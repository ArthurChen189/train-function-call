"""Evaluation driver: run a fixed, held-out task suite against one or more
model snapshots (base, post-SFT, post-RL) and emit a comparison table.

Held-out is enforced two ways:
- `train` split: same task families the policy trained on (base tools), but a
  disjoint seed range -> measures in-domain generalisation to new task instances.
- `test` split (holdout): tasks that use a *disjoint tool set* (`record_read`,
  `gcd`, `lcm`, `digit_sum`) — no `calculator` or `kv_lookup` — the policy
  never saw in SFT/RL. Measures transfer of tool-calling behaviour to new
  names/schemas, not memorised train tools.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .env import sample_tasks
from .rollout import rollout


EVAL_SEED = 999_001  # disjoint from SFT (seed=0..n) and RL (42..42+steps)


def _load_model(base_model: str, adapter: str | None):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map="auto")
    if adapter:
        model = PeftModel.from_pretrained(base, adapter)
    else:
        model = base
    model.eval()
    return model


def evaluate(model, tokenizer, n: int, *, split: str, max_turns: int,
             max_new_tokens: int, greedy: bool) -> dict[str, Any]:
    tasks = sample_tasks(n, seed=EVAL_SEED, split=split)
    rewards: list[float] = []
    correct: list[int] = []
    turns: list[int] = []
    valid_calls: list[int] = []
    malformed: list[int] = []
    has_answer: list[int] = []

    examples: list[dict[str, Any]] = []
    for i, t in enumerate(tasks):
        tr = rollout(
            model, tokenizer, t,
            max_turns=max_turns, max_new_tokens=max_new_tokens, greedy=greedy,
        )
        rewards.append(tr.reward)
        correct.append(int(tr.stats.correct))
        turns.append(tr.stats.turns)
        valid_calls.append(tr.stats.valid_tool_calls)
        malformed.append(tr.stats.malformed_tool_calls + tr.stats.unknown_tool_calls)
        has_answer.append(int(tr.stats.final_answer is not None))
        if i < 3:  # keep a few full trajectories for the writeup
            examples.append({
                "task_prompt": t.prompt,
                "ground_truth": t.answer,
                "messages": tr.messages,
                "stats": tr.stats.__dict__,
                "reward": tr.reward,
            })

    return {
        "n": n,
        "success_rate": sum(correct) / n,
        "mean_reward": statistics.mean(rewards),
        "reward_std": statistics.pstdev(rewards) if n > 1 else 0.0,
        "mean_turns": statistics.mean(turns),
        "mean_valid_tool_calls": statistics.mean(valid_calls),
        "mean_malformed_or_unknown": statistics.mean(malformed),
        "answer_emit_rate": sum(has_answer) / n,
        "examples": examples,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--adapters", nargs="*", default=[],
                   help="Pairs of NAME=PATH, e.g. sft=checkpoints/sft/final rl=checkpoints/rl/final.")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--splits", nargs="*", default=["train", "test"],
                   help="Which task splits to evaluate. 'train' = base tools (in-domain), "
                        "'test' = held-out tools the policy never trained on.")
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--out", default="results/eval_summary.json")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    runs: list[tuple[str, str | None]] = [("base", None)]
    for entry in args.adapters:
        name, path = entry.split("=", 1)
        runs.append((name, path))

    # summary[name][split] = metrics (+ examples). Load each model once, eval all splits.
    summary: dict[str, Any] = {name: {} for name, _ in runs}
    for name, adapter in runs:
        model = _load_model(args.base_model, adapter)
        for split in args.splits:
            print(f"\n[eval] === {name} (adapter={adapter}) | split={split} ===")
            res = evaluate(
                model, tokenizer, args.n, split=split,
                max_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
                greedy=args.greedy,
            )
            headline = {k: v for k, v in res.items() if k != "examples"}
            print(json.dumps(headline, indent=2))
            summary[name][split] = res
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # One comparison table per split.
    keys = ["success_rate", "mean_reward", "answer_emit_rate", "mean_turns",
            "mean_valid_tool_calls", "mean_malformed_or_unknown"]
    for split in args.splits:
        title = ("in-domain (base tools)" if split == "train"
                 else "held-out tools" if split == "test" else split)
        print("\n" + "=" * 80)
        print(f"split = {split}  [{title}]")
        print(f"{'metric':35s}" + "".join(f"{name:>15s}" for name, _ in runs))
        print("-" * 80)
        for k in keys:
            row = f"{k:35s}"
            for name, _ in runs:
                row += f"{summary[name][split][k]:>15.4f}"
            print(row)
    print("=" * 80)
    print(f"[eval] wrote full results to {out_path}")


if __name__ == "__main__":
    main()
