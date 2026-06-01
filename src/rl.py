"""Custom GRPO loop for multi-turn tool-use.

Why not `trl.GRPOTrainer`?
- TRL's GRPOTrainer is built around single-turn `(prompt -> completion -> reward(completion))`.
  Our trajectories are multi-turn with tool messages interleaved, and the
  policy log-probs we need are over MULTIPLE assistant segments per trajectory.
  Bending GRPOTrainer to that shape is more code (and more bugs) than just
  writing the ~150-line core loop directly.

GRPO recap (Shao et al., 2024 / DeepSeekMath):
  For each prompt, sample K completions; let R_i be their rewards.
  Advantage A_i = (R_i - mean(R)) / (std(R) + eps)   <- group-relative, no critic
  Loss:
      -mean_i [ A_i * sum_t log pi(a_t|s_t) ]  + beta * KL(pi || pi_ref)
  We use the "REINFORCE-with-group-baseline" form (no PPO ratio clip) for
  simplicity; it's the same as GRPO with `epsilon -> infty`. With LoRA at low
  LR this is stable enough for a 4-hour demo.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .env import sample_tasks
from .rollout import Trajectory, rollout


# ---------- log-prob computation -------------------------------------------------

def _segment_logprobs(model, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
    """Return per-token log-probs of `response_ids` given `prompt_ids` (no grad context controlled by caller).

    Shapes:
        prompt_ids:   [L_p]
        response_ids: [L_r]
    Returns:
        logprobs:     [L_r]
    """
    device = next(model.parameters()).device
    full = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(device)
    out = model(full)
    logits = out.logits[0]                         # [L, V]
    # Predict token t from logits at position t-1.
    logits = logits[prompt_ids.shape[0] - 1 : -1]  # [L_r, V]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    target = response_ids.to(device).long()
    return log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [L_r]


def _trajectory_logprob_sum(model, traj: Trajectory) -> torch.Tensor:
    """Sum of per-token log-probs across ALL assistant segments of a trajectory."""
    total = None
    for seg in traj.assistant_segments:
        lp = _segment_logprobs(model, seg["prompt_ids"], seg["response_ids"]).sum()
        total = lp if total is None else total + lp
    assert total is not None, "trajectory had no assistant segments"
    return total


def _trajectory_logprob_mean_per_token(model, traj: Trajectory) -> torch.Tensor:
    total_tokens = sum(seg["response_ids"].shape[0] for seg in traj.assistant_segments)
    return _trajectory_logprob_sum(model, traj) / max(1, total_tokens)


# ---------- GRPO step ------------------------------------------------------------

def grpo_step(
    policy,
    ref_model,
    tokenizer,
    tasks,
    *,
    k_rollouts: int,
    kl_beta: float,
    temperature: float,
    max_turns: int,
    max_new_tokens: int,
) -> dict[str, float]:
    """Sample K rollouts per task, compute GRPO loss, backprop. Returns metrics."""
    policy.eval()  # generation uses eval mode; we re-enable grad mode for loss only.

    all_rewards: list[list[float]] = []
    all_trajs: list[list[Trajectory]] = []
    for t in tasks:
        per: list[Trajectory] = []
        for _ in range(k_rollouts):
            tr = rollout(
                policy, tokenizer, t,
                max_turns=max_turns,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            per.append(tr)
        all_trajs.append(per)
        all_rewards.append([tr.reward for tr in per])

    # Group-relative advantages.
    advantages: list[list[float]] = []
    for group in all_rewards:
        mu = sum(group) / len(group)
        var = sum((r - mu) ** 2 for r in group) / len(group)
        std = math.sqrt(var) + 1e-6
        advantages.append([(r - mu) / std for r in group])

    # Compute loss.
    policy.train()
    total_loss = torch.zeros((), device=next(policy.parameters()).device)
    n_traj = 0
    kl_sum = 0.0
    for trajs, advs in zip(all_trajs, advantages):
        for tr, A in zip(trajs, advs):
            if not tr.assistant_segments:
                continue
            # Policy log-prob (with grad).
            lp_policy = _trajectory_logprob_mean_per_token(policy, tr)
            # Reference log-prob (no grad).
            with torch.no_grad():
                lp_ref = _trajectory_logprob_mean_per_token(ref_model, tr).detach()

            # Per-token approximate KL: (log p - log p_ref) summed over tokens, here we used
            # the per-token mean so this is already a mean KL estimate.
            kl_approx = lp_policy - lp_ref
            kl_sum += kl_approx.detach().item()

            # REINFORCE with group baseline + KL penalty.
            total_loss = total_loss + (-A * lp_policy + kl_beta * kl_approx)
            n_traj += 1

    if n_traj == 0:
        return {"loss": 0.0, "mean_reward": 0.0, "kl": 0.0, "n_traj": 0}

    loss = total_loss / n_traj
    loss.backward()

    flat_rewards = [r for g in all_rewards for r in g]
    correct = [tr.stats.correct for g in all_trajs for tr in g]
    return {
        "loss": loss.item(),
        "mean_reward": sum(flat_rewards) / len(flat_rewards),
        "max_reward": max(flat_rewards),
        "min_reward": min(flat_rewards),
        "success_rate": sum(correct) / len(correct),
        "kl": kl_sum / n_traj,
        "n_traj": n_traj,
    }


# ---------- driver --------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--sft-adapter", default="checkpoints/sft/final",
                   help="Path to SFT LoRA adapter (also used as the frozen reference policy).")
    p.add_argument("--out", default="checkpoints/rl")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--k-rollouts", type=int, default=4)
    p.add_argument("--kl-beta", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    print(f"[rl] loading base model {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto",
    )
    print(f"[rl] attaching trainable LoRA from {args.sft_adapter}")
    policy = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)
    policy.config.use_cache = False

    # Frozen reference = a second copy of base + SFT adapter, no grads.
    ref_base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto",
    )
    ref_model = PeftModel.from_pretrained(ref_base, args.sft_adapter, is_trainable=False)
    ref_model.eval()
    for p_ in ref_model.parameters():
        p_.requires_grad_(False)

    trainable = [p_ for p_ in policy.parameters() if p_.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"[rl] trainable params: {sum(p.numel() for p in trainable):,}")

    with open(log_path, "w") as f:
        for step in range(args.steps):
            tasks = sample_tasks(args.prompts_per_step, seed=args.seed + step, split="train")
            optim.zero_grad(set_to_none=True)
            metrics = grpo_step(
                policy, ref_model, tokenizer, tasks,
                k_rollouts=args.k_rollouts,
                kl_beta=args.kl_beta,
                temperature=args.temperature,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
            )
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optim.step()
            metrics["step"] = step
            print(f"[rl] step={step} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                  for k, v in metrics.items() if k != "step"))
            f.write(json.dumps(metrics) + "\n")
            f.flush()

    final = out_dir / "final"
    policy.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"[rl] saved adapter to {final}")


if __name__ == "__main__":
    main()
