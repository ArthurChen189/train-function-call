"""SFT data from `Salesforce/APIGen-MT-5k` (real multi-turn tool-use trajectories).

APIGen-MT-5k is a ShareGPT-format dataset of 5,000 verified multi-turn agent
trajectories in the retail and airline (tau-bench) domains. Each row is:

    {
      "system": "<domain policy>",
      "tools":  "<JSON list of tool schemas>",
      "conversations": [
        {"from": "human",         "value": "..."},   # user
        {"from": "function_call", "value": "{name, arguments}"},  # assistant tool call
        {"from": "observation",   "value": "..."},    # tool result
        {"from": "gpt",           "value": "..."},     # assistant reply
        ...
      ],
    }

Why this instead of the previous synthetic generator:
  + Real, diverse tool use (dozens of tools, two domains, genuine multi-turn
    flows) instead of two hand-written task families -> the format generalises.
  + Human-verified trajectories (99% success in the paper's audit) -> clean
    teacher signal.

What we convert it *into*: we keep the exact tag protocol the rest of the
pipeline already speaks -- the tool call lives as text inside the assistant
turn (`<tool_call>{...}</tool_call>`) and the tool result comes back as a
`tool`-role message wrapped in `<tool_response>...</tool_response>`. This is
byte-identical to what `rollout.py` emits/parses at RL and eval time, so the
SFT format and the env format do not drift (the classic silent-failure mode).

Trade-off we accept (see README): APIGen's tools are retail/airline, *not* the
env's `calculator`/`kv_lookup`. SFT therefore teaches the general tool-calling
*format*, while the env-specific behaviour (which two tools exist, the
`<answer>` tag) is left for the system prompt + RL to instil.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from typing import Any

from datasets import Dataset, load_dataset

APIGEN_DATASET = "Salesforce/APIGen-MT-5k"

# Rows reserved (from a fixed canonical shuffle) for the SFT eval split. Held
# out from training so the in-loop eval is genuinely disjoint.
EVAL_HOLDOUT = 256
# Fixed seed for the canonical shuffle that defines the train/eval split. This
# is independent of the per-call sampling seed, so train and eval pools never
# overlap regardless of what seed a caller passes.
_SPLIT_SEED = 1234

# Mirrors the tool-call protocol described in `env.build_system_prompt` so the
# emit format the model learns here matches what the env consumes at RL/eval.
_TOOL_CALL_PROTOCOL = (
    "\n\nTo call a tool, emit exactly one tool call per assistant turn:\n"
    '  <tool_call>{"name": "<tool_name>", "arguments": {...}}</tool_call>\n'
    "You will receive the result in a tool message wrapped in "
    "<tool_response>...</tool_response>.\n"
    "Call tools rather than guessing; reply to the user in natural language "
    "when no tool call is needed."
)


def _build_system_prompt(domain_policy: str, tools_json: str) -> str:
    """Compose the domain policy, the available tools, and the tag protocol."""
    return (
        f"{domain_policy.strip()}\n\n"
        f"Available tools:\n{tools_json.strip()}"
        f"{_TOOL_CALL_PROTOCOL}"
    )


def _normalise_tool_call(value: str) -> str:
    """Compact a function_call JSON string; pass through unchanged if unparseable."""
    try:
        obj = json.loads(value)
        obj.setdefault("arguments", {})
        return json.dumps(obj, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return value


def _convert_row(row: dict[str, Any]) -> list[dict[str, str]] | None:
    """ShareGPT row -> chat `messages` in the pipeline's tag protocol.

    Returns `None` if the row has no usable assistant turn after cleaning.
    The `think` pseudo-tool (a tau-bench reasoning scaffold, always paired with
    an empty observation) is dropped: it is not a real tool, and emitting it
    would teach the model to call a tool the env would penalise as unknown.
    """
    system = _build_system_prompt(row["system"], row["tools"])
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    n_assistant = 0
    for turn in row["conversations"]:
        src, value = turn.get("from"), turn.get("value", "")
        if src == "human":
            messages.append({"role": "user", "content": value})
        elif src == "gpt":
            messages.append({"role": "assistant", "content": value})
            n_assistant += 1
        elif src == "function_call":
            try:
                name = json.loads(value).get("name")
            except (json.JSONDecodeError, TypeError):
                name = None
            if name == "think":
                continue  # drop reasoning scaffold (paired obs is empty)
            messages.append({
                "role": "assistant",
                "content": f"<tool_call>{_normalise_tool_call(value)}</tool_call>",
            })
            n_assistant += 1
        elif src == "observation":
            if not value.strip():
                continue  # empty obs only ever follows a dropped `think` call
            messages.append({
                "role": "tool",
                "content": f"<tool_response>{value}</tool_response>",
            })
        # unknown roles are ignored

    if n_assistant == 0:
        return None
    return messages


@lru_cache(maxsize=1)
def _split_pools() -> tuple[tuple[tuple[dict, ...], ...], tuple[tuple[dict, ...], ...]]:
    """Load + convert the whole dataset once; return (train_pool, eval_pool).

    Cached so repeated calls (train then eval) don't re-download/re-convert.
    Tuples are used so the result is hashable/immutable under the cache.
    """
    raw = load_dataset(APIGEN_DATASET, split="train")
    converted = [m for m in (_convert_row(r) for r in raw) if m is not None]

    rng = random.Random(_SPLIT_SEED)
    rng.shuffle(converted)

    eval_pool = converted[:EVAL_HOLDOUT]
    train_pool = converted[EVAL_HOLDOUT:]
    # Freeze into tuples of tuples so lru_cache can hold them immutably.
    to_t = lambda pool: tuple(tuple(msg.items() for msg in conv) for conv in pool)  # noqa: E731
    return to_t(train_pool), to_t(eval_pool)


def _thaw(conv: tuple) -> list[dict[str, str]]:
    return [dict(msg) for msg in conv]


def build_sft_dataset(
    n: int,
    seed: int = 0,
    split: str = "train",
    tokenizer: Any | None = None,
    max_tokens: int | None = None,
) -> Dataset:
    """Return a HF `Dataset` with a single `messages` column (chat format).

    Args:
        n: number of trajectories to return (capped by what's available/fits).
        seed: shuffles the pool so different calls pick different rows.
        split: "train" or "eval" -- disjoint pools (see `EVAL_HOLDOUT`).
        tokenizer / max_tokens: if both given, only trajectories whose rendered
            length is <= `max_tokens` are kept. This avoids training on
            right-truncated trajectories (APIGen rows are long: the shared
            system prompt alone is ~3.6k tokens), which would otherwise drop
            later assistant turns and, in the worst case, leave a row with zero
            trainable tokens.
    """
    train_pool, eval_pool = _split_pools()
    pool = list(train_pool if split == "train" else eval_pool)
    if not pool:
        raise ValueError(f"empty pool for split={split!r}")

    rng = random.Random(seed)
    rng.shuffle(pool)

    fits = _make_length_filter(tokenizer, max_tokens)

    chosen: list[list[dict[str, str]]] = []
    for conv_t in pool:
        if len(chosen) >= n:
            break
        conv = _thaw(conv_t)
        if fits(conv):
            chosen.append(conv)

    if len(chosen) < n:
        print(f"[data] split={split}: requested {n}, returning {len(chosen)} "
              f"(pool={len(pool)}, max_tokens={max_tokens}).")
    return Dataset.from_list([{"messages": c} for c in chosen])


def _make_length_filter(tokenizer: Any | None, max_tokens: int | None):
    """Return a predicate `fits(messages) -> bool`.

    When filtering is active we render with TRL's training chat template (the
    same one `SFTTrainer` uses for assistant-only loss), so the measured length
    matches what training actually sees.
    """
    if tokenizer is None or max_tokens is None:
        return lambda _conv: True

    chat_template = None
    try:  # use the exact template SFTTrainer will train with, if available
        from trl.chat_template_utils import get_training_chat_template

        chat_template = get_training_chat_template(tokenizer)
    except Exception:  # noqa: BLE001 -- fall back to the tokenizer's default template
        chat_template = None

    def fits(conv: list[dict[str, str]]) -> bool:
        ids = tokenizer.apply_chat_template(
            conv, chat_template=chat_template, tokenize=True,
            return_dict=False, add_generation_prompt=False,
        )
        return len(ids) <= max_tokens

    return fits


if __name__ == "__main__":
    ds = build_sft_dataset(2, seed=1)
    for row in ds:
        print(json.dumps(row, indent=2)[:1500], "...\n---")
