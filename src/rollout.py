"""Multi-turn rollout loop used by both RL training and eval.

Conversation grows turn-by-turn: each assistant turn is generated with
`model.generate`, then we parse for either a `<tool_call>` (continue the loop
with a tool message) or an `<answer>` (terminate). A hard `max_turns` cutoff
stops runaways.

The same function is used at eval time and inside the GRPO sampler. Keeping
it in one place means the reward we train against == the reward we eval on,
which removes a major source of train/eval drift bugs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .env import (
    ANSWER_RE,
    Task,
    TrajectoryStats,
    build_system_prompt,
    compute_reward,
    parse_answer,
    parse_tool_calls,
    run_tool,
)


@dataclass
class Trajectory:
    task: Task
    messages: list[dict[str, str]]                  # full chat trace
    assistant_segments: list[dict[str, Any]] = field(default_factory=list)
    # Each segment: {"prompt_ids": LongTensor[L_p], "response_ids": LongTensor[L_r]}
    stats: TrajectoryStats = field(default_factory=TrajectoryStats)
    reward: float = 0.0


def _initial_messages(task: Task) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(task.tools)},
        {"role": "user", "content": task.prompt},
    ]


def _apply_template(tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]],
                    add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
    )


@torch.no_grad()
def rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task: Task,
    *,
    max_turns: int = 6,
    max_new_tokens: int = 192,
    temperature: float = 0.7,
    top_p: float = 0.95,
    greedy: bool = False,
) -> Trajectory:
    """Run a full multi-turn rollout on a single task. Returns a Trajectory."""
    device = next(model.parameters()).device
    messages = _initial_messages(task)
    traj = Trajectory(task=task, messages=messages)

    # `tool` role compatibility: some tokenizers don't render it. Fall back to user.
    tool_role = "tool"
    try:
        _apply_template(tokenizer, messages + [{"role": "tool", "content": "x"}],
                        add_generation_prompt=True)
    except Exception:  # noqa: BLE001
        tool_role = "user"

    for turn in range(max_turns):
        prompt_text = _apply_template(tokenizer, traj.messages, add_generation_prompt=True)
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        prompt_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
        )
        if greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)

        out = model.generate(prompt_ids, **gen_kwargs)
        response_ids = out[0, prompt_ids.shape[1]:].detach().cpu()
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        traj.assistant_segments.append({
            "prompt_ids": prompt_ids[0].detach().cpu(),
            "response_ids": response_ids,
        })
        traj.messages.append({"role": "assistant", "content": response_text})
        traj.stats.turns += 1

        # Parse what the model just emitted.
        calls = parse_tool_calls(response_text)
        answer = parse_answer(response_text)

        if answer is not None:
            traj.stats.final_answer = answer
            traj.stats.correct = (answer == task.answer)
            break

        if not calls:
            # No tool call and no answer -> nudge the model and continue.
            traj.messages.append({
                "role": tool_role,
                "content": "<tool_response>ERROR: no tool_call or answer detected. "
                           "Emit one tool_call or an answer.</tool_response>",
            })
            traj.stats.malformed_tool_calls += 1
            continue

        # Execute the FIRST tool call only (encourages one-call-per-turn discipline).
        call = calls[0]
        if isinstance(call, str):  # "MALFORMED"
            traj.stats.malformed_tool_calls += 1
            result = "ERROR: malformed tool_call JSON"
        else:
            name = call.get("name")
            allowed = frozenset(task.tools)
            if name not in allowed:
                traj.stats.unknown_tool_calls += 1
            else:
                traj.stats.valid_tool_calls += 1
            result = run_tool(call, task)

        traj.messages.append({
            "role": tool_role,
            "content": f"<tool_response>{result}</tool_response>",
        })

    traj.reward = compute_reward(task, traj.stats)
    return traj


def rollout_batch(
    model, tokenizer, tasks: list[Task], *, k_per_task: int = 1, **kwargs,
) -> list[list[Trajectory]]:
    """Sequential batched rollouts (k samples per task). Returns [[Trajectory]*k]*N."""
    all_trajs: list[list[Trajectory]] = []
    for t in tasks:
        per: list[Trajectory] = []
        for _ in range(k_per_task):
            per.append(rollout(model, tokenizer, t, **kwargs))
        all_trajs.append(per)
    return all_trajs
