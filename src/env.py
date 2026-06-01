"""Self-contained tool-use environment with grounded, verifiable rewards.

Why this env instead of Open Trajectory Gym / BFCL / tau-bench:
- Rewards are 100% programmatic (no graders, no LLM-as-judge, no flakiness).
- Tasks are sampled fresh per seed, so the model can't memorise answers from SFT.
- Two tools (`calculator`, `kv_lookup`) force at least 2-3 tool turns, which
  exercises the multi-turn loop without blowing up the rollout budget.

Tool-call protocol (decoupled from any specific chat template):
    Assistant emits:
        <tool_call>{"name": "calculator", "arguments": {"expression": "3+4"}}</tool_call>
    Tool result is fed back as a `tool` role message:
        <tool_response>7</tool_response>
    Final answer:
        <answer>NUMERIC_VALUE</answer>

Reward shaping (per trajectory, terminal):
    +1.0  correct final answer
    +0.1  emitted a parseable <answer>...</answer> (even if wrong) -- shapes format
    +0.05 * (#valid tool calls), capped at 0.2  -- encourages tool use
    -0.1  per malformed tool call or unknown tool   -- discourages garbage
    -0.05 * max(0, turns - target_turns)            -- discourages stalling

The +0.1 "format" shaping is small enough that it can't out-weigh correctness,
which is the dominant signal. We watched for the classic hack of "spam tool
calls to farm shaping reward" -- the cap on tool-call bonus plus the turn
penalty mean the optimal policy is still "use the tools you need, then answer".
"""
from __future__ import annotations

import ast
import json
import operator
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------- Tool implementations -------------------------------------------------

_SAFE_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_SAFE_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        return _SAFE_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
        return _SAFE_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsafe expression node: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression. Returns a stringified number."""
    if not isinstance(expression, str):
        return "ERROR: expression must be a string"
    try:
        tree = ast.parse(expression, mode="eval")
        val = _safe_eval(tree)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return str(val)


def kv_lookup(key: str, kb: dict[str, Any]) -> str:
    """Look up `key` in the task-local knowledge base. Returns value or NOT_FOUND."""
    if not isinstance(key, str):
        return "ERROR: key must be a string"
    if key in kb:
        return str(kb[key])
    return "NOT_FOUND"


# ---------- Task generation ------------------------------------------------------

KNOWN_TOOLS = frozenset({"calculator", "kv_lookup"})

TOOL_SCHEMA = [
    {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression (+, -, *, /, **, parentheses).",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "kv_lookup",
        "description": "Look up the integer value associated with a key in the user's database.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
]


@dataclass
class Task:
    """A single tool-use task with a verifiable ground-truth answer."""
    prompt: str
    kb: dict[str, int]
    answer: int  # exact integer answer
    target_turns: int = 3  # used for the turn-penalty in reward shaping
    meta: dict[str, Any] = field(default_factory=dict)


_KEY_POOL = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
]


def _gen_two_key_arith(rng: random.Random) -> Task:
    """Look up two values then combine them with a small arithmetic expression."""
    k1, k2 = rng.sample(_KEY_POOL, 2)
    v1, v2 = rng.randint(2, 50), rng.randint(2, 50)
    distractors = rng.sample([k for k in _KEY_POOL if k not in (k1, k2)], 3)
    kb = {k1: v1, k2: v2, **{d: rng.randint(2, 50) for d in distractors}}

    op_name, op_fn, op_sym = rng.choice([
        ("sum", lambda a, b: a + b, "+"),
        ("product", lambda a, b: a * b, "*"),
        ("difference (first minus second)", lambda a, b: a - b, "-"),
    ])
    mult = rng.randint(2, 6)
    answer = op_fn(v1, v2) * mult
    prompt = (
        f"Look up the values for '{k1}' and '{k2}' in the database, "
        f"then compute the {op_name} of those two values and multiply by {mult}. "
        f"Respond with the final integer wrapped in <answer>...</answer>."
    )
    return Task(
        prompt=prompt,
        kb=kb,
        answer=answer,
        target_turns=3,
        meta={"keys": [k1, k2], "op": op_sym, "mult": mult},
    )


def _gen_three_key_arith(rng: random.Random) -> Task:
    """Slightly harder: three lookups + a 2-op expression."""
    k1, k2, k3 = rng.sample(_KEY_POOL, 3)
    v1, v2, v3 = (rng.randint(2, 30) for _ in range(3))
    distractors = rng.sample([k for k in _KEY_POOL if k not in (k1, k2, k3)], 2)
    kb = {k1: v1, k2: v2, k3: v3, **{d: rng.randint(2, 30) for d in distractors}}
    answer = (v1 + v2) * v3
    prompt = (
        f"Look up '{k1}', '{k2}', and '{k3}' in the database, then compute "
        f"(value_of_{k1} + value_of_{k2}) * value_of_{k3}. "
        f"Respond with the final integer wrapped in <answer>...</answer>."
    )
    return Task(prompt=prompt, kb=kb, answer=answer, target_turns=4,
                meta={"keys": [k1, k2, k3]})


_GENERATORS = [_gen_two_key_arith, _gen_two_key_arith, _gen_three_key_arith]


def sample_tasks(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    return [rng.choice(_GENERATORS)(rng) for _ in range(n)]


# ---------- Tool dispatch + reward ----------------------------------------------

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


def parse_tool_calls(text: str) -> list[dict[str, Any] | str]:
    """Return a list of parsed tool-call dicts or error strings (one per match)."""
    out: list[dict[str, Any] | str] = []
    for m in TOOL_CALL_RE.finditer(text):
        raw = m.group(1)
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict) or "name" not in obj:
                out.append("MALFORMED")
            else:
                obj.setdefault("arguments", {})
                out.append(obj)
        except json.JSONDecodeError:
            out.append("MALFORMED")
    return out


def parse_answer(text: str) -> int | None:
    m = ANSWER_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def run_tool(call: dict[str, Any], task: Task) -> str:
    name = call.get("name")
    args = call.get("arguments") or {}
    if name == "calculator":
        return calculator(args.get("expression", ""))
    if name == "kv_lookup":
        return kv_lookup(args.get("key", ""), task.kb)
    return f"ERROR: unknown tool '{name}'"


@dataclass
class TrajectoryStats:
    valid_tool_calls: int = 0  # calculator or kv_lookup calls that parsed and dispatched
    malformed_tool_calls: int = 0  # invalid JSON, missing name, or turn with no tool_call/answer
    unknown_tool_calls: int = 0  # parseable calls whose name is not calculator or kv_lookup
    turns: int = 0  # assistant turns taken before termination or max_turns
    final_answer: int | None = None  # integer from <answer>...</answer>, if emitted
    correct: bool = False  # final_answer equals task.answer


def compute_reward(task: Task, stats: TrajectoryStats) -> float:
    r = 0.0
    if stats.correct:
        r += 1.0
    if stats.final_answer is not None:
        r += 0.1
    r += min(0.2, 0.05 * stats.valid_tool_calls)
    r -= 0.1 * (stats.malformed_tool_calls + stats.unknown_tool_calls)
    r -= 0.05 * max(0, stats.turns - task.target_turns)
    return r


# ---------- System prompt builder -----------------------------------------------

def build_system_prompt() -> str:
    schema = json.dumps(TOOL_SCHEMA, indent=2)
    return (
        "You are a helpful assistant with access to two tools.\n\n"
        f"Available tools:\n{schema}\n\n"
        "To call a tool, emit exactly one tool call per assistant turn:\n"
        '  <tool_call>{"name": "<tool_name>", "arguments": {...}}</tool_call>\n'
        "You will receive the result in a tool message wrapped in <tool_response>...</tool_response>.\n"
        "When you have the final numeric answer, output it as:\n"
        "  <answer>INTEGER</answer>\n"
        "Use the tools rather than guessing. Do not invent values."
    )


if __name__ == "__main__":
    # Smoke test
    tasks = sample_tasks(3, seed=0)
    for t in tasks:
        print("PROMPT:", t.prompt)
        print("KB:", t.kb)
        print("ANSWER:", t.answer)
        print("---")
