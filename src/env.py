"""Self-contained tool-use environment with grounded, verifiable rewards.

Why this env instead of Open Trajectory Gym / BFCL / tau-bench:
- Rewards are 100% programmatic (no graders, no LLM-as-judge, no flakiness).
- Tasks are sampled fresh per seed, so the model can't memorise answers from SFT.
- Two tools (`calculator`, `kv_lookup`) force at least 2-3 tool turns, which
  exercises the multi-turn loop without blowing up the rollout budget.

Train/holdout tool split (held-out-tool generalisation):
- The `train` split only exposes `calculator` and `kv_lookup`.
- The `test` (holdout) split exposes a *disjoint* tool set: `record_read`
  (lookup with a new name/schema), plus `gcd`, `lcm`, and `digit_sum`. No train
  tool names appear at holdout time. Each task advertises only its tool list in
  the system prompt so eval measures transfer of tool-calling behaviour, not
  memorised tool names. Answers stay generator-computed and exactly verifiable.

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
import math
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


# ---------- Held-out tools (test split only) -------------------------------------
# Deterministic and integer-valued so rewards stay exactly verifiable. The model
# never sees these during SFT/RL; they exist purely to probe tool generalisation.

def _as_int(value: Any) -> int:
    """Coerce a JSON number or numeric string to int (raises on bad input)."""
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
        raise ValueError("expected a number, got bool")
    return int(value)


def gcd(a: Any, b: Any) -> str:
    """Greatest common divisor of two integers."""
    try:
        return str(math.gcd(_as_int(a), _as_int(b)))
    except (TypeError, ValueError):
        return "ERROR: gcd requires integer arguments 'a' and 'b'"


def lcm(a: Any, b: Any) -> str:
    """Least common multiple of two integers."""
    try:
        return str(math.lcm(_as_int(a), _as_int(b)))
    except (TypeError, ValueError):
        return "ERROR: lcm requires integer arguments 'a' and 'b'"


def digit_sum(n: Any) -> str:
    """Sum of the decimal digits of an integer."""
    try:
        return str(sum(int(d) for d in str(abs(_as_int(n)))))
    except (TypeError, ValueError):
        return "ERROR: digit_sum requires an integer argument 'n'"


# ---------- Tool registry --------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    """A tool's advertised schema plus its dispatch function `(args, task) -> str`."""
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[[dict[str, Any], "Task"], str]

    @property
    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters}


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


TOOLS: dict[str, Tool] = {
    # --- base tools (seen during SFT/RL) ---
    "calculator": Tool(
        "calculator",
        "Evaluate a basic arithmetic expression (+, -, *, /, **, parentheses).",
        _obj({"expression": {"type": "string"}}, ["expression"]),
        lambda a, t: calculator(a.get("expression", "")),
    ),
    "kv_lookup": Tool(
        "kv_lookup",
        "Look up the integer value associated with a key in the user's database.",
        _obj({"key": {"type": "string"}}, ["key"]),
        lambda a, t: kv_lookup(a.get("key", ""), t.kb),
    ),
    # --- holdout tools (test split only; names disjoint from train) ---
    "record_read": Tool(
        "record_read",
        "Read the integer value stored under record_id in the task database.",
        _obj({"record_id": {"type": "string"}}, ["record_id"]),
        lambda a, t: kv_lookup(a.get("record_id", ""), t.kb),
    ),
    "gcd": Tool(
        "gcd",
        "Compute the greatest common divisor of two integers a and b.",
        _obj({"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a", "b"]),
        lambda a, t: gcd(a.get("a"), a.get("b")),
    ),
    "lcm": Tool(
        "lcm",
        "Compute the least common multiple of two integers a and b.",
        _obj({"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a", "b"]),
        lambda a, t: lcm(a.get("a"), a.get("b")),
    ),
    "digit_sum": Tool(
        "digit_sum",
        "Compute the sum of the decimal digits of an integer n.",
        _obj({"n": {"type": "integer"}}, ["n"]),
        lambda a, t: digit_sum(a.get("n")),
    ),
}

TRAIN_TOOLS = ("calculator", "kv_lookup")         # train split only
HOLDOUT_TOOLS = ("record_read", "gcd", "lcm", "digit_sum")  # test split only
BASE_TOOLS = TRAIN_TOOLS  # default for train tasks / build_system_prompt()
KNOWN_TOOLS = frozenset(TOOLS)                    # full registry (dispatch)
assert frozenset(TRAIN_TOOLS).isdisjoint(HOLDOUT_TOOLS)


# ---------- Task generation ------------------------------------------------------


@dataclass
class Task:
    """A single tool-use task with a verifiable ground-truth answer."""
    prompt: str
    kb: dict[str, int]
    answer: int  # exact integer answer
    target_turns: int = 3  # used for the turn-penalty in reward shaping
    # Tool names advertised to the model for this task (built into the system
    # prompt). Train tasks expose only base tools; test tasks add a held-out one.
    tools: tuple[str, ...] = BASE_TOOLS
    meta: dict[str, Any] = field(default_factory=dict)


_KEY_POOL = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
]


# --- train split: base tools only ---

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
        tools=BASE_TOOLS,
        meta={"keys": [k1, k2], "op": op_sym, "mult": mult, "family": "two_key"},
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
                tools=BASE_TOOLS, meta={"keys": [k1, k2, k3], "family": "three_key"})


# --- test split: each task requires a held-out tool the policy never trained on ---

def _gen_gcd(rng: random.Random) -> Task:
    k1, k2 = rng.sample(_KEY_POOL, 2)
    v1, v2 = rng.randint(10, 99), rng.randint(10, 99)
    distractors = rng.sample([k for k in _KEY_POOL if k not in (k1, k2)], 3)
    kb = {k1: v1, k2: v2, **{d: rng.randint(10, 99) for d in distractors}}
    answer = math.gcd(v1, v2)
    prompt = (
        f"Use record_read to read the values for record_ids '{k1}' and '{k2}', "
        f"then use the gcd tool to compute the greatest common divisor of those "
        f"two values. Respond with the final integer wrapped in <answer>...</answer>."
    )
    return Task(prompt=prompt, kb=kb, answer=answer, target_turns=3,
                tools=("record_read", "gcd"), meta={"keys": [k1, k2], "family": "gcd"})


def _gen_lcm(rng: random.Random) -> Task:
    k1, k2 = rng.sample(_KEY_POOL, 2)
    v1, v2 = rng.randint(2, 20), rng.randint(2, 20)
    distractors = rng.sample([k for k in _KEY_POOL if k not in (k1, k2)], 3)
    kb = {k1: v1, k2: v2, **{d: rng.randint(2, 20) for d in distractors}}
    answer = math.lcm(v1, v2)
    prompt = (
        f"Use record_read to read the values for record_ids '{k1}' and '{k2}', "
        f"then use the lcm tool to compute the least common multiple of those two "
        f"values. Respond with the final integer wrapped in <answer>...</answer>."
    )
    return Task(prompt=prompt, kb=kb, answer=answer, target_turns=3,
                tools=("record_read", "lcm"), meta={"keys": [k1, k2], "family": "lcm"})


def _gen_digit_sum(rng: random.Random) -> Task:
    k = rng.choice(_KEY_POOL)
    v = rng.randint(100, 999)
    distractors = rng.sample([x for x in _KEY_POOL if x != k], 3)
    kb = {k: v, **{d: rng.randint(100, 999) for d in distractors}}
    answer = sum(int(d) for d in str(v))
    prompt = (
        f"Use record_read to read the value for record_id '{k}', then use the "
        f"digit_sum tool to compute the sum of the digits of that value. "
        f"Respond with the final integer wrapped in <answer>...</answer>."
    )
    return Task(prompt=prompt, kb=kb, answer=answer, target_turns=2,
                tools=("record_read", "digit_sum"), meta={"keys": [k], "family": "digit_sum"})


_TRAIN_GENERATORS = [_gen_two_key_arith, _gen_two_key_arith, _gen_three_key_arith]
_TEST_GENERATORS = [_gen_gcd, _gen_lcm, _gen_digit_sum]


def sample_tasks(n: int, seed: int, split: str = "train") -> list[Task]:
    """Sample `n` tasks.

    `split="train"`: tasks using TRAIN_TOOLS only (`calculator`, `kv_lookup`).
    `split="test"`: holdout tasks using HOLDOUT_TOOLS only (disjoint names/schemas).
    """
    if split not in ("train", "test"):
        raise ValueError(f"unknown split {split!r}; use 'train' or 'test'")
    gens = _TRAIN_GENERATORS if split == "train" else _TEST_GENERATORS
    allowed = frozenset(TRAIN_TOOLS if split == "train" else HOLDOUT_TOOLS)
    rng = random.Random(seed)
    tasks = [rng.choice(gens)(rng) for _ in range(n)]
    for t in tasks:
        if frozenset(t.tools) - allowed:
            raise AssertionError(f"split={split} task exposes tools outside {allowed}: {t.tools}")
        if split == "test" and frozenset(t.tools) & frozenset(TRAIN_TOOLS):
            raise AssertionError(f"holdout task must not expose train tools: {t.tools}")
    return tasks


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
    """Dispatch a tool call. Only tools advertised on `task.tools` are allowed."""
    name = call.get("name")
    args = call.get("arguments") or {}
    if name not in task.tools:
        return f"ERROR: unknown tool '{name}'"
    tool = TOOLS.get(name)
    if tool is None:
        return f"ERROR: unknown tool '{name}'"
    return tool.fn(args, task)


@dataclass
class TrajectoryStats:
    valid_tool_calls: int = 0  # parsed calls dispatched to a registered tool
    malformed_tool_calls: int = 0  # invalid JSON, missing name, or turn with no tool_call/answer
    unknown_tool_calls: int = 0  # parseable calls whose name is not a registered tool
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

def build_system_prompt(tool_names: tuple[str, ...] | list[str] | None = None) -> str:
    """Render the system prompt advertising `tool_names` (defaults to base tools)."""
    if tool_names is None:
        tool_names = BASE_TOOLS
    schema = json.dumps([TOOLS[n].schema for n in tool_names], indent=2)
    return (
        "You are a helpful assistant with access to the following tools.\n\n"
        f"Available tools:\n{schema}\n\n"
        "To call a tool, emit exactly one tool call per assistant turn:\n"
        '  <tool_call>{"name": "<tool_name>", "arguments": {...}}</tool_call>\n'
        "You will receive the result in a tool message wrapped in <tool_response>...</tool_response>.\n"
        "When you have the final numeric answer, output it as:\n"
        "  <answer>INTEGER</answer>\n"
        "Use the tools rather than guessing. Do not invent values."
    )


if __name__ == "__main__":
    # Smoke test both splits + verify the held-out tools dispatch and score.
    for split in ("train", "test"):
        print(f"===== split={split} =====")
        for t in sample_tasks(3, seed=0, split=split):
            print("FAMILY:", t.meta.get("family"), "| TOOLS:", list(t.tools))
            print("PROMPT:", t.prompt)
            print("KB:", t.kb)
            print("ANSWER:", t.answer)
            print("---")
    # Holdout tool dispatch sanity (train tool names must fail).
    demo = sample_tasks(1, seed=1, split="test")[0]
    keys = demo.meta["keys"]
    rr = run_tool({"name": "record_read", "arguments": {"record_id": keys[0]}}, demo)
    print(f"record_read -> {rr}")
    bad = run_tool({"name": "kv_lookup", "arguments": {"key": keys[0]}}, demo)
    assert "unknown tool" in bad, bad
    name = demo.tools[-1]
    args = ({"a": demo.kb[keys[0]], "b": demo.kb[keys[1]]} if name in ("gcd", "lcm")
            else {"n": demo.kb[keys[0]]})
    print(f"dispatch {name}{args} -> {run_tool({'name': name, 'arguments': args}, demo)} "
          f"(answer={demo.answer})")
