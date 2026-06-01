# Qwen3-0.6B multi-turn tool-use: SFT → GRPO → Eval

Minimal end-to-end pipeline that takes `Qwen/Qwen3-0.6B` from base → instruction-tuned
on tool-use trajectories → online RL with grounded rewards → evaluated against a
held-out task suite. Designed to close the loop in ~3 hours on a single 24 GB GPU.

```
src/
  env.py       # Two-tool environment + programmatic task generator + reward
  rollout.py   # Multi-turn rollout (shared by RL sampler and eval)
  data.py      # SFT data: Salesforce/APIGen-MT-5k (ShareGPT) -> tag protocol
  sft.py       # LoRA SFT via TRL SFTTrainer
  rl.py        # Custom GRPO loop (REINFORCE w/ group baseline + KL-to-ref)
  eval.py      # Held-out evaluation; emits base vs SFT vs RL table
scripts/
  run_pipeline.sh
```

Run everything:

```bash
bash scripts/run_pipeline.sh
```

Knobs (env vars): `N_SFT`, `SFT_EPOCHS`, `RL_STEPS`, `RL_PROMPTS`, `RL_K`, `N_EVAL`.

---

## The pipeline

### 1. Environment with verifiable rewards (`src/env.py`)

Two tools, both completely deterministic:


| Tool         | Signature         | Behaviour                                                        |
| ------------ | ----------------- | ---------------------------------------------------------------- |
| `calculator` | `expression: str` | Evaluates a safe arithmetic AST (`+ - * / **`, parens).          |
| `kv_lookup`  | `key: str`        | Looks up `key` in a per-task dict, returns value or `NOT_FOUND`. |


Tasks are generated programmatically. Two families:

- *2-key* — “look up `alpha` and `bravo`, take their sum, multiply by 4.”
- *3-key* — “look up three keys, compute `(v1 + v2) * v3`.”

Each task carries its own `kb` (with distractor keys) and a pre-computed integer
answer. Because the answer is computed by the generator, scoring a rollout is
just `int(parsed_answer) == task.answer` — no graders, no LLM judges, no flakiness.

**Tool-call protocol** (deliberately decoupled from any chat template so we can
parse it the same way at SFT, RL, and eval time):

```text
<tool_call>{"name": "kv_lookup", "arguments": {"key": "alpha"}}</tool_call>
<tool_response>17</tool_response>
...
<answer>42</answer>
```

### 2. SFT (`src/sft.py`, `src/data.py`)

- LoRA (`r=16, α=32`) on all attention + MLP projections of Qwen3-0.6B.
- TRL `SFTTrainer` with `assistant_only_loss=True` — gradient only flows on
the assistant tokens, never on system / user / tool messages. (TRL auto-patches
Qwen3's chat template with `{% generation %}` markers to make this work.)
- Dataset is `**Salesforce/APIGen-MT-5k**` — 5,000 human-verified multi-turn
tool-use trajectories in the retail/airline (τ-bench) domains, in ShareGPT
format. `src/data.py` converts each trajectory into the **same tag protocol
the env speaks**: the tool call lives as text inside the assistant turn
(`<tool_call>{...}</tool_call>`) and the result comes back as a `tool`-role
message wrapped in `<tool_response>...</tool_response>`. So the SFT emit
format is byte-identical to what `rollout.py` parses at RL/eval time, even
though the *tools themselves* differ (APIGen = retail/airline; env =
`calculator`/`kv_lookup`).
- Conversion details: the `think` pseudo-tool (a τ-bench reasoning scaffold,
always paired with an empty observation) is dropped so we never teach the
model to call a tool the env would penalise as unknown. A canonical 256-row
held-out split (fixed seed) backs the in-loop eval, disjoint from training.
- **Length handling:** APIGen trajectories are long — the shared domain-policy +
tools system prompt alone is ~3.6k tokens, and full trajectories run ~5k
(median). We train at `--max-seq-len 6144` (captures ~84% of trajectories in
full) and **filter out** anything longer rather than right-truncating it, so
we never train on a trajectory whose later assistant turns were cut. Gradient
checkpointing keeps the long sequences in memory.

**Why this is the more honest setup.** The old pipeline synthesised SFT data
from the *same* env, so SFT alone nearly solved the task. Training on
out-of-domain real data instead means SFT teaches the general tool-calling
*format/behaviour*, and the env-specific bits (which two tools exist, the
`<answer>` tag) have to come from the env's system prompt + RL — a more
realistic test of whether the loop actually transfers.

Defaults: 512 trajectories × 2 epochs.

### 3. GRPO RL (`src/rl.py`)

I wrote the loop directly rather than using `trl.GRPOTrainer`, because TRL's
GRPOTrainer assumes a single-turn `(prompt → completion → scalar reward)`
setup. Our trajectories have multiple assistant segments interleaved with
tool outputs, and we need policy log-probs across **all** of them.

Notable choices:

- **No PPO ratio clip.** We're using the REINFORCE form of GRPO (i.e. ε→∞).
With LoRA and a tiny LR (`1e-5`) it's stable for the scales we're running,
and it cuts ~30 lines of implementation. Easy to swap in a ratio clip later.
- **Per-token mean log-prob**, not sum. With variable-length multi-turn
rollouts the un-normalised sum just biases the loss toward longer trajectories.
- **KL is the cheap one-sample estimator** `log π(τ) − log π_ref(τ)` averaged
per token. Faster than the unbiased `e^{Δ} − Δ − 1` form and good enough for
the small step sizes here.
- **LoRA-only updates.** The reference is the same base + saved SFT adapter,
which keeps the KL pull anchored to “what SFT learned”, not “what the base
model thinks”.

Defaults: 40 steps × 4 prompts × 4 rollouts = 640 trajectories. ~90 min wall.

### 4. Reward design

```text
+1.0   correct final answer
+0.1   emitted ANY parseable <answer> (format shaping)
+0.05  per valid tool call, capped at 0.2
−0.1   per malformed/unknown tool call
−0.05  per turn over the task's target turn count
```

How I thought about hacking risk:

- **Format-only farming** — the `+0.1` for emitting a numeric answer is bounded
and dominated by correctness (`+1.0`). I checked this by hand on a few seeds:
guessing a random integer beats “refuse to answer” by 0.1, but loses 1.0 to
the “use the tools and get it right” strategy.
- **Tool-call spam** — capped at 0.2 total bonus, and the turn penalty kicks in
beyond `target_turns`, so spamming actively hurts past 3-4 calls.
- **Memorisation** — task values are sampled fresh per task `seed`. Eval seeds
(`999_001+`) are disjoint from SFT (`0..N`) and RL (`42..42+steps`), so the
model has to learn the procedure, not the answers.
- **Sparse signal early on** — the `+0.1` answer-format shaping is the
workaround for cold-start sparsity. Without it, a freshly-RL'd model that
doesn't yet emit `<answer>` gets 0 reward on every rollout in a group →
zero variance → zero advantage → no signal. With it, even partial credit
carries information about format.

### 5. Eval (`src/eval.py`)

50 held-out tasks, greedy decoding. Run pre-SFT (base), post-SFT, and post-RL
sequentially; print a side-by-side table; dump full trajectories (first 3 per
condition) to `results/eval_summary.json`.

Metrics reported:

- `success_rate` — fraction with `parsed_answer == ground_truth`. The headline.
- `mean_reward` — same reward function as training. Sanity-checks the loop.
- `answer_emit_rate` — did the model ever output `<answer>...</answer>`?
- `mean_turns` / `mean_valid_tool_calls` / `mean_malformed_or_unknown` — behavioural
signals to spot regressions even when success rate is flat.

---

## Trade-offs and scope cuts


| Choice                                                           | Why                                                                                                                 | What I gave up                                                                        |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Qwen3-0.6B                                                       | Fits comfortably with LoRA, fast iter, native tool template.                                                        | A 7B+ would close the loop with much less RL noise.                                   |
| LoRA, not full FT                                                | 4-hour budget.                                                                                                      | A small ceiling on what RL can move.                                                  |
| Custom env, not BFCL / τ-bench / Open Trajectory Gym             | Wanted 100% verifiable, no-graders rewards. BFCL needs OpenAI-style scoring infra and τ-bench needs a longer setup. | External benchmark comparability.                                                     |
| Custom GRPO loop, not `trl.GRPOTrainer`                          | TRL's GRPOTrainer is single-turn. Multi-turn requires bending it.                                                   | TRL's PPO ratio clip, batched generation, vLLM integration.                           |
| `APIGen-MT-5k` for SFT (not synthetic)                           | Real, diverse, human-verified multi-turn tool use; honest train/eval domain gap.                                    | Out-of-domain vs the env's two tools, so SFT no longer "almost solves" the eval task. |
| Convert to the env's tag protocol (not Qwen-native `tool_calls`) | SFT emit format == what `rollout.py` parses, no schema-translation drift.                                           | Doesn't exercise Qwen's structured tool-call API.                                     |
| `transformers` + `peft`, not Unsloth                             | One fewer moving dep at 0.6B scale.                                                                                 | ~2× SFT speed.                                                                        |
| REINFORCE form of GRPO (no ε clip)                               | Simpler, stable at this LR.                                                                                         | Robustness to larger step sizes.                                                      |
| Single GPU, sequential rollouts                                  | Simplest correct implementation.                                                                                    | ~4-8× wall via vLLM-based rollout.                                                    |


## Honest evaluation: what I expect to see, and why

Now that SFT data is **out-of-domain** (`APIGen-MT-5k` = retail/airline tools)
relative to the env (`calculator`/`kv_lookup`), the shape of the result shifts —
SFT teaches *format*, RL has to teach *correctness*. A 24-example × 1-epoch SFT
smoke run already shows the format transfer cleanly:


| Stage                  | success_rate   | answer_emit_rate | mean_valid_tool_calls | mean_malformed/unknown | Why                                                                                                                                                                                                     |
| ---------------------- | -------------- | ---------------- | --------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| base (Qwen3-0.6B)      | ~0.00          | ~0.0             | ~0.0                  | high                   | Base model rarely emits the `<tool_call>{...}</tool_call>` schema; mostly malformed.                                                                                                                    |
| + SFT (`APIGen-MT-5k`) | low (~0 – 0.3) | ~1.0             | ~2–3                  | low                    | The tag protocol + "call a tool, then answer" behaviour transfers from the airline/retail data to the env's two tools, but the model has never seen *these* tools or arithmetic, so correctness is low. |
| + GRPO (40 steps)      | bump from SFT  | 1.0              | ~3                    | low                    | RL is now doing the real work — grounding the transferred format on the env's actual tools + reward. 4 hours is not enough to fully converge.                                                           |


(The base→SFT format jump above is from an actual smoke run; the absolute
`success_rate` will depend on the full SFT budget and RL steps.)

A flat or even slightly-negative RL delta on `success_rate` is *expected* and
fine — the methodology question is whether the loop runs and whether the
signal moves in the right direction in `mean_reward` and `mean_malformed_or_unknown`.
The training log at `checkpoints/rl/train_log.jsonl` is the better lens than
the post-eval delta at this budget.

The realistic failure mode to watch for:

- `mean_reward` rising while `success_rate` is flat ⇒ format shaping is being
exploited. Mitigation already in place: shaping bonus is bounded.
- `kl` running away → policy collapse onto a single mode. Mitigations: KL
penalty, grad clip, low LR.

## What I'd do with a week instead of an afternoon

1. **More tools, more task families.** Add a web-search stub, a date-arithmetic
  tool, a unit-conversion tool. Mix task families so RL can't overfit a
   single recipe.
2. **Broaden + align the SFT mix.** SFT now uses `APIGen-MT-5k` (retail/airline).
  A week's version would mix in `xlam-function-calling-60k` *and* add a few
   `calculator`/`kv_lookup`-style trajectories so the SFT distribution overlaps
   the env's tools — closing the domain gap that currently leaves correctness
   almost entirely to RL.
3. **vLLM-backed rollouts.** Drop in vLLM for the sampling half of GRPO — easily
  8× faster wall-clock, makes K=16 rollouts/prompt feasible.
4. **Real external benchmark.** Add a τ-bench or BFCL slice as a *secondary*
  eval, kept disjoint from training. Report both in-domain and out-of-domain.
5. **Ablate the reward.** Run the same RL with (a) only correctness reward,
  (b) full shaped reward, (c) shaped + tool-call cost penalty. Compare.
6. **Process supervision experiment.** Score each turn (valid tool call?
  sensible arg?), not just the trajectory. Compare per-turn vs per-trajectory
   credit assignment.
7. **PPO ratio clip + GAE-style off-policy correction** so we can run multiple
  gradient steps per rollout batch without policy collapse.
8. **Reference-model schedule.** Re-anchor `π_ref ← π_current` every N steps so
  RL can keep moving without an ever-growing KL.

## Reproducibility notes

- Seeds: SFT draws from `APIGen-MT-5k` via a fixed canonical shuffle
(`_SPLIT_SEED=1234`) that reserves a disjoint 256-row eval split; the env eval
uses `seed=999_001` and RL uses `seed=42 + step`. The env eval tasks are
freshly generated and never seen during SFT (different domain entirely) or RL.
- All assistant-only loss masking is delegated to TRL's chat-template-aware
collator (`assistant_only_loss=True`) — no hand-rolled label masking.
- The env smoke test runs first (`python -m src.env`) so format regressions
in `Task`/reward code blow up before we waste GPU time.

