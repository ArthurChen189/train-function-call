# Qwen3-0.6B multi-turn tool-use: SFT → GRPO → Eval

Minimal end-to-end pipeline that takes `Qwen/Qwen3-0.6B` from base → instruction-tuned
on tool-use trajectories → online RL with grounded rewards → evaluated against a
held-out task suite. Designed to close the loop in ~3 hours on a single 24 GB GPU.

```
src/
  env.py       # Tool-use env (train tools + held-out eval tools) + reward
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

**Headline eval** (50 tasks per split): on the **train** split (base tools,
in-domain) base 0% → SFT 54% → RL 96% success; on the **test** holdout split
(disjoint tools: `record_read`, `gcd`, `lcm`, `digit_sum`) base 4% → SFT 92% →
RL 98%. See [Train vs eval tools](#train-vs-eval-tools) below.
Full metrics in [Results](#results) below.

---

## The pipeline

### 1. Environment with verifiable rewards (`src/env.py`)

#### Train vs eval tools

| Split   | When used                         | Tools exposed per task (disjoint sets) | Task families                                      |
| ------- | --------------------------------- | -------------------------------------- | -------------------------------------------------- |
| `train` | SFT env demos, GRPO, eval in-domain | `calculator`, `kv_lookup`              | 2-key / 3-key arithmetic after lookups             |
| `test`  | Eval only (holdout)               | `record_read` + one of `gcd`/`lcm`/`digit_sum` | Read records → GCD/LCM, or read one record → digit sum |

Train and holdout tool **names do not overlap** (`kv_lookup` vs `record_read` are
the same lookup semantics under different schemas).

GRPO always samples `split="train"`. Eval runs both splits by default
(`--splits train test`) so you get in-domain numbers *and* whether the model
transfers the tool-calling *behaviour* to brand-new tools it never saw in RL.

Base tools (deterministic):


| Tool         | Signature         | Behaviour                                                        |
| ------------ | ----------------- | ---------------------------------------------------------------- |
| `calculator` | `expression: str` | Evaluates a safe arithmetic AST (`+ - * / **`, parens).          |
| `kv_lookup`  | `key: str`        | Looks up `key` in a per-task dict, returns value or `NOT_FOUND`. |

Holdout eval tools (disjoint from train; never advertised during GRPO):


| Tool          | Signature                    | Behaviour                              |
| ------------- | ---------------------------- | -------------------------------------- |
| `record_read` | `record_id: str`             | Same KB lookup as `kv_lookup`, new name/arg. |
| `gcd`         | `a: int`, `b: int`           | Greatest common divisor.               |
| `lcm`         | `a: int`, `b: int`           | Least common multiple.                 |
| `digit_sum`   | `n: int`                     | Sum of decimal digits of `n`.          |

Each task advertises only its own tool list in the system prompt; calling a train
tool (e.g. `calculator` or `kv_lookup` on a holdout task) returns
`ERROR: unknown tool`.

**Train** tasks are generated programmatically. Two families:

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
*format/behaviour* (and, in practice, transfers ~54% zero-shot success), while
the env-specific completion recipe — reliable `<answer>` emission, correct
arithmetic, efficient turn count — comes from RL.

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

50 tasks per split (`train` + `test` by default), greedy decoding, disjoint seed
`999_001` from SFT/RL. Run pre-SFT (base), post-SFT, and post-RL sequentially;
print one comparison table per split; dump full trajectories (first 3 per
condition) to `results/eval_summary_trained_on_apigen.json`.

```bash
# In-domain only (matches what GRPO trained on):
python3 -m src.eval --adapters rl=checkpoints/rl/final --splits train

# Held-out tools only:
python3 -m src.eval --adapters rl=checkpoints/rl/final --splits test
```

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
| `APIGen-MT-5k` for SFT (not synthetic)                           | Real, diverse, human-verified multi-turn tool use; honest train/eval domain gap.                                    | Out-of-domain vs the env's two tools — SFT transfers format and most correctness (incl. held-out tools); RL still needed for completion and polish. |
| Convert to the env's tag protocol (not Qwen-native `tool_calls`) | SFT emit format == what `rollout.py` parses, no schema-translation drift.                                           | Doesn't exercise Qwen's structured tool-call API.                                     |
| `transformers` + `peft`, not Unsloth                             | One fewer moving dep at 0.6B scale.                                                                                 | ~2× SFT speed.                                                                        |
| REINFORCE form of GRPO (no ε clip)                               | Simpler, stable at this LR.                                                                                         | Robustness to larger step sizes.                                                      |
| Single GPU, sequential rollouts                                  | Simplest correct implementation.                                                                                    | ~4-8× wall via vLLM-based rollout.                                                    |


## Results

Full pipeline run with defaults (512 APIGen trajectories × 2 SFT epochs, 40 GRPO
steps, 50 eval tasks per split, greedy decode). Saved to
`results/eval_summary_trained_on_apigen.json`.

### Split = `train` (in-domain, base tools)

| metric                    | base   | sft    | rl     |
| ------------------------- | ------ | ------ | ------ |
| success_rate              | 0.0000 | 0.5400 | 0.9600 |
| mean_reward               | −0.589 | 0.651  | 1.173  |
| answer_emit_rate          | 0.1400 | 0.6000 | 1.0000 |
| mean_turns                | 5.70   | 5.24   | 3.26   |
| mean_valid_tool_calls     | 0.50   | 4.22   | 2.26   |
| mean_malformed_or_unknown | 5.06   | 0.42   | 0.00   |

### Split = `test` (held-out tools)

| metric                    | base   | sft    | rl     |
| ------------------------- | ------ | ------ | ------ |
| success_rate              | 0.0400 | 0.9200 | 0.9800 |
| mean_reward               | −0.622 | 1.078  | 1.157  |
| answer_emit_rate          | 0.1200 | 0.9200 | 1.0000 |
| mean_turns                | 5.80   | 3.96   | 3.46   |
| mean_valid_tool_calls     | 0.32   | 2.88   | 2.40   |
| mean_malformed_or_unknown | 5.36   | 0.16   | 0.06   |

**Base** — Near-zero success on both splits. The model occasionally emits
something tool-shaped but mostly produces malformed output (~5 bad calls per
episode) and rarely wraps a final answer (12–14% `answer_emit_rate`).

**+ SFT (`APIGen-MT-5k`, out-of-domain)** — Format transfer is clear on train:
`mean_malformed_or_unknown` drops 5.1 → 0.4 and valid tool calls jump to 4.2.
SFT also carries strong procedural transfer — **54% success** on in-domain env
tasks it was never trained on, and **92%** on held-out tools (`record_read`,
`gcd`/`lcm`/`digit_sum`) it has never seen. The remaining in-domain gap is
mostly *completion*: only 60% of train rollouts emit `<answer>`, and episodes
run long (5.2 turns, often over-calling relative to the 2–3 step recipes).

**+ GRPO (40 steps)** — RL closes what SFT left open on train. Success 54% →
96%, `answer_emit_rate` → 1.0, malformed calls → 0. Mean turns fall to 3.3 and
tool calls to 2.3 — the policy learns the env's lookup→calculate→answer recipe
rather than spamming tools. `mean_reward` tracks the same story (−0.59 → 1.17).
On the holdout split RL adds a smaller lift (92% → 98%) because SFT already
generalises the tool-calling behaviour to unseen tool names. In-loop RL success
in `checkpoints/rl/train_log.jsonl` rises from ~44% at step 0 to ~94% by step
15.

So the honest out-of-domain setup works: SFT teaches the tag protocol and gets
most of the way on correctness (including on disjoint eval tools); GRPO does the
env-specific grounding and answer completion. The failure modes I was watching
for did *not* show up here — `mean_reward` and `success_rate` moved together,
and KL stayed bounded (~0.02 by step 19).

## What I'd do with a week instead of an afternoon

1. **More tools, more task families.** The holdout split already uses disjoint
   tools (`record_read`, `gcd`, `lcm`, `digit_sum`); extend with web-search stubs,
   date arithmetic, unit conversion, etc.
2. **Broaden + align the SFT mix.** SFT now uses `APIGen-MT-5k` (retail/airline)
  and already reaches 54% in-domain / 92% held-out-tool zero-shot success — but
  answer completion and turn efficiency on the train split still need RL. A week's
  version would mix in
  `xlam-function-calling-60k` *and* add a few `calculator`/`kv_lookup`-style
  trajectories to close the remaining domain gap before RL.
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

