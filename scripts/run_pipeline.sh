#!/usr/bin/env bash
# End-to-end pipeline: env smoke test -> SFT -> RL -> eval.
# Designed to fit in ~3 hours on a single 24GB GPU. Tune the knobs at the top.
#
# SFT uses the gated dataset Salesforce/APIGen-MT-5k: accept the terms on its HF
# page and run `huggingface-cli login` once before running this script.
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
SFT_OUT="${SFT_OUT:-checkpoints/sft}"
RL_OUT="${RL_OUT:-checkpoints/rl}"
EVAL_OUT="${EVAL_OUT:-results/eval_summary_trained_on_apigen.json}"

# Small but non-trivial defaults. Override via env vars.
N_SFT="${N_SFT:-128}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"

RL_STEPS="${RL_STEPS:-20}"
RL_PROMPTS="${RL_PROMPTS:-4}" # number of tasks per step
RL_K="${RL_K:-4}" # number of rollouts per prompt

N_EVAL="${N_EVAL:-50}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "== [0/4] env sanity check =="
python -m src.env

# echo "== [1/4] SFT (n=$N_SFT, epochs=$SFT_EPOCHS) =="
# python -m src.sft \
#   --base-model "$BASE_MODEL" \
#   --n-train "$N_SFT" \
#   --epochs "$SFT_EPOCHS" \
#   --out "$SFT_OUT"

echo "== [2/4] Eval base + SFT (sanity check) =="
python -m src.eval \
  --base-model "$BASE_MODEL" \
  --adapters sft="$SFT_OUT/final" \
  --n "$N_EVAL" \
  --greedy \
  --out "results/eval_pre_rl.json"

echo "== [3/4] GRPO RL (steps=$RL_STEPS, prompts=$RL_PROMPTS, k=$RL_K) =="
python -m src.rl \
  --base-model "$BASE_MODEL" \
  --sft-adapter "$SFT_OUT/final" \
  --steps "$RL_STEPS" \
  --prompts-per-step "$RL_PROMPTS" \
  --k-rollouts "$RL_K" \
  --out "$RL_OUT"

echo "== [4/4] Final eval: base vs SFT vs RL =="
python -m src.eval \
  --base-model "$BASE_MODEL" \
  --adapters sft="$SFT_OUT/final" rl="$RL_OUT/final" \
  --n "$N_EVAL" \
  --greedy \
  --out "$EVAL_OUT"

echo "Done. See $EVAL_OUT and $RL_OUT/train_log.jsonl."
