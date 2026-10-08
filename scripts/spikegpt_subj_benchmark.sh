#!/usr/bin/env bash
# LoRA vs SpikeLoRA fine-tuning SpikeGPT on the subjectivity dataset. (Paper Table 4.)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPIKEGPT_DIR="$REPO_ROOT/examples/SpikeGPT-NLU"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/SpikeGPT-NLU}"
SEEDS="${SEEDS:-0 1 2 3 4}"
PYTHON="${PYTHON:-python}"

finetune_subj() {
  "$PYTHON" "$SPIKEGPT_DIR/finetune_subj.py" "$@" ${EXTRA_ARGS:-}
}

for seed in $SEEDS; do
  finetune_subj \
    --output_dir "$OUT_ROOT/subj-benchmark/lora-seed${seed}" --seed "$seed" \
    --lora_r 16 --lora_alpha 16

  finetune_subj \
    --output_dir "$OUT_ROOT/subj-benchmark/spikelora-seed${seed}" --seed "$seed" \
    --use_spikelora --v_threshold 0.25 --lora_r 16 --lora_alpha 16
done
