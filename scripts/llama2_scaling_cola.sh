#!/usr/bin/env bash
# 4-bit-quantised Llama2-7B on CoLA, LoRA vs SpikeLoRA, QV projections only. Needs a CUDA
# GPU + bitsandbytes + gated access to meta-llama/Llama-2-7b-hf (set HF_TOKEN).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA2_DIR="$REPO_ROOT/examples/llama2-qlora-scaling"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-hf}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/llama2-qlora-scaling}"
cd "$LLAMA2_DIR"

python finetune_llama_cola.py \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUT_ROOT/scaling-lora" \
    --lora_r 8 --lora_alpha 16 --lora_dropout 0.0 \
    --learning_rate 3e-4 --batch_size 32 --num_epochs 10 \
    --weight_decay 0.01 --max_grad_norm 1.0 \
    --quantize true --seed 0 ${EXTRA_ARGS:-}

python finetune_llama_cola.py \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUT_ROOT/scaling-spikelora" \
    --use_spikelora --spikelora_v_threshold 0.1 \
    --lora_r 8 --lora_alpha 16 --lora_dropout 0.0 \
    --learning_rate 3e-4 --batch_size 32 --num_epochs 10 \
    --weight_decay 0.01 --max_grad_norm 1.0 \
    --quantize true --seed 0 ${EXTRA_ARGS:-}
