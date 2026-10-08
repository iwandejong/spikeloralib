#!/usr/bin/env bash
# Per-block, per-module-type sparsity for SpikeLoRA on CoLA. (Paper Fig. 3.)
cd "$(dirname "${BASH_SOURCE[0]}")"
source nlu_common.sh

TASK=cola

run_glue \
  --task_name "$TASK" --output_dir "$OUT_ROOT/sparsity-breakdown" --seed "${SPARSITY_SEED:-0}" \
  --use_spikelora --v_threshold 0.1 \
  --lora_r 8 --use_rslora \
  --learning_rate "$(task_lr "$TASK")" \
  --per_device_train_batch_size "$(task_bsz "$TASK")" \
  --num_train_epochs "$(task_epochs "$TASK")"
