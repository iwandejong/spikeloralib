#!/usr/bin/env bash
# Per-step gradient norm, SpikeLoRA vs. LoRA, matched seed, on CoLA. (Paper Fig. 4.)
cd "$(dirname "${BASH_SOURCE[0]}")"
source nlu_common.sh

TASK=cola
SEED="${GRADNORM_SEED:-0}"

run_glue \
  --task_name "$TASK" --output_dir "$OUT_ROOT/gradnorm-comparison/lora" --seed "$SEED" \
  --lora_r 8 --use_rslora \
  --learning_rate "$(task_lr "$TASK")" \
  --per_device_train_batch_size "$(task_bsz "$TASK")" \
  --num_train_epochs "$(task_epochs "$TASK")" \
  --log_history_path "$OUT_ROOT/gradnorm-comparison/lora/log_history.json"

run_glue \
  --task_name "$TASK" --output_dir "$OUT_ROOT/gradnorm-comparison/spikelora" --seed "$SEED" \
  --use_spikelora --v_threshold 0.1 \
  --lora_r 8 --use_rslora \
  --learning_rate "$(task_lr "$TASK")" \
  --per_device_train_batch_size "$(task_bsz "$TASK")" \
  --num_train_epochs "$(task_epochs "$TASK")" \
  --log_history_path "$OUT_ROOT/gradnorm-comparison/spikelora/log_history.json"
