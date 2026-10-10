#!/usr/bin/env bash
# Re-run the CoLA LoRA vs SpikeLoRA rank experiments at a fixed learning rate.
cd "$(dirname "${BASH_SOURCE[0]}")"
source nlu_common.sh

TASK=cola
LEARNING_RATE=3e-4
RANKS="1 2 4 16 32 64"

for r in $RANKS; do
  for seed in $SEEDS; do
    run_glue \
      --task_name "$TASK" --output_dir "$OUT_ROOT/rank-rerun/lora-r${r}-seed${seed}" --seed "$seed" \
      --lora_r "$r" --use_rslora \
      --learning_rate "$LEARNING_RATE" \
      --per_device_train_batch_size "$(task_bsz "$TASK")" \
      --num_train_epochs "$(task_epochs "$TASK")"

    run_glue \
      --task_name "$TASK" --output_dir "$OUT_ROOT/rank-rerun/spikelora-r${r}-seed${seed}" --seed "$seed" \
      --use_spikelora --v_threshold 0.1 \
      --lora_r "$r" --use_rslora \
      --learning_rate "$LEARNING_RATE" \
      --per_device_train_batch_size "$(task_bsz "$TASK")" \
      --num_train_epochs "$(task_epochs "$TASK")"
  done
done
