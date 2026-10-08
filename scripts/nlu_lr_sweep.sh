#!/usr/bin/env bash
# LoRA vs SpikeLoRA on CoLA across learning rates. (Paper Table 2.)
cd "$(dirname "${BASH_SOURCE[0]}")"
source nlu_common.sh

TASK=cola
LRS="1e-4 3e-4 5e-4 7e-4"

for lr in $LRS; do
  for seed in $SEEDS; do
    run_glue \
      --task_name "$TASK" --output_dir "$OUT_ROOT/lr-sweep/lora-lr${lr}-seed${seed}" --seed "$seed" \
      --lora_r 8 --use_rslora \
      --learning_rate "$lr" \
      --per_device_train_batch_size "$(task_bsz "$TASK")" \
      --num_train_epochs "$(task_epochs "$TASK")"

    run_glue \
      --task_name "$TASK" --output_dir "$OUT_ROOT/lr-sweep/spikelora-lr${lr}-seed${seed}" --seed "$seed" \
      --use_spikelora --v_threshold 0.1 \
      --lora_r 8 --use_rslora \
      --learning_rate "$lr" \
      --per_device_train_batch_size "$(task_bsz "$TASK")" \
      --num_train_epochs "$(task_epochs "$TASK")"
  done
done
