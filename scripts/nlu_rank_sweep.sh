#!/usr/bin/env bash
# LoRA vs SpikeLoRA on CoLA across ranks. (Paper Table 1.)
cd "$(dirname "${BASH_SOURCE[0]}")"
source nlu_common.sh

TASK=cola
RANKS="1 2 4 8 16 32 64"

for r in $RANKS; do
  for seed in $SEEDS; do
    run_glue \
      --task_name "$TASK" --output_dir "$OUT_ROOT/rank-sweep/lora-r${r}-seed${seed}" --seed "$seed" \
      --lora_r "$r" --use_rslora \
      --learning_rate "$(task_lr "$TASK")" \
      --per_device_train_batch_size "$(task_bsz "$TASK")" \
      --num_train_epochs "$(task_epochs "$TASK")"

    run_glue \
      --task_name "$TASK" --output_dir "$OUT_ROOT/rank-sweep/spikelora-r${r}-seed${seed}" --seed "$seed" \
      --use_spikelora --v_threshold 0.1 \
      --lora_r "$r" --use_rslora \
      --learning_rate "$(task_lr "$TASK")" \
      --per_device_train_batch_size "$(task_bsz "$TASK")" \
      --num_train_epochs "$(task_epochs "$TASK")"
  done
done
