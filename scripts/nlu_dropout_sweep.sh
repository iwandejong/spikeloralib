#!/usr/bin/env bash
# LoRA dropout sweep, LoRA vs. SpikeLoRA, on CoLA. (Paper Fig. 5.)
cd "$(dirname "${BASH_SOURCE[0]}")"
source nlu_common.sh

TASK=cola
DROPOUTS="0.0 0.025 0.05 0.075 0.1"

for p in $DROPOUTS; do
  for seed in $SEEDS; do
    run_glue \
      --task_name "$TASK" --output_dir "$OUT_ROOT/dropout-sweep/lora-p${p}-seed${seed}" --seed "$seed" \
      --lora_r 8 --use_rslora --lora_dropout "$p" \
      --learning_rate "$(task_lr "$TASK")" \
      --per_device_train_batch_size "$(task_bsz "$TASK")" \
      --num_train_epochs "$(task_epochs "$TASK")"

    run_glue \
      --task_name "$TASK" --output_dir "$OUT_ROOT/dropout-sweep/spikelora-p${p}-seed${seed}" --seed "$seed" \
      --use_spikelora --v_threshold 0.1 \
      --lora_r 8 --use_rslora --lora_dropout "$p" \
      --learning_rate "$(task_lr "$TASK")" \
      --per_device_train_batch_size "$(task_bsz "$TASK")" \
      --num_train_epochs "$(task_epochs "$TASK")"
  done
done
