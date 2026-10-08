#!/usr/bin/env bash
# LoRA vs SpikeLoRA across the full GLUE benchmark. (Paper Table 3.)
cd "$(dirname "${BASH_SOURCE[0]}")"
source nlu_common.sh

TASKS="cola sst2 mrpc stsb mnli qnli rte qqp"

for task in $TASKS; do
  for seed in $SEEDS; do
    run_glue \
      --task_name "$task" --output_dir "$OUT_ROOT/glue-benchmark/lora-${task}-seed${seed}" --seed "$seed" \
      --lora_r 8 --use_rslora \
      --learning_rate "$(task_lr "$task")" \
      --per_device_train_batch_size "$(task_bsz "$task")" \
      --num_train_epochs "$(task_epochs "$task")"

    run_glue \
      --task_name "$task" --output_dir "$OUT_ROOT/glue-benchmark/spikelora-${task}-seed${seed}" --seed "$seed" \
      --use_spikelora --v_threshold 0.1 \
      --lora_r 8 --use_rslora \
      --learning_rate "$(task_lr "$task")" \
      --per_device_train_batch_size "$(task_bsz "$task")" \
      --num_train_epochs "$(task_epochs "$task")"
  done
done
