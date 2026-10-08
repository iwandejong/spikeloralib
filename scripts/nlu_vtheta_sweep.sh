#!/usr/bin/env bash
# V_theta sweep on CoLA, SpikeLoRA only. (Paper Fig. 2.)
cd "$(dirname "${BASH_SOURCE[0]}")"
source nlu_common.sh

TASK=cola
THRESHOLDS="0.0 0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0"

for vth in $THRESHOLDS; do
  for seed in $SEEDS; do
    out="$OUT_ROOT/vtheta-sweep/vth${vth}-seed${seed}"
    run_glue \
      --task_name "$TASK" --output_dir "$out" --seed "$seed" \
      --use_spikelora --v_threshold "$vth" \
      --lora_r 8 --use_rslora \
      --learning_rate "$(task_lr "$TASK")" \
      --per_device_train_batch_size "$(task_bsz "$TASK")" \
      --num_train_epochs "$(task_epochs "$TASK")"
  done
done
