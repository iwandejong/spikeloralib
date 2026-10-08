# SpikeGPT + LoRA/SpikeLoRA on the Subjectivity Dataset

Fine-tunes [SpikeGPT](https://huggingface.co/ridger/SpikeGPT-OpenWebText-216M) (an RWKV-style
spiking language model) on the Pang & Lee subjectivity dataset with plain LoRA and with
SpikeLoRA.

## Setup

```bash
pip install -e ../..            # spikeloralib itself
pip install -r requirements.txt
```

## Quick start

```bash
# Plain LoRA
python finetune_subj.py --output_dir runs/subj-lora-seed0 --seed 0 \
    --lora_r 16 --lora_alpha 16

# SpikeLoRA
python finetune_subj.py --output_dir runs/subj-spikelora-seed0 --seed 0 \
    --use_spikelora --v_threshold 0.25 --lora_r 16 --lora_alpha 16
```

The first run downloads the SpikeGPT checkpoint and the subjectivity dataset automatically
(cached thereafter). See `scripts/spikegpt_subj_benchmark.sh` (from the repo root) to
reproduce this comparison across 5 seeds (paper Table 4).
