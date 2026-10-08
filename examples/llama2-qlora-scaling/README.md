# Llama2-7B + QLoRA/SpikeLoRA on CoLA

Fine-tunes a 4-bit-quantised [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) on
CoLA (GLUE) with QLoRA, applied to the Q/V attention projections, with and without
SpikeLoRA's spiking gate.

Needs a CUDA GPU, a working GPU build of
[`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes), and gated access
to `meta-llama/Llama-2-7b-hf` (set `HF_TOKEN`).

## Setup

```bash
pip install -e ../.. # spikeloralib itself
pip install -r requirements.txt
```

## Quick start

```bash
./scripts/llama2_scaling_cola.sh   # from the repo root
```

Equivalently, call the script directly:

```bash
# Plain QLoRA
python finetune_llama_cola.py --output_dir runs/scaling-lora \
    --lora_r 8 --lora_alpha 16 --learning_rate 3e-4 --batch_size 32 --num_epochs 10

# SpikeLoRA on top of the same QLoRA config
python finetune_llama_cola.py --output_dir runs/scaling-spikelora \
    --use_spikelora --spikelora_v_threshold 0.1 \
    --lora_r 8 --lora_alpha 16 --learning_rate 3e-4 --batch_size 32 --num_epochs 10
```

Pass `--quantize false` to skip 4-bit loading (full precision, no `bitsandbytes` required) --
useful for a quick CPU sanity check, not for reproducing the paper's numbers.
