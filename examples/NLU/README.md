# DeBERTaV3-Base + LoRA/SpikeLoRA on GLUE

Fine-tunes [DeBERTaV3-Base](https://huggingface.co/microsoft/deberta-v3-base) on the GLUE
benchmark with plain LoRA and with SpikeLoRA.

## Setup

```bash
pip install -e ../..            # spikeloralib itself
pip install -r requirements.txt
```

## Quick start

```bash
# Plain LoRA on CoLA
python run_glue.py --task_name cola --output_dir runs/cola-lora \
    --lora_r 8 --use_rslora --learning_rate 3e-4 --num_train_epochs 20

# SpikeLoRA on CoLA
python run_glue.py --task_name cola --output_dir runs/cola-spikelora \
    --use_spikelora --v_threshold 0.1 --lora_r 8 --use_rslora \
    --learning_rate 3e-4 --num_train_epochs 20
```

See `scripts/nlu_*.sh` (from the repo root) for the scripts reproducing the paper's
tables/figures; each run writes its raw `eval_metrics.json` under `--output_dir`.
