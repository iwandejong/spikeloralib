import argparse
import json
import math
import os

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model

from spikelora_peft import average_sparsity, enable_spikelora, patch_peft_for_spikelora

TARGET_MODULES = ["q_proj", "v_proj"]

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_seq_length", type=int, default=256)

    p.add_argument("--use_spikelora", action="store_true")
    p.add_argument("--spikelora_v_threshold", type=float, default=0.1)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=float, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--quantize", type=lambda s: s.lower() not in ("0", "false", "no"), default=torch.cuda.is_available())

    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    return p


def build_base_model(args, config_override=None):
    if config_override is not None:
        from transformers import LlamaForSequenceClassification

        return LlamaForSequenceClassification(config_override)

    quantization_kwargs = {}
    if args.quantize:
        if not torch.cuda.is_available():
            print("WARNING: --quantize is on but no CUDA GPU is available; bitsandbytes' "
                  "4-bit loading will likely fail on CPU/MPS. This is the real 4-bit QLoRA "
                  "setup from the paper and needs a CUDA GPU -- use --quantize false for a "
                  "full-precision approximation instead.")
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        quantization_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    if not args.quantize:
        quantization_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, **quantization_kwargs,
    )
    if args.quantize:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def apply_lora_or_spikelora(model, args):
    patch_peft_for_spikelora()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=True,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    if args.use_spikelora:
        n_enabled = enable_spikelora(model, v_threshold=args.spikelora_v_threshold)
        print(f"SpikeLoRA enabled on {n_enabled} (module, adapter) pairs (v_threshold={args.spikelora_v_threshold})")

    return model


def build_model(args, config_override=None):
    model = build_base_model(args, config_override=config_override)
    return apply_lora_or_spikelora(model, args)


def _tokenize(tokenizer, max_seq_length):
    def fn(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=max_seq_length)

    return fn


def build_compute_metrics():
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"matthews_correlation": matthews_corrcoef(p.label_ids, preds)}

    return compute_metrics


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(args)
    model.config.pad_token_id = tokenizer.pad_token_id

    raw_datasets = load_dataset("nyu-mll/glue", "cola")
    encoded = raw_datasets.map(
        _tokenize(tokenizer, args.max_seq_length),
        batched=True, remove_columns=[c for c in raw_datasets["train"].column_names if c != "label"],
    )
    train_dataset = encoded["train"]
    eval_dataset = encoded["validation"]
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.num_epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=build_compute_metrics(),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    results = {"eval_matthews_correlation": metrics.get("eval_matthews_correlation")}
    if args.use_spikelora:
        results["eval_sparsity"] = average_sparsity(model)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
