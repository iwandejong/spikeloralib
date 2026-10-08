import argparse
import json
import math
import os

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from model_patch import DEBERTA_TARGET_MODULES, apply_lora, average_sparsity, mark_trainable, module_sparsity

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}
NUM_LABELS = {"mnli": 3, "stsb": 1}
EVAL_SPLIT = {"mnli": "validation_matched"}


class SpikeLoRATrainer(Trainer):
    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        prefix = kwargs.get("metric_key_prefix", "eval")

        sparsity = average_sparsity(self.model)
        if sparsity is not None:
            metrics[f"{prefix}_sparsity"] = sparsity

        train_loss = self._last_train_loss()
        loss_key = f"{prefix}_loss"
        if train_loss is not None and loss_key in metrics:
            metrics[f"{prefix}_generalisation_gap"] = metrics[loss_key] - train_loss

        self.log(metrics)
        return metrics

    def _last_train_loss(self):
        for entry in reversed(self.state.log_history):
            if "loss" in entry:
                return entry["loss"]
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", required=True, choices=sorted(TASK_TO_KEYS))
    p.add_argument("--model_name_or_path", default="microsoft/deberta-v3-base")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_seq_length", type=int, default=256)

    p.add_argument("--use_spikelora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=None)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--use_rslora", action="store_true")
    p.add_argument("--v_threshold", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--target_modules", nargs="+", default=None)

    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--num_train_epochs", type=float, default=10)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    return p


def preprocess(task_name, tokenizer, max_seq_length):
    key1, key2 = TASK_TO_KEYS[task_name]

    def fn(examples):
        texts = (examples[key1],) if key2 is None else (examples[key1], examples[key2])
        return tokenizer(*texts, truncation=True, max_length=max_seq_length)

    return fn


def _glue_metric(task_name, preds, labels):
    if task_name == "cola":
        return {"matthews_correlation": matthews_corrcoef(labels, preds)}
    if task_name == "stsb":
        pearson = pearsonr(preds, labels)[0]
        spearman = spearmanr(preds, labels)[0]
        return {"pearson": pearson, "spearmanr": spearman}
    if task_name in ("mrpc", "qqp"):
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}
    return {"accuracy": accuracy_score(labels, preds)}


def build_compute_metrics(task_name):
    is_regression = task_name == "stsb"

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        return _glue_metric(task_name, preds, p.label_ids)

    return compute_metrics


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    if args.lora_alpha is None:
        args.lora_alpha = math.sqrt(args.lora_r) if args.use_rslora else float(args.lora_r)

    raw_datasets = load_dataset("nyu-mll/glue", args.task_name)
    num_labels = NUM_LABELS.get(args.task_name, 2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=num_labels,
        problem_type="regression" if args.task_name == "stsb" else None,
        torch_dtype=torch.float32,
    )

    replaced = apply_lora(
        model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        use_rslora=args.use_rslora, use_spikelora=args.use_spikelora, v_threshold=args.v_threshold,
        tau=args.tau, target_modules=args.target_modules,
    )
    print(f"Patched {len(replaced)} linear layers with {'SpikeLoRA' if args.use_spikelora else 'LoRA'} (r={args.lora_r}, alpha={args.lora_alpha:.3g}):")
    for name in replaced:
        print(f"{name}")
    mark_trainable(model)

    encoded = raw_datasets.map(
        preprocess(args.task_name, tokenizer, args.max_seq_length),
        batched=True, remove_columns=[c for c in raw_datasets["train"].column_names if c != "label"],
    )
    eval_split = EVAL_SPLIT.get(args.task_name, "validation")
    train_dataset = encoded["train"]
    eval_dataset = encoded[eval_split]
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    steps_per_epoch = math.ceil(len(train_dataset) / args.per_device_train_batch_size)
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.num_train_epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        seed=args.seed,
        report_to=[],
    )

    trainer = SpikeLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=build_compute_metrics(args.task_name),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if args.use_spikelora:
        with open(os.path.join(args.output_dir, "module_sparsity.json"), "w") as f:
            json.dump(module_sparsity(model), f, indent=2)


if __name__ == "__main__":
    main()
