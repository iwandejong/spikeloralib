import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from data import OBJECTIVE_FILE, SUBJECTIVE_FILE, SubjDataset, download_subj, load_subj, make_collate_fn, train_val_split
from lora_patch import apply_lora, average_sparsity, mark_trainable, reset_all_lif
from model import GPT, GPTConfig, RWKV_Init

NUM_CLASSES = 2
CHECKPOINT_FILE = "SpikeGPT-216M.pth"
VOCAB_SIZE = 50277
CTX_LEN = 1024
N_LAYER = 18
N_EMBD = 768
TOKENIZER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "20B_tokenizer.json")
DEFAULT_DATA_DIR = os.path.expanduser("~/.cache/spikelora/subj")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(model_name_or_path: str, n_layer: int, n_embd: int, vocab_size: int, ctx_len: int):
    config = GPTConfig(vocab_size=vocab_size, ctx_len=ctx_len, n_layer=n_layer, n_embd=n_embd, model_type="RWKV")
    model = GPT(config)

    if model_name_or_path == "random":
        RWKV_Init(model, config)
    else:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo_id=model_name_or_path, filename=CHECKPOINT_FILE)
        state_dict = torch.load(local_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    model.head = nn.Linear(config.n_embd, NUM_CLASSES, bias=False)
    nn.init.normal_(model.head.weight, mean=0.0, std=0.02)
    return model, config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="ridger/SpikeGPT-OpenWebText-216M")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_layer", type=int, default=N_LAYER)
    p.add_argument("--n_embd", type=int, default=N_EMBD)
    p.add_argument("--vocab_size", type=int, default=VOCAB_SIZE)
    p.add_argument("--ctx_len", type=int, default=CTX_LEN)

    p.add_argument("--use_spikelora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--use_rslora", action="store_true")
    p.add_argument("--v_threshold", type=float, default=0.25)
    p.add_argument("--tau", type=float, default=2.0)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=pick_device())
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--wandb", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)
    tokenizer.pad_token = "<|padding|>"

    model, config = build_model(args.model_name_or_path, args.n_layer, args.n_embd, args.vocab_size, args.ctx_len)

    replaced = apply_lora(
        model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        use_rslora=args.use_rslora, use_spikelora=args.use_spikelora,
        v_threshold=args.v_threshold, tau=args.tau,
    )
    expected = config.n_layer * 6 + 1
    print(f"Patched {len(replaced)} linear layers (expected {expected}) with "
          f"{'SpikeLoRA' if args.use_spikelora else 'LoRA'} (r={args.lora_r}, alpha={args.lora_alpha}):")
    for name in replaced:
        print(f"  {name}")
    assert len(replaced) == expected, f"expected {expected} replaced modules, got {len(replaced)}"

    mark_trainable(model)
    model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params} / {total_params} ({trainable_params / total_params:.2%})")

    data_dir = download_subj(args.data_dir or DEFAULT_DATA_DIR)
    texts, labels = load_subj(data_dir)
    train_texts, val_texts, train_labels, val_labels = train_val_split(texts, labels, test_size=0.1, seed=args.seed)

    if args.max_train_samples:
        train_texts = train_texts[: args.max_train_samples]
        train_labels = train_labels[: args.max_train_samples]
    if args.max_eval_samples:
        val_texts = val_texts[: args.max_eval_samples]
        val_labels = val_labels[: args.max_eval_samples]

    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(
        SubjDataset(train_texts, train_labels, tokenizer, max_length=config.ctx_len),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        SubjDataset(val_texts, val_labels, tokenizer, max_length=config.ctx_len),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="spikegpt-nlu-subj", config=vars(args))

    os.makedirs(args.output_dir, exist_ok=True)
    history = []
    global_step = 0
    stop_early = False

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        with tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs} [train]", leave=False) as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                reset_all_lif(model)

                train_loss_sum += loss.item() * inputs.size(0)
                train_correct += (outputs.argmax(dim=1) == targets).sum().item()
                train_total += inputs.size(0)
                global_step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                if wandb_run is not None:
                    wandb_run.log({"train/batch_loss": loss.item()}, step=global_step)
                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop_early = True
                    break

        avg_train_loss = train_loss_sum / max(train_total, 1)
        avg_train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad(), tqdm(val_loader, desc=f"epoch {epoch + 1}/{args.epochs} [eval]", leave=False) as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                reset_all_lif(model)
                val_loss_sum += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(dim=1) == targets).sum().item()
                val_total += inputs.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_val_loss = val_loss_sum / max(val_total, 1)
        avg_val_acc = val_correct / max(val_total, 1)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss, "train_accuracy": avg_train_acc,
            "val_loss": avg_val_loss, "val_accuracy": avg_val_acc,
        }
        if args.use_spikelora:
            epoch_metrics["sparsity"] = average_sparsity(model)
        history.append(epoch_metrics)
        print(f"Epoch {epoch + 1}/{args.epochs}: train_loss={avg_train_loss:.4f} "
              f"train_acc={avg_train_acc:.4f} val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.4f}"
              + (f" sparsity={epoch_metrics['sparsity']:.4f}" if "sparsity" in epoch_metrics else ""))
        if wandb_run is not None:
            wandb_run.log(epoch_metrics, step=global_step)

        if stop_early:
            break

    final_metrics = {
        "args": vars(args),
        "replaced_modules": replaced,
        "history": history,
        **history[-1],
    }
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Wrote {os.path.join(args.output_dir, 'eval_metrics.json')}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
