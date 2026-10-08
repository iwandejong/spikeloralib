from typing import Iterable, List, Optional

import torch
import torch.nn as nn

import spikeloralib as lora
from spikeloralib import spikelora

DEBERTA_TARGET_MODULES = [
    "query_proj",
    "key_proj",
    "value_proj",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
]


def _get_parent(model: nn.Module, dotted_name: str) -> nn.Module:
    if "." not in dotted_name:
        return model
    parent_name = dotted_name.rsplit(".", 1)[0]
    return model.get_submodule(parent_name)


def _replace_linear(model: nn.Module, name: str, old: nn.Linear, layer_cls, **kwargs) -> None:
    new = layer_cls(old.in_features, old.out_features, bias=old.bias is not None, **kwargs)
    new = new.to(dtype=old.weight.dtype, device=old.weight.device)
    with torch.no_grad():
        new.weight.copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    parent = _get_parent(model, name)
    child_name = name.rsplit(".", 1)[-1]
    setattr(parent, child_name, new)


def apply_lora(
    model: nn.Module,
    r: int,
    lora_alpha: int,
    lora_dropout: float = 0.0,
    use_rslora: bool = False,
    use_spikelora: bool = False,
    v_threshold: float = 1.0,
    tau: float = 2.0,
    target_modules: Optional[Iterable[str]] = None,
) -> List[str]:
    target_modules = list(target_modules) if target_modules is not None else DEBERTA_TARGET_MODULES
    layer_cls = spikelora.Linear if use_spikelora else lora.Linear

    extra_kwargs = {}
    if use_spikelora:
        extra_kwargs.update(v_threshold=v_threshold, tau=tau)

    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            _replace_linear(
                model, name, module, layer_cls,
                r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, use_rslora=use_rslora,
                **extra_kwargs,
            )
            replaced.append(name)
    return replaced


def mark_trainable(model: nn.Module, train_classifier_substrings: Iterable[str] = ("classifier", "pooler")) -> None:
    lora.mark_only_lora_as_trainable(model)
    for name, param in model.named_parameters():
        if any(s in name for s in train_classifier_substrings):
            param.requires_grad = True


def spikelora_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, spikelora.Linear)]


def average_sparsity(model: nn.Module) -> Optional[float]:
    values = [m.sparsity for m in spikelora_modules(model) if m.sparsity is not None]
    if not values:
        return None
    return sum(values) / len(values)


def module_sparsity(model: nn.Module) -> dict:
    return {
        name: module.sparsity
        for name, module in model.named_modules()
        if isinstance(module, spikelora.Linear) and module.sparsity is not None
    }
