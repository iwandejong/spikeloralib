from typing import Optional

import torch
import torch.nn as nn

import peft.tuners.lora.layer as _peft_lora_layer
from spikeloralib.spiking import LIFNeuron

try:
    import peft.tuners.lora.bnb as _peft_lora_bnb
except ImportError:
    _peft_lora_bnb = None

_PATCHED_FLAG = "_spikelora_shim_patched"
_LIF_ATTR = "_spikelora_lif"
_SPARSITY_ATTR = "_spikelora_sparsity"

def _module_spikelora_adapters(module: nn.Module):
    lif_dict = getattr(module, _LIF_ATTR, None)
    if lif_dict is None:
        return ()
    return tuple(lif_dict.keys())

def _spikelora_lora_term(module: nn.Module, active_adapter: str, x: torch.Tensor) -> torch.Tensor:
    lora_A = module.lora_A[active_adapter]
    lora_B = module.lora_B[active_adapter]
    dropout = module.lora_dropout[active_adapter]
    scaling = module.scaling[active_adapter]
    lif: LIFNeuron = getattr(module, _LIF_ATTR)[active_adapter]

    a_out = lora_A(dropout(x))

    lif.reset()
    spikes = lif(a_out)
    gated = a_out * spikes

    sparsity_dict = getattr(module, _SPARSITY_ATTR, None)
    if sparsity_dict is None:
        sparsity_dict = {}
        setattr(module, _SPARSITY_ATTR, sparsity_dict)
    sparsity_dict[active_adapter] = (spikes == 0).float().mean().item()

    return lora_B(gated) * scaling


def _make_patched_linear_forward(orig_forward):
    def patched_forward(self, x: torch.Tensor, *args, **kwargs):
        spikelora_adapters = _module_spikelora_adapters(self)
        active = getattr(self, "active_adapters", [])
        if not spikelora_adapters or not any(a in spikelora_adapters for a in active):
            return orig_forward(self, x, *args, **kwargs)

        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        from peft.tuners.lora.layer import VARIANT_KWARG_KEYS
        variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if adapter_names is not None:
            return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **variant_kwargs, **kwargs)
        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        lora_A_keys = self.lora_A.keys()
        for active_adapter in self.active_adapters:
            if active_adapter not in lora_A_keys:
                continue
            lora_A = self.lora_A[active_adapter]
            x_cast = self._cast_input_dtype(x, lora_A.weight.dtype)
            if active_adapter not in self.lora_variant and active_adapter in spikelora_adapters:
                result = result + _spikelora_lora_term(self, active_adapter, x_cast)
            elif active_adapter not in self.lora_variant:  # vanilla LoRA
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                result = result + lora_B(lora_A(dropout(x_cast))) * scaling
            else:
                result = self.lora_variant[active_adapter].forward(
                    self, active_adapter=active_adapter, x=x_cast, result=result,
                    **variant_kwargs, **kwargs,
                )

        return result.to(torch_result_dtype)

    return patched_forward


def _make_patched_linear4bit_forward(orig_forward):
    def patched_forward(self, x: torch.Tensor, *args, **kwargs):
        spikelora_adapters = _module_spikelora_adapters(self)
        active = getattr(self, "active_adapters", [])
        if not spikelora_adapters or not any(a in spikelora_adapters for a in active):
            return orig_forward(self, x, *args, **kwargs)

        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        from peft.tuners.lora.layer import VARIANT_KWARG_KEYS
        variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if adapter_names is not None:
            return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **variant_kwargs, **kwargs)
        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)
        result = result.clone()  # some torch versions error on in-place backprop otherwise

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            x_cast = x
            if requires_conversion:
                expected_dtype = result.dtype
                x_cast = self._cast_input_dtype(x_cast, lora_A.weight.dtype)

            if active_adapter not in self.lora_variant and active_adapter in spikelora_adapters:
                output = _spikelora_lora_term(self, active_adapter, x_cast)
                if requires_conversion:
                    output = output.to(expected_dtype)
                result = result + output
            elif active_adapter not in self.lora_variant:  # vanilla LoRA
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                output = lora_B(lora_A(dropout(x_cast))) * scaling
                if requires_conversion:
                    output = output.to(expected_dtype)
                result = result + output
            else:
                result = self.lora_variant[active_adapter].forward(
                    self, active_adapter=active_adapter, x=x_cast, result=result,
                    **variant_kwargs, **kwargs,
                )
                if requires_conversion:
                    result = result.to(expected_dtype)

        return result

    return patched_forward


def patch_peft_for_spikelora() -> None:
    linear_cls = _peft_lora_layer.Linear
    if not getattr(linear_cls.forward, _PATCHED_FLAG, False):
        orig_forward = linear_cls.forward
        patched = _make_patched_linear_forward(orig_forward)
        patched._spikelora_orig_forward = orig_forward
        setattr(patched, _PATCHED_FLAG, True)
        linear_cls.forward = patched

    if _peft_lora_bnb is not None and hasattr(_peft_lora_bnb, "Linear4bit"):
        linear4bit_cls = _peft_lora_bnb.Linear4bit
        if not getattr(linear4bit_cls.forward, _PATCHED_FLAG, False):
            orig_forward_4bit = linear4bit_cls.forward
            patched_4bit = _make_patched_linear4bit_forward(orig_forward_4bit)
            patched_4bit._spikelora_orig_forward = orig_forward_4bit
            setattr(patched_4bit, _PATCHED_FLAG, True)
            linear4bit_cls.forward = patched_4bit


def enable_spikelora(
    peft_model: nn.Module,
    v_threshold: float = 0.1,
    tau: float = 2.0,
    surrogate_alpha: float = 2.0,
) -> int:
    patched_classes = (_peft_lora_layer.Linear,)
    if _peft_lora_bnb is not None and hasattr(_peft_lora_bnb, "Linear4bit"):
        patched_classes = patched_classes + (_peft_lora_bnb.Linear4bit,)

    enabled = 0
    for _, module in peft_model.named_modules():
        if not isinstance(module, patched_classes):
            continue
        lora_A = getattr(module, "lora_A", None)
        if not lora_A:
            continue

        lif_dict = getattr(module, _LIF_ATTR, None)
        if lif_dict is None:
            lif_dict = nn.ModuleDict()
            setattr(module, _LIF_ATTR, lif_dict)  # registers as a submodule, moves with .to(device)

        for adapter_name in lora_A.keys():
            if adapter_name in lif_dict:
                continue
            lif_dict[adapter_name] = LIFNeuron(
                tau=tau, v_threshold=v_threshold, surrogate_alpha=surrogate_alpha,
            )
            enabled += 1

    return enabled


def average_sparsity(peft_model: nn.Module) -> Optional[float]:
    values = []
    for _, module in peft_model.named_modules():
        sparsity_dict = getattr(module, _SPARSITY_ATTR, None)
        if sparsity_dict:
            values.extend(sparsity_dict.values())
    if not values:
        return None
    return sum(values) / len(values)
