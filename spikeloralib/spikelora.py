import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LoRALayer
from .spiking import LIFNeuron

class SpikeLoRALayer(LoRALayer):
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        use_rslora: bool = False,
        v_threshold: float = 1.0,
        tau: float = 2.0,
        v_reset: Optional[float] = 0.0,
        surrogate_alpha: float = 2.0,
        reset_each_forward: bool = True,
    ):
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          merge_weights=False, use_rslora=use_rslora)
        self.v_threshold = v_threshold
        self.tau = tau
        self.v_reset = v_reset
        self.surrogate_alpha = surrogate_alpha
        self.reset_each_forward = reset_each_forward
        self.sparsity: Optional[float] = None
        if r > 0:
            self.lif = LIFNeuron(
                tau=tau, v_threshold=v_threshold, v_reset=v_reset,
                surrogate_alpha=surrogate_alpha,
            )

    def reset_lif(self) -> None:
        if hasattr(self, "lif"):
            self.lif.reset()

    def _spike_gate(self, a_out: torch.Tensor) -> torch.Tensor:
        if self.reset_each_forward:
            self.lif.reset()
        spikes = self.lif(a_out)
        self.sparsity = (spikes == 0).float().mean().item()
        return a_out * spikes

class Linear(nn.Linear, SpikeLoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        use_rslora: bool = False,
        v_threshold: float = 1.0,
        tau: float = 2.0,
        v_reset: Optional[float] = 0.0,
        surrogate_alpha: float = 2.0,
        reset_each_forward: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SpikeLoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=use_rslora,
            v_threshold=v_threshold,
            tau=tau,
            v_reset=v_reset,
            surrogate_alpha=surrogate_alpha,
            reset_each_forward=reset_each_forward,
        )

        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self._lora_scaling()
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            a_out = self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
            gated = self._spike_gate(a_out)
            result = result + (gated @ self.lora_B.transpose(0, 1)) * self.scaling
        return result
