import math
from typing import Optional, Union

import torch
import torch.nn as nn

class _ATanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_threshold: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(v_minus_threshold)
        ctx.alpha = alpha
        return (v_minus_threshold >= 0).to(v_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v_minus_threshold,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = alpha / (2 * (1 + (math.pi / 2 * alpha * v_minus_threshold) ** 2))
        return grad * grad_output, None


def atan_spike(v_minus_threshold: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return _ATanSpike.apply(v_minus_threshold, alpha)


class LIFNeuron(nn.Module):
    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_alpha: float = 2.0,
        detach_reset: bool = True,
    ):
        super().__init__()
        if tau <= 1.0:
            raise ValueError(f"tau must be > 1.0 (got {tau})")
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_alpha = surrogate_alpha
        self.detach_reset = detach_reset
        self.v: Union[float, torch.Tensor] = 0.0 if v_reset is None else v_reset

    def reset(self) -> None:
        self.v = 0.0 if self.v_reset is None else self.v_reset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v = self.v + (x - self.v) / self.tau
        spike = atan_spike(self.v - self.v_threshold, self.surrogate_alpha)
        # Detaching avoids a second gradient path through the already-surrogate spike.
        spike_for_reset = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - spike_for_reset * self.v_threshold
        else:
            self.v = (1.0 - spike_for_reset) * self.v + spike_for_reset * self.v_reset

        return spike

    def extra_repr(self) -> str:
        return f"tau={self.tau}, v_threshold={self.v_threshold}, v_reset={self.v_reset}"
