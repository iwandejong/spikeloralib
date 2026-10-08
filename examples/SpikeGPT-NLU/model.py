# adapted from SpikeGPT, loads with strict=True
import math

import torch
import torch.nn as nn

from spikeloralib.spiking import LIFNeuron, atan_spike

class GPTConfig:
    def __init__(self, vocab_size: int, ctx_len: int, n_layer: int, n_embd: int, model_type: str = "RWKV"):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.model_type = model_type

def _wkv_linear_attention(time_decay: torch.Tensor, time_first: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, T, C = k.shape
    w = -torch.exp(time_decay.float())
    u = time_first.float()
    out = torch.empty((B, T, C), dtype=torch.float32, device=k.device)
    num_state = torch.zeros(B, C, dtype=torch.float32, device=k.device)
    den_state = torch.zeros(B, C, dtype=torch.float32, device=k.device)
    max_state = torch.full((B, C), -1e38, dtype=torch.float32, device=k.device)
    for t in range(T):
        kt = k[:, t].float()
        vt = v[:, t].float()

        max_for_output = torch.maximum(max_state, kt + u)
        e1 = torch.exp(max_state - max_for_output)
        e2 = torch.exp(kt + u - max_for_output)
        out[:, t] = (e1 * num_state + e2 * vt) / (e1 * den_state + e2)

        max_for_state = torch.maximum(max_state + w, kt)
        e1 = torch.exp(max_state + w - max_for_state)
        e2 = torch.exp(kt - max_for_state)
        num_state = e1 * num_state + e2 * vt
        den_state = e1 * den_state + e2
        max_state = max_for_state
    return out.to(k.dtype)

class _SpikeInputActivation(nn.Module):
    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return atan_spike(x, self.alpha)

def _step_over_time(lif: LIFNeuron, x: torch.Tensor) -> torch.Tensor:
    return torch.stack([lif(x[:, t]) for t in range(x.size(1))], dim=1)

def RWKV_Init(model: nn.Module, config: GPTConfig) -> None:
    for m in model.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        ww = m.weight
        shape = ww.shape
        gain = 1.0
        scale = 1.0
        if isinstance(m, nn.Embedding):
            gain = math.sqrt(max(shape[0], shape[1]))
            scale = 1e-4 if shape[0] == config.vocab_size and shape[1] == config.n_embd else 0.0
        if isinstance(m, nn.Linear):
            if shape[0] > shape[1]:
                gain = math.sqrt(shape[0] / shape[1])
            if shape[0] == config.vocab_size and shape[1] == config.n_embd:
                scale = 0.5
        if hasattr(m, "scale_init"):
            scale = m.scale_init
        gain *= scale
        with torch.no_grad():
            if scale == -999:
                nn.init.eye_(ww)
            elif gain == 0:
                nn.init.zeros_(ww)
            elif gain > 0:
                nn.init.orthogonal_(ww, gain=gain)
            else:
                nn.init.normal_(ww, mean=0.0, std=-scale)

class RWKV_TimeMix(nn.Module):
    def __init__(self, config: GPTConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd
        attn_sz = config.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)

            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)

            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, config.n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        rwkv = sr * _wkv_linear_attention(self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)
        return rwkv

class RWKV_ChannelMix(nn.Module):
    def __init__(self, config: GPTConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * config.n_embd
        self.key = nn.Linear(config.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, config.n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv

class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)

        self.lif1 = LIFNeuron(tau=2.0, v_threshold=1.0, surrogate_alpha=2.0)
        self.lif2 = LIFNeuron(tau=2.0, v_threshold=1.0, surrogate_alpha=2.0)
        self.dropout = nn.Dropout(0.03)  # hardcoded in the original, not config-driven

        self.att = RWKV_TimeMix(config, layer_id)
        self.ffn = RWKV_ChannelMix(config, layer_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + _step_over_time(self.lif1, self.att(self.ln1(x)))
        x = x + _step_over_time(self.lif2, self.ffn(self.ln2(x)))
        x = self.dropout(x)
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])
        self.spike_in = _SpikeInputActivation(alpha=2.0)
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.ctx_len = config.ctx_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(self.emb.weight.device)
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.spike_in(self.emb(idx))
        x = self.blocks(x)
        x = self.ln_out(x)
        x = self.head(x)
        return x[:, -1, :]
