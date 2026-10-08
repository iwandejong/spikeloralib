# SpikeLoRA

`spikeloralib` is a fork of [microsoft/LoRA](https://github.com/microsoft/LoRA)'s `loralib`, extended with **SpikeLoRA**: a spiking-neuron gate (`spikeloralib.spikelora.Linear`) that sparsifies LoRA's low-rank activations with a learned, parameter-free LIF neuron. It's a drop-in replacement for `spikeloralib.Linear`, which itself is a drop-in replacement for `nn.Linear`.

## Install

```bash
pip install -e .
```

## Quickstart

```python
from spikeloralib import spikelora
import spikeloralib as sl

layer = spikelora.Linear(
    in_features, out_features, r=8, lora_alpha=16,
    v_threshold=0.1, # LIF voltage threshold
    tau=2.0, # LIF leak constant
)
sl.mark_only_lora_as_trainable(model)
```

See `examples/NLU`, `examples/SpikeGPT-NLU`, `examples/llama2-qlora-scaling`, and
`examples/low-resource` for the fine-tuning code itself, `scripts/` for the shell scripts
that reproduce the paper's results, or `./scripts/run.sh` to run everything end to end.

## Citation

Citation to the paper will be added on release.
