#!/usr/bin/env bash
# Pre-fetches the checkpoints the examples use. Set HF_TOKEN for gated repos.
hf download microsoft/deberta-v3-base
hf download microsoft/mdeberta-v3-base
hf download ridger/SpikeGPT-OpenWebText-216M SpikeGPT-216M.pth
hf download meta-llama/Llama-2-7b-hf
