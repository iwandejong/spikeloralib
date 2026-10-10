#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

echo "[install_nlu_deps] Installing spikeloralib..."
"$PYTHON" -m pip install -e "$REPO_ROOT"

echo "[install_nlu_deps] Installing NLU requirements..."
"$PYTHON" -m pip install -r "$REPO_ROOT/examples/NLU/requirements.txt"

echo "[install_nlu_deps] Dependencies installed."
