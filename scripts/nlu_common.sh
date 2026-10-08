#!/usr/bin/env bash
# Shared defaults for scripts/nlu_*.sh. Source this, don't run it.
set -euo pipefail

MODEL="${MODEL:-microsoft/deberta-v3-base}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NLU_DIR="$REPO_ROOT/examples/NLU"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/NLU}"
SEEDS="${SEEDS:-0 1 2 3 4}"
PYTHON="${PYTHON:-python}"

# Plain functions, not `declare -A`: macOS's /bin/bash is 3.2, which has no associative arrays.
task_lr() {
  case "$1" in
    cola|sst2|mrpc|stsb|rte) echo 3e-4 ;;
    mnli|qnli|qqp) echo 1e-4 ;;
    *) echo "task_lr: unknown task '$1'" >&2; exit 1 ;;
  esac
}
task_bsz() {
  case "$1" in
    cola|sst2|mrpc|stsb|mnli|qnli|rte|qqp) echo 32 ;;
    *) echo "task_bsz: unknown task '$1'" >&2; exit 1 ;;
  esac
}
task_epochs() {
  case "$1" in
    cola|mrpc|stsb) echo 20 ;;
    sst2) echo 6 ;;
    mnli|qnli) echo 5 ;;
    rte) echo 30 ;;
    qqp) echo 5 ;;
    *) echo "task_epochs: unknown task '$1'" >&2; exit 1 ;;
  esac
}

run_glue() {
  "$PYTHON" "$NLU_DIR/run_glue.py" --model_name_or_path "$MODEL" "$@" ${EXTRA_ARGS:-}
}
