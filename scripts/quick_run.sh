#!/usr/bin/env bash
# Same as run.sh (same args, same --help), but 10 steps per run, into results-quick/.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export EXTRA_ARGS="--max_steps 10 --max_train_samples 32 --max_eval_samples 16"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results-quick}" exec "$REPO_ROOT/scripts/run.sh" "$@"
