#!/usr/bin/env bash
# Reproduces every result in the SpikeLoRA paper. Long-running (hours-to-days); llama2
# needs a CUDA GPU and HF_TOKEN set for gated access. See --help for usage.
set -uo pipefail  # not -e: one stage failing shouldn't stop the rest

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python3}"

DRY_RUN=0
STAGES=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    nlu|spikegpt|llama2) STAGES+=("$arg") ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./scripts/run.sh [--dry-run] [nlu] [spikegpt] [llama2]

No stage argument runs the paper's three tracks (nlu, spikegpt, llama2).
--dry-run prints every command that would run, without installing, downloading, or
training anything. Results go to results/<stage>/; combined output is logged to
logs/run_<timestamp>.log. llama2 needs a CUDA GPU and HF_TOKEN set.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg (expected --dry-run, nlu, spikegpt, or llama2)" >&2
      exit 1
      ;;
  esac
done
if [ ${#STAGES[@]} -eq 0 ]; then
  STAGES=(nlu spikegpt llama2)
fi

RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"

if [ "$DRY_RUN" = "1" ]; then
  exec "$PYTHON" "$REPO_ROOT/scripts/run_all.py" --results-root "$RESULTS_ROOT" --stages "${STAGES[@]}" --dry-run
fi

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$RESULTS_ROOT" "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
echo "[run.sh] logging combined output to $LOG_FILE (also printed below)"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[run.sh] $(date) starting -- stages: ${STAGES[*]}, results -> $RESULTS_ROOT"

echo "[run.sh] [1/3] installing spikeloralib + example dependencies..."
"$PYTHON" -m pip install -q -e "$REPO_ROOT" || { echo "[run.sh] FATAL: pip install -e . failed"; exit 1; }
"$PYTHON" -m pip install -q tqdm "huggingface_hub[cli]>=0.20"
for stage in "${STAGES[@]}"; do
  case "$stage" in
    nlu) req="$REPO_ROOT/examples/NLU/requirements.txt" ;;
    spikegpt) req="$REPO_ROOT/examples/SpikeGPT-NLU/requirements.txt" ;;
    llama2) req="$REPO_ROOT/examples/llama2-qlora-scaling/requirements.txt" ;;
  esac
  "$PYTHON" -m pip install -q -r "$req" || echo "[run.sh] WARNING: failed installing $req -- continuing"
done

echo "[run.sh] [2/3] pre-fetching pretrained checkpoints..."
"$REPO_ROOT/scripts/pull_models.sh"

echo "[run.sh] [3/3] running the paper reproduction..."
"$PYTHON" "$REPO_ROOT/scripts/run_all.py" --results-root "$RESULTS_ROOT" --stages "${STAGES[@]}"
STATUS=$?

echo "[run.sh] done (exit $STATUS). results -> $RESULTS_ROOT, full log -> $LOG_FILE"
exit "$STATUS"
