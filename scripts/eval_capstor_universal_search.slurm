#!/bin/bash -l
#SBATCH --job-name=universal-sds
#SBATCH --account=a0105
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=/ritom/scratch/cscs/%u/logs/universal-sds-%j.out
#SBATCH --error=/ritom/scratch/cscs/%u/logs/universal-sds-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smassoudi@ethz.ch  # Update with your email

# =============================================================================
# Universal Solver Search for SDS (CPU-only)
# =============================================================================
#
# This runs an adaptive "universal code" search over an existing pool of generated
# solutions (e.g., Base Best-of-64 generations.jsonl) WITHOUT any new LLM inference.
#
# Usage (recommended, batch-aware):
#   sbatch scripts/eval_capstor_universal_search.slurm \
#     --seed 101 \
#     --generations-jsonl evaluation/sds/results_batches/<BATCH>/qwen2.5-coder-14b/base/seed101/job-*/generations.jsonl \
#     --metrics-csv evaluation/sds/results_batches/<BATCH>/qwen2.5-coder-14b/base/seed101/job-*/metrics_final.csv \
#     --batch-id <BATCH_ID>
#
# If --batch-id is omitted, outputs go to evaluation/sds/universal_search/ (legacy/moving).
#
# Notes:
# - Requires HF token access inside the container (same as other scripts).
# - Uses python (not python3) to match the conda/env expectations.
# =============================================================================

set -e

SEED=""
BATCH_ID=""
GENERATIONS_GLOB=""
METRICS_CSV=""
WORKERS="64"
TIMEOUT="5.0"
TOURNAMENT_MISSIONS="30"
SURVIVORS="10"
KEEP_PER_ROUND="500"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --batch-id) BATCH_ID="$2"; shift 2 ;;
    --generations-jsonl) GENERATIONS_GLOB="$2"; shift 2 ;;
    --metrics-csv) METRICS_CSV="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --tournament-missions) TOURNAMENT_MISSIONS="$2"; shift 2 ;;
    --survivors) SURVIVORS="$2"; shift 2 ;;
    --keep-per-round) KEEP_PER_ROUND="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$SEED" ]]; then
  echo "ERROR: --seed is required (101|202|303)"
  exit 1
fi
if [[ ! "$SEED" =~ ^(101|202|303)$ ]]; then
  echo "ERROR: Seed must be 101, 202, or 303. Got: $SEED"
  exit 1
fi
if [[ -z "$GENERATIONS_GLOB" ]]; then
  echo "ERROR: --generations-jsonl <glob-or-path> is required"
  exit 1
fi
if [[ -z "$METRICS_CSV" ]]; then
  echo "ERROR: --metrics-csv <path> is required (use metrics_final.csv from the same base run)"
  exit 1
fi

# Batch-aware output root (legacy compatible)
OUT_ROOT="evaluation/sds/universal_search"
if [[ -n "${BATCH_ID:-}" ]]; then
  OUT_ROOT="evaluation/sds/universal_search_batches/${BATCH_ID}"
fi
OUT_DIR="${OUT_ROOT}/seed${SEED}"

echo "=== Universal SDS Search ==="
echo "Seed:               $SEED"
echo "Batch ID:           ${BATCH_ID:-(none)}"
echo "Generations glob:   $GENERATIONS_GLOB"
echo "Metrics CSV:        $METRICS_CSV"
echo "Timeout (s):        $TIMEOUT"
echo "Workers:            $WORKERS"
echo "Tournament missions:$TOURNAMENT_MISSIONS"
echo "Survivors (final):  $SURVIVORS"
echo "Keep per round:     $KEEP_PER_ROUND"
echo "Output dir:         $OUT_DIR"
echo "==========================="
echo ""

export SEED
export BATCH_ID
export OUTPUT_DIR="$OUT_DIR"

export HF_HOME=/ritom/scratch/cscs/${USER}/models
export HF_DATASETS_CACHE=/ritom/scratch/cscs/${USER}/hf_datasets_cache
export EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gh200-llm-sds-training-daints}"
export HF_TOKEN=$(< $HOME/llm/hf_token.txt)

# Execute inside the standard container environment used by other SDS jobs.
srun -u --environment=${EDF_ENVIRONMENT} bash -c "
  set -e
  REPO_ROOT=/workspace/llm-finetuning
  cd \$REPO_ROOT
  mkdir -p \"$OUT_DIR\"

  # Paths are already relative (converted in run_universal_search_base.sh)
  # They work directly inside container because repo is mounted at /workspace/llm-finetuning
  METRICS_CSV=\"$METRICS_CSV\"
  GENERATIONS_GLOB=\"$GENERATIONS_GLOB\"

  echo \"🔍 Expanding generations glob: \$GENERATIONS_GLOB\"
  
  # Expand glob inside container context (bash array expansion)
  # Use shopt -s nullglob to make empty globs result in empty array (not the literal glob string)
  shopt -s nullglob
  GEN_FILES=( \$GENERATIONS_GLOB )
  shopt -u nullglob
  
  echo \"   Found \${#GEN_FILES[@]} file(s)\"
  if [ \${#GEN_FILES[@]} -eq 0 ]; then
    echo \"❌ ERROR: no generations.jsonl files matched: \$GENERATIONS_GLOB\"
    echo \"   Attempting to list parent directory...\"
    PARENT_DIR=\$(dirname \"\$GENERATIONS_GLOB\" 2>/dev/null || echo \".\")
    if [ -d \"\$PARENT_DIR\" ]; then
      echo \"   Directory contents:\"
      ls -la \"\$PARENT_DIR\" 2>/dev/null || echo \"   (ls failed)\"
    else
      echo \"   Parent directory does not exist: \$PARENT_DIR\"
    fi
    echo \"   Full glob pattern was: \$GENERATIONS_GLOB\"
    exit 1
  fi
  
  # Validate each file exists
  for f in \"\${GEN_FILES[@]}\"; do
    if [ ! -f \"\$f\" ]; then
      echo \"❌ ERROR: file does not exist: \$f\"
      exit 1
    fi
    echo \"   ✓ \$(basename \$f)\"
  done

  # Build Python command with all generation files
  PYTHON_ARGS=(
    --seed $SEED
    --metrics-csv \"\$METRICS_CSV\"
    --output-dir \"$OUT_DIR\"
    --timeout $TIMEOUT
    --workers $WORKERS
    --tournament-missions $TOURNAMENT_MISSIONS
    --survivors $SURVIVORS
    --keep-per-round $KEEP_PER_ROUND
  )
  
  # Add each generation file as a separate --generations-jsonl argument
  for f in \"\${GEN_FILES[@]}\"; do
    PYTHON_ARGS+=(--generations-jsonl \"\$f\")
  done

  python evaluation/sds/universal_solver_search.py \"\${PYTHON_ARGS[@]}\"
"

echo ""
echo "✅ Universal search completed."
echo "   Results: $OUT_DIR"
