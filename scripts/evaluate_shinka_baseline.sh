#!/bin/bash
# =============================================================================
# Evaluate ShinkaEvolve Baseline on SDS for Multiple Seeds
# =============================================================================
# Usage:
#   # Evaluate all seeds
#   ./scripts/evaluate_shinka_baseline.sh --batch-id 20251230_baselines-v1
#
#   # Evaluate single seed
#   ./scripts/evaluate_shinka_baseline.sh --seed 101 --batch-id 20251230_baselines-v1
#
#   # Without W&B logging
#   ./scripts/evaluate_shinka_baseline.sh --batch-id 20251230_baselines-v1 --no-wandb
#
# This script:
#   1. Evaluates ShinkaEvolve evolved codes against the SDS test set
#   2. Uses ShinkaEvolve-SDS-1000 augmented datasets (1000 codes per seed)
#   3. Runs locally (no GPU needed, no SLURM required)
#   4. Requires conda environment: llm-finetuning
#
# Output per seed:
#   - metrics_final.csv (per-problem metrics)
#   - experiment_metadata.json
#
# Dependencies:
#   - conda activate llm-finetuning
#   - syndeopt (pip install -e deps/syndeopt)
#   - HuggingFace access to SoheylM/ShinkaEvolve-SDS-1000-seed{101,202,303}
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
SHINKA_DATASET_PREFIX="SoheylM/ShinkaEvolve-SDS-1000"
BASELINES="greedy local_search cpsat bnb"
TIME_BUDGET=5.0
REPEATS=3
WORKERS=4
ALL_SEEDS=(101 202 303)

# Parse arguments
SELECTED_SEED=""
BATCH_ID=""
LOG_TO_WANDB="true"
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SELECTED_SEED="$2"
            shift 2
            ;;
        --batch-id)
            BATCH_ID="$2"
            shift 2
            ;;
        --no-wandb)
            LOG_TO_WANDB="false"
            shift 1
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--seed 101|202|303] [--batch-id YYYYMMDD_name] [--no-wandb]"
            exit 1
            ;;
    esac
done

# Determine which seeds to run
if [ -n "$SELECTED_SEED" ]; then
    if [[ ! " ${ALL_SEEDS[@]} " =~ " ${SELECTED_SEED} " ]]; then
        echo "ERROR: Invalid seed '$SELECTED_SEED'. Must be one of: ${ALL_SEEDS[*]}"
        exit 1
    fi
    SEEDS=("$SELECTED_SEED")
else
    SEEDS=("${ALL_SEEDS[@]}")
fi

# Check we are in the correct conda environment
if ! python -c "import syndeopt" 2>/dev/null; then
    echo "ERROR: syndeopt not found. Please activate the conda environment:"
    echo "  conda activate llm-finetuning"
    exit 1
fi

echo "=============================================================================="
echo "Evaluating ShinkaEvolve Baseline on SDS"
echo "=============================================================================="
echo "Dataset:      ${SHINKA_DATASET_PREFIX}-seed{SEED}"
echo "Seeds:        ${SEEDS[*]}"
echo "Baselines:    $BASELINES"
echo "Time Budget:  ${TIME_BUDGET}s"
if [ -n "$BATCH_ID" ]; then
    echo "Batch ID:     $BATCH_ID"
fi
echo "W&B Logging:  $LOG_TO_WANDB"
echo "Total Runs:   ${#SEEDS[@]}"
echo "=============================================================================="
echo ""

cd "$REPO_ROOT"

for seed in "${SEEDS[@]}"; do
    echo "Processing: ShinkaEvolve (Seed $seed)"
    echo "  Dataset: ${SHINKA_DATASET_PREFIX}-seed${seed}"

    EVAL_ARGS=(
        --shinka-dataset "${SHINKA_DATASET_PREFIX}-seed${seed}"
        --baselines $BASELINES
        --time_budget "$TIME_BUDGET"
        --repeats "$REPEATS"
        --workers "$WORKERS"
    )

    if [ -n "$BATCH_ID" ]; then
        export BATCH_ID
    fi

    if [ "$LOG_TO_WANDB" = "true" ]; then
        EVAL_ARGS+=(--log-to-wandb)
    fi

    python evaluation/sds/evaluate.py "${EVAL_ARGS[@]}"

    echo "  Evaluation complete (Seed $seed)"
    echo ""
done

echo "=============================================================================="
echo "All ${#SEEDS[@]} ShinkaEvolve evaluations complete!"
echo "=============================================================================="
echo ""
echo "Results saved in:"
echo "  evaluation/sds/results_batches/${BATCH_ID:-results}/shinka-evolve/"
echo ""
