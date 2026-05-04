#!/bin/bash
# =============================================================================
# Evaluate Base Model (Best-of-64) on SDS for Multiple Seeds
# =============================================================================
# Usage:
#   # Evaluate all seeds (3 SLURM jobs)
#   ./scripts/evaluate_base_model_sds.sh --batch-id 20251230_baselines-v1
#
#   # Evaluate single seed
#   ./scripts/evaluate_base_model_sds.sh --seed 101 --batch-id 20251230_baselines-v1
#
# This script:
#   1. Submits SDS evaluation jobs for the untrained base model
#   2. Uses 64 samples per problem (temperature=0.6) for Pass@k scaling analysis
#   3. Produces ~1 GB of generations per seed (~4.4 GB total for 3 seeds)
#   4. Bootstrap resampling (500 iterations) for k in {1,2,4,8,16,32,64}
#
# Output per seed:
#   - generations.jsonl (~1 GB, 64k lines)
#   - metrics_final.csv (per-problem metrics for Best-of-64)
#   - scaling_stats.csv (Pass@k statistics for Figure 5)
#
# Dependencies:
#   - Requires eval_capstor_sds_pipeline.slurm
#   - Requires HuggingFace access to Qwen/Qwen2.5-Coder-14B-Instruct
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
BASE_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
N_SAMPLES=64
TEMPERATURE=0.6
BOOTSTRAP_N=500
ALL_SEEDS=(101 202 303)

# Parse arguments
SELECTED_SEED=""
BATCH_ID=""
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
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--seed 101|202|303] [--batch-id YYYYMMDD_name]"
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

echo "=============================================================================="
echo "Evaluating Base Model (Best-of-64) on SDS"
echo "=============================================================================="
echo "Model:        $BASE_MODEL"
echo "Seeds:        ${SEEDS[*]}"
echo "Samples:      $N_SAMPLES per problem (temperature=$TEMPERATURE)"
echo "Bootstrap:    $BOOTSTRAP_N iterations"
if [ -n "$BATCH_ID" ]; then
    echo "Batch ID:     $BATCH_ID"
else
    echo "Batch ID:     (none) -> writing to evaluation/sds/results/ (legacy path)"
fi
echo "Total Jobs:   ${#SEEDS[@]}"
echo "=============================================================================="
echo ""

TOTAL_JOBS=0
for seed in "${SEEDS[@]}"; do
    echo "Processing: Base Model (Seed $seed)"

    if [ -n "$BATCH_ID" ]; then
        export BATCH_ID
    fi
    sbatch \
        "$SCRIPT_DIR/eval_capstor_sds_pipeline.slurm" \
        --base-model "$BASE_MODEL" \
        --n-samples "$N_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --bootstrap-n "$BOOTSTRAP_N" \
        "$seed"

    echo "  Job submitted (Seed $seed)"
    echo ""
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
done

echo "=============================================================================="
echo "All $TOTAL_JOBS Base model evaluation jobs submitted!"
echo "=============================================================================="
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check job outputs in:"
echo "  /ritom/scratch/cscs/\$USER/logs/eval-*.out"
echo ""
echo "Expected output per seed (~1 GB):"
echo "  evaluation/sds/results_batches/${BATCH_ID:-results}/qwen2.5-coder-14b/base/seed{SEED}/"
echo ""
