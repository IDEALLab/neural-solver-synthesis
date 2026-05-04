#!/bin/bash
# Submit frozen-Hero and hand-written-SA evaluations plus timing reruns.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="all"
BATCH_ID=""
SEEDS=(101 202 303)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-id)
            BATCH_ID="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --batch-id YYYYMMDD_baseline-eval-v1 [--mode fixed|timing|all]"
            exit 1
            ;;
    esac
done

if [[ -z "$BATCH_ID" ]]; then
    echo "ERROR: --batch-id is required"
    exit 1
fi

if [[ "$MODE" != "fixed" && "$MODE" != "timing" && "$MODE" != "all" ]]; then
    echo "ERROR: --mode must be one of: fixed, timing, all"
    exit 1
fi

cd "$REPO_ROOT"
export EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gh200-llm-sds-training-baseline-evaluation-daints}"

pick_python() {
    if command -v python >/dev/null 2>&1; then
        echo "python"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
        return 0
    fi
    return 1
}

echo "=============================================================================="
echo "Baseline Evaluation Launcher"
echo "=============================================================================="
echo "Batch ID:  $BATCH_ID"
echo "Mode:      $MODE"
echo "EDF Env:   $EDF_ENVIRONMENT"
echo "=============================================================================="

echo ""
echo "Extracting canonical frozen Hero solvers..."
PYTHON_BIN="$(pick_python || true)"
for seed in "${SEEDS[@]}"; do
    solver_path="evaluation/sds/frozen_solvers/hero_seed${seed}.py"
    solver_meta="evaluation/sds/frozen_solvers/hero_seed${seed}.json"

    if [[ -n "${PYTHON_BIN:-}" ]] && "$PYTHON_BIN" -c "import pandas" >/dev/null 2>&1; then
        "$PYTHON_BIN" evaluation/sds/extract_frozen_solver.py --seed "$seed"
        continue
    fi

    if [[ -f "$solver_path" && -f "$solver_meta" ]]; then
        echo "Using existing frozen solver for seed ${seed}: $solver_path"
        continue
    fi

    echo "ERROR: Cannot extract frozen solver for seed ${seed} because pandas is unavailable on the submit host and no pre-extracted solver was found."
    exit 1
done

if [[ "$MODE" == "fixed" || "$MODE" == "all" ]]; then
    echo ""
    echo "Submitting fixed-code jobs..."
    export BATCH_ID
    for seed in "${SEEDS[@]}"; do
        sbatch \
            "$SCRIPT_DIR/eval_capstor_sds_fixed_code.slurm" \
            --fixed-code-file "evaluation/sds/frozen_solvers/hero_seed${seed}.py" \
            --label "frozen-hero" \
            --seed "$seed" \
            --method-name "Frozen Hero" \
            --code-source-type "frozen-hero" \
            --code-source-seed "$seed" \
            --model-name "qwen2.5-coder-14b"

        sbatch \
            "$SCRIPT_DIR/eval_capstor_sds_fixed_code.slurm" \
            --fixed-code-file "evaluation/sds/manual_solvers/constraint_aware_sa.py" \
            --label "hand-written-sa" \
            --seed "$seed" \
            --method-name "Hand-written SA" \
            --code-source-type "manual-sa" \
            --model-name "manual-sa"
    done
fi

if [[ "$MODE" == "timing" || "$MODE" == "all" ]]; then
    echo ""
    echo "Submitting timing reruns..."
    export BATCH_ID

    HERO_CHECKPOINT="/ritom/scratch/cscs/${USER}/checkpoints/Qwen2.5-Coder-14B-Instruct-GRPO-SDS-seed101-config_hero/job-1315163/checkpoint-60"

    sbatch \
        "$SCRIPT_DIR/eval_capstor_sds_pipeline.slurm" \
        --checkpoint-dir "$HERO_CHECKPOINT" \
        grpo \
        101

    sbatch \
        "$SCRIPT_DIR/eval_capstor_sds_pipeline.slurm" \
        --base-model "Qwen/Qwen2.5-Coder-14B-Instruct" \
        --n-samples 64 \
        --temperature 0.6 \
        --bootstrap-n 500 \
        101
fi

echo ""
echo "Run Shinka timing locally after cluster submissions:"
echo "  BATCH_ID=$BATCH_ID ./scripts/evaluate_shinka_baseline.sh --seed 101 --batch-id $BATCH_ID"
