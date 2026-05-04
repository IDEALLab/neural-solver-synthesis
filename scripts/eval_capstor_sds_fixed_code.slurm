#!/bin/bash -l
#SBATCH --job-name=eval-sds-fixed
#SBATCH --account=a0105
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ritom/scratch/cscs/%u/logs/eval-fixed-%j.out
#SBATCH --error=/ritom/scratch/cscs/%u/logs/eval-fixed-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smassoudi@ethz.ch

set -euo pipefail

LABEL=""
SEED=""
FIXED_CODE_FILE=""
METHOD_NAME=""
CODE_SOURCE_TYPE=""
CODE_SOURCE_SEED=""
MODEL_NAME="fixed-code"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fixed-code-file)
            FIXED_CODE_FILE="$2"
            shift 2
            ;;
        --label)
            LABEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --method-name)
            METHOD_NAME="$2"
            shift 2
            ;;
        --code-source-type)
            CODE_SOURCE_TYPE="$2"
            shift 2
            ;;
        --code-source-seed)
            CODE_SOURCE_SEED="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$FIXED_CODE_FILE" || -z "$LABEL" || -z "$SEED" || -z "$METHOD_NAME" || -z "$CODE_SOURCE_TYPE" ]]; then
    echo "Missing required arguments."
    echo "Usage: sbatch eval_capstor_sds_fixed_code.slurm --fixed-code-file <path> --label <label> --seed <101|202|303> --method-name <name> --code-source-type <type> [--code-source-seed <seed>] [--model-name <name>]"
    exit 1
fi

if [[ ! "$SEED" =~ ^(101|202|303)$ ]]; then
    echo "ERROR: Seed must be 101, 202, or 303. Got: '$SEED'"
    exit 1
fi

export HF_HOME=/ritom/scratch/cscs/${USER}/models
export HF_DATASETS_CACHE=/ritom/scratch/cscs/${USER}/hf_datasets_cache
export HF_TOKEN=$(< $HOME/llm/hf_token.txt)
export REPO_HOST_ROOT="${PWD}"
export TEST_DATASET="SoheylM/OpenR1-SDS-10k-seed${SEED}"
export EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gh200-llm-sds-training-daints}"
export LABEL
export SEED
export FIXED_CODE_FILE
export METHOD_NAME
export CODE_SOURCE_TYPE
export CODE_SOURCE_SEED
export MODEL_NAME

RESULTS_ROOT="evaluation/sds/results"
if [[ -n "${BATCH_ID:-}" ]]; then
    RESULTS_ROOT="evaluation/sds/results_batches/${BATCH_ID}"
fi
export OUTPUT_DIR="${RESULTS_ROOT}/fixed-code/${LABEL}/seed${SEED}"

echo "=== Fixed-Code SDS Evaluation ==="
echo "Seed:              $SEED"
echo "Method:            $METHOD_NAME"
echo "Label:             $LABEL"
echo "Code Source Type:  $CODE_SOURCE_TYPE"
echo "Code Source Seed:  ${CODE_SOURCE_SEED:-(none)}"
echo "Fixed Code File:   $FIXED_CODE_FILE"
echo "Dataset:           $TEST_DATASET"
echo "Output:            $OUTPUT_DIR"
echo "Batch ID:          ${BATCH_ID:-(none)}"
echo "================================="

srun -u --environment="$EDF_ENVIRONMENT" bash -c '
    set -e

    REPO_ROOT="/workspace/llm-finetuning"
    cd "$REPO_ROOT"
    export PYTHONPATH=$PYTHONPATH:$(pwd):/workspace/open-r1

    if [[ "$FIXED_CODE_FILE" = /* ]]; then
        CONTAINER_CODE_PATH="${FIXED_CODE_FILE/$REPO_HOST_ROOT/\/workspace\/llm-finetuning}"
    else
        CONTAINER_CODE_PATH="$REPO_ROOT/$FIXED_CODE_FILE"
    fi

    if [[ ! -f "$CONTAINER_CODE_PATH" ]]; then
        echo "❌ ERROR: Fixed code file not found inside container: $CONTAINER_CODE_PATH"
        exit 1
    fi

    EXTRA_CODE_ARGS=()
    if [[ -n "${CODE_SOURCE_SEED:-}" ]]; then
        EXTRA_CODE_ARGS+=(--code-source-seed "$CODE_SOURCE_SEED")
    fi

    EVALUATION_START=$(python -c "import time; print(time.time())")
    python evaluation/sds/evaluate.py \
        --dataset "$TEST_DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --model "$MODEL_NAME" \
        --training-scheme fixed-code \
        --seed "$SEED" \
        --fixed-code-file "$CONTAINER_CODE_PATH" \
        --method-name-override "$METHOD_NAME" \
        --code-label "$LABEL" \
        --code-source-type "$CODE_SOURCE_TYPE" \
        --code-source-path "$FIXED_CODE_FILE" \
        "${EXTRA_CODE_ARGS[@]}" \
        --baselines greedy local_search cpsat bnb \
        --time_budget 5.0 \
        --repeats 3 \
        --workers 4
    EVALUATION_END=$(python -c "import time; print(time.time())")
    export EVALUATION_WALL_CLOCK_SECONDS=$(python -c "print(${EVALUATION_END} - ${EVALUATION_START})")

    python - <<PY
import json
import os
from pathlib import Path

path = Path(os.environ["OUTPUT_DIR"]) / "timing_summary.json"
data = json.loads(path.read_text()) if path.exists() else {}
data.update(
    {
        "run_type": "fixed-code",
        "dataset": os.environ.get("TEST_DATASET"),
        "model": os.environ.get("MODEL_NAME"),
        "method_name": os.environ.get("METHOD_NAME"),
        "label": os.environ.get("LABEL"),
        "code_source_type": os.environ.get("CODE_SOURCE_TYPE"),
        "code_source_seed": int(os.environ["CODE_SOURCE_SEED"]) if os.environ.get("CODE_SOURCE_SEED") else None,
        "code_source_path": os.environ.get("FIXED_CODE_FILE"),
        "generation_wall_clock_seconds": 0.0,
        "evaluation_wall_clock_seconds": float(os.environ.get("EVALUATION_WALL_CLOCK_SECONDS", "0.0")),
        "batch_id": os.environ.get("BATCH_ID") or None,
    }
)
path.write_text(json.dumps(data, indent=2))
print(f"⏱️  Updated timing summary: {path}")
PY
'
