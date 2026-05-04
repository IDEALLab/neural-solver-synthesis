#!/bin/bash
# Submit universal SDS search for Base (Best-of-64) generations across seeds 101/202/303.
#
# Usage:
#   ./scripts/run_universal_search_base.sh --base-root evaluation/sds/results_batches/20251230_baselines-v1/qwen2.5-coder-14b/base --batch-id 20251230_baselines-v1
#
# Notes:
# - Expects per-seed folders under --base-root: seed101/, seed202/, seed303/
# - Expects generations.jsonl + metrics_final.csv to exist (either directly under seed folder, or inside job-* subfolder)

set -e

BASE_ROOT=""
BATCH_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-root) BASE_ROOT="$2"; shift 2 ;;
    --batch-id) BATCH_ID="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$BASE_ROOT" ]]; then
  echo "ERROR: --base-root is required"
  exit 1
fi

# Convert absolute paths to relative paths (strip Alps scratch repo prefix).
# This matches the pattern used in eval_capstor_sds_pipeline.slurm
if [[ "$BASE_ROOT" == /ritom/scratch/cscs/*/llm-finetuning/* || "$BASE_ROOT" == /capstor/scratch/cscs/*/llm-finetuning/* || "$BASE_ROOT" == /iopsstor/scratch/cscs/*/llm-finetuning/* ]]; then
  # Extract relative path after /llm-finetuning/
  BASE_ROOT_REL=$(echo "$BASE_ROOT" | sed 's|.*/llm-finetuning/||')
  echo "📁 Converting absolute path to relative:"
  echo "   Absolute: $BASE_ROOT"
  echo "   Relative: $BASE_ROOT_REL"
  BASE_ROOT="$BASE_ROOT_REL"
fi

SEEDS=(101 202 303)
for SEED in "${SEEDS[@]}"; do
  GEN_GLOB="${BASE_ROOT}/seed${SEED}/job-*/generations.jsonl"
  MET_GLOB="${BASE_ROOT}/seed${SEED}/job-*/metrics_final.csv"

  # Fallback: some legacy jobs may not have job-* nesting
  if ! ls ${GEN_GLOB} >/dev/null 2>&1; then
    GEN_GLOB="${BASE_ROOT}/seed${SEED}/generations.jsonl"
  fi
  if ! ls ${MET_GLOB} >/dev/null 2>&1; then
    MET_GLOB="${BASE_ROOT}/seed${SEED}/metrics_final.csv"
  fi

  # Resolve a single metrics file (universal search expects a single CSV path)
  METRICS_CSV=$(ls -1 ${MET_GLOB} 2>/dev/null | head -n 1 || true)
  if [[ -z "$METRICS_CSV" ]]; then
    echo "⚠️  WARNING: missing metrics_final.csv for seed ${SEED} under ${BASE_ROOT}; skipping"
    echo "   Tried glob: ${MET_GLOB}"
    continue
  fi

  # Convert metrics CSV to relative path if needed
  if [[ "$METRICS_CSV" == /ritom/scratch/cscs/*/llm-finetuning/* || "$METRICS_CSV" == /capstor/scratch/cscs/*/llm-finetuning/* || "$METRICS_CSV" == /iopsstor/scratch/cscs/*/llm-finetuning/* ]]; then
    METRICS_CSV=$(echo "$METRICS_CSV" | sed 's|.*/llm-finetuning/||')
  fi

  # Validate that at least one generations.jsonl file exists (check locally before submitting)
  # Note: We check the glob pattern, but the actual expansion happens inside the container
  GEN_FILES_CHECK=$(ls -1 ${GEN_GLOB} 2>/dev/null | head -n 1 || true)
  if [[ -z "$GEN_FILES_CHECK" ]]; then
    echo "⚠️  WARNING: no generations.jsonl files found for seed ${SEED} under ${BASE_ROOT}; skipping"
    echo "   Tried glob: ${GEN_GLOB}"
    echo "   Base root: ${BASE_ROOT}"
    continue
  fi

  echo "Submitting universal search for seed ${SEED}"
  echo "   Metrics CSV: ${METRICS_CSV}"
  echo "   Generations glob: ${GEN_GLOB}"
  echo "   (Found at least one file: $(basename ${GEN_FILES_CHECK}))"
  if [[ -n "$BATCH_ID" ]]; then
    sbatch scripts/eval_capstor_universal_search.slurm \
      --seed "${SEED}" \
      --generations-jsonl "${GEN_GLOB}" \
      --metrics-csv "${METRICS_CSV}" \
      --batch-id "${BATCH_ID}"
  else
    sbatch scripts/eval_capstor_universal_search.slurm \
      --seed "${SEED}" \
      --generations-jsonl "${GEN_GLOB}" \
      --metrics-csv "${METRICS_CSV}"
  fi
done
