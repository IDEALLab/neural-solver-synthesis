#!/bin/bash
# =============================================================================
# Evaluate 14B GRPO Training Experiments (Default: 4 Configs) for Multiple Seeds
# =============================================================================
# Usage:
#   # Evaluate all seeds (12 jobs: 4 experiments × 3 seeds)
#   ./scripts/evaluate_14b_grpo_experiments.sh
#   
#   # Evaluate single seed (4 jobs: 4 experiments × 1 seed)
#   ./scripts/evaluate_14b_grpo_experiments.sh --seed 101
#   ./scripts/evaluate_14b_grpo_experiments.sh --seed 202
#   ./scripts/evaluate_14b_grpo_experiments.sh --seed 303
#
#   # Optional: include Generalization ablation (if checkpoints exist)
#   ./scripts/evaluate_14b_grpo_experiments.sh --include-generalization
#
# This script:
#   1. Finds the latest checkpoint for each experiment
#   2. Submits evaluation jobs for all 5 experiments
#   3. Uses the correct output directory structure with ablation tags
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
SELECTED_SEED=""
INCLUDE_GENERALIZATION="false"
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
        --include-generalization)
            INCLUDE_GENERALIZATION="true"
            shift 1
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--seed 101|202|303] [--batch-id YYYYMMDD_name] [--include-generalization]"
            exit 1
            ;;
    esac
done

# Configuration
MODEL_SIZE="14B"
ALL_SEEDS=(101 202 303)
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/ritom/scratch/cscs/${USER}/checkpoints}"

# Determine which seeds to run
if [ -n "$SELECTED_SEED" ]; then
    # Validate seed
    if [[ ! " ${ALL_SEEDS[@]} " =~ " ${SELECTED_SEED} " ]]; then
        echo "ERROR: Invalid seed '$SELECTED_SEED'. Must be one of: ${ALL_SEEDS[*]}"
        exit 1
    fi
    SEEDS=("$SELECTED_SEED")
else
    SEEDS=("${ALL_SEEDS[@]}")
fi

# Config names (default: 5 configs; optionally include generalization)
declare -A CONFIG_NAMES=(
    ["config_hero"]="Hero"
    ["config_ablation_oracle"]="Ablation: Oracle"
    ["config_ablation_diversity"]="Ablation: Diversity"
    ["config_minimalist"]="Ablation: Minimalist"
    ["config_ablation_prompt"]="Ablation: Prompt"
)

if [[ "$INCLUDE_GENERALIZATION" == "true" ]]; then
    CONFIG_NAMES["config_ablation_generalization"]="Ablation: Generalization"
fi

echo "=============================================================================="
if [ ${#SEEDS[@]} -eq 1 ]; then
    echo "Evaluating 14B GRPO Training Experiments (Single Seed)"
else
    echo "Evaluating 14B GRPO Training Experiments (Multiple Seeds)"
fi
echo "=============================================================================="
echo "Model:        $MODEL_SIZE"
echo "Seeds:        ${SEEDS[*]}"
echo "Checkpoint Root: $CHECKPOINT_ROOT"
if [ -n "$BATCH_ID" ]; then
    echo "Batch ID:     $BATCH_ID"
else
    echo "Batch ID:     (none) -> writing to evaluation/sds/results/ (legacy path)"
fi
echo "Total Jobs:   $((${#CONFIG_NAMES[@]} * ${#SEEDS[@]})) (${#CONFIG_NAMES[@]} experiments × ${#SEEDS[@]} seed(s))"
echo "=============================================================================="
echo ""

# Function to find latest checkpoint for a given config and seed
find_latest_checkpoint() {
    local config_name=$1
    local seed=$2
    local base_pattern="Qwen2.5-Coder-${MODEL_SIZE}-Instruct-GRPO-SDS-seed${seed}"
    local base_dir="${CHECKPOINT_ROOT}/${base_pattern}-${config_name}"
    
    if [ ! -d "$base_dir" ]; then
        echo "  ⚠️  WARNING: Base directory not found: $base_dir" >&2
        return 1
    fi
    
    # Check for both structures:
    # 1. New structure: base_dir/job-*/checkpoint-*
    # 2. Old structure: base_dir/checkpoint-* (directly in base)
    
    local latest_checkpoint=""
    local latest_job_id=""
    local latest_checkpoint_num=0
    
    # First, try new structure (job-* subdirectories)
    local job_dirs=("$base_dir"/job-*)
    if [ -e "${job_dirs[0]}" ] 2>/dev/null; then
        # New structure: find latest job by job ID, then latest checkpoint in that job
        local latest_job_dir=""
        local latest_job_id_num=0
        
        # Step 1: Find the job directory with the highest job ID
        for job_dir in "${job_dirs[@]}"; do
            if [ ! -d "$job_dir" ]; then
                continue
            fi
            
            # Extract job ID as number
            local job_id_str=$(basename "$job_dir" | sed 's/job-//')
            if [[ "$job_id_str" =~ ^[0-9]+$ ]]; then
                local job_id_num=$((10#$job_id_str))  # Force base-10 interpretation
                if [ "$job_id_num" -gt "$latest_job_id_num" ] 2>/dev/null; then
                    latest_job_id_num=$job_id_num
                    latest_job_dir="$job_dir"
                    latest_job_id="$job_id_str"
                fi
            fi
        done
        
        # Step 2: Find the latest checkpoint within the latest job directory
        if [ -n "$latest_job_dir" ] && [ -d "$latest_job_dir" ]; then
            local checkpoint_dirs=("$latest_job_dir"/checkpoint-*)
            if [ -e "${checkpoint_dirs[0]}" ] 2>/dev/null; then
                for checkpoint_dir in "${checkpoint_dirs[@]}"; do
                    if [ -d "$checkpoint_dir" ]; then
                        local checkpoint_num=$(basename "$checkpoint_dir" | sed 's/checkpoint-//')
                        # Validate it's a number
                        if [[ "$checkpoint_num" =~ ^[0-9]+$ ]] && [ "$checkpoint_num" -gt "$latest_checkpoint_num" ] 2>/dev/null; then
                            latest_checkpoint_num=$checkpoint_num
                            latest_checkpoint="$checkpoint_dir"
                        fi
                    fi
                done
            fi
        fi
    fi
    
    # If no checkpoints found in job-* structure, try old structure (directly in base)
    if [ -z "$latest_checkpoint" ]; then
        local checkpoint_dirs=("$base_dir"/checkpoint-*)
        if [ -e "${checkpoint_dirs[0]}" ] 2>/dev/null; then
            for checkpoint_dir in "${checkpoint_dirs[@]}"; do
                if [ -d "$checkpoint_dir" ]; then
                    local checkpoint_num=$(basename "$checkpoint_dir" | sed 's/checkpoint-//')
                    # Validate it's a number
                    if [[ "$checkpoint_num" =~ ^[0-9]+$ ]] && [ "$checkpoint_num" -gt "$latest_checkpoint_num" ] 2>/dev/null; then
                        latest_checkpoint_num=$checkpoint_num
                        latest_checkpoint="$checkpoint_dir"
                        # Try to extract job ID from path if possible
                        if [[ "$checkpoint_dir" =~ /job-([0-9]+)/ ]]; then
                            latest_job_id="${BASH_REMATCH[1]}"
                        else
                            latest_job_id="unknown"
                        fi
                    fi
                fi
            done
        fi
    fi
    
    if [ -z "$latest_checkpoint" ]; then
        echo "  ⚠️  WARNING: Could not find any valid checkpoints in: $base_dir" >&2
        return 1
    fi
    
    echo "$latest_checkpoint|$latest_job_id"
    return 0
}

# Function to submit an evaluation job
submit_eval_job() {
    local config_name=$1
    local experiment_name=$2
    local seed=$3
    
    echo "Processing: $experiment_name (Seed $seed)"
    echo "  Config: $config_name"
    
    # Find latest checkpoint
    local result=$(find_latest_checkpoint "$config_name" "$seed")
    if [ $? -ne 0 ] || [ -z "$result" ]; then
        echo "  ⚠️  WARNING: Could not find checkpoint; skipping"
        echo ""
        return 0
    fi
    
    local checkpoint_dir=$(echo "$result" | cut -d'|' -f1)
    local job_id=$(echo "$result" | cut -d'|' -f2)
    
    echo "  Checkpoint: $checkpoint_dir"
    echo "  Job ID: $job_id"
    
    # Submit evaluation job
    if [ -n "$BATCH_ID" ]; then
        export BATCH_ID
    fi
    sbatch \
        "$SCRIPT_DIR/eval_capstor_sds_pipeline.slurm" \
        --checkpoint-dir "$checkpoint_dir" \
        grpo \
        "$seed"
    
    echo "  ✓ Evaluation job submitted"
    echo ""
}

# Submit evaluations for all experiments for each seed
TOTAL_JOBS=0
for seed in "${SEEDS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Seed: $seed"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    for config_name in "${!CONFIG_NAMES[@]}"; do
        submit_eval_job "$config_name" "${CONFIG_NAMES[$config_name]}" "$seed"
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    done
done

echo "=============================================================================="
echo "All $TOTAL_JOBS evaluation jobs submitted!"
echo "=============================================================================="
echo ""
echo "Breakdown:"
for seed in "${SEEDS[@]}"; do
    echo "  Seed $seed: ${#CONFIG_NAMES[@]} jobs"
done
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check job outputs in:"
echo "  /ritom/scratch/cscs/\$USER/logs/eval-*.out"
echo ""
