#!/bin/bash
# =============================================================================
# Launch 14B GRPO Training Experiments (Default: 4 Configs) for Multiple Seeds
# =============================================================================
# Usage:
#   # Run all seeds (12 jobs: 4 experiments × 3 seeds)
#   ./scripts/launch_14b_grpo_experiments.sh
#   
#   # Run single seed (4 jobs: 4 experiments × 1 seed)
#   ./scripts/launch_14b_grpo_experiments.sh --seed 101
#   ./scripts/launch_14b_grpo_experiments.sh --seed 202
#   ./scripts/launch_14b_grpo_experiments.sh --seed 303
#
#   # Optional: include Generalization ablation (if config exists)
#   ./scripts/launch_14b_grpo_experiments.sh --include-generalization
#
# This script submits 5 training jobs for each seed:
#   1. Hero
#   2. Ablation: Oracle
#   3. Ablation: Diversity
#   4. Ablation: Minimalist
#   5. Ablation: Prompt
#   (Optional) Ablation: Generalization
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
SELECTED_SEED=""
INCLUDE_GENERALIZATION="false"
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SELECTED_SEED="$2"
            shift 2
            ;;
        --include-generalization)
            INCLUDE_GENERALIZATION="true"
            shift 1
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--seed 101|202|303] [--include-generalization]"
            exit 1
            ;;
    esac
done

# Training parameters
MODEL_SIZE="14B"
ALL_SEEDS=(101 202 303)
TIME_LIMIT="04:00:00"
NODES=3
TASKS=3

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

# Config directory
CONFIG_DIR="$REPO_ROOT/deps/open-r1/recipes/Qwen2.5-Coder-14B-Instruct/grpo"

# Experiment configurations (default: 5 configs; optionally include generalization)
declare -A EXPERIMENTS=(
    ["config_hero.yaml"]="Hero"
    ["config_ablation_oracle.yaml"]="Ablation: Oracle"
    ["config_ablation_diversity.yaml"]="Ablation: Diversity"
    ["config_minimalist.yaml"]="Ablation: Minimalist"
    ["config_ablation_prompt.yaml"]="Ablation: Prompt"
)

if [[ "$INCLUDE_GENERALIZATION" == "true" ]]; then
    # Legacy config location (archived by default). If you want to re-enable it,
    # you can either restore it to the main config dir or point the SLURM training
    # script at the archived path manually.
    EXPERIMENTS["config_ablation_generalization.yaml"]="Ablation: Generalization"
fi

echo "=============================================================================="
if [ ${#SEEDS[@]} -eq 1 ]; then
    echo "Launching 14B GRPO Training Experiments (Single Seed)"
else
    echo "Launching 14B GRPO Training Experiments (Multiple Seeds)"
fi
echo "=============================================================================="
echo "Model:        $MODEL_SIZE"
echo "Seeds:        ${SEEDS[*]}"
echo "Time Limit:   $TIME_LIMIT"
echo "Nodes:        $NODES"
echo "Tasks:        $TASKS"
echo "Total Jobs:   $((${#EXPERIMENTS[@]} * ${#SEEDS[@]})) (${#EXPERIMENTS[@]} experiments × ${#SEEDS[@]} seed(s))"
echo "=============================================================================="
echo ""

# Function to calculate time limit (3x for generalization ablation, only when enabled)
calculate_time_limit() {
    local config_name=$1
    local base_time=$2
    
    # Check if this is the generalization ablation config
    if [[ "$INCLUDE_GENERALIZATION" == "true" && "$config_name" == "config_ablation_generalization.yaml" ]]; then
        # Parse time format (HH:MM:SS) and multiply hours by 3
        IFS=':' read -r hours minutes seconds <<< "$base_time"
        hours=$((hours * 3))
        # Format back to HH:MM:SS with zero padding
        printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
    else
        echo "$base_time"
    fi
}

# Function to submit a job
submit_job() {
    local config_name=$1
    local experiment_name=$2
    local seed=$3
    local dataset=$4
    local config_path_abs="$CONFIG_DIR/$config_name"
    # Use relative path from repo root (required by SLURM script for path translation)
    local config_path_rel="deps/open-r1/recipes/Qwen2.5-Coder-14B-Instruct/grpo/$config_name"
    
    # Calculate actual time limit (3x for generalization ablation)
    local actual_time_limit=$(calculate_time_limit "$config_name" "$TIME_LIMIT")
    
    echo "Submitting: $experiment_name (Seed $seed)"
    echo "  Config: $config_name"
    echo "  Dataset: $dataset"
    if [[ "$actual_time_limit" != "$TIME_LIMIT" ]]; then
        echo "  Time Limit: $actual_time_limit (3x standard: $TIME_LIMIT)"
    else
        echo "  Time Limit: $actual_time_limit"
    fi
    
    if [ ! -f "$config_path_abs" ]; then
        echo "  ⚠️  WARNING: Config file not found; skipping: $config_path_abs"
        echo ""
        return 0
    fi
    
    sbatch \
        --nodes=$NODES \
        --ntasks=$TASKS \
        --time=$actual_time_limit \
        "$SCRIPT_DIR/train_capstor_unified_sds_qwen_coder.slurm" \
        --mode grpo_cold \
        --model $MODEL_SIZE \
        --seed $seed \
        --dataset-name "$dataset" \
        --config "$config_path_rel"
    
    echo "  ✓ Submitted"
    echo ""
}

# Submit all experiments for each seed
TOTAL_JOBS=0
for seed in "${SEEDS[@]}"; do
    dataset="SoheylM/OpenR1-SDS-10k-seed${seed}"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Seed: $seed | Dataset: $dataset"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    for config_name in "${!EXPERIMENTS[@]}"; do
        submit_job "$config_name" "${EXPERIMENTS[$config_name]}" "$seed" "$dataset"
        # Count only attempted submissions; skipping missing configs is still "attempted"
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    done
done

echo "=============================================================================="
echo "All $TOTAL_JOBS jobs submitted successfully!"
echo "=============================================================================="
echo ""
echo "Breakdown:"
for seed in "${SEEDS[@]}"; do
    echo "  Seed $seed: ${#EXPERIMENTS[@]} jobs"
done
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check job outputs in:"
echo "  /ritom/scratch/cscs/\$USER/logs/unified-*.out"
echo ""
echo "Checkpoints will be saved in:"
if [ ${#SEEDS[@]} -eq 1 ]; then
    echo "  /ritom/scratch/cscs/\$USER/checkpoints/Qwen2.5-Coder-14B-Instruct-GRPO-SDS-seed${SEEDS[0]}-config_*/"
else
    echo "  /ritom/scratch/cscs/\$USER/checkpoints/Qwen2.5-Coder-14B-Instruct-GRPO-SDS-seed{101,202,303}-config_*/"
fi
echo ""
