#!/bin/bash -l
#SBATCH --job-name=unified-sds-train
#SBATCH --account=a0105
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1          # Default: single node (override with sbatch --nodes=3 for 14B/32B)
#SBATCH --ntasks=1         # Default: single task (override with sbatch --ntasks=3 for 14B/32B)
#SBATCH --gpus-per-node=4
#SBATCH --output=/ritom/scratch/cscs/%u/logs/unified-%j.out
#SBATCH --error=/ritom/scratch/cscs/%u/logs/unified-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smassoudi@ethz.ch  # Update with your email

# =============================================================================
# UNIFIED SDS TRAINING SCRIPT
# =============================================================================
# Usage:
#   # 7B models (single node - default)
#   sbatch train_unified_sds_qwen_coder.slurm --mode sft --model 7B --seed 101
#   sbatch train_unified_sds_qwen_coder.slurm --mode grpo_cold --model 7B --seed 101
#   sbatch train_unified_sds_qwen_coder.slurm --mode grpo_sft --model 7B --seed 101
#   
#   # 14B/32B models (multi-node - override nodes/tasks)
#   sbatch --nodes=3 --ntasks=3 train_unified_sds_qwen_coder.slurm --mode grpo_cold --model 14B --seed 101
#   sbatch --nodes=3 --ntasks=3 train_unified_sds_qwen_coder.slurm --mode grpo_cold --model 32B --seed 101
#
# Modes:
#   sft:        SFT training from base model
#   grpo_cold:  GRPO training from base model (no SFT)
#   grpo_sft:   GRPO training from SFT checkpoint
#   grpo_resume: Resume GRPO training from existing checkpoint
#
# Model sizes: 
#   7B:  Single node (default SBATCH settings)
#   14B: Multi-node (requires --nodes=3 --ntasks=3: 1 vLLM + 2 training)
#   32B: Multi-node (requires --nodes=3 --ntasks=3: 1 vLLM + 2 training)
# =============================================================================

# ----------- 1. ARGUMENT PARSING & CONFIGURATION ----------------------------
MODE="grpo_cold"       # sft | grpo_cold | grpo_sft | grpo_resume
MODEL_SIZE="7B"        # 7B | 14B | 32B
SEED=101
DATASET_SIZE="10k"
RESUME_CHECKPOINT=""   # For grpo_resume mode
USE_LORA=false         # Default to full fine-tuning (set to true via --use-lora flag)
CUSTOM_CONFIG=""       # Custom config file path for ablation experiments
CUSTOM_DATASET=""      # Custom dataset name (full override, e.g., "SoheylM/ShinkaEvolve-SDS-1000-seed101")

# Safety: by default, do NOT allow the legacy Generalization ablation to run.
# To explicitly re-enable it, submit with: sbatch --export=INCLUDE_GENERALIZATION=1 ...
INCLUDE_GENERALIZATION="${INCLUDE_GENERALIZATION:-0}"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift ;;
        --model) MODEL_SIZE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --dataset) DATASET_SIZE="$2"; shift ;;
        --dataset-name) CUSTOM_DATASET="$2"; shift ;;  # Full dataset name override
        --checkpoint) RESUME_CHECKPOINT="$2"; shift ;;
        --config) CUSTOM_CONFIG="$2"; shift ;;  # Custom config file for ablations
        --use-lora) USE_LORA=true ;;  # Enable LoRA (PEFT) for reduced VRAM/storage
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate seed
if [[ ! "$SEED" =~ ^(101|202|303)$ ]]; then
    echo "ERROR: Seed must be 101, 202, or 303 for paper reproducibility"
    exit 1
fi

# Validate mode
if [[ ! "$MODE" =~ ^(sft|grpo_cold|grpo_sft|grpo_resume)$ ]]; then
    echo "ERROR: Mode must be sft, grpo_cold, grpo_sft, or grpo_resume"
    exit 1
fi

# Validate model size
if [[ ! "$MODEL_SIZE" =~ ^(7B|14B|32B)$ ]]; then
    echo "ERROR: Model size must be 7B, 14B, or 32B"
    exit 1
fi

echo "=== Unified SDS Training Configuration ==="
echo "Mode:         $MODE"
echo "Model:        $MODEL_SIZE"
echo "Seed:         $SEED"
echo "Dataset:      ${CUSTOM_DATASET:-${DATASET_SIZE}}"
echo "LoRA:         $USE_LORA"
echo "Custom Config: ${CUSTOM_CONFIG:-(default)}"
echo "Job ID:       $SLURM_JOB_ID"
echo "Date:         $(date)"
echo "=========================================="

# Validate custom config file if provided
if [ -n "$CUSTOM_CONFIG" ]; then
    if [ ! -f "$CUSTOM_CONFIG" ]; then
        echo "CRITICAL ERROR: Custom config file not found at: $CUSTOM_CONFIG"
        exit 1
    fi
    # Guardrail: prevent accidental training of legacy generalization ablation
    if [[ "$CUSTOM_CONFIG" == *"config_ablation_generalization.yaml"* ]] && [[ "$INCLUDE_GENERALIZATION" != "1" ]]; then
        echo "⚠️  WARNING: Refusing to run legacy Generalization ablation by default."
        echo "    Config: $CUSTOM_CONFIG"
        echo "    To override: sbatch --export=INCLUDE_GENERALIZATION=1 ..."
        echo "    Exiting without launching training."
        exit 0
    fi
    echo ">>> Verified Custom Config exists: $CUSTOM_CONFIG"
fi

# ----------- 2. LORA CONFIGURATION --------------------------------------------
# Configure PEFT (LoRA) arguments if enabled
PEFT_ARGS=""
LORA_GRPO_OVERRIDES=""  # Additional config overrides for GRPO when using LoRA
if [ "$USE_LORA" = true ]; then
    echo ">>> LoRA ENABLED (PEFT)"
    # Standard LoRA config for reasoning models
    # Target all linear layers in attention and MLP blocks
    PEFT_ARGS="--use_peft true --lora_r 16 --lora_alpha 32 --lora_dropout 0.1 --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
    # For GRPO with LoRA: use SDPA instead of flash_attention_2, disable gradient checkpointing
    LORA_GRPO_OVERRIDES="--attn_implementation sdpa --gradient_checkpointing false"
else
    echo ">>> LoRA DISABLED (Full Fine-Tuning)"
fi

# ----------- 3. PATH & MODEL CONFIGURATION -----------------------------------
USER_SCRATCH="/ritom/scratch/cscs/${USER}"
CHECKPOINT_ROOT="${USER_SCRATCH}/checkpoints"
LOG_DIR="${USER_SCRATCH}/logs"

# Function to convert config name to readable ablation tag for W&B
normalize_config_name() {
    local config_name=$1
    # Remove "config_" prefix if present
    local name="${config_name#config_}"
    
    # Map to readable abbreviations
    case "$name" in
        "hero")
            echo "hero"
            ;;
        "discovery")
            echo "discovery"
            ;;
        "ablation_no_oracle")
            echo "no-oracle"
            ;;
        "ablation_oracle")
            echo "oracle"
            ;;
        "ablation_no_gen")
            echo "no-gen"
            ;;
        "ablation_generalization")
            echo "gen"
            ;;
        "ablation_no_diversity")
            echo "no-div"
            ;;
        "ablation_diversity")
            echo "div"
            ;;
        "ablation_neutral_prompt")
            echo "neutral-prompt"
            ;;
        "ablation_prompt")
            echo "prompt"
            ;;
        "minimalist")
            echo "minimalist"
            ;;
        *)
            # Fallback: remove "ablation_" prefix and replace underscores with hyphens
            echo "$name" | sed 's/^ablation_//' | tr '_' '-'
            ;;
    esac
}

# Map model size to HF model and configs
case $MODEL_SIZE in
    "7B")
        BASE_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
        ACCEL_CONFIG_SFT="recipes/accelerate_configs/zero2_single_node_4GPUs.yaml"
        ACCEL_CONFIG_GRPO_SINGLE="recipes/accelerate_configs/zero2_single_node.yaml"
        GRPO_CONFIG="recipes/Qwen2.5-Coder-7B-Instruct/grpo/config_demo_sm_sds_opt.yaml"
        SFT_CONFIG="recipes/Qwen2.5-Coder-7B-Instruct/sft/config_sds.yaml"
        MODEL_NAME_LOWER="qwen2.5-coder-7b"
        ;;
    "14B")
        BASE_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
        ACCEL_CONFIG_GRPO_MULTI="recipes/accelerate_configs/zero3_multi_node_14b.yaml"
        ACCEL_CONFIG_SFT="recipes/accelerate_configs/zero3_single_node_14b.yaml"  # For SFT: single-node with ZeRO-3
        GRPO_CONFIG="recipes/Qwen2.5-Coder-14B-Instruct/grpo/config_demo_sm_sds_opt.yaml"
        SFT_CONFIG="recipes/Qwen2.5-Coder-14B-Instruct/sft/config_sds.yaml"
        MODEL_NAME_LOWER="qwen2.5-coder-14b"
        # 14B GRPO requires multi-node setup (needs vLLM server), but SFT can run on single-node
        if [ "$SLURM_NNODES" -eq 1 ] && [ "$MODE" != "sft" ]; then
            echo "ERROR: 14B GRPO requires multi-node setup (needs vLLM server)"
            echo "Please run: sbatch --nodes=3 --ntasks=3 train_unified_sds_qwen_coder.slurm --mode $MODE --model 14B --seed $SEED"
            echo "Or use --mode sft for single-node SFT training"
            exit 1
        fi
        ;;
    "32B")
        BASE_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
        ACCEL_CONFIG_GRPO_MULTI="recipes/accelerate_configs/zero3_multi_node_32b.yaml"
        ACCEL_CONFIG_SFT="recipes/accelerate_configs/zero3_single_node_4GPUs.yaml"  # For SFT: single-node with ZeRO-3
        GRPO_CONFIG="recipes/Qwen2.5-Coder-32B-Instruct/grpo/config_demo_sm_sds_opt.yaml"
        SFT_CONFIG="recipes/Qwen2.5-Coder-32B-Instruct/sft/config_sds.yaml"
        MODEL_NAME_LOWER="qwen2.5-coder-32b"
        # 32B requires 3-node setup for GRPO (1 vLLM + 2 training), but SFT can run on single-node
        if [ "$SLURM_NNODES" -lt 3 ] && [ "$MODE" != "sft" ]; then
            echo "ERROR: 32B GRPO requires 3-node setup (1 vLLM + 2 training)"
            echo "Please run: sbatch --nodes=3 --ntasks=3 train_unified_sds_qwen_coder.slurm --mode $MODE --model 32B --seed $SEED"
            echo "Or use --mode sft for single-node SFT training"
            exit 1
        fi
        ;;
esac

# Determine input model and output directory based on mode
case $MODE in
    "sft")
        INPUT_MODEL="$BASE_MODEL"
        BASE_OUTPUT_DIR="${CHECKPOINT_ROOT}/Qwen2.5-Coder-${MODEL_SIZE}-Instruct-SFT-SDS-seed${SEED}"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/job-${SLURM_JOB_ID}"
        TASK_SCRIPT="src/open_r1/sft.py"
        TASK_CONFIG="$SFT_CONFIG"
        SCHEME_NAME="sft"
        ;;
    "grpo_cold")
        INPUT_MODEL="$BASE_MODEL"
        TASK_SCRIPT="src/open_r1/grpo.py"
        
        # Handle Config Override
        if [ -n "$CUSTOM_CONFIG" ]; then
            TASK_CONFIG="$CUSTOM_CONFIG"
            # Extract filename without extension for tagging (e.g., "config_ablation_no_gen")
            CONFIG_NAME=$(basename "$CUSTOM_CONFIG" .yaml)
            # Use full config name for directory (backward compatibility)
            CONFIG_TAG="-${CONFIG_NAME}"
            # Use readable abbreviation for W&B run name
            WANDB_CONFIG_TAG=$(normalize_config_name "$CONFIG_NAME")
        else
            TASK_CONFIG="$GRPO_CONFIG"
            CONFIG_TAG=""
            WANDB_CONFIG_TAG=""
        fi
        
        SCHEME_NAME="grpo${CONFIG_TAG}"
        BASE_OUTPUT_DIR="${CHECKPOINT_ROOT}/Qwen2.5-Coder-${MODEL_SIZE}-Instruct-GRPO-SDS-seed${SEED}${CONFIG_TAG}"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/job-${SLURM_JOB_ID}"
        ;;
    "grpo_sft")
        # Input is the SFT checkpoint (need to find latest job)
        SFT_BASE_DIR="${CHECKPOINT_ROOT}/Qwen2.5-Coder-${MODEL_SIZE}-Instruct-SFT-SDS-seed${SEED}"
        if [ ! -d "$SFT_BASE_DIR" ]; then
            echo "CRITICAL ERROR: SFT checkpoint directory not found at $SFT_BASE_DIR"
            exit 1
        fi
        # Find latest job directory
        LATEST_SFT_JOB=$(ls -d "${SFT_BASE_DIR}"/job-* 2>/dev/null | sort -V | tail -n 1)
        if [ -z "$LATEST_SFT_JOB" ]; then
            # Backward compatibility: check if old structure (no job-* subdirs)
            if [ -f "${SFT_BASE_DIR}/config.json" ]; then
                INPUT_MODEL="$SFT_BASE_DIR"
            else
                echo "CRITICAL ERROR: No SFT checkpoints found in $SFT_BASE_DIR"
                exit 1
            fi
        else
            # Find latest checkpoint in latest job
            LATEST_CHECKPOINT=$(find "$LATEST_SFT_JOB" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
            if [ -z "$LATEST_CHECKPOINT" ]; then
                INPUT_MODEL="$LATEST_SFT_JOB"
            else
                INPUT_MODEL="$LATEST_CHECKPOINT"
            fi
        fi
        TASK_SCRIPT="src/open_r1/grpo.py"
        
        # Handle Config Override
        if [ -n "$CUSTOM_CONFIG" ]; then
            TASK_CONFIG="$CUSTOM_CONFIG"
            # Extract filename without extension for tagging (e.g., "config_ablation_no_gen")
            CONFIG_NAME=$(basename "$CUSTOM_CONFIG" .yaml)
            # Use full config name for directory (backward compatibility)
            CONFIG_TAG="-${CONFIG_NAME}"
            # Use readable abbreviation for W&B run name
            WANDB_CONFIG_TAG=$(normalize_config_name "$CONFIG_NAME")
        else
            TASK_CONFIG="$GRPO_CONFIG"
            CONFIG_TAG=""
            WANDB_CONFIG_TAG=""
        fi
        
        SCHEME_NAME="sft-grpo${CONFIG_TAG}"
        BASE_OUTPUT_DIR="${CHECKPOINT_ROOT}/Qwen2.5-Coder-${MODEL_SIZE}-Instruct-SFT-GRPO-SDS-seed${SEED}${CONFIG_TAG}"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/job-${SLURM_JOB_ID}"
        echo "Using SFT checkpoint: $INPUT_MODEL"
        ;;
    "grpo_resume")
        # Input is base model, but we'll resume from existing checkpoint
        INPUT_MODEL="$BASE_MODEL"
        BASE_OUTPUT_DIR="${CHECKPOINT_ROOT}/Qwen2.5-Coder-${MODEL_SIZE}-Instruct-GRPO-SDS-seed${SEED}"
        if [ -z "$RESUME_CHECKPOINT" ]; then
            # Find latest job and checkpoint
            LATEST_JOB=$(ls -d "${BASE_OUTPUT_DIR}"/job-* 2>/dev/null | sort -V | tail -n 1)
            if [ -z "$LATEST_JOB" ]; then
                echo "CRITICAL ERROR: No existing GRPO checkpoints found to resume from"
                echo "  Expected directory: $BASE_OUTPUT_DIR"
                exit 1
            fi
            RESUME_CHECKPOINT=$(find "$LATEST_JOB" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -n 1)
            if [ -z "$RESUME_CHECKPOINT" ]; then
                echo "CRITICAL ERROR: No checkpoint found in $LATEST_JOB"
                echo "  Available directories:"
                ls -la "$LATEST_JOB" 2>/dev/null || echo "  (directory not accessible)"
                exit 1
            fi
        fi
        # Validate that the resume checkpoint actually exists
        if [ ! -d "$RESUME_CHECKPOINT" ]; then
            echo "CRITICAL ERROR: Resume checkpoint directory does not exist: $RESUME_CHECKPOINT"
            exit 1
        fi
        # Validate that it's a valid checkpoint (has trainer_state.json or adapter files)
        if [ ! -f "$RESUME_CHECKPOINT/trainer_state.json" ] && [ ! -f "$RESUME_CHECKPOINT/adapter_config.json" ]; then
            echo "WARNING: Resume checkpoint may be incomplete (missing trainer_state.json or adapter_config.json)"
            echo "  Proceeding anyway, but training may fail..."
        fi
        TASK_SCRIPT="src/open_r1/grpo.py"
        
        # Handle Config Override
        if [ -n "$CUSTOM_CONFIG" ]; then
            TASK_CONFIG="$CUSTOM_CONFIG"
            # Extract filename without extension for tagging (e.g., "config_ablation_no_gen")
            CONFIG_NAME=$(basename "$CUSTOM_CONFIG" .yaml)
            # Use full config name for directory (backward compatibility)
            CONFIG_TAG="-${CONFIG_NAME}"
            # Use readable abbreviation for W&B run name
            WANDB_CONFIG_TAG=$(normalize_config_name "$CONFIG_NAME")
        else
            TASK_CONFIG="$GRPO_CONFIG"
            CONFIG_TAG=""
            WANDB_CONFIG_TAG=""
        fi
        
        SCHEME_NAME="grpo${CONFIG_TAG}"
        BASE_OUTPUT_DIR="${CHECKPOINT_ROOT}/Qwen2.5-Coder-${MODEL_SIZE}-Instruct-GRPO-SDS-seed${SEED}${CONFIG_TAG}"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/job-${SLURM_JOB_ID}"
        echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
        ;;
esac

# ----------- 3. ENVIRONMENT SETUP ---------------------------------------------
export HF_HOME="${USER_SCRATCH}/models"
export HF_DATASETS_CACHE="${USER_SCRATCH}/hf_datasets_cache"
export WANDB_DIR="${LOG_DIR}/wandb"
export TRITON_CACHE_DIR="${LOG_DIR}/.triton_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gh200-llm-sds-training-clariden}"

export HF_TOKEN=$(< $HOME/llm/hf_token.txt)
export HF_HUB_TOKEN=$(< $HOME/llm/hf_token.txt)
export WANDB_API_KEY=$(< $HOME/llm/wandb_token.txt)
export WANDB_ENTITY="${WANDB_ENTITY:-smassoudi-eth-z-rich}"  # Update with your W&B entity
export WANDB_PROJECT="qwen-coder-sds-rl"

# Construct WandB run name: qwen2.5-coder-7b-grpo-sds-seed101-job12345
# Use readable config tag for W&B if available, otherwise use full scheme name
if [ -n "$WANDB_CONFIG_TAG" ]; then
    # Extract base scheme (remove config tag from SCHEME_NAME)
    # SCHEME_NAME is like "grpo-config_ablation_no_gen" or "sft-grpo-config_ablation_no_gen"
    # We want just "grpo" or "sft-grpo"
    BASE_SCHEME="${SCHEME_NAME%%-config_*}"  # Remove everything from "-config_" onwards
    if [ "$BASE_SCHEME" == "$SCHEME_NAME" ]; then
        # No config tag found, try alternative pattern
        BASE_SCHEME="${SCHEME_NAME%%-config*}"
    fi
    # Use readable abbreviation (e.g., "grpo-no-gen" instead of "grpo-config_ablation_no_gen")
    WANDB_RUN_NAME="${MODEL_NAME_LOWER}-${BASE_SCHEME}-${WANDB_CONFIG_TAG}-sds-seed${SEED}-job${SLURM_JOB_ID}"
else
    # Fallback to original scheme name
    WANDB_RUN_NAME="${MODEL_NAME_LOWER}-${SCHEME_NAME}-sds-seed${SEED}-job${SLURM_JOB_ID}"
fi

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$WANDB_DIR" "$TRITON_CACHE_DIR" "$LOG_DIR" "$CHECKPOINT_ROOT"

# Validate and clean incomplete checkpoints before training
echo "Validating checkpoints for seed $SEED..."
if [ -d "$BASE_OUTPUT_DIR" ]; then
    # Check if new structure (job-* subdirectories) or old structure
    if ls -d "${BASE_OUTPUT_DIR}"/job-* 1> /dev/null 2>&1; then
        # New structure: clean in all job directories
        for job_dir in "${BASE_OUTPUT_DIR}"/job-*; do
            for checkpoint in "$job_dir"/checkpoint-*; do
                if [ -d "$checkpoint" ] && [ ! -f "$checkpoint/trainer_state.json" ]; then
                    echo "Removing incomplete checkpoint: $checkpoint"
                    rm -rf "$checkpoint"
                fi
            done
        done
    else
        # Old structure: clean directly in base directory
        for checkpoint in "${BASE_OUTPUT_DIR}"/checkpoint-*; do
            if [ -d "$checkpoint" ] && [ ! -f "$checkpoint/trainer_state.json" ]; then
                echo "Removing incomplete checkpoint: $checkpoint"
                rm -rf "$checkpoint"
            fi
        done
    fi
    echo "Checkpoint validation complete"
else
    echo "No existing checkpoint directory found for seed $SEED"
fi

# ----------- 4. NETWORK SETUP (for multi-node) --------------------------------
export MASTER_NODE=$(hostname -i)
NUM_NODES=$SLURM_NNODES

if [ "$NUM_NODES" -gt 1 ]; then
    export ALL_NODES=($(hostlist -e $SLURM_NODELIST))
    export MASTER_WORKER_NODE=${ALL_NODES[1]}
    export MASTER_WORKER_IP=$(srun -n1 -N1 --nodelist=$MASTER_WORKER_NODE hostname -i)
    echo "Multi-node setup:"
    echo "  Master node (vLLM): $MASTER_NODE"
    echo "  Master worker IP:  $MASTER_WORKER_IP"
else
    echo "Single-node setup: $MASTER_NODE"
fi

# ----------- 5. DATASET CONFIGURATION -----------------------------------------
# Different datasets for SFT vs GRPO
# SFT uses: SoheylM/ShinkaEvolve-SDS-100-seed{SEED}
# GRPO uses: SoheylM/OpenR1-SDS-10k-seed{SEED}
# If CUSTOM_DATASET is provided, it overrides the default
if [[ -n "$CUSTOM_DATASET" ]]; then
    DATASET_FULL_NAME="$CUSTOM_DATASET"
elif [[ "$MODE" == "sft" ]]; then
    DATASET_FULL_NAME="SoheylM/ShinkaEvolve-SDS-100-seed${SEED}"
else
    DATASET_FULL_NAME="SoheylM/OpenR1-SDS-${DATASET_SIZE}-seed${SEED}"
fi

# ----------- 6. EXECUTION LOGIC ------------------------------------------------
srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -c '
    # Export variables needed inside container
    export SEED='"${SEED}"'
    export MODE='"${MODE}"'
    export MODEL_SIZE='"${MODEL_SIZE}"'
    export DATASET_FULL_NAME='"${DATASET_FULL_NAME}"'
    export INPUT_MODEL='"${INPUT_MODEL}"'
    export OUTPUT_DIR='"${OUTPUT_DIR}"'
    export TASK_SCRIPT='"${TASK_SCRIPT}"'
    export TASK_CONFIG='"${TASK_CONFIG}"'
    export WANDB_ENTITY='"${WANDB_ENTITY}"'
    export WANDB_PROJECT='"${WANDB_PROJECT}"'
    export WANDB_RUN_NAME='"${WANDB_RUN_NAME}"'
    export MASTER_NODE='"${MASTER_NODE}"'
    export NUM_NODES='"${NUM_NODES}"'
    export RESUME_CHECKPOINT='"${RESUME_CHECKPOINT}"'
    
    # Export HuggingFace cache directories (ensure they are set inside container)
    export HF_HOME='"${HF_HOME}"'
    export HF_DATASETS_CACHE='"${HF_DATASETS_CACHE}"'
    
    # Clear cache for the specific dataset to force fresh download
    # This ensures we get the latest version from HuggingFace Hub
    if [[ -n "$DATASET_FULL_NAME" ]]; then
        echo ">>> Clearing cache for dataset: $DATASET_FULL_NAME"
        # Extract dataset org and name
        DATASET_ORG=$(echo "$DATASET_FULL_NAME" | cut -d/ -f1)
        DATASET_NAME=$(echo "$DATASET_FULL_NAME" | cut -d/ -f2)
        
        if [[ -d "$HF_DATASETS_CACHE" ]]; then
            # Clear all cache entries matching the dataset name (comprehensive search)
            echo ">>> Searching for cached files matching: $DATASET_NAME"
            CACHE_COUNT=$(find "$HF_DATASETS_CACHE" -type f -o -type d 2>/dev/null | grep -i "${DATASET_NAME}" | wc -l || echo "0")
            if [[ "$CACHE_COUNT" -gt 0 ]]; then
                echo ">>> Found $CACHE_COUNT cache entries, removing..."
                find "$HF_DATASETS_CACHE" -type d -name "*${DATASET_NAME}*" -exec rm -rf {} + 2>/dev/null || true
                find "$HF_DATASETS_CACHE" -type f -name "*${DATASET_NAME}*" -delete 2>/dev/null || true
                # Also clear by org/name path if it exists
                if [[ -d "$HF_DATASETS_CACHE/${DATASET_ORG}___${DATASET_NAME}" ]]; then
                    rm -rf "$HF_DATASETS_CACHE/${DATASET_ORG}___${DATASET_NAME}" 2>/dev/null || true
                fi
                echo ">>> Cache cleared for $DATASET_FULL_NAME"
            else
                echo ">>> No cached files found for $DATASET_FULL_NAME (will download fresh)"
            fi
        else
            echo ">>> Warning: HF_DATASETS_CACHE directory not found: $HF_DATASETS_CACHE"
            echo ">>> Will create cache directory and download fresh dataset"
        fi
    fi
    
    # NOTE: PEFT_ARGS and LORA_GRPO_OVERRIDES are NOT exported here.
    # They are injected directly into the command string below via '"${PEFT_ARGS}"' syntax.
    
    # Translate paths for container (if local paths)
    # NOTE: This assumes container mount points match these patterns.
    # If your container mounts differ, update these replacements accordingly.
    # Check if INPUT_MODEL is a local path on Alps scratch.
    if [[ "$INPUT_MODEL" =~ ^/(ritom|iopsstor|capstor)/ ]]; then
        CONTAINER_INPUT_MODEL="${INPUT_MODEL/\/ritom\/scratch\/cscs\/$USER/\/workspace}"
        CONTAINER_INPUT_MODEL="${CONTAINER_INPUT_MODEL/\/iopsstor\/scratch\/cscs\/$USER/\/workspace}"
        CONTAINER_INPUT_MODEL="${CONTAINER_INPUT_MODEL/\/capstor\/scratch\/cscs\/$USER/\/workspace}"
    else
        # HuggingFace model ID - use as-is
        CONTAINER_INPUT_MODEL="$INPUT_MODEL"
    fi
    
    CONTAINER_OUTPUT_DIR="${OUTPUT_DIR/\/ritom\/scratch\/cscs\/$USER/\/workspace}"
    CONTAINER_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR/\/iopsstor\/scratch\/cscs\/$USER/\/workspace}"
    CONTAINER_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR/\/capstor\/scratch\/cscs\/$USER/\/workspace}"
    
    CONTAINER_RESUME_CHECKPOINT=""
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        CONTAINER_RESUME_CHECKPOINT="${RESUME_CHECKPOINT/\/ritom\/scratch\/cscs\/$USER/\/workspace}"
        CONTAINER_RESUME_CHECKPOINT="${CONTAINER_RESUME_CHECKPOINT/\/iopsstor\/scratch\/cscs\/$USER/\/workspace}"
        CONTAINER_RESUME_CHECKPOINT="${CONTAINER_RESUME_CHECKPOINT/\/capstor\/scratch\/cscs\/$USER/\/workspace}"
    fi
    
    # Set accelerate config based on model size (for multi-node)
    if [[ "$MODEL_SIZE" == "14B" ]]; then
        ACCEL_CONFIG_MULTI="recipes/accelerate_configs/zero3_multi_node_14b.yaml"
    elif [[ "$MODEL_SIZE" == "32B" ]]; then
        ACCEL_CONFIG_MULTI="recipes/accelerate_configs/zero3_multi_node_32b.yaml"
    fi
    
    # --------------------------------------------------------------------------
    # CASE A: SFT MODE
    # --------------------------------------------------------------------------
    if [[ "$MODE" == "sft" ]]; then
        # Select accelerate config based on model size and number of nodes
        if [[ "$NUM_NODES" -eq 1 ]]; then
            # Single-node SFT
            echo ">>> Starting SFT training on single node (all GPUs)"
            if [[ "$MODEL_SIZE" == "7B" ]]; then
                ACCEL_CONFIG_SFT_EXEC="recipes/accelerate_configs/zero2_single_node_4GPUs.yaml"
            elif [[ "$MODEL_SIZE" == "14B" ]]; then
                ACCEL_CONFIG_SFT_EXEC="recipes/accelerate_configs/zero3_single_node_4GPUs.yaml"
            elif [[ "$MODEL_SIZE" == "32B" ]]; then
                ACCEL_CONFIG_SFT_EXEC="recipes/accelerate_configs/zero3_single_node_4GPUs.yaml"
            else
                ACCEL_CONFIG_SFT_EXEC="recipes/accelerate_configs/zero2_single_node_4GPUs.yaml"
            fi
            
            CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
                --config_file "$ACCEL_CONFIG_SFT_EXEC" \
                "$TASK_SCRIPT" \
                --config "$TASK_CONFIG" \
                --model_name_or_path "$CONTAINER_INPUT_MODEL" \
                --output_dir "$CONTAINER_OUTPUT_DIR" \
                --dataset_name "$DATASET_FULL_NAME" \
                --dataset_config "" \
                --wandb_entity "$WANDB_ENTITY" \
                --wandb_project "$WANDB_PROJECT" \
                --run_name "$WANDB_RUN_NAME" \
                --seed "$SEED" \
                '"${PEFT_ARGS}"'
        else
            # Multi-node SFT (no vLLM needed, all nodes run training)
            echo ">>> Starting SFT training on $NUM_NODES nodes (all GPUs on all nodes)"
            if [[ "$MODEL_SIZE" == "14B" ]]; then
                ACCEL_CONFIG_SFT_EXEC="recipes/accelerate_configs/zero3_multi_node_14b.yaml"
                TOTAL_PROCESSES=$((4 * NUM_NODES))  # 4 GPUs per node
            elif [[ "$MODEL_SIZE" == "32B" ]]; then
                ACCEL_CONFIG_SFT_EXEC="recipes/accelerate_configs/zero3_multi_node_32b.yaml"
                TOTAL_PROCESSES=$((4 * NUM_NODES))  # 4 GPUs per node
            else
                echo "ERROR: Multi-node SFT only supported for 14B and 32B models"
                exit 1
            fi
            
            # For multi-node SFT, all nodes participate in training
            # Unlike GRPO where node 0 runs vLLM, here all nodes train
            accelerate launch \
                --config_file "$ACCEL_CONFIG_SFT_EXEC" \
                --main_process_ip "$MASTER_NODE" \
                --main_process_port 29500 \
                --num_machines "$NUM_NODES" \
                --num_processes "$TOTAL_PROCESSES" \
                --machine_rank $SLURM_PROCID \
                "$TASK_SCRIPT" \
                --config "$TASK_CONFIG" \
                --model_name_or_path "$CONTAINER_INPUT_MODEL" \
                --output_dir "$CONTAINER_OUTPUT_DIR" \
                --dataset_name "$DATASET_FULL_NAME" \
                --dataset_config "" \
                --wandb_entity "$WANDB_ENTITY" \
                --wandb_project "$WANDB_PROJECT" \
                --run_name "$WANDB_RUN_NAME" \
                --seed "$SEED" \
                '"${PEFT_ARGS}"'
        fi

    # --------------------------------------------------------------------------
    # CASE B: GRPO MODE (Note: 'else' matches the 'if' above)
    # --------------------------------------------------------------------------
    else
        # Execution path determined by NUM_NODES
        if [[ "$NUM_NODES" -eq 1 ]]; then
            echo ">>> Single Node GRPO: GPU 0=vLLM, GPUs 1-3=Training"
            
            # 1. Start vLLM in background on GPU 0
            CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
                --model "$CONTAINER_INPUT_MODEL" \
                --tensor-parallel-size 1 \
                --host localhost \
                --port 8000 &> /workspace/logs/vllm_server_'"${SLURM_JOB_ID}"'.log &
            
            sleep 60  # Wait for vLLM to start
            
            # 2. Start Training on GPUs 1-3
            GRPO_ARGS=""
            if [[ -n "$CONTAINER_RESUME_CHECKPOINT" ]]; then
                GRPO_ARGS="--resume_from_checkpoint $CONTAINER_RESUME_CHECKPOINT"
            fi
            
            CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
                --config_file recipes/accelerate_configs/zero2_single_node.yaml \
                "$TASK_SCRIPT" \
                --config "$TASK_CONFIG" \
                --model_name_or_path "$CONTAINER_INPUT_MODEL" \
                --output_dir "$CONTAINER_OUTPUT_DIR" \
                --dataset_name "$DATASET_FULL_NAME" \
                --vllm_server_host localhost \
                --vllm_server_timeout 300 \
                --wandb_entity "$WANDB_ENTITY" \
                --wandb_project "$WANDB_PROJECT" \
                --run_name "$WANDB_RUN_NAME" \
                --seed "$SEED" \
                $GRPO_ARGS \
                '"${PEFT_ARGS}"' \
                '"${LORA_GRPO_OVERRIDES}"'
        
        # MULTI NODE LOGIC
        else
            export MASTER_WORKER_IP='"${MASTER_WORKER_IP}"'
            NUM_WORKER_NODES=$((NUM_NODES - 1))
            
            if [[ $SLURM_PROCID -eq 0 ]]; then
                # Node 0: vLLM
                echo ">>> Node 0: Starting vLLM Serving"
                CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve \
                    --model "$CONTAINER_INPUT_MODEL" \
                    --tensor-parallel-size 4 \
                    --host "$MASTER_NODE" \
                    --max-model-len 4096 \
                    --port 8000 &> /workspace/logs/vllm_server_'"${SLURM_JOB_ID}"'.log
            else
                # Nodes 1+: Training
                echo ">>> Node $SLURM_PROCID: Starting GRPO training"
                
                GRPO_ARGS=""
                if [[ -n "$CONTAINER_RESUME_CHECKPOINT" ]]; then
                    GRPO_ARGS="--resume_from_checkpoint $CONTAINER_RESUME_CHECKPOINT"
                fi
                
                accelerate launch \
                    --config_file "$ACCEL_CONFIG_MULTI" \
                    --machine_rank $((SLURM_PROCID - 1)) \
                    --num_machines $NUM_WORKER_NODES \
                    --num_processes $((4 * NUM_WORKER_NODES)) \
                    --main_process_ip "$MASTER_WORKER_IP" \
                    --main_process_port 10000 \
                    "$TASK_SCRIPT" \
                    --config "$TASK_CONFIG" \
                    --model_name_or_path "$CONTAINER_INPUT_MODEL" \
                    --output_dir "$CONTAINER_OUTPUT_DIR" \
                    --dataset_name "$DATASET_FULL_NAME" \
                    --vllm_server_host "$MASTER_NODE" \
                    --vllm_server_timeout 300 \
                    --wandb_entity "$WANDB_ENTITY" \
                    --wandb_project "$WANDB_PROJECT" \
                    --run_name "$WANDB_RUN_NAME" \
                    --seed "$SEED" \
                    $GRPO_ARGS \
                    '"${PEFT_ARGS}"' \
                    '"${LORA_GRPO_OVERRIDES}"'
            fi
        fi
    fi
'
