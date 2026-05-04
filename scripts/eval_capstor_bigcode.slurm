#!/bin/bash -l
#SBATCH --job-name=eval-bigcode
#SBATCH --account=a0105
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=/ritom/scratch/cscs/%u/logs/eval-bigcode-%j.out
#SBATCH --error=/ritom/scratch/cscs/%u/logs/eval-bigcode-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smassoudi@ethz.ch  # Update with your email

# ----------- CONFIGURATION ---------------------------------------------------
# Usage: sbatch eval_capstor_bigcode.slurm <SEED> [MODEL] [TRAINING_SCHEME] [CHECKPOINT_NUM]
#   OR:  sbatch eval_capstor_bigcode.slurm --checkpoint-dir <PATH> [TRAINING_SCHEME] [SEED]
#   OR:  sbatch eval_capstor_bigcode.slurm --base-model <HF_ID> [SEED]
# Examples:
#   sbatch eval_capstor_bigcode.slurm 303                                    # 7B, GRPO, latest checkpoint
#   sbatch eval_capstor_bigcode.slurm 303 qwen2.5-coder-14b grpo 40         # 14B, GRPO, checkpoint-40
#   sbatch eval_capstor_bigcode.slurm --checkpoint-dir /path/to/Qwen2.5-Coder-14B-Instruct-GRPO-SDS-OPT-seed303/checkpoint-40 grpo 303
#   sbatch eval_capstor_bigcode.slurm --base-model Qwen/Qwen2.5-Coder-14B-Instruct 303  # Base model from HF

# Fixed parameters for Qwen technical report comparison (greedy decoding)
export TEMPERATURE="0"        # Greedy decoding (matches Qwen report)
export N_SAMPLES="1"          # Pass@1 evaluation (matches Qwen report)
export TASKS="humaneval,mbpp" # Tasks from Qwen technical report

# Check if first arg is --checkpoint-dir (explicit path mode) or --base-model (HF identifier)
if [[ "$1" == "--checkpoint-dir" ]]; then
    export CHECKPOINT_DIR="$2"
    export TRAINING_SCHEME=${3:-"grpo"}  # Default to GRPO
    export SEED=${4:-303}                 # Default to 303, extract from path if possible
    
    # Extract seed from path if not provided
    if [[ -z "$4" ]]; then
        if [[ "$CHECKPOINT_DIR" =~ seed([0-9]+) ]]; then
            export SEED="${BASH_REMATCH[1]}"
        fi
    fi
    
    # Extract job ID from path (e.g., /path/to/job-1291239/checkpoint-80 -> 1291239)
    export JOB_ID=""
    if [[ "$CHECKPOINT_DIR" =~ /job-([0-9]+)/ ]]; then
        export JOB_ID="${BASH_REMATCH[1]}"
    elif [[ "$CHECKPOINT_DIR" =~ /job-([0-9]+)$ ]]; then
        export JOB_ID="${BASH_REMATCH[1]}"
    fi
    
    # Extract ablation tag from base directory name (e.g., ...-config_ablation_no_gen/job-...)
    export ABLATION_TAG=""
    export WANDB_ABLATION_TAG=""
    BASE_DIR=$(dirname "$CHECKPOINT_DIR")
    # If checkpoint is inside a job directory, get the parent (base) directory
    if [[ "$BASE_DIR" =~ /job- ]]; then
        BASE_DIR=$(dirname "$BASE_DIR")
    fi
    BASE_DIR_NAME=$(basename "$BASE_DIR")
    # Check for ablation tags: -config_ablation_no_gen, -config_ablation_no_oracle, -config_discovery, -config_hero, -config_minimalist, etc.
    if [[ "$BASE_DIR_NAME" =~ -(config_(?:ablation_[a-z_]+|discovery|hero|minimalist))$ ]]; then
        export ABLATION_TAG="${BASH_REMATCH[1]}"
        # Convert to readable abbreviation for W&B
        CONFIG_NAME="${BASH_REMATCH[1]}"
        # Remove "config_" prefix if present
        name="${CONFIG_NAME#config_}"
        # Map to readable abbreviations
        case "$name" in
            "hero")
                WANDB_ABLATION_TAG="hero"
                ;;
            "discovery")
                WANDB_ABLATION_TAG="discovery"
                ;;
            "ablation_no_oracle")
                WANDB_ABLATION_TAG="no-oracle"
                ;;
            "ablation_oracle")
                WANDB_ABLATION_TAG="oracle"
                ;;
            "ablation_no_gen")
                WANDB_ABLATION_TAG="no-gen"
                ;;
            "ablation_generalization")
                WANDB_ABLATION_TAG="gen"
                ;;
            "ablation_no_diversity")
                WANDB_ABLATION_TAG="no-div"
                ;;
            "ablation_diversity")
                WANDB_ABLATION_TAG="div"
                ;;
            "ablation_neutral_prompt")
                WANDB_ABLATION_TAG="neutral-prompt"
                ;;
            "ablation_prompt")
                WANDB_ABLATION_TAG="prompt"
                ;;
            "minimalist")
                WANDB_ABLATION_TAG="minimalist"
                ;;
            *)
                # Fallback: remove "ablation_" prefix and replace underscores with hyphens
                WANDB_ABLATION_TAG=$(echo "$name" | sed 's/^ablation_//' | tr '_' '-')
                ;;
        esac
        export WANDB_ABLATION_TAG
        # Store original training scheme for output directory
        export TRAINING_SCHEME_ORIGINAL="$TRAINING_SCHEME"
        # Create modified training scheme for output directory only (use full tag for directory)
        export TRAINING_SCHEME_OUTPUT="${TRAINING_SCHEME}-${ABLATION_TAG}"
    else
        # No ablation tag, use same scheme for both
        export TRAINING_SCHEME_ORIGINAL="$TRAINING_SCHEME"
        export TRAINING_SCHEME_OUTPUT="$TRAINING_SCHEME"
    fi

    # Guardrail: prevent accidental evaluation of legacy Generalization ablation by default.
    # To explicitly re-enable it, submit with: sbatch --export=INCLUDE_GENERALIZATION=1 ...
    INCLUDE_GENERALIZATION="${INCLUDE_GENERALIZATION:-0}"
    if [[ "$ABLATION_TAG" == "config_ablation_generalization" ]] && [[ "$INCLUDE_GENERALIZATION" != "1" ]]; then
        echo "⚠️  WARNING: Refusing to evaluate legacy Generalization ablation by default."
        echo "    Detected tag: $ABLATION_TAG"
        echo "    To override: sbatch --export=INCLUDE_GENERALIZATION=1 ..."
        echo "    Exiting without running BigCode evaluation."
        exit 0
    fi
    
    # Extract model name from path for output directory
    # Try to infer from checkpoint directory name (e.g., "Qwen2.5-Coder-14B-Instruct" -> "qwen2.5-coder-14b")
    CHECKPOINT_BASE=$(basename "$BASE_DIR")
    if [[ "$CHECKPOINT_BASE" =~ Qwen2\.5-Coder-([0-9.]+B)-Instruct ]]; then
        SIZE="${BASH_REMATCH[1]}"  # e.g., "14B" or "7B"
        # Convert "14B" -> "14b", "7B" -> "7b"
        SIZE_LOWER=$(echo "$SIZE" | tr '[:upper:]' '[:lower:]')
        export MODEL_NAME="qwen2.5-coder-${SIZE_LOWER}"
    else
        # Fallback: try to extract from full path
        if [[ "$CHECKPOINT_DIR" =~ Qwen2\.5-Coder-([0-9.]+B) ]]; then
            SIZE="${BASH_REMATCH[1]}"
            SIZE_LOWER=$(echo "$SIZE" | tr '[:upper:]' '[:lower:]')
            export MODEL_NAME="qwen2.5-coder-${SIZE_LOWER}"
        else
            export MODEL_NAME="qwen2.5-coder-7b"  # Default fallback
        fi
    fi
    
    # Use the checkpoint directory directly
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "❌ CRITICAL ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
        exit 1
    fi
    
    export MODEL_PATH="$CHECKPOINT_DIR"
    echo "=== Explicit Checkpoint Path Mode ==="
    echo "Checkpoint:    $CHECKPOINT_DIR"
    echo "Training:      $TRAINING_SCHEME"
    echo "Seed:          $SEED"
    echo "Job ID:        ${JOB_ID:-(none)}"
    echo "Ablation Tag:  ${ABLATION_TAG:-(none)}"
    echo "Model (inferred): $MODEL_NAME"
    echo ""
    
    # Skip to output directory construction
    SKIP_DISCOVERY=true
    IS_HF_MODEL=false
elif [[ "$1" == "--base-model" ]]; then
    # Base model mode: HuggingFace identifier
    export HF_MODEL_ID="$2"  # e.g., "Qwen/Qwen2.5-Coder-14B-Instruct"
    export TRAINING_SCHEME="base"  # Always "base" for base models
    # Get seed from 3rd argument, default to 303 if not provided
    if [ -n "$3" ]; then
        export SEED="$3"
    else
        export SEED=303
    fi
    
    # Extract model name from HF identifier (e.g., "Qwen/Qwen2.5-Coder-7B-Instruct" -> "qwen2.5-coder-7b")
    if [[ "$HF_MODEL_ID" =~ Qwen2\.5-Coder-([0-9.]+B)-Instruct ]]; then
        SIZE="${BASH_REMATCH[1]}"  # e.g., "7B" or "14B"
        SIZE_LOWER=$(echo "$SIZE" | tr '[:upper:]' '[:lower:]')
        export MODEL_NAME="qwen2.5-coder-${SIZE_LOWER}"
    else
        export MODEL_NAME="qwen2.5-coder-7b"  # Default fallback
    fi
    
    export MODEL_PATH="$HF_MODEL_ID"
    export JOB_ID=""  # No job ID for base models
    export TRAINING_SCHEME_ORIGINAL="$TRAINING_SCHEME"  # No ablation for base models
    export TRAINING_SCHEME_OUTPUT="$TRAINING_SCHEME"
    echo "=== Base Model (HuggingFace) Mode ==="
    echo "HF Model ID:   $HF_MODEL_ID"
    echo "Training:      base (untrained)"
    echo "Seed:          $SEED"
    echo "Model (inferred): $MODEL_NAME"
    echo ""
    
    SKIP_DISCOVERY=true
    export IS_HF_MODEL=true
else
    # Original mode: discover from model/training/seed
    export SEED=${1:-101}
    export MODEL_NAME=${2:-"qwen2.5-coder-7b"}  # Default to 7B
    export TRAINING_SCHEME=${3:-"grpo"}          # Default to GRPO only
    export CHECKPOINT_NUM=${4:-""}               # Optional: specific checkpoint (e.g., "40"), empty = latest
    export JOB_ID=""                             # No job ID in discovery mode
    export TRAINING_SCHEME_ORIGINAL="$TRAINING_SCHEME"  # No ablation in discovery mode
    export TRAINING_SCHEME_OUTPUT="$TRAINING_SCHEME"
    SKIP_DISCOVERY=false
    export IS_HF_MODEL=false
fi

# Initialize JOB_ID if not set
if [ -z "$JOB_ID" ]; then
    export JOB_ID=""
fi

# Initialize TRAINING_SCHEME_ORIGINAL if not set (fallback)
if [ -z "$TRAINING_SCHEME_ORIGINAL" ]; then
    export TRAINING_SCHEME_ORIGINAL="$TRAINING_SCHEME"
fi
if [ -z "$TRAINING_SCHEME_OUTPUT" ]; then
    export TRAINING_SCHEME_OUTPUT="$TRAINING_SCHEME"
fi

# Validate Seed
if [ -z "$SEED" ]; then
    echo "ERROR: Seed not set. Please provide a seed value."
    exit 1
fi
if [[ ! "$SEED" =~ ^(101|202|303)$ ]]; then
    echo "ERROR: Seed must be 101, 202, or 303. Got: '$SEED'"
    exit 1
fi

# Skip discovery if explicit checkpoint path was provided
if [[ "$SKIP_DISCOVERY" == "true" ]]; then
    # Already set MODEL_PATH above, skip to output directory construction
    :
else
    # Normalize model name for checkpoint directory
    # Convert "qwen2.5-coder-7b" -> "Qwen2.5-Coder-7B-Instruct"
    # Convert "qwen2.5-coder-14b" -> "Qwen2.5-Coder-14B-Instruct"
    normalize_model() {
        local model=$1
        local parts=($(echo "$model" | tr '-' ' '))
        if [[ "${parts[0]}" == "qwen" && "${parts[1]}" == "2.5" ]]; then
            local size=$(echo "${parts[2]}" | tr '[:lower:]' '[:upper:]')  # 7b -> 7B, 14b -> 14B
            echo "Qwen2.5-Coder-${size}-Instruct"
        else
            echo "$model"
        fi
    }

    # Map training scheme to checkpoint suffix
    # GRPO only -> "GRPO-SDS-OPT", SFT+GRPO -> "SFT-GRPO", SFT only -> "SFT"
    map_training_scheme() {
        local scheme=$1
        case "$scheme" in
            "grpo")
                echo "GRPO-SDS-OPT"
                ;;
            "sft-grpo"|"sft_grpo")
                echo "SFT-GRPO"
                ;;
            "sft")
                echo "SFT"
                ;;
            *)
                echo "GRPO-SDS-OPT"  # Default fallback
                ;;
        esac
    }

    NORMALIZED_MODEL=$(normalize_model "$MODEL_NAME")
    SCHEME_SUFFIX=$(map_training_scheme "$TRAINING_SCHEME")
    BASE_MODEL_DIR="/ritom/scratch/cscs/${USER}/checkpoints/${NORMALIZED_MODEL}-${SCHEME_SUFFIX}-seed${SEED}"

    echo "=== Model Discovery ==="
    echo "Model:        $MODEL_NAME -> $NORMALIZED_MODEL"
    echo "Training:     $TRAINING_SCHEME -> $SCHEME_SUFFIX"
    echo "Seed:         $SEED"
    echo "Base Dir:     $BASE_MODEL_DIR"

    if [ ! -d "$BASE_MODEL_DIR" ]; then
        echo "❌ CRITICAL ERROR: Base directory not found: $BASE_MODEL_DIR"
        echo "   Please check that the model was trained with this configuration."
        exit 1
    fi

    # Find checkpoint (specific or latest)
    if [ -n "$CHECKPOINT_NUM" ]; then
        # Use specific checkpoint
        SPECIFIC_CHECKPOINT="$BASE_MODEL_DIR/checkpoint-${CHECKPOINT_NUM}"
        if [ -d "$SPECIFIC_CHECKPOINT" ]; then
            export MODEL_PATH="$SPECIFIC_CHECKPOINT"
            echo "✅ Using specified checkpoint: checkpoint-${CHECKPOINT_NUM}"
        else
            echo "❌ CRITICAL ERROR: Checkpoint checkpoint-${CHECKPOINT_NUM} not found in $BASE_MODEL_DIR"
            echo "   Available checkpoints:"
            ls -d "$BASE_MODEL_DIR"/checkpoint-* 2>/dev/null | head -5 || echo "   (none found)"
            exit 1
        fi
    else
        # Find the latest checkpoint
        LATEST_CHECKPOINT=$(find "$BASE_MODEL_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
        
        if [ -z "$LATEST_CHECKPOINT" ]; then
            if [ -f "$BASE_MODEL_DIR/config.json" ]; then
                export MODEL_PATH="$BASE_MODEL_DIR"
                echo "✅ Using base directory (no checkpoint-* found)"
            else
                echo "❌ CRITICAL ERROR: No checkpoints found in $BASE_MODEL_DIR"
                exit 1
            fi
        else
            export MODEL_PATH="$LATEST_CHECKPOINT"
            CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT")
            echo "✅ Using latest checkpoint: $CHECKPOINT_NAME"
        fi
    fi
fi

# Construct hierarchical output directory
# Normalize model name for directory (lowercase, keep structure)
OUTPUT_MODEL=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
# Use TRAINING_SCHEME_OUTPUT if available (includes ablation tag), otherwise use TRAINING_SCHEME
TRAINING_FOR_OUTPUT="${TRAINING_SCHEME_OUTPUT:-$TRAINING_SCHEME}"
OUTPUT_TRAINING=$(echo "$TRAINING_FOR_OUTPUT" | tr '_' '-')
# Batch-aware output root (backward compatible)
RESULTS_ROOT="evaluation/bigcode/results"
if [ -n "${BATCH_ID:-}" ]; then
    RESULTS_ROOT="evaluation/bigcode/results_batches/${BATCH_ID}"
fi
# Include job-id in path if available
if [ -n "$JOB_ID" ]; then
    export OUTPUT_DIR="${RESULTS_ROOT}/${OUTPUT_MODEL}/${OUTPUT_TRAINING}/seed${SEED}/job-${JOB_ID}"
else
    export OUTPUT_DIR="${RESULTS_ROOT}/${OUTPUT_MODEL}/${OUTPUT_TRAINING}/seed${SEED}"
fi

echo ""
echo "=== Starting BigCode Evaluation Pipeline ==="
echo "Seed:           $SEED"
echo "Model:          $MODEL_NAME"
echo "Training:       ${TRAINING_SCHEME_ORIGINAL:-$TRAINING_SCHEME}"
echo "Ablation:       ${ABLATION_TAG:-(none)}"
echo "Checkpoint:     $MODEL_PATH"
echo "Tasks:          $TASKS"
echo "Output:         $OUTPUT_DIR"
echo "Temperature:    $TEMPERATURE (greedy)"
echo "N Samples:      $N_SAMPLES (Pass@1)"
echo "==========================================="
echo ""

# ----------- ENV SETUP -------------------------------------------------------
export HF_HOME=/ritom/scratch/cscs/${USER}/models
export HF_DATASETS_CACHE=/ritom/scratch/cscs/${USER}/hf_datasets_cache
export EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gh200-llm-sds-training-daints}"
export HF_TOKEN=$(< $HOME/llm/hf_token.txt)
export HF_ALLOW_CODE_EVAL="1"  # Required for code execution
export WANDB_API_KEY=$(< $HOME/llm/wandb_token.txt)

# ----------- EXECUTION -------------------------------------------------------
srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -c '
    set -e # Fail on first error
    
    # 0. Setup Environment
    REPO_ROOT="/workspace/llm-finetuning"
    
    if [ ! -d "$REPO_ROOT" ]; then
        echo "❌ ERROR: Repo mount not found at $REPO_ROOT"
        ls -la /workspace
        exit 1
    fi
    
    cd "$REPO_ROOT"
    echo "📂 Working Directory: $(pwd)"
    
    # Path Translation Logic (skip for HF identifiers)
    if [[ "$IS_HF_MODEL" == "true" ]]; then
        # HuggingFace identifier - use as-is (harness will download automatically)
        CONTAINER_MODEL_PATH="$MODEL_PATH"
        echo "HF Model ID:   $MODEL_PATH"
        echo "Harness will download from HuggingFace Hub automatically"
    else
        # Local checkpoint path - translate for container
        CONTAINER_MODEL_PATH="${MODEL_PATH/\/ritom\/scratch\/cscs\/$USER/\/workspace}"
        CONTAINER_MODEL_PATH="${CONTAINER_MODEL_PATH/\/capstor\/scratch\/cscs\/$USER/\/workspace}"
        CONTAINER_MODEL_PATH="${CONTAINER_MODEL_PATH/\/iopsstor\/scratch\/cscs\/$USER/\/workspace}"
        echo "Physical Path: $MODEL_PATH"
        echo "Container Path: $CONTAINER_MODEL_PATH"
        
        if [ ! -d "$CONTAINER_MODEL_PATH" ]; then
            echo "❌ ERROR: Model not found inside container at $CONTAINER_MODEL_PATH"
            echo "   Listing /workspace/checkpoints:"
            ls -d /workspace/checkpoints/* 2>/dev/null | head -5 || echo "   (directory empty or not accessible)"
            exit 1
        fi
    fi
    
    # 1. Export Variables & Paths (passed from outer shell)
    export SEED=${SEED}
    export MODEL_NAME=${MODEL_NAME}
    export TRAINING_SCHEME=${TRAINING_SCHEME_ORIGINAL:-${TRAINING_SCHEME}}
    export TRAINING_SCHEME_ORIGINAL=${TRAINING_SCHEME_ORIGINAL:-${TRAINING_SCHEME}}
    export WANDB_ABLATION_TAG=${WANDB_ABLATION_TAG:-}
    export MODEL_PATH=${MODEL_PATH}
    export JOB_ID=${JOB_ID}
    export OUTPUT_DIR=${OUTPUT_DIR}
    export TASKS=${TASKS}
    export TEMPERATURE=${TEMPERATURE}
    export N_SAMPLES=${N_SAMPLES}
    export IS_HF_MODEL=${IS_HF_MODEL}
    export CONTAINER_MODEL_PATH=${CONTAINER_MODEL_PATH}
    export PYTHONPATH=$PYTHONPATH:$(pwd):/workspace/open-r1
    export BATCH_ID=${BATCH_ID:-}
    
    # --- Install BigCode Harness and dependencies ---
    echo "🔧 Installing BigCode evaluation harness..."
    
    # Install harness with dependencies, but skip pyext (incompatible with Python 3.11+)
    # pyext is only needed for some multilingual tasks, not for HumanEval/MBPP
    echo "🔧 Installing harness dependencies (skipping pyext)..."
    pip install --no-cache-dir \
        evaluate>=0.3.0 \
        scikit-learn \
        appdirs \
        mosestokenizer==1.0.0 \
        huggingface_hub>=0.11.1 \
        fsspec>=2023.12.2 || echo "⚠️  Some optional deps failed, continuing..."
    
    # Install harness itself (editable, no deps to avoid conflicts)
    pip install --no-deps -e deps/bigcode-evaluation-harness
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Split tasks for wrapper script
    IFS="," read -ra TASK_ARRAY <<< "$TASKS"
    
    # Call wrapper script (handles harness execution, metadata, and W&B logging)
    python evaluation/bigcode/evaluate.py \
        --model "$CONTAINER_MODEL_PATH" \
        --tasks "${TASK_ARRAY[@]}" \
        --temperature "$TEMPERATURE" \
        --n_samples "$N_SAMPLES" \
        --max_length_generation 2048 \
        --batch_size 1 \
        --precision bf16 \
        --allow_code_execution \
        --save_generations \
        --output_dir "$OUTPUT_DIR" \
        --seed "$SEED" \
        --job-id "$JOB_ID" \
        --training-scheme "$TRAINING_SCHEME_OUTPUT" \
        --log-to-wandb
'
