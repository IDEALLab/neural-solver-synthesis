#!/bin/bash -l
#SBATCH --job-name=plot-sds
#SBATCH --account=a0105
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --output=/ritom/scratch/cscs/%u/logs/plot-%j.out
#SBATCH --error=/ritom/scratch/cscs/%u/logs/plot-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smassoudi@ethz.ch  # Update with your email

# ----------- CONFIGURATION ---------------------------------------------------
export SEED=${1:-303}

# Validate Seed
if [[ ! "$SEED" =~ ^(101|202|303)$ ]]; then
    echo "ERROR: Seed must be 101, 202, or 303."
    exit 1
fi

export OUTPUT_DIR="evaluation/sds/results_seed${SEED}"

echo "=== Plot-Only Mode ==="
echo "Seed:    $SEED"
echo "Output:  $OUTPUT_DIR"
echo "====================="

# ----------- ENV SETUP -------------------------------------------------------
export HF_HOME=/ritom/scratch/cscs/${USER}/models
export HF_DATASETS_CACHE=/ritom/scratch/cscs/${USER}/hf_datasets_cache
export EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gh200-llm-sds-training-clariden}"

# ----------- EXECUTION -------------------------------------------------------
srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -c '
    set -e # Fail on first error
    
    # Setup Environment
    REPO_ROOT="/workspace/llm-finetuning"
    
    # Install plotting dependencies
    echo "🔧 Installing plotting packages..."
    pip install seaborn scipy --quiet
    
    if [ ! -d "$REPO_ROOT" ]; then
        echo "❌ ERROR: Repo mount not found at $REPO_ROOT"
        exit 1
    fi
    
    cd "$REPO_ROOT"
    echo "📂 Working Directory: $(pwd)"
    
    # Export Variables & Paths
    export SEED='"${SEED}"'
    export OUTPUT_DIR='"${OUTPUT_DIR}"'
    export PYTHONPATH=$PYTHONPATH:$(pwd):/workspace/open-r1
    
    # Check if metrics file exists
    METRICS_FILE="$OUTPUT_DIR/metrics_final.csv"
    if [ ! -f "$METRICS_FILE" ]; then
        echo "❌ ERROR: Metrics file not found: $METRICS_FILE"
        echo "   Run full evaluation first: sbatch scripts/eval_sds_pipeline.slurm $SEED"
        exit 1
    fi
    
    echo "📊 Generating plots from existing results..."
    
    # Run plotting only
    python evaluation/sds/evaluate.py \
        --output_dir "$OUTPUT_DIR" \
        --plot-only
    
    echo "✅ Plotting Complete!"
    echo "   Plots saved in: $OUTPUT_DIR"
    echo "   - plot_pareto_variant1.png"
    echo "   - plot_pareto_variant2.png"
    echo "   - plot_pareto_variant3.png"
    echo "   - plot_perf_profile.png"
    echo "   - error_distribution.png"
'
