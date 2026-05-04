#!/bin/bash
# Generate the paper result bundles from a report set manifest
#
# Usage:
#   ./scripts/generate_paper_results.sh [report_set.json]
#
# Default: experiments/report_sets/paper_public_main_v1.json

set -e

REPORT_SET="${1:-experiments/report_sets/paper_public_main_v1.json}"

if [ ! -f "$REPORT_SET" ]; then
    echo "❌ Report set not found: $REPORT_SET"
    exit 1
fi

echo "📋 Generating paper results from: $REPORT_SET"
echo ""

# Activate conda environment
# Handle both conda init (Mac default) and conda.sh sourcing
if ! conda activate llm-finetuning 2>/dev/null; then
    # If direct activation fails, try sourcing conda.sh (for non-init setups)
    if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate llm-finetuning || {
            echo "⚠️  Failed to activate conda environment. Make sure 'llm-finetuning' environment exists."
            exit 1
        }
    else
        echo "⚠️  Failed to activate conda environment. Make sure 'llm-finetuning' environment exists."
        echo "   Try running: conda activate llm-finetuning"
        exit 1
    fi
fi

echo "📊 Step 1/4: Convergence Analysis (dynamic - fetches from W&B)..."
# Extract Hero job directories from report set (only jobs actually included in report set)
# Then run analyze_convergence.py which fetches generated code from W&B and analyzes it
HERO_JOB_DIRS=$(python -c "
import json
import os
import glob
import sys

with open('$REPORT_SET', 'r') as f:
    data = json.load(f)

# Get SDS result roots from report set (only analyze jobs actually in the report set)
sds_roots = data.get('sds', {}).get('result_roots', [])
if not sds_roots:
    sys.exit(1)

# Find all Hero job directories in these roots (only from report set, not brute search)
hero_dirs = []
for root in sds_roots:
    # Look for experiment_metadata.json files
    pattern = os.path.join(root, '**', 'experiment_metadata.json')
    for metadata_path in glob.glob(pattern, recursive=True):
        try:
            with open(metadata_path) as mf:
                metadata = json.load(mf)
            # Check if this is a Hero config (only Hero, not ablations)
            if (metadata.get('method_name') == 'Ours (Hero)' and 
                metadata.get('config_name') == 'config_hero'):
                job_dir = os.path.dirname(metadata_path)
                hero_dirs.append(job_dir)
        except:
            continue

# Print space-separated list
if hero_dirs:
    print(' '.join(hero_dirs))
    sys.exit(0)
else:
    sys.exit(1)
" 2>/dev/null)

if [ -n "$HERO_JOB_DIRS" ]; then
    echo "   Found Hero jobs in report set, running analyze_convergence.py..."
    echo "   (This fetches generated code from W&B and analyzes for Simulated Annealing template)"
    # Run analyze_convergence.py - fetches from W&B (no --skip-wandb flag)
    # This is dynamic: re-runs every time to get latest Hero code from W&B
    python evaluation/sds/analyze_convergence.py --job-dirs $HERO_JOB_DIRS || {
        echo "   ⚠️  Convergence analysis failed (check W&B access or run manually)"
        echo "   ℹ️  Aggregation will use existing convergence files if present"
    }
else
    echo "   ⚠️  No Hero job directories found in report set"
    echo "   ℹ️  Skipping convergence analysis (will aggregate existing files if present)"
fi

echo ""
echo "📊 Step 2/4: Aggregating SDS results (plots + tables)..."
python evaluation/sds/aggregate_plots.py \
    --report-set "$REPORT_SET" \
    --model-filter qwen2.5-coder-14b

echo ""
echo "📊 Step 3/4: Aggregating BigCode results (table)..."
python evaluation/bigcode/aggregate_results.py \
    --report-set "$REPORT_SET"

echo ""
echo "📊 Step 4/4: Aggregating Universal Solver Search results..."

# Extract universal solver batch ID from report set
UNIVERSAL_BATCH_ID=$(python -c "
import json
import sys
with open('$REPORT_SET', 'r') as f:
    data = json.load(f)
    location = data.get('additional_analyses', {}).get('universal_solver_search', {}).get('location', '')
    if location:
        # Extract batch ID from path: evaluation/sds/universal_search_batches/{batch_id}/
        location = location.rstrip('/')
        parts = location.split('/')
        if 'universal_search_batches' in parts:
            idx = parts.index('universal_search_batches')
            if idx + 1 < len(parts):
                print(parts[idx + 1])
                sys.exit(0)
    sys.exit(1)
" 2>/dev/null)

if [ -n "$UNIVERSAL_BATCH_ID" ]; then
    UNIVERSAL_DIR="evaluation/sds/universal_search_batches/$UNIVERSAL_BATCH_ID"
    
    if [ -d "$UNIVERSAL_DIR" ]; then
        python evaluation/sds/aggregate_universal_search.py \
            --batch-id "$UNIVERSAL_BATCH_ID"
        echo "   ✅ Universal solver aggregation complete"
    else
        echo "   ⚠️  Universal solver directory not found: $UNIVERSAL_DIR"
        echo "   ℹ️  Skipping universal solver aggregation"
    fi
else
    echo "   ⚠️  No universal solver search location found in report set"
    echo "   ℹ️  Skipping universal solver aggregation"
fi

echo ""
echo "✅ All paper results generated!"
echo ""
echo "📁 Output locations:"
REPORT_NAME=$(basename "$REPORT_SET" .json)
echo "   - SDS: evaluation/sds/aggregated_report_batches/$REPORT_NAME/"
echo "     (includes convergence_statistics.json if Hero jobs were analyzed)"
echo "   - BigCode: evaluation/bigcode/aggregated_report_batches/$REPORT_NAME/"
if [ -n "$UNIVERSAL_BATCH_ID" ] && [ -d "evaluation/sds/universal_search_batches/$UNIVERSAL_BATCH_ID/aggregated" ]; then
    echo "   - Universal Solver: evaluation/sds/universal_search_batches/$UNIVERSAL_BATCH_ID/aggregated/"
fi
