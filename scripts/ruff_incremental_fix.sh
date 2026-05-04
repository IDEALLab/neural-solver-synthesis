#!/bin/bash
# Incremental ruff fixing - run fixes in safe batches with testing
# Usage: ./scripts/ruff_incremental_fix.sh [step]
# Steps: format, imports, safe, unsafe, all

set -e

STEP=${1:-format}

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^llm-finetuning "; then
        conda activate llm-finetuning
    fi
fi

cd "$(dirname "$0")/.."

case "$STEP" in
    format)
        echo "🎨 Step 1: Formatting code (100% safe)..."
        ruff format evaluation/ data/ tests/ scripts/
        echo "✅ Formatting complete"
        ;;
    imports)
        echo "📦 Step 2: Fixing import sorting (safe)..."
        ruff check evaluation/ data/ tests/ scripts/ --select I --fix
        echo "✅ Import sorting complete"
        ;;
    safe)
        echo "🔧 Step 3: Safe auto-fixes (review recommended)..."
        ruff check evaluation/ data/ tests/ scripts/ --fix
        echo "✅ Safe fixes complete"
        ;;
    unsafe)
        echo "⚠️  Step 4: Unsafe fixes (review carefully)..."
        ruff check evaluation/ data/ tests/ scripts/ --fix --unsafe-fixes
        echo "✅ Unsafe fixes complete"
        ;;
    all)
        echo "🚀 Running all fixes..."
        ruff format evaluation/ data/ tests/ scripts/
        ruff check evaluation/ data/ tests/ scripts/ --fix --unsafe-fixes
        echo "✅ All fixes complete"
        ;;
    *)
        echo "Usage: $0 [format|imports|safe|unsafe|all]"
        exit 1
        ;;
esac

echo ""
echo "📊 Remaining errors:"
ruff check evaluation/ data/ tests/ scripts/ --statistics 2>&1 | tail -5

echo ""
echo "🧪 Next: Run tests to verify nothing broke:"
echo "   pytest tests/ -v"
