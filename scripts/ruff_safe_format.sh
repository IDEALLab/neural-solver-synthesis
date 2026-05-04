#!/bin/bash
# Safe ruff formatting - only formats code, doesn't change logic
# This is 100% safe and won't break functionality

set -e

echo "🎨 Running safe ruff formatting (format only, no logic changes)..."

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^llm-finetuning "; then
        conda activate llm-finetuning
    fi
fi

# Only format - this is 100% safe
ruff format evaluation/ data/ tests/ scripts/

echo "✅ Formatting complete!"
echo ""
echo "This only changed code formatting (whitespace, line breaks)."
echo "No logic was changed - your code functionality is preserved."
