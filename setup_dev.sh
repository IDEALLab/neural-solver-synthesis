#!/bin/bash
# Setup script for development environment (without LLM training dependencies)
# This installs only what's needed for syndeopt and SDS simulator testing
# Uses conda environment similar to syndeopt

set -e  # Exit on any error

echo "🚀 Setting up llm-finetuning development environment..."
echo "   (without LLM training dependencies - PyTorch, vLLM, etc.)"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install miniforge first:"
    echo "   https://github.com/conda-forge/miniforge"
    exit 1
fi

echo "✅ Conda found"

# Check if environment already exists
if conda env list | grep -q "^llm-finetuning "; then
    echo "⚠️  Environment 'llm-finetuning' already exists."
    read -p "Do you want to remove it and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n llm-finetuning -y
    else
        echo "📦 Using existing environment..."
        eval "$(conda shell.bash hook)"
        conda activate llm-finetuning
        echo "🔄 Updating environment..."
        conda env update -f environment.yml --prune --solver=classic
    fi
else
    echo "🫙 Creating conda environment..."
    conda env create -f environment.yml --solver=classic
fi

echo "🔄 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate llm-finetuning

# Install Python dependencies from pyproject.toml
echo "📦 Installing Python dependencies from pyproject.toml..."
pip install -e .

# Install syndeopt as editable
echo "📦 Installing syndeopt (editable)..."
cd deps/syndeopt
pip install -e .
cd ../..

# Install open-r1 as editable (but skip LLM training deps)
echo "📦 Installing open-r1 (editable, minimal deps)..."
cd deps/open-r1
# Install only the package structure, not all dependencies
pip install -e . --no-deps
cd ../..

# Install the lightweight open-r1 parsing/eval dependencies that are required
# by the public reward-function validation path, without pulling the full TRL /
# training stack.
echo "📦 Installing lightweight open-r1 reward-test dependencies..."
pip install "latex2sympy2_extended>=1.0.6" "math-verify==0.5.2"

# Install ShinkaEvolve as editable
echo "📦 Installing ShinkaEvolve (editable)..."
cd deps/ShinkaEvolve
pip install -e .
cd ../..

# Install bigcode-evaluation-harness as editable
echo "📦 Installing bigcode-evaluation-harness (editable)..."
cd deps/bigcode-evaluation-harness
pip install -e . --no-deps
cd ../..

# Parse arguments
INSTALL_DEV=false

for arg in "$@"; do
    case "$arg" in
        --dev)
            INSTALL_DEV=true
            ;;
    esac
done

# Install dev dependencies if requested
if [ "$INSTALL_DEV" = true ]; then
    echo "📦 Installing dev dependencies..."
    pip install -e .[dev]
    echo "🪝 Installing pre-commit hooks..."
    pre-commit install || echo "⚠️  pre-commit not installed, skipping hooks"
fi

echo ""
echo "🧪 Testing the setup..."
python -c "from syndeopt.core.instance import SDSInstance; print('✅ syndeopt: OK')" || echo "❌ syndeopt import failed"
python -c "from open_r1.simulators import SDSSimulator; print('✅ simulators: OK')" || echo "❌ simulators import failed"
python -c "from open_r1.rewards_unified_v2 import unified_soft_nominal_reward; print('✅ open-r1 rewards: OK')" || echo "❌ open-r1 rewards import failed"
python -c "from data.gen_sds_dataset import sds_sample; print('✅ datasets: OK')" || echo "❌ datasets import failed"
python -c "import shinka; print('✅ ShinkaEvolve: OK')" || echo "❌ ShinkaEvolve import failed"
python -c "import bigcode_eval; print('✅ bigcode-evaluation-harness: OK')" || echo "❌ bigcode-evaluation-harness import failed"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate llm-finetuning"
echo "2. Run tests to verify setup: pytest tests/ (optional)"
echo ""
echo "Useful commands:"
echo "  conda activate llm-finetuning  # Activate environment"
echo "  python -c 'from syndeopt.core.instance import SDSInstance'  # Test syndeopt"
echo "  python -c 'from open_r1.simulators import SDSSimulator'  # Test simulators"
