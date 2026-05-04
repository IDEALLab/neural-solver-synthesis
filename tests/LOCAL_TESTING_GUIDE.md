# Local Testing Guide

This document describes the useful tests that can be run locally on a MacBook Pro (using the `llm-finetuning` conda environment) **without requiring cluster resources, GPUs, or expensive computations**.

## Overview

The heavy training and evaluation runs happen on the cluster. However, there are many critical components that can and should be tested locally to catch bugs early and ensure correctness.

## Test Categories

### ✅ 1. Aggregation Scripts (High Priority)

**Why**: These scripts process results from cluster runs. Bugs here can corrupt paper results.

**What to Test**:
- **Path Parsing**: Correctly identify methods, seeds, job IDs from file paths
- **Metadata Loading**: Priority handling of `experiment_metadata.json` vs. path inference
- **Report Set Loading**: JSON manifest parsing for batch aggregation
- **Job Selection**: Latest job selection, filtering by method/seed, specific job IDs
- **Data Loading**: CSV/JSON loading, column validation, VBS calculation
- **Table Generation**: LaTeX table formatting, statistics calculation

**Files**:
- `tests/evaluation/test_aggregate_plots.py` - SDS aggregation
- `tests/evaluation/test_bigcode_aggregate.py` - BigCode aggregation
- `tests/integration/test_aggregation_pipeline.py` - End-to-end aggregation

**Status**: ✅ 21 tests passing, some need API adjustments

### ✅ 2. Convergence Analysis (Static Code Analysis)

**Why**: Detects algorithmic patterns in generated code without executing it.

**What to Test**:
- **Hyperparameter Extraction**: Regex patterns for T, cooling_rate, n_iterations
- **Template Matching**: Hero template detection (constraint guard + Metropolis)
- **Code Structure Analysis**: Pattern detection in code strings

**Files**:
- `tests/evaluation/test_convergence_analysis.py`

**Status**: ✅ 8 tests passing

### ✅ 3. Dataset Generation Utilities

**Why**: Validate instance creation and dataset structure before expensive generation.

**What to Test**:
- **Instance Creation**: Small SDS instances (n=5-10 variables)
- **Bounds Validation**: Cardinality bounds (L, U)
- **Weight Validation**: Numeric weights, interaction structure
- **Dataset Generation**: `sds_sample()` function with small datasets
- **Problem Conversion**: `_instance_to_problem()` conversion logic
- **Prompt Rendering**: `sds_render_prompt()` template generation

**Files**:
- `tests/data/test_dataset_utils.py` - Instance creation and validation
- `tests/data/test_dataset_generation.py` - Full dataset generation pipeline

**Status**: ✅ 7 tests passing (3 instance tests + 4 generation tests)

### ✅ 4. Reward Functions

**Why**: Reward calculation logic is pure Python and critical for training.

**What to Test** (with small mock data):
- **Format Reward**: Syntax/schema validation, block detection
- **Execution Reward**: Code execution logic (with simple test cases)
- **Minimal Feasibility Reward**: Basic feasibility checking
- **Batch Handling**: Multiple completions processed correctly

**Files**:
- `tests/evaluation/test_reward_functions.py`

**Status**: ✅ 10 tests passing

**Note**: Tests use small mock SDS instances and simple code snippets. No GPU or cluster needed.

### ✅ 5. Configuration Validation

**Why**: Catch YAML errors before launching expensive cluster jobs.

**What to Test**:
- **YAML Parsing**: Valid config files load correctly
- **Schema Validation**: Required fields present, types correct
- **Field Types**: Numeric fields are numbers, lists are lists, etc.
- **Value Ranges**: Epsilon values in [0, 1], batch sizes positive
- **Reward Functions**: Registered reward function names are valid

**Files**:
- `tests/evaluation/test_config_validation.py`

**Status**: ✅ 8 tests passing

**Note**: Tests validate Hero and ablation configs without executing training.

### ✅ 6. Evaluation Utilities

**Why**: Core evaluation logic (VBS, difficulty, gap calculation) is critical and has many edge cases.

**What to Test**:
- **VBS Calculation**: Max score across methods, handling infeasible solutions
- **Difficulty Calculation**: Hardness formula, classification (Trivial/Moderate/Hard)
- **Gap Calculation**: Optimality gap formula, edge cases (negative, small VBS)
- **Mission Conversion**: Converting mission dicts to SDSInstance objects
- **Constraint Checking**: Cardinality, precedence, mutex violations
- **Score Calculation**: True score with weights and interactions

**Files**:
- `tests/evaluation/test_evaluation_utils.py` - Core evaluation utilities (VBS, difficulty, gap, constraints)
- `tests/evaluation/test_sds_evaluate.py` - PassAtKAnalyzer bootstrap and calculate_true_score

**Status**: ✅ 24 tests passing (22 utils + 2 evaluate)

**Note**: Tests use small mock instances and pure functions - no execution needed.

### ✅ 7. Code Extraction and Parsing

**Why**: Parsing LLM completions is error-prone and critical for evaluation.

**What to Test**:
- **Code Block Extraction**: Regex patterns, multiple blocks, case-insensitive
- **Answer Parsing**: "Selected: X, Y, Z" format, JSON format
- **Edge Cases**: Missing blocks, empty blocks, whitespace handling

**Files**:
- `tests/evaluation/test_code_extraction.py`

**Status**: ✅ 9 tests passing

**Note**: Pure text parsing - no execution needed.

## Running Tests

```bash
# Activate environment
conda activate llm-finetuning

# Run all tests
pytest

# Run specific category
pytest tests/evaluation/          # Aggregation tests
pytest tests/data/                 # Dataset tests
pytest tests/integration/          # Integration tests

# Skip slow tests
pytest -m "not slow"

# Run with coverage
pytest --cov=evaluation --cov=data

# Verbose output
pytest -v
```

## Current Test Status

```
✅ 91 tests passing
⏭️  1 test skipped (missing optional dependency)
```

**Breakdown**:
- Aggregation tests: 15 tests
- Convergence analysis: 8 tests
- Dataset utilities: 3 tests (instance creation, bounds, weights)
- Dataset generation: 4 tests (sds_sample, different modes, instance conversion, prompt rendering)
- Integration tests: 2 tests
- Reward functions: 10 tests
- Config validation: 8 tests
- Evaluation utilities: 22 tests (VBS, difficulty, gap, constraints, mission conversion)
- Evaluation core: 2 tests (PassAtKAnalyzer bootstrap, calculate_true_score)
- Code extraction: 9 tests
- Basic structure tests: 9 tests

## What NOT to Test Locally

These require cluster resources and should be tested on the cluster:

- ❌ **Model Training**: Requires GPUs, multi-node setup
- ❌ **Full Evaluation**: Requires GPUs, takes hours
- ❌ **Large Dataset Generation**: 10k+ instances, takes too long
- ❌ **W&B Integration**: Requires API keys, network access (but can mock)
- ❌ **Code Execution**: Full SDS simulator runs (but can test with tiny instances)

## Best Practices

1. **Use Mock Data**: Create small CSV/JSON files in `tmp_path` fixtures
2. **Test Edge Cases**: Empty files, missing columns, malformed paths
3. **Test Path Parsing**: This is critical for correct aggregation
4. **Test Metadata Priority**: Ensure `experiment_metadata.json` takes precedence
5. **Test Report Sets**: Validate batch aggregation logic

## Next Steps

1. ✅ Basic test structure created
2. ✅ All API mismatches fixed
3. ✅ Reward function tests added
4. ✅ Configuration validation tests added
5. ✅ Placeholder tests implemented (dataset generation, evaluation utilities)
6. ⚠️ Set up CI/CD to run tests automatically (optional)

## Example: Testing Path Parsing

This is one of the most critical tests because path parsing errors can cause:
- Wrong method identification
- Duplicate results
- Missing results in aggregation

```python
def test_parse_hero_path():
    path = "evaluation/sds/results/.../grpo-config_hero/seed101/job-1315163/metrics_final.csv"
    method, seed, model, job_id = parse_path_metadata(path)
    assert method == "Ours (Hero)"
    assert seed == 101
    assert job_id == 1315163
```

## Example: Testing with Mock Data

```python
def test_sds_aggregation_with_mock_data(tmp_path):
    # Create mock CSV
    csv_data = pd.DataFrame({
        'uuid': ['uuid1'],
        'feasible': [True],
        'llm_score': [100.0],
        'vbs_score': [100.0],
        # ... required columns
    })
    csv_path = tmp_path / "metrics_final.csv"
    csv_data.to_csv(csv_path, index=False)
    
    # Test loading
    df = load_all_data([str(csv_path)])
    assert len(df) == 1
```
