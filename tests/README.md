# Test Suite

This directory contains the automated test suite for the llm-finetuning project.

## Structure

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures and configuration
├── evaluation/          # Tests for evaluation package
│   ├── __init__.py
│   ├── test_sds_evaluate.py
│   └── test_bigcode_evaluate.py
├── data/                # Tests for data generation
│   ├── __init__.py
│   └── test_dataset_generation.py
└── integration/         # End-to-end integration tests
    ├── __init__.py
    └── test_full_pipeline.py
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=evaluation --cov=data

# Run specific test file
pytest tests/evaluation/test_sds_evaluate.py

# Run with verbose output
pytest -v

# Run only fast tests (skip slow/integration)
pytest -m "not slow"
```

## Writing Tests

- Test files should start with `test_`
- Test functions should start with `test_`
- Use pytest fixtures from `conftest.py` for shared setup
- Mark slow tests with `@pytest.mark.slow`
- Mark integration tests with `@pytest.mark.integration`

## Example Test

```python
import pytest
from evaluation.sds.evaluate import evaluate_model

def test_evaluate_model_basic():
    """Test basic model evaluation."""
    # Your test code here
    assert True
```
