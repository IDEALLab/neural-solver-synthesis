"""Tests for convergence analysis (static code analysis)."""

import pytest

HAS_CONVERGENCE = False
IMPORT_ERROR = None
try:
    from evaluation.sds.analyze_convergence import (
        analyze_code_structure,
        check_hero_template,
    )

    HAS_CONVERGENCE = True
except ImportError as e:
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(
    not HAS_CONVERGENCE, reason=f"Could not import analyze_convergence: {IMPORT_ERROR}"
)
class TestHyperparameterExtraction:
    """Test hyperparameter extraction from code."""

    def test_extract_temperature(self):
        """Test extracting temperature from code."""
        code = """
        T = 1000
        cooling_rate = 0.99
        for i in range(1000):
            T *= cooling_rate
        """
        meta = analyze_code_structure(code)
        assert meta["T"] == 1000
        assert meta["cooling"] == 0.99

    def test_extract_iterations(self):
        """Test extracting iteration count."""
        code = """
        n_iterations = 5000
        for i in range(n_iterations):
            pass
        """
        meta = analyze_code_structure(code)
        assert meta["iters"] == 5000

    def test_extract_cooling_rate(self):
        """Test extracting cooling rate."""
        code = """
        cooling_rate = 0.95
        T = 2000
        while T > 0.1:
            T *= cooling_rate
        """
        meta = analyze_code_structure(code)
        assert meta["cooling"] == 0.95
        assert meta["T"] == 2000


@pytest.mark.skipif(
    not HAS_CONVERGENCE, reason=f"Could not import analyze_convergence: {IMPORT_ERROR}"
)
class TestTemplateMatching:
    """Test Hero template matching."""

    def test_valid_hero_template(self):
        """Test that valid Hero template is recognized."""
        meta = {
            "has_constraint_guard": True,
            "has_metropolis": True,
            "T": 1000,
            "cooling": 0.99,
            "iters": 1000,
        }
        assert check_hero_template(meta) is True

    def test_missing_constraint_guard(self):
        """Test that missing constraint guard fails."""
        meta = {
            "has_constraint_guard": False,
            "has_metropolis": True,
            "T": 1000,
            "cooling": 0.99,
            "iters": 1000,
        }
        assert check_hero_template(meta) is False

    def test_missing_metropolis(self):
        """Test that missing Metropolis criterion fails."""
        meta = {
            "has_constraint_guard": True,
            "has_metropolis": False,
            "T": 1000,
            "cooling": 0.99,
            "iters": 1000,
        }
        assert check_hero_template(meta) is False

    def test_invalid_temperature(self):
        """Test that too-low temperature fails."""
        meta = {
            "has_constraint_guard": True,
            "has_metropolis": True,
            "T": 50,  # Too low
            "cooling": 0.99,
            "iters": 1000,
        }
        assert check_hero_template(meta) is False

    def test_invalid_cooling(self):
        """Test that invalid cooling rate fails."""
        meta = {
            "has_constraint_guard": True,
            "has_metropolis": True,
            "T": 1000,
            "cooling": 0.5,  # Too low
            "iters": 1000,
        }
        assert check_hero_template(meta) is False
