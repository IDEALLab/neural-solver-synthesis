"""Tests for dataset generation utilities."""

import sys
from pathlib import Path

import pytest

HAS_SYNDEOPT = False
IMPORT_ERROR = None
try:
    # Try importing dataset generation utilities
    _workspace_root = (Path(__file__).parent / "../..").resolve()
    _syndeopt_path = _workspace_root / "deps" / "syndeopt" / "src"
    if str(_syndeopt_path) not in sys.path:
        sys.path.insert(0, str(_syndeopt_path))

    from syndeopt.gen import make_tree_instance

    HAS_SYNDEOPT = True
except ImportError as e:
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(
    not HAS_SYNDEOPT, reason=f"Could not import syndeopt: {IMPORT_ERROR}"
)
class TestInstanceCreation:
    """Test SDS instance creation with small examples."""

    def test_create_small_tree_instance(self):
        """Test creating a small tree instance."""
        instance = make_tree_instance(n=10, seed=42)
        assert instance.n == 10
        assert len(instance.w) == 10
        assert len(instance.W) >= 0  # Tree has n-1 edges

    def test_instance_bounds(self):
        """Test instance cardinality bounds."""
        instance = make_tree_instance(n=5, seed=42, card=(2, 4))  # Explicit bounds
        assert instance.card.L >= 0
        assert instance.n >= instance.card.U
        assert instance.card.L <= instance.card.U

    def test_instance_weights(self):
        """Test that instance has valid weights."""
        instance = make_tree_instance(n=5, seed=42)
        # Weights should be numeric (w is a list)
        assert all(isinstance(w, (int, float)) for w in instance.w)
        # Interaction weights should be numeric (W is a dict)
        assert all(isinstance(w, (int, float)) for w in instance.W.values())


@pytest.mark.slow
@pytest.mark.skipif(
    not HAS_SYNDEOPT, reason=f"Could not import syndeopt: {IMPORT_ERROR}"
)
class TestDatasetGeneration:
    """Slow tests for dataset generation (can be skipped with -m 'not slow')."""

    def test_generate_small_dataset(self):
        """Test generating a very small dataset with sds_sample."""
        try:
            from data.gen_sds_dataset import sds_sample  # noqa: PLC0415
        except ImportError as e:
            pytest.skip(f"Could not import sds_sample: {e}")

        # Generate a small dataset (3 problems)
        problems = sds_sample(mode="tree", n_problems=3, seed=42, compute_optimal=False)

        assert len(problems) == 3
        assert all("uuid" in p for p in problems)
        assert all("requirements" in p for p in problems)
        assert all("catalog" in p for p in problems)

        # Verify all problems have consistent structure
        problems[0]["requirements"]["n_variables"]
        for prob in problems:
            assert prob["requirements"]["n_variables"] == len(
                prob["requirements"]["weights"]
            )
            assert (
                len(prob["catalog"]["variables"]) == prob["requirements"]["n_variables"]
            )
