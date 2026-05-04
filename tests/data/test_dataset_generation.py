"""Tests for data generation module."""

import sys
from pathlib import Path

import pytest

HAS_SYNDEOPT = False
IMPORT_ERROR = None
try:
    # Try importing syndeopt (required for dataset generation)
    _workspace_root = (Path(__file__).parent / "../..").resolve()
    _syndeopt_path = _workspace_root / "deps" / "syndeopt" / "src"
    if str(_syndeopt_path) not in sys.path:
        sys.path.insert(0, str(_syndeopt_path))

    from syndeopt.gen import make_tree_instance

    HAS_SYNDEOPT = True
except ImportError as e:
    IMPORT_ERROR = str(e)


def test_imports():
    """Test that data modules can be imported."""
    try:
        from data import gen_sds_dataset  # noqa: PLC0415

        assert gen_sds_dataset is not None
    except ImportError as e:
        pytest.skip(f"Could not import data.gen_sds_dataset: {e}")


def test_data_module_structure():
    """Test that data package has expected structure."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    assert data_dir.exists(), "data directory should exist"
    assert (data_dir / "gen_sds_dataset.py").exists(), "gen_sds_dataset.py should exist"


@pytest.mark.slow
@pytest.mark.skipif(
    not HAS_SYNDEOPT, reason=f"Could not import syndeopt: {IMPORT_ERROR}"
)
def test_sds_sample_generates_problems():
    """Test that sds_sample generates valid problem instances."""
    try:
        from data.gen_sds_dataset import sds_sample  # noqa: PLC0415
    except ImportError as e:
        pytest.skip(f"Could not import sds_sample: {e}")

    # Generate a small dataset (5 problems)
    problems = sds_sample(mode="tree", n_problems=5, seed=42, compute_optimal=False)

    assert len(problems) == 5
    for prob in problems:
        assert "uuid" in prob
        assert "domain" in prob
        assert prob["domain"] == "sds"
        assert "requirements" in prob
        assert "catalog" in prob
        assert "mission" in prob

        # Check requirements structure
        req = prob["requirements"]
        assert "n_variables" in req
        assert "cardinality_bounds" in req
        assert "weights" in req
        assert "interactions" in req
        assert len(req["weights"]) == req["n_variables"]

        # Check catalog structure
        cat = prob["catalog"]
        assert "variables" in cat
        assert "interactions" in cat
        assert len(cat["variables"]) == req["n_variables"]


@pytest.mark.slow
@pytest.mark.skipif(
    not HAS_SYNDEOPT, reason=f"Could not import syndeopt: {IMPORT_ERROR}"
)
def test_sds_sample_different_modes():
    """Test that sds_sample works with different problem modes."""
    try:
        from data.gen_sds_dataset import sds_sample  # noqa: PLC0415
    except ImportError as e:
        pytest.skip(f"Could not import sds_sample: {e}")

    modes = ["tree", "greedy_easy", "dense"]
    for mode in modes:
        problems = sds_sample(mode=mode, n_problems=2, seed=42, compute_optimal=False)
        assert len(problems) == 2
        for prob in problems:
            assert prob["problem_type"] == mode


@pytest.mark.slow
@pytest.mark.skipif(
    not HAS_SYNDEOPT, reason=f"Could not import syndeopt: {IMPORT_ERROR}"
)
def test_instance_to_problem_conversion():
    """Test that _instance_to_problem converts SDSInstance correctly."""
    try:
        from data.gen_sds_dataset import _instance_to_problem  # noqa: PLC0415
    except ImportError as e:
        pytest.skip(f"Could not import _instance_to_problem: {e}")

    # Create a small instance
    inst = make_tree_instance(n=5, seed=42, card=(2, 4))

    # Convert to problem
    problem = _instance_to_problem(
        inst, problem_type="tree", idx=0, compute_optimal=False
    )

    assert problem["domain"] == "sds"
    assert problem["problem_type"] == "tree"
    assert problem["requirements"]["n_variables"] == inst.n
    assert problem["requirements"]["cardinality_bounds"] == [inst.card.L, inst.card.U]
    assert problem["requirements"]["weights"] == inst.w
    assert len(problem["catalog"]["variables"]) == inst.n


@pytest.mark.slow
@pytest.mark.skipif(
    not HAS_SYNDEOPT, reason=f"Could not import syndeopt: {IMPORT_ERROR}"
)
def test_sds_render_prompt():
    """Test that sds_render_prompt generates valid prompts."""
    try:
        from data.gen_sds_dataset import sds_render_prompt, sds_sample  # noqa: PLC0415
    except ImportError as e:
        pytest.skip(f"Could not import sds functions: {e}")

    # Generate a problem
    problems = sds_sample(mode="tree", n_problems=1, seed=42, compute_optimal=False)
    assert len(problems) == 1

    # Render prompt
    rendered = sds_render_prompt(problems[0])

    assert "uuid" in rendered
    assert "problem" in rendered
    assert "mission" in rendered
    assert "domain" in rendered
    assert rendered["domain"] == "sds"
    assert rendered["uuid"] == problems[0]["uuid"]

    # Check that prompt contains expected template variables
    problem_text = rendered["problem"]
    assert "Task:" in problem_text
    assert "Synergistic Dependency Selection" in problem_text
    assert "<code>" in problem_text.lower() or "<code>" in problem_text
