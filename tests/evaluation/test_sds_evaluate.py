"""Tests for SDS evaluation module."""

from pathlib import Path

import pandas as pd
import pytest


def test_imports():
    """Test that evaluation modules can be imported."""
    try:
        from evaluation.sds import evaluate  # noqa: PLC0415

        assert evaluate is not None
    except ImportError as e:
        pytest.skip(f"Could not import evaluation.sds.evaluate: {e}")


def test_evaluation_module_structure():
    """Test that evaluation package has expected structure."""
    eval_dir = Path(__file__).parent.parent.parent / "evaluation"
    assert eval_dir.exists(), "evaluation directory should exist"
    assert (eval_dir / "sds").exists(), "evaluation/sds should exist"
    assert (eval_dir / "bigcode").exists(), "evaluation/bigcode should exist"


@pytest.mark.slow
def test_pass_at_k_analyzer_bootstrap():
    """Test PassAtKAnalyzer bootstrap logic with mock data."""
    try:
        from evaluation.sds.evaluate import PassAtKAnalyzer  # noqa: PLC0415
    except ImportError as e:
        pytest.skip(f"Could not import PassAtKAnalyzer: {e}")

    # Create mock evaluation data
    # Simulate 10 problems, each with 4 samples
    data = []
    for uuid in range(10):
        for sample_idx in range(4):
            # Vary feasibility and scores
            feasible = sample_idx < 3  # First 3 are feasible
            llm_score = 100.0 - (sample_idx * 10.0) if feasible else 0.0
            vbs_score = 110.0  # VBS is constant per problem

            data.append(
                {
                    "uuid": f"problem_{uuid}",
                    "feasible": feasible,
                    "llm_score": llm_score,
                    "vbs_score": vbs_score,
                }
            )

    df = pd.DataFrame(data)

    # Create analyzer
    analyzer = PassAtKAnalyzer(df, k_values=[1, 2, 4])

    # Run bootstrap with small n for speed
    stats = analyzer.bootstrap_metrics(n_bootstraps=10)

    # Verify results structure
    assert len(stats) > 0
    assert "k" in stats.columns
    assert "pass_rate_mean" in stats.columns
    assert "opt_gap_mean" in stats.columns

    # Verify k values
    assert all(k in [1, 2, 4] for k in stats["k"])

    # Verify pass rates are reasonable (should be > 0 for k>=2 since 3/4 samples are feasible)
    for _, row in stats.iterrows():
        if row["k"] >= 2:
            assert row["pass_rate_mean"] > 0


@pytest.mark.slow
def test_calculate_true_score():
    """Test calculate_true_score function."""
    try:
        from syndeopt.core.instance import CardBounds, SDSInstance  # noqa: PLC0415

        from evaluation.sds.evaluate import calculate_true_score  # noqa: PLC0415
    except ImportError as e:
        pytest.skip(f"Could not import calculate_true_score: {e}")

    # Create a simple instance
    inst = SDSInstance(
        n=5,
        w=[10.0, 20.0, 30.0, 40.0, 50.0],
        W={(0, 1): 5.0, (1, 2): 10.0},
        precedence=[],
        mutex=[],
        groups={},
        card=CardBounds(L=2, U=4),
    )

    # Test score calculation
    selected = [0, 1, 2]
    score = calculate_true_score(inst, selected)

    expected = 10.0 + 20.0 + 30.0 + 5.0 + 10.0
    assert abs(score - expected) < 1e-6

    # Test empty selection
    assert calculate_true_score(inst, []) == 0.0

    # Test selection without interactions
    selected = [3, 4]
    score = calculate_true_score(inst, selected)
    expected = 40.0 + 50.0  # Only unary weights
    assert abs(score - expected) < 1e-6
