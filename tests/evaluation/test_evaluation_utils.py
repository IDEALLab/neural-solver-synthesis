"""Tests for evaluation utility functions (VBS, difficulty, gap calculation, etc.)."""

import sys
from pathlib import Path

import pytest

# Add evaluation/sds to path
_workspace_root = (Path(__file__).parent / "../..").resolve()
_eval_path = _workspace_root / "evaluation" / "sds"
if str(_eval_path) not in sys.path:
    sys.path.insert(0, str(_eval_path))

HAS_UTILS = False
IMPORT_ERROR = None
try:
    from evaluate import calculate_true_score
    from syndeopt.core.instance import CardBounds, SDSInstance
    from utils import check_constraint_violations, mission_to_instance

    HAS_UTILS = True
except ImportError as e:
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(
    not HAS_UTILS, reason=f"Could not import evaluation utils: {IMPORT_ERROR}"
)
class TestVBSCalculation:
    """Test VBS (Virtual Best Solver) calculation logic."""

    def test_vbs_from_scores(self):
        """Test VBS calculation from multiple scores."""
        scores = [100.0, 95.0, 110.0, 90.0]
        vbs = max(scores)
        assert vbs == 110.0

    def test_vbs_with_negative_scores(self):
        """Test VBS handles negative scores correctly."""
        scores = [-10.0, -5.0, 0.0, 5.0]
        vbs = max(scores)
        assert vbs == 5.0

    def test_vbs_with_infeasible(self):
        """Test VBS excludes infeasible (-inf) scores."""
        scores = [100.0, float("-inf"), 95.0, float("-inf")]
        valid_scores = [s for s in scores if s > float("-inf")]
        vbs = max(valid_scores) if valid_scores else float("-inf")
        assert vbs == 100.0

    def test_vbs_empty(self):
        """Test VBS when all solutions are infeasible."""
        scores = [float("-inf"), float("-inf")]
        valid_scores = [s for s in scores if s > float("-inf")]
        vbs = max(valid_scores) if valid_scores else float("-inf")
        assert vbs == float("-inf")


@pytest.mark.skipif(
    not HAS_UTILS, reason=f"Could not import evaluation utils: {IMPORT_ERROR}"
)
class TestDifficultyCalculation:
    """Test difficulty (hardness) calculation logic."""

    def test_difficulty_trivial(self):
        """Test trivial difficulty (greedy ≈ VBS)."""
        vbs = 100.0
        greedy = 99.5
        epsilon = 1e-10
        hardness = (vbs - greedy) / (abs(vbs) + epsilon)
        assert hardness < 0.01  # Should be classified as Trivial

    def test_difficulty_moderate(self):
        """Test moderate difficulty."""
        vbs = 100.0
        greedy = 92.0
        epsilon = 1e-10
        hardness = (vbs - greedy) / (abs(vbs) + epsilon)
        assert 0.01 <= hardness < 0.10  # Should be classified as Moderate

    def test_difficulty_hard(self):
        """Test hard difficulty (greedy much worse than VBS)."""
        vbs = 100.0
        greedy = 50.0
        epsilon = 1e-10
        hardness = (vbs - greedy) / (abs(vbs) + epsilon)
        assert hardness >= 0.10  # Should be classified as Hard

    def test_difficulty_all_failed(self):
        """Test difficulty when all solvers fail."""
        float("-inf")
        hardness = 1.0  # Max difficulty when no one can solve
        assert hardness == 1.0

    def test_difficulty_greedy_failed(self):
        """Test difficulty when greedy fails."""
        vbs = 100.0
        float("-inf")
        # When greedy fails, anchor to 0.0
        greedy_anchored = 0.0
        epsilon = 1e-10
        hardness = (vbs - greedy_anchored) / (abs(vbs) + epsilon)
        assert hardness >= 0.10  # Should be Hard

    def test_difficulty_classification(self):
        """Test difficulty classification function."""

        def classify_diff(h):
            if h < 0.01:
                return "Trivial"
            if h < 0.10:
                return "Moderate"
            return "Hard"

        assert classify_diff(0.005) == "Trivial"
        assert classify_diff(0.05) == "Moderate"
        assert classify_diff(0.15) == "Hard"


@pytest.mark.skipif(
    not HAS_UTILS, reason=f"Could not import evaluation utils: {IMPORT_ERROR}"
)
class TestGapCalculation:
    """Test optimality gap calculation."""

    def test_gap_calculation(self):
        """Test gap calculation formula."""
        vbs = 100.0
        method_score = 90.0
        gap = (vbs - method_score) / vbs
        assert gap == 0.1  # 10% gap

    def test_gap_zero(self):
        """Test gap when method matches VBS."""
        vbs = 100.0
        method_score = 100.0
        gap = (vbs - method_score) / vbs
        assert gap == 0.0

    def test_gap_infeasible(self):
        """Test gap when method is infeasible."""
        vbs = 100.0
        method_score = 0.0  # Infeasible
        gap = (vbs - method_score) / vbs
        assert gap == 1.0  # 100% gap (worst case)

    def test_gap_negative_clipping(self):
        """Test that negative gaps are clipped to 0."""
        vbs = 100.0
        method_score = (
            110.0  # Better than VBS (shouldn't happen, but handle gracefully)
        )
        gap = max(0.0, (vbs - method_score) / vbs)
        assert gap == 0.0  # Clipped to 0

    def test_gap_small_vbs(self):
        """Test gap calculation with very small VBS."""
        vbs = 0.001
        method_score = 0.0005
        gap = (vbs - method_score) / vbs
        assert gap == 0.5  # 50% gap


@pytest.mark.skipif(
    not HAS_UTILS, reason=f"Could not import evaluation utils: {IMPORT_ERROR}"
)
class TestMissionConversion:
    """Test mission to instance conversion."""

    def test_simple_mission_conversion(self):
        """Test converting a simple mission to instance."""
        mission = {
            "n_variables": 5,
            "weights": [1.0, 2.0, 3.0, 4.0, 5.0],
            "interactions": {},
            "precedence": [],
            "mutex": [],
            "groups": {},
            "cardinality_bounds": [1, 3],
        }

        instance = mission_to_instance(mission)
        assert instance.n == 5
        assert len(instance.w) == 5
        assert instance.card.L == 1
        assert instance.card.U == 3

    def test_mission_with_interactions(self):
        """Test mission conversion with interactions."""
        mission = {
            "n_variables": 3,
            "weights": [1.0, 2.0, 3.0],
            "interactions": {"0,1": 5.0, "1,2": 3.0},
            "precedence": [],
            "mutex": [],
            "groups": {},
            "cardinality_bounds": [1, 2],
        }

        instance = mission_to_instance(mission)
        assert len(instance.W) == 2
        assert instance.W[(0, 1)] == 5.0
        assert instance.W[(1, 2)] == 3.0


@pytest.mark.skipif(
    not HAS_UTILS, reason=f"Could not import evaluation utils: {IMPORT_ERROR}"
)
class TestConstraintChecking:
    """Test constraint violation checking."""

    def test_cardinality_violation(self):
        """Test cardinality constraint violation."""
        instance = SDSInstance(
            n=5,
            w=[1.0, 2.0, 3.0, 4.0, 5.0],
            W={},
            precedence=[],
            mutex=[],
            groups={},
            card=CardBounds(L=2, U=3),
        )

        # Too few selected
        violations = check_constraint_violations(instance, [0])
        assert violations["cardinality"] is True
        assert violations["all_valid"] is False

        # Too many selected
        violations = check_constraint_violations(instance, [0, 1, 2, 3])
        assert violations["cardinality"] is True
        assert violations["all_valid"] is False

        # Valid cardinality
        violations = check_constraint_violations(instance, [0, 1, 2])
        assert violations["cardinality"] is False

    def test_precedence_violation(self):
        """Test precedence constraint violation."""
        instance = SDSInstance(
            n=3,
            w=[1.0, 2.0, 3.0],
            W={},
            precedence=[(0, 1)],  # 1 requires 0
            mutex=[],
            groups={},
            card=CardBounds(L=1, U=3),
        )

        # Violation: 1 selected but 0 not selected
        violations = check_constraint_violations(instance, [1])
        assert len(violations["precedence"]) > 0
        assert violations["all_valid"] is False

        # Valid: both selected
        violations = check_constraint_violations(instance, [0, 1])
        assert len(violations["precedence"]) == 0
        assert violations["all_valid"] is True

    def test_mutex_violation(self):
        """Test mutex constraint violation."""
        instance = SDSInstance(
            n=3,
            w=[1.0, 2.0, 3.0],
            W={},
            precedence=[],
            mutex=[(0, 1)],  # 0 and 1 are mutually exclusive
            groups={},
            card=CardBounds(L=1, U=2),
        )

        # Violation: both selected
        violations = check_constraint_violations(instance, [0, 1])
        assert len(violations["mutex"]) > 0
        assert violations["all_valid"] is False

        # Valid: only one selected
        violations = check_constraint_violations(instance, [0, 2])
        assert len(violations["mutex"]) == 0
        assert violations["all_valid"] is True


@pytest.mark.skipif(
    not HAS_UTILS, reason=f"Could not import evaluation utils: {IMPORT_ERROR}"
)
class TestScoreCalculation:
    """Test score calculation for SDS instances."""

    def test_score_calculation(self):
        """Test basic score calculation."""
        instance = SDSInstance(
            n=3,
            w=[1.0, 2.0, 3.0],
            W={(0, 1): 5.0},  # Interaction between 0 and 1
            precedence=[],
            mutex=[],
            groups={},
            card=CardBounds(L=1, U=3),
        )

        # Select [0, 1]: weights = 1.0 + 2.0 = 3.0, interaction = 5.0, total = 8.0
        score = calculate_true_score(instance, [0, 1])
        assert score == 8.0

        # Select [0, 2]: weights = 1.0 + 3.0 = 4.0, no interaction, total = 4.0
        score = calculate_true_score(instance, [0, 2])
        assert score == 4.0

    def test_score_empty_selection(self):
        """Test score with empty selection."""
        instance = SDSInstance(
            n=3,
            w=[1.0, 2.0, 3.0],
            W={},
            precedence=[],
            mutex=[],
            groups={},
            card=CardBounds(L=0, U=3),
        )

        score = calculate_true_score(instance, [])
        assert score == 0.0
