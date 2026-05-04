"""Tests for reward functions used in GRPO training."""

import sys
from pathlib import Path

import pytest

# Add open-r1 to path
_workspace_root = (Path(__file__).parent / "../..").resolve()
_open_r1_path = _workspace_root / "deps" / "open-r1" / "src"
if str(_open_r1_path) not in sys.path:
    sys.path.insert(0, str(_open_r1_path))

HAS_REWARDS = False
IMPORT_ERROR = None
try:
    from open_r1.rewards_unified_v2 import (
        minimal_feasibility_reward,
        unified_code_execution_reward_no_oracle,
        unified_format_reward,
        unified_nominal_reward_topk_interaction_bound,
        unified_soft_nominal_reward,
    )
    from open_r1.rewards import get_reward_funcs

    HAS_REWARDS = True
except ImportError as e:
    IMPORT_ERROR = str(e)


# Sample SDS completion with all required blocks
SDS_COMPLETION_VALID = """<think>
I need to select variables that maximize the score while respecting constraints.
</think>
<code>
import json
import sys

def select_components(requirements, catalog):
    # Simple greedy selection
    n = requirements["n_variables"]
    weights = catalog["variables"]
    selected = []
    for i in range(min(requirements["cardinality_bounds"][1], n)):
        if i < len(weights):
            selected.append(i)
    return {"variables": selected}

if __name__ == "__main__":
    data = json.load(sys.stdin)
    result = select_components(data["requirements"], data["catalog"])
    print(json.dumps({"selection": result}))
</code>
<answer>
Selected: 0, 1, 2
</answer>
"""

SDS_COMPLETION_INVALID = "No blocks here"

# Sample SDS requirements (minimal valid structure)
SDS_REQUIREMENTS = {
    "n_variables": 5,
    "cardinality_bounds": [1, 3],
    "precedence": [],
    "mutex": [],
    "groups": {},
    "weights": [1.0, 2.0, 3.0, 4.0, 5.0],
    "interactions": {},
}


@pytest.mark.skipif(
    not HAS_REWARDS, reason=f"Could not import reward functions: {IMPORT_ERROR}"
)
class TestFormatReward:
    """Test format reward function."""

    def test_valid_completion(self):
        """Test that valid completion gets full reward."""
        completions = [{"content": SDS_COMPLETION_VALID}]
        rewards = unified_format_reward(completions)
        assert len(rewards) == 1
        assert rewards[0] > 0.0, "Valid completion should get positive reward"

    def test_invalid_completion(self):
        """Test that invalid completion gets zero reward."""
        completions = [{"content": SDS_COMPLETION_INVALID}]
        rewards = unified_format_reward(completions)
        assert len(rewards) == 1
        assert rewards[0] == 0.0, "Invalid completion should get 0.0"

    def test_empty_completion(self):
        """Test that empty completion gets zero reward."""
        completions = [{"content": ""}]
        rewards = unified_format_reward(completions)
        assert len(rewards) == 1
        assert rewards[0] == 0.0, "Empty completion should get 0.0"

    def test_missing_blocks(self):
        """Test that completion missing required blocks gets reduced reward."""
        completion_no_code = "<think>Some reasoning</think>"
        completions = [{"content": completion_no_code}]
        rewards = unified_format_reward(completions)
        assert len(rewards) == 1
        assert rewards[0] < 1.0, "Missing code block should get reduced reward"


@pytest.mark.skipif(
    not HAS_REWARDS, reason=f"Could not import reward functions: {IMPORT_ERROR}"
)
class TestExecutionReward:
    """Test code execution reward function."""

    def test_execution_reward_basic(self):
        """Test that execution reward returns valid values."""
        completions = [{"content": SDS_COMPLETION_VALID}]
        rewards = unified_code_execution_reward_no_oracle(
            completions, domain="sds", mission=SDS_REQUIREMENTS
        )
        assert len(rewards) == 1
        assert 0.0 <= rewards[0] <= 1.0, "Reward should be in [0, 1]"

    def test_execution_reward_invalid_code(self):
        """Test that invalid code gets lower reward."""
        invalid_code = """<code>
def broken():
    return {invalid json}
</code>"""
        completions = [{"content": invalid_code}]
        rewards = unified_code_execution_reward_no_oracle(
            completions, domain="sds", mission=SDS_REQUIREMENTS
        )
        assert len(rewards) == 1
        assert rewards[0] < 1.0, "Invalid code should get reduced reward"

    def test_soft_nominal_reward_basic(self):
        """Test that the soft-gate nominal reward executes and stays bounded."""
        completions = [{"content": SDS_COMPLETION_VALID}]
        rewards = unified_soft_nominal_reward(
            completions, domain="sds", mission=SDS_REQUIREMENTS
        )
        assert len(rewards) == 1
        assert 0.0 <= rewards[0] <= 1.0, "Reward should be in [0, 1]"

    def test_topk_nominal_reward_basic(self):
        """Test that the normalization-ablation nominal reward executes and stays bounded."""
        completions = [{"content": SDS_COMPLETION_VALID}]
        rewards = unified_nominal_reward_topk_interaction_bound(
            completions, domain="sds", mission=SDS_REQUIREMENTS
        )
        assert len(rewards) == 1
        assert 0.0 <= rewards[0] <= 1.0, "Reward should be in [0, 1]"


@pytest.mark.skipif(
    not HAS_REWARDS, reason=f"Could not import reward functions: {IMPORT_ERROR}"
)
class TestMinimalFeasibilityReward:
    """Test minimalist feasibility reward function."""

    def test_minimal_feasibility_reward_basic(self):
        """Test that minimal feasibility reward returns valid values."""
        completions = [{"content": SDS_COMPLETION_VALID}]
        rewards = minimal_feasibility_reward(
            completions, domain="sds", mission=SDS_REQUIREMENTS
        )
        assert len(rewards) == 1
        assert 0.0 <= rewards[0] <= 1.0, "Reward should be in [0, 1]"

    def test_minimal_feasibility_reward_invalid(self):
        """Test that invalid completion gets zero reward."""
        completions = [{"content": SDS_COMPLETION_INVALID}]
        rewards = minimal_feasibility_reward(
            completions, domain="sds", mission=SDS_REQUIREMENTS
        )
        assert len(rewards) == 1
        assert rewards[0] == 0.0, "Invalid completion should get 0.0"


@pytest.mark.skipif(
    not HAS_REWARDS, reason=f"Could not import reward functions: {IMPORT_ERROR}"
)
class TestRewardBatchHandling:
    """Test that reward functions handle batches correctly."""

    def test_batch_format_reward(self):
        """Test format reward with multiple completions."""
        completions = [
            {"content": SDS_COMPLETION_VALID},
            {"content": SDS_COMPLETION_INVALID},
            {"content": ""},
        ]
        rewards = unified_format_reward(completions)
        assert len(rewards) == 3
        assert rewards[0] > 0.0, "First (valid) should get positive reward"
        assert rewards[1] == 0.0, "Second (invalid) should get 0.0"
        assert rewards[2] == 0.0, "Third (empty) should get 0.0"

    def test_batch_execution_reward(self):
        """Test execution reward with multiple completions."""
        completions = [
            {"content": SDS_COMPLETION_VALID},
            {"content": SDS_COMPLETION_INVALID},
        ]
        rewards = unified_code_execution_reward_no_oracle(
            completions, domain="sds", mission=SDS_REQUIREMENTS
        )
        assert len(rewards) == 2
        assert all(0.0 <= r <= 1.0 for r in rewards), "All rewards should be in [0, 1]"


@pytest.mark.skipif(
    not HAS_REWARDS, reason=f"Could not import reward functions: {IMPORT_ERROR}"
)
class TestRewardRegistry:
    """Test that the public reward registry exposes the release reward names."""

    def test_registry_resolves_release_reward_functions(self):
        script_args = type(
            "ScriptArgs",
            (),
            {
                "cosine_min_value_wrong": -1.0,
                "cosine_max_value_wrong": -0.5,
                "cosine_min_value_correct": 0.5,
                "cosine_max_value_correct": 1.0,
                "cosine_max_len": 1000,
                "repetition_n_grams": 3,
                "repetition_max_penalty": -1.0,
                "parallel_code_exec_per_proc": 1,
                "e2b_router_url": None,
                "code_eval_test_batch_size": 1,
                "code_language": "python",
                "reward_funcs": [
                    "unified_format_reward",
                    "unified_code_execution_reward_no_oracle",
                    "unified_soft_nominal_reward",
                    "unified_nominal_reward_topk_interaction_bound",
                    "minimal_feasibility_reward",
                ],
            },
        )()

        reward_funcs = get_reward_funcs(script_args)
        assert len(reward_funcs) == len(script_args.reward_funcs)
        assert [func.__name__ for func in reward_funcs] == script_args.reward_funcs
