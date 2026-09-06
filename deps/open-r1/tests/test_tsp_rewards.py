from __future__ import annotations

import pytest
from open_r1.rewards_tsp import (
    tsp_feasible_execution_reward,
    tsp_feasible_format_reward,
    tsp_feasible_quality_reward,
)
from open_r1.simulators.tsp_simulator import (
    TSPSimulator,
    euc_2d_distance,
    one_tree_lower_bound,
)
from open_r1.simulators.utils import validate_safety

ROUNDED_HALF = 3
THREE_FOUR_FIVE = 5
UNIT_SQUARE_TOUR = 4.0


MISSION = {
    "n_cities": 4,
    "cities": [
        {"id": 0, "x": 0, "y": 0},
        {"id": 1, "x": 1, "y": 0},
        {"id": 2, "x": 1, "y": 1},
        {"id": 3, "x": 0, "y": 1},
    ],
    "time_limit_sec": 5.0,
}


def _completion(tour: list[int]) -> list[dict[str, str]]:
    code = (
        "import json, sys\n"
        "def solve_tsp():\n"
        f"    print(json.dumps({{'selection': {{'tour': {tour!r}}}}}))\n"
        "solve_tsp()\n"
    )
    return [{"content": f"<think>check</think><code>{code}</code>"}]


def test_euc_2d_uses_floor_plus_half_not_bankers_rounding() -> None:
    assert euc_2d_distance({"x": 0, "y": 0}, {"x": 2.5, "y": 0}) == ROUNDED_HALF
    assert euc_2d_distance({"x": 0, "y": 0}, {"x": 3, "y": 4}) == THREE_FOUR_FIVE


def test_tsp_simulator_closes_cycle_and_rejects_repeated_terminal() -> None:
    simulator = TSPSimulator()
    assert simulator.simulate([0, 1, 2, 3], MISSION)["tour_length"] == UNIT_SQUARE_TOUR
    assert one_tree_lower_bound(MISSION["cities"]) == UNIT_SQUARE_TOUR
    assert not simulator.validate_design([0, 1, 2, 3, 0], MISSION)
    assert not simulator.validate_design([0, 1, 1, 3], MISSION)


def test_rewards_are_hard_gated_and_quality_is_monotone() -> None:
    feasible = [_completion([0, 1, 2, 3])]
    invalid = [_completion([0, 1, 1, 3])]
    kwargs = {"mission": [MISSION]}
    assert tsp_feasible_format_reward(feasible, **kwargs) == [1.0]
    assert tsp_feasible_execution_reward(feasible, **kwargs) == [1.0]
    assert tsp_feasible_quality_reward(feasible, **kwargs) == pytest.approx([1.0])
    assert tsp_feasible_format_reward(invalid, **kwargs) == [0.0]
    assert tsp_feasible_execution_reward(invalid, **kwargs) == [0.0]
    assert tsp_feasible_quality_reward(invalid, **kwargs) == [0.0]

    stretched = {
        **MISSION,
        "n_cities": 5,
        "cities": [
            {"id": 0, "x": 0, "y": 0},
            {"id": 1, "x": 10, "y": 0},
            {"id": 2, "x": 11, "y": 0},
            {"id": 3, "x": 11, "y": 10},
            {"id": 4, "x": 0, "y": 10},
        ],
    }
    short = tsp_feasible_quality_reward(
        [_completion([0, 1, 2, 3, 4])], mission=[stretched]
    )[0]
    long = tsp_feasible_quality_reward(
        [_completion([0, 2, 4, 1, 3])], mission=[stretched]
    )[0]
    assert short > long >= 0.0


@pytest.mark.parametrize(
    "code",
    [
        "import socket\ndef solve_tsp(): pass",
        "import pathlib\ndef solve_tsp(): pass",
        "import requests\ndef solve_tsp(): pass",
        "import sys\ndef solve_tsp(): print(sys.modules)",
        "def solve_tsp(): open('/tmp/x', 'w')",
    ],
)
def test_tsp_generated_code_cannot_access_network_or_files(code: str) -> None:
    assert validate_safety(code)[0] is False
