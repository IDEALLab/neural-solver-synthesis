"""Feasibility-gated reward stack for the zero-SFT TSP experiment."""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from typing import Any

from .simulators.tsp_simulator import TSPSimulator
from .simulators.utils import extract_block, run_candidate, validate_code_structure

_STRICT_FORMAT = re.compile(
    r"^\s*<think>.*?</think>\s*<code>.*?</code>\s*$",
    re.IGNORECASE | re.DOTALL,
)
_CACHE_LIMIT = 4096
_EVALUATION_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion:
        return str(completion[0].get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _mission_at(kwargs: dict[str, Any], index: int) -> dict[str, Any]:
    missions = kwargs.get("mission", {})
    mission = (
        missions[index]
        if isinstance(missions, list) and index < len(missions)
        else missions
    )
    if isinstance(mission, str):
        try:
            mission = json.loads(mission)
        except json.JSONDecodeError:
            return {}
    return mission if isinstance(mission, dict) else {}


def _cache_key(text: str, mission: dict[str, Any]) -> str:
    payload = json.dumps(mission, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{text}\0{payload}".encode()).hexdigest()


def _extract_tour(result: Any) -> Any:
    if not isinstance(result, dict) or set(result) != {"selection"}:
        return None
    selection = result.get("selection")
    if not isinstance(selection, dict) or set(selection) != {"tour"}:
        return None
    return selection.get("tour")


def _evaluate(text: str, mission: dict[str, Any]) -> dict[str, Any]:
    key = _cache_key(text, mission)
    cached = _EVALUATION_CACHE.get(key)
    if cached is not None:
        _EVALUATION_CACHE.move_to_end(key)
        return cached

    outcome: dict[str, Any] = {
        "strict_format": bool(_STRICT_FORMAT.fullmatch(text)),
        "executed": False,
        "feasible": False,
        "quality": 0.0,
    }

    def remember() -> dict[str, Any]:
        _EVALUATION_CACHE[key] = outcome
        _EVALUATION_CACHE.move_to_end(key)
        while len(_EVALUATION_CACHE) > _CACHE_LIMIT:
            _EVALUATION_CACHE.popitem(last=False)
        return outcome

    code = extract_block(text, "code")
    if not code or not validate_code_structure(code):
        return remember()

    requirements = {
        "n_cities": int(mission.get("n_cities", 0)),
        "cities": mission.get("cities", []),
        "edge_weight_type": "EUC_2D",
        "time_limit_sec": float(mission.get("time_limit_sec", 5.0)),
        "random_seed": int(mission.get("random_seed", 0)),
    }
    result = run_candidate(
        code,
        stdin_obj={
            "requirements": requirements,
            "catalog": {"cities": requirements["cities"]},
        },
        timeout=5.0,
    )
    if not isinstance(result, dict) or "error" in result:
        return remember()
    outcome["executed"] = True
    tour = _extract_tour(result)
    simulator = TSPSimulator()
    if not simulator.validate_design(tour, requirements):
        return remember()
    simulation = simulator.simulate(tour, requirements)
    if not bool(simulation.get("feasible", False)):
        return remember()
    outcome["feasible"] = True
    outcome["quality"] = simulator.get_reward(tour, requirements)

    return remember()


def _batch(completions: list[Any], kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _evaluate(_completion_text(completion), _mission_at(kwargs, index))
        for index, completion in enumerate(completions)
    ]


def tsp_feasible_format_reward(completions, **kwargs) -> list[float]:
    """Award format credit only when the generated solver returns a feasible tour."""
    return [
        float(row["strict_format"] and row["feasible"])
        for row in _batch(completions, kwargs)
    ]


def tsp_feasible_execution_reward(completions, **kwargs) -> list[float]:
    """Award execution credit only for an exact feasible permutation."""
    return [
        float(row["executed"] and row["feasible"])
        for row in _batch(completions, kwargs)
    ]


def tsp_feasible_quality_reward(completions, **kwargs) -> list[float]:
    """Return the deterministic one-tree-bound ratio, hard-gated by feasibility."""
    return [
        float(row["quality"]) if row["feasible"] else 0.0
        for row in _batch(completions, kwargs)
    ]
