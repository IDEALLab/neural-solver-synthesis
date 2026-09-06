"""Strict symmetric EUC_2D travelling-salesperson simulator."""

from __future__ import annotations

import math
from typing import Any

from .base import BaseSimulator

MIN_TOUR_CITIES = 3


def euc_2d_distance(first: dict[str, Any], second: dict[str, Any]) -> int:
    """Return the TSPLIB EUC_2D distance, avoiding Python bankers rounding."""
    distance = math.hypot(
        float(first["x"]) - float(second["x"]),
        float(first["y"]) - float(second["y"]),
    )
    return math.floor(distance + 0.5)


def one_tree_lower_bound(cities: list[dict[str, Any]]) -> int:
    """Compute the deterministic minimum 1-tree lower bound rooted at city 0."""
    n_cities = len(cities)
    if n_cities < MIN_TOUR_CITIES:
        return 0

    root_edges = sorted(
        (euc_2d_distance(cities[0], cities[index]), index)
        for index in range(1, n_cities)
    )
    root_cost = root_edges[0][0] + root_edges[1][0]

    remaining = set(range(1, n_cities))
    first = min(remaining)
    remaining.remove(first)
    tree = {first}
    mst_cost = 0
    while remaining:
        cost, source, target = min(
            (
                euc_2d_distance(cities[source], cities[target]),
                source,
                target,
            )
            for source in tree
            for target in remaining
        )
        del source
        mst_cost += cost
        tree.add(target)
        remaining.remove(target)
    return int(root_cost + mst_cost)


class TSPSimulator(BaseSimulator):
    """Validate Hamiltonian tours and score their closed-cycle EUC_2D length."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.name = "tsp"
        self.domain = "tsp"

    def _setup(self) -> None:
        pass

    @staticmethod
    def _cities(requirements: dict[str, Any]) -> list[dict[str, Any]]:
        cities = requirements.get("cities")
        if not isinstance(cities, list):
            return []
        normalized = []
        for city in cities:
            if not isinstance(city, dict):
                return []
            try:
                normalized.append(
                    {
                        "id": int(city["id"]),
                        "x": float(city["x"]),
                        "y": float(city["y"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                return []
        normalized.sort(key=lambda row: row["id"])
        return normalized

    def validate_design(self, design: Any, requirements: dict[str, Any]) -> bool:
        cities = self._cities(requirements)
        n_cities = int(requirements.get("n_cities", len(cities)))
        if n_cities < MIN_TOUR_CITIES or len(cities) != n_cities:
            return False
        if [city["id"] for city in cities] != list(range(n_cities)):
            return False
        if not isinstance(design, list) or len(design) != n_cities:
            return False
        if any(
            isinstance(city_id, bool) or not isinstance(city_id, int)
            for city_id in design
        ):
            return False
        return sorted(design) == list(range(n_cities))

    def simulate(self, design: Any, requirements: dict[str, Any]) -> dict[str, float]:
        cities = self._cities(requirements)
        if not self.validate_design(design, requirements):
            return {
                "feasible": False,
                "tour_length": float("inf"),
                "one_tree_lower_bound": float(one_tree_lower_bound(cities)),
            }
        tour = list(design)
        tour_length = sum(
            euc_2d_distance(cities[tour[index]], cities[tour[(index + 1) % len(tour)]])
            for index in range(len(tour))
        )
        return {
            "feasible": True,
            "tour_length": float(tour_length),
            "one_tree_lower_bound": float(one_tree_lower_bound(cities)),
        }

    def _calculate_reward(
        self,
        results: dict[str, float],
        requirements: dict[str, Any],  # noqa: ARG002
    ) -> float:
        if not bool(results.get("feasible", False)):
            return 0.0
        tour_length = float(results.get("tour_length", float("inf")))
        lower_bound = float(results.get("one_tree_lower_bound", 0.0))
        if not math.isfinite(tour_length) or tour_length <= 0.0 or lower_bound <= 0.0:
            return 0.0
        return max(0.0, min(1.0, lower_bound / tour_length))

    def _get_capabilities(self) -> dict[str, Any]:
        return {
            "input_format": "A permutation of zero-based city IDs",
            "objective": "minimize_closed_cycle_euc_2d_length",
            "constraints": ["exact_permutation", "no_repeated_terminal_city"],
            "feasibility_check": True,
        }
