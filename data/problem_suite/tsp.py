"""Deterministic synthetic EUC_2D travelling-salesperson instances."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Callable

TSP_FAMILIES = ("uniform", "clustered", "grid_jittered", "radial", "anisotropic")
MIN_CITIES = 20
MAX_CITIES = 100
COORDINATE_LIMIT = 10_000

TSP_PROMPT_TEMPLATE = """
Output exactly two top-level blocks, in this order, with nothing else:
<think>...algorithm design and contract checks...</think>
<code>...a self-contained Python program...</code>

Task: Write a high-performance white-box solver for variable-size symmetric
EUC_2D travelling-salesperson problems.

The program must read one JSON object from stdin with keys "requirements" and
"catalog", and print exactly this JSON shape:
{{"selection":{{"tour":[...]}}}}

Contract:
- `tour` contains every zero-based city ID exactly once.
- Do not repeat the first city; the evaluator closes the cycle.
- Distances use TSPLIB EUC_2D rounding: floor(Euclidean distance + 0.5).
- Read all coordinates from stdin and support any size; never hardcode this instance.
- Honor requirements.time_limit_sec and seed randomness from requirements.random_seed.
- Use Python builtins, `math`, `random`, and optional `numpy` only. Do not access
  files or networks and do not call external or black-box solvers.

Instance summary:
- Number of cities: {n_cities}
- Coordinate family: {family}
- Edge-weight type: EUC_2D

Aim for low tour length while preserving exact feasibility. Validate the final
permutation and output schema before printing.
""".strip()


def _derived_seed(seed: int, index: int, family: str) -> int:
    digest = hashlib.sha256(f"N26-E12|{seed}|{index}|{family}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _unique_points(
    n_cities: int,
    point: Callable[[random.Random, int], tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(points) < n_cities:
        candidate = point(rng, attempts)
        attempts += 1
        candidate = (
            max(0, min(COORDINATE_LIMIT, int(candidate[0]))),
            max(0, min(COORDINATE_LIMIT, int(candidate[1]))),
        )
        if candidate in seen:
            continue
        seen.add(candidate)
        points.append(candidate)
        if attempts > n_cities * 10_000:
            raise RuntimeError("failed to generate unique TSP coordinates")
    return points


def _uniform(n_cities: int, rng: random.Random) -> list[tuple[int, int]]:
    return _unique_points(
        n_cities,
        lambda local, _: (
            local.randint(0, COORDINATE_LIMIT),
            local.randint(0, COORDINATE_LIMIT),
        ),
        rng,
    )


def _clustered(n_cities: int, rng: random.Random) -> list[tuple[int, int]]:
    count = rng.randint(3, 8)
    centers = [(rng.randint(1000, 9000), rng.randint(1000, 9000)) for _ in range(count)]
    spread = rng.randint(180, 900)

    def point(local: random.Random, _: int) -> tuple[int, int]:
        center_x, center_y = centers[local.randrange(count)]
        return (
            round(center_x + local.gauss(0, spread)),
            round(center_y + local.gauss(0, spread)),
        )

    return _unique_points(n_cities, point, rng)


def _grid_jittered(n_cities: int, rng: random.Random) -> list[tuple[int, int]]:
    side = math.ceil(math.sqrt(n_cities))
    spacing = COORDINATE_LIMIT // (side + 1)
    cells = [(row, column) for row in range(side) for column in range(side)]
    rng.shuffle(cells)
    jitter = max(1, spacing // 4)
    points = [
        (
            (column + 1) * spacing + rng.randint(-jitter, jitter),
            (row + 1) * spacing + rng.randint(-jitter, jitter),
        )
        for row, column in cells[:n_cities]
    ]
    if len(set(points)) != n_cities:
        raise RuntimeError("grid-jittered generator produced duplicate coordinates")
    return points


def _radial(n_cities: int, rng: random.Random) -> list[tuple[int, int]]:
    center_x = rng.randint(4200, 5800)
    center_y = rng.randint(4200, 5800)
    rings = rng.randint(2, 5)

    def point(local: random.Random, attempt: int) -> tuple[int, int]:
        angle = local.uniform(0.0, 2.0 * math.pi)
        radius = 900 + (attempt % rings) * 700 + local.randint(-120, 120)
        return (
            center_x + round(radius * math.cos(angle)),
            center_y + round(radius * math.sin(angle)),
        )

    return _unique_points(n_cities, point, rng)


def _anisotropic(n_cities: int, rng: random.Random) -> list[tuple[int, int]]:
    angle = rng.uniform(0.0, math.pi)
    cosine, sine = math.cos(angle), math.sin(angle)

    def point(local: random.Random, _: int) -> tuple[int, int]:
        along = local.uniform(-4300, 4300)
        across = local.gauss(0.0, 280)
        return (
            round(5000 + along * cosine - across * sine),
            round(5000 + along * sine + across * cosine),
        )

    return _unique_points(n_cities, point, rng)


_GENERATORS: dict[str, Callable[[int, random.Random], list[tuple[int, int]]]] = {
    "uniform": _uniform,
    "clustered": _clustered,
    "grid_jittered": _grid_jittered,
    "radial": _radial,
    "anisotropic": _anisotropic,
}


def tsp_sample(
    n_problems: int, seed: int, start_index: int = 0
) -> list[dict[str, Any]]:
    """Generate a balanced deterministic sequence over all five frozen families."""
    problems = []
    for offset in range(n_problems):
        index = start_index + offset
        family = TSP_FAMILIES[index % len(TSP_FAMILIES)]
        rng = random.Random(_derived_seed(seed, index, family))
        n_cities = rng.randint(MIN_CITIES, MAX_CITIES)
        points = _GENERATORS[family](n_cities, rng)
        cities = [
            {"id": city_id, "x": x, "y": y} for city_id, (x, y) in enumerate(points)
        ]
        requirements = {
            "n_cities": n_cities,
            "cities": cities,
            "edge_weight_type": "EUC_2D",
            "time_limit_sec": 5.0,
            "random_seed": _derived_seed(seed, index, "execution") % (2**31),
        }
        problems.append(
            {
                "uuid": f"tsp_{seed}_{family}_{index:06d}",
                "domain": "tsp",
                "problem_type": f"symmetric_euc_2d_{family}",
                "family": family,
                "mission": requirements,
                "requirements": requirements,
                "catalog": {"cities": cities},
                "target": {"one_tree_lower_bound": None, "optimal_length": None},
            }
        )
    return problems


def tsp_render_prompt(problem: dict[str, Any]) -> dict[str, Any]:
    requirements = problem["requirements"]
    return {
        "uuid": problem["uuid"],
        "problem": TSP_PROMPT_TEMPLATE.format(
            n_cities=requirements["n_cities"], family=problem["family"]
        ),
        "mission": problem["mission"],
        "domain": "tsp",
        "family": problem["family"],
    }
