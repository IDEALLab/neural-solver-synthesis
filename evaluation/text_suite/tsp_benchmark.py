"""Frozen TSPLIB benchmark and aggregation for N26-E12."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from evaluation.sds.utils import run_candidate
from evaluation.text_suite.tsp_compile_once import (
    CONDITIONS,
    SEEDS,
    TSPLIB_NAMES,
    audit_prompt_leakage,
    load_protocol,
    parse_tsplib,
    pretest_gate,
    sha256_file,
    sha256_text,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OPEN_R1_SRC = REPO_ROOT / "deps" / "open-r1" / "src"

if str(OPEN_R1_SRC) not in sys.path:
    sys.path.insert(0, str(OPEN_R1_SRC))

from open_r1.simulators.tsp_simulator import (  # noqa: E402
    TSPSimulator,
    euc_2d_distance,
)

SIMPLE_METHODS = (
    "nearest_neighbor",
    "insertion",
    "two_opt",
    "ortools_single_worker",
)
ORTOOLS_RANDOM_SEED_MODULUS = 2**31


def ortools_random_seed(seed: int) -> int:
    """Project a SHA-derived seed into OR-Tools' nonnegative int32 domain."""
    return seed % ORTOOLS_RANDOM_SEED_MODULUS


def distance_matrix(cities: list[dict[str, Any]]) -> list[list[int]]:
    return [[euc_2d_distance(first, second) for second in cities] for first in cities]


def tour_length(tour: list[int], matrix: list[list[int]]) -> int:
    return sum(
        matrix[tour[index]][tour[(index + 1) % len(tour)]] for index in range(len(tour))
    )


def nearest_neighbor(matrix: list[list[int]], start: int = 0) -> list[int]:
    remaining = set(range(len(matrix)))
    remaining.remove(start)
    tour = [start]
    while remaining:
        current = tour[-1]
        next_city = min(remaining, key=lambda city: (matrix[current][city], city))
        remaining.remove(next_city)
        tour.append(next_city)
    return tour


def cheapest_insertion(matrix: list[list[int]]) -> list[int]:
    n_cities = len(matrix)
    farthest = max(range(1, n_cities), key=lambda city: (matrix[0][city], -city))
    tour = [0, farthest]
    remaining = set(range(n_cities)) - set(tour)
    while remaining:
        delta, city, position = min(
            (
                matrix[tour[index]][candidate]
                + matrix[candidate][tour[(index + 1) % len(tour)]]
                - matrix[tour[index]][tour[(index + 1) % len(tour)]],
                candidate,
                index + 1,
            )
            for candidate in remaining
            for index in range(len(tour))
        )
        del delta
        tour.insert(position, city)
        remaining.remove(city)
    return tour


def two_opt(
    matrix: list[list[int]], seed: int, deadline: float, max_restarts: int = 32
) -> list[int]:
    """Deterministic-seed multistart 2-opt bounded by the common wall-time limit."""
    rng = random.Random(seed)
    incumbent = min(
        (nearest_neighbor(matrix, start) for start in range(len(matrix))),
        key=lambda candidate: (tour_length(candidate, matrix), candidate),
    )
    incumbent_cost = tour_length(incumbent, matrix)
    for restart in range(max_restarts):
        if time.perf_counter() >= deadline:
            break
        candidate = (
            list(incumbent)
            if restart == 0
            else rng.sample(range(len(matrix)), len(matrix))
        )
        candidate_cost = tour_length(candidate, matrix)
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            best_move: tuple[int, int, int] | None = None
            for left in range(1, len(candidate) - 1):
                a, b = candidate[left - 1], candidate[left]
                for right in range(left + 1, len(candidate)):
                    c = candidate[right]
                    d = candidate[(right + 1) % len(candidate)]
                    delta = matrix[a][c] + matrix[b][d] - matrix[a][b] - matrix[c][d]
                    move = (delta, left, right)
                    if delta < 0 and (best_move is None or move < best_move):
                        best_move = move
                if time.perf_counter() >= deadline:
                    break
            if best_move is not None:
                delta, left, right = best_move
                candidate[left : right + 1] = reversed(candidate[left : right + 1])
                candidate_cost += delta
                improved = True
        if (candidate_cost, candidate) < (incumbent_cost, incumbent):
            incumbent, incumbent_cost = candidate, candidate_cost
    return incumbent


def ortools_tsp(
    matrix: list[list[int]], seed: int, time_limit: float
) -> tuple[list[int] | None, str]:
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:
        raise RuntimeError("OR-Tools is a required E12 benchmark dependency") from exc

    manager = pywrapcp.RoutingIndexManager(len(matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit(from_index: int, to_index: int) -> int:
        return matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    callback = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(callback)
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.FromMilliseconds(round(time_limit * 1000))
    parameters.log_search = False
    if hasattr(parameters, "sat_parameters"):
        parameters.sat_parameters.num_search_workers = 1
        parameters.sat_parameters.random_seed = ortools_random_seed(seed)
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        return None, "NO_SOLUTION"
    tour = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        tour.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    return tour, "FEASIBLE"


def _strict_generated_tour(result: Any) -> Any:
    if not isinstance(result, dict) or set(result) not in (
        {"selection"},
        {"selection", "execution_time"},
    ):
        return None
    selection = result["selection"]
    if not isinstance(selection, dict) or set(selection) != {"tour"}:
        return None
    return selection["tour"]


def _trial_seed(seed: int, method: str, instance: str, repetition: int) -> int:
    return int(sha256_text(f"N26-E12|{seed}|{method}|{instance}|{repetition}")[:8], 16)


def _run_trial(payload: dict[str, Any]) -> dict[str, Any]:
    mission = payload["mission"]
    method = payload["method"]
    seed = int(payload["seed"])
    repetition = int(payload["repetition"])
    trial_seed = _trial_seed(seed, method, payload["instance"], repetition)
    time_limit = float(payload["time_limit"])
    matrix = distance_matrix(mission["cities"])
    status = "FEASIBLE"
    started = time.perf_counter()
    if method in CONDITIONS:
        requirements = {
            **mission,
            "time_limit_sec": time_limit,
            "random_seed": trial_seed,
        }
        result = run_candidate(
            payload["solver_code"],
            {"requirements": requirements, "catalog": {"cities": mission["cities"]}},
            timeout=time_limit,
        )
        tour = _strict_generated_tour(result)
        status = (
            "FEASIBLE"
            if tour is not None and "error" not in result
            else str(result.get("error", "INVALID"))
        )
    elif method == "nearest_neighbor":
        tour = min(
            (nearest_neighbor(matrix, start) for start in range(len(matrix))),
            key=lambda candidate: (tour_length(candidate, matrix), candidate),
        )
    elif method == "insertion":
        tour = cheapest_insertion(matrix)
    elif method == "two_opt":
        tour = two_opt(matrix, trial_seed, started + time_limit)
    elif method == "ortools_single_worker":
        tour, status = ortools_tsp(matrix, trial_seed, time_limit)
    else:
        raise ValueError(f"unknown E12 method: {method}")
    runtime = time.perf_counter() - started
    simulator = TSPSimulator()
    feasible = simulator.validate_design(tour, mission)
    simulation = simulator.simulate(tour, mission) if feasible else {}
    feasible = feasible and bool(simulation.get("feasible", False))
    length = float(simulation.get("tour_length", float("inf")))
    optimum = float(payload["optimum"])
    gap = max(0.0, (length - optimum) / optimum) if feasible else 1.0
    return {
        "seed": seed,
        "condition": payload["condition"],
        "method": method,
        "instance": payload["instance"],
        "family": payload["family"],
        "n_cities": mission["n_cities"],
        "repetition": repetition,
        "trial_seed": trial_seed,
        "feasible": feasible,
        "tour_length": length,
        "optimum": optimum,
        "optimality_gap": gap,
        "runtime_seconds": runtime,
        "status": status,
        "solver_sha256": payload.get("solver_sha256", ""),
    }


def _family(name: str) -> str:
    match = re.match(r"[A-Za-z]+", name)
    return match.group(0).lower() if match else "unknown"


def build_trials(
    protocol_path: Path,
    global_freeze_path: Path,
    external_root: Path,
) -> list[dict[str, Any]]:
    protocol = load_protocol(protocol_path, require_launch_ready=True)
    pretest_gate(protocol_path, global_freeze_path, external_root)
    freeze = json.loads(global_freeze_path.read_text())
    trials = []
    missions = {
        name: parse_tsplib(external_root / f"{name}.tsp", name) for name in TSPLIB_NAMES
    }
    for condition in CONDITIONS:
        for seed in SEEDS:
            selected = freeze["selected"][condition][str(seed)]
            solver_path = Path(selected["solver_path"])
            code = solver_path.read_text()
            audit_prompt_leakage([code], protocol)
            if sha256_file(solver_path) != selected["solver_sha256"]:
                raise ValueError("frozen generated solver drift")
            for name, mission in missions.items():
                for repetition in range(protocol["benchmark"]["repetitions"]):
                    trials.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "method": condition,
                            "instance": name,
                            "family": _family(name),
                            "repetition": repetition,
                            "mission": mission,
                            "optimum": protocol["benchmark"]["optima"][name],
                            "time_limit": protocol["benchmark"]["time_limit_seconds"],
                            "solver_code": code,
                            "solver_sha256": selected["solver_sha256"],
                        }
                    )
    for method in SIMPLE_METHODS:
        for name, mission in missions.items():
            for repetition in range(protocol["benchmark"]["repetitions"]):
                trials.append(
                    {
                        "seed": 0,
                        "condition": "native_baseline",
                        "method": method,
                        "instance": name,
                        "family": _family(name),
                        "repetition": repetition,
                        "mission": mission,
                        "optimum": protocol["benchmark"]["optima"][name],
                        "time_limit": protocol["benchmark"]["time_limit_seconds"],
                    }
                )
    return trials


def run_benchmark(
    protocol_path: Path,
    global_freeze_path: Path,
    external_root: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(
            f"refusing to reuse immutable E12 benchmark directory: {output_dir}"
        )
    trials = build_trials(protocol_path, global_freeze_path, external_root)
    output_dir.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_run_trial, trials))
    with (output_dir / "per_trial.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    optional = {
        executable: shutil.which(executable) for executable in ("LKH", "concorde")
    }
    protocol = load_protocol(protocol_path)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "global_freeze_sha256": sha256_file(global_freeze_path),
        "external_provenance_sha256": sha256_file(external_root / "provenance.json"),
        "trial_count": len(rows),
        "generated_trial_count": 2 * 3 * 12 * 5,
        "native_trial_count": 4 * 12 * 5,
        "workers": workers,
        "optional_solver_preflight": optional,
        "optional_solver_completion_gate": False,
        "all_outcomes_reported": True,
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [bool(row["feasible"]) for row in rows]
    gaps = [float(row["optimality_gap"]) for row in rows]
    runtimes = [float(row["runtime_seconds"]) for row in rows]
    lengths = [
        float(row["tour_length"])
        for row in rows
        if math.isfinite(float(row["tour_length"]))
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["instance"]].append(row)
    within_gap_std = [
        statistics.pstdev(float(row["optimality_gap"]) for row in group)
        for group in grouped.values()
    ]
    best_gaps = [
        min(float(row["optimality_gap"]) for row in group) for group in grouped.values()
    ]
    return {
        "trial_count": len(rows),
        "instance_count": len(grouped),
        "feasibility": sum(feasible) / len(feasible),
        "mean_gap": statistics.fmean(gaps),
        "median_gap": statistics.median(gaps),
        "best_gap": min(gaps),
        "mean_best_of_five_gap": statistics.fmean(best_gaps),
        "mean_tour_length": statistics.fmean(lengths) if lengths else None,
        "runtime_mean": statistics.fmean(runtimes),
        "runtime_stddev": statistics.pstdev(runtimes),
        "mean_within_instance_gap_stddev": statistics.fmean(within_gap_std),
        "max_within_instance_gap_stddev": max(within_gap_std),
    }


def validate_benchmark_pairing(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> None:
    """Require the complete predeclared trial grid with no extras or duplicates."""
    repetitions = range(protocol["benchmark"]["repetitions"])
    generated = {
        (seed, condition, condition, instance, repetition)
        for condition in CONDITIONS
        for seed in SEEDS
        for instance in TSPLIB_NAMES
        for repetition in repetitions
    }
    native = {
        (0, "native_baseline", method, instance, repetition)
        for method in SIMPLE_METHODS
        for instance in TSPLIB_NAMES
        for repetition in repetitions
    }
    expected = generated | native
    actual = [
        (
            int(row["seed"]),
            row["condition"],
            row["method"],
            row["instance"],
            int(row["repetition"]),
        )
        for row in rows
    ]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate E12 benchmark pairing key")
    actual_set = set(actual)
    if actual_set != expected:
        missing = sorted(expected - actual_set)[:5]
        unexpected = sorted(actual_set - expected)[:5]
        raise ValueError(
            f"incomplete E12 benchmark pairing: missing={missing}, "
            f"unexpected={unexpected}"
        )


def aggregate(
    protocol_path: Path, benchmark_dir: Path, output_dir: Path
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path, require_launch_ready=True)
    with (benchmark_dir / "per_trial.csv").open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    expected = 2 * 3 * 12 * 5 + 4 * 12 * 5
    if len(raw) != expected:
        raise ValueError(f"E12 benchmark rows: {len(raw)} != {expected}")
    rows = [
        {
            **row,
            "seed": int(row["seed"]),
            "n_cities": int(row["n_cities"]),
            "repetition": int(row["repetition"]),
            "feasible": row["feasible"].lower() == "true",
            "tour_length": float(row["tour_length"]),
            "optimality_gap": float(row["optimality_gap"]),
            "runtime_seconds": float(row["runtime_seconds"]),
        }
        for row in raw
    ]
    validate_benchmark_pairing(rows, protocol)

    summaries = {}
    for method in (*CONDITIONS, *SIMPLE_METHODS):
        method_rows = [row for row in rows if row["method"] == method]
        if method in CONDITIONS:
            summaries[method] = {
                str(seed): _summary([row for row in method_rows if row["seed"] == seed])
                for seed in SEEDS
            }
            summaries[method]["macro"] = {
                key: statistics.fmean(
                    summaries[method][str(seed)][key] for seed in SEEDS
                )
                for key in (
                    "feasibility",
                    "mean_gap",
                    "median_gap",
                    "mean_best_of_five_gap",
                    "runtime_mean",
                    "mean_within_instance_gap_stddev",
                )
            }
        else:
            summaries[method] = _summary(method_rows)

    trained = summaries["trained"]
    base = summaries["base64"]
    simple_best = min(summaries[method]["median_gap"] for method in SIMPLE_METHODS[:-1])
    gate_components = {
        "all_trained_seeds_feasibility_at_least_0_95": all(
            trained[str(seed)]["feasibility"] >= 0.95 for seed in SEEDS
        ),
        "all_trained_seeds_median_gap_below_0_10": all(
            trained[str(seed)]["median_gap"] < 0.10 for seed in SEEDS
        ),
        "stable_feasibility_range_at_most_0_05": (
            max(trained[str(seed)]["feasibility"] for seed in SEEDS)
            - min(trained[str(seed)]["feasibility"] for seed in SEEDS)
            <= 0.05
        ),
        "stable_median_gap_range_at_most_0_05": (
            max(trained[str(seed)]["median_gap"] for seed in SEEDS)
            - min(trained[str(seed)]["median_gap"] for seed in SEEDS)
            <= 0.05
        ),
        "trained_improves_over_base_each_seed": all(
            trained[str(seed)]["median_gap"] < base[str(seed)]["median_gap"]
            for seed in SEEDS
        ),
        "trained_macro_improves_over_best_simple_heuristic": (
            trained["macro"]["median_gap"] < simple_best
        ),
    }
    primary_gate = all(gate_components.values())

    by_family = []
    by_size = []
    for method in (*CONDITIONS, *SIMPLE_METHODS):
        method_rows = [row for row in rows if row["method"] == method]
        for family in sorted({row["family"] for row in method_rows}):
            group = [row for row in method_rows if row["family"] == family]
            by_family.append({"method": method, "family": family, **_summary(group)})
        for size in sorted({row["n_cities"] for row in method_rows}):
            group = [row for row in method_rows if row["n_cities"] == size]
            by_size.append({"method": method, "n_cities": size, **_summary(group)})

    output_dir.mkdir(parents=True, exist_ok=False)
    for name, values in (("by_family.csv", by_family), ("by_size.csv", by_size)):
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    result = {
        "protocol_id": protocol["protocol_id"],
        "summaries": summaries,
        "primary_gate": {
            "passed": primary_gate,
            "components": gate_components,
            "interpretation_if_passed": protocol["outcome_policy"]["success"],
            "interpretation_if_failed": protocol["outcome_policy"]["failure"],
        },
        "all_outcomes_reported": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--global-freeze", type=Path, required=True)
    run.add_argument("--external-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--workers", type=int, default=64)
    collect = commands.add_parser("aggregate")
    collect.add_argument("--protocol", type=Path, required=True)
    collect.add_argument("--benchmark-dir", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_benchmark(
            args.protocol,
            args.global_freeze,
            args.external_root,
            args.output_dir,
            args.workers,
        )
    else:
        result = aggregate(args.protocol, args.benchmark_dir, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
