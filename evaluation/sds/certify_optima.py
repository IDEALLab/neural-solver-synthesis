#!/usr/bin/env python3
"""Certify SDS reference objectives with an auditable CP-SAT run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from ortools import __version__ as ortools_version
from ortools.sat.python import cp_model
from syndeopt.core.feasibility import feasible
from syndeopt.core.instance import SDSInstance
from syndeopt.core.scoring import score

from evaluation.sds.utils import mission_to_instance

OPTIMAL_STATUS = "OPTIMAL"
FEASIBLE_STATUSES = {"OPTIMAL", "FEASIBLE"}
OBJECTIVE_VALIDATION_TOLERANCE = 1e-6


class DirtyCheckoutError(RuntimeError):
    def __init__(self, status: str):
        super().__init__(f"source checkout is dirty:\n{status}")


class DatasetCountError(RuntimeError):
    def __init__(self, actual: int, expected: int):
        super().__init__(f"dataset row count {actual} != expected {expected}")


class DuplicateUuidError(RuntimeError):
    def __init__(self, uuid: str):
        super().__init__(f"duplicate UUID in dataset: {uuid}")


class CertificationValidationError(RuntimeError):
    def __init__(self):
        super().__init__("independent certification validation failed")


@dataclass(frozen=True)
class CertificationConfig:
    dataset: str
    split: str
    max_time_sec: float
    solver_seed: int
    num_workers: int
    expected_count: int | None
    limit: int | None


def canonical_json(value: Any) -> str:
    """Serialize data deterministically for content fingerprints."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_cp_model(
    instance: SDSInstance,
) -> tuple[cp_model.CpModel, list[cp_model.IntVar]]:
    """Build the exact binary quadratic SDS model used for certification."""
    model = cp_model.CpModel()
    selected = [model.new_bool_var(f"x_{idx}") for idx in range(instance.n)]
    interactions: dict[tuple[int, int], cp_model.IntVar] = {}

    for left, right in instance.W:
        active = model.new_bool_var(f"y_{left}_{right}")
        interactions[(left, right)] = active
        model.add(active <= selected[left])
        model.add(active <= selected[right])
        model.add(active >= selected[left] + selected[right] - 1)

    model.add(sum(selected) >= instance.card.L)
    model.add(sum(selected) <= instance.card.U)

    for required, dependent in instance.precedence:
        model.add(selected[dependent] <= selected[required])
    for left, right in instance.mutex:
        model.add(selected[left] + selected[right] <= 1)
    for members in instance.groups.values():
        model.add(sum(selected[idx] for idx in members) <= 1)

    objective = sum(
        instance.w[idx] * selected[idx] for idx in range(instance.n)
    )
    objective += sum(
        instance.W[pair] * interaction for pair, interaction in interactions.items()
    )
    model.maximize(objective)
    return model, selected


def solve_instance(
    instance: SDSInstance,
    *,
    max_time_sec: float,
    solver_seed: int,
    num_workers: int,
) -> dict[str, Any]:
    """Solve one instance and independently verify every returned incumbent."""
    model, selected = build_cp_model(instance)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_sec
    solver.parameters.random_seed = solver_seed
    solver.parameters.num_search_workers = num_workers

    started = time.perf_counter()
    status_code = solver.solve(model)
    elapsed = time.perf_counter() - started
    status = solver.status_name(status_code)

    result: dict[str, Any] = {
        "status": status,
        "status_code": int(status_code),
        "certified_optimal": status == OPTIMAL_STATUS,
        "proved_infeasible": status == "INFEASIBLE",
        "wall_time_sec": elapsed,
        "solver_wall_time_sec": solver.wall_time,
        "objective": None,
        "best_bound": None,
        "absolute_bound_gap": None,
        "relative_bound_gap": None,
        "selected_count": None,
        "independently_feasible": None,
        "independently_scored_objective": None,
        "objective_validation_error": None,
    }

    if status not in FEASIBLE_STATUSES:
        return result

    mask = 0
    for idx, variable in enumerate(selected):
        if solver.boolean_value(variable):
            mask |= 1 << idx

    objective = float(solver.objective_value)
    bound = float(solver.best_objective_bound)
    independent_feasible = feasible(instance, mask)
    independent_score = float(score(instance, mask)) if independent_feasible else None
    validation_error = (
        abs(objective - independent_score)
        if independent_score is not None
        else None
    )
    absolute_gap = max(0.0, bound - objective)
    denominator = max(abs(bound), 1e-12)

    result.update(
        {
            "objective": objective,
            "best_bound": bound,
            "absolute_bound_gap": absolute_gap,
            "relative_bound_gap": absolute_gap / denominator,
            "selected_count": mask.bit_count(),
            "independently_feasible": independent_feasible,
            "independently_scored_objective": independent_score,
            "objective_validation_error": validation_error,
        }
    )
    return result


def git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def collect_provenance(repo_root: Path) -> dict[str, Any]:
    """Capture code and execution provenance before loading the dataset."""
    dirty = git_output(repo_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise DirtyCheckoutError(dirty)

    return {
        "top_level_commit": git_output(repo_root, "rev-parse", "HEAD"),
        "branch": git_output(repo_root, "branch", "--show-current"),
        "submodules": git_output(repo_root, "submodule", "status", "--recursive").splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "ortools_version": ortools_version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": platform.node(),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def register_uuid(seen_uuids: set[str], uuid: str) -> None:
    if uuid in seen_uuids:
        raise DuplicateUuidError(uuid)
    seen_uuids.add(uuid)


def require_valid_certifications(failures: bool) -> None:
    if failures:
        raise CertificationValidationError


def summarize(rows: list[dict[str, Any]], config: CertificationConfig) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        statuses[status] = statuses.get(status, 0) + 1

    validation_errors = [
        float(row["objective_validation_error"])
        for row in rows
        if row["objective_validation_error"] is not None
    ]
    certified = sum(bool(row["certified_optimal"]) for row in rows)
    return {
        "dataset": config.dataset,
        "split": config.split,
        "row_count": len(rows),
        "expected_count": config.expected_count,
        "status_counts": statuses,
        "certified_optimal_count": certified,
        "certified_optimal_rate": certified / len(rows) if rows else 0.0,
        "independent_feasibility_failures": sum(
            row["independently_feasible"] is False for row in rows
        ),
        "max_objective_validation_error": max(validation_errors, default=0.0),
        "total_wall_time_sec": sum(float(row["wall_time_sec"]) for row in rows),
    }


def run_certification(
    config: CertificationConfig, output_dir: Path, repo_root: Path
) -> dict[str, Any]:
    """Run a non-overwriting certification batch and return its summary."""
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance = collect_provenance(repo_root)
    write_json(output_dir / "run_config.json", asdict(config))
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "run_state.json", {"status": "running"})

    dataset = load_dataset(config.dataset, split=config.split)
    row_count = min(len(dataset), config.limit) if config.limit else len(dataset)
    if (
        config.expected_count is not None
        and config.limit is None
        and row_count != config.expected_count
    ):
        raise DatasetCountError(row_count, config.expected_count)

    rows: list[dict[str, Any]] = []
    seen_uuids: set[str] = set()
    mission_hashes: list[str] = []
    jsonl_path = output_dir / "certifications.jsonl"

    try:
        with jsonl_path.open("x") as stream:
            for index in range(row_count):
                item = dataset[index]
                uuid = str(item.get("uuid") or f"row-{index:06d}")
                register_uuid(seen_uuids, uuid)

                mission = item.get("mission")
                parsed_mission = json.loads(mission) if isinstance(mission, str) else mission
                mission_json = canonical_json(parsed_mission)
                mission_hash = sha256_text(mission_json)
                mission_hashes.append(mission_hash)
                instance = mission_to_instance(parsed_mission)

                row = {
                    "dataset": config.dataset,
                    "split": config.split,
                    "dataset_index": index,
                    "uuid": uuid,
                    "mission_sha256": mission_hash,
                    "n_variables": instance.n,
                    **solve_instance(
                        instance,
                        max_time_sec=config.max_time_sec,
                        solver_seed=config.solver_seed,
                        num_workers=config.num_workers,
                    ),
                }
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()

        csv_path = output_dir / "certifications.csv"
        with csv_path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        summary = summarize(rows, config)
        summary["dataset_content_sha256"] = sha256_text("\n".join(mission_hashes))
        write_json(output_dir / "summary.json", summary)

        failures = (
            summary["independent_feasibility_failures"]
            or summary["max_objective_validation_error"]
            > OBJECTIVE_VALIDATION_TOLERANCE
        )
        state = "failed_validation" if failures else "complete"
        write_json(output_dir / "run_state.json", {"status": state})
        require_valid_certifications(bool(failures))
    except BaseException as error:
        write_json(
            output_dir / "run_state.json",
            {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
        raise
    else:
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-time-sec", type=float, default=60.0)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CertificationConfig(
        dataset=args.dataset,
        split=args.split,
        max_time_sec=args.max_time_sec,
        solver_seed=args.solver_seed,
        num_workers=args.num_workers,
        expected_count=args.expected_count,
        limit=args.limit,
    )
    summary = run_certification(config, args.output_dir, args.repo_root.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
