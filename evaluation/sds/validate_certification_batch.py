#!/usr/bin/env python3
# ruff: noqa: TRY003
"""Validate an SDS certification batch against immutable data revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


class BatchValidationError(RuntimeError):
    """Raised when an artifact or independent check does not match the manifest."""


@dataclass(frozen=True)
class FeasibilityCertificate:
    maximum_cardinality: int
    required_minimum: int
    independently_infeasible: bool
    solver_message: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise BatchValidationError(f"{label}: {actual!r} != {expected!r}")


def mission_constraints(mission: dict[str, Any]) -> tuple[int, int, int, list]:
    """Return the binary feasibility model without importing the paper solver."""
    n_variables = int(mission["n_variables"])
    lower, upper = (int(value) for value in mission["cardinality_bounds"])
    rows: list[tuple[dict[int, float], float]] = []

    for required, dependent in mission.get("precedence", []):
        rows.append(({int(dependent): 1.0, int(required): -1.0}, 0.0))
    for left, right in mission.get("mutex", []):
        rows.append(({int(left): 1.0, int(right): 1.0}, 1.0))

    if "groups_list" in mission:
        groups = [item["members"] for item in mission["groups_list"]]
    else:
        groups = mission.get("groups", {}).values()
    rows.extend(({int(member): 1.0 for member in members}, 1.0) for members in groups)

    rows.append((dict.fromkeys(range(n_variables), 1.0), float(upper)))
    return n_variables, lower, upper, rows


def maximum_feasible_cardinality(
    mission: dict[str, Any], *, time_limit_sec: float = 60.0
) -> FeasibilityCertificate:
    """Independently maximize cardinality with SciPy/HiGHS MILP."""
    n_variables, lower, _upper, rows = mission_constraints(mission)
    matrix = lil_matrix((len(rows), n_variables), dtype=float)
    upper_bounds = np.empty(len(rows), dtype=float)
    for row_index, (coefficients, row_upper) in enumerate(rows):
        for variable, coefficient in coefficients.items():
            matrix[row_index, variable] = coefficient
        upper_bounds[row_index] = row_upper

    result = milp(
        c=-np.ones(n_variables, dtype=float),
        integrality=np.ones(n_variables, dtype=np.int8),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=LinearConstraint(
            matrix.tocsr(),
            lb=np.full(len(rows), -np.inf),
            ub=upper_bounds,
        ),
        options={"time_limit": time_limit_sec},
    )
    if not result.success or result.fun is None:
        raise BatchValidationError(
            f"independent feasibility MILP failed: status={result.status}, "
            f"message={result.message}"
        )

    maximum = round(-float(result.fun))
    return FeasibilityCertificate(
        maximum_cardinality=maximum,
        required_minimum=lower,
        independently_infeasible=maximum < lower,
        solver_message=str(result.message),
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate_source_files(seed_dir: Path, spec: dict[str, Any]) -> None:
    for name, expected_hash in spec["source_sha256"].items():
        path = seed_dir / name
        if not path.is_file():
            raise BatchValidationError(f"missing source artifact: {path}")
        require_equal(sha256_file(path), expected_hash, f"SHA-256 {path}")

    require_equal(
        load_json(seed_dir / "run_state.json"),
        {"status": "complete"},
        f"run state seed {spec['seed']}",
    )
    provenance = load_json(seed_dir / "provenance.json")
    require_equal(
        provenance["top_level_commit"],
        spec["source_commit"],
        f"source commit seed {spec['seed']}",
    )


def validate_seed(
    spec: dict[str, Any], batch_root: Path, *, milp_time_limit_sec: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one seed against its exact Hub revision and source hashes."""
    from datasets import load_dataset  # noqa: PLC0415

    seed = int(spec["seed"])
    seed_dir = batch_root / f"seed{seed}"
    validate_source_files(seed_dir, spec)
    rows = load_jsonl(seed_dir / "certifications.jsonl")
    expected_count = int(spec["expected_count"])
    require_equal(len(rows), expected_count, f"certification row count seed {seed}")

    dataset = load_dataset(
        spec["dataset"],
        revision=spec["revision"],
        split=spec["split"],
    )
    require_equal(len(dataset), expected_count, f"dataset row count seed {seed}")

    mission_hashes: list[str] = []
    statuses: dict[str, int] = {}
    infeasibility_certificates: list[dict[str, Any]] = []
    seen_uuids: set[str] = set()

    for index, (item, row) in enumerate(zip(dataset, rows, strict=True)):
        mission = item["mission"]
        mission = json.loads(mission) if isinstance(mission, str) else mission
        mission_hash = sha256_text(canonical_json(mission))
        mission_hashes.append(mission_hash)
        uuid = str(item.get("uuid") or f"row-{index:06d}")
        if uuid in seen_uuids:
            raise BatchValidationError(f"duplicate UUID in seed {seed}: {uuid}")
        seen_uuids.add(uuid)

        require_equal(row["dataset_index"], index, f"dataset index seed {seed}")
        require_equal(row["uuid"], uuid, f"UUID seed {seed} index {index}")
        require_equal(
            row["mission_sha256"],
            mission_hash,
            f"mission hash seed {seed} index {index}",
        )
        require_equal(row["dataset"], spec["dataset"], f"dataset seed {seed}")
        require_equal(row["split"], spec["split"], f"split seed {seed}")

        status = str(row["status"])
        statuses[status] = statuses.get(status, 0) + 1
        if status in {"OPTIMAL", "FEASIBLE"}:
            if row["independently_feasible"] is not True:
                raise BatchValidationError(
                    f"unvalidated incumbent seed {seed} index {index}"
                )
            error = float(row["objective_validation_error"])
            if error > float(spec["maximum_objective_validation_error"]):
                raise BatchValidationError(
                    f"objective mismatch seed {seed} index {index}: {error}"
                )
            if float(row["objective"]) > float(row["best_bound"]) + 1e-6:
                raise BatchValidationError(
                    f"objective exceeds upper bound seed {seed} index {index}"
                )
        elif status == "INFEASIBLE":
            certificate = maximum_feasible_cardinality(
                mission, time_limit_sec=milp_time_limit_sec
            )
            if not certificate.independently_infeasible:
                raise BatchValidationError(
                    f"HiGHS found feasible cardinality for seed {seed} index {index}: "
                    f"max={certificate.maximum_cardinality}, "
                    f"required={certificate.required_minimum}"
                )
            infeasibility_certificates.append(
                {
                    "seed": seed,
                    "dataset_index": index,
                    "uuid": uuid,
                    "mission_sha256": mission_hash,
                    **certificate.__dict__,
                }
            )
        else:
            raise BatchValidationError(
                f"unsupported certification status seed {seed} index {index}: {status}"
            )

    content_hash = sha256_text("\n".join(mission_hashes))
    require_equal(
        content_hash,
        spec["dataset_content_sha256"],
        f"dataset content hash seed {seed}",
    )
    optimal_count = statuses.get("OPTIMAL", 0)
    infeasible_count = statuses.get("INFEASIBLE", 0)
    return (
        {
            "seed": seed,
            "dataset": spec["dataset"],
            "revision": spec["revision"],
            "row_count": len(rows),
            "dataset_content_sha256": content_hash,
            "status_counts": statuses,
            "certified_optimal_count": optimal_count,
            "exact_status_count": optimal_count + infeasible_count,
            "independently_reproved_infeasible_count": len(
                infeasibility_certificates
            ),
        },
        infeasibility_certificates,
    )


def validate_batch(
    manifest_path: Path,
    batch_root: Path,
    output_dir: Path,
    *,
    milp_time_limit_sec: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = load_json(manifest_path)
    require_equal(
        batch_root.name,
        manifest["source_batch"],
        "source batch directory",
    )

    summaries: list[dict[str, Any]] = []
    infeasibility_certificates: list[dict[str, Any]] = []
    for spec in manifest["datasets"]:
        summary, certificates = validate_seed(
            spec, batch_root, milp_time_limit_sec=milp_time_limit_sec
        )
        summaries.append(summary)
        infeasibility_certificates.extend(certificates)

    row_count = sum(item["row_count"] for item in summaries)
    optimal_count = sum(item["certified_optimal_count"] for item in summaries)
    exact_count = sum(item["exact_status_count"] for item in summaries)
    feasible_instances = row_count - len(infeasibility_certificates)
    report = {
        "status": "validated",
        "manifest_sha256": sha256_file(manifest_path),
        "source_batch": manifest["source_batch"],
        "source_commit": manifest["source_commit"],
        "row_count": row_count,
        "certified_optimal_count": optimal_count,
        "certified_optimal_rate_all_rows": optimal_count / row_count,
        "certified_optimal_rate_feasible_instances": (
            optimal_count / feasible_instances if feasible_instances else math.nan
        ),
        "exact_status_count": exact_count,
        "exact_status_rate": exact_count / row_count,
        "independently_reproved_infeasible_count": len(
            infeasibility_certificates
        ),
        "seeds": summaries,
    }
    (output_dir / "independent_infeasibility_certificates.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in infeasibility_certificates)
    )
    (output_dir / "validation_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--milp-time-limit-sec", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_batch(
        args.manifest,
        args.batch_root,
        args.output_dir,
        milp_time_limit_sec=args.milp_time_limit_sec,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
