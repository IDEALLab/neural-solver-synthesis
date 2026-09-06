#!/usr/bin/env python3
"""Validate the immutable N26-E4 result registration against raw artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(root: Path, hashes: dict[str, str]) -> None:
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"registered artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"artifact hash drift for {path}: {actual} != {expected}")


def require_close(actual: Any, expected: Any, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"registered result drift for {label}: {actual} != {expected}")


def _row_for(path: Path, key: str, value: str) -> dict[str, str]:
    with path.open(newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row[key] == value]
    if len(matches) != 1:
        raise ValueError(f"expected one {key}={value} row in {path}, found {len(matches)}")
    return matches[0]


def validate_registration(path: Path) -> dict[str, Any]:
    registration = json.loads(path.read_text())
    if registration.get("status") != "complete":
        raise ValueError("adaptive result registration is not complete")
    synthesis_root = Path(registration["synthesis_root"])
    test_root = Path(registration["test_root"])
    aggregation_root = Path(registration["aggregation_root"])
    verify_hashes(test_root, registration["test_files_sha256"])
    verify_hashes(aggregation_root, registration["aggregation_files_sha256"])

    integrity = registration["integrity"]
    for seed in (101, 202, 303):
        seed_key = str(seed)
        solver = synthesis_root / f"seed{seed}" / "synthesis" / "selected_solver.py"
        if sha256_file(solver) != registration["selected_solver_sha256"][seed_key]:
            raise ValueError(f"seed {seed} selected solver hash drift")
        reevaluation = json.loads(
            (test_root / f"seed{seed}" / "reevaluation_manifest.json").read_text()
        )
        if reevaluation["source_solver_sha256"] != sha256_file(solver):
            raise ValueError(f"seed {seed} reevaluated a different solver")
        if any(reevaluation[key] for key in ("regenerated", "reselected")):
            raise ValueError(f"seed {seed} was regenerated or reselected")
        if int(reevaluation["execution_repeats"]) != int(integrity["execution_repeats"]):
            raise ValueError(f"seed {seed} execution-repeat drift")
        metrics_path = test_root / f"seed{seed}" / "test-evaluation" / "metrics_final.csv"
        with metrics_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != int(integrity["test_rows_per_seed"]):
            raise ValueError(f"seed {seed} test row count drift")
        uuids = [row["uuid"] for row in rows]
        if len(uuids) != len(set(uuids)):
            raise ValueError(f"seed {seed} duplicate test UUIDs")

    result = registration["result"]
    summary = aggregation_root / "certified_summary.csv"
    adaptive = _row_for(summary, "Method", "Adaptive Repair (64 completions)")
    frozen = _row_for(summary, "Method", "Frozen Hero")
    summary_checks = (
        (adaptive, "PassRateFeasibleBenchmarkMean", "adaptive_repair_pass_rate_feasible_benchmark_mean"),
        (adaptive, "PassRateFeasibleBenchmarkStd", "adaptive_repair_pass_rate_feasible_benchmark_sample_stddev"),
        (adaptive, "CertifiedOptimalGapLowerMean", "adaptive_repair_cpsat_optimal_gap_interval_mean", 0),
        (adaptive, "CertifiedOptimalGapUpperMean", "adaptive_repair_cpsat_optimal_gap_interval_mean", 1),
        (adaptive, "AllFeasibleGapLowerMean", "adaptive_repair_all_feasible_gap_interval_mean", 0),
        (adaptive, "AllFeasibleGapUpperMean", "adaptive_repair_all_feasible_gap_interval_mean", 1),
        (frozen, "PassRateFeasibleBenchmarkMean", "frozen_hero_pass_rate_feasible_benchmark_mean"),
        (frozen, "CertifiedOptimalGapLowerMean", "frozen_hero_cpsat_optimal_gap_interval_mean", 0),
        (frozen, "CertifiedOptimalGapUpperMean", "frozen_hero_cpsat_optimal_gap_interval_mean", 1),
    )
    for check in summary_checks:
        row, column, result_key, *index = check
        expected = result[result_key] if not index else result[result_key][index[0]]
        require_close(row[column], expected, result_key)

    paired = _row_for(
        aggregation_root / "paired_frozen_hero_comparisons.csv",
        "ComparatorMethod",
        "Adaptive Repair (64 completions)",
    )
    paired_checks = {
        "PairedPassDifference": "paired_frozen_hero_minus_adaptive_pass_difference",
        "PairedPassDifferenceCI025": ("paired_pass_difference_95ci", 0),
        "PairedPassDifferenceCI975": ("paired_pass_difference_95ci", 1),
        "PairedCertifiedGapDifference": "paired_frozen_hero_minus_adaptive_gap_difference",
        "PairedCertifiedGapDifferenceCI025": ("paired_gap_difference_95ci", 0),
        "PairedCertifiedGapDifferenceCI975": ("paired_gap_difference_95ci", 1),
    }
    for column, result_spec in paired_checks.items():
        if isinstance(result_spec, tuple):
            expected = result[result_spec[0]][result_spec[1]]
        else:
            expected = result[result_spec]
        require_close(paired[column], expected, column)

    return {
        "status": "valid",
        "protocol_id": registration["protocol_id"],
        "test_rows": 3 * int(integrity["test_rows_per_seed"]),
        "selected_solver_sha256": registration["selected_solver_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_registration(args.manifest), indent=2))


if __name__ == "__main__":
    main()
