"""Validate and register the exact historical Hypothesize experiment."""

# ruff: noqa: PLR0913, TRY003

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

METHODS = {
    "control": "Historical exact Hero control",
    "ablation": "Historical exact Hero without Hypothesize",
}
SEEDS = (101, 202, 303)
ROWS_PER_SEED = 1000
REPRODUCTION_TOLERANCE = 0.02
CORRECTION_PROTOCOL_ID = "N26-E11-v3-seed101-90-correction"
CORRECTION_SEED = 101
CORRECTION_FREEZE_STATUS = (
    "seed101_pair_frozen_before_its_test_prior_seeds_preserved"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register_evaluation(
    root: Path,
    condition: str,
    seed: int,
    training_job_id: str,
    evaluation_job_id: str,
    test_started_unix: float,
) -> dict:
    """Register an already-completed immutable evaluation after validation."""
    if condition not in METHODS or seed not in SEEDS:
        raise ValueError("invalid historical evaluation condition or seed")
    freeze_path = root / "freeze" / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("status") != "all_six_frozen_before_test":
        raise ValueError("all-six checkpoint freeze is not valid")
    expected_training_job = str(freeze["training_job_ids"][condition][str(seed)])
    if str(training_job_id) != expected_training_job:
        raise ValueError("historical training job mapping drift")
    if float(test_started_unix) < float(freeze["created_unix"]):
        raise ValueError("test evaluation preceded the global freeze")

    seed_root = root / "evaluation" / condition / f"seed{seed}"
    output = seed_root / "run_manifest.json"
    metrics_path = seed_root / "evaluation" / "metrics_final.csv"
    metadata_path = seed_root / "evaluation" / "experiment_metadata.json"
    generations_path = seed_root / "generations.jsonl"
    audit_rows_path = seed_root / "program-audit" / "row_audit.jsonl"
    audit_summary_path = seed_root / "program-audit" / "summary.json"
    checkpoint_validation_path = seed_root / "checkpoint_validation.json"

    with metrics_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    uuids = [str(row["uuid"]) for row in rows]
    if len(rows) != ROWS_PER_SEED or len(set(uuids)) != ROWS_PER_SEED:
        raise ValueError("historical evaluation key or row-count drift")
    generations = [line for line in generations_path.read_text().splitlines() if line]
    audit_rows = [line for line in audit_rows_path.read_text().splitlines() if line]
    if len(generations) != ROWS_PER_SEED or len(audit_rows) != ROWS_PER_SEED:
        raise ValueError("historical generation or audit workload is incomplete")

    method = METHODS[condition]
    metadata = json.loads(metadata_path.read_text())
    expected_metadata = {
        "method_name": method,
        "seed": seed,
        "job_id": str(training_job_id),
        "model": "qwen2.5-coder-14b",
        "training_scheme": "grpo",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"historical evaluation metadata drift: {key}")

    payload: dict = {
        "status": "complete",
        "condition": condition,
        "seed": seed,
        "method": method,
        "training_job_id": str(training_job_id),
        "evaluation_job_id": str(evaluation_job_id),
        "test_started_unix": float(test_started_unix),
        "freeze_created_unix": float(freeze["created_unix"]),
        "registered_unix": time.time(),
        "rows": len(rows),
        "unique_uuids": len(set(uuids)),
        "generations_sha256": _sha256(generations_path),
        "metrics_sha256": _sha256(metrics_path),
        "experiment_metadata_sha256": _sha256(metadata_path),
        "program_audit_rows_sha256": _sha256(audit_rows_path),
        "program_audit_summary_sha256": _sha256(audit_summary_path),
        "checkpoint_validation_sha256": _sha256(checkpoint_validation_path),
        "freeze_manifest_sha256": _sha256(freeze_path),
    }
    if output.exists():
        existing = json.loads(output.read_text())
        payload["registered_unix"] = existing.get("registered_unix")
        if existing != payload:
            raise ValueError("immutable historical evaluation registration drift")
        return existing
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def register_correction_evaluation(
    root: Path,
    condition: str,
    training_job_id: str,
    evaluation_job_id: str,
    test_started_unix: float,
) -> dict:
    """Register one seed-101 correction after its matched pair was frozen."""
    if condition not in METHODS:
        raise ValueError("invalid historical correction condition")
    freeze_path = root / "freeze" / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("status") != CORRECTION_FREEZE_STATUS:
        raise ValueError("seed-101 correction freeze is not valid")
    expected_training_job = str(freeze["training_job_ids"][condition])
    if str(training_job_id) != expected_training_job:
        raise ValueError("correction training job mapping drift")
    if float(test_started_unix) < float(freeze["created_unix"]):
        raise ValueError("correction evaluation preceded the pair freeze")

    seed_root = root / "evaluation" / condition / "seed101"
    output = seed_root / "run_manifest.json"
    metrics_path = seed_root / "evaluation" / "metrics_final.csv"
    metadata_path = seed_root / "evaluation" / "experiment_metadata.json"
    generations_path = seed_root / "generations.jsonl"
    audit_rows_path = seed_root / "program-audit" / "row_audit.jsonl"
    audit_summary_path = seed_root / "program-audit" / "summary.json"
    checkpoint_validation_path = seed_root / "checkpoint_validation.json"

    with metrics_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    uuids = [str(row["uuid"]) for row in rows]
    generations = [line for line in generations_path.read_text().splitlines() if line]
    audit_rows = [line for line in audit_rows_path.read_text().splitlines() if line]
    if (
        len(rows) != ROWS_PER_SEED
        or len(set(uuids)) != ROWS_PER_SEED
        or len(generations) != ROWS_PER_SEED
        or len(audit_rows) != ROWS_PER_SEED
    ):
        raise ValueError("correction evaluation workload is incomplete")

    method = METHODS[condition]
    metadata = json.loads(metadata_path.read_text())
    expected_metadata = {
        "method_name": method,
        "seed": 101,
        "job_id": str(training_job_id),
        "model": "qwen2.5-coder-14b",
        "training_scheme": "grpo",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"correction evaluation metadata drift: {key}")

    payload: dict = {
        "protocol_id": CORRECTION_PROTOCOL_ID,
        "status": "complete",
        "condition": condition,
        "seed": 101,
        "method": method,
        "training_job_id": str(training_job_id),
        "evaluation_job_id": str(evaluation_job_id),
        "test_started_unix": float(test_started_unix),
        "freeze_created_unix": float(freeze["created_unix"]),
        "registered_unix": time.time(),
        "rows": len(rows),
        "unique_uuids": len(set(uuids)),
        "generations_sha256": _sha256(generations_path),
        "metrics_sha256": _sha256(metrics_path),
        "experiment_metadata_sha256": _sha256(metadata_path),
        "program_audit_rows_sha256": _sha256(audit_rows_path),
        "program_audit_summary_sha256": _sha256(audit_summary_path),
        "checkpoint_validation_sha256": _sha256(checkpoint_validation_path),
        "freeze_manifest_sha256": _sha256(freeze_path),
    }
    if output.exists():
        existing = json.loads(output.read_text())
        payload["registered_unix"] = existing.get("registered_unix")
        if existing != payload:
            raise ValueError("immutable correction evaluation registration drift")
        return existing
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def prepare_correction_composite(root: Path, prior_root: Path) -> dict:
    """Compose corrected seed 101 with immutable v2 seed 202/303 evaluations."""
    composite = root / "composite" / "evaluation"
    if composite.exists():
        raise ValueError("immutable correction composite already exists")
    links: dict[str, dict] = {}
    for condition in METHODS:
        for seed in SEEDS:
            source_root = root if seed == CORRECTION_SEED else prior_root
            source = source_root / "evaluation" / condition / f"seed{seed}"
            run_path = source / "run_manifest.json"
            run = json.loads(run_path.read_text())
            if (
                run.get("condition") != condition
                or int(run.get("seed", -1)) != seed
                or int(run.get("rows", -1)) != ROWS_PER_SEED
            ):
                raise ValueError("correction composite source manifest drift")
            metrics = source / "evaluation" / "metrics_final.csv"
            metadata = source / "evaluation" / "experiment_metadata.json"
            target = composite / condition / f"seed{seed}" / "evaluation"
            target.mkdir(parents=True)
            (target / metrics.name).symlink_to(metrics.resolve())
            (target / metadata.name).symlink_to(metadata.resolve())
            links[str(target.parent.relative_to(root))] = {
                "source": str(source.resolve()),
                "run_manifest_sha256": _sha256(run_path),
                "metrics_sha256": _sha256(metrics),
                "experiment_metadata_sha256": _sha256(metadata),
            }
    payload = {
        "protocol_id": CORRECTION_PROTOCOL_ID,
        "status": "complete",
        "seed101_source": "corrective step-90 pair",
        "seed202_303_source": "immutable N26-E11-v2 evaluations",
        "links": links,
    }
    output = root / "composite" / "composite_manifest.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def register_correction(
    root: Path, prior_root: Path, certified_root: Path, output: Path
) -> dict:
    """Finalize the corrected three-seed result without rewriting v2 history."""
    freeze_path = root / "freeze" / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("status") != CORRECTION_FREEZE_STATUS:
        raise ValueError("seed-101 correction freeze is not valid")
    prior_freeze_path = prior_root / "freeze" / "freeze_manifest.json"
    prior_freeze = json.loads(prior_freeze_path.read_text())
    if prior_freeze.get("status") != "all_six_frozen_before_test":
        raise ValueError("prior historical freeze is not valid")
    if _sha256(prior_freeze_path) != freeze["prior_freeze_manifest_sha256"]:
        raise ValueError("prior historical freeze hash drift")

    files: dict[str, str] = {}
    for condition, method in METHODS.items():
        for seed in SEEDS:
            source_root = root if seed == CORRECTION_SEED else prior_root
            seed_root = source_root / "evaluation" / condition / f"seed{seed}"
            run_path = seed_root / "run_manifest.json"
            run = json.loads(run_path.read_text())
            reference_freeze = freeze if seed == CORRECTION_SEED else prior_freeze
            if float(run["test_started_unix"]) < float(reference_freeze["created_unix"]):
                raise ValueError("evaluation preceded its governing freeze")
            if (
                run.get("method") != method
                or int(run.get("seed", -1)) != seed
                or int(run.get("rows", -1)) != ROWS_PER_SEED
            ):
                raise ValueError("corrected historical evaluation manifest drift")
            metrics_path = seed_root / "evaluation" / "metrics_final.csv"
            rows = list(csv.DictReader(metrics_path.open()))
            keys = {row["uuid"] for row in rows}
            if len(rows) != ROWS_PER_SEED or len(keys) != ROWS_PER_SEED:
                raise ValueError("corrected historical evaluation key drift")
            if _sha256(metrics_path) != run["metrics_sha256"]:
                raise ValueError("corrected historical metrics hash drift")
            files[f"{condition}/seed{seed}"] = _sha256(metrics_path)

    summary_rows = {
        row["Method"]: row
        for row in csv.DictReader((certified_root / "certified_summary.csv").open())
    }
    for method in ("Ours (Hero)", *METHODS.values()):
        if method not in summary_rows:
            raise ValueError(f"missing certified summary method: {method}")
    historical = summary_rows["Ours (Hero)"]
    control = summary_rows[METHODS["control"]]
    pass_delta = abs(
        float(control["PassRateFeasibleBenchmarkMean"])
        - float(historical["PassRateFeasibleBenchmarkMean"])
    )
    gap_delta = abs(
        float(control["CertifiedOptimalGapMean"])
        - float(historical["CertifiedOptimalGapMean"])
    )
    gate = pass_delta <= REPRODUCTION_TOLERANCE and gap_delta <= REPRODUCTION_TOLERANCE
    payload = {
        "protocol_id": CORRECTION_PROTOCOL_ID,
        "status": "complete",
        "rows_per_condition": 3000,
        "seeds": list(SEEDS),
        "seed101_pair_frozen_before_its_test": True,
        "prior_seed_pairs_preserved_from_v2": [202, 303],
        "all_six_frozen_before_single_global_test_opening": False,
        "global_freeze_qualification": (
            "Seed 202/303 outcomes were already open under their immutable v2 freeze "
            "when the artifact-mismatch correction for seed 101 was declared."
        ),
        "metrics_sha256": files,
        "historical_reproduction_gate": {
            "passes": gate,
            "pass_rate_absolute_delta": pass_delta,
            "certified_gap_absolute_delta": gap_delta,
            "absolute_tolerance_each": REPRODUCTION_TOLERANCE,
            "causal_interpretation_allowed": gate,
        },
        "freeze_manifest_sha256": _sha256(freeze_path),
        "prior_freeze_manifest_sha256": _sha256(prior_freeze_path),
        "certified_summary_sha256": _sha256(certified_root / "certified_summary.csv"),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def register(root: Path, certified_root: Path, output: Path) -> dict:
    freeze_path = root / "freeze" / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["status"] != "all_six_frozen_before_test":
        raise ValueError("all-six checkpoint freeze is not valid")
    files = {}
    for condition, method in METHODS.items():
        for seed in SEEDS:
            seed_root = root / "evaluation" / condition / f"seed{seed}"
            run_path = seed_root / "run_manifest.json"
            run = json.loads(run_path.read_text())
            if run["test_started_unix"] < freeze["created_unix"]:
                raise ValueError("test evaluation preceded the global freeze")
            if (
                run["method"] != method
                or run["seed"] != seed
                or run["rows"] != ROWS_PER_SEED
            ):
                raise ValueError("historical evaluation manifest drift")
            metrics_path = seed_root / "evaluation" / "metrics_final.csv"
            rows = list(csv.DictReader(metrics_path.open()))
            keys = {row["uuid"] for row in rows}
            if len(rows) != ROWS_PER_SEED or len(keys) != ROWS_PER_SEED:
                raise ValueError("historical evaluation key or row-count drift")
            metadata_path = seed_root / "evaluation" / "experiment_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("method_name") != method or metadata.get("seed") != seed:
                raise ValueError("historical evaluation metadata drift")
            if _sha256(metrics_path) != run["metrics_sha256"]:
                raise ValueError("historical metrics hash drift")
            if _sha256(metadata_path) != run["experiment_metadata_sha256"]:
                raise ValueError("historical metadata hash drift")
            files[str(metrics_path.relative_to(root))] = _sha256(metrics_path)

    summary_rows = {row["Method"]: row for row in csv.DictReader((certified_root / "certified_summary.csv").open())}
    for method in ("Ours (Hero)", *METHODS.values()):
        if method not in summary_rows:
            raise ValueError(f"missing certified summary method: {method}")
    historical = summary_rows["Ours (Hero)"]
    control = summary_rows[METHODS["control"]]
    pass_delta = abs(
        float(control["PassRateFeasibleBenchmarkMean"])
        - float(historical["PassRateFeasibleBenchmarkMean"])
    )
    gap_delta = abs(
        float(control["CertifiedOptimalGapMean"])
        - float(historical["CertifiedOptimalGapMean"])
    )
    gate = (
        pass_delta <= REPRODUCTION_TOLERANCE
        and gap_delta <= REPRODUCTION_TOLERANCE
    )
    payload = {
        "protocol_id": "N26-E11-v2-historical-exact-hypothesize",
        "status": "complete",
        "all_six_frozen_before_test": True,
        "rows_per_condition": 3000,
        "seeds": list(SEEDS),
        "metrics_sha256": files,
        "historical_reproduction_gate": {
            "passes": gate,
            "pass_rate_absolute_delta": pass_delta,
            "certified_gap_absolute_delta": gap_delta,
            "absolute_tolerance_each": REPRODUCTION_TOLERANCE,
            "causal_interpretation_allowed": gate,
        },
        "freeze_manifest_sha256": _sha256(freeze_path),
        "certified_summary_sha256": _sha256(certified_root / "certified_summary.csv"),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("register-run")
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--condition", choices=tuple(METHODS), required=True)
    run_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run_parser.add_argument("--training-job-id", required=True)
    run_parser.add_argument("--evaluation-job-id", required=True)
    run_parser.add_argument("--test-started-unix", type=float, required=True)
    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--root", type=Path, required=True)
    final_parser.add_argument("--certified-root", type=Path, required=True)
    final_parser.add_argument("--output", type=Path, required=True)
    correction_run_parser = subparsers.add_parser("register-correction-run")
    correction_run_parser.add_argument("--root", type=Path, required=True)
    correction_run_parser.add_argument(
        "--condition", choices=tuple(METHODS), required=True
    )
    correction_run_parser.add_argument("--training-job-id", required=True)
    correction_run_parser.add_argument("--evaluation-job-id", required=True)
    correction_run_parser.add_argument(
        "--test-started-unix", type=float, required=True
    )
    composite_parser = subparsers.add_parser("prepare-correction-composite")
    composite_parser.add_argument("--root", type=Path, required=True)
    composite_parser.add_argument("--prior-root", type=Path, required=True)
    correction_parser = subparsers.add_parser("finalize-correction")
    correction_parser.add_argument("--root", type=Path, required=True)
    correction_parser.add_argument("--prior-root", type=Path, required=True)
    correction_parser.add_argument("--certified-root", type=Path, required=True)
    correction_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "register-run":
        register_evaluation(
            args.root,
            args.condition,
            args.seed,
            args.training_job_id,
            args.evaluation_job_id,
            args.test_started_unix,
        )
    elif args.command == "register-correction-run":
        register_correction_evaluation(
            args.root,
            args.condition,
            args.training_job_id,
            args.evaluation_job_id,
            args.test_started_unix,
        )
    elif args.command == "prepare-correction-composite":
        prepare_correction_composite(args.root, args.prior_root)
    elif args.command == "finalize-correction":
        register_correction(
            args.root, args.prior_root, args.certified_root, args.output
        )
    else:
        register(args.root, args.certified_root, args.output)


if __name__ == "__main__":
    main()
