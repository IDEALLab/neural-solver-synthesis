"""Freeze and validate N26-E10 artifacts before and after test evaluation."""

# ruff: noqa: PLR0912, PLR0915, PLR2004, TRY003

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from evaluation.sds.adaptive_repair import ProtocolError, sha256_text
from evaluation.sds.openai_adaptive_repair import load_openai_protocol

SEEDS = (101, 202, 303)


def sha256_file(path: Path) -> str:
    return sha256_text(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def freeze(protocol_path: Path, root: Path, slurm_job_id: str | None) -> None:
    protocol, _, _ = load_openai_protocol(protocol_path)
    destination = root / "freeze"
    if destination.exists():
        raise FileExistsError(f"immutable freeze already exists: {destination}")
    selected: dict[str, dict[str, Any]] = {}
    all_response_ids: list[str] = []
    observed_campaign_cost = 0.0
    source_commits: set[str] = set()
    initial_source_commits: set[str] = set()
    for seed in SEEDS:
        synthesis = root / "synthesis" / f"seed{seed}"
        run_path = synthesis / "run_manifest.json"
        solver_path = synthesis / "selected_solver.py"
        generations_path = synthesis / "generations.jsonl"
        summaries_path = synthesis / "candidate_summaries.jsonl"
        evaluations_path = synthesis / "development_evaluations.jsonl"
        for path in (
            run_path,
            solver_path,
            generations_path,
            summaries_path,
            evaluations_path,
        ):
            if not path.is_file():
                raise ProtocolError(f"missing E10 synthesis artifact: {path}")
        run = json.loads(run_path.read_text())
        expected = {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_text(protocol_path.read_text()),
            "seed": seed,
            "model": protocol["model"],
            "completion_count": 64,
            "development_execution_count": 2560,
            "test_outcomes_opened": False,
        }
        for key, value in expected.items():
            if run.get(key) != value:
                raise ProtocolError(f"E10 seed {seed} {key} drifted")
        if run.get("provenance", {}).get("dirty") is not False:
            raise ProtocolError(f"E10 seed {seed} ran from a dirty checkout")
        source_commit = str(run["provenance"]["top_level_commit"])
        initial_source_commit = str(
            run.get("paid_response_initial_source_commit", source_commit)
        )
        resume_execution_source_commit = str(
            run.get("resume_execution_source_commit", source_commit)
        )
        source_commits.add(source_commit)
        initial_source_commits.add(initial_source_commit)
        if resume_execution_source_commit != source_commit:
            raise ProtocolError(f"E10 seed {seed} resume provenance drifted")
        generations = read_jsonl(generations_path)
        summaries = read_jsonl(summaries_path)
        evaluations = read_jsonl(evaluations_path)
        if [row.get("completion_index") for row in generations] != list(range(64)):
            raise ProtocolError(f"E10 seed {seed} completion slots drifted")
        if len(summaries) != 64 or len(evaluations) != 2560:
            raise ProtocolError(f"E10 seed {seed} workload is incomplete")
        round_counts = {
            round_index: sum(int(row["round_index"]) == round_index for row in generations)
            for round_index in range(8)
        }
        if round_counts != dict.fromkeys(range(8), 8):
            raise ProtocolError(f"E10 seed {seed} round schedule drifted")
        response_ids = [str(row.get("api", {}).get("response_id", "")) for row in generations]
        if not all(response_ids) or len(set(response_ids)) != 64:
            raise ProtocolError(f"E10 seed {seed} Responses IDs are invalid")
        all_response_ids.extend(response_ids)
        observed_campaign_cost += float(run["observed_api_cost_usd"])
        solver_sha = sha256_file(solver_path)
        if solver_sha != run.get("selected_code_sha256"):
            raise ProtocolError(f"E10 seed {seed} selected solver hash drifted")
        code = solver_path.read_text()
        embedded_uuids = sorted(
            uuid for uuid in run["development_uuids"] if str(uuid) in code
        )
        if embedded_uuids:
            raise ProtocolError(
                f"E10 seed {seed} solver embeds development UUIDs: {embedded_uuids[:5]}"
            )
        seed_destination = destination / f"seed{seed}"
        seed_destination.mkdir(parents=True)
        shutil.copy2(solver_path, seed_destination / "selected_solver.py")
        selected[str(seed)] = {
            "selected_code_sha256": solver_sha,
            "source_run_manifest_sha256": sha256_file(run_path),
            "source_generations_sha256": sha256_file(generations_path),
            "response_count": len(response_ids),
            "observed_api_cost_usd": float(run["observed_api_cost_usd"]),
            "paid_response_initial_source_commit": initial_source_commit,
            "resume_execution_source_commit": resume_execution_source_commit,
        }
    if len(initial_source_commits) != 1:
        raise ProtocolError("E10 seeds did not begin from one source commit")
    if len(set(all_response_ids)) != 192:
        raise ProtocolError("E10 campaign Responses IDs are not globally unique")
    if observed_campaign_cost > float(protocol["cost_caps_usd"]["campaign"]):
        raise ProtocolError("E10 observed campaign cost exceeds the hard cap")
    freeze_manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_text(protocol_path.read_text()),
        "status": "complete",
        "seeds": list(SEEDS),
        "source_commit": (
            next(iter(source_commits)) if len(source_commits) == 1 else None
        ),
        "resume_execution_source_commits": sorted(source_commits),
        "paid_response_initial_source_commit": next(iter(initial_source_commits)),
        "slurm_job_id": slurm_job_id,
        "frozen_unix": time.time(),
        "test_outcomes_opened": False,
        "response_count": len(all_response_ids),
        "observed_campaign_cost_usd": observed_campaign_cost,
        "selected": selected,
    }
    (destination / "freeze_manifest.json").write_text(
        json.dumps(freeze_manifest, indent=2) + "\n"
    )


def register_test(
    protocol_path: Path, root: Path, seed: int, slurm_job_id: str | None
) -> None:
    protocol, _, _ = load_openai_protocol(protocol_path)
    freeze_path = root / "freeze" / "freeze_manifest.json"
    freeze_manifest = json.loads(freeze_path.read_text())
    if freeze_manifest.get("test_outcomes_opened") is not False:
        raise ProtocolError("E10 solver freeze did not precede test access")
    solver_path = root / "freeze" / f"seed{seed}" / "selected_solver.py"
    expected_sha = freeze_manifest["selected"][str(seed)]["selected_code_sha256"]
    if sha256_file(solver_path) != expected_sha:
        raise ProtocolError("E10 frozen solver hash drifted")
    metrics_path = root / "test" / f"seed{seed}" / "test-evaluation" / "metrics_final.csv"
    metadata_path = (
        root
        / "test"
        / f"seed{seed}"
        / "test-evaluation"
        / "experiment_metadata.json"
    )
    with metrics_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(protocol["test_evaluation"]["rows_per_seed"]):
        raise ProtocolError("E10 test row count drifted")
    uuids = {str(row["uuid"]) for row in rows}
    if len(uuids) != len(rows):
        raise ProtocolError("E10 test UUIDs are not unique")
    expected_method = protocol.get(
        "method_name", "GPT-5.4-mini Adaptive Repair (64 completions)"
    )
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("method_name") != expected_method:
        raise ProtocolError(
            f"E10 test method drifted: {metadata.get('method_name')!r}"
        )
    if int(metadata.get("seed", -1)) != seed:
        raise ProtocolError("E10 test metadata seed drifted")
    if metadata.get("model") != "fixed-code":
        raise ProtocolError("E10 test metadata model drifted")
    if metadata.get("fixed_code_file") != str(solver_path):
        raise ProtocolError("E10 test metadata solver path drifted")
    payload = {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "slurm_job_id": slurm_job_id,
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "selected_code_sha256": expected_sha,
        "metrics_sha256": sha256_file(metrics_path),
        "experiment_metadata_sha256": sha256_file(metadata_path),
        "test_rows": len(rows),
        "unique_test_uuids": len(uuids),
        "registered_unix": time.time(),
    }
    output = root / "test" / f"seed{seed}" / "test_manifest.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--root", type=Path, required=True)
    test_parser = subparsers.add_parser("register-test")
    test_parser.add_argument("--protocol", type=Path, required=True)
    test_parser.add_argument("--root", type=Path, required=True)
    test_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        freeze(args.protocol, args.root, os.environ.get("SLURM_JOB_ID"))
    else:
        register_test(
            args.protocol, args.root, args.seed, os.environ.get("SLURM_JOB_ID")
        )


if __name__ == "__main__":
    main()
