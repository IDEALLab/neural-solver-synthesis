#!/usr/bin/env python3
"""Validate and register the complete content-clean E4/E8 control artifacts."""

# ruff: noqa: PLR0912, PLR0913, PLR0915, PLR2004, TRY003

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SEEDS = (101, 202, 303)
EXPECTED_TEST_ROWS = 1000
EXPECTED_ADAPTIVE_COMPLETIONS = 64
EXPECTED_DEVELOPMENT_ROWS = 40
EXPECTED_ADAPTIVE_EXECUTIONS = 2560
EXPECTED_TEST_REPEATS = 3
EXPECTED_ALLOCATED_GPUS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing control artifact: {path}")


def validate_scheduler(
    scheduler: dict[str, Any], submission: dict[str, Any], kind: str
) -> dict[int, dict[str, Any]]:
    records = scheduler.get(kind)
    if not isinstance(records, dict) or set(records) != {str(seed) for seed in SEEDS}:
        raise ValueError(f"scheduler {kind} seeds drifted")
    submission_key = "adaptive_jobs" if kind == "adaptive" else "frozen_hero_jobs"
    submitted = submission[submission_key]
    result: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        record = records[str(seed)]
        if str(record.get("job_id")) != str(submitted[str(seed)]):
            raise ValueError(f"{kind} seed {seed} scheduler/submission mismatch")
        if record.get("state") != "COMPLETED" or record.get("exit_code") != "0:0":
            raise ValueError(f"{kind} seed {seed} did not complete cleanly")
        if int(record.get("allocated_gpus", 0)) != EXPECTED_ALLOCATED_GPUS:
            raise ValueError(f"{kind} seed {seed} must record four allocated GPUs")
        if float(record.get("elapsed_seconds", 0)) <= 0:
            raise ValueError(f"{kind} seed {seed} elapsed time is invalid")
        result[seed] = record
    return result


def load_scheduler_evidence(
    path: Path, submission: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    if path.suffix == ".json":
        return json.loads(path.read_text())
    records: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(
            stream,
            fieldnames=[
                "job_id",
                "state",
                "elapsed_seconds",
                "exit_code",
                "alloc_tres",
            ],
            delimiter="|",
        )
        for row in reader:
            if not row["job_id"]:
                continue
            match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", row["alloc_tres"])
            if not match:
                raise ValueError(f"scheduler job {row['job_id']} lacks gres/gpu")
            records[row["job_id"]] = {
                "job_id": row["job_id"],
                "state": row["state"].split("+", 1)[0],
                "elapsed_seconds": int(row["elapsed_seconds"]),
                "exit_code": row["exit_code"],
                "allocated_gpus": int(match.group(1)),
                "alloc_tres": row["alloc_tres"],
            }
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for kind, submission_key in (
        ("adaptive", "adaptive_jobs"),
        ("frozen_hero", "frozen_hero_jobs"),
    ):
        result[kind] = {}
        for seed in SEEDS:
            job_id = str(submission[submission_key][str(seed)])
            if job_id not in records:
                raise ValueError(f"scheduler evidence lacks job {job_id}")
            result[kind][str(seed)] = records[job_id]
    return result


def validate_metrics(path: Path) -> dict[str, Any]:
    require_file(path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    uuids = [str(row["uuid"]) for row in rows]
    if len(rows) != EXPECTED_TEST_ROWS or len(set(uuids)) != EXPECTED_TEST_ROWS:
        raise ValueError(f"{path} does not contain 1,000 unique test rows")
    required = {"uuid", "feasible", "llm_score", "execution_time"}
    if not required.issubset(rows[0]):
        raise ValueError(f"{path} lacks required metric columns")
    return {
        "rows": len(rows),
        "unique_uuids": len(set(uuids)),
        "uuid_set_sha256": hashlib.sha256(
            ("\n".join(sorted(uuids)) + "\n").encode()
        ).hexdigest(),
    }


def metric_uuids(path: Path) -> set[str]:
    with path.open(newline="") as stream:
        return {str(row["uuid"]) for row in csv.DictReader(stream)}


def validate_prompt_hash(row: dict[str, Any]) -> None:
    prompt = str(row["prompt"])
    actual_prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    if row.get("prompt_sha256") != actual_prompt_sha:
        raise ValueError("adaptive repair prompt hash drifted")


def validate_exact_requirement_blocks(
    row: dict[str, Any], development_uuids: set[str]
) -> tuple[str, ...]:
    """Require every declared exact-requirement block to be complete JSON."""
    validate_prompt_hash(row)
    prompt = str(row["prompt"])
    parsed: list[str] = []
    prefix_pattern = re.compile(r"^- ([^ ]+) exact_development_requirements=(.*)$")
    for line in prompt.splitlines():
        match = prefix_pattern.match(line)
        if match is None:
            continue
        uuid, payload = match.groups()
        payload = payload.removesuffix("<|im_end|>")
        if uuid not in development_uuids:
            raise ValueError(f"adaptive repair prompt uses unknown UUID: {uuid}")
        try:
            requirements = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"adaptive repair prompt has malformed requirements for {uuid}"
            ) from error
        required_fields = {
            "n_variables",
            "cardinality_bounds",
            "precedence",
            "mutex",
            "groups",
        }
        if not isinstance(requirements, dict) or not required_fields.issubset(
            requirements
        ):
            raise ValueError(
                f"adaptive repair prompt has incomplete requirements for {uuid}"
            )
        parsed.append(uuid)
    declared = tuple(str(uuid) for uuid in row["exact_requirement_uuids"])
    if tuple(parsed) != declared:
        raise ValueError("adaptive repair exact requirement declarations drifted")
    return declared


def validate_adaptive_seed(
    root: Path,
    seed: int,
    protocol: dict[str, Any],
    protocol_sha256: str,
    manifest_sha256: str,
    scheduler: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    seed_root = root / f"seed{seed}"
    paths = {
        "references": seed_root / "development_references.jsonl",
        "run_manifest": seed_root / "synthesis/run_manifest.json",
        "generations": seed_root / "synthesis/generations.jsonl",
        "summaries": seed_root / "synthesis/candidate_summaries.jsonl",
        "evaluations": seed_root / "synthesis/development_evaluations.jsonl",
        "selected_solver": seed_root / "synthesis/selected_solver.py",
        "test_metrics": seed_root / "test-evaluation/metrics_final.csv",
        "timing": seed_root / "timing_summary.json",
    }
    for path in paths.values():
        require_file(path)
    run = json.loads(paths["run_manifest"].read_text())
    expected_run = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "frozen_manifest_sha256": manifest_sha256,
        "seed": seed,
        "completion_count": EXPECTED_ADAPTIVE_COMPLETIONS,
        "development_execution_count": EXPECTED_ADAPTIVE_EXECUTIONS,
        "slurm_job_id": str(scheduler["job_id"]),
        "model": protocol["model"],
    }
    for key, expected in expected_run.items():
        if run.get(key) != expected:
            raise ValueError(f"adaptive seed {seed} {key} drifted")
    if not run.get("resolved_model_commit"):
        raise ValueError(f"adaptive seed {seed} lacks a resolved model commit")
    if run.get("provenance", {}).get("dirty") is not False:
        raise ValueError(f"adaptive seed {seed} source was dirty")

    generations = read_jsonl(paths["generations"])
    if [row.get("completion_index") for row in generations] != list(
        range(EXPECTED_ADAPTIVE_COMPLETIONS)
    ):
        raise ValueError(f"adaptive seed {seed} completion indices drifted")
    expected_rounds = dict.fromkeys(range(8), 8)
    if Counter(int(row["round_index"]) for row in generations) != expected_rounds:
        raise ValueError(f"adaptive seed {seed} round schedule drifted")
    available_prompt_tokens = int(protocol["sampling"]["max_model_len"]) - int(
        protocol["sampling"]["max_tokens"]
    )
    prompt_tokens = [int(row["prompt_token_count"]) for row in generations]
    if min(prompt_tokens) <= 0 or max(prompt_tokens) > available_prompt_tokens:
        raise ValueError(f"adaptive seed {seed} prompt exceeded native context")
    if any(
        not row.get("prompt") or not row.get("generated_text") for row in generations
    ):
        raise ValueError(f"adaptive seed {seed} lacks full prompt/output provenance")
    for row in generations:
        validate_prompt_hash(row)
    repair_generations = [row for row in generations if int(row["round_index"]) > 0]
    development_rows = read_jsonl(paths["references"])
    development_uuids = {str(row["uuid"]) for row in development_rows}
    if (
        len(development_rows) != EXPECTED_DEVELOPMENT_ROWS
        or len(development_uuids) != EXPECTED_DEVELOPMENT_ROWS
    ):
        raise ValueError(f"adaptive seed {seed} lacks 40 unique development UUIDs")
    for row in repair_generations:
        requirement_uuids = row.get("exact_requirement_uuids")
        requirement_count = int(row.get("exact_requirement_count", -1))
        if (
            not isinstance(requirement_uuids, list)
            or requirement_count != len(requirement_uuids)
            or not 1 <= requirement_count <= 6
            or len(set(requirement_uuids)) != requirement_count
        ):
            raise ValueError(
                f"adaptive seed {seed} has invalid context-budgeted requirements"
            )
        validate_exact_requirement_blocks(row, development_uuids)
    exact_requirements_by_round: dict[int, tuple[str, ...]] = {}
    for round_index in range(1, 8):
        round_prompts = {
            (row["prompt_sha256"], tuple(row["exact_requirement_uuids"]))
            for row in repair_generations
            if int(row["round_index"]) == round_index
        }
        if len(round_prompts) != 1:
            raise ValueError(
                f"adaptive seed {seed} repair round {round_index} prompt drifted"
            )
        _, requirement_uuids = next(iter(round_prompts))
        exact_requirements_by_round[round_index] = requirement_uuids

    summaries = read_jsonl(paths["summaries"])
    evaluations = read_jsonl(paths["evaluations"])
    if len(summaries) != EXPECTED_ADAPTIVE_COMPLETIONS:
        raise ValueError(f"adaptive seed {seed} summary count drifted")
    if len(evaluations) != EXPECTED_ADAPTIVE_EXECUTIONS:
        raise ValueError(f"adaptive seed {seed} execution count drifted")
    evaluations_by_completion: dict[int, set[str]] = defaultdict(set)
    for row in evaluations:
        evaluations_by_completion[int(row["completion_index"])].add(str(row["uuid"]))
    if set(evaluations_by_completion) != set(range(EXPECTED_ADAPTIVE_COMPLETIONS)):
        raise ValueError(f"adaptive seed {seed} lacks a completion block")
    if any(
        len(uuids) != EXPECTED_DEVELOPMENT_ROWS
        for uuids in evaluations_by_completion.values()
    ):
        raise ValueError(f"adaptive seed {seed} lacks a complete development block")

    selected_sha = sha256_file(paths["selected_solver"])
    if selected_sha != run["selected_code_sha256"]:
        raise ValueError(f"adaptive seed {seed} selected solver hash drifted")
    timing = json.loads(paths["timing"].read_text())
    expected_timing = {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "completion_count": EXPECTED_ADAPTIVE_COMPLETIONS,
        "development_execution_count": EXPECTED_ADAPTIVE_EXECUTIONS,
        "test_rows": EXPECTED_TEST_ROWS,
        "test_execution_repeats": EXPECTED_TEST_REPEATS,
        "test_classical_baselines": "none",
        "selected_code_sha256": selected_sha,
        "slurm_job_id": str(scheduler["job_id"]),
    }
    for key, expected in expected_timing.items():
        if timing.get(key) != expected:
            raise ValueError(f"adaptive seed {seed} timing {key} drifted")
    metrics = validate_metrics(paths["test_metrics"])
    selected_code = paths["selected_solver"].read_text()
    frozen_identifiers = development_uuids | metric_uuids(paths["test_metrics"])
    hardcoded_identifiers = sorted(
        identifier for identifier in frozen_identifiers if identifier in selected_code
    )
    if hardcoded_identifiers:
        raise ValueError(
            f"adaptive seed {seed} selected code contains frozen identifiers: "
            f"{hardcoded_identifiers[:5]}"
        )
    return (
        {str(path.relative_to(root)): sha256_file(path) for path in paths.values()},
        {
            "resolved_model_commit": run["resolved_model_commit"],
            "source_commit": run["provenance"]["top_level_commit"],
            "max_prompt_tokens": max(prompt_tokens),
            "exact_requirement_counts_by_round": {
                str(round_index): len(requirements)
                for round_index, requirements in exact_requirements_by_round.items()
            },
            "available_prompt_tokens": available_prompt_tokens,
            "completion_count": len(generations),
            "development_execution_count": len(evaluations),
            "test": metrics,
            "selected_code_sha256": selected_sha,
            "selected_code_frozen_identifier_matches": [],
            "scheduler": scheduler,
        },
    )


def validate_frozen_seed(
    root: Path,
    seed: int,
    protocol: dict[str, Any],
    protocol_sha256: str,
    manifest_sha256: str,
    scheduler: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    seed_root = root / f"seed{seed}"
    test_root = seed_root / f"test/fixed-code/frozen-hero/seed{seed}"
    paths = {
        "generation_provenance": seed_root / "generation_provenance.json",
        "generations": seed_root / "validation_generations.jsonl",
        "references": seed_root / "development_references.jsonl",
        "evaluations": seed_root / "selection/candidate_development_evaluations.jsonl",
        "summaries": seed_root / "selection/candidate_summaries.jsonl",
        "selected_solver": seed_root / "selection/selected_solver.py",
        "freeze": seed_root / "selection/pretest_freeze_manifest.json",
        "test_metrics": test_root / "metrics_final.csv",
        "timing": seed_root / "timing_accounting.json",
    }
    for path in paths.values():
        require_file(path)
    freeze = json.loads(paths["freeze"].read_text())
    expected_freeze = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "seed": seed,
        "candidate_completions": 64,
        "selection_instances": EXPECTED_DEVELOPMENT_ROWS,
        "slurm_job_id": str(scheduler["job_id"]),
        "test_outcomes_accessed": False,
    }
    for key, expected in expected_freeze.items():
        if freeze.get(key) != expected:
            raise ValueError(f"Frozen Hero seed {seed} freeze {key} drifted")
    if freeze.get("source", {}).get("dirty") is not False:
        raise ValueError(f"Frozen Hero seed {seed} source was dirty")
    selected_sha = sha256_file(paths["selected_solver"])
    if selected_sha != freeze["selected_code_sha256"]:
        raise ValueError(f"Frozen Hero seed {seed} solver hash drifted")

    generations = read_jsonl(paths["generations"])
    if len(generations) != 64 or len({str(row["uuid"]) for row in generations}) != 64:
        raise ValueError(f"Frozen Hero seed {seed} synthesis workload drifted")
    references = read_jsonl(paths["references"])
    reference_uuids = {str(row["uuid"]) for row in references}
    if len(references) != EXPECTED_DEVELOPMENT_ROWS or len(reference_uuids) != 40:
        raise ValueError(f"Frozen Hero seed {seed} reference workload drifted")
    evaluations = read_jsonl(paths["evaluations"])
    evaluations_by_code: dict[str, set[str]] = defaultdict(set)
    for row in evaluations:
        evaluations_by_code[str(row["code_sha256"])].add(str(row["uuid"]))
    if not evaluations_by_code or len(evaluations_by_code) > 64:
        raise ValueError(f"Frozen Hero seed {seed} unique-code count drifted")
    if any(uuids != reference_uuids for uuids in evaluations_by_code.values()):
        raise ValueError(f"Frozen Hero seed {seed} lacks a development block")
    metrics = validate_metrics(paths["test_metrics"])
    timing = json.loads(paths["timing"].read_text())
    expected_timing = {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "active_generation_gpus": 1,
        "synthesis_completions": 64,
        "test_outcomes_accessed_after_freeze": True,
        "slurm_job_id": str(scheduler["job_id"]),
    }
    for key, expected in expected_timing.items():
        if timing.get(key) != expected:
            raise ValueError(f"Frozen Hero seed {seed} timing {key} drifted")
    for key in (
        "generation_gpu_wall_seconds",
        "development_selection_wall_seconds",
        "test_evaluation_wall_seconds",
    ):
        if float(timing.get(key, 0)) <= 0:
            raise ValueError(f"Frozen Hero seed {seed} has invalid {key}")
    return (
        {str(path.relative_to(root)): sha256_file(path) for path in paths.values()},
        {
            "candidate_completions": len(generations),
            "unique_candidates": len(evaluations_by_code),
            "candidate_development_evaluations": len(evaluations),
            "test": metrics,
            "selected_code_sha256": selected_sha,
            "scheduler": scheduler,
        },
    )


def register(
    adaptive_root: Path,
    frozen_root: Path,
    scheduler_path: Path,
    adaptive_submission_path: Path,
    frozen_submission_path: Path,
    adaptive_protocol_path: Path,
    adaptive_manifest_path: Path,
    frozen_protocol_path: Path,
    frozen_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output_dir}")
    adaptive_submission = json.loads(adaptive_submission_path.read_text())
    frozen_submission = json.loads(frozen_submission_path.read_text())
    submission = {
        "adaptive_jobs": adaptive_submission["adaptive_jobs"],
        "frozen_hero_jobs": frozen_submission["frozen_hero_jobs"],
    }
    scheduler = load_scheduler_evidence(scheduler_path, submission)
    adaptive_scheduler = validate_scheduler(scheduler, submission, "adaptive")
    frozen_scheduler = validate_scheduler(scheduler, submission, "frozen_hero")
    adaptive_protocol = json.loads(adaptive_protocol_path.read_text())
    frozen_protocol = json.loads(frozen_protocol_path.read_text())
    adaptive_protocol_sha = sha256_file(adaptive_protocol_path)
    adaptive_manifest_sha = sha256_file(adaptive_manifest_path)
    frozen_protocol_sha = sha256_file(frozen_protocol_path)
    frozen_manifest_sha = sha256_file(frozen_manifest_path)

    adaptive_files: dict[str, str] = {}
    frozen_files: dict[str, str] = {}
    adaptive_validation: dict[str, Any] = {}
    frozen_validation: dict[str, Any] = {}
    for seed in SEEDS:
        files, validation = validate_adaptive_seed(
            adaptive_root,
            seed,
            adaptive_protocol,
            adaptive_protocol_sha,
            adaptive_manifest_sha,
            adaptive_scheduler[seed],
        )
        adaptive_files.update(files)
        adaptive_validation[str(seed)] = validation
        files, validation = validate_frozen_seed(
            frozen_root,
            seed,
            frozen_protocol,
            frozen_protocol_sha,
            frozen_manifest_sha,
            frozen_scheduler[seed],
        )
        frozen_files.update(files)
        frozen_validation[str(seed)] = validation

    adaptive_commits = {row["source_commit"] for row in adaptive_validation.values()}
    if adaptive_commits != {adaptive_submission["source_commit"]}:
        raise ValueError("adaptive runtime/source submission commit drifted")
    resolved_model_commits = {
        row["resolved_model_commit"] for row in adaptive_validation.values()
    }
    if len(resolved_model_commits) != 1:
        raise ValueError("adaptive seeds resolved different base-model commits")

    output_dir.mkdir(parents=True)
    common = {
        "status": "complete",
        "adaptive_submission": str(adaptive_submission_path),
        "adaptive_submission_sha256": sha256_file(adaptive_submission_path),
        "frozen_submission": str(frozen_submission_path),
        "frozen_submission_sha256": sha256_file(frozen_submission_path),
        "scheduler_sha256": sha256_file(scheduler_path),
    }
    adaptive_result = {
        **common,
        "protocol_id": "N26-E4-content-clean-result-v1",
        "timing_protocol_id": adaptive_protocol["protocol_id"],
        "artifact_root": str(adaptive_root),
        "protocol_sha256": adaptive_protocol_sha,
        "manifest_sha256": adaptive_manifest_sha,
        "scheduler": {str(seed): adaptive_scheduler[seed] for seed in SEEDS},
        "validation": adaptive_validation,
        "files_sha256": adaptive_files,
    }
    frozen_result = {
        **common,
        "protocol_id": "N26-E8-cost-registration-v3",
        "timing_protocol_id": frozen_protocol["protocol_id"],
        "artifact_root": str(frozen_root),
        "protocol_sha256": frozen_protocol_sha,
        "manifest_sha256": frozen_manifest_sha,
        "scheduler": {str(seed): frozen_scheduler[seed] for seed in SEEDS},
        "validation": frozen_validation,
        "files_sha256": frozen_files,
    }
    (output_dir / "adaptive_repair_result_manifest.json").write_text(
        json.dumps(adaptive_result, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "frozen_hero_result_manifest.json").write_text(
        json.dumps(frozen_result, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "protocol_id": "N26-content-clean-control-registration-v1",
        "adaptive_result_sha256": sha256_file(
            output_dir / "adaptive_repair_result_manifest.json"
        ),
        "frozen_result_sha256": sha256_file(
            output_dir / "frozen_hero_result_manifest.json"
        ),
    }
    (output_dir / "registration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, required=True)
    parser.add_argument("--adaptive-submission", type=Path, required=True)
    parser.add_argument("--frozen-submission", type=Path, required=True)
    parser.add_argument("--adaptive-protocol", type=Path, required=True)
    parser.add_argument("--adaptive-manifest", type=Path, required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            register(
                args.adaptive_root,
                args.frozen_root,
                args.scheduler,
                args.adaptive_submission,
                args.frozen_submission,
                args.adaptive_protocol,
                args.adaptive_manifest,
                args.frozen_protocol,
                args.frozen_manifest,
                args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
