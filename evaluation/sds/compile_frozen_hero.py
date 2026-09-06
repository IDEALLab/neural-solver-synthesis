#!/usr/bin/env python3
"""Select one Hero-generated SDS solver using validation data only."""

# ruff: noqa: PLC0415, PLR0912, PLR0913, PLR0915, PLR2004, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.sds.adaptive_repair import (
    Candidate,
    ProtocolError,
    assert_content_disjoint,
    assert_disjoint,
    certify_dev_rows,
    deduplicate_rows_by_mission,
    evaluate_candidates,
    extract_code,
    family_from_uuid,
    git_provenance,
    load_development_references,
    mission_sha256,
    select_balanced_rows,
    sha256_text,
    uuid_set_sha256,
    write_jsonl,
)


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    content_policy = protocol.get("development_content_policy")
    if content_policy not in {None, "exclude_all_test_missions_and_deduplicate"}:
        raise ProtocolError(f"unknown development content policy: {content_policy}")
    if int(protocol["synthesis_candidates"]) != 64:
        raise ProtocolError("Frozen Hero requires exactly 64 synthesis candidates")
    expected_families = int(protocol.get("expected_development_families", 10))
    selection_base = int(protocol["selection_cases_per_family"]) * expected_families
    expected_selection = int(protocol["expected_selection_cases"])
    if not selection_base <= expected_selection <= selection_base + expected_families:
        raise ProtocolError("selection design is not near-balanced across SDS families")
    if int(protocol["synthesis_cases_per_family"]) * expected_families + int(
        protocol["synthesis_extra_cases"]
    ) != int(protocol["synthesis_candidates"]):
        raise ProtocolError("synthesis family allocation does not total 64")
    expected_generation = {
        "temperature": 0.0,
        "completions_per_prompt": 1,
        "max_tokens": 4096,
        "tensor_parallel_size": 1,
    }
    if protocol.get("generation") != expected_generation:
        raise ProtocolError(
            f"generation settings must be the frozen full protocol: {expected_generation}"
        )
    expected_development_execution = {
        "timeout_seconds": 5.0,
        "workers": 64,
        "infeasible_gap_penalty": 1.0,
    }
    if protocol.get("development_execution") != expected_development_execution:
        raise ProtocolError(
            "development execution settings must be the frozen full protocol: "
            f"{expected_development_execution}"
        )
    if content_policy:
        expected_reference = {
            "solver": "OR-Tools CP-SAT",
            "ortools_version": "9.14.6206",
            "max_time_seconds": 600.0,
            "solver_seed": 0,
            "num_search_workers": 1,
        }
        if protocol.get("development_reference") != expected_reference:
            raise ProtocolError(
                "development reference settings must be the frozen full protocol "
                f"matched to Adaptive Repair: {expected_reference}"
            )
    expected_test_evaluation = {
        "rows_per_seed": 1000,
        "time_budget_seconds": 5.0,
        "repeats": 3,
        "workers": 64,
        "classical_baselines": "none",
        "quality_uses_first_execution": True,
        "join_keys": ["seed", "uuid"],
    }
    test_evaluation = protocol.get("test_evaluation", {})
    historical_test_core = {
        key: expected_test_evaluation[key]
        for key in (
            "rows_per_seed",
            "time_budget_seconds",
            "repeats",
            "join_keys",
        )
    }
    observed_test_core = {key: test_evaluation.get(key) for key in historical_test_core}
    if content_policy and test_evaluation != expected_test_evaluation:
        raise ProtocolError(
            f"test evaluation settings must be the frozen full protocol: {expected_test_evaluation}"
        )
    if not content_policy and observed_test_core != historical_test_core:
        raise ProtocolError(
            f"historical test evaluation settings drifted: {historical_test_core}"
        )
    return protocol


def _select_synthesis_rows(
    rows: list[dict[str, Any]],
    excluded_uuids: set[str],
    cases_per_family: int,
    extra_cases: int,
    salt: str,
    expected_families: int = 10,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        uuid = str(row["uuid"])
        if uuid not in excluded_uuids:
            grouped[family_from_uuid(uuid)].append(row)
    if len(grouped) != expected_families:
        raise ProtocolError(
            f"expected {expected_families} SDS families, found {len(grouped)}"
        )

    selected: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for family in sorted(grouped):
        ranked = sorted(
            grouped[family],
            key=lambda row: sha256_text(
                f"{salt}:{row['uuid']}:{mission_sha256(row['mission'])}"
            ),
        )
        if len(ranked) < cases_per_family:
            raise ProtocolError(f"family {family} lacks synthesis rows")
        selected.extend(ranked[:cases_per_family])
        remaining.extend(ranked[cases_per_family:])

    extras = sorted(
        remaining,
        key=lambda row: sha256_text(
            f"{salt}:extra:{row['uuid']}:{mission_sha256(row['mission'])}"
        ),
    )[:extra_cases]
    selected.extend(extras)
    return sorted(selected, key=lambda row: sha256_text(f"{salt}:order:{row['uuid']}"))


def freeze_manifest(protocol_path: Path, output_path: Path) -> None:
    from datasets import load_dataset

    protocol = _load_protocol(protocol_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output_path}")

    test_uuids_by_seed: dict[int, set[str]] = {}
    test_mission_hashes_by_seed: dict[int, set[str]] = {}
    test_hashes: dict[str, str] = {}
    test_content_hashes: dict[str, str] = {}
    for seed_text, dataset in protocol["datasets"].items():
        test_rows = load_dataset(
            dataset["repo_id"],
            split=protocol["test_split"],
            revision=dataset["revision"],
        )
        if len(test_rows) != int(protocol["expected_split_count"]):
            raise ProtocolError(f"unexpected test count for seed {seed_text}")
        uuids = {str(uuid) for uuid in test_rows["uuid"]}
        if len(uuids) != len(test_rows):
            raise ProtocolError(f"duplicate test UUID for seed {seed_text}")
        test_uuids_by_seed[int(seed_text)] = uuids
        test_mission_hashes_by_seed[int(seed_text)] = {
            mission_sha256(row["mission"]) for row in test_rows
        }
        test_hashes[seed_text] = uuid_set_sha256(uuids)
        test_content_hashes[seed_text] = uuid_set_sha256(
            test_mission_hashes_by_seed[int(seed_text)]
        )

    excluded_test_content = set().union(*test_mission_hashes_by_seed.values())

    frozen_seeds: dict[str, Any] = {}
    for seed_text, dataset in protocol["datasets"].items():
        validation = list(
            load_dataset(
                dataset["repo_id"],
                split=protocol["development_split"],
                revision=dataset["revision"],
            )
        )
        if len(validation) != int(protocol["expected_split_count"]):
            raise ProtocolError(f"unexpected validation count for seed {seed_text}")
        if protocol.get("development_content_policy") == (
            "exclude_all_test_missions_and_deduplicate"
        ):
            validation = [
                row
                for row in validation
                if mission_sha256(row["mission"]) not in excluded_test_content
            ]
            validation = deduplicate_rows_by_mission(
                validation,
                f"{protocol['manifest_salt']}:deduplicate:seed{seed_text}",
            )
        selection = select_balanced_rows(
            validation,
            int(protocol["selection_cases_per_family"]),
            f"{protocol['manifest_salt']}:selection:seed{seed_text}",
            int(protocol["expected_selection_cases"]),
        )
        selection_uuids = {str(row["uuid"]) for row in selection}
        synthesis = _select_synthesis_rows(
            validation,
            selection_uuids,
            int(protocol["synthesis_cases_per_family"]),
            int(protocol["synthesis_extra_cases"]),
            f"{protocol['manifest_salt']}:synthesis:seed{seed_text}",
            int(protocol.get("expected_development_families", 10)),
        )
        synthesis_uuids = {str(row["uuid"]) for row in synthesis}
        if len(selection_uuids) != int(protocol["expected_selection_cases"]):
            raise ProtocolError("unexpected selection row count")
        if len(synthesis_uuids) != int(protocol["synthesis_candidates"]):
            raise ProtocolError("unexpected synthesis row count")
        if selection_uuids & synthesis_uuids:
            raise ProtocolError("synthesis and selection rows overlap")
        assert_disjoint(selection_uuids | synthesis_uuids, test_uuids_by_seed)
        if protocol.get("development_content_policy"):
            selection_hashes = {mission_sha256(row["mission"]) for row in selection}
            synthesis_hashes = {mission_sha256(row["mission"]) for row in synthesis}
            if len(selection_hashes) != len(selection):
                raise ProtocolError("selection manifest contains duplicate missions")
            if len(synthesis_hashes) != len(synthesis):
                raise ProtocolError("synthesis manifest contains duplicate missions")
            if selection_hashes & synthesis_hashes:
                raise ProtocolError("synthesis and selection mission content overlaps")
            assert_content_disjoint(
                selection_hashes | synthesis_hashes, test_mission_hashes_by_seed
            )

        def records(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
            return [
                {
                    "uuid": str(row["uuid"]),
                    "family": family_from_uuid(str(row["uuid"])),
                    "mission_sha256": mission_sha256(row["mission"]),
                }
                for row in rows
            ]

        frozen_seeds[seed_text] = {
            "dataset": dataset,
            "split": protocol["development_split"],
            "synthesis_rows": records(synthesis),
            "selection_rows": records(selection),
        }

    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_text(protocol_path.read_text()),
        "test_uuid_set_sha256": test_hashes,
        "test_mission_content_set_sha256": test_content_hashes,
        "selection_rule": protocol["selection_rule"],
        "development_content_policy": protocol.get("development_content_policy"),
        "expected_development_families": protocol.get(
            "expected_development_families", 10
        ),
        "seeds": frozen_seeds,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _load_rows(
    protocol: dict[str, Any], manifest: dict[str, Any], seed: int, section: str
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    seed_text = str(seed)
    if manifest["protocol_id"] != protocol["protocol_id"]:
        raise ProtocolError("manifest/protocol ID mismatch")
    dataset = protocol["datasets"][seed_text]
    validation = load_dataset(
        dataset["repo_id"],
        split=protocol["development_split"],
        revision=dataset["revision"],
    )
    by_uuid = {str(row["uuid"]): dict(row) for row in validation}
    frozen = manifest["seeds"][seed_text][f"{section}_rows"]
    rows: list[dict[str, Any]] = []
    for record in frozen:
        uuid = record["uuid"]
        if uuid not in by_uuid:
            raise ProtocolError(f"frozen validation UUID missing: {uuid}")
        row = by_uuid[uuid]
        if mission_sha256(row["mission"]) != record["mission_sha256"]:
            raise ProtocolError(f"mission drift for validation UUID: {uuid}")
        rows.append(row)
    return rows


def _read_verified_manifest(
    protocol_path: Path, protocol: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ProtocolError("manifest/protocol ID mismatch")
    if manifest.get("protocol_sha256") != sha256_text(protocol_path.read_text()):
        raise ProtocolError("manifest/protocol content mismatch")
    return manifest


def write_uuid_list(
    protocol_path: Path, manifest_path: Path, seed: int, output_path: Path
) -> None:
    protocol = _load_protocol(protocol_path)
    manifest = _read_verified_manifest(protocol_path, protocol, manifest_path)
    rows = _load_rows(protocol, manifest, seed, "synthesis")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite UUID list: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{row['uuid']}\n" for row in rows))


def certify_selection(
    protocol_path: Path, manifest_path: Path, seed: int, output_path: Path
) -> None:
    import ortools

    protocol = _load_protocol(protocol_path)
    manifest = _read_verified_manifest(protocol_path, protocol, manifest_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite references: {output_path}")
    expected = str(protocol["development_reference"]["ortools_version"])
    if ortools.__version__ != expected:
        raise ProtocolError(f"OR-Tools version {ortools.__version__} != {expected}")
    rows = _load_rows(protocol, manifest, seed, "selection")
    references = certify_dev_rows(rows, protocol)
    for reference in references:
        reference["ortools_version"] = ortools.__version__
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, references)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank_key(summary: dict[str, Any]) -> tuple[float, float, str]:
    return (
        -float(summary["pass_rate"]),
        float(summary["mean_penalized_gap"]),
        str(summary["code_sha256"] or "~"),
    )


def summarize_validation_candidate(
    summary: dict[str, Any],
    evaluations: list[dict[str, Any]],
    references_by_uuid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply benchmark-feasible pass semantics to validation selection."""
    benchmark_rows = [
        row
        for row in evaluations
        if references_by_uuid[str(row["uuid"])]["status"] != "INFEASIBLE"
    ]
    proved_infeasible = [
        row
        for row in evaluations
        if references_by_uuid[str(row["uuid"])]["status"] == "INFEASIBLE"
    ]
    false_feasible = sum(bool(row["feasible"]) for row in proved_infeasible)
    if false_feasible:
        raise ProtocolError(
            "candidate is feasible on a proved-INFEASIBLE validation row"
        )
    bounded_gaps = [
        float(row["penalized_gap"])
        for row in benchmark_rows
        if row["penalized_gap"] is not None
    ]
    if not benchmark_rows or not bounded_gaps:
        raise ProtocolError(
            "validation selection lacks benchmark-feasible bounded rows"
        )
    return {
        **summary,
        "benchmark_feasible_development_evaluations": len(benchmark_rows),
        "proved_infeasible_development_evaluations": len(proved_infeasible),
        "feasible_count": sum(bool(row["feasible"]) for row in benchmark_rows),
        "pass_rate": sum(bool(row["feasible"]) for row in benchmark_rows)
        / len(benchmark_rows),
        "bounded_reference_evaluations": len(bounded_gaps),
        "mean_penalized_gap": sum(bounded_gaps) / len(bounded_gaps),
        "false_feasible_on_proved_infeasible": false_feasible,
    }


def select_solver(
    protocol_path: Path,
    manifest_path: Path,
    references_path: Path,
    generations_path: Path,
    seed: int,
    output_dir: Path,
    source_freeze_path: Path | None = None,
) -> None:
    protocol = _load_protocol(protocol_path)
    manifest = _read_verified_manifest(protocol_path, protocol, manifest_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse immutable output: {output_dir}")
    provenance = git_provenance()
    if provenance["dirty"]:
        raise ProtocolError("source checkout is dirty")

    synthesis_rows = _load_rows(protocol, manifest, seed, "synthesis")
    selection_rows = _load_rows(protocol, manifest, seed, "selection")
    references_raw, references = load_development_references(
        references_path, selection_rows, protocol
    )
    generation_records = [
        json.loads(line) for line in generations_path.read_text().splitlines() if line
    ]
    expected_uuids = [str(row["uuid"]) for row in synthesis_rows]
    actual_uuids = [str(row.get("uuid")) for row in generation_records]
    if actual_uuids != expected_uuids:
        raise ProtocolError(
            "generation UUID order/content does not match frozen synthesis rows"
        )
    if len(generation_records) != int(protocol["synthesis_candidates"]):
        raise ProtocolError("generation count is not exactly 64")
    expected_dataset_revision = protocol["datasets"][str(seed)]["revision"]
    expected_model_revision = protocol["models"][str(seed)]["revision"]
    for record in generation_records:
        if record.get("dataset_revision") != expected_dataset_revision:
            raise ProtocolError("generation dataset revision mismatch")
        if record.get("model_revision") != expected_model_revision:
            raise ProtocolError("generation model revision mismatch")
        if record.get("system_prompt_sha256") != protocol["system_prompt_sha256"]:
            raise ProtocolError("generation system-prompt hash mismatch")
        if int(record.get("sample_idx", -1)) != 0:
            raise ProtocolError(
                "generation sample index is not the single frozen sample"
            )

    candidates: list[Candidate] = []
    source_uuids: dict[int, str] = {}
    seen_code: set[str | None] = set()
    for index, record in enumerate(generation_records):
        code = extract_code(str(record.get("generated_text", "")))
        code_sha = sha256_text(code) if code else None
        if code_sha in seen_code:
            continue
        seen_code.add(code_sha)
        candidates.append(
            Candidate(
                index, 0, str(record.get("generated_text", "")), code, code_sha, None
            )
        )
        source_uuids[index] = str(record["uuid"])
    if not candidates:
        raise ProtocolError("no generated candidates were available")

    summaries, evaluations = evaluate_candidates(
        candidates, selection_rows, references, protocol
    )
    references_by_uuid = {str(row["uuid"]): row for row in references_raw}
    summaries = [
        summarize_validation_candidate(
            summary,
            evaluations[int(summary["completion_index"])],
            references_by_uuid,
        )
        for summary in summaries
    ]
    for summary in summaries:
        summary["synthesis_uuid"] = source_uuids[int(summary["completion_index"])]
    selected = min(summaries, key=_rank_key)
    selected_candidate = next(
        candidate
        for candidate in candidates
        if candidate.completion_index == int(selected["completion_index"])
    )
    if not selected_candidate.code:
        raise ProtocolError("selected candidate contains no executable code")

    correction = None
    if source_freeze_path is not None:
        source_freeze = json.loads(source_freeze_path.read_text())
        if source_freeze.get("protocol_id") != protocol["protocol_id"]:
            raise ProtocolError("source freeze protocol mismatch")
        if source_freeze.get("generation_file_sha256") != _file_sha256(
            generations_path
        ):
            raise ProtocolError("source freeze generation hash mismatch")
        correction = {
            "reason": "Exclude proved-INFEASIBLE validation rows from the pass-rate denominator while auditing false-feasible outputs.",
            "source_freeze_path": str(source_freeze_path),
            "source_freeze_sha256": _file_sha256(source_freeze_path),
            "original_selected_code_sha256": source_freeze.get("selected_code_sha256"),
            "selection_changed": source_freeze.get("selected_code_sha256")
            != selected_candidate.code_sha256,
            "generations_reused_without_regeneration": True,
        }

    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "candidate_summaries.jsonl", summaries)
    write_jsonl(
        output_dir / "candidate_development_evaluations.jsonl",
        [row for index in sorted(evaluations) for row in evaluations[index]],
    )
    write_jsonl(output_dir / "development_references.jsonl", references_raw)
    solver_path = output_dir / "selected_solver.py"
    solver_path.write_text(selected_candidate.code)
    freeze = {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "selection_rule": protocol["selection_rule"],
        "selected_completion_index": selected_candidate.completion_index,
        "selected_synthesis_uuid": source_uuids[selected_candidate.completion_index],
        "selected_code_sha256": selected_candidate.code_sha256,
        "candidate_completions": len(generation_records),
        "unique_executable_or_malformed_candidates": len(candidates),
        "selection_instances": len(selection_rows),
        "selected_development_summary": selected,
        "protocol_sha256": _file_sha256(protocol_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "generation_file_sha256": _file_sha256(generations_path),
        "references_sha256": _file_sha256(references_path),
        "source": provenance,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "test_outcomes_accessed": False,
    }
    if correction is not None:
        freeze["correction"] = correction
    freeze_path = output_dir / "pretest_freeze_manifest.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    os.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-manifest")
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    uuids = subparsers.add_parser("write-uuid-list")
    uuids.add_argument("--protocol", type=Path, required=True)
    uuids.add_argument("--manifest", type=Path, required=True)
    uuids.add_argument("--seed", type=int, required=True)
    uuids.add_argument("--output", type=Path, required=True)
    certify = subparsers.add_parser("certify-selection")
    certify.add_argument("--protocol", type=Path, required=True)
    certify.add_argument("--manifest", type=Path, required=True)
    certify.add_argument("--seed", type=int, required=True)
    certify.add_argument("--output", type=Path, required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--protocol", type=Path, required=True)
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--references", type=Path, required=True)
    select.add_argument("--generations", type=Path, required=True)
    select.add_argument("--seed", type=int, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--source-freeze", type=Path)
    args = parser.parse_args()
    if args.command == "freeze-manifest":
        freeze_manifest(args.protocol, args.output)
    elif args.command == "write-uuid-list":
        write_uuid_list(args.protocol, args.manifest, args.seed, args.output)
    elif args.command == "certify-selection":
        certify_selection(args.protocol, args.manifest, args.seed, args.output)
    else:
        select_solver(
            args.protocol,
            args.manifest,
            args.references,
            args.generations,
            args.seed,
            args.output_dir,
            args.source_freeze,
        )


if __name__ == "__main__":
    main()
