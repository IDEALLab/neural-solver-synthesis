"""Leakage-free, certification-aware rerun of the Base universal-code search."""

# Protocol checks intentionally fail with the violated invariant, and the two
# orchestration functions keep the full audit flow together.
# ruff: noqa: PLR0912, PLR0915, TRY003

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from evaluation.sds.adaptive_repair import (
    Candidate,
    ProtocolError,
    evaluate_one,
    load_frozen_rows,
    mission_sha256,
    summarize_candidate,
)
from evaluation.sds.adaptive_repair import (
    load_protocol as load_development_protocol,
)

CODE_BLOCK_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE)
EXPECTED_SEEDS = {101, 202, 303}
EXPECTED_RAW_RECORDS = 192000
EXPECTED_UNIQUE_CODES = 191699
EXPECTED_DEVELOPMENT_ROWS = 40
EXPECTED_TIMEOUT_SECONDS = 5.0
EXPECTED_SHARDS_PER_SEED = 4
SHA256_HEX_LENGTH = 64


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def extract_historical_code(generated_text: str) -> str | None:
    """Match the original 191,699-program tournament extraction exactly."""
    match = CODE_BLOCK_RE.search(generated_text or "")
    if not match:
        return None
    code = match.group(1).strip()
    return code or None


def canonicalize_historical_code(code: str) -> str:
    code = (code or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in code.split("\n")).strip() + "\n"


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    if (
        protocol.get("protocol_id")
        != "N26-E9-v3-certified-input-disjoint-universal-search"
    ):
        raise ProtocolError("unexpected certified universal-search protocol ID")
    if {int(seed) for seed in protocol.get("sources", {})} != EXPECTED_SEEDS:
        raise ProtocolError("universal-search source seed set drift")
    if int(protocol.get("expected_raw_records", -1)) != EXPECTED_RAW_RECORDS:
        raise ProtocolError("universal-search raw population drift")
    if int(protocol.get("expected_global_unique_codes", -1)) != EXPECTED_UNIQUE_CODES:
        raise ProtocolError("universal-search unique population drift")
    development = protocol.get("development", {})
    expected_selection = [
        "highest pass rate on benchmark-feasible development rows",
        "lowest mean failure-penalized gap to finite CP-SAT upper bounds",
        "lexicographically smallest full code SHA-256",
    ]
    if development.get("selection") != expected_selection:
        raise ProtocolError("universal-search selection rule drift")
    if int(development.get("rows_per_seed", -1)) != EXPECTED_DEVELOPMENT_ROWS:
        raise ProtocolError("universal-search development workload drift")
    if float(development.get("timeout_seconds", -1)) != EXPECTED_TIMEOUT_SECONDS:
        raise ProtocolError("universal-search timeout drift")
    if int(development.get("shards_per_seed", -1)) != EXPECTED_SHARDS_PER_SEED:
        raise ProtocolError("universal-search shard-count drift")
    if protocol.get("test", {}).get("join_keys") != ["seed", "uuid"]:
        raise ProtocolError("universal-search join-key drift")
    return protocol


def rank_key(summary: dict[str, Any]) -> tuple[float, float, str]:
    """Frozen lexicographic development-only selection rule."""
    return (
        -float(summary["pass_rate"]),
        float(summary["mean_penalized_gap"]),
        str(summary["code_sha256"]),
    )


def eligible_for_target(
    record: dict[str, Any], target_seed: int, target_test_mission_hashes: set[str]
) -> tuple[bool, str]:
    source_seeds = {int(seed) for seed in record["source_seeds"]}
    if target_seed in source_seeds:
        return False, "target_seed_source"
    source_missions = set(record["source_mission_sha256"])
    if source_missions & target_test_mission_hashes:
        return False, "target_test_content_source"
    return True, "eligible"


def shard_for_code(code_sha256: str, shard_count: int) -> int:
    if shard_count <= 0 or len(code_sha256) != SHA256_HEX_LENGTH:
        raise ProtocolError("invalid deterministic shard input")
    return int(code_sha256, 16) % shard_count


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProtocolError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ProtocolError(f"non-object JSONL at {path}:{line_number}")
            yield value


def prepare_pool(protocol_path: Path, source_root: Path, output_dir: Path) -> None:
    protocol = load_protocol(protocol_path)
    protocol_sha256 = sha256_file(protocol_path)
    if output_dir.exists():
        raise ProtocolError(f"immutable pool output already exists: {output_dir}")
    in_progress = output_dir.with_name(output_dir.name + ".inprogress")
    if in_progress.exists():
        raise ProtocolError(f"incomplete pool output already exists: {in_progress}")
    in_progress.mkdir(parents=True)

    candidates: dict[str, dict[str, Any]] = {}
    source_results: dict[str, Any] = {}
    total_extractable = 0
    for seed_text, source in sorted(
        protocol["sources"].items(), key=lambda item: int(item[0])
    ):
        seed = int(seed_text)
        source_path = source_root / f"seed{seed}" / source["filename"]
        if not source_path.is_file():
            raise ProtocolError(f"missing pinned Base source: {source_path}")
        actual_source_sha = sha256_file(source_path)
        if actual_source_sha != source["sha256"]:
            raise ProtocolError(f"pinned Base source hash drift for seed {seed}")

        raw_records = 0
        extractable_records = 0
        seed_hashes: set[str] = set()
        for record in _iter_jsonl(source_path):
            raw_records += 1
            code = extract_historical_code(str(record.get("generated_text", "")))
            if not code:
                continue
            extractable_records += 1
            total_extractable += 1
            canonical_code = canonicalize_historical_code(code)
            code_sha = sha256_text(canonical_code)
            seed_hashes.add(code_sha)
            source_uuid = str(record.get("uuid", ""))
            if not source_uuid or "mission" not in record:
                raise ProtocolError(f"source provenance missing for seed {seed}")
            source_mission_sha = mission_sha256(record["mission"])
            incumbent = candidates.setdefault(
                code_sha,
                {
                    "code_sha256": code_sha,
                    "code": canonical_code,
                    "source_seeds": set(),
                    "source_uuids": set(),
                    "source_mission_sha256": set(),
                },
            )
            if incumbent["code"] != canonical_code:
                raise ProtocolError("full SHA-256 collision in candidate pool")
            incumbent["source_seeds"].add(seed)
            incumbent["source_uuids"].add(source_uuid)
            incumbent["source_mission_sha256"].add(source_mission_sha)

        if raw_records != int(source["records"]):
            raise ProtocolError(
                f"raw source count drift for seed {seed}: {raw_records}"
            )
        expected_unique = int(source["extractable_unique_codes"])
        if (
            extractable_records != expected_unique
            or len(seed_hashes) != expected_unique
        ):
            raise ProtocolError(
                f"extractable/unique source count drift for seed {seed}: "
                f"{extractable_records}/{len(seed_hashes)}"
            )
        source_results[seed_text] = {
            "path": str(source_path),
            "repo": source["repo"],
            "revision": source["revision"],
            "sha256": actual_source_sha,
            "raw_records": raw_records,
            "extractable_records": extractable_records,
            "unique_codes": len(seed_hashes),
        }

    if total_extractable != int(protocol["expected_extractable_records"]):
        raise ProtocolError(f"global extractable count drift: {total_extractable}")
    if len(candidates) != int(protocol["expected_global_unique_codes"]):
        raise ProtocolError(f"global unique code count drift: {len(candidates)}")

    pool_path = in_progress / "candidate_pool.jsonl"
    with pool_path.open("w") as handle:
        for code_sha in sorted(candidates):
            record = candidates[code_sha]
            payload = {
                "code_sha256": code_sha,
                "code": record["code"],
                "source_seeds": sorted(record["source_seeds"]),
                "source_uuids": sorted(record["source_uuids"]),
                "source_mission_sha256": sorted(record["source_mission_sha256"]),
            }
            handle.write(canonical_json(payload) + "\n")

    manifest = {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "source_policy": protocol["source_policy"],
        "source_results": source_results,
        "raw_records": sum(item["raw_records"] for item in source_results.values()),
        "extractable_records": total_extractable,
        "global_unique_codes": len(candidates),
        "candidate_pool_sha256": sha256_file(pool_path),
        "candidate_pool_bytes": pool_path.stat().st_size,
        "source_seed_sets": dict(
            Counter(
                ",".join(str(seed) for seed in sorted(record["source_seeds"]))
                for record in candidates.values()
            )
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(in_progress / "pool_manifest.json", manifest)
    in_progress.replace(output_dir)


def _load_development_inputs(
    protocol: dict[str, Any],
    repo_root: Path,
    development_manifest_path: Path,
    references_path: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float | None], dict[str, str]]:
    development = protocol["development"]
    development_protocol_path = repo_root / development["protocol"]
    if sha256_file(development_protocol_path) != development["protocol_sha256"]:
        raise ProtocolError("development protocol hash drift")
    if sha256_file(development_manifest_path) != development["manifest_sha256"]:
        raise ProtocolError("development manifest hash drift")
    if sha256_file(references_path) != development["reference_sha256"][str(seed)]:
        raise ProtocolError(f"development reference hash drift for seed {seed}")

    development_protocol = load_development_protocol(development_protocol_path)
    development_manifest = load_json(development_manifest_path)
    rows = load_frozen_rows(development_protocol, development_manifest, seed)
    reference_rows = list(_iter_jsonl(references_path))
    references: dict[str, float | None] = {}
    statuses: dict[str, str] = {}
    for reference in reference_rows:
        uuid = str(reference["uuid"])
        status = str(reference["status"])
        statuses[uuid] = status
        bound = reference.get("best_bound")
        references[uuid] = (
            float(bound)
            if status != "INFEASIBLE"
            and bound is not None
            and math.isfinite(float(bound))
            else None
        )
    row_uuids = {str(row["uuid"]) for row in rows}
    if set(references) != row_uuids or set(statuses) != row_uuids:
        raise ProtocolError("development reference UUID drift")
    if len(rows) != int(development["rows_per_seed"]):
        raise ProtocolError("development row count drift")
    return rows, references, statuses


def _target_test_hashes(
    protocol: dict[str, Any], repo_root: Path, seed: int
) -> set[str]:
    clean_path = repo_root / protocol["clean_test_manifest"]
    if sha256_file(clean_path) != protocol["clean_test_manifest_sha256"]:
        raise ProtocolError("clean test manifest hash drift")
    clean_manifest = load_json(clean_path)
    retained = clean_manifest["seeds"][str(seed)]["retained"]
    hashes = {str(row["mission_sha256"]) for row in retained}
    if len(hashes) != int(clean_manifest["seeds"][str(seed)]["retained_rows"]):
        raise ProtocolError(f"clean test mission hash drift for seed {seed}")
    return hashes


def _summarize_chunk(
    candidates: list[Candidate],
    evaluations: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_candidate: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for evaluation in evaluations:
        by_candidate[int(evaluation["completion_index"])].append(evaluation)
    summaries: list[dict[str, Any]] = []
    for candidate, source in zip(candidates, provenance, strict=True):
        rows = sorted(
            by_candidate[candidate.completion_index], key=lambda row: row["uuid"]
        )
        summary = summarize_candidate(candidate, rows)
        summary.update(
            {
                "source_seeds": source["source_seeds"],
                "source_uuids": source["source_uuids"],
                "source_mission_sha256": source["source_mission_sha256"],
                "duplicate_selection_rows": sum(
                    int(row.get("duplicate_selection_count", 0)) > 0 for row in rows
                ),
                "duplicate_id_occurrences": sum(
                    int(row.get("duplicate_selection_count", 0)) for row in rows
                ),
                "error_counts": dict(
                    sorted(Counter(str(row["error_type"]) for row in rows).items())
                ),
            }
        )
        summaries.append(summary)
    return summaries


def _validate_cached_chunk(
    summary_path: Path,
    manifest_path: Path,
    expected_hashes: list[str],
    rows_per_candidate: int,
) -> list[dict[str, Any]] | None:
    if not summary_path.exists() and not manifest_path.exists():
        return None
    if not summary_path.is_file() or not manifest_path.is_file():
        raise ProtocolError(f"partial deterministic chunk: {summary_path.parent}")
    manifest = load_json(manifest_path)
    if manifest.get("candidate_hashes_sha256") != sha256_text(
        "\n".join(expected_hashes) + "\n"
    ):
        raise ProtocolError(f"cached chunk candidate drift: {summary_path}")
    if manifest.get("summary_sha256") != sha256_file(summary_path):
        raise ProtocolError(f"cached chunk summary hash drift: {summary_path}")
    summaries = list(_iter_jsonl(summary_path))
    if [str(row["code_sha256"]) for row in summaries] != expected_hashes:
        raise ProtocolError(f"cached chunk order drift: {summary_path}")
    if (
        int(manifest.get("evaluations", -1))
        != len(expected_hashes) * rows_per_candidate
    ):
        raise ProtocolError(f"cached chunk workload drift: {summary_path}")
    return summaries


def select_solver(  # noqa: PLR0913
    protocol_path: Path,
    pool_dir: Path,
    development_manifest_path: Path,
    references_path: Path,
    output_dir: Path,
    seed: int,
    shard_index: int,
) -> None:
    if seed not in EXPECTED_SEEDS:
        raise ProtocolError(f"unsupported target seed: {seed}")
    protocol = load_protocol(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    repo_root = Path.cwd()
    pool_manifest_path = pool_dir / "pool_manifest.json"
    pool_path = pool_dir / "candidate_pool.jsonl"
    pool_manifest = load_json(pool_manifest_path)
    if pool_manifest.get("status") != "complete":
        raise ProtocolError("candidate pool is incomplete")
    if pool_manifest.get("protocol_sha256") != protocol_sha:
        raise ProtocolError("candidate pool protocol drift")
    if pool_manifest.get("candidate_pool_sha256") != sha256_file(pool_path):
        raise ProtocolError("candidate pool hash drift")
    if int(pool_manifest.get("global_unique_codes", -1)) != int(
        protocol["expected_global_unique_codes"]
    ):
        raise ProtocolError("candidate pool count drift")

    rows, references, statuses = _load_development_inputs(
        protocol, repo_root, development_manifest_path, references_path, seed
    )
    target_hashes = _target_test_hashes(protocol, repo_root, seed)
    execution_protocol = {
        "development_execution": {
            "timeout_seconds": float(protocol["development"]["timeout_seconds"]),
            "workers": int(protocol["development"]["workers"]),
            "diagnostic_cases": 0,
            "infeasible_gap_penalty": float(
                protocol["development"]["infeasible_gap_penalty"]
            ),
        }
    }
    chunk_size = int(protocol["development"]["chunk_size"])
    workers = int(protocol["development"]["workers"])
    shard_count = int(protocol["development"]["shards_per_seed"])
    if not 0 <= shard_index < shard_count:
        raise ProtocolError(f"invalid shard index: {shard_index}")

    if output_dir.exists():
        raise ProtocolError(f"immutable selection output already exists: {output_dir}")
    in_progress = output_dir.with_name(output_dir.name + ".inprogress")
    in_progress.mkdir(parents=True, exist_ok=True)
    chunks_dir = in_progress / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    run_header = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "pool_sha256": pool_manifest["candidate_pool_sha256"],
        "development_manifest_sha256": sha256_file(development_manifest_path),
        "development_references_sha256": sha256_file(references_path),
        "target_seed": seed,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "target_clean_test_mission_set_sha256": sha256_text(
            "\n".join(sorted(target_hashes)) + "\n"
        ),
        "chunk_size": chunk_size,
        "workers": workers,
    }
    header_path = in_progress / "run_header.json"
    if header_path.exists():
        if load_json(header_path) != run_header:
            raise ProtocolError("in-progress selection header drift")
    else:
        write_json(header_path, run_header)

    fold_eligible_count = 0
    shard_eligible_count = 0
    exclusion_counts: Counter[str] = Counter()
    chunk_records: list[dict[str, Any]] = []
    chunk_index = 0
    best_summary: dict[str, Any] | None = None
    best_code: str | None = None
    total_evaluations = 0
    total_duplicate_rows = 0
    total_duplicate_occurrences = 0
    total_false_feasible = 0
    aggregate_errors: Counter[str] = Counter()
    merged_summary_path = in_progress / "development_summaries.jsonl"
    if merged_summary_path.exists():
        merged_summary_path.unlink()

    executor: ProcessPoolExecutor | None = None

    def process_chunk(
        records: list[dict[str, Any]], index: int
    ) -> list[dict[str, Any]]:
        nonlocal executor
        expected_hashes = [str(record["code_sha256"]) for record in records]
        summary_path = chunks_dir / f"chunk-{index:06d}.jsonl"
        manifest_path = chunks_dir / f"chunk-{index:06d}.manifest.json"
        cached = _validate_cached_chunk(
            summary_path, manifest_path, expected_hashes, len(rows)
        )
        if cached is not None:
            return cached
        if executor is None:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        candidates = [
            Candidate(
                completion_index=shard_eligible_count - len(records) + offset,
                round_index=0,
                generated_text="",
                code=str(record["code"]),
                code_sha256=str(record["code_sha256"]),
                parent_code_sha256=None,
            )
            for offset, record in enumerate(records)
        ]
        payloads = [
            (
                candidate,
                row,
                references[str(row["uuid"])],
                statuses[str(row["uuid"])],
                float(execution_protocol["development_execution"]["timeout_seconds"]),
                float(
                    execution_protocol["development_execution"][
                        "infeasible_gap_penalty"
                    ]
                ),
            )
            for candidate in candidates
            for row in rows
        ]
        evaluations = list(executor.map(evaluate_one, payloads, chunksize=1))
        summaries = _summarize_chunk(candidates, evaluations, records)
        temporary = summary_path.with_suffix(".jsonl.tmp")
        with temporary.open("w") as handle:
            for summary in summaries:
                handle.write(canonical_json(summary) + "\n")
        temporary.replace(summary_path)
        write_json(
            manifest_path,
            {
                "chunk_index": index,
                "candidates": len(records),
                "evaluations": len(evaluations),
                "candidate_hashes_sha256": sha256_text(
                    "\n".join(expected_hashes) + "\n"
                ),
                "summary_sha256": sha256_file(summary_path),
            },
        )
        return summaries

    def register_summaries(summaries: list[dict[str, Any]]) -> None:
        nonlocal best_summary, best_code, total_evaluations
        nonlocal total_duplicate_rows, total_duplicate_occurrences, total_false_feasible
        records_by_hash = {
            str(record["code_sha256"]): record for record in chunk_records
        }
        with merged_summary_path.open("a") as merged:
            for summary in summaries:
                code_sha = str(summary["code_sha256"])
                merged.write(canonical_json(summary) + "\n")
                total_evaluations += int(summary["development_evaluations"])
                total_duplicate_rows += int(summary["duplicate_selection_rows"])
                total_duplicate_occurrences += int(summary["duplicate_id_occurrences"])
                total_false_feasible += int(
                    summary["false_feasible_on_proved_infeasible"]
                )
                aggregate_errors.update(summary["error_counts"])
                if best_summary is None or rank_key(summary) < rank_key(best_summary):
                    best_summary = summary
                    best_code = str(records_by_hash[code_sha]["code"])

    try:
        for record in _iter_jsonl(pool_path):
            is_eligible, reason = eligible_for_target(record, seed, target_hashes)
            if not is_eligible:
                exclusion_counts[reason] += 1
                continue
            fold_eligible_count += 1
            if shard_for_code(str(record["code_sha256"]), shard_count) != shard_index:
                continue
            shard_eligible_count += 1
            chunk_records.append(record)
            if len(chunk_records) == chunk_size:
                summaries = process_chunk(chunk_records, chunk_index)
                register_summaries(summaries)
                chunk_records = []
                chunk_index += 1
        if chunk_records:
            summaries = process_chunk(chunk_records, chunk_index)
            register_summaries(summaries)
            chunk_index += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    if shard_eligible_count <= 0 or best_summary is None or best_code is None:
        raise ProtocolError("no eligible universal-search candidates")
    if total_evaluations != shard_eligible_count * len(rows):
        raise ProtocolError("full development evaluation workload drift")
    if sha256_text(best_code) != best_summary["code_sha256"]:
        raise ProtocolError("selected code hash drift")
    selected_record = {
        "target_seed": seed,
        "code_sha256": best_summary["code_sha256"],
        "source_seeds": best_summary["source_seeds"],
        "source_uuids": best_summary["source_uuids"],
        "source_mission_sha256": best_summary["source_mission_sha256"],
        "development_summary": best_summary,
    }
    is_eligible, reason = eligible_for_target(selected_record, seed, target_hashes)
    if not is_eligible:
        raise ProtocolError(f"selected solver violates target isolation: {reason}")

    selected_solver_path = in_progress / "selected_solver.py"
    selected_solver_path.write_text(best_code)
    write_json(in_progress / "selected.json", selected_record)
    manifest = {
        "status": "complete",
        **run_header,
        "global_pool_candidates": int(pool_manifest["global_unique_codes"]),
        "fold_eligible_candidates": fold_eligible_count,
        "eligible_candidates": shard_eligible_count,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "development_rows": len(rows),
        "development_evaluations": total_evaluations,
        "development_chunks": chunk_index,
        "proved_infeasible_development_rows": sum(
            status == "INFEASIBLE" for status in statuses.values()
        ),
        "finite_bound_development_rows": sum(
            value is not None for value in references.values()
        ),
        "duplicate_selection_rows": total_duplicate_rows,
        "duplicate_id_occurrences": total_duplicate_occurrences,
        "false_feasible_on_proved_infeasible": total_false_feasible,
        "aggregate_error_counts": dict(sorted(aggregate_errors.items())),
        "selection_rule": protocol["development"]["selection"],
        "selected_code_sha256": best_summary["code_sha256"],
        "selected_solver_sha256": sha256_file(selected_solver_path),
        "development_summaries_sha256": sha256_file(merged_summary_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(in_progress / "selection_manifest.json", manifest)
    in_progress.replace(output_dir)


def reduce_selection(
    protocol_path: Path, shard_root: Path, output_dir: Path, seed: int
) -> None:
    protocol = load_protocol(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    shard_count = int(protocol["development"]["shards_per_seed"])
    if output_dir.exists():
        raise ProtocolError(f"immutable reduced selection exists: {output_dir}")
    in_progress = output_dir.with_name(output_dir.name + ".inprogress")
    if in_progress.exists():
        raise ProtocolError(f"incomplete reduced selection exists: {in_progress}")

    manifests: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    pool_hashes: set[str] = set()
    fold_counts: set[int] = set()
    for shard_index in range(shard_count):
        root = shard_root / f"shard{shard_index}"
        manifest_path = root / "selection_manifest.json"
        selected_path = root / "selected.json"
        solver_path = root / "selected_solver.py"
        manifest = load_json(manifest_path)
        selected = load_json(selected_path)
        if manifest.get("status") != "complete":
            raise ProtocolError(f"incomplete selection shard: {shard_index}")
        expected = {
            "target_seed": seed,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "protocol_sha256": protocol_sha,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ProtocolError(f"selection shard invariant drift: {key}")
        if manifest.get("selected_code_sha256") != sha256_file(solver_path):
            raise ProtocolError(f"selection shard solver drift: {shard_index}")
        if selected.get("code_sha256") != manifest["selected_code_sha256"]:
            raise ProtocolError(f"selection shard record drift: {shard_index}")
        if shard_for_code(str(selected["code_sha256"]), shard_count) != shard_index:
            raise ProtocolError(f"selection shard partition drift: {shard_index}")
        pool_hashes.add(str(manifest["pool_sha256"]))
        fold_counts.add(int(manifest["fold_eligible_candidates"]))
        manifests.append(manifest)
        selections.append(selected)

    if len(pool_hashes) != 1 or len(fold_counts) != 1:
        raise ProtocolError("selection shards do not share one fold and candidate pool")
    fold_eligible = next(iter(fold_counts))
    if sum(int(item["eligible_candidates"]) for item in manifests) != fold_eligible:
        raise ProtocolError("selection shards do not partition the eligible fold")
    expected_evaluations = fold_eligible * int(protocol["development"]["rows_per_seed"])
    if (
        sum(int(item["development_evaluations"]) for item in manifests)
        != expected_evaluations
    ):
        raise ProtocolError("selection shard evaluation coverage drift")

    winner_index = min(
        range(shard_count),
        key=lambda index: rank_key(selections[index]["development_summary"]),
    )
    winner_root = shard_root / f"shard{winner_index}"
    winner = selections[winner_index]
    target_hashes = _target_test_hashes(protocol, Path.cwd(), seed)
    is_eligible, reason = eligible_for_target(winner, seed, target_hashes)
    if not is_eligible:
        raise ProtocolError(f"reduced solver isolation failure: {reason}")

    in_progress.mkdir(parents=True)
    shutil.copy2(winner_root / "selected_solver.py", in_progress / "selected_solver.py")
    shutil.copy2(winner_root / "selected.json", in_progress / "selected.json")
    write_json(
        in_progress / "selection_manifest.json",
        {
            "status": "complete",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "pool_sha256": next(iter(pool_hashes)),
            "target_seed": seed,
            "shard_count": shard_count,
            "selected_shard_index": winner_index,
            "eligible_candidates": fold_eligible,
            "development_rows": int(protocol["development"]["rows_per_seed"]),
            "development_evaluations": expected_evaluations,
            "selection_rule": protocol["development"]["selection"],
            "selected_code_sha256": winner["code_sha256"],
            "selected_solver_sha256": sha256_file(in_progress / "selected_solver.py"),
            "shards": [
                {
                    "shard_index": index,
                    "eligible_candidates": manifest["eligible_candidates"],
                    "development_evaluations": manifest["development_evaluations"],
                    "selected_code_sha256": manifest["selected_code_sha256"],
                    "selection_manifest_sha256": sha256_file(
                        shard_root / f"shard{index}" / "selection_manifest.json"
                    ),
                }
                for index, manifest in enumerate(manifests)
            ],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )
    in_progress.replace(output_dir)


def freeze_solvers(
    protocol_path: Path, selection_root: Path, test_root: Path, output_dir: Path
) -> None:
    protocol = load_protocol(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    if output_dir.exists():
        raise ProtocolError(f"immutable freeze output already exists: {output_dir}")
    if test_root.exists() and any(test_root.iterdir()):
        raise ProtocolError("test artifacts exist before solver freeze")
    in_progress = output_dir.with_name(output_dir.name + ".inprogress")
    if in_progress.exists():
        raise ProtocolError(f"incomplete freeze output already exists: {in_progress}")
    in_progress.mkdir(parents=True)

    selected: dict[str, Any] = {}
    pool_hashes: set[str] = set()
    for seed in sorted(EXPECTED_SEEDS):
        seed_root = selection_root / f"seed{seed}"
        manifest_path = seed_root / "selection_manifest.json"
        solver_path = seed_root / "selected_solver.py"
        selected_path = seed_root / "selected.json"
        manifest = load_json(manifest_path)
        selection = load_json(selected_path)
        if manifest.get("status") != "complete" or manifest.get("target_seed") != seed:
            raise ProtocolError(f"selection result incomplete for seed {seed}")
        if manifest.get("protocol_sha256") != protocol_sha:
            raise ProtocolError(f"selection protocol drift for seed {seed}")
        if manifest.get("selected_code_sha256") != sha256_file(solver_path):
            raise ProtocolError(f"selected solver hash drift for seed {seed}")
        if selection.get("code_sha256") != manifest["selected_code_sha256"]:
            raise ProtocolError(f"selected record hash drift for seed {seed}")
        target_hashes = _target_test_hashes(protocol, Path.cwd(), seed)
        is_eligible, reason = eligible_for_target(selection, seed, target_hashes)
        if not is_eligible:
            raise ProtocolError(
                f"selected solver isolation failure for seed {seed}: {reason}"
            )
        frozen_seed_root = in_progress / f"seed{seed}"
        frozen_seed_root.mkdir()
        shutil.copy2(solver_path, frozen_seed_root / "selected_solver.py")
        shutil.copy2(selected_path, frozen_seed_root / "selected.json")
        pool_hashes.add(str(manifest["pool_sha256"]))
        selected[str(seed)] = {
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": sha256_file(manifest_path),
            "selected_code_sha256": manifest["selected_code_sha256"],
            "eligible_candidates": manifest["eligible_candidates"],
            "development_evaluations": manifest["development_evaluations"],
        }
    if len(pool_hashes) != 1:
        raise ProtocolError("selection folds did not use one immutable candidate pool")
    write_json(
        in_progress / "freeze_manifest.json",
        {
            "status": "complete",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "candidate_pool_sha256": next(iter(pool_hashes)),
            "test_outcomes_opened": False,
            "selected": selected,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )
    in_progress.replace(output_dir)


def register_test_result(
    protocol_path: Path,
    freeze_root: Path,
    test_seed_root: Path,
    seed: int,
    evaluation_job_id: str,
) -> None:
    protocol = load_protocol(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    freeze_manifest_path = freeze_root / "freeze_manifest.json"
    freeze_manifest = load_json(freeze_manifest_path)
    if freeze_manifest.get("status") != "complete":
        raise ProtocolError("test registration requires a complete freeze")
    if freeze_manifest.get("protocol_sha256") != protocol_sha:
        raise ProtocolError("test registration protocol drift")
    selected_solver = freeze_root / f"seed{seed}" / "selected_solver.py"
    selected_sha = sha256_file(selected_solver)
    if freeze_manifest["selected"][str(seed)]["selected_code_sha256"] != selected_sha:
        raise ProtocolError("test registration solver hash drift")

    evaluation_root = test_seed_root / "test-evaluation"
    metrics_path = evaluation_root / "metrics_final.csv"
    metadata_path = evaluation_root / "experiment_metadata.json"
    timing_path = evaluation_root / "timing_summary.json"
    duplicate_audit_path = evaluation_root / "duplicate_selection_audit.json"
    for path in (metrics_path, metadata_path, timing_path, duplicate_audit_path):
        if not path.is_file():
            raise ProtocolError(f"test registration artifact missing: {path}")
    manifest_path = test_seed_root / "test_manifest.json"
    if manifest_path.exists():
        raise ProtocolError(f"immutable test manifest exists: {manifest_path}")

    metadata = load_json(metadata_path)
    timing = load_json(timing_path)
    expected_dataset_protocol = load_json(Path(protocol["development"]["protocol"]))
    expected_metadata = {
        "seed": seed,
        "method_name": protocol["test"]["method_name"],
        "dataset": expected_dataset_protocol["datasets"][str(seed)],
        "dataset_revision": expected_dataset_protocol["dataset_revisions"][str(seed)],
        "fixed_code_file": str(selected_solver),
        "code_source_type": "input-disjoint-development-selected-base-pool",
        "code_source_path": str(freeze_manifest_path),
        "code_source_seed": seed,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ProtocolError(f"test metadata drift: {key}")
    if int(timing.get("num_records_evaluated", -1)) != int(
        protocol["test"]["rows_per_seed_before_clean_filter"]
    ):
        raise ProtocolError("test timing row count drift")

    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = int(protocol["test"]["rows_per_seed_before_clean_filter"])
    if len(rows) != expected_rows:
        raise ProtocolError(f"test metrics row count drift: {len(rows)}")
    uuids = [str(row["uuid"]) for row in rows]
    if len(set(uuids)) != expected_rows:
        raise ProtocolError("test metrics UUID uniqueness drift")
    duplicate_rows = sum(
        int(row.get("duplicate_selection_count") or 0) > 0 for row in rows
    )
    duplicate_occurrences = sum(
        int(row.get("duplicate_selection_count") or 0) for row in rows
    )
    write_json(
        manifest_path,
        {
            "status": "complete",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "seed": seed,
            "evaluation_job_id": str(evaluation_job_id),
            "registration_job_id": os.environ.get("SLURM_JOB_ID"),
            "freeze_manifest_sha256": sha256_file(freeze_manifest_path),
            "selected_code_sha256": selected_sha,
            "rows": len(rows),
            "unique_uuids": len(set(uuids)),
            "uuid_set_sha256": sha256_text("\n".join(sorted(uuids)) + "\n"),
            "metrics_sha256": sha256_file(metrics_path),
            "experiment_metadata_sha256": sha256_file(metadata_path),
            "timing_summary_sha256": sha256_file(timing_path),
            "duplicate_selection_audit_sha256": sha256_file(duplicate_audit_path),
            "duplicate_selection_rows": duplicate_rows,
            "duplicate_id_occurrences": duplicate_occurrences,
            "execution_repeats": int(protocol["test"]["execution_repeats"]),
            "quality_uses_first_execution": bool(
                protocol["test"]["quality_uses_first_execution"]
            ),
            "regenerated": False,
            "retrained": False,
            "reselected_after_test": False,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-pool")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--protocol", type=Path, required=True)
    select.add_argument("--pool-dir", type=Path, required=True)
    select.add_argument("--development-manifest", type=Path, required=True)
    select.add_argument("--references", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument(
        "--seed", type=int, choices=sorted(EXPECTED_SEEDS), required=True
    )
    select.add_argument("--shard-index", type=int, required=True)

    reduce = subparsers.add_parser("reduce")
    reduce.add_argument("--protocol", type=Path, required=True)
    reduce.add_argument("--shard-root", type=Path, required=True)
    reduce.add_argument("--output-dir", type=Path, required=True)
    reduce.add_argument(
        "--seed", type=int, choices=sorted(EXPECTED_SEEDS), required=True
    )

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--selection-root", type=Path, required=True)
    freeze.add_argument("--test-root", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)

    register = subparsers.add_parser("register-test")
    register.add_argument("--protocol", type=Path, required=True)
    register.add_argument("--freeze-root", type=Path, required=True)
    register.add_argument("--test-seed-root", type=Path, required=True)
    register.add_argument(
        "--seed", type=int, choices=sorted(EXPECTED_SEEDS), required=True
    )
    register.add_argument("--evaluation-job-id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-pool":
        prepare_pool(args.protocol, args.source_root, args.output_dir)
    elif args.command == "select":
        select_solver(
            args.protocol,
            args.pool_dir,
            args.development_manifest,
            args.references,
            args.output_dir,
            args.seed,
            args.shard_index,
        )
    elif args.command == "reduce":
        reduce_selection(args.protocol, args.shard_root, args.output_dir, args.seed)
    elif args.command == "freeze":
        freeze_solvers(
            args.protocol, args.selection_root, args.test_root, args.output_dir
        )
    elif args.command == "register-test":
        register_test_result(
            args.protocol,
            args.freeze_root,
            args.test_seed_root,
            args.seed,
            args.evaluation_job_id,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
