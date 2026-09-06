#!/usr/bin/env python3
"""Validate and register immutable full-throughput generation measurements."""

# ruff: noqa: PLR0913, PLR0915, PLR2004, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SEEDS = (202, 303)
EXPECTED_ROWS = 1000
EXPECTED_ALLOCATED_GPUS = 4
EXPECTED_FILES = (
    "calibration.json",
    "hero_provenance.json",
    "base64_provenance.json",
    "hero_generations_calibration_only.jsonl",
    "base64_generations_calibration_only.jsonl",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_generations(
    path: Path,
    *,
    expected_rows: int,
    samples_per_uuid: int,
    expected_model_revision: str,
    expected_dataset_revision: str,
    expected_prompt_sha256: str,
) -> dict[str, Any]:
    samples: dict[str, set[int]] = defaultdict(set)
    missions: dict[str, str] = {}
    rows = 0
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            uuid = str(row["uuid"])
            mission = str(row["mission"])
            sample_idx = row["sample_idx"]
            if isinstance(sample_idx, bool) or not isinstance(sample_idx, int):
                raise TypeError(f"invalid sample_idx at {path}:{line_number}")
            if row.get("model_revision") != expected_model_revision:
                raise ValueError(f"model revision drift at {path}:{line_number}")
            if row.get("dataset_revision") != expected_dataset_revision:
                raise ValueError(f"dataset revision drift at {path}:{line_number}")
            if row.get("prompt_variant") != "canonical":
                raise ValueError(f"prompt variant drift at {path}:{line_number}")
            if row.get("system_prompt_sha256") != expected_prompt_sha256:
                raise ValueError(f"system prompt drift at {path}:{line_number}")
            if uuid in missions and missions[uuid] != mission:
                raise ValueError(f"UUID maps to multiple missions in {path}: {uuid}")
            missions[uuid] = mission
            if sample_idx in samples[uuid]:
                raise ValueError(f"duplicate ({uuid}, {sample_idx}) in {path}")
            samples[uuid].add(sample_idx)
            rows += 1

    expected_indices = set(range(samples_per_uuid))
    if len(samples) != expected_rows:
        raise ValueError(f"{path} has {len(samples)} UUIDs, expected {expected_rows}")
    for uuid, indices in samples.items():
        if indices != expected_indices:
            raise ValueError(f"incomplete sample indices for {uuid} in {path}")
    if rows != expected_rows * samples_per_uuid:
        raise ValueError(f"{path} has {rows} rows")
    return {
        "rows": rows,
        "unique_uuids": len(samples),
        "samples_per_uuid": samples_per_uuid,
        "uuid_order_sha256": hashlib.sha256(
            ("\n".join(missions) + "\n").encode()
        ).hexdigest(),
        "mission_sha256_by_uuid_sha256": hashlib.sha256(
            json.dumps(
                {
                    uuid: hashlib.sha256(mission.encode()).hexdigest()
                    for uuid, mission in sorted(missions.items())
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }


def split_pin(value: str) -> tuple[str, str]:
    repo_id, revision = value.rsplit("@", 1)
    return repo_id, revision


def validate_provenance(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"provenance {key}={payload.get(key)!r}, expected {value!r}"
            )


def register(  # noqa: PLR0912
    root: Path,
    scheduler_path: Path,
    protocol_path: Path,
    addendum_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable result: {output}")
    scheduler = json.loads(scheduler_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("protocol_id") != "N26-E3-v2":
        raise ValueError("unexpected cost protocol")
    addendum = json.loads(addendum_path.read_text())
    if addendum.get("protocol_id") != "N26-E3-v3-full-throughput":
        raise ValueError("unexpected full-throughput addendum")
    expected_rows = int(addendum["row_count"])
    if expected_rows != EXPECTED_ROWS:
        raise ValueError("full-throughput addendum must declare 1,000 prompts")
    if set(addendum["seeds"]) != set(SEEDS):
        raise ValueError("full-throughput addendum seed set drifted")
    if addendum.get("efficacy_excluded") is not True:
        raise ValueError("full-throughput addendum must exclude efficacy use")
    if int(addendum["hero_completions_per_prompt"]) != 1:
        raise ValueError("full-throughput Hero completion count drifted")
    if int(addendum["base_completions_per_prompt"]) != 64:
        raise ValueError("full-throughput Base completion count drifted")
    generation_protocol = protocol["generation_calibration"]
    prompt_sha256 = generation_protocol["canonical_prompt_sha256"]
    if set(scheduler) != {str(seed) for seed in SEEDS}:
        raise ValueError("scheduler evidence must contain exactly seeds 202 and 303")

    files_sha256: dict[str, str] = {}
    validation: dict[str, Any] = {}
    for seed in SEEDS:
        seed_root = root / f"seed{seed}"
        record = scheduler[str(seed)]
        if record.get("state") != "COMPLETED" or record.get("exit_code") != "0:0":
            raise ValueError(f"seed {seed} scheduler job did not complete cleanly")
        if int(record.get("allocated_gpus", 0)) != EXPECTED_ALLOCATED_GPUS:
            raise ValueError(f"seed {seed} allocation must record four node GPUs")
        if int(record.get("elapsed_seconds", 0)) <= 0:
            raise ValueError(f"seed {seed} scheduler elapsed time is invalid")

        for name in EXPECTED_FILES:
            path = seed_root / name
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"missing full calibration artifact: {path}")
            files_sha256[str(path.relative_to(root))] = sha256_file(path)

        calibration = json.loads((seed_root / "calibration.json").read_text())
        expected = {
            "protocol_id": "N26-E3-v2",
            "calibration_scope": "full-test-throughput",
            "seed": seed,
            "row_count": expected_rows,
            "hero_completion_count": expected_rows,
            "base_completion_count": expected_rows * 64,
            "active_generation_gpus": 1,
            "slurm_allocation_gpus": 4,
            "efficacy_excluded": True,
            "slurm_job_id": str(record["job_id"]),
        }
        for key, value in expected.items():
            if calibration.get(key) != value:
                raise ValueError(
                    f"seed {seed} calibration {key}={calibration.get(key)!r}, expected {value!r}"
                )
        for field in ("hero_generation_wall_seconds", "base64_generation_wall_seconds"):
            if float(calibration.get(field, 0)) <= 0:
                raise ValueError(f"seed {seed} has invalid {field}")

        hero_provenance = json.loads((seed_root / "hero_provenance.json").read_text())
        base_provenance = json.loads((seed_root / "base64_provenance.json").read_text())
        dataset_repo, dataset_revision = split_pin(
            generation_protocol["dataset_revisions"][str(seed)]
        )
        hero_repo, hero_revision = split_pin(
            generation_protocol["hero"]["model_revisions"][str(seed)]
        )
        base_model = generation_protocol["base_model"]
        common = {
            "schema_version": 1,
            "dataset": dataset_repo,
            "dataset_revision": dataset_revision,
            "dataset_split": generation_protocol["split"],
            "prompt_variant": "canonical",
            "canonical_system_prompt_sha256": prompt_sha256,
            "effective_system_prompt_sha256": prompt_sha256,
            "max_samples": expected_rows,
        }
        validate_provenance(
            hero_provenance,
            {
                **common,
                "model_path": hero_repo,
                "model_revision": hero_revision,
                "temperature": generation_protocol["hero"]["temperature"],
                "n_samples": generation_protocol["hero"]["completions_per_prompt"],
            },
        )
        validate_provenance(
            base_provenance,
            {
                **common,
                "model_path": base_model["repo_id"],
                "model_revision": base_model["revision"],
                "temperature": base_model["temperature"],
                "n_samples": base_model["completions_per_prompt"],
            },
        )
        if hero_provenance["dataset_revision"] != base_provenance["dataset_revision"]:
            raise ValueError(f"seed {seed} dataset provenance differs by method")
        validation[str(seed)] = {
            "hero": validate_generations(
                seed_root / "hero_generations_calibration_only.jsonl",
                expected_rows=expected_rows,
                samples_per_uuid=1,
                expected_model_revision=hero_provenance["model_revision"],
                expected_dataset_revision=hero_provenance["dataset_revision"],
                expected_prompt_sha256=prompt_sha256,
            ),
            "base64": validate_generations(
                seed_root / "base64_generations_calibration_only.jsonl",
                expected_rows=expected_rows,
                samples_per_uuid=64,
                expected_model_revision=base_provenance["model_revision"],
                expected_dataset_revision=base_provenance["dataset_revision"],
                expected_prompt_sha256=prompt_sha256,
            ),
        }
        if (
            validation[str(seed)]["hero"]["mission_sha256_by_uuid_sha256"]
            != validation[str(seed)]["base64"]["mission_sha256_by_uuid_sha256"]
        ):
            raise ValueError(f"seed {seed} Hero and Base prompt identities differ")
        if (
            validation[str(seed)]["hero"]["uuid_order_sha256"]
            != validation[str(seed)]["base64"]["uuid_order_sha256"]
        ):
            raise ValueError(f"seed {seed} Hero and Base prompt order differs")

    result = {
        "protocol_id": "N26-E3-v3-full-throughput-result",
        "status": "complete",
        "artifact_root": str(root),
        "expected_row_count": expected_rows,
        "efficacy_excluded": True,
        "scheduler": scheduler,
        "scheduler_sha256": sha256_file(scheduler_path),
        "source_protocol": str(protocol_path),
        "source_protocol_sha256": sha256_file(protocol_path),
        "full_throughput_addendum": str(addendum_path),
        "full_throughput_addendum_sha256": sha256_file(addendum_path),
        "validation": validation,
        "files_sha256": files_sha256,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            register(
                args.root,
                args.scheduler,
                args.protocol,
                args.addendum,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
