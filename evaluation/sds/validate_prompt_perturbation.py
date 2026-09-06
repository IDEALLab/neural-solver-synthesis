"""Validate immutable inputs and outputs for the N26-E6 prompt perturbation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from evaluation.sds.prompt_variants import (
    HYPOTHESIZE_INSTRUCTION,
    apply_prompt_variant,
    sha256_text,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


class PerturbationValidationError(ValueError):
    """Raised when an immutable N26-E6 input or output fails validation."""

    def __init__(self, check: str, actual: Any, expected: Any) -> None:
        super().__init__(f"{check}: {actual!r} != {expected!r}")


def require_equal(check: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise PerturbationValidationError(check, actual, expected)


def validate_generation_row(
    row: dict[str, Any],
    protocol: dict[str, Any],
    seed_text: str,
    checkpoint: dict[str, Any],
) -> None:
    require_equal("prompt_variant", row.get("prompt_variant"), protocol["condition"])
    require_equal(
        "system_prompt_sha256",
        row.get("system_prompt_sha256"),
        protocol["perturbed_system_prompt_sha256"],
    )
    require_equal(
        "dataset_revision",
        row.get("dataset_revision"),
        protocol["dataset_revisions"][seed_text],
    )
    require_equal(
        "model_revision", row.get("model_revision"), checkpoint["revision"]
    )
    if HYPOTHESIZE_INSTRUCTION in row.get("prompt", ""):
        raise PerturbationValidationError(
            "hypothesize_instruction_present", True, False
        )


def validate_run(
    protocol_path: Path,
    seed: int,
    config_path: Path,
    provenance_path: Path,
    generations_path: Path,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    provenance = load_json(provenance_path)
    seed_text = str(seed)
    checkpoint = protocol["checkpoints"][seed_text]

    config_sha256 = sha256_file(config_path)
    require_equal(
        "system_prompt_config_sha256",
        config_sha256,
        protocol["system_prompt_config_sha256"],
    )

    canonical_prompt = yaml.safe_load(config_path.read_text())["system_prompt"].strip()
    effective_prompt = apply_prompt_variant(
        canonical_prompt, protocol["condition"]
    )
    require_equal(
        "removed_instruction",
        protocol["removed_instruction"],
        HYPOTHESIZE_INSTRUCTION,
    )
    require_equal(
        "canonical_system_prompt_sha256",
        sha256_text(canonical_prompt),
        protocol["canonical_system_prompt_sha256"],
    )
    require_equal(
        "perturbed_system_prompt_sha256",
        sha256_text(effective_prompt),
        protocol["perturbed_system_prompt_sha256"],
    )

    expected_provenance = {
        "model_path": checkpoint["repo_id"],
        "model_revision": checkpoint["revision"],
        "dataset": protocol["datasets"][seed_text],
        "dataset_revision": protocol["dataset_revisions"][seed_text],
        "config_file_sha256": protocol["system_prompt_config_sha256"],
        "prompt_variant": protocol["condition"],
        "canonical_system_prompt_sha256": protocol[
            "canonical_system_prompt_sha256"
        ],
        "effective_system_prompt_sha256": protocol[
            "perturbed_system_prompt_sha256"
        ],
        "temperature": protocol["sampling"]["temperature"],
        "n_samples": protocol["sampling"]["n_samples"],
    }
    for key, expected in expected_provenance.items():
        require_equal(key, provenance.get(key), expected)

    rows = [json.loads(line) for line in generations_path.read_text().splitlines() if line]
    expected_count = protocol["expected_count"] * protocol["sampling"]["n_samples"]
    require_equal("generation_row_count", len(rows), expected_count)

    uuids = [row.get("uuid") for row in rows]
    require_equal("missing_uuid_count", uuids.count(None), 0)
    require_equal("unique_uuid_count", len(set(uuids)), protocol["expected_count"])

    for row in rows:
        validate_generation_row(row, protocol, seed_text, checkpoint)

    return {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "row_count": len(rows),
        "unique_uuid_count": len(set(uuids)),
        "config_sha256": config_sha256,
        "canonical_system_prompt_sha256": sha256_text(canonical_prompt),
        "effective_system_prompt_sha256": sha256_text(effective_prompt),
        "generations_sha256": sha256_file(generations_path),
        "status": "validated",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[101, 202, 303], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = validate_run(
        args.protocol,
        args.seed,
        args.config,
        args.provenance,
        args.generations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
