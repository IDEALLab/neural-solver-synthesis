# ruff: noqa: TRY003
"""Freeze a deterministic stratified sample for manual program-taxonomy review."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_key(salt: str, condition: str, family: str, code_hash: str) -> str:
    payload = f"{salt}|{condition}|{family}|{code_hash}".encode()
    return hashlib.sha256(payload).hexdigest()


def sample_condition(
    paths: Iterable[Path], condition: str, *, per_stratum: int, salt: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_paths = list(paths)
    if not source_paths:
        raise ValueError(f"no sources for condition: {condition}")
    rows: list[dict[str, Any]] = []
    for path in source_paths:
        source_rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        source_hashes = [str(row.get("code_sha256") or "") for row in source_rows]
        if len(source_hashes) != len(set(source_hashes)):
            raise ValueError(f"duplicate code hash within {path}")
        rows.extend(source_rows)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_hashes: set[str] = set()
    for row in rows:
        code_hash = str(row.get("code_sha256") or "")
        family = str(row.get("algorithm_family") or "")
        code = str(row.get("canonical_code") or "")
        if not code_hash or not family or not code:
            raise ValueError(f"incomplete unique-program record for {condition}")
        if code_hash in seen_hashes:
            continue
        seen_hashes.add(code_hash)
        by_family[family].append(row)

    selected: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    for family, family_rows in sorted(by_family.items()):
        ranked = sorted(
            family_rows,
            key=lambda row: (
                rank_key(salt, condition, family, row["code_sha256"]),
                row["code_sha256"],
            ),
        )
        sampled = ranked[:per_stratum]
        counts[family] = {
            "population_unique_programs": len(family_rows),
            "sampled_programs": len(sampled),
        }
        selected.extend(
            [
                {
                    "condition": condition,
                    "automatic_algorithm_family": family,
                    "automatic_sa_acceptance": row["sa_acceptance"],
                    "code_sha256": row["code_sha256"],
                    "selection_rank_sha256": rank_key(
                        salt, condition, family, row["code_sha256"]
                    ),
                    "canonical_code": row["canonical_code"],
                }
                for row in sampled
            ]
        )
    return selected, {
        "sources": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in source_paths
        ],
        "source_rows": len(rows),
        "deduplicated_unique_programs": len(seen_hashes),
        "strata": counts,
    }


def freeze_sample(
    inputs: dict[str, list[Path]], output_dir: Path, *, per_stratum: int, salt: str
) -> dict[str, Any]:
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    samples: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for condition, paths in sorted(inputs.items()):
        selected, source = sample_condition(
            paths, condition, per_stratum=per_stratum, salt=salt
        )
        samples.extend(selected)
        sources[condition] = source

    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("x") as handle:
        for index, row in enumerate(samples):
            handle.write(json.dumps({"sample_id": index, **row}, sort_keys=True) + "\n")
    manifest = {
        "protocol_id": "N26-E7-manual-taxonomy-audit-v1",
        "status": "sample_frozen",
        "salt": salt,
        "per_stratum": per_stratum,
        "sample_count": len(samples),
        "samples_sha256": sha256_file(samples_path),
        "sources": sources,
    }
    (output_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", required=True, metavar="CONDITION=PATH"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--salt", default="N26-E7-manual-taxonomy-audit-v1")
    args = parser.parse_args()
    inputs: dict[str, list[Path]] = defaultdict(list)
    for value in args.input:
        if "=" not in value:
            parser.error(f"invalid --input: {value}")
        condition, path = value.split("=", 1)
        if not condition:
            parser.error(f"invalid --input condition: {condition}")
        inputs[condition].append(Path(path))
    manifest = freeze_sample(
        inputs,
        args.output_dir,
        per_stratum=args.per_stratum,
        salt=args.salt,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
