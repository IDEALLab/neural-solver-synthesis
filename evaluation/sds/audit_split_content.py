#!/usr/bin/env python3
"""Audit pinned SDS development manifests against canonical test content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.sds.adaptive_repair import mission_sha256


def overlap_summary(
    development_hashes_by_seed: dict[int, set[str]],
    test_hashes_by_seed: dict[int, set[str]],
) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    total_overlaps: set[str] = set()
    for development_seed, development_hashes in sorted(
        development_hashes_by_seed.items()
    ):
        for test_seed, test_hashes in sorted(test_hashes_by_seed.items()):
            overlap = sorted(development_hashes & test_hashes)
            total_overlaps.update(overlap)
            pairs[f"development_seed{development_seed}_test_seed{test_seed}"] = {
                "development_unique_missions": len(development_hashes),
                "test_unique_missions": len(test_hashes),
                "overlap_unique_missions": len(overlap),
                "overlap_mission_sha256": overlap,
            }
    return {
        "status": "disjoint" if not total_overlaps else "overlap_detected",
        "overlap_unique_missions_across_all_pairs": len(total_overlaps),
        "pairs": pairs,
    }


def _load_test_hashes(protocol: dict[str, Any], frozen_kind: str) -> dict[int, set[str]]:
    from datasets import load_dataset

    test_hashes: dict[int, set[str]] = {}
    for seed_text, dataset in protocol["datasets"].items():
        if frozen_kind == "adaptive":
            repo_id = dataset
            revision = protocol["dataset_revisions"][seed_text]
        else:
            repo_id = dataset["repo_id"]
            revision = dataset["revision"]
        rows = load_dataset(
            repo_id,
            split=protocol["test_split"],
            revision=revision,
        )
        test_hashes[int(seed_text)] = {
            mission_sha256(row["mission"]) for row in rows
        }
    return test_hashes


def _manifest_hashes(
    manifest: dict[str, Any], frozen_kind: str
) -> dict[str, dict[int, set[str]]]:
    sections = ("development",) if frozen_kind == "adaptive" else ("synthesis", "selection")
    result: dict[str, dict[int, set[str]]] = {}
    for section in sections:
        by_seed: dict[int, set[str]] = {}
        for seed_text, seed_manifest in manifest["seeds"].items():
            key = "rows" if frozen_kind == "adaptive" else f"{section}_rows"
            by_seed[int(seed_text)] = {
                str(row["mission_sha256"]) for row in seed_manifest[key]
            }
        result[section] = by_seed
    return result


def audit(
    protocol_path: Path,
    manifest_path: Path,
    frozen_kind: str,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("protocol/manifest ID mismatch")
    test_hashes = _load_test_hashes(protocol, frozen_kind)
    sections = {
        section: overlap_summary(hashes, test_hashes)
        for section, hashes in _manifest_hashes(manifest, frozen_kind).items()
    }
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "manifest_path": str(manifest_path),
        "kind": frozen_kind,
        "status": (
            "disjoint"
            if all(section["status"] == "disjoint" for section in sections.values())
            else "overlap_detected"
        ),
        "sections": sections,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kind", choices=("adaptive", "frozen-hero"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.protocol, args.manifest, args.kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
