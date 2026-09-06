#!/usr/bin/env python3
"""Measure exact canonical-mission overlap across pinned SDS dataset splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.sds.adaptive_repair import mission_sha256


def summarize_split_overlap(
    source_hashes: list[str], target_hashes: list[str]
) -> dict[str, Any]:
    source_unique = set(source_hashes)
    target_counts = Counter(target_hashes)
    overlap = sorted(source_unique & set(target_counts))
    return {
        "source_rows": len(source_hashes),
        "source_unique_missions": len(source_unique),
        "source_duplicate_rows": len(source_hashes) - len(source_unique),
        "target_rows": len(target_hashes),
        "target_unique_missions": len(target_counts),
        "target_duplicate_rows": len(target_hashes) - len(target_counts),
        "overlap_unique_missions": len(overlap),
        "target_rows_with_source_content": sum(target_counts[value] for value in overlap),
        "overlap_mission_sha256": overlap,
    }


def audit(protocol_path: Path, kind: str) -> dict[str, Any]:
    from datasets import load_dataset

    protocol = json.loads(protocol_path.read_text())
    hashes: dict[int, dict[str, list[str]]] = {}
    for seed_text, dataset in protocol["datasets"].items():
        if kind == "adaptive":
            repo_id = dataset
            revision = protocol["dataset_revisions"][seed_text]
        else:
            repo_id = dataset["repo_id"]
            revision = dataset["revision"]
        dataset_hashes: dict[str, list[str]] = {}
        for split in ("train", protocol["development_split"], protocol["test_split"]):
            rows = load_dataset(repo_id, split=split, revision=revision)
            dataset_hashes[split] = [mission_sha256(row["mission"]) for row in rows]
        hashes[int(seed_text)] = dataset_hashes

    comparisons: dict[str, Any] = {}
    for source_seed, split_hashes in sorted(hashes.items()):
        for source_split in ("train", protocol["development_split"]):
            for test_seed, test_split_hashes in sorted(hashes.items()):
                key = f"{source_split}_seed{source_seed}_test_seed{test_seed}"
                comparisons[key] = summarize_split_overlap(
                    split_hashes[source_split],
                    test_split_hashes[protocol["test_split"]],
                )

    matching_seed_train_test_rows = sum(
        comparisons[f"train_seed{seed}_test_seed{seed}"][
            "target_rows_with_source_content"
        ]
        for seed in hashes
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "status": (
            "disjoint"
            if matching_seed_train_test_rows == 0
            else "matching_seed_train_test_overlap_detected"
        ),
        "matching_seed_train_test_rows_with_overlap": matching_seed_train_test_rows,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--kind", choices=("adaptive", "frozen-hero"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.protocol, args.kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
