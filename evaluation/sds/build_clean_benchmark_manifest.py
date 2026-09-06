#!/usr/bin/env python3
"""Freeze a test subset disjoint from every pinned SDS training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.sds.adaptive_repair import family_from_uuid, mission_sha256, sha256_text

GLOBAL_DEDUPLICATION_SALT = "N26-SDS-clean-test-v2-global-deduplication"


def deduplicate_across_seeds(seeds: dict[str, Any]) -> set[str]:
    """Keep one deterministic test row for mission content repeated across seeds."""
    ranked: dict[str, list[tuple[str, str, dict[str, str]]]] = {}
    for seed_text, seed in seeds.items():
        for row in seed["retained"]:
            rank = sha256_text(
                f"{GLOBAL_DEDUPLICATION_SALT}:{seed_text}:{row['uuid']}:"
                f"{row['mission_sha256']}"
            )
            ranked.setdefault(row["mission_sha256"], []).append(
                (rank, seed_text, row)
            )

    winners = {
        mission_hash: min(candidates)[0:2]
        for mission_hash, candidates in ranked.items()
    }
    duplicate_hashes = {
        mission_hash for mission_hash, candidates in ranked.items() if len(candidates) > 1
    }
    for seed_text, seed in seeds.items():
        retained = []
        for row in seed["retained"]:
            rank = sha256_text(
                f"{GLOBAL_DEDUPLICATION_SALT}:{seed_text}:{row['uuid']}:"
                f"{row['mission_sha256']}"
            )
            if winners[row["mission_sha256"]] == (rank, seed_text):
                retained.append(row)
            else:
                seed["excluded"].append(
                    {**row, "exclusion_reason": "cross_seed_duplicate"}
                )
        seed["retained"] = retained
        seed["retained_rows"] = len(retained)
        seed["excluded_rows"] = len(seed["excluded"])
    return duplicate_hashes


def build(protocol_path: Path) -> dict[str, Any]:
    from datasets import load_dataset

    protocol = json.loads(protocol_path.read_text())
    datasets: dict[str, tuple[str, str]] = {}
    training_hashes: set[str] = set()
    for seed_text, dataset in protocol["datasets"].items():
        repo_id = dataset if isinstance(dataset, str) else dataset["repo_id"]
        revision = (
            protocol["dataset_revisions"][seed_text]
            if isinstance(dataset, str)
            else dataset["revision"]
        )
        datasets[seed_text] = (repo_id, revision)
        train = load_dataset(repo_id, split="train", revision=revision)
        training_hashes.update(mission_sha256(row["mission"]) for row in train)

    seeds: dict[str, Any] = {}
    for seed_text, (repo_id, revision) in datasets.items():
        test = load_dataset(
            repo_id,
            split=protocol["test_split"],
            revision=revision,
        )
        retained: list[dict[str, str]] = []
        excluded: list[dict[str, str]] = []
        for row in test:
            record = {
                "uuid": str(row["uuid"]),
                "family": family_from_uuid(str(row["uuid"])),
                "mission_sha256": mission_sha256(row["mission"]),
            }
            if record["mission_sha256"] in training_hashes:
                excluded.append({**record, "exclusion_reason": "training_overlap"})
            else:
                retained.append(record)
        retained_hashes = [row["mission_sha256"] for row in retained]
        if len(retained_hashes) != len(set(retained_hashes)):
            raise ValueError(f"clean test subset still has duplicate missions for seed {seed_text}")
        seeds[seed_text] = {
            "dataset": repo_id,
            "dataset_revision": revision,
            "split": protocol["test_split"],
            "original_rows": len(test),
            "retained_rows": len(retained),
            "excluded_rows": len(excluded),
            "retained": retained,
            "excluded": excluded,
        }

    cross_seed_duplicates = deduplicate_across_seeds(seeds)
    all_retained_hashes = [
        row["mission_sha256"]
        for seed in seeds.values()
        for row in seed["retained"]
    ]
    if len(all_retained_hashes) != len(set(all_retained_hashes)):
        raise ValueError("clean test subset still has cross-seed duplicate missions")
    for seed in seeds.values():
        if seed["original_rows"] != seed["retained_rows"] + seed["excluded_rows"]:
            raise ValueError("clean test manifest lost or duplicated source rows")

    return {
        "protocol_id": "N26-SDS-clean-test-v2",
        "source_protocol_id": protocol["protocol_id"],
        "source_protocol_sha256": sha256_text(protocol_path.read_text()),
        "policy": (
            "Exclude a test row when its canonical mission occurs in any of the three "
            "pinned SDS training splits; then retain exactly one deterministic row for "
            "mission content repeated across test seeds."
        ),
        "join_keys": ["seed", "uuid"],
        "training_union_unique_missions": len(training_hashes),
        "excluded_training_mission_sha256": sorted(
            {
                row["mission_sha256"]
                for seed in seeds.values()
                for row in seed["excluded"]
                if row["exclusion_reason"] == "training_overlap"
            }
        ),
        "excluded_cross_seed_duplicate_mission_sha256": sorted(
            cross_seed_duplicates
        ),
        "retained_global_unique_missions": len(all_retained_hashes),
        "seeds": seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")
    manifest = build(args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "protocol_id": manifest["protocol_id"],
                "retained_rows": {
                    seed: data["retained_rows"] for seed, data in manifest["seeds"].items()
                },
                "excluded_rows": {
                    seed: data["excluded_rows"] for seed, data in manifest["seeds"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
