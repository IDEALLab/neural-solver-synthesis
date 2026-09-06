#!/usr/bin/env python3
"""Generate the exact per-seed N26-E12 synthetic dataset splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.problem_suite.common import write_jsonl
from data.problem_suite.schemas import validate_problem_record, validate_prompt_record
from data.problem_suite.tsp import tsp_render_prompt, tsp_sample


def canonical_sha256(records: list[dict]) -> str:
    payload = "\n".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def split_counts(
    train_count: int, validation_count: int, test_count: int
) -> dict[str, int]:
    counts = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }
    if train_count < 1 or validation_count < 1 or test_count < 1:
        raise ValueError("all TSP split counts must be positive")
    return counts


def generate(
    seed: int,
    output_dir: Path,
    *,
    train_count: int = 10_000,
    validation_count: int = 1_000,
    test_count: int = 1_000,
    protocol_id: str = "N26-E12-v1",
) -> dict:
    counts = split_counts(train_count, validation_count, test_count)
    records = tsp_sample(sum(counts.values()), seed=seed)
    split_records = {}
    cursor = 0
    for split, count in counts.items():
        split_records[split] = records[cursor : cursor + count]
        cursor += count

    prefix = f"tsp_zero_sft_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_id": protocol_id,
        "seed": seed,
        "counts": counts,
        "splits": {},
    }
    for split, rows in split_records.items():
        prompts = [tsp_render_prompt(row) for row in rows]
        if not all(validate_problem_record(row) for row in rows):
            raise ValueError(f"invalid TSP problem in {split}")
        if not all(validate_prompt_record(row) for row in prompts):
            raise ValueError(f"invalid TSP prompt in {split}")
        problem_path = output_dir / f"{prefix}_problems_{split}.jsonl"
        prompt_path = output_dir / f"{prefix}_prompts_{split}.jsonl"
        write_jsonl(problem_path, rows)
        write_jsonl(prompt_path, prompts)
        manifest["splits"][split] = {
            "problem_sha256": canonical_sha256(rows),
            "prompt_sha256": canonical_sha256(prompts),
            "problem_file_sha256": hashlib.sha256(
                problem_path.read_bytes()
            ).hexdigest(),
            "prompt_file_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        }
    manifest_path = output_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def save_dataset_dict(
    seed: int,
    output_dir: Path,
    *,
    train_count: int = 10_000,
    validation_count: int = 1_000,
    test_count: int = 1_000,
    protocol_id: str = "N26-E12-v1",
) -> dict:
    """Materialize a local DatasetDict with a content-addressed revision manifest."""
    from datasets import Dataset, DatasetDict

    counts = split_counts(train_count, validation_count, test_count)
    records = tsp_sample(sum(counts.values()), seed=seed)
    cursor = 0
    prompt_splits = {}
    split_hashes = {}
    for split, count in counts.items():
        rows = records[cursor : cursor + count]
        cursor += count
        prompts = [tsp_render_prompt(row) for row in rows]
        prompt_splits[split] = Dataset.from_list(prompts)
        split_hashes[split] = canonical_sha256(prompts)
    revision = hashlib.sha256(
        json.dumps(split_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    DatasetDict(prompt_splits).save_to_disk(str(output_dir))
    manifest = {
        "protocol_id": protocol_id,
        "seed": seed,
        "counts": counts,
        "split_sha256": split_hashes,
        "dataset_revision": revision,
    }
    (output_dir / "n26_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(101, 202, 303), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-to-disk", action="store_true")
    parser.add_argument("--train-count", type=int, default=10_000)
    parser.add_argument("--validation-count", type=int, default=1_000)
    parser.add_argument("--test-count", type=int, default=1_000)
    parser.add_argument("--protocol-id", default="N26-E12-v1")
    args = parser.parse_args()
    operation = save_dataset_dict if args.save_to_disk else generate
    print(
        json.dumps(
            operation(
                args.seed,
                args.output_dir,
                train_count=args.train_count,
                validation_count=args.validation_count,
                test_count=args.test_count,
                protocol_id=args.protocol_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
