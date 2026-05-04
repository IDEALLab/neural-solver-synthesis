#!/usr/bin/env python3
"""Verify pushed Shinka SDS datasets and their run metadata."""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a Shinka SDS dataset on Hugging Face."
    )
    parser.add_argument("--dataset", required=True, help="Dataset repo id.")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Expected number of rows in the train split.",
    )
    parser.add_argument(
        "--require-run-config",
        action="store_true",
        help="Fail if run_config.json is missing.",
    )
    args = parser.parse_args()

    try:
        dataset = load_dataset(args.dataset, split="train")
    except Exception as exc:
        print(f"❌ Failed to load dataset {args.dataset}: {exc}")
        return 1

    row_count = len(dataset)
    print(f"✅ Loaded {args.dataset}")
    print(f"   row_count: {row_count}")

    if args.expected_count is not None and row_count != args.expected_count:
        print(
            f"❌ Dataset row count mismatch: expected {args.expected_count}, got {row_count}"
        )
        return 1

    try:
        run_config_path = hf_hub_download(
            repo_id=args.dataset,
            filename="run_config.json",
            repo_type="dataset",
        )
        run_config = json.loads(Path(run_config_path).read_text())
        print("✅ Found run_config.json")
        for key in [
            "prompt_variant",
            "push_to",
            "source_dataset",
            "target_dataset",
            "num_output_samples",
        ]:
            if key in run_config:
                print(f"   {key}: {run_config[key]}")
    except Exception as exc:
        if args.require_run_config:
            print(f"❌ Missing or unreadable run_config.json for {args.dataset}: {exc}")
            return 1
        print(f"⚠️  Could not read run_config.json for {args.dataset}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
