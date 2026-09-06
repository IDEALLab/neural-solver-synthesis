#!/usr/bin/env python3
"""Account prompt and completion tokens for frozen synthesis experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def summarize_file(
    path: Path, tokenizer: Any, *, seed: int, expected_completions: int
) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != expected_completions:
        raise ValueError(
            f"{path} has {len(rows)} completions, expected {expected_completions}"
        )
    prompt_tokens = []
    completion_tokens = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        completion = row.get("generated_text")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise ValueError(f"{path} row {index} lacks prompt/generated_text")
        prompt_tokens.append(count_tokens(tokenizer, prompt))
        completion_tokens.append(count_tokens(tokenizer, completion))
    return {
        "seed": seed,
        "path": str(path),
        "sha256": sha256_file(path),
        "completion_count": len(rows),
        "prompt_tokens": sum(prompt_tokens),
        "generated_tokens": sum(completion_tokens),
        "total_tokens": sum(prompt_tokens) + sum(completion_tokens),
        "prompt_tokens_min": min(prompt_tokens),
        "prompt_tokens_max": max(prompt_tokens),
        "generated_tokens_min": min(completion_tokens),
        "generated_tokens_max": max(completion_tokens),
    }


def audit_protocol(protocol_path: Path, tokenizer: Any) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    conditions: dict[str, Any] = {}
    for condition, spec in protocol["conditions"].items():
        rows = [
            summarize_file(
                Path(path),
                tokenizer,
                seed=int(seed),
                expected_completions=int(spec["expected_completions_per_seed"]),
            )
            for seed, path in sorted(spec["files"].items())
        ]
        conditions[condition] = {
            "seeds": rows,
            "completion_count": sum(row["completion_count"] for row in rows),
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "generated_tokens": sum(row["generated_tokens"] for row in rows),
            "total_tokens": sum(row["total_tokens"] for row in rows),
        }
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "tokenizer": protocol["tokenizer"],
        "conditions": conditions,
        "status": "complete",
        "interpretation_policy": protocol["interpretation_policy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite immutable output: {args.output}")

    protocol = json.loads(args.protocol.read_text())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        protocol["tokenizer"]["repo_id"],
        revision=protocol["tokenizer"]["revision"],
    )
    result = audit_protocol(args.protocol, tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
