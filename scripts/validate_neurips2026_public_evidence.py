#!/usr/bin/env python3
"""Validate the compact public NeurIPS 2026 evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".tex", ".toml", ".yaml", ".yml"}
FORBIDDEN_FILENAMES = {
    ".env",
    "hf_token.txt",
    "wandb_token",
    "id_rsa",
    "id_ed25519",
}
FORBIDDEN_PATTERNS = {
    "review material": re.compile(
        r"official review|official comment|meta review|area chair|senior area chair|program chair|confidential comment",
        re.IGNORECASE,
    ),
    "discussion metadata": re.compile(r"openreview\.net/(?:forum|attachment)|submission\s*23665|reviewer\s+[A-Za-z0-9]{4}", re.IGNORECASE),
    "credential name": re.compile(r"OPENAI_API_KEY|WANDB_API_KEY|HF_TOKEN|HUGGING_FACE_HUB_TOKEN"),
    "credential value": re.compile(r"(?:sk-|hf_)[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]+PRIVATE KEY-----"),
    "private filesystem": re.compile(r"/(?:Users|home|capstor|ritom)/|\.edf(?:/|\b)", re.IGNORECASE),
    "cluster identity": re.compile(r"\b(?:smassoudi|smassoud|a0225|clariden|daint)\b", re.IGNORECASE),
    "internal workflow": re.compile(r"\b(?:rebuttal|meta-review|reviewer-requested|ac-complete)\b", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum_file(path: Path) -> dict[str, str]:
    expected = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, rel_path = line.split("  ", 1)
        expected[rel_path] = digest
    return expected


def validate(root: Path, index_path: Path) -> list[str]:
    errors: list[str] = []
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        return [f"missing checksum file: {checksum_path}"]

    expected = parse_checksum_file(checksum_path)
    actual_files = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if set(expected) != set(actual_files):
        errors.append("checksum inventory does not match package membership")
    for rel_path, path in actual_files.items():
        if expected.get(rel_path) != sha256(path):
            errors.append(f"checksum mismatch: {rel_path}")

    for path in root.rglob("*"):
        if path.name in FORBIDDEN_FILENAMES:
            errors.append(f"forbidden filename: {path.relative_to(root)}")
        if path.is_symlink():
            errors.append(f"symlink is not permitted: {path.relative_to(root)}")
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 text file: {path.relative_to(root)}")
            continue
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} in {path.relative_to(root)}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    claims = {claim["claim_id"]: claim for claim in index["claims"]}
    claim_ids = [claim["claim_id"] for claim in index["claims"]]
    if len(claim_ids) != len(set(claim_ids)):
        errors.append("duplicate claim_id in evidence index")
    for claim in index["claims"]:
        mapped = index_path.parents[1] / claim["public_result_path"]
        if not mapped.is_file():
            errors.append(f"missing mapped result: {claim['public_result_path']}")
        elif sha256(mapped) != claim["public_result_sha256"]:
            errors.append(f"mapped result checksum mismatch: {claim['claim_id']}")

    sds_by_seed = json.loads((root / "sds" / "certified_summary_by_seed.json").read_text())
    sds_summary = {
        row["Method"]: row
        for row in json.loads((root / "sds" / "certified_summary.json").read_text())
    }
    observed_seeds = {int(row["Seed"]) for row in sds_by_seed}
    if observed_seeds != {101, 202, 303}:
        errors.append(f"unexpected SDS seed set: {sorted(observed_seeds)}")
    for row in sds_by_seed:
        if row["CertifiedOptimalGapLowerMean"] < -1e-12:
            errors.append(f"negative certified gap: {row['Method']} seed {row['Seed']}")
        if row["CertifiedOptimalGapLowerMean"] > row["CertifiedOptimalGapUpperMean"] + 1e-12:
            errors.append(f"reversed certified interval: {row['Method']} seed {row['Seed']}")
        if row["FalseFeasibleOnProvedInfeasible"] != 0:
            errors.append(f"false feasible on proved-infeasible rows: {row['Method']} seed {row['Seed']}")

    method_claims = {
        "sds-hero-certified": "Ours (Hero)",
        "sds-base-best-of-64": "Base (Best-of-64)",
        "sds-same-model-adaptive-repair": "Adaptive Repair (64 completions)",
        "sds-hosted-adaptive-repair": "GPT-5.4-mini Medium-Reasoning Repair (64 completions)",
        "sds-input-disjoint-universal-search": "Certified Universal Search (191,699 pool)",
    }
    for claim_id, method in method_claims.items():
        row = sds_summary[method]
        reported = claims[claim_id]["reported_value"]
        expected_values = {
            "pass_rate_mean": row["PassRateFeasibleBenchmarkMean"],
            "pass_rate_seed_std": row["PassRateFeasibleBenchmarkStd"],
            "certified_optimal_gap_mean": row["CertifiedOptimalGapMean"],
            "certified_optimal_gap_seed_std": row["CertifiedOptimalGapStd"],
        }
        if reported != expected_values:
            errors.append(f"evidence-index metric drift: {claim_id}")

    same_model = json.loads(
        (root / "adaptive_repair" / "same_model_protocol_summary.json").read_text()
    )
    if same_model["clean_certified_result"] != sds_summary["Adaptive Repair (64 completions)"]:
        errors.append("same-model adaptive-repair protocol summary does not match final SDS aggregate")
    if same_model["workload"]["completions_total"] != 192:
        errors.append("same-model adaptive-repair completion count is not 192")

    hosted = json.loads((root / "adaptive_repair" / "hosted_protocol_summary.json").read_text())
    hosted_result = hosted["clean_certified_result"]
    hosted_row = sds_summary["GPT-5.4-mini Medium-Reasoning Repair (64 completions)"]
    if not math.isclose(
        hosted_result["seed_mean_pass_rate"],
        hosted_row["PassRateFeasibleBenchmarkMean"],
        abs_tol=1e-9,
    ):
        errors.append("hosted adaptive-repair pass rate does not match final SDS aggregate")
    if not math.isclose(
        hosted_result["seed_mean_certified_gap"],
        hosted_row["CertifiedOptimalGapMean"],
        abs_tol=1e-9,
    ):
        errors.append("hosted adaptive-repair gap does not match final SDS aggregate")

    certification = json.loads((root / "sds" / "certification_validation_summary.json").read_text())
    status_total = sum(
        sum(seed["status_counts"].values()) for seed in certification["seeds"]
    )
    if status_total != 3000:
        errors.append(f"certification status counts total {status_total}, expected 3000")
    if certification["certified_optimal_count"] != 2865:
        errors.append("certified-optimal SDS count is not 2865")
    if certification["independently_reproved_infeasible_count"] != 51:
        errors.append("proved-infeasible SDS count is not 51")

    tsp = json.loads((root / "tsp" / "summary.json").read_text())
    if tsp["primary_gate"]["passed"] is not False:
        errors.append("TSP failed primary gate is not preserved")

    hosted_hashes = hosted["freeze_and_test_integrity"]["selected_solvers"]
    for seed, expected_hash in hosted_hashes.items():
        solver = root / "solvers" / "hosted_repair" / f"seed{seed}.py"
        if sha256(solver) != expected_hash:
            errors.append(f"hosted solver checksum mismatch: seed {seed}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/neurips2026")
    parser.add_argument("--index", default="docs/final_evidence_index.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    index_path = Path(args.index).resolve()
    errors = validate(root, index_path)
    if errors:
        raise SystemExit("Evidence validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    print(f"Validated {root} with {len(parse_checksum_file(root / 'checksums.sha256'))} files.")


if __name__ == "__main__":
    main()
