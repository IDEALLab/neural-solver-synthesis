#!/usr/bin/env python3
"""Extract one deterministic frozen Hero solver from canonical evaluation artifacts."""

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_REPORT_SET = "experiments/report_sets/paper_main_results_v1.json"
DEFAULT_RESULTS_ROOT = "evaluation/sds/results_batches/20251230_struct-feas-v1"
SEEDS = (101, 202, 303)
SELECTION_RULE = (
    "filter feasible == True and error_type == 'none', sort by uuid ascending, take first"
)


def canonical_metrics_csv_for_seed(seed: int, report_set_path: str) -> Path:
    """Resolve the canonical Hero metrics CSV from the main report set."""
    report_set = json.loads(Path(report_set_path).read_text())
    hero_seed = report_set["checkpoints"]["models"]["Hero"]["seeds"][str(seed)]
    job_id = hero_seed["job_id"]
    root = Path(DEFAULT_RESULTS_ROOT)
    return root / "qwen2.5-coder-14b" / "grpo" / f"seed{seed}" / f"job-{job_id}" / "metrics_final.csv"


def extract_solver(metrics_csv: Path) -> tuple[str, dict]:
    """Return the selected code snippet and provenance metadata."""
    df = pd.read_csv(metrics_csv)
    valid = df[(df["feasible"] == True) & (df["error_type"] == "none")].copy()  # noqa: E712
    if valid.empty:
        raise RuntimeError(f"No valid solver rows found in {metrics_csv}")

    selected = valid.sort_values("uuid").iloc[0]
    code = selected.get("code_snippet", "")
    if not isinstance(code, str) or not code.strip():
        raise RuntimeError(f"Selected row in {metrics_csv} has empty code_snippet")

    metadata = {
        "source_metrics_csv": str(metrics_csv),
        "selected_uuid": selected["uuid"],
        "selection_rule": SELECTION_RULE,
        "job_id": metrics_csv.parent.name.replace("job-", ""),
        "code_source_type": "frozen-hero",
        "code_length": len(code),
    }
    return code, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract deterministic frozen Hero solvers from canonical metrics."
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        help="Seed to extract from the canonical main report set.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Explicit metrics_final.csv path. Overrides --seed lookup if provided.",
    )
    parser.add_argument(
        "--report-set",
        type=str,
        default=DEFAULT_REPORT_SET,
        help="Main report-set JSON used to locate canonical Hero jobs.",
    )
    parser.add_argument(
        "--output-py",
        type=str,
        default=None,
        help="Path to write the extracted solver.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write provenance metadata JSON.",
    )
    args = parser.parse_args()

    if not args.metrics_csv and args.seed is None:
        parser.error("Provide either --metrics-csv or --seed")

    metrics_csv = (
        Path(args.metrics_csv)
        if args.metrics_csv
        else canonical_metrics_csv_for_seed(args.seed, args.report_set)
    )
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    seed = args.seed
    if seed is None:
        seed_part = next(part for part in metrics_csv.parts if part.startswith("seed"))
        seed = int(seed_part.replace("seed", ""))

    output_py = Path(args.output_py or f"evaluation/sds/frozen_solvers/hero_seed{seed}.py")
    output_json = Path(
        args.output_json or f"evaluation/sds/frozen_solvers/hero_seed{seed}.json"
    )
    output_py.parent.mkdir(parents=True, exist_ok=True)

    code, metadata = extract_solver(metrics_csv)
    metadata["seed"] = seed
    metadata["output_py"] = str(output_py)

    output_py.write_text(code if code.endswith("\n") else f"{code}\n")
    output_json.write_text(json.dumps(metadata, indent=2))

    print(f"✅ Extracted frozen solver for seed {seed}")
    print(f"   Source: {metrics_csv}")
    print(f"   Solver: {output_py}")
    print(f"   Metadata: {output_json}")


if __name__ == "__main__":
    main()
