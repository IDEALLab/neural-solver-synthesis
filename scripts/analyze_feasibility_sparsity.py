#!/usr/bin/env python3
"""
Aggregate feasibility sparsity logs from instrumented Clariden GRPO runs.

This script reconstructs full 64-sample GRPO groups from the rank-sharded
summary logs under `feasibility_sparsity/<job_id>/rank*.jsonl`. If the raw
generation traces are available under `feasibility_generation_traces/<job_id>/`,
it also runs integrity checks to confirm that the per-group summary matches the
underlying sample-level logs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


STAGE_NAMES = ("early", "middle", "late")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        type=Path,
        required=True,
        help="Root logs directory containing feasibility_sparsity/ and feasibility_generation_traces/.",
    )
    parser.add_argument(
        "--job-seed",
        action="append",
        required=True,
        metavar="JOB:SEED",
        help="Job/seed pair to aggregate. Repeat for multiple jobs, e.g. 1743423:101.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory for CSV/JSON artifacts.",
    )
    return parser.parse_args()


def parse_job_seed_mapping(values: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for value in values:
        job_id, seed = value.split(":", 1)
        mapping[job_id] = int(seed)
    return mapping


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_jsonl_rows(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("rank*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_job(logs_root: Path, job_id: str, seed: int) -> tuple[list[dict], dict, list[dict]]:
    sparse_dir = logs_root / "feasibility_sparsity" / job_id
    trace_dir = logs_root / "feasibility_generation_traces" / job_id

    sparse_groups: dict[int, list[dict]] = defaultdict(list)
    for row in load_jsonl_rows(sparse_dir):
        sparse_groups[row["reward_call_index"]].append(row)

    trace_groups: dict[int, list[dict]] = defaultdict(list)
    if trace_dir.exists():
        for row in load_jsonl_rows(trace_dir):
            trace_groups[row["reward_call_index"]].append(row)

    call_indices = sorted(sparse_groups)
    max_call = max(call_indices)
    n_calls = len(call_indices)
    stage_bounds = (n_calls // 3, (2 * n_calls) // 3)

    per_group_rows: list[dict] = []
    integrity_issues: list[dict] = []

    for call_idx in call_indices:
        sparse_rows = sparse_groups[call_idx]
        feasible_count = sum(row["feasible_count_in_group"] for row in sparse_rows)
        group_size = sum(row["local_group_size"] for row in sparse_rows)
        stage = (
            STAGE_NAMES[0]
            if call_idx < stage_bounds[0]
            else STAGE_NAMES[1]
            if call_idx < stage_bounds[1]
            else STAGE_NAMES[2]
        )

        mission_hashes = sorted({row["mission_hash"] for row in sparse_rows})
        prompt_hashes = sorted({row["prompt_hash"] for row in sparse_rows})
        row = {
            "job_id": job_id,
            "seed": seed,
            "reward_call_index": call_idx,
            "max_reward_call_index": max_call,
            "progress_fraction": (call_idx / max_call) if max_call else 0.0,
            "stage": stage,
            "group_size": group_size,
            "feasible_count_in_group": feasible_count,
            "has_any_feasible_in_group": int(feasible_count > 0),
            "feasible_fraction_in_group": feasible_count / group_size if group_size else 0.0,
            "mission_hash": mission_hashes[0] if len(mission_hashes) == 1 else "MULTI",
            "prompt_hash": prompt_hashes[0] if len(prompt_hashes) == 1 else "MULTI",
        }

        trace_rows = trace_groups.get(call_idx, [])
        if trace_rows:
            problem_uuids = sorted({trace_row.get("problem_uuid", "") for trace_row in trace_rows})
            trace_feasible = sum(int(bool(trace_row.get("exact_feasible"))) for trace_row in trace_rows)
            row["problem_uuid"] = problem_uuids[0] if len(problem_uuids) == 1 else "MULTI"
            row["trace_sample_count"] = len(trace_rows)
            row["trace_feasible_count"] = trace_feasible

            expected_ordinals = Counter({i: 8 for i in range(8)})
            actual_ordinals = Counter(trace_row.get("sample_ordinal_in_group") for trace_row in trace_rows)
            if len(trace_rows) != 64:
                integrity_issues.append({"job_id": job_id, "reward_call_index": call_idx, "issue": "trace_sample_count", "value": len(trace_rows)})
            if len(problem_uuids) != 1:
                integrity_issues.append({"job_id": job_id, "reward_call_index": call_idx, "issue": "problem_uuid_count", "value": len(problem_uuids)})
            if trace_feasible != feasible_count:
                integrity_issues.append(
                    {
                        "job_id": job_id,
                        "reward_call_index": call_idx,
                        "issue": "feasible_count_mismatch",
                        "value": {"trace": trace_feasible, "summary": feasible_count},
                    }
                )
            if actual_ordinals != expected_ordinals:
                integrity_issues.append(
                    {
                        "job_id": job_id,
                        "reward_call_index": call_idx,
                        "issue": "sample_ordinal_distribution",
                        "value": dict(actual_ordinals),
                    }
                )
        else:
            row["problem_uuid"] = ""
            row["trace_sample_count"] = 0
            row["trace_feasible_count"] = ""

        if group_size != 64:
            integrity_issues.append({"job_id": job_id, "reward_call_index": call_idx, "issue": "summary_group_size", "value": group_size})
        if len(sparse_rows) != 8:
            integrity_issues.append({"job_id": job_id, "reward_call_index": call_idx, "issue": "summary_rank_count", "value": len(sparse_rows)})
        if len(mission_hashes) != 1:
            integrity_issues.append({"job_id": job_id, "reward_call_index": call_idx, "issue": "mission_hash_count", "value": len(mission_hashes)})
        if len(prompt_hashes) != 1:
            integrity_issues.append({"job_id": job_id, "reward_call_index": call_idx, "issue": "prompt_hash_count", "value": len(prompt_hashes)})

        per_group_rows.append(row)

    per_seed_summary = {
        "seed": seed,
        "job_id": job_id,
        "n_groups": len(per_group_rows),
        "frac_groups_with_any_feasible": mean([row["has_any_feasible_in_group"] for row in per_group_rows]),
        "mean_feasible_count_in_group": mean([row["feasible_count_in_group"] for row in per_group_rows]),
        "feasible_completion_rate": mean([row["feasible_fraction_in_group"] for row in per_group_rows]),
    }

    for stage in STAGE_NAMES:
        stage_rows = [row for row in per_group_rows if row["stage"] == stage]
        per_seed_summary[stage] = {
            "n_groups": len(stage_rows),
            "frac_groups_with_any_feasible": mean([row["has_any_feasible_in_group"] for row in stage_rows]),
            "mean_feasible_count_in_group": mean([row["feasible_count_in_group"] for row in stage_rows]),
            "feasible_completion_rate": mean([row["feasible_fraction_in_group"] for row in stage_rows]),
        }

    return per_group_rows, per_seed_summary, integrity_issues


def aggregate(logs_root: Path, job_to_seed: dict[str, int]) -> dict:
    all_group_rows: list[dict] = []
    per_seed_rows: list[dict] = []
    integrity_issues: list[dict] = []

    for job_id, seed in job_to_seed.items():
        per_group, per_seed, issues = summarize_job(logs_root, job_id, seed)
        all_group_rows.extend(per_group)
        per_seed_rows.append(per_seed)
        integrity_issues.extend(issues)

    pooled = {
        "n_groups": len(all_group_rows),
        "frac_groups_with_any_feasible": mean([row["has_any_feasible_in_group"] for row in all_group_rows]),
        "mean_feasible_count_in_group": mean([row["feasible_count_in_group"] for row in all_group_rows]),
        "feasible_completion_rate": mean([row["feasible_fraction_in_group"] for row in all_group_rows]),
    }

    stage_pooled = {}
    for stage in STAGE_NAMES:
        rows = [row for row in all_group_rows if row["stage"] == stage]
        stage_pooled[stage] = {
            "n_groups": len(rows),
            "frac_groups_with_any_feasible": mean([row["has_any_feasible_in_group"] for row in rows]),
            "mean_feasible_count_in_group": mean([row["feasible_count_in_group"] for row in rows]),
            "feasible_completion_rate": mean([row["feasible_fraction_in_group"] for row in rows]),
        }

    progress_bins: list[dict] = []
    for bin_idx in range(10):
        lo = bin_idx / 10
        hi = (bin_idx + 1) / 10
        rows = [
            row
            for row in all_group_rows
            if row["progress_fraction"] >= lo and (row["progress_fraction"] < hi if bin_idx < 9 else row["progress_fraction"] <= hi)
        ]
        progress_bins.append(
            {
                "bin_start": lo,
                "bin_end": hi,
                "n_groups": len(rows),
                "frac_groups_with_any_feasible": mean([row["has_any_feasible_in_group"] for row in rows]),
                "mean_feasible_count_in_group": mean([row["feasible_count_in_group"] for row in rows]),
                "feasible_completion_rate": mean([row["feasible_fraction_in_group"] for row in rows]),
            }
        )

    return {
        "metadata": {
            "job_to_seed": job_to_seed,
            "logs_root": str(logs_root),
            "stage_definition": "Per-seed terciles over reward_call_index.",
            "progress_bins_definition": "Pooled normalized progress bins over reward_call_index/max_reward_call_index.",
        },
        "per_seed": sorted(per_seed_rows, key=lambda row: row["seed"]),
        "pooled": pooled,
        "stage_pooled": stage_pooled,
        "progress_bins": progress_bins,
        "integrity_issues": integrity_issues,
        "integrity_issue_count": len(integrity_issues),
        "per_group_rows": all_group_rows,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = summary.copy()
    per_group_rows = summary_json.pop("per_group_rows")
    (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    per_seed_rows = []
    for row in summary["per_seed"]:
        per_seed_rows.append(
            {
                "seed": row["seed"],
                "job_id": row["job_id"],
                "n_groups": row["n_groups"],
                "frac_groups_with_any_feasible": row["frac_groups_with_any_feasible"],
                "mean_feasible_count_in_group": row["mean_feasible_count_in_group"],
                "feasible_completion_rate": row["feasible_completion_rate"],
                "early_frac_groups_with_any_feasible": row["early"]["frac_groups_with_any_feasible"],
                "early_mean_feasible_count_in_group": row["early"]["mean_feasible_count_in_group"],
                "early_feasible_completion_rate": row["early"]["feasible_completion_rate"],
                "middle_frac_groups_with_any_feasible": row["middle"]["frac_groups_with_any_feasible"],
                "middle_mean_feasible_count_in_group": row["middle"]["mean_feasible_count_in_group"],
                "middle_feasible_completion_rate": row["middle"]["feasible_completion_rate"],
                "late_frac_groups_with_any_feasible": row["late"]["frac_groups_with_any_feasible"],
                "late_mean_feasible_count_in_group": row["late"]["mean_feasible_count_in_group"],
                "late_feasible_completion_rate": row["late"]["feasible_completion_rate"],
            }
        )
    write_csv(
        output_dir / "per_seed_summary.csv",
        per_seed_rows,
        [
            "seed",
            "job_id",
            "n_groups",
            "frac_groups_with_any_feasible",
            "mean_feasible_count_in_group",
            "feasible_completion_rate",
            "early_frac_groups_with_any_feasible",
            "early_mean_feasible_count_in_group",
            "early_feasible_completion_rate",
            "middle_frac_groups_with_any_feasible",
            "middle_mean_feasible_count_in_group",
            "middle_feasible_completion_rate",
            "late_frac_groups_with_any_feasible",
            "late_mean_feasible_count_in_group",
            "late_feasible_completion_rate",
        ],
    )

    stage_rows = []
    for stage, row in summary["stage_pooled"].items():
        stage_rows.append({"stage": stage, **row})
    write_csv(
        output_dir / "stage_pooled_summary.csv",
        stage_rows,
        [
            "stage",
            "n_groups",
            "frac_groups_with_any_feasible",
            "mean_feasible_count_in_group",
            "feasible_completion_rate",
        ],
    )

    write_csv(
        output_dir / "progress_bins.csv",
        summary["progress_bins"],
        [
            "bin_start",
            "bin_end",
            "n_groups",
            "frac_groups_with_any_feasible",
            "mean_feasible_count_in_group",
            "feasible_completion_rate",
        ],
    )

    write_csv(
        output_dir / "per_group_summary.csv",
        per_group_rows,
        [
            "job_id",
            "seed",
            "reward_call_index",
            "max_reward_call_index",
            "progress_fraction",
            "stage",
            "group_size",
            "feasible_count_in_group",
            "has_any_feasible_in_group",
            "feasible_fraction_in_group",
            "mission_hash",
            "prompt_hash",
            "problem_uuid",
            "trace_sample_count",
            "trace_feasible_count",
        ],
    )


def main() -> None:
    args = parse_args()
    job_to_seed = parse_job_seed_mapping(args.job_seed)
    summary = aggregate(args.logs_root, job_to_seed)

    if args.output_dir is not None:
        write_outputs(args.output_dir, summary)

    printable = summary.copy()
    printable.pop("per_group_rows")
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
