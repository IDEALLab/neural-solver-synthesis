"""Aggregate three-seed generated-program audits by condition."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, stdev

FEATURE_FIELDS = (
    "has_feasibility_guard",
    "has_best_tracking",
    "has_two_way_neighbor_logic",
)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid Boolean value: {value!r}")


def parse_optional_float(value: object) -> float | None:
    text = str(value).strip()
    return float(text) if text else None


def load_certified_quality(
    path: Path, condition_methods: dict[str, str]
) -> dict[tuple[str, int, str], dict]:
    method_conditions = {
        method: condition for condition, method in condition_methods.items()
    }
    if len(method_conditions) != len(condition_methods):
        raise ValueError("each audit condition must map to a distinct method")

    quality = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Method",
            "Seed",
            "uuid",
            "CertificateStatus",
            "BenchmarkFeasible",
            "Pass",
            "CertifiedGapLower",
            "CertifiedGapUpper",
            "CertifiedOptimumAttained",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"certified metrics missing columns: {sorted(missing)}")
        for row in reader:
            condition = method_conditions.get(row["Method"])
            if condition is None:
                continue
            key = (condition, int(row["Seed"]), row["uuid"])
            if key in quality:
                raise ValueError(f"duplicate certified quality key: {key}")
            quality[key] = {
                "benchmark_feasible": parse_bool(row["BenchmarkFeasible"]),
                "passed": parse_bool(row["Pass"]),
                "certificate_status": row["CertificateStatus"],
                "gap_lower": parse_optional_float(row["CertifiedGapLower"]),
                "gap_upper": parse_optional_float(row["CertifiedGapUpper"]),
                "optimum_attained": (
                    parse_bool(row["CertifiedOptimumAttained"])
                    if row["CertifiedOptimumAttained"].strip()
                    else None
                ),
            }
    return quality


def quality_summary(rows: list[dict], quality: dict) -> dict:
    keyed = [quality[(row["condition"], int(row["seed"]), row["uuid"])] for row in rows]
    feasible = [row for row in keyed if row["benchmark_feasible"]]
    exact = [row for row in keyed if row["certificate_status"] == "OPTIMAL"]
    exact_gaps = [row["gap_lower"] for row in exact if row["gap_lower"] is not None]
    lower = [row["gap_lower"] for row in feasible if row["gap_lower"] is not None]
    upper = [row["gap_upper"] for row in feasible if row["gap_upper"] is not None]
    attained = [
        row["optimum_attained"] for row in exact if row["optimum_attained"] is not None
    ]
    return {
        "row_count": len(rows),
        "benchmark_feasible_rows": len(feasible),
        "pass_rate_feasible_benchmark": (
            mean(float(row["passed"]) for row in feasible) if feasible else None
        ),
        "certified_optimal_rows": len(exact),
        "certified_optimal_gap_mean": mean(exact_gaps) if exact_gaps else None,
        "certified_optimum_attainment_rate": mean(float(value) for value in attained)
        if attained
        else None,
        "all_feasible_gap_lower_mean": mean(lower) if lower else None,
        "all_feasible_gap_upper_mean": mean(upper) if upper else None,
    }


def summarize_uncertainty(per_seed: list[dict]) -> dict:
    metrics = (
        "pass_rate_feasible_benchmark",
        "certified_optimal_gap_mean",
        "certified_optimum_attainment_rate",
        "all_feasible_gap_lower_mean",
        "all_feasible_gap_upper_mean",
    )
    result = {}
    for metric in metrics:
        values = [float(row[metric]) for row in per_seed if row[metric] is not None]
        result[metric] = {
            "mean": mean(values) if values else None,
            "sample_stddev": stdev(values)
            if len(values) > 1
            else 0.0
            if values
            else None,
            "seed_count": len(values),
        }
    return result


def load_benchmark_keys(path: Path) -> set[tuple[int, str]]:
    """Load an immutable benchmark subset keyed strictly by seed and UUID."""
    manifest = json.loads(path.read_text())
    if manifest.get("join_keys") != ["seed", "uuid"]:
        raise ValueError("benchmark subset must declare (seed, uuid) joins")
    keys: set[tuple[int, str]] = set()
    for seed_text, seed_manifest in manifest.get("seeds", {}).items():
        seed = int(seed_text)
        retained = seed_manifest.get("retained", [])
        if len(retained) != int(seed_manifest.get("retained_rows", -1)):
            raise ValueError(f"benchmark subset retained count drift for seed {seed}")
        for row in retained:
            key = (seed, str(row["uuid"]))
            if key in keys:
                raise ValueError(f"duplicate benchmark subset key: {key}")
            keys.add(key)
    if not keys:
        raise ValueError("benchmark subset is empty")
    return keys


def aggregate_audits(  # noqa: PLR0912, PLR0915
    inputs: Iterable[Path],
    *,
    certified_metrics: Path | None = None,
    condition_methods: dict[str, str] | None = None,
    benchmark_keys: set[tuple[int, str]] | None = None,
) -> dict:
    rows = []
    seen = set()
    for path in inputs:
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            if not row.get("uuid"):
                raise ValueError(f"missing audit UUID in {path}")
            row_key = (int(row["seed"]), str(row["uuid"]))
            if benchmark_keys is not None and row_key not in benchmark_keys:
                continue
            key = (row["condition"], int(row["seed"]), row["uuid"])
            if key in seen:
                raise ValueError(f"duplicate audit key: {key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("no audit rows remain after benchmark filtering")

    condition_seeds = defaultdict(set)
    family_counts = defaultdict(Counter)
    acceptance_counts = defaultdict(Counter)
    feature_counts = defaultdict(Counter)
    code_hashes = defaultdict(set)
    for row in rows:
        condition = row["condition"]
        condition_seeds[condition].add(int(row["seed"]))
        family_counts[condition][row["algorithm_family"]] += 1
        acceptance_counts[condition][row["sa_acceptance"]] += 1
        for feature in FEATURE_FIELDS:
            feature_counts[condition][feature] += int(bool(row.get(feature, False)))
        if row.get("code_sha256"):
            code_hashes[condition].add(row["code_sha256"])

    expected_seeds = {101, 202, 303}
    for condition, seeds in condition_seeds.items():
        condition_keys = {
            (int(row["seed"]), str(row["uuid"]))
            for row in rows
            if row["condition"] == condition
        }
        if benchmark_keys is None:
            if len(condition_keys) != 3000:
                raise ValueError(f"condition {condition} has {len(condition_keys)} rows")
        elif condition_keys != benchmark_keys:
            missing = sorted(benchmark_keys - condition_keys)[:5]
            extra = sorted(condition_keys - benchmark_keys)[:5]
            raise ValueError(
                f"benchmark subset key mismatch for {condition}: "
                f"missing={missing}, extra={extra}"
            )
        if seeds != expected_seeds:
            raise ValueError(f"condition {condition} has seeds {sorted(seeds)}")

    quality = None
    if certified_metrics is not None:
        if not condition_methods:
            raise ValueError(
                "condition-method mappings are required with certified metrics"
            )
        if set(condition_seeds) != set(condition_methods):
            raise ValueError(
                "condition-method mappings must cover exactly the audit conditions"
            )
        quality = load_certified_quality(certified_metrics, condition_methods)
        audit_keys = {(row["condition"], int(row["seed"]), row["uuid"]) for row in rows}
        quality_keys = set(quality)
        if audit_keys != quality_keys:
            missing = sorted(audit_keys - quality_keys)[:5]
            extra = sorted(quality_keys - audit_keys)[:5]
            raise ValueError(
                f"audit/quality key mismatch; missing={missing}, extra={extra}"
            )

    conditions = {}
    for condition in sorted(condition_seeds):
        condition_rows = [row for row in rows if row["condition"] == condition]
        total = len(condition_rows)
        valid_code_rows = sum(bool(row.get("code_sha256")) for row in condition_rows)
        condition_result = {
            "row_count": total,
            "valid_code_row_count": valid_code_rows,
            "unique_code_count": len(code_hashes[condition]),
            "repeated_code_row_count": valid_code_rows - len(code_hashes[condition]),
            "algorithm_family_counts": dict(sorted(family_counts[condition].items())),
            "algorithm_family_rates": {
                label: count / total
                for label, count in sorted(family_counts[condition].items())
            },
            "sa_acceptance_counts": dict(sorted(acceptance_counts[condition].items())),
            "sa_acceptance_rates": {
                label: count / total
                for label, count in sorted(acceptance_counts[condition].items())
            },
            "feature_counts": dict(sorted(feature_counts[condition].items())),
            "feature_rates": {
                label: count / total
                for label, count in sorted(feature_counts[condition].items())
            },
        }
        family_by_seed = []
        all_families = sorted(family_counts[condition])
        for seed in sorted(condition_seeds[condition]):
            seed_rows = [row for row in condition_rows if int(row["seed"]) == seed]
            seed_counts = Counter(row["algorithm_family"] for row in seed_rows)
            family_by_seed.append(
                {
                    "seed": seed,
                    "row_count": len(seed_rows),
                    "unique_code_count": len(
                        {
                            row.get("code_sha256")
                            for row in seed_rows
                            if row.get("code_sha256")
                        }
                    ),
                    "algorithm_family_counts": {
                        family: seed_counts.get(family, 0) for family in all_families
                    },
                    "algorithm_family_rates": {
                        family: seed_counts.get(family, 0) / len(seed_rows)
                        for family in all_families
                    },
                }
            )
        condition_result["algorithm_families_by_seed"] = family_by_seed
        condition_result["algorithm_family_rate_across_seeds"] = {
            family: {
                "mean": mean(
                    row["algorithm_family_rates"][family] for row in family_by_seed
                ),
                "sample_stddev": stdev(
                    row["algorithm_family_rates"][family] for row in family_by_seed
                ),
                "seed_count": len(family_by_seed),
            }
            for family in all_families
        }
        if quality is not None:
            per_seed = []
            for seed in sorted(condition_seeds[condition]):
                seed_rows = [row for row in condition_rows if int(row["seed"]) == seed]
                seed_summary = quality_summary(seed_rows, quality)
                seed_summary["seed"] = seed
                seed_summary["unique_code_count"] = len(
                    {
                        row.get("code_sha256")
                        for row in seed_rows
                        if row.get("code_sha256")
                    }
                )
                per_seed.append(seed_summary)
            condition_result["quality_by_seed"] = per_seed
            condition_result["quality_across_seeds"] = summarize_uncertainty(per_seed)
            condition_result["quality_by_family"] = {
                family: quality_summary(
                    [
                        row
                        for row in condition_rows
                        if row["algorithm_family"] == family
                    ],
                    quality,
                )
                for family in sorted(family_counts[condition])
            }
        conditions[condition] = condition_result

    return {
        "conditions": conditions,
        "join_keys": ["condition", "seed", "uuid"],
        "certified_metrics": str(certified_metrics) if certified_metrics else None,
        "benchmark_subset_rows": len(benchmark_keys) if benchmark_keys else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--certified-metrics", type=Path)
    parser.add_argument("--benchmark-subset", type=Path)
    parser.add_argument(
        "--condition-method",
        action="append",
        default=[],
        metavar="CONDITION=METHOD",
    )
    args = parser.parse_args()
    condition_methods = {}
    for mapping in args.condition_method:
        if "=" not in mapping:
            parser.error(f"invalid --condition-method mapping: {mapping}")
        condition, method = mapping.split("=", 1)
        if not condition or not method or condition in condition_methods:
            parser.error(f"invalid --condition-method mapping: {mapping}")
        condition_methods[condition] = method
    result = aggregate_audits(
        args.input,
        certified_metrics=args.certified_metrics,
        condition_methods=condition_methods,
        benchmark_keys=(
            load_benchmark_keys(args.benchmark_subset)
            if args.benchmark_subset
            else None
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "program_audit_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    with (args.output_dir / "program_audit_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "measure", "label", "count", "rate"])
        for condition, condition_data in result["conditions"].items():
            total = condition_data["row_count"]
            for measure in (
                "algorithm_family_counts",
                "sa_acceptance_counts",
                "feature_counts",
            ):
                for label, count in condition_data[measure].items():
                    writer.writerow([condition, measure, label, count, count / total])

    with (args.output_dir / "program_families_by_seed.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["condition", "seed", "algorithm_family", "count", "rate", "unique_code_count"]
        )
        for condition, condition_data in result["conditions"].items():
            for seed_row in condition_data["algorithm_families_by_seed"]:
                for family, count in seed_row["algorithm_family_counts"].items():
                    writer.writerow(
                        [
                            condition,
                            seed_row["seed"],
                            family,
                            count,
                            seed_row["algorithm_family_rates"][family],
                            seed_row["unique_code_count"],
                        ]
                    )

    if args.certified_metrics:
        seed_fields = [
            "condition",
            "seed",
            "row_count",
            "unique_code_count",
            "benchmark_feasible_rows",
            "pass_rate_feasible_benchmark",
            "certified_optimal_rows",
            "certified_optimal_gap_mean",
            "certified_optimum_attainment_rate",
            "all_feasible_gap_lower_mean",
            "all_feasible_gap_upper_mean",
        ]
        with (args.output_dir / "program_quality_by_seed.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=seed_fields)
            writer.writeheader()
            for condition, condition_data in result["conditions"].items():
                for row in condition_data["quality_by_seed"]:
                    writer.writerow({"condition": condition, **row})

        family_fields = [
            "condition",
            "algorithm_family",
            "row_count",
            "benchmark_feasible_rows",
            "pass_rate_feasible_benchmark",
            "certified_optimal_rows",
            "certified_optimal_gap_mean",
            "certified_optimum_attainment_rate",
            "all_feasible_gap_lower_mean",
            "all_feasible_gap_upper_mean",
        ]
        with (args.output_dir / "program_quality_by_family.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=family_fields)
            writer.writeheader()
            for condition, condition_data in result["conditions"].items():
                for family, row in condition_data["quality_by_family"].items():
                    writer.writerow(
                        {
                            "condition": condition,
                            "algorithm_family": family,
                            **row,
                        }
                    )


if __name__ == "__main__":
    main()
