#!/usr/bin/env python3
"""Aggregate fixed-code publication evidence against canonical SDS baselines."""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from evaluation.sds.aggregate_plots import (
        compute_virtual_best_scores,
        find_all_metrics_files,
        find_all_metrics_files_from_roots,
        load_all_data,
        load_report_set,
        select_latest_jobs,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from evaluation.sds.aggregate_plots import (
        compute_virtual_best_scores,
        find_all_metrics_files,
        find_all_metrics_files_from_roots,
        load_all_data,
        load_report_set,
        select_latest_jobs,
    )


MAIN_METHODS = [
    "Ours (Hero)",
    "Contemporary Hero control",
    "Frozen Hero",
    "Hand-written SA",
    "Base (Best-of-64)",
    "Adaptive Repair (64 completions)",
    "ShinkaEvolve",
    "CP-SAT",
    "Local Search",
    "Greedy",
    "BnB",
]
RUNTIME_METHODS = [
    "Ours (Hero)",
    "Frozen Hero",
    "Hand-written SA",
    "Base (Best-of-64)",
    "ShinkaEvolve",
]
HERO_METHODS = {"Ours (Hero)", "Frozen Hero"}
CERTIFIED_STATUSES = {"OPTIMAL", "FEASIBLE", "INFEASIBLE"}
CERTIFICATION_TOLERANCE = 1e-5


def infer_runtime_method_name(data: dict) -> str | None:
    """Normalize timing-summary method names onto the publication labels."""
    method_name = data.get("method_name")
    training_scheme = data.get("training_scheme")
    run_type = data.get("run_type")
    n_samples = data.get("n_samples")

    if method_name == "LLM (Ours)":
        if training_scheme == "grpo":
            return "Ours (Hero)"
        if training_scheme == "base" and run_type == "best-of-n" and n_samples == 64:
            return "Base (Best-of-64)"
    return method_name


def summarise_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-method pass/gap/cost summary across seeds."""
    per_seed = summarise_methods_by_seed(df)
    summary = (
        per_seed.groupby("Method", observed=True)
        .agg(
            PassMean=("Pass", "mean"),
            PassStd=("Pass", "std"),
            GapMean=("Gap", "mean"),
            GapStd=("Gap", "std"),
            CostMean=("Cost", "mean"),
            CostStd=("Cost", "std"),
        )
        .reset_index()
    )
    summary = summary[summary["Method"].isin(MAIN_METHODS)].copy()
    summary["Method"] = pd.Categorical(summary["Method"], MAIN_METHODS, ordered=True)
    summary = summary.sort_values("Method").reset_index(drop=True)
    numeric_cols = summary.select_dtypes(include=["number"]).columns
    summary[numeric_cols] = summary[numeric_cols].fillna(0.0)
    return summary


def recompute_global_vbs(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute shared VBS/gap after combining separately loaded method groups."""
    final_df = df.copy()
    global_vbs_data = []
    for (uuid, seed), group in final_df.groupby(["uuid", "Seed"], observed=True):
        feasible_scores = group[group["feasible"]]["llm_score"].dropna()
        if len(feasible_scores) > 0:
            global_vbs = feasible_scores.max()
        else:
            vbs_scores = group["vbs_score"].dropna()
            global_vbs = vbs_scores.max() if len(vbs_scores) > 0 else float("-inf")
        global_vbs_data.append({"uuid": uuid, "Seed": seed, "global_vbs": global_vbs})

    global_vbs_df = pd.DataFrame(global_vbs_data)
    final_df = final_df.merge(global_vbs_df, on=["uuid", "Seed"], how="left")

    valid_vbs_mask = (final_df["global_vbs"] > 1e-6) & final_df["global_vbs"].notna()
    method_scores = final_df.loc[valid_vbs_mask, "llm_score"].fillna(0.0).clip(lower=0.0)
    global_vbs_vals = final_df.loc[valid_vbs_mask, "global_vbs"]
    final_df.loc[valid_vbs_mask, "Gap"] = (
        (global_vbs_vals - method_scores) / global_vbs_vals
    ).to_numpy()
    final_df["vbs_score"] = final_df["global_vbs"].fillna(final_df["vbs_score"])
    final_df = final_df.drop(columns=["global_vbs"], errors="ignore")

    valid_gap_mask = (
        (final_df["Gap"] >= 0) & (final_df["Gap"] <= 1.0) & (~final_df["Gap"].isna())
    )
    final_df.loc[~valid_gap_mask, "Gap"] = pd.NA
    return final_df


def load_certified_references(
    certification_root: Path, validation_summary_path: Path
) -> tuple[pd.DataFrame, dict]:
    """Load an independently validated certification batch."""
    validation = json.loads(validation_summary_path.read_text())
    if validation.get("status") != "validated":
        raise RuntimeError("certification validation summary is not validated")
    if validation.get("source_batch") != certification_root.name:
        raise RuntimeError("certification root does not match validation source_batch")

    rows = []
    for seed_dir in sorted(certification_root.glob("seed*")):
        if not seed_dir.is_dir() or not seed_dir.name.removeprefix("seed").isdigit():
            continue
        seed = int(seed_dir.name.removeprefix("seed"))
        source = seed_dir / "certifications.jsonl"
        if not source.is_file():
            raise RuntimeError(f"missing certification rows: {source}")
        for line in source.read_text().splitlines():
            if line:
                rows.append({"Seed": seed, **json.loads(line)})
    certifications = pd.DataFrame(rows)
    required = {
        "Seed",
        "uuid",
        "status",
        "objective",
        "best_bound",
        "certified_optimal",
        "proved_infeasible",
    }
    missing = required - set(certifications.columns)
    if missing:
        raise RuntimeError(f"certifications are missing columns: {sorted(missing)}")
    if len(certifications) != int(validation["row_count"]):
        raise RuntimeError("certification row count disagrees with validation summary")
    if certifications.duplicated(["Seed", "uuid"]).any():
        raise RuntimeError("certifications contain duplicate (Seed, uuid) keys")
    statuses = set(certifications["status"])
    if not statuses <= CERTIFIED_STATUSES:
        raise RuntimeError(f"unsupported certification statuses: {sorted(statuses)}")
    infeasible = certifications["status"] == "INFEASIBLE"
    if not certifications.loc[infeasible, "proved_infeasible"].fillna(False).all():
        raise RuntimeError("an INFEASIBLE row lacks a proof marker")
    return certifications, validation


def attach_certified_references(
    results: pd.DataFrame, certifications: pd.DataFrame
) -> pd.DataFrame:
    """Attach certified reference intervals with strict per-seed UUID joins."""
    if results.duplicated(["Method", "Seed", "uuid"]).any():
        raise RuntimeError("result rows are not unique on (Method, Seed, uuid)")
    result_keys = set(results[["Seed", "uuid"]].itertuples(index=False, name=None))
    certification_keys = set(
        certifications[["Seed", "uuid"]].itertuples(index=False, name=None)
    )
    if result_keys != certification_keys:
        missing = sorted(result_keys - certification_keys)[:5]
        extra = sorted(certification_keys - result_keys)[:5]
        raise RuntimeError(
            f"certification/result key mismatch; missing={missing}, extra={extra}"
        )

    reference_columns = certifications[
        [
            "Seed",
            "uuid",
            "status",
            "objective",
            "best_bound",
            "certified_optimal",
            "proved_infeasible",
        ]
    ].rename(
        columns={
            "status": "CertificateStatus",
            "objective": "ReferenceLower",
            "best_bound": "ReferenceUpper",
            "certified_optimal": "CertifiedOptimal",
            "proved_infeasible": "ProvedInfeasible",
        }
    )
    merged = results.merge(
        reference_columns,
        on=["Seed", "uuid"],
        how="left",
        validate="many_to_one",
    )
    gap_lower: list[float] = []
    gap_upper: list[float] = []
    attained: list[bool | None] = []
    benchmark_feasible: list[bool] = []
    for row in merged.itertuples(index=False):
        status = str(row.CertificateStatus)
        method_feasible = bool(row.feasible)
        score = float(row.llm_score) if method_feasible else 0.0
        if status == "INFEASIBLE":
            if method_feasible:
                raise RuntimeError(
                    f"{row.Method} is feasible on proved-INFEASIBLE key "
                    f"({row.Seed}, {row.uuid})"
                )
            gap_lower.append(math.nan)
            gap_upper.append(math.nan)
            attained.append(None)
            benchmark_feasible.append(False)
            continue

        lower = float(row.ReferenceLower)
        upper = float(row.ReferenceUpper)
        tolerance = CERTIFICATION_TOLERANCE * max(1.0, abs(upper))
        if method_feasible and score > upper + tolerance:
            raise RuntimeError(
                f"{row.Method} score {score} exceeds certified upper bound {upper} "
                f"for ({row.Seed}, {row.uuid})"
            )
        if not method_feasible:
            lower_gap = upper_gap = 1.0
        else:
            effective_lower = max(lower, score)
            lower_gap = (
                max(0.0, (effective_lower - max(0.0, score)) / effective_lower)
                if effective_lower > CERTIFICATION_TOLERANCE
                else 0.0
            )
            upper_gap = (
                max(0.0, (upper - max(0.0, score)) / upper)
                if upper > CERTIFICATION_TOLERANCE
                else 0.0
            )
        gap_lower.append(min(1.0, lower_gap))
        gap_upper.append(min(1.0, upper_gap))
        attained.append(
            status == "OPTIMAL" and method_feasible and abs(score - lower) <= tolerance
        )
        benchmark_feasible.append(True)

    merged["BenchmarkFeasible"] = benchmark_feasible
    merged["CertifiedGapLower"] = gap_lower
    merged["CertifiedGapUpper"] = gap_upper
    merged["CertifiedOptimumAttained"] = attained
    feasible_reference = merged["CertificateStatus"] != "INFEASIBLE"
    merged["ReferenceIntervalWidth"] = math.nan
    merged["ReferenceRelativeIntervalWidth"] = math.nan
    merged.loc[feasible_reference, "ReferenceIntervalWidth"] = (
        merged.loc[feasible_reference, "ReferenceUpper"]
        - merged.loc[feasible_reference, "ReferenceLower"]
    ).clip(lower=0.0)
    merged.loc[feasible_reference, "ReferenceRelativeIntervalWidth"] = (
        merged.loc[feasible_reference, "ReferenceIntervalWidth"]
        / merged.loc[feasible_reference, "ReferenceUpper"]
        .abs()
        .clip(lower=CERTIFICATION_TOLERANCE)
    )
    return merged


def load_benchmark_subset(path: Path) -> tuple[set[tuple[int, str]], dict]:
    """Load an immutable benchmark subset keyed strictly by seed and UUID."""
    manifest = json.loads(path.read_text())
    if manifest.get("join_keys") != ["seed", "uuid"]:
        raise RuntimeError("benchmark subset must declare (seed, uuid) joins")
    keys: set[tuple[int, str]] = set()
    for seed_text, seed_manifest in manifest.get("seeds", {}).items():
        seed = int(seed_text)
        rows = seed_manifest.get("retained", [])
        if len(rows) != int(seed_manifest.get("retained_rows", -1)):
            raise RuntimeError(f"benchmark subset retained count drift for seed {seed}")
        for row in rows:
            key = (seed, str(row["uuid"]))
            if key in keys:
                raise RuntimeError(f"duplicate benchmark subset key: {key}")
            keys.add(key)
    if not keys:
        raise RuntimeError("benchmark subset is empty")
    return keys, manifest


def restrict_to_benchmark_subset(
    results: pd.DataFrame, keys: set[tuple[int, str]]
) -> pd.DataFrame:
    """Require and retain the same immutable key set for every method."""
    methods = results["Method"].drop_duplicates().tolist()
    retained = results[
        [
            (int(seed), str(uuid)) in keys
            for seed, uuid in results[["Seed", "uuid"]].itertuples(index=False, name=None)
        ]
    ].copy()
    for method in methods:
        group = retained[retained["Method"] == method]
        duplicate_mask = group.duplicated(["Seed", "uuid"], keep=False)
        if duplicate_mask.any():
            duplicate = group.loc[duplicate_mask, ["Seed", "uuid"]].iloc[0]
            raise RuntimeError(
                "duplicate benchmark result key for "
                f"{method}: {(int(duplicate['Seed']), str(duplicate['uuid']))}"
            )
        method_keys = set(
            (int(seed), str(uuid))
            for seed, uuid in group[["Seed", "uuid"]].itertuples(index=False, name=None)
        )
        if method_keys != keys:
            missing = sorted(keys - method_keys)[:5]
            extra = sorted(method_keys - keys)[:5]
            raise RuntimeError(
                f"benchmark subset key mismatch for {method}: missing={missing}, extra={extra}"
            )
    return retained


def summarise_certified_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize CP-SAT-optimal, unresolved, and infeasible rows by seed."""
    rows = []
    for (method, seed), group in results.groupby(["Method", "Seed"], observed=True):
        feasible_benchmark = group[group["BenchmarkFeasible"]]
        exact = group[group["CertificateStatus"] == "OPTIMAL"]
        unresolved = group[group["CertificateStatus"] == "FEASIBLE"]
        proved_infeasible = group[group["CertificateStatus"] == "INFEASIBLE"]
        rows.append(
            {
                "Method": method,
                "Seed": seed,
                "Rows": len(group),
                "FeasibleBenchmarkRows": len(feasible_benchmark),
                "ProvedInfeasibleRows": len(proved_infeasible),
                "PassRateFeasibleBenchmark": feasible_benchmark["Pass"].mean(),
                "PassRateAllRows": group["Pass"].mean(),
                "CertifiedOptimalRows": len(exact),
                "CertifiedOptimalGapMean": exact["CertifiedGapLower"].mean(),
                "CertifiedOptimalGapLowerMean": exact["CertifiedGapLower"].mean(),
                "CertifiedOptimalGapUpperMean": exact["CertifiedGapUpper"].mean(),
                "CertifiedOptimalReferenceIntervalMean": exact[
                    "ReferenceIntervalWidth"
                ].mean(),
                "CertifiedOptimalReferenceIntervalMax": exact[
                    "ReferenceIntervalWidth"
                ].max(),
                "CertifiedOptimalReferenceRelativeIntervalMax": exact[
                    "ReferenceRelativeIntervalWidth"
                ].max(),
                "CertifiedOptimumAttainmentRate": exact[
                    "CertifiedOptimumAttained"
                ].astype(float).mean(),
                "UnresolvedFeasibleRows": len(unresolved),
                "UnresolvedGapLowerMean": unresolved["CertifiedGapLower"].mean(),
                "UnresolvedGapUpperMean": unresolved["CertifiedGapUpper"].mean(),
                "AllFeasibleGapLowerMean": feasible_benchmark[
                    "CertifiedGapLower"
                ].mean(),
                "AllFeasibleGapUpperMean": feasible_benchmark[
                    "CertifiedGapUpper"
                ].mean(),
                "FalseFeasibleOnProvedInfeasible": int(proved_infeasible["Pass"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["Method", "Seed"]).reset_index(drop=True)


def stratified_bootstrap_mean_ci(
    values: pd.Series,
    seeds: pd.Series,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap a paired mean while preserving equal seed weighting."""
    seed_values = [
        values[seeds == seed].to_numpy(dtype=float)
        for seed in sorted(seeds.unique())
    ]
    if not seed_values or any(len(group) == 0 for group in seed_values):
        return math.nan, math.nan
    bootstrapped = np.empty((samples, len(seed_values)), dtype=float)
    for column, group in enumerate(seed_values):
        indices = rng.integers(0, len(group), size=(samples, len(group)))
        bootstrapped[:, column] = group[indices].mean(axis=1)
    draws = bootstrapped.mean(axis=1)
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return float(lower), float(upper)


def paired_certified_comparisons(
    results: pd.DataFrame,
    *,
    reference_method: str = "Ours (Hero)",
    bootstrap_samples: int = 10_000,
    random_seed: int = 260726,
) -> pd.DataFrame:
    """Compare one method with every alternative on paired certified rows."""
    if results.duplicated(["Method", "Seed", "uuid"]).any():
        raise RuntimeError("paired comparisons require unique method/seed/UUID rows")
    reference = results[results["Method"] == reference_method].copy()
    if reference.empty:
        raise RuntimeError(f"reference method is missing: {reference_method}")
    reference = reference[
        [
            "Seed",
            "uuid",
            "BenchmarkFeasible",
            "CertificateStatus",
            "Pass",
            "CertifiedGapLower",
            "CertifiedGapUpper",
        ]
    ].rename(
        columns={
            "BenchmarkFeasible": "ReferenceBenchmarkFeasible",
            "CertificateStatus": "ReferenceCertificateStatus",
            "Pass": "ReferencePass",
            "CertifiedGapLower": "ReferenceGapLower",
            "CertifiedGapUpper": "ReferenceGapUpper",
        }
    )

    rows = []
    methods = sorted(set(results["Method"]) - {reference_method})
    for offset, method in enumerate(methods):
        comparator = results[results["Method"] == method][
            [
                "Seed",
                "uuid",
                "BenchmarkFeasible",
                "CertificateStatus",
                "Pass",
                "CertifiedGapLower",
                "CertifiedGapUpper",
            ]
        ].rename(
            columns={
                "BenchmarkFeasible": "ComparatorBenchmarkFeasible",
                "CertificateStatus": "ComparatorCertificateStatus",
                "Pass": "ComparatorPass",
                "CertifiedGapLower": "ComparatorGapLower",
                "CertifiedGapUpper": "ComparatorGapUpper",
            }
        )
        paired = reference.merge(
            comparator,
            on=["Seed", "uuid"],
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(reference):
            raise RuntimeError(f"paired key mismatch for {method}")
        if not (
            paired["ReferenceBenchmarkFeasible"]
            == paired["ComparatorBenchmarkFeasible"]
        ).all():
            raise RuntimeError(f"benchmark-feasibility mismatch for {method}")
        if not (
            paired["ReferenceCertificateStatus"]
            == paired["ComparatorCertificateStatus"]
        ).all():
            raise RuntimeError(f"certificate-status mismatch for {method}")

        feasible = paired[paired["ReferenceBenchmarkFeasible"]].copy()
        exact = paired[paired["ReferenceCertificateStatus"] == "OPTIMAL"].copy()
        pass_difference = feasible["ReferencePass"].astype(float) - feasible[
            "ComparatorPass"
        ].astype(float)
        gap_difference = exact["ReferenceGapLower"] - exact["ComparatorGapLower"]
        gap_difference_upper = exact["ReferenceGapUpper"] - exact["ComparatorGapUpper"]
        pass_rng = np.random.default_rng(random_seed + 2 * offset)
        gap_rng = np.random.default_rng(random_seed + 2 * offset + 1)
        pass_lower, pass_upper = stratified_bootstrap_mean_ci(
            pass_difference,
            feasible["Seed"],
            samples=bootstrap_samples,
            rng=pass_rng,
        )
        gap_lower, gap_upper = stratified_bootstrap_mean_ci(
            gap_difference,
            exact["Seed"],
            samples=bootstrap_samples,
            rng=gap_rng,
        )
        rows.append(
            {
                "ReferenceMethod": reference_method,
                "ComparatorMethod": method,
                "FeasibleBenchmarkRows": len(feasible),
                "CertifiedOptimalRows": len(exact),
                "ReferencePassRate": feasible["ReferencePass"].mean(),
                "ComparatorPassRate": feasible["ComparatorPass"].mean(),
                "PairedPassDifference": pass_difference.mean(),
                "PairedPassDifferenceCI025": pass_lower,
                "PairedPassDifferenceCI975": pass_upper,
                "ReferenceCertifiedGap": exact["ReferenceGapLower"].mean(),
                "ComparatorCertifiedGap": exact["ComparatorGapLower"].mean(),
                "ReferenceCertifiedGapLower": exact["ReferenceGapLower"].mean(),
                "ReferenceCertifiedGapUpper": exact["ReferenceGapUpper"].mean(),
                "ComparatorCertifiedGapLower": exact["ComparatorGapLower"].mean(),
                "ComparatorCertifiedGapUpper": exact["ComparatorGapUpper"].mean(),
                "PairedCertifiedGapDifference": gap_difference.mean(),
                "PairedCertifiedGapDifferenceUpperReference": gap_difference_upper.mean(),
                "PairedCertifiedGapDifferenceCI025": gap_lower,
                "PairedCertifiedGapDifferenceCI975": gap_upper,
                "ReferenceGapWins": int((gap_difference < -CERTIFICATION_TOLERANCE).sum()),
                "CertifiedGapTies": int(
                    (gap_difference.abs() <= CERTIFICATION_TOLERANCE).sum()
                ),
                "ComparatorGapWins": int(
                    (gap_difference > CERTIFICATION_TOLERANCE).sum()
                ),
                "BootstrapSamples": bootstrap_samples,
                "BootstrapSeed": random_seed,
            }
        )
    return pd.DataFrame(rows).sort_values("ComparatorMethod").reset_index(drop=True)


def write_paired_comparison_bundle(
    comparisons: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str = "paired_certified_comparisons",
) -> None:
    """Persist paired certified comparisons and deterministic bootstrap CIs."""
    comparisons.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.json").write_text(
        comparisons.to_json(orient="records", indent=2)
    )


def paired_comparison_stem(reference_method: str) -> str:
    """Return a stable filename stem for an additional paired reference."""
    normalized = re.sub(r"[^a-z0-9]+", "_", reference_method.lower()).strip("_")
    if not normalized:
        raise ValueError("paired reference method has no filename-safe characters")
    return f"paired_{normalized}_comparisons"


def build_hero_excluded_vbs(results: pd.DataFrame) -> pd.DataFrame:
    """Compare Hero to a VBS that excludes both ordinary and frozen Hero."""
    references = compute_virtual_best_scores(
        results,
        excluded_methods=HERO_METHODS,
        output_column="HeroExcludedVBS",
    )
    hero = results[results["Method"] == "Ours (Hero)"].copy()
    if hero.duplicated(["Seed", "uuid"]).any():
        raise RuntimeError("Hero rows are not unique on (Seed, uuid)")
    columns = ["Seed", "uuid", "Pass", "llm_score"]
    if "CertificateStatus" in hero.columns:
        columns.extend(["CertificateStatus", "BenchmarkFeasible"])
    comparison = hero[columns].merge(
        references,
        on=["Seed", "uuid"],
        how="left",
        validate="one_to_one",
    )
    comparison = comparison.rename(
        columns={"Pass": "HeroPass", "llm_score": "HeroScore"}
    )
    comparison["HeroExcludedVBSAvailable"] = comparison["HeroExcludedVBS"].notna()
    comparable = comparison["HeroPass"] & comparison["HeroExcludedVBSAvailable"]
    comparison["HeroMinusExcludedVBS"] = math.nan
    comparison.loc[comparable, "HeroMinusExcludedVBS"] = (
        comparison.loc[comparable, "HeroScore"]
        - comparison.loc[comparable, "HeroExcludedVBS"]
    )
    comparison["HeroExcludedVBSGap"] = math.nan
    positive_reference = comparable & (comparison["HeroExcludedVBS"] > 1e-6)
    comparison.loc[positive_reference, "HeroExcludedVBSGap"] = (
        comparison.loc[positive_reference, "HeroExcludedVBS"]
        - comparison.loc[positive_reference, "HeroScore"]
    ) / comparison.loc[positive_reference, "HeroExcludedVBS"]
    comparison.loc[~comparison["HeroPass"] & comparison["HeroExcludedVBSAvailable"], "HeroExcludedVBSGap"] = 1.0
    return comparison


def write_certification_bundle(
    certified_results: pd.DataFrame,
    validation: dict,
    output_dir: Path,
) -> None:
    columns = [
        "Method",
        "Seed",
        "uuid",
        "CertificateStatus",
        "BenchmarkFeasible",
        "Pass",
        "llm_score",
        "ReferenceLower",
        "ReferenceUpper",
        "CertifiedGapLower",
        "CertifiedGapUpper",
        "CertifiedOptimumAttained",
        "ReferenceIntervalWidth",
        "ReferenceRelativeIntervalWidth",
    ]
    certified_results[columns].to_csv(
        output_dir / "certified_instance_metrics.csv", index=False
    )
    per_seed = summarise_certified_results(certified_results)
    per_seed.to_csv(output_dir / "certified_summary_by_seed.csv", index=False)
    (output_dir / "certified_summary_by_seed.json").write_text(
        per_seed.to_json(orient="records", indent=2)
    )
    aggregate = (
        per_seed.groupby("Method", observed=True)
        .agg(
            Seeds=("Seed", "nunique"),
            PassRateFeasibleBenchmarkMean=("PassRateFeasibleBenchmark", "mean"),
            PassRateFeasibleBenchmarkStd=("PassRateFeasibleBenchmark", "std"),
            CertifiedOptimalGapMean=("CertifiedOptimalGapMean", "mean"),
            CertifiedOptimalGapStd=("CertifiedOptimalGapMean", "std"),
            CertifiedOptimalGapLowerMean=("CertifiedOptimalGapLowerMean", "mean"),
            CertifiedOptimalGapUpperMean=("CertifiedOptimalGapUpperMean", "mean"),
            CertifiedOptimalReferenceIntervalMean=(
                "CertifiedOptimalReferenceIntervalMean",
                "mean",
            ),
            CertifiedOptimalReferenceIntervalMax=(
                "CertifiedOptimalReferenceIntervalMax",
                "max",
            ),
            CertifiedOptimalReferenceRelativeIntervalMax=(
                "CertifiedOptimalReferenceRelativeIntervalMax",
                "max",
            ),
            CertifiedOptimumAttainmentMean=(
                "CertifiedOptimumAttainmentRate",
                "mean",
            ),
            AllFeasibleGapLowerMean=("AllFeasibleGapLowerMean", "mean"),
            AllFeasibleGapUpperMean=("AllFeasibleGapUpperMean", "mean"),
            ProvedInfeasibleRows=("ProvedInfeasibleRows", "sum"),
            FalseFeasibleOnProvedInfeasible=(
                "FalseFeasibleOnProvedInfeasible",
                "sum",
            ),
        )
        .reset_index()
    )
    aggregate.to_csv(output_dir / "certified_summary.csv", index=False)
    (output_dir / "certified_summary.json").write_text(
        aggregate.to_json(orient="records", indent=2)
    )
    (output_dir / "certification_validation_summary.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    write_certified_family_bundle(certified_results, output_dir)


def write_certified_family_bundle(
    certified_results: pd.DataFrame, output_dir: Path
) -> None:
    """Report micro and equal-family sensitivity after any benchmark filtering."""
    rows = certified_results.copy()
    rows["Family"] = rows["uuid"].astype(str).map(
        lambda value: re.sub(r"_[0-9]+$", "", value)
    )
    feasible = rows[rows["BenchmarkFeasible"]].copy()
    grouped = (
        feasible.groupby(["Method", "Seed", "Family"], observed=True)
        .agg(
            Rows=("uuid", "size"),
            PassRate=("Pass", "mean"),
            CertifiedOptimalRows=("CertifiedOptimal", "sum"),
            CertifiedOptimalGapLower=(
                "CertifiedGapLower",
                lambda values: values[
                    feasible.loc[values.index, "CertifiedOptimal"]
                ].mean(),
            ),
            CertifiedOptimalGapUpper=(
                "CertifiedGapUpper",
                lambda values: values[
                    feasible.loc[values.index, "CertifiedOptimal"]
                ].mean(),
            ),
        )
        .reset_index()
    )
    grouped.to_csv(output_dir / "certified_summary_by_family.csv", index=False)
    (output_dir / "certified_summary_by_family.json").write_text(
        grouped.to_json(orient="records", indent=2)
    )
    per_seed = (
        grouped.groupby(["Method", "Seed"], observed=True)
        .agg(
            Families=("Family", "nunique"),
            FamilyMacroPassRate=("PassRate", "mean"),
            FamilyMacroCertifiedOptimalGapLower=("CertifiedOptimalGapLower", "mean"),
            FamilyMacroCertifiedOptimalGapUpper=("CertifiedOptimalGapUpper", "mean"),
        )
        .reset_index()
    )
    macro = (
        per_seed.groupby("Method", observed=True)
        .agg(
            Seeds=("Seed", "nunique"),
            FamiliesMin=("Families", "min"),
            FamiliesMax=("Families", "max"),
            FamilyMacroPassRateMean=("FamilyMacroPassRate", "mean"),
            FamilyMacroPassRateStd=("FamilyMacroPassRate", "std"),
            FamilyMacroCertifiedOptimalGapLowerMean=(
                "FamilyMacroCertifiedOptimalGapLower",
                "mean",
            ),
            FamilyMacroCertifiedOptimalGapUpperMean=(
                "FamilyMacroCertifiedOptimalGapUpper",
                "mean",
            ),
        )
        .reset_index()
    )
    macro.to_csv(output_dir / "certified_family_macro_summary.csv", index=False)
    (output_dir / "certified_family_macro_summary.json").write_text(
        macro.to_json(orient="records", indent=2)
    )


def write_hero_excluded_vbs_bundle(comparison: pd.DataFrame, output_dir: Path) -> None:
    comparison.to_csv(output_dir / "hero_excluded_vbs_instances.csv", index=False)
    summary_input = (
        comparison[comparison["BenchmarkFeasible"]]
        if "BenchmarkFeasible" in comparison.columns
        else comparison
    )
    summary = (
        summary_input.groupby("Seed", observed=True)
        .agg(
            Rows=("uuid", "size"),
            HeroPassRate=("HeroPass", "mean"),
            HeroExcludedVBSCoverage=("HeroExcludedVBSAvailable", "mean"),
            HeroMinusExcludedVBSMean=("HeroMinusExcludedVBS", "mean"),
            HeroExcludedVBSGapMean=("HeroExcludedVBSGap", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "hero_excluded_vbs_summary_by_seed.csv", index=False)
    (output_dir / "hero_excluded_vbs_summary_by_seed.json").write_text(
        summary.to_json(orient="records", indent=2)
    )
    (output_dir / "hero_excluded_vbs_manifest.json").write_text(
        json.dumps(
            {
                "join_keys": ["Seed", "uuid"],
                "excluded_methods": sorted(HERO_METHODS),
                "proved_infeasible_rows_excluded_from_summary": (
                    "BenchmarkFeasible" in comparison.columns
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def summarise_methods_by_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-method metrics for each seed."""
    per_seed = (
        df.groupby(["Method", "Seed"], observed=True)
        .agg(Pass=("Pass", "mean"), Gap=("Gap", "mean"), Cost=("Cost", "mean"))
        .reset_index()
    )
    per_seed = per_seed[per_seed["Method"].isin(MAIN_METHODS)].copy()
    per_seed["Method"] = pd.Categorical(per_seed["Method"], MAIN_METHODS, ordered=True)
    return per_seed.sort_values(["Method", "Seed"]).reset_index(drop=True)


def write_summary_bundle(summary: pd.DataFrame, output_dir: Path) -> None:
    """Persist summary CSV, JSON, and LaTeX."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "baseline_summary.csv", index=False)
    (output_dir / "baseline_summary.json").write_text(
        summary.to_json(orient="records", indent=2)
    )

    latex_df = pd.DataFrame(
        {
            "Method": summary["Method"],
            "Pass": summary.apply(
                lambda row: f"{row['PassMean'] * 100:.1f}$\\pm${row['PassStd'] * 100:.1f}\\%", axis=1
            ),
            "Gap": summary.apply(
                lambda row: f"{row['GapMean'] * 100:.2f}$\\pm${row['GapStd'] * 100:.2f}\\%", axis=1
            ),
            "Cost": summary.apply(
                lambda row: f"{row['CostMean']:.4f}$\\pm${row['CostStd']:.4f}", axis=1
            ),
        }
    )
    latex = latex_df.to_latex(index=False, escape=False)
    (output_dir / "baseline_comparison_table.tex").write_text(latex)


def write_per_seed_summary_bundle(per_seed: pd.DataFrame, output_dir: Path) -> None:
    """Persist per-seed summary CSV/JSON for auditability."""
    per_seed.to_csv(output_dir / "baseline_summary_by_seed.csv", index=False)
    (output_dir / "baseline_summary_by_seed.json").write_text(
        per_seed.to_json(orient="records", indent=2)
    )


def is_refreshed_shinka_result(path: Path) -> bool:
    """Return True for refreshed Shinka v2 test outputs only."""
    path_str = path.as_posix()
    return "ShinkaEvolve-SDS-1000-v2-seed" in path_str and "/test/" in path_str


def collect_timing_rows(
    timing_root: Path, runtime_seed: int, shinka_root: Path | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Load timing summaries and keep one representative row per method."""
    all_runtime_df = collect_all_timing_rows(timing_root, shinka_root)
    timing_df = all_runtime_df[all_runtime_df["Seed"] == runtime_seed].copy()
    timing_df = timing_df.drop_duplicates(subset=["Method", "Seed"], keep="last")
    missing_methods = [
        method for method in RUNTIME_METHODS if method not in timing_df["Method"].tolist()
    ]
    if missing_methods:
        placeholder_df = pd.DataFrame(
            [
                {
                    "Method": method,
                    "Seed": runtime_seed,
                    "RunType": "pending",
                    "GenerationWallClock": pd.NA,
                    "EvaluationWallClock": pd.NA,
                    "TotalWallClock": pd.NA,
                    "Dataset": None,
                    "Path": None,
                    "TimingStatus": "pending",
                }
                for method in missing_methods
            ]
        )
        timing_df = pd.concat([timing_df, placeholder_df], ignore_index=True)
    timing_df["Method"] = pd.Categorical(timing_df["Method"], RUNTIME_METHODS, ordered=True)
    return timing_df.sort_values("Method").reset_index(drop=True), missing_methods


def collect_all_timing_rows(
    timing_root: Path, shinka_root: Path | None = None
) -> pd.DataFrame:
    """Load all available timing summaries without restricting to one seed."""
    rows = []
    for timing_file in timing_root.rglob("timing_summary.json"):
        data = json.loads(timing_file.read_text())
        rows.append(
            {
                "Method": infer_runtime_method_name(data),
                "Seed": data.get("seed"),
                "RunType": data.get("run_type"),
                "GenerationWallClock": data.get("generation_wall_clock_seconds", 0.0),
                "EvaluationWallClock": data.get("evaluation_wall_clock_seconds", 0.0),
                "Dataset": data.get("dataset"),
                "Path": str(timing_file),
                "TimingStatus": "available",
            }
        )

    if shinka_root is not None:
        for timing_file in shinka_root.rglob("timing_summary.json"):
            if not is_refreshed_shinka_result(timing_file):
                continue
            data = json.loads(timing_file.read_text())
            rows.append(
                {
                    "Method": infer_runtime_method_name(data),
                    "Seed": data.get("seed"),
                    "RunType": data.get("run_type"),
                    "GenerationWallClock": data.get("generation_wall_clock_seconds", 0.0),
                    "EvaluationWallClock": data.get("evaluation_wall_clock_seconds", 0.0),
                    "Dataset": data.get("dataset"),
                    "Path": str(timing_file),
                    "TimingStatus": "available",
                }
            )

    timing_df = pd.DataFrame(rows)
    if timing_df.empty:
        raise RuntimeError(f"No timing_summary.json files found under {timing_root}")

    timing_df = timing_df[timing_df["Method"].isin(RUNTIME_METHODS)].copy()
    timing_df["TotalWallClock"] = (
        timing_df["GenerationWallClock"] + timing_df["EvaluationWallClock"]
    )
    timing_df = timing_df.drop_duplicates(subset=["Method", "Seed"], keep="last")
    timing_df["Method"] = pd.Categorical(timing_df["Method"], RUNTIME_METHODS, ordered=True)
    return timing_df.sort_values(["Method", "Seed"]).reset_index(drop=True)


def augment_runtime_with_classical_baselines(
    timing_df: pd.DataFrame, combined_df: pd.DataFrame, runtime_seed: int
) -> pd.DataFrame:
    """Append classical baseline totals derived from per-instance costs."""
    seed_df = combined_df[
        (combined_df["Seed"] == runtime_seed)
        & (combined_df["Method"].isin(["CP-SAT", "Local Search", "Greedy", "BnB"]))
    ].copy()
    classical = (
        seed_df.groupby("Method", observed=True)["Cost"].sum().reset_index(name="TotalWallClock")
    )
    classical["GenerationWallClock"] = 0.0
    classical["EvaluationWallClock"] = classical["TotalWallClock"]
    classical["Dataset"] = None
    classical["Path"] = "derived-from-metrics_final.csv"
    classical["TimingStatus"] = "derived"

    out = pd.concat([timing_df, classical], ignore_index=True)
    method_order = MAIN_METHODS
    out["Method"] = pd.Categorical(out["Method"], method_order, ordered=True)
    return out.sort_values("Method").reset_index(drop=True)


def write_runtime_bundle(runtime_df: pd.DataFrame, output_dir: Path) -> None:
    """Persist runtime CSV/JSON."""
    runtime_df.to_csv(output_dir / "runtime_summary.csv", index=False)
    (output_dir / "runtime_summary.json").write_text(
        runtime_df.to_json(orient="records", indent=2)
    )


def write_runtime_audit_bundle(all_runtime_df: pd.DataFrame, output_dir: Path) -> None:
    """Persist all available runtime rows and their aggregate over available seeds."""
    all_runtime_df.to_csv(output_dir / "runtime_summary_by_seed.csv", index=False)
    (output_dir / "runtime_summary_by_seed.json").write_text(
        all_runtime_df.to_json(orient="records", indent=2)
    )

    numeric = ["GenerationWallClock", "EvaluationWallClock", "TotalWallClock"]
    aggregate = (
        all_runtime_df.groupby("Method", observed=True)
        .agg(
            SeedsReported=("Seed", "nunique"),
            GenerationWallClockMean=("GenerationWallClock", "mean"),
            GenerationWallClockStd=("GenerationWallClock", "std"),
            EvaluationWallClockMean=("EvaluationWallClock", "mean"),
            EvaluationWallClockStd=("EvaluationWallClock", "std"),
            TotalWallClockMean=("TotalWallClock", "mean"),
            TotalWallClockStd=("TotalWallClock", "std"),
        )
        .reset_index()
    )
    aggregate = aggregate.sort_values("Method").reset_index(drop=True)
    aggregate.to_csv(output_dir / "runtime_summary_available_seeds.csv", index=False)
    (output_dir / "runtime_summary_available_seeds.json").write_text(
        aggregate.to_json(orient="records", indent=2)
    )


def write_results_note(
    output_dir: Path, summary: pd.DataFrame, runtime_df: pd.DataFrame, missing_runtime_methods: list[str]
) -> None:
    """Create a short publication-facing note."""
    frozen_row = summary[summary["Method"] == "Frozen Hero"].iloc[0]
    sa_row = summary[summary["Method"] == "Hand-written SA"].iloc[0]
    if missing_runtime_methods:
        runtime_note = (
            "- Runtime summary placeholder: "
            + ", ".join(missing_runtime_methods)
            + " timing is still pending and should be refreshed after the planned rerun."
        )
    else:
        runtime_note = (
            "- Runtime summary uses fresh seed101 timing reruns for Ours Hero, Base Best-of-64, "
            "ShinkaEvolve, Frozen Hero, and Hand-written SA."
        )
    note = f"""# Baseline Evaluation Notes

- Quality summary (`baseline_summary.csv`) is aggregated across all three seeds: 101, 202, and 303. Per-seed values are in `baseline_summary_by_seed.csv`.
- Frozen Hero selection provenance is validated from its immutable freeze manifest; no test outcome is used for synthesis or selection.
- Failure semantics: a non-pass means the method did not return a feasible solution within the fixed evaluation budget. It should not be described generically as a "violation" unless the returned program actually violated constraints.
- Frozen Hero mean pass rate: {frozen_row['PassMean'] * 100:.1f}%.
- Frozen Hero mean optimality gap: {frozen_row['GapMean'] * 100:.2f}%.
- Hand-written SA mean pass rate: {sa_row['PassMean'] * 100:.1f}%.
- Hand-written SA mean optimality gap: {sa_row['GapMean'] * 100:.2f}%.
- Runtime audit across all available seeds is in `runtime_summary_by_seed.csv` and `runtime_summary_available_seeds.csv`.
{runtime_note}
"""
    (output_dir / "results_notes.md").write_text(note)


def load_fixed_code_results(fixed_code_root: Path) -> pd.DataFrame:
    """Load and summarize Frozen Hero + Hand-written SA results."""
    files = find_all_metrics_files(str(fixed_code_root))
    if not files:
        raise RuntimeError(f"No metrics_final.csv files found under {fixed_code_root}")
    fixed_df = load_all_data(files, include_baselines=False)
    fixed_df = fixed_df[fixed_df["Method"].isin(["Frozen Hero", "Hand-written SA"])].copy()
    if fixed_df.empty:
        raise RuntimeError(f"No fixed-code methods found under {fixed_code_root}")
    return fixed_df


def load_validation_selected_frozen_hero(
    correction_root: Path,
    protocol_path: Path = Path(
        "experiments/neurips2026/frozen_hero_validation_protocol.json"
    ),
) -> pd.DataFrame:
    """Load the corrected validation-selected Frozen Hero without latest-file discovery."""
    protocol = json.loads(protocol_path.read_text())
    expected_seeds = {int(seed) for seed in protocol["datasets"]}
    metrics_files: list[str] = []
    for seed in sorted(expected_seeds):
        seed_root = correction_root / f"seed{seed}"
        freeze_path = seed_root / "selection" / "pretest_freeze_manifest.json"
        solver_path = seed_root / "selection" / "selected_solver.py"
        summaries_path = seed_root / "selection" / "candidate_summaries.jsonl"
        if not freeze_path.is_file() or not solver_path.is_file() or not summaries_path.is_file():
            raise RuntimeError(f"Frozen Hero seed {seed} corrected selection is incomplete")
        freeze = json.loads(freeze_path.read_text())
        correction = freeze.get("correction")
        if freeze.get("protocol_id") != protocol["protocol_id"] or int(
            freeze.get("seed", -1)
        ) != seed:
            raise RuntimeError(f"Frozen Hero seed {seed} freeze identity drift")
        if freeze.get("test_outcomes_accessed") is not False:
            raise RuntimeError(f"Frozen Hero seed {seed} was not frozen before test")
        source_freeze_path = None
        if correction is not None:
            if correction.get("generations_reused_without_regeneration") is not True:
                raise RuntimeError(
                    f"Frozen Hero seed {seed} correction regenerated candidates"
                )
            source_freeze_path = Path(correction["source_freeze_path"])
            if not source_freeze_path.is_file() or hashlib.sha256(
                source_freeze_path.read_bytes()
            ).hexdigest() != correction.get("source_freeze_sha256"):
                raise RuntimeError(f"Frozen Hero seed {seed} source freeze hash drift")
            source_freeze = json.loads(source_freeze_path.read_text())
            if source_freeze.get("generation_file_sha256") != freeze.get(
                "generation_file_sha256"
            ):
                raise RuntimeError(f"Frozen Hero seed {seed} generation provenance drift")
        selected_sha = hashlib.sha256(solver_path.read_bytes()).hexdigest()
        if selected_sha != freeze.get("selected_code_sha256"):
            raise RuntimeError(f"Frozen Hero seed {seed} selected code hash drift")
        summaries = [
            json.loads(line) for line in summaries_path.read_text().splitlines() if line
        ]
        if not 1 <= len(summaries) <= int(protocol["synthesis_candidates"]):
            raise RuntimeError(f"Frozen Hero seed {seed} unique candidate count drift")
        selected_summary = freeze["selected_development_summary"]
        if "benchmark_feasible_development_evaluations" not in selected_summary:
            raise RuntimeError(f"Frozen Hero seed {seed} lacks corrected pass semantics")
        if selected_summary.get("false_feasible_on_proved_infeasible") != 0:
            raise RuntimeError(f"Frozen Hero seed {seed} false-feasible validation output")

        changed = correction is not None and correction.get("selection_changed") is True
        test_seed_root = (
            seed_root
            if correction is None or changed
            else source_freeze_path.parents[1]
        )
        test_dir = test_seed_root / "test" / "fixed-code" / "frozen-hero" / f"seed{seed}"
        metrics_path = test_dir / "metrics_final.csv"
        metadata_path = test_dir / "experiment_metadata.json"
        if not metrics_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Frozen Hero seed {seed} selected test result is incomplete")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("method_name") != "Frozen Hero":
            raise RuntimeError(f"Frozen Hero seed {seed} method label drift")
        metrics = pd.read_csv(metrics_path)
        if len(metrics) != int(protocol["test_evaluation"]["rows_per_seed"]):
            raise RuntimeError(f"Frozen Hero seed {seed} test row count drift")
        if "uuid" not in metrics or metrics["uuid"].duplicated().any():
            raise RuntimeError(f"Frozen Hero seed {seed} test UUID drift")
        metrics_files.append(str(metrics_path))

    observed = {
        int(path.name.removeprefix("seed"))
        for path in correction_root.glob("seed*")
        if path.is_dir() and path.name.removeprefix("seed").isdigit()
    }
    if observed != expected_seeds:
        raise RuntimeError(f"Frozen Hero corrected seed set drift: {sorted(observed)}")
    frozen = load_all_data(metrics_files, include_baselines=False)
    frozen = frozen[frozen["Method"] == "Frozen Hero"].copy()
    if len(frozen) != len(expected_seeds) * int(
        protocol["test_evaluation"]["rows_per_seed"]
    ):
        raise RuntimeError("Frozen Hero loaded test rows drift")
    return frozen


def load_main_report_results(main_report_set: str) -> pd.DataFrame:
    """Load canonical comparison methods from the main report set."""
    report_set = load_report_set(main_report_set)
    explicit = report_set["sds"].get("canonical_metrics_files")
    if explicit:
        selected = [str(Path(path)) for path in explicit]
        if len(selected) != len(set(selected)):
            raise RuntimeError("canonical_metrics_files contains duplicate paths")
        missing = [path for path in selected if not Path(path).is_file()]
        if missing:
            raise RuntimeError(f"canonical metrics files are missing: {missing}")
    else:
        result_roots = report_set["sds"]["result_roots"]
        files = find_all_metrics_files_from_roots(result_roots)
        selected = select_latest_jobs(
            files,
            max_jobs=25,
            jobs_per_seed=1,
            allowed_methods=["Ours (Hero)", "Base (Best-of-64)", "ShinkaEvolve"],
        )
    main_df = load_all_data(selected, include_baselines=True)
    return main_df[main_df["Method"].isin(MAIN_METHODS)].copy()


def load_shinka_override_results(shinka_root: Path) -> pd.DataFrame:
    """Load refreshed Shinka results and present them as the active Shinka line."""
    files = [
        str(path)
        for path in shinka_root.rglob("metrics_final.csv")
        if is_refreshed_shinka_result(path)
    ]
    if not files:
        raise RuntimeError(f"No metrics_final.csv files found under {shinka_root}")
    shinka_df = load_all_data(files, include_baselines=False)
    shinka_df = shinka_df[shinka_df["Method"] == "ShinkaEvolve"].copy()
    if shinka_df.empty:
        raise RuntimeError(f"No ShinkaEvolve rows found under {shinka_root}")
    return shinka_df


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base_rescore_results(
    base_rescore_root: Path, result_manifest_path: Path
) -> pd.DataFrame:
    """Load only the registered corrected immutable-completion Base rescore."""
    manifest = json.loads(result_manifest_path.read_text())
    protocol_path = result_manifest_path.parent / manifest["protocol_file"]
    protocol = json.loads(protocol_path.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("Base rescore result manifest is not complete")
    if manifest.get("protocol_id") != protocol.get("protocol_id"):
        raise RuntimeError("Base rescore protocol ID drift")
    if manifest.get("protocol_sha256") != _sha256_file(protocol_path):
        raise RuntimeError("Base rescore protocol hash drift")
    if Path(manifest.get("artifact_root", "")) != base_rescore_root:
        raise RuntimeError("Base rescore artifact root drift")
    if manifest.get("account") != "PUBLIC_ALLOCATION_ACCOUNT":
        raise RuntimeError("Base rescore allocation account drift")

    expected_seeds = {101, 202, 303}
    manifest_seeds = {int(seed) for seed in manifest.get("seeds", {})}
    if manifest_seeds != expected_seeds:
        raise RuntimeError(f"Base rescore manifest seed drift: {sorted(manifest_seeds)}")
    protocol_datasets = {
        int(row["seed"]): row for row in protocol.get("datasets", [])
    }
    if set(protocol_datasets) != expected_seeds:
        raise RuntimeError("Base rescore protocol seed drift")

    metrics_files: list[str] = []
    for seed in sorted(expected_seeds):
        registered = manifest["seeds"][str(seed)]
        protocol_dataset = protocol_datasets[seed]
        if registered.get("dataset_repo") != protocol_dataset["repo"]:
            raise RuntimeError(f"Base rescore seed {seed} dataset repository drift")
        if registered.get("dataset_revision") != protocol_dataset["revision"]:
            raise RuntimeError(f"Base rescore seed {seed} dataset revision drift")
        if int(registered.get("source_records", -1)) != int(
            protocol_dataset["expected_count"]
        ):
            raise RuntimeError(f"Base rescore seed {seed} source record drift")
        if int(registered.get("collapsed_rows", -1)) != 1000:
            raise RuntimeError(f"Base rescore seed {seed} collapsed row drift")
        scheduler = registered.get("scheduler", {})
        if scheduler.get("state") != "COMPLETED":
            raise RuntimeError(f"Base rescore seed {seed} scheduler state drift")
        if int(scheduler.get("elapsed_seconds", 0)) <= 0:
            raise RuntimeError(f"Base rescore seed {seed} scheduler elapsed drift")
        if int(scheduler.get("allocated_gpus", 0)) != 4:
            raise RuntimeError(f"Base rescore seed {seed} scheduler allocation drift")

        seed_root = base_rescore_root / f"seed{seed}"
        artifacts = registered.get("artifacts_sha256", {})
        required_artifacts = {
            "metrics_final.csv",
            "duplicate_selection_audit.json",
            "duplicate_selection_audit.csv",
            "timing_summary.json",
            "experiment_metadata.json",
            "input_revision.txt",
            "input_sha256.txt",
        }
        if set(artifacts) != required_artifacts:
            raise RuntimeError(f"Base rescore seed {seed} artifact registration drift")
        for relative, expected_sha in artifacts.items():
            artifact = seed_root / relative
            if not artifact.is_file() or _sha256_file(artifact) != expected_sha:
                raise RuntimeError(
                    f"Base rescore seed {seed} artifact hash drift: {relative}"
                )

        expected_revision = (
            f"{registered['dataset_repo']}@{registered['dataset_revision']}"
        )
        if (seed_root / "input_revision.txt").read_text().strip() != expected_revision:
            raise RuntimeError(f"Base rescore seed {seed} input revision drift")
        input_sha = (seed_root / "input_sha256.txt").read_text().split(maxsplit=1)[0]
        if input_sha != registered.get("source_file_sha256"):
            raise RuntimeError(f"Base rescore seed {seed} input file hash drift")

        audit = json.loads((seed_root / "duplicate_selection_audit.json").read_text())
        if int(audit.get("evaluated_record_count", -1)) != int(
            registered["source_records"]
        ):
            raise RuntimeError(f"Base rescore seed {seed} audit count drift")
        registered_audit = registered.get("audit", {})
        expected_audit_keys = {
            "records_with_duplicate_selections",
            "duplicate_id_occurrences",
            "invalid_selection_records",
        }
        if set(registered_audit) != expected_audit_keys:
            raise RuntimeError(f"Base rescore seed {seed} audit registration drift")
        if any(
            int(audit.get(key, -1)) != int(registered_audit[key])
            for key in expected_audit_keys
        ):
            raise RuntimeError(f"Base rescore seed {seed} audit value drift")
        timing = json.loads((seed_root / "timing_summary.json").read_text())
        if int(timing.get("num_records_evaluated", -1)) != int(
            registered["source_records"]
        ) or int(timing.get("num_unique_instances", -1)) != int(
            registered["collapsed_rows"]
        ):
            raise RuntimeError(f"Base rescore seed {seed} timing count drift")
        metrics_path = seed_root / "metrics_final.csv"
        metrics = pd.read_csv(metrics_path)
        if len(metrics) != int(registered["collapsed_rows"]):
            raise RuntimeError(f"Base rescore seed {seed} metric count drift")
        if "uuid" not in metrics or metrics["uuid"].duplicated().any():
            raise RuntimeError(f"Base rescore seed {seed} metric UUID drift")
        metrics_files.append(str(metrics_path))

    observed_seed_dirs = {
        int(path.name.removeprefix("seed"))
        for path in base_rescore_root.glob("seed*")
        if path.is_dir() and path.name.removeprefix("seed").isdigit()
    }
    if observed_seed_dirs != expected_seeds:
        raise RuntimeError(
            f"Base rescore seed directory drift: {sorted(observed_seed_dirs)}"
        )

    base_df = load_all_data(metrics_files, include_baselines=False)
    base_df = base_df[base_df["Method"] == "Base (Best-of-64)"].copy()
    if len(base_df) != 3000:
        raise RuntimeError("Base rescore loaded row count drift")
    if base_df.duplicated(["Seed", "uuid"]).any():
        raise RuntimeError("Base rescore loaded key drift")
    return base_df


def load_additional_method_results(additional_root: Path) -> pd.DataFrame:
    """Load a predeclared extra method without creating a second aggregator."""
    files = find_all_metrics_files(str(additional_root))
    if not files:
        raise RuntimeError(f"No metrics_final.csv files found under {additional_root}")
    additional_df = load_all_data(files, include_baselines=False)
    if additional_df.empty:
        raise RuntimeError(f"No method rows found under {additional_root}")
    duplicate = additional_df.duplicated(["Method", "Seed", "uuid"], keep=False)
    if duplicate.any():
        keys = additional_df.loc[duplicate, ["Method", "Seed", "uuid"]].head()
        raise RuntimeError(f"Duplicate additional-method keys: {keys.to_dict('records')}")
    return additional_df


def load_adaptive_repair_results(  # noqa: PLR0912, PLR0915
    adaptive_root: Path,
    protocol_path: Path = Path("experiments/neurips2026/adaptive_repair_protocol.json"),
    test_root: Path | None = None,
    test_addendum_path: Path = Path(
        "experiments/neurips2026/adaptive_repair_test_addendum.json"
    ),
) -> pd.DataFrame:
    """Validate and load completed development-only Adaptive Repair runs."""
    protocol_text = protocol_path.read_text()
    protocol = json.loads(protocol_text)
    protocol_sha256 = hashlib.sha256(protocol_text.encode()).hexdigest()
    expected_seeds = {int(seed) for seed in protocol["datasets"]}
    expected_completions = int(protocol["total_completion_budget"])
    expected_dev_rows = int(protocol["expected_development_cases"])
    expected_test_rows = int(protocol["expected_split_count"])
    expected_round_counts = {
        0: int(protocol["initial_completions"]),
        **{
            round_index: int(protocol["completions_per_repair_round"])
            for round_index in range(1, int(protocol["repair_rounds"]) + 1)
        },
    }
    aligned_test = test_root is not None
    test_root = test_root or adaptive_root
    test_addendum = json.loads(test_addendum_path.read_text()) if aligned_test else None

    metrics_files: list[str] = []
    for seed in sorted(expected_seeds):
        seed_dir = adaptive_root / f"seed{seed}"
        test_seed_dir = test_root / f"seed{seed}"
        synthesis = seed_dir / "synthesis"
        correction_path = synthesis / "correction_manifest.json"
        corrected = correction_path.is_file()
        required = {
            "candidate summaries": synthesis / "candidate_summaries.jsonl",
            "development evaluations": synthesis / "development_evaluations.jsonl",
            "selected solver": synthesis / "selected_solver.py",
            "test metrics": test_seed_dir / "test-evaluation" / "metrics_final.csv",
            "test metadata": test_seed_dir / "test-evaluation" / "experiment_metadata.json",
        }
        if corrected:
            required["correction manifest"] = correction_path
        else:
            required.update(
                {
                    "run manifest": synthesis / "run_manifest.json",
                    "generations": synthesis / "generations.jsonl",
                }
            )
        missing = [label for label, path in required.items() if not path.is_file()]
        if missing:
            raise RuntimeError(f"Adaptive Repair seed {seed} is incomplete: {missing}")

        if corrected:
            correction = json.loads(required["correction manifest"].read_text())
            if correction.get("correction_id") != "N26-E4-unique-selection-rescore-v1":
                raise RuntimeError(f"Adaptive Repair seed {seed} correction ID drift")
            source_synthesis = Path(correction["source_synthesis_dir"])
            source_manifest_path = source_synthesis / "run_manifest.json"
            source_generations_path = source_synthesis / "generations.jsonl"
            if not source_manifest_path.is_file() or not source_generations_path.is_file():
                raise RuntimeError(f"Adaptive Repair seed {seed} correction source missing")
            if hashlib.sha256(source_manifest_path.read_bytes()).hexdigest() != correction.get(
                "source_run_manifest_sha256"
            ):
                raise RuntimeError(f"Adaptive Repair seed {seed} source manifest hash drift")
            if hashlib.sha256(source_generations_path.read_bytes()).hexdigest() != correction.get(
                "source_generations_sha256"
            ):
                raise RuntimeError(f"Adaptive Repair seed {seed} source generations hash drift")
            run_manifest = json.loads(source_manifest_path.read_text())
            generations_path = source_generations_path
            selected_index = int(correction["corrected_selected_completion_index"])
            selected_hash = correction.get("selected_code_sha256")
            correction_invariants = {
                "protocol_id": (correction.get("protocol_id"), protocol["protocol_id"]),
                "seed": (correction.get("seed"), seed),
                "completion_count": (correction.get("completion_count"), expected_completions),
                "development_execution_count": (
                    correction.get("development_execution_count"),
                    expected_completions * expected_dev_rows,
                ),
            }
            correction_drift = {
                name: {"actual": actual, "expected": expected}
                for name, (actual, expected) in correction_invariants.items()
                if actual != expected
            }
            if correction_drift:
                raise RuntimeError(
                    f"Adaptive Repair seed {seed} correction drift: {correction_drift}"
                )
        else:
            run_manifest = json.loads(required["run manifest"].read_text())
            generations_path = required["generations"]
            selected_index = int(run_manifest["selected_completion_index"])
            selected_hash = run_manifest.get("selected_code_sha256")
        invariants = {
            "protocol_id": (run_manifest.get("protocol_id"), protocol["protocol_id"]),
            "protocol_sha256": (
                run_manifest.get("protocol_sha256"),
                protocol_sha256,
            ),
            "seed": (run_manifest.get("seed"), seed),
            "model": (run_manifest.get("model"), protocol["model"]),
            "completion_count": (
                run_manifest.get("completion_count"),
                expected_completions,
            ),
            "development_execution_count": (
                run_manifest.get("development_execution_count"),
                expected_completions * expected_dev_rows,
            ),
        }
        drift = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in invariants.items()
            if actual != expected
        }
        if drift:
            raise RuntimeError(f"Adaptive Repair seed {seed} manifest drift: {drift}")

        generations = [
            json.loads(line)
            for line in generations_path.read_text().splitlines()
            if line
        ]
        completion_indices = [int(row["completion_index"]) for row in generations]
        if sorted(completion_indices) != list(range(expected_completions)):
            raise RuntimeError(f"Adaptive Repair seed {seed} completion indices drift")
        round_counts = {
            round_index: sum(
                int(row["round_index"]) == round_index for row in generations
            )
            for round_index in expected_round_counts
        }
        if round_counts != expected_round_counts:
            raise RuntimeError(
                f"Adaptive Repair seed {seed} completion schedule drift: {round_counts}"
            )

        summaries = [
            json.loads(line)
            for line in required["candidate summaries"].read_text().splitlines()
            if line
        ]
        if len(summaries) != expected_completions:
            raise RuntimeError(f"Adaptive Repair seed {seed} candidate count drift")
        evaluations = [
            json.loads(line)
            for line in required["development evaluations"].read_text().splitlines()
            if line
        ]
        evaluation_keys = {
            (int(row["completion_index"]), str(row["uuid"])) for row in evaluations
        }
        if len(evaluations) != expected_completions * expected_dev_rows or len(
            evaluation_keys
        ) != len(evaluations):
            raise RuntimeError(f"Adaptive Repair seed {seed} development audit drift")

        selected_code = required["selected solver"].read_text()
        selected_sha256 = hashlib.sha256(selected_code.encode()).hexdigest()
        if selected_sha256 != selected_hash:
            raise RuntimeError(f"Adaptive Repair seed {seed} selected code hash drift")
        if selected_index not in completion_indices:
            raise RuntimeError(f"Adaptive Repair seed {seed} selected index drift")
        if selected_index not in {
            int(summary["completion_index"]) for summary in summaries
        }:
            raise RuntimeError(f"Adaptive Repair seed {seed} selected summary missing")

        if aligned_test:
            reevaluation_path = test_seed_dir / "reevaluation_manifest.json"
            if not reevaluation_path.is_file():
                raise RuntimeError(
                    f"Adaptive Repair seed {seed} aligned reevaluation manifest missing"
                )
            reevaluation = json.loads(reevaluation_path.read_text())
            aligned_invariants = {
                "protocol_id": (
                    reevaluation.get("protocol_id"),
                    test_addendum["protocol_id"],
                ),
                "seed": (reevaluation.get("seed"), seed),
                "source_solver_sha256": (
                    reevaluation.get("source_solver_sha256"),
                    selected_sha256,
                ),
                "expected_solver_sha256": (
                    reevaluation.get("expected_solver_sha256"),
                    selected_sha256,
                ),
                "execution_repeats": (
                    reevaluation.get("execution_repeats"),
                    int(test_addendum["execution_repeats"]),
                ),
                "quality_uses_first_execution": (
                    reevaluation.get("quality_uses_first_execution"),
                    True,
                ),
                "regenerated": (reevaluation.get("regenerated"), False),
                "reselected": (reevaluation.get("reselected"), False),
            }
            aligned_drift = {
                name: {"actual": actual, "expected": expected}
                for name, (actual, expected) in aligned_invariants.items()
                if actual != expected
            }
            if aligned_drift:
                raise RuntimeError(
                    f"Adaptive Repair seed {seed} aligned test drift: {aligned_drift}"
                )

        metadata = json.loads(required["test metadata"].read_text())
        if metadata.get("method_name") != "Adaptive Repair (64 completions)":
            raise RuntimeError(f"Adaptive Repair seed {seed} method label drift")
        test_metrics = pd.read_csv(required["test metrics"])
        if len(test_metrics) != expected_test_rows:
            raise RuntimeError(f"Adaptive Repair seed {seed} test row count drift")
        if "uuid" not in test_metrics or test_metrics["uuid"].duplicated().any():
            raise RuntimeError(f"Adaptive Repair seed {seed} test UUID drift")
        metrics_files.append(str(required["test metrics"]))

    observed_seed_dirs = {
        int(path.name.removeprefix("seed"))
        for path in adaptive_root.glob("seed*")
        if path.is_dir() and path.name.removeprefix("seed").isdigit()
    }
    if observed_seed_dirs != expected_seeds:
        raise RuntimeError(
            f"Adaptive Repair seed directory drift: {sorted(observed_seed_dirs)}"
        )
    observed_test_seed_dirs = {
        int(path.name.removeprefix("seed"))
        for path in test_root.glob("seed*")
        if path.is_dir() and path.name.removeprefix("seed").isdigit()
    }
    if observed_test_seed_dirs != expected_seeds:
        raise RuntimeError(
            "Adaptive Repair test seed directory drift: "
            f"{sorted(observed_test_seed_dirs)}"
        )

    adaptive_df = load_all_data(metrics_files, include_baselines=False)
    adaptive_df = adaptive_df[
        adaptive_df["Method"] == "Adaptive Repair (64 completions)"
    ].copy()
    if len(adaptive_df) != expected_test_rows * len(expected_seeds):
        raise RuntimeError("Adaptive Repair loaded test rows drift")
    return adaptive_df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate fixed-code SDS results against canonical baselines."
    )
    parser.add_argument("--main-report-set", required=True, help="Canonical main SDS report set JSON.")
    parser.add_argument(
        "--fixed-code-root",
        default=None,
        help="Fixed-code root; defaults to the report set publication evidence entry.",
    )
    parser.add_argument(
        "--timing-root",
        default=None,
        help="Timing root; defaults to the report set publication evidence entry.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for aggregation outputs.")
    parser.add_argument(
        "--shinka-root",
        default=None,
        help="Optional root containing refreshed ShinkaEvolve metrics_final.csv files to override the canonical Shinka line.",
    )
    parser.add_argument(
        "--base-rescore-root",
        default=None,
        help="Optional root containing corrected immutable-completion Base Best-of-64 metrics.",
    )
    parser.add_argument(
        "--base-rescore-result-manifest",
        default=None,
        help="Registered hashes, source revisions, and scheduler records for --base-rescore-root.",
    )
    parser.add_argument(
        "--certification-root",
        default=None,
        help="Optional root containing seed*/certifications.jsonl from the frozen batch.",
    )
    parser.add_argument(
        "--certification-validation-summary",
        default=None,
        help="Independent validation_summary.json for --certification-root.",
    )
    parser.add_argument(
        "--adaptive-repair-root",
        default=None,
        help="Optional immutable root containing validated seed*/ Adaptive Repair runs.",
    )
    parser.add_argument(
        "--adaptive-repair-test-root",
        default=None,
        help="Optional immutable aligned test root; synthesis integrity remains under --adaptive-repair-root.",
    )
    parser.add_argument(
        "--frozen-hero-validation-root",
        default=None,
        help="Corrected validation-selected Frozen Hero root with explicit per-seed manifests.",
    )
    parser.add_argument(
        "--frozen-hero-protocol",
        default="experiments/neurips2026/frozen_hero_validation_protocol.json",
        help="Frozen Hero protocol used to validate --frozen-hero-validation-root.",
    )
    parser.add_argument(
        "--adaptive-repair-protocol",
        default="experiments/neurips2026/adaptive_repair_protocol.json",
        help="Adaptive Repair protocol used to validate its immutable root.",
    )
    parser.add_argument(
        "--additional-method-root",
        action="append",
        default=[],
        help="Optional immutable result root for another predeclared method; may be repeated.",
    )
    parser.add_argument(
        "--runtime-seed",
        type=int,
        default=101,
        help="Seed used for representative runtime summaries (default: 101).",
    )
    parser.add_argument(
        "--benchmark-subset",
        default=None,
        help="Optional immutable JSON manifest restricting every method and certificate by (seed, uuid).",
    )
    parser.add_argument(
        "--required-method",
        action="append",
        default=[],
        help="Fail if this method is absent after all overrides and additions; may be repeated.",
    )
    parser.add_argument(
        "--paired-reference-method",
        action="append",
        default=[],
        help=(
            "Write a paired certified comparison using this method as the reference; "
            "may be repeated."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_set = load_report_set(args.main_report_set)
    sds_report = report_set.get("sds", {})
    evidence = sds_report.get("publication_evidence")
    if evidence is None:
        evidence = sds_report.get("publication_evidence", {})
    fixed_code_root = args.fixed_code_root or evidence.get("fixed_code_root")
    timing_root = args.timing_root or evidence.get("timing_root")
    shinka_root = args.shinka_root or evidence.get("shinka_root")
    base_rescore_root = args.base_rescore_root or evidence.get("base_rescore_root")
    base_rescore_result_manifest = args.base_rescore_result_manifest or evidence.get(
        "base_rescore_result_manifest"
    )
    certification_root = args.certification_root or evidence.get("certification_root")
    adaptive_repair_root = args.adaptive_repair_root or evidence.get(
        "adaptive_repair_root"
    )
    certification_validation_summary = (
        args.certification_validation_summary
        or evidence.get("certification_validation_summary")
    )
    if not fixed_code_root or not timing_root:
        parser.error(
            "fixed-code and timing roots are required via CLI or report-set publication evidence"
        )
    main_df = load_main_report_results(args.main_report_set)
    if bool(certification_root) != bool(certification_validation_summary):
        parser.error(
            "--certification-root and --certification-validation-summary must be provided together"
        )
    if bool(base_rescore_root) != bool(base_rescore_result_manifest):
        parser.error(
            "--base-rescore-root and --base-rescore-result-manifest must be provided together"
        )
    if base_rescore_root:
        main_df = main_df[main_df["Method"] != "Base (Best-of-64)"].copy()
        main_df = pd.concat(
            [
                main_df,
                load_base_rescore_results(
                    Path(base_rescore_root), Path(base_rescore_result_manifest)
                ),
            ],
            ignore_index=True,
        )
    if shinka_root:
        main_df = main_df[main_df["Method"] != "ShinkaEvolve"].copy()
        main_df = pd.concat(
            [main_df, load_shinka_override_results(Path(shinka_root))],
            ignore_index=True,
        )
    fixed_df = load_fixed_code_results(Path(fixed_code_root))
    if args.frozen_hero_validation_root:
        fixed_df = fixed_df[fixed_df["Method"] != "Frozen Hero"].copy()
    result_frames = [main_df, fixed_df]
    if args.frozen_hero_validation_root:
        result_frames.append(
            load_validation_selected_frozen_hero(
                Path(args.frozen_hero_validation_root),
                protocol_path=Path(args.frozen_hero_protocol),
            )
        )
    if adaptive_repair_root:
        result_frames.append(
            load_adaptive_repair_results(
                Path(adaptive_repair_root),
                protocol_path=Path(args.adaptive_repair_protocol),
                test_root=(
                    Path(args.adaptive_repair_test_root)
                    if args.adaptive_repair_test_root
                    else None
                ),
            )
        )
    for additional_root in args.additional_method_root:
        result_frames.append(load_additional_method_results(Path(additional_root)))
    combined_df = pd.concat(result_frames, ignore_index=True)
    missing_required_methods = sorted(
        set(args.required_method) - set(combined_df["Method"])
    )
    if missing_required_methods:
        raise RuntimeError(
            f"required methods are missing: {missing_required_methods}"
        )
    benchmark_keys = None
    benchmark_manifest = None
    if args.benchmark_subset:
        benchmark_keys, benchmark_manifest = load_benchmark_subset(
            Path(args.benchmark_subset)
        )
        combined_df = restrict_to_benchmark_subset(combined_df, benchmark_keys)
    combined_df = recompute_global_vbs(combined_df)

    per_seed_summary = summarise_methods_by_seed(combined_df)
    summary = summarise_methods(combined_df)
    write_summary_bundle(summary, output_dir)
    write_per_seed_summary_bundle(per_seed_summary, output_dir)
    if certification_root:
        certifications, validation = load_certified_references(
            Path(certification_root),
            Path(certification_validation_summary),
        )
        if benchmark_keys is not None:
            certifications = certifications[
                [
                    (int(seed), str(uuid)) in benchmark_keys
                    for seed, uuid in certifications[["Seed", "uuid"]].itertuples(
                        index=False, name=None
                    )
                ]
            ].copy()
            if set(
                certifications[["Seed", "uuid"]].itertuples(index=False, name=None)
            ) != benchmark_keys:
                raise RuntimeError("certification benchmark subset key mismatch")
            validation = {
                **validation,
                "benchmark_subset": {
                    "manifest": str(args.benchmark_subset),
                    "protocol_id": benchmark_manifest.get("protocol_id"),
                    "row_count": len(benchmark_keys),
                },
            }
        certified_results = attach_certified_references(combined_df, certifications)
        write_certification_bundle(certified_results, validation, output_dir)
        write_paired_comparison_bundle(
            paired_certified_comparisons(certified_results), output_dir
        )
        if "Frozen Hero" in set(certified_results["Method"]):
            write_paired_comparison_bundle(
                paired_certified_comparisons(
                    certified_results, reference_method="Frozen Hero"
                ),
                output_dir,
                stem="paired_frozen_hero_comparisons",
            )
        reserved_references = {"Ours (Hero)", "Frozen Hero"}
        seen_custom_references: set[str] = set()
        for reference_method in args.paired_reference_method:
            if reference_method in reserved_references:
                continue
            if reference_method in seen_custom_references:
                raise RuntimeError(
                    f"duplicate paired reference method: {reference_method}"
                )
            seen_custom_references.add(reference_method)
            write_paired_comparison_bundle(
                paired_certified_comparisons(
                    certified_results, reference_method=reference_method
                ),
                output_dir,
                stem=paired_comparison_stem(reference_method),
            )
        hero_vbs = build_hero_excluded_vbs(certified_results)
    else:
        hero_vbs = build_hero_excluded_vbs(combined_df)
    write_hero_excluded_vbs_bundle(hero_vbs, output_dir)
    if benchmark_manifest is not None:
        (output_dir / "benchmark_subset_manifest.json").write_text(
            json.dumps(benchmark_manifest, indent=2, sort_keys=True) + "\n"
        )

    timing_df, missing_runtime_methods = collect_timing_rows(
        Path(timing_root),
        args.runtime_seed,
        Path(shinka_root) if shinka_root else None,
    )
    all_runtime_df = collect_all_timing_rows(
        Path(timing_root), Path(shinka_root) if shinka_root else None
    )
    runtime_df = augment_runtime_with_classical_baselines(
        timing_df, combined_df, args.runtime_seed
    )
    write_runtime_bundle(runtime_df, output_dir)
    write_runtime_audit_bundle(all_runtime_df, output_dir)
    write_results_note(output_dir, summary, runtime_df, missing_runtime_methods)

    print(f"Wrote publication bundle to {output_dir}")


if __name__ == "__main__":
    main()
