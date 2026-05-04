#!/usr/bin/env python3
"""Aggregate fixed-code appendix evidence against canonical SDS baselines."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    from evaluation.sds.aggregate_plots import (
        find_all_metrics_files,
        find_all_metrics_files_from_roots,
        load_all_data,
        load_report_set,
        select_latest_jobs,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from evaluation.sds.aggregate_plots import (
        find_all_metrics_files,
        find_all_metrics_files_from_roots,
        load_all_data,
        load_report_set,
        select_latest_jobs,
    )


MAIN_METHODS = [
    "Ours (Hero)",
    "Frozen Hero",
    "Hand-written SA",
    "Base (Best-of-64)",
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


def infer_runtime_method_name(data: dict) -> str | None:
    """Normalize timing-summary method names onto the appendix labels."""
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
    (output_dir / "baseline_appendix_table.tex").write_text(latex)


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
    """Create a short reviewer-facing note."""
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
- Frozen Hero selection rule: filter canonical Hero `metrics_final.csv` rows to `feasible == True` and `error_type == "none"`, sort by `uuid`, and take the first row.
- Failure semantics: a non-pass means the method did not return a feasible solution within the fixed evaluation budget. It should not be described as a reviewer-confusing "violation" unless the returned program actually violated constraints.
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


def load_main_report_results(main_report_set: str) -> pd.DataFrame:
    """Load canonical comparison methods from the main report set."""
    report_set = load_report_set(main_report_set)
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


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate fixed-code SDS appendix results against canonical baselines."
    )
    parser.add_argument("--main-report-set", required=True, help="Canonical main SDS report set JSON.")
    parser.add_argument("--fixed-code-root", required=True, help="Root containing fixed-code results for the appendix batch.")
    parser.add_argument("--timing-root", required=True, help="Batch root containing timing_summary.json files.")
    parser.add_argument("--output-dir", required=True, help="Directory for appendix aggregation outputs.")
    parser.add_argument(
        "--shinka-root",
        default=None,
        help="Optional root containing refreshed ShinkaEvolve metrics_final.csv files to override the canonical Shinka line.",
    )
    parser.add_argument(
        "--runtime-seed",
        type=int,
        default=101,
        help="Seed used for representative runtime summaries (default: 101).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    main_df = load_main_report_results(args.main_report_set)
    if args.shinka_root:
        main_df = main_df[main_df["Method"] != "ShinkaEvolve"].copy()
        main_df = pd.concat(
            [main_df, load_shinka_override_results(Path(args.shinka_root))],
            ignore_index=True,
        )
    fixed_df = load_fixed_code_results(Path(args.fixed_code_root))
    combined_df = pd.concat([main_df, fixed_df], ignore_index=True)
    combined_df = recompute_global_vbs(combined_df)

    per_seed_summary = summarise_methods_by_seed(combined_df)
    summary = summarise_methods(combined_df)
    write_summary_bundle(summary, output_dir)
    write_per_seed_summary_bundle(per_seed_summary, output_dir)

    timing_df, missing_runtime_methods = collect_timing_rows(
        Path(args.timing_root),
        args.runtime_seed,
        Path(args.shinka_root) if args.shinka_root else None,
    )
    all_runtime_df = collect_all_timing_rows(
        Path(args.timing_root), Path(args.shinka_root) if args.shinka_root else None
    )
    runtime_df = augment_runtime_with_classical_baselines(
        timing_df, combined_df, args.runtime_seed
    )
    write_runtime_bundle(runtime_df, output_dir)
    write_runtime_audit_bundle(all_runtime_df, output_dir)
    write_results_note(output_dir, summary, runtime_df, missing_runtime_methods)

    print(f"✅ Wrote appendix bundle to {output_dir}")


if __name__ == "__main__":
    main()
