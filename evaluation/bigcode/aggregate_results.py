#!/usr/bin/env python3
"""
Aggregate BigCode evaluation results across seeds and experiments.

This script:
1. Scans evaluation/bigcode/results/ recursively for metrics_*.json files
2. Parses paths to identify methods (Base, Hero, ablations)
3. Aggregates across seeds (mean ± std dev) for HumanEval and MBPP
4. Generates LaTeX table similar to Qwen technical report format

Usage:
    python evaluation/bigcode/aggregate_results.py [--output-dir DIR]
"""

import argparse
import contextlib
import json
import re
from pathlib import Path

import pandas as pd


class DuplicateDataError(ValueError):
    """Raised when duplicate data points are detected in aggregation."""

    def __init__(self):
        super().__init__(
            "Duplicate data points detected. Aggregation would be invalid."
        )


# --- CONFIGURATION ---
# NOTE: BASE_RESULT_DIR is the default "moving" directory.
# For reproducible paper results, prefer using --results-root or --report-set.
BASE_RESULT_DIR = "evaluation/bigcode/results"
DEFAULT_OUTPUT_DIR = "evaluation/bigcode/aggregated_report"

# Method name mapping from path patterns
METHOD_PATTERNS = {
    r"/base/": "Base",
    r"grpo-config_hero": "Ours (Hero)",
    r"grpo-config_ablation_oracle": "Ours (+Oracle)",
    r"grpo-config_ablation_diversity": "Ours (+Diversity)",
    r"grpo-config_minimalist": "Ours (w/o Structure)",
    r"grpo-config_ablation_prompt": "Ours (w/o Prompt)",
    # Optional/legacy (excluded from default aggregation, but supported via --include-generalization)
    r"grpo-config_ablation_generalization": "Ours (+Generalization)",
    # Also handle paths without "grpo-" prefix
    r"/grpo$": "Ours (Hero)",  # Default grpo without config
    r"config_hero": "Ours (Hero)",
    r"config_ablation_oracle": "Ours (+Oracle)",
    r"config_ablation_diversity": "Ours (+Diversity)",
    r"config_minimalist": "Ours (w/o Structure)",
    r"config_ablation_prompt": "Ours (w/o Prompt)",
    # Optional/legacy (excluded from default aggregation, but supported via --include-generalization)
    r"config_ablation_generalization": "Ours (+Generalization)",
}

# Task names (matching the technical report)
TASKS = ["humaneval", "mbpp"]


DEFAULT_ALLOWED_METHODS = [
    "Base",
    "Ours (Hero)",
    "Ours (+Oracle)",
    "Ours (+Diversity)",
    "Ours (w/o Structure)",
    "Ours (w/o Prompt)",
]


def load_report_set(report_set_path: str) -> dict:
    """
    Load a report-set JSON manifest describing which result roots to aggregate.
    Expected format:
      {
        "name": "...",
        "sds": { "result_roots": [...] },
        "bigcode": { "result_roots": ["evaluation/bigcode/results_batches/<batch_id>", ...] }
      }
    """
    with Path(report_set_path).open() as f:
        return json.load(f)


def find_all_metrics_files(base_dir: str) -> list[str]:
    base_path = Path(base_dir)
    return sorted(str(p) for p in base_path.rglob("metrics_*.json"))


def find_all_metrics_files_from_roots(result_roots: list[str]) -> list[str]:
    all_files: list[str] = []
    for root in result_roots:
        all_files.extend(find_all_metrics_files(root))
    # De-dupe while preserving order
    seen = set()
    out = []
    for fp in all_files:
        if fp in seen:
            continue
        seen.add(fp)
        out.append(fp)
    return out


def infer_method_from_model_path(model_path: str) -> str | None:  # noqa: PLR0911
    """
    Infer method name from the model path embedded in BigCode metrics JSON.
    This is needed because some result directories are stored under:
      evaluation/bigcode/results/<model>/grpo/seed*/job-*/metrics_*.json
    and therefore do not encode the config in the filesystem path.
    """
    if not model_path:
        return None
    lowered = str(model_path).lower()
    # Base model mode is a HF identifier (e.g., "Qwen/Qwen2.5-Coder-14B-Instruct")
    if "qwen/qwen2.5-coder" in lowered:
        return "Base"
    # Checkpoint paths often include "...-config_ablation_oracle/..."
    if "config_ablation_oracle" in lowered:
        return "Ours (+Oracle)"
    if "config_ablation_diversity" in lowered:
        return "Ours (+Diversity)"
    if "config_ablation_prompt" in lowered:
        return "Ours (w/o Prompt)"
    if "config_ablation_generalization" in lowered:
        return "Ours (+Generalization)"
    if "config_minimalist" in lowered:
        return "Ours (w/o Structure)"
    if "config_hero" in lowered or "config_hero.yaml" in lowered:
        return "Ours (Hero)"
    return None


def maybe_write_experiment_metadata(  # noqa: PLR0913
    json_path: str,
    method: str | None,
    seed: int,
    model: str | None,
    job_id: int | None,
    embedded_model_path: str | None,
) -> None:
    """
    Best-effort backfill of `experiment_metadata.json` next to legacy BigCode outputs.
    This keeps future aggregations stable and removes reliance on brittle heuristics.

    Safe behavior:
    - Only writes if metadata file does not already exist
    - Never throws (aggregation should not fail because metadata couldn't be written)
    """
    try:
        json_path_obj = Path(json_path)
        metadata_path = json_path_obj.parent / "experiment_metadata.json"
        if metadata_path.exists():
            return

        # Infer a config_name from the method label (best-effort)
        config_name = None
        if method == "Ours (Hero)":
            config_name = "config_hero"
        elif method == "Ours (+Oracle)":
            config_name = "config_ablation_oracle"
        elif method == "Ours (+Diversity)":
            config_name = "config_ablation_diversity"
        elif method == "Ours (w/o Structure)":
            config_name = "config_minimalist"
        elif method == "Ours (w/o Prompt)":
            config_name = "config_ablation_prompt"
        elif method == "Ours (+Generalization)":
            config_name = "config_ablation_generalization"
        elif method == "Base":
            config_name = "base"

        metadata = {
            "model": model,
            "seed": int(seed) if seed is not None else 0,
            "job_id": str(job_id) if job_id is not None else None,
            "config_name": config_name,
            "method_name": method,
            # Helpful for auditing:
            "embedded_model_path": embedded_model_path,
        }

        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        # Never fail aggregation due to metadata backfill
        return


def parse_path_metadata(  # noqa: PLR0912, PLR0915
    json_path: str,
) -> tuple[str | None, int, str | None, int | None]:
    """
    Parse JSON file path to extract method name, seed, model, and job_id.

    First tries to read experiment_metadata.json if it exists (most reliable).
    Falls back to path parsing.

    Expected paths:
    - Base: {base_dir}/{model}/base/seed{seed}/metrics_{task}.json
    - Fine-tuned: {base_dir}/{model}/grpo/seed{seed}/job-{job_id}/metrics_{task}.json

    Returns:
        (method_name, seed, model, job_id)
    """
    # Try to read metadata file first (most reliable, like SDS eval)
    json_path_obj = Path(json_path)
    metadata_path = json_path_obj.parent / "experiment_metadata.json"

    if metadata_path.exists():
        try:
            with metadata_path.open() as f:
                metadata = json.load(f)

            method = metadata.get("method_name")
            seed = metadata.get("seed", 0)
            model = metadata.get("model")
            job_id = metadata.get("job_id")

            # Convert job_id to int if it's a string
            if job_id is not None and isinstance(job_id, str):
                with contextlib.suppress(ValueError):
                    job_id = int(job_id)

            if method:
                return (method, seed, model, job_id)
        except Exception as e:
            print(f"⚠️  Failed to read metadata from {metadata_path}: {e}")

    # Fallback to path parsing
    json_path_obj = Path(json_path)
    path_parts = list(json_path_obj.parts)

    # Extract seed (allow seed-101, seed_101, or seed101)
    seed_match = re.search(r"seed[-_]?(\d+)", json_path)
    if seed_match:
        seed = int(seed_match.group(1))
    else:
        print(f"⚠️  WARNING: Could not extract seed from {json_path}. Defaulting to 0.")
        seed = 0

    # Extract job_id (if present)
    job_match = re.search(r"job-(\d+)", json_path)
    job_id = int(job_match.group(1)) if job_match else None

    # Extract model name (usually the first directory after results/)
    model = None
    if "results" in path_parts:
        results_idx = path_parts.index("results")
        if results_idx + 1 < len(path_parts):
            model = path_parts[results_idx + 1]

    # TERTIARY METHOD: Detect method from path patterns
    method = None
    for pattern, method_name in METHOD_PATTERNS.items():
        if re.search(pattern, json_path, re.IGNORECASE):
            method = method_name
            break

    # If no pattern matched, try to infer from path
    if method is None:
        if "/base/" in json_path:
            method = "Base"
        elif "/grpo/" in json_path or "grpo-" in json_path:
            # Check if there's a config in the path
            if "config_ablation" in json_path:
                # Try to extract the specific ablation type
                if "oracle" in json_path:
                    method = "Ours (+Oracle)"
                elif "diversity" in json_path:
                    method = "Ours (+Diversity)"
                elif "minimalist" in json_path:
                    method = "Ours (w/o Structure)"
                elif "prompt" in json_path:
                    method = "Ours (w/o Prompt)"
                else:
                    method = "Ours (Hero)"  # Fallback
            elif "config_hero" in json_path or "hero" in json_path.lower():
                method = "Ours (Hero)"
            else:
                method = "Ours (Hero)"  # Default assumption for grpo without config

    return (method, seed, model, job_id)


def load_all_data(files: list[str]) -> pd.DataFrame:  # noqa: PLR0912, PLR0915
    """
    Load and merge all metrics JSON files.

    Returns DataFrame with columns:
    - Method: Method name (e.g., "Base", "Hero", "Ablation: Oracle")
    - Seed: Seed value
    - Task: Task name (humaneval, mbpp)
    - Pass@1: Pass@1 score (0-1)
    """
    rows = []

    print(f"📥 Loading {len(files)} metric files...")

    for json_path in files:
        try:
            with Path(json_path).open() as f:
                data = json.load(f)

            method, seed, model, job_id = parse_path_metadata(json_path)

            # Extract embedded model path for metadata backfill
            embedded_model_path = None
            if isinstance(data, dict):
                embedded_model_path = (data.get("config") or {}).get("model")

            # Only infer from checkpoint path if method wasn't found in metadata
            # (Metadata is more reliable, especially for migrated jobs)
            if method is None:
                inferred = (
                    infer_method_from_model_path(embedded_model_path)
                    if embedded_model_path
                    else None
                )
                if inferred:
                    method = inferred

            # Backfill experiment_metadata.json for legacy runs (best-effort).
            maybe_write_experiment_metadata(
                json_path=json_path,
                method=method,
                seed=seed,
                model=model,
                job_id=job_id,
                embedded_model_path=embedded_model_path,
            )

            if method is None:
                print(f"⚠️  Skipping {json_path}: Could not identify method")
                continue

            # Extract task name from filename
            task_match = re.search(r"metrics_(\w+)\.json", json_path)
            if not task_match:
                print(f"⚠️  Skipping {json_path}: Could not extract task name")
                continue

            task = task_match.group(1)

            # Extract Pass@1 score from JSON
            # Structure: {task: {"pass@1": value, ...}, "config": {...}}
            if task not in data:
                print(f"⚠️  Skipping {json_path}: Task '{task}' not found in JSON")
                continue

            task_data = data[task]
            if not isinstance(task_data, dict):
                print(f"⚠️  Skipping {json_path}: Task data is not a dict")
                continue

            pass_at_1 = task_data.get("pass@1")
            if pass_at_1 is None:
                print(f"⚠️  Skipping {json_path}: pass@1 not found")
                continue

            # Validate pass@1 value is in expected range (0.0 to 1.0, not percentage)
            if pass_at_1 > 1.0:
                print(
                    f"⚠️  WARNING: {json_path} has pass@1 = {pass_at_1} (looks like percentage, not fraction)"
                )
                print(
                    "   Expected range: 0.0 to 1.0. If this is a percentage, divide by 100."
                )
            elif pass_at_1 < 0.0:
                print(
                    f"⚠️  WARNING: {json_path} has pass@1 = {pass_at_1} (negative value, unexpected)"
                )

            rows.append(
                {
                    "Method": method,
                    "Seed": seed,
                    "Task": task,
                    "Pass@1": pass_at_1,
                    "Model": model,
                    "JobID": job_id,
                    "File": json_path,  # Keep file path for debugging
                }
            )

        except Exception as e:
            print(f"❌ Error reading {json_path}: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Check for and handle duplicates (shouldn't happen, but be safe)
    duplicates = df.groupby(["Method", "Seed", "Task"]).size()
    if (duplicates > 1).any():
        print(
            "❌ CRITICAL ERROR: Found duplicate results for some (Method, Seed, Task) combinations:"
        )
        for (method, seed, task), count in duplicates[duplicates > 1].items():
            print(f"   {method}, seed {seed}, {task}: {count} results")
            # Show which files are duplicates
            dup_rows = df[
                (df["Method"] == method) & (df["Seed"] == seed) & (df["Task"] == task)
            ]
            if "File" in dup_rows.columns:
                for _idx, row in dup_rows.iterrows():
                    print(f"      - {row.get('File', 'unknown')}")
        print("\n❌ This indicates a bug in seed extraction or path parsing.")
        print(
            "   Check if seeds are being extracted correctly (should be 101, 202, 303, not 0)."
        )
        print("   Aggregation would be invalid. Please fix the path parsing logic.")
        raise DuplicateDataError()

    print(f"✅ Loaded {len(df)} results across {df['Method'].nunique()} methods")
    return df


def generate_latex_table(df: pd.DataFrame, output_path: str):  # noqa: PLR0912, PLR0915
    """
    Generate LaTeX table with Mean ± Std across seeds.

    Format similar to Qwen technical report:
    - Rows: Methods (Base, Hero, Ablations)
    - Columns: HumanEval Pass@1, MBPP Pass@1

    Format: $88.4_{\\pm 0.1}$ or just $88.4$ if std is negligible
    """
    # First: Get one value per seed per method per task
    # Ensure we have exactly one value per (Method, Seed, Task) combination
    per_seed = df.groupby(["Method", "Seed", "Task"])["Pass@1"].first().reset_index()

    # Second: Mean/Std across seeds per method per task
    # Use ddof=0 for population std (matching pandas default) to ensure consistency
    agg = (
        per_seed.groupby(["Method", "Task"])["Pass@1"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Pivot to have tasks as columns
    pivot_mean = (
        agg.pivot_table(index="Method", columns="Task", values="mean") * 100
    )  # Convert to %
    pivot_std = agg.pivot_table(index="Method", columns="Task", values="std") * 100

    # Fill missing values with 0
    pivot_mean = pivot_mean.fillna(0.0)
    pivot_std = pivot_std.fillna(0.0)

    def _is_close(a: float, b: float, tol: float = 1e-9) -> bool:
        return bool(pd.notna(a) and pd.notna(b) and abs(float(a) - float(b)) <= tol)

    best_by_task: dict[str, float] = {}
    for task in TASKS:
        if task in pivot_mean.columns:
            best_by_task[task] = float(pivot_mean[task].max())

    # Format for LaTeX
    rows = []
    for method in pivot_mean.index:
        row = {"Method": method}

        for task in TASKS:
            if task not in pivot_mean.columns:
                row[task] = "N/A"
                continue

            mean_val = pivot_mean.loc[method, task]
            std_val = pivot_std.loc[method, task] if task in pivot_std.columns else 0.0

            # Format as $88.4_{\pm 0.1}$ or just $88.4$ if std is negligible
            _STD_THRESHOLD = 0.01  # noqa: N806
            if std_val > _STD_THRESHOLD:
                inner = f"{mean_val:.1f}_{{\\pm {std_val:.1f}}}"
            else:
                inner = f"{mean_val:.1f}"

            is_best = _is_close(float(mean_val), best_by_task.get(task, float("nan")))
            formatted = f"$\\\\mathbf{{{inner}}}$" if is_best else f"${inner}$"

            # Use readable column name
            task_label = (
                (r"HumanEval (\%) $\uparrow$")
                if task == "humaneval"
                else (r"MBPP (\%) $\uparrow$")
            )
            row[task_label] = formatted

        rows.append(row)

    tex_df = pd.DataFrame(rows)

    # Sort by method (Base first, then Hero, then ablations) - matching SDS table style
    method_order = [
        "Base",
        "Ours (Hero)",
        "Ours (+Oracle)",
        "Ours (+Diversity)",
        "Ours (w/o Structure)",
        "Ours (w/o Prompt)",
    ]
    tex_df["sort_key"] = tex_df["Method"].apply(
        lambda x: method_order.index(x) if x in method_order else 999
    )
    tex_df = tex_df.sort_values("sort_key").drop("sort_key", axis=1)

    # Reorder columns: Method, HumanEval, MBPP
    column_order = ["Method"] + [
        col
        for col in [r"HumanEval (\%) $\uparrow$", r"MBPP (\%) $\uparrow$"]
        if col in tex_df.columns
    ]
    tex_df = tex_df[column_order]

    # Generate LaTeX with proper formatting (matching SDS table style)
    num_cols = len(tex_df.columns) - 1  # -1 for Method column
    col_format = f"l{'c' * num_cols}"
    latex_str = tex_df.to_latex(
        index=False, escape=False, column_format=col_format, float_format="%.1f"
    )

    # Add toprule and bottomrule for better formatting
    latex_str = latex_str.replace("\\begin{tabular}", "\\begin{tabular}")
    if "\\toprule" not in latex_str:
        # Insert toprule after \begin{tabular} line
        lines = latex_str.split("\n")
        for i, line in enumerate(lines):
            if "\\begin{tabular}" in line:
                lines.insert(i + 1, "\\toprule")
                break
        latex_str = "\n".join(lines)

    if "\\bottomrule" not in latex_str:
        # Replace last \\hline with \\bottomrule
        latex_str = latex_str.replace(
            "\\hline\n\\end{tabular}", "\\bottomrule\n\\end{tabular}"
        )
        if "\\hline" in latex_str and "\\bottomrule" not in latex_str:
            lines = latex_str.split("\n")
            for i in range(len(lines) - 1, -1, -1):
                if (
                    "\\hline" in lines[i] and "\\end{tabular}" in lines[i + 1]
                    if i + 1 < len(lines)
                    else False
                ):
                    lines[i] = "\\bottomrule"
                    break
            latex_str = "\n".join(lines)

    with Path(output_path).open("w") as f:
        f.write(latex_str)

    print(f"✅ LaTeX table saved to {output_path}")


def print_summary_table(df: pd.DataFrame):
    """Print a human-readable summary table to console."""
    # Aggregate across seeds
    per_seed = df.groupby(["Method", "Seed", "Task"])["Pass@1"].mean().reset_index()
    agg = (
        per_seed.groupby(["Method", "Task"])["Pass@1"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Pivot
    pivot_mean = agg.pivot_table(index="Method", columns="Task", values="mean") * 100
    pivot_std = agg.pivot_table(index="Method", columns="Task", values="std") * 100

    # Fill missing
    pivot_mean = pivot_mean.fillna(0.0)
    pivot_std = pivot_std.fillna(0.0)

    print("\n" + "=" * 60)
    print("BigCode Evaluation Results (Mean ± Std across seeds)")
    print("=" * 60)
    print(f"{'Method':<30} {'HumanEval':<15} {'MBPP':<15}")
    print("-" * 60)

    # Sort methods - matching SDS table style
    method_order = [
        "Base",
        "Ours (Hero)",
        "Ours (+Oracle)",
        "Ours (+Diversity)",
        "Ours (w/o Structure)",
        "Ours (w/o Prompt)",
    ]

    for method in method_order:
        if method not in pivot_mean.index:
            continue

        he_mean = (
            pivot_mean.loc[method, "humaneval"]
            if "humaneval" in pivot_mean.columns
            else 0.0
        )
        he_std = (
            pivot_std.loc[method, "humaneval"]
            if "humaneval" in pivot_std.columns
            else 0.0
        )
        mbpp_mean = (
            pivot_mean.loc[method, "mbpp"] if "mbpp" in pivot_mean.columns else 0.0
        )
        mbpp_std = pivot_std.loc[method, "mbpp"] if "mbpp" in pivot_std.columns else 0.0

        _STD_THRESHOLD = 0.01  # noqa: N806
        he_str = (
            f"{he_mean:.1f} ± {he_std:.1f}"
            if he_std > _STD_THRESHOLD
            else f"{he_mean:.1f}"
        )
        mbpp_str = (
            f"{mbpp_mean:.1f} ± {mbpp_std:.1f}"
            if mbpp_std > _STD_THRESHOLD
            else f"{mbpp_mean:.1f}"
        )

        print(f"{method:<30} {he_str:<15} {mbpp_str:<15}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate BigCode evaluation results across seeds and experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate all results
  python evaluation/bigcode/aggregate_results.py
  
  # Custom output directory
  python evaluation/bigcode/aggregate_results.py --output-dir my_aggregated_results
        """,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for aggregated results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Root directory to scan for BigCode results (overrides default evaluation/bigcode/results)",
    )
    parser.add_argument(
        "--report-set",
        type=str,
        default=None,
        help="Path to a report-set JSON manifest. If set, aggregates across all BigCode roots listed there.",
    )
    parser.add_argument(
        "--include-generalization",
        action="store_true",
        help="Include the (+Generalization) ablation in aggregation (default: excluded).",
    )
    args = parser.parse_args()

    # Find all metrics files (default: moving dir; optionally override)
    if args.report_set:
        rs = load_report_set(args.report_set)
        roots = rs.get("bigcode", {}).get("result_roots", [])
        if not roots:
            print(f"❌ Report set has no BigCode result_roots: {args.report_set}")
            return

        # When using --report-set, automatically use aggregated_report_batches/{report_set_name}/
        report_set_name = rs.get("name", Path(args.report_set).stem)
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            # Only override if user didn't specify a custom output dir
            args.output_dir = (
                f"evaluation/bigcode/aggregated_report_batches/{report_set_name}"
            )

        all_files = find_all_metrics_files_from_roots(roots)
        print(f"📌 Report set: {report_set_name}")
        print(f"📌 BigCode roots: {roots}")
    else:
        root = args.results_root or BASE_RESULT_DIR
        all_files = find_all_metrics_files(root)
        if args.results_root:
            print(f"📌 Results root override: {args.results_root}")

    # Create output directory (after potentially overriding it for report-set)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")

    print(f"📊 Found {len(all_files)} metrics JSON files")

    if not all_files:
        print("❌ No metrics files found!")
        return

    # Load and merge data
    df = load_all_data(all_files)

    if df.empty:
        print("❌ No data loaded!")
        return

    # Filter to default allowlist (exclude generalization unless explicitly enabled)
    allowed = set(DEFAULT_ALLOWED_METHODS)
    if args.include_generalization:
        allowed.add("Ours (+Generalization)")
    df = df[df["Method"].isin(allowed)].copy()

    # Print summary
    print("\n📊 Data Summary:")
    print(f"   Methods: {', '.join(sorted(df['Method'].unique()))}")
    print(f"   Seeds: {sorted(df['Seed'].unique())}")
    print(f"   Tasks: {sorted(df['Task'].unique())}")
    print(f"   Total results: {len(df)}")

    # Print summary table
    print_summary_table(df)

    # Generate LaTeX table
    table_path = Path(args.output_dir) / "bigcode_results_table.tex"
    print("\n📝 Generating LaTeX table...")
    generate_latex_table(df, str(table_path))

    print(f"\n✅ Done! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
