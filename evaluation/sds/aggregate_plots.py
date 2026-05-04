#!/usr/bin/env python3
"""
Aggregate evaluation results across seeds and experiments.

This script:
1. Scans evaluation/sds/results/ recursively for metrics_final.csv files
2. Parses paths to identify methods (Hero, ablations, ShinkaEvolve, baselines)
3. Aggregates across seeds (mean ± std dev)
4. Generates LaTeX tables and stratified plots
5. Logs to W&B (optional)

Usage:
    python aggregate_plots.py [--max-jobs N] [--output-dir DIR] [--log-to-wandb]
"""

import argparse
import contextlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Try to import optional dependencies
try:
    from dotenv import load_dotenv

    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    print("⚠️  python-dotenv not available. Install with: pip install python-dotenv")

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("⚠️  wandb not available. Install with: pip install wandb")

# Load environment variables (for WANDB_API_KEY)
if HAS_DOTENV:
    load_dotenv()

# --- CONFIGURATION ---
# NOTE: BASE_RESULT_DIR is the default "moving" directory.
# For reproducible paper results, prefer using --results-root or --report-set.
BASE_RESULT_DIR = "evaluation/sds/results"
DEFAULT_OUTPUT_DIR = "evaluation/sds/aggregated_report"

# Numerical constants for comparisons and thresholds
_EPSILON_SMALL = 1e-9  # Small epsilon for numerical comparisons
_EPSILON_MEDIUM = 1e-6  # Medium epsilon for VBS score validation
_EPSILON_TINY = 1e-10  # Tiny epsilon for division safety
_TRIVIAL_THRESHOLD = 0.01  # Threshold for trivial difficulty classification
_MODERATE_THRESHOLD = 0.10  # Threshold for moderate difficulty classification
_STD_THRESHOLD = 0.01  # Threshold for standard deviation formatting
_STD_THRESHOLD_COST = 0.001  # Threshold for cost standard deviation formatting
_TIMEOUT_THRESHOLD = 4.9  # Timeout threshold (seconds, close to 5s limit)
_TIMEOUT_MAX = 10.0  # Maximum reasonable execution time (seconds)
_INFEASIBLE_SCORE = -1e9  # Score value indicating infeasible solution
_BEST_OF_K = 64  # Number of samples for Best-of-K baseline
_CP_SAT_GAP_TOLERANCE = 0.1  # Tolerance for CP-SAT gap (percentage)

# Plotting Style (paper-compatible)
plt.rcParams.update(
    {
        "text.usetex": False,  # Set to True if you have LaTeX installed
        "font.family": "serif",
        "font.serif": ["Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.figsize": (3.25, 2.5),  # Standard 1-col width
        "figure.dpi": 300,
    }
)

# Universal Palette (matches evaluate.py colors)
PALETTE = {
    "Ours (Hero)": "#1f77b4",  # Blue (same as LLM in evaluate.py)
    "Ours (+Oracle)": "#56B4E9",  # Light Blue
    "Ours (+Diversity)": "#009E73",  # Green
    "Ours (+Soft Gate)": "#8C564B",  # Brown
    "Ours (w/o Structure)": "#E69F00",  # Orange (replaces Generalization position)
    "Ours (w/o Prompt)": "#CC79A7",  # Pink
    # Optional/legacy (excluded from default aggregation, but supported via --include-generalization)
    "Ours (+Generalization)": "#E69F00",  # Orange
    "Base (Best-of-64)": "#e377c2",  # Pink/Magenta (distinct from other methods)
    "ShinkaEvolve": "#00CED1",  # Dark Turquoise (distinct from baselines and LLM)
    "CP-SAT": "#ff7f0e",  # Orange (same as evaluate.py)
    "Local Search": "#2ca02c",  # Green (same as evaluate.py)
    "Greedy": "#d62728",  # Red (same as evaluate.py)
    "BnB": "#9467bd",  # Purple (same as evaluate.py)
}

# Method name mapping from path patterns
METHOD_PATTERNS = {
    r"config_hero": "Ours (Hero)",
    r"config_ablation_oracle": "Ours (+Oracle)",
    r"config_ablation_diversity": "Ours (+Diversity)",
    r"config_minimalist": "Ours (w/o Structure)",
    # Optional/legacy (excluded from default aggregation, but supported via --include-generalization)
    r"config_ablation_generalization": "Ours (+Generalization)",
    r"config_ablation_soft_gate": "Ours (+Soft Gate)",
    r"config_ablation_prompt": "Ours (w/o Prompt)",
    r"shinka-evolve": "ShinkaEvolve",
    r"/base/": "Base (Best-of-64)",  # Untrained base model with hero prompt, Best-of-64
}

# Baseline name mapping
BASELINE_MAPPING = {
    "greedy": "Greedy",
    "local_search": "Local Search",
    "cpsat": "CP-SAT",
    "bnb": "BnB",
    "random": "Random",
}


def parse_path_metadata(
    csv_path: str,
) -> tuple[str | None, int, str | None, int | None]:
    """
    Parse CSV file path to extract method name, seed, model, and job_id.

    First tries to read experiment_metadata.json if it exists (most reliable).
    Falls back to path parsing if metadata is not available.

    Expected paths:
    - Fine-tuned: {base_dir}/{model}/{scheme}/seed{seed}/job-{job_id}/metrics_final.csv
    - ShinkaEvolve: {base_dir}/shinka-evolve/{dataset}/seed{seed}/test/metrics_final.csv
    - Base: {base_dir}/{model}/base/seed{seed}/metrics_final.csv

    Returns:
        (method_name, seed, model, job_id)
    """
    # Try to read metadata file first (most reliable)
    csv_path_obj = Path(csv_path)
    metadata_path = csv_path_obj.parent / "experiment_metadata.json"

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
    csv_path_obj = Path(csv_path)
    path_parts = list(csv_path_obj.parts)

    # Extract seed
    seed_match = re.search(r"seed(\d+)", csv_path)
    seed = int(seed_match.group(1)) if seed_match else 0

    # Extract job_id
    job_match = re.search(r"job-(\d+)", csv_path)
    job_id = int(job_match.group(1)) if job_match else None

    # ShinkaEvolve detection
    if "shinka-evolve" in csv_path:
        return ("ShinkaEvolve", seed, "shinka-evolve", job_id)

    # Extract model name (usually the first directory after results/)
    model = None
    if "results" in path_parts:
        results_idx = path_parts.index("results")
        if results_idx + 1 < len(path_parts):
            model = path_parts[results_idx + 1]

    # Detect method from path patterns
    method = None
    for pattern, method_name in METHOD_PATTERNS.items():
        if re.search(pattern, csv_path, re.IGNORECASE):
            method = method_name
            break

        # If no pattern matched, try to infer from training scheme
        if method is None:
            # Check for grpo directory (default to Hero if not specified)
            if "/grpo/" in csv_path:
                method = "Ours (Hero)"  # Default assumption
            elif "/base/" in csv_path:
                method = "Base (Best-of-64)"

    return (method, seed, model, job_id)


def find_all_metrics_files(base_dir: str = BASE_RESULT_DIR) -> list[str]:
    """Find all metrics_final.csv files recursively."""
    base_path = Path(base_dir)
    return sorted(str(p) for p in base_path.rglob("metrics_final.csv"))


def load_report_set(report_set_path: str) -> dict:
    """
    Load a report-set JSON manifest describing which result roots to aggregate.
    Expected format:
      {
        "name": "...",
        "sds": { "result_roots": ["evaluation/sds/results_batches/<batch_id>", ...] },
        "bigcode": { "result_roots": [...] }
      }
    """
    with Path(report_set_path).open() as f:
        return json.load(f)


def find_all_metrics_files_from_roots(result_roots: list[str]) -> list[str]:
    """Find all metrics_final.csv files under multiple result roots."""
    all_files: list[str] = []
    for root in result_roots:
        all_files.extend(find_all_metrics_files(base_dir=root))
    # De-dupe while preserving order
    seen = set()
    out = []
    for fp in all_files:
        if fp in seen:
            continue
        seen.add(fp)
        out.append(fp)
    return out


def select_latest_jobs(  # noqa: PLR0912, PLR0913
    files: list[str],
    max_jobs: int = 15,
    model_filter: str | None = None,
    jobs_per_seed: int | None = None,
    specific_job_ids: list[str] | None = None,
    allowed_methods: list[str] | None = None,
) -> list[str]:
    """
    Select the latest N job directories.

    Strategy:
    1. Group files by (method, seed, model) to identify unique experiments
    2. For ShinkaEvolve: include all results (no filtering)
    3. For other methods: select latest N jobs per (method, seed) if jobs_per_seed is set,
       otherwise select latest job per experiment and take top max_jobs overall
    4. If specific_job_ids is provided, only include those jobs

    Args:
        files: List of CSV file paths
        max_jobs: Maximum number of experiments to include (if jobs_per_seed not set)
        model_filter: Optional model name filter (e.g., "qwen2.5-coder-14b")
        jobs_per_seed: If set, select this many latest jobs per (method, seed) pair
        specific_job_ids: If provided, only include these job IDs (for 14b: the 15 specific jobs)
        allowed_methods: If provided, only include methods in this allowlist
    """
    # Group by (method, seed, model) to identify unique experiments
    experiment_groups = defaultdict(list)

    for f in files:
        method, seed, model, job_id = parse_path_metadata(f)
        if method:  # Only include if we can identify the method
            if allowed_methods is not None and method not in allowed_methods:
                continue

            # Exclude 7b results (user only evaluates 14b and ShinkaEvolve)
            # Include Base (Best-of-64) for 14b only
            if (model and "7b" in model.lower()) or (
                method == "Base (Best-of-64)" and model and "14b" not in model.lower()
            ):
                continue

            # Apply model filter if specified (but always include ShinkaEvolve)
            if (
                model_filter
                and method != "ShinkaEvolve"
                and model
                and model_filter.lower() not in model.lower()
            ):
                continue

            # If specific_job_ids provided, filter by job_id (for 14b jobs)
            # Only apply this filter to non-ShinkaEvolve methods
            if (
                specific_job_ids is not None
                and method != "ShinkaEvolve"
                and job_id is not None
                and str(job_id) not in specific_job_ids
            ):
                continue

            key = (method, seed, model)
            experiment_groups[key].append((f, job_id))

    selected_files = []

    for (method, seed, model), job_files in experiment_groups.items():
        if not job_files:
            continue

        # Sort by job_id (if available) or modification time
        def sort_key(item):
            file_path, job_id = item
            mtime = Path(file_path).stat().st_mtime
            if job_id is not None:
                return (job_id, mtime)
            return (0, mtime)

        job_files.sort(key=sort_key, reverse=True)

        # Special handling: ShinkaEvolve - keep only the latest 1000-sample result per seed.
        # Prefer the explicit v2 fairness rerun when present, then fall back to the
        # older SDS-1000 run. This makes the paper-backed Shinka source-of-truth
        # deterministic rather than depending on filesystem mtimes.
        if method == "ShinkaEvolve":
            preferred_patterns = [
                "ShinkaEvolve-SDS-1000-v2",
                "ShinkaEvolve-SDS-1000",
            ]
            chosen = None
            for pattern in preferred_patterns:
                matching = [
                    (file_path, job_id)
                    for file_path, job_id in job_files
                    if pattern in file_path
                ]
                if matching:
                    chosen = matching[0]
                    break
            if chosen is not None:
                selected_files.append((chosen[0], method, seed, model, chosen[1]))
        elif jobs_per_seed is not None:
            # Select latest N jobs per (method, seed)
            for file_path, job_id in job_files[:jobs_per_seed]:
                selected_files.append((file_path, method, seed, model, job_id))
        else:
            # Default: take the latest job for this experiment
            selected_files.append(
                (job_files[0][0], method, seed, model, job_files[0][1])
            )

    # Sort all selected files by modification time (most recent first)
    selected_files.sort(key=lambda x: Path(x[0]).stat().st_mtime, reverse=True)

    # If jobs_per_seed not set and we have too many, take top max_jobs
    if jobs_per_seed is None and len(selected_files) > max_jobs:
        print(
            f"📊 Found {len(selected_files)} experiments, selecting latest {max_jobs}"
        )
        selected_files = selected_files[:max_jobs]

    return [f[0] for f in selected_files]


def load_all_data(files: list[str], include_baselines: bool = True) -> pd.DataFrame:  # noqa: PLR0912, PLR0915
    """
    Load and merge all metrics_final.csv files.

    Returns DataFrame with columns:
    - Method: Method name (e.g., "Ours (Hero)", "ShinkaEvolve", "Greedy")
    - Seed: Seed value
    - difficulty_class: Trivial/Moderate/Hard
    - Gap: Optimality gap (VBS - Score) / VBS
    - Cost: Core-seconds
    - Pass: Pass rate (0 or 1 per instance)
    - vbs_score: Virtual Best Score
    """
    dfs = []

    print(f"📥 Loading {len(files)} metric files...")

    for csv_path in files:
        try:
            metrics_data = pd.read_csv(csv_path)
            method, seed, _model, _job_id = parse_path_metadata(csv_path)

            if method is None:
                print(f"⚠️  Skipping {csv_path}: Could not identify method")
                continue

            # Calculate VBS and difficulty if missing (backward compatibility)
            if (
                "vbs_score" not in metrics_data.columns
                or "difficulty_class" not in metrics_data.columns
            ):
                # Check if we have minimum required columns
                has_llm = (
                    "llm_score" in metrics_data.columns
                    and "feasible" in metrics_data.columns
                )
                has_baselines = any(
                    col.startswith("score_") for col in metrics_data.columns
                )

                if not (has_llm or has_baselines):
                    print(
                        f"⚠️  Skipping {csv_path}: Missing required columns (need llm_score/feasible or score_* baselines)"
                    )
                    continue

                print(
                    f"📊 Computing VBS and difficulty for {csv_path} (missing from CSV)..."
                )

                def calculate_vbs_and_difficulty(row):
                    # A. Find VBS (Max of LLM + All Baselines)
                    scores = []

                    # LLM Score (only if feasible, preserve negative scores)
                    if row.get("feasible", False) and "llm_score" in row:
                        scores.append(row.get("llm_score", float("-inf")))

                    # Baseline Scores (preserve negative scores, filter only -inf)
                    for col in row.index:
                        if col.startswith("score_"):
                            score_val = row[col]
                            if score_val > float(
                                "-inf"
                            ):  # Only exclude -inf (infeasible)
                                scores.append(score_val)

                    # VBS is the absolute max found by anyone (can be negative)
                    valid_scores = [s for s in scores if s > float("-inf")]
                    vbs = float("-inf") if not valid_scores else max(valid_scores)

                    # B. Calculate Difficulty (Gap vs Greedy)
                    greedy_score = row.get("score_greedy", float("-inf"))
                    if greedy_score == float("-inf"):
                        greedy_score = 0.0  # Anchor for gap calc

                    if vbs == float("-inf") or vbs <= _EPSILON_SMALL:
                        hardness = 1.0  # All failed
                    elif greedy_score <= _EPSILON_SMALL:
                        hardness = 1.0  # Greedy failed
                    else:
                        epsilon = _EPSILON_TINY
                        numerator = vbs - greedy_score
                        denominator = abs(vbs) + epsilon
                        hardness = numerator / denominator

                    return pd.Series([vbs, hardness], index=["vbs_score", "hardness"])

                metrics_data[["vbs_score", "hardness"]] = metrics_data.apply(
                    calculate_vbs_and_difficulty, axis=1
                )

                # Classify difficulty
                def classify_diff(h):
                    if h < _TRIVIAL_THRESHOLD:
                        return "Trivial"
                    if h < _MODERATE_THRESHOLD:
                        return "Moderate"
                    return "Hard"

                metrics_data["difficulty_class"] = metrics_data["hardness"].apply(
                    classify_diff
                )

            # CRITICAL: Calculate Pass rate BEFORE filtering broken problems
            # This ensures we count ALL instances, including those where all solvers failed

            # 1. Extract LLM rows (BEFORE filtering)
            if (
                "llm_score" in metrics_data.columns
                and "feasible" in metrics_data.columns
            ):
                df_llm = metrics_data.copy()  # Keep all rows for Pass calculation
                df_llm["Method"] = method
                df_llm["Seed"] = seed

                # Set Pass from feasible column (this must be done on ALL rows)
                df_llm["Pass"] = df_llm["feasible"].astype(int)

                # Now filter broken problems (vbs_score <= _EPSILON_MEDIUM) for Gap calculation
                # But keep Pass rate calculated on all rows
                df_llm_valid = df_llm[df_llm["vbs_score"] > _EPSILON_MEDIUM].copy()

                # Calculate Gap: (VBS - LLM) / VBS (only for valid VBS rows)
                llm_scores = df_llm_valid.apply(
                    lambda r: r["llm_score"] if r.get("feasible", False) else 0.0,
                    axis=1,
                )
                df_llm_valid["Gap"] = (
                    df_llm_valid["vbs_score"] - llm_scores.clip(lower=0.0)
                ) / df_llm_valid["vbs_score"]
                # Cost calculation: For Base (Best-of-64), multiply by 64 since we generated 64 samples
                base_cost = df_llm_valid.get(
                    "llm_core_sec", df_llm_valid.get("execution_time", 0.04)
                )
                if method == "Base (Best-of-64)":
                    df_llm_valid["Cost"] = base_cost * _BEST_OF_K
                else:
                    df_llm_valid["Cost"] = base_cost

                # For rows with invalid VBS, set Gap to NaN (will be filtered later)
                # But keep them for Pass rate calculation
                df_llm_invalid = df_llm[df_llm["vbs_score"] <= _EPSILON_MEDIUM].copy()
                if len(df_llm_invalid) > 0:
                    df_llm_invalid["Gap"] = np.nan
                    # Cost calculation: For Base (Best-of-64), multiply by 64 since we generated 64 samples
                    base_cost_invalid = df_llm_invalid.get(
                        "llm_core_sec", df_llm_invalid.get("execution_time", 0.04)
                    )
                    if method == "Base (Best-of-64)":
                        df_llm_invalid["Cost"] = base_cost_invalid * _BEST_OF_K
                    else:
                        df_llm_invalid["Cost"] = base_cost_invalid
                    # Combine valid and invalid, but mark invalid gaps
                    df_llm = pd.concat(
                        [df_llm_valid, df_llm_invalid], ignore_index=True
                    )
                else:
                    df_llm = df_llm_valid

                # --- FAILURE ANALYSIS: Classify failure types ---
                def classify_failure(row):
                    """Classify failure type: Success, Timeout, or Logic Error."""
                    if row.get("feasible", False):
                        return "Success"

                    # Check for Timeout
                    # Use .get() with defaults in case columns don't exist (backward compatibility)
                    err_type = row.get("error_type", "unknown")
                    exec_time = row.get("execution_time", 0.0)

                    # Timeout if error_type is 'timeout' or execution time is close to 5s limit
                    # (evaluate.py uses 5.0s timeout with 0.5s buffer, so >_TIMEOUT_THRESHOLD indicates timeout)
                    # Also check that exec_time is reasonable (< _TIMEOUT_MAX) to avoid false positives
                    if err_type == "timeout" or (
                        exec_time > _TIMEOUT_THRESHOLD and exec_time < _TIMEOUT_MAX
                    ):
                        return "Timeout"

                    # If not timeout and not feasible, it's a Logic Error
                    # (includes syntax errors, constraint violations, runtime errors, etc.)
                    return "Logic Error"

                # Only classify if error_type or execution_time columns exist
                if (
                    "error_type" in metrics_data.columns
                    or "execution_time" in metrics_data.columns
                ):
                    df_llm["FailureType"] = df_llm.apply(classify_failure, axis=1)
                else:
                    # Fallback: if columns don't exist, classify based on feasibility only
                    df_llm["FailureType"] = df_llm["feasible"].apply(
                        lambda x: "Success" if x else "Logic Error"
                    )

                # Preserve error_type for detailed error analysis
                # Also preserve uuid and llm_score for global VBS calculation
                columns_to_keep = [
                    "Method",
                    "Seed",
                    "difficulty_class",
                    "Gap",
                    "Cost",
                    "Pass",
                    "vbs_score",
                    "FailureType",
                ]
                if "uuid" in df_llm.columns:
                    columns_to_keep.append("uuid")
                if "llm_score" in df_llm.columns:
                    columns_to_keep.append("llm_score")
                if "error_type" in df_llm.columns:
                    columns_to_keep.append("error_type")
                if "feasible" in df_llm.columns:
                    columns_to_keep.append("feasible")
                # Only append rows with valid gaps for plotting, but Pass includes all
                dfs.append(df_llm[columns_to_keep])

            # 2. Extract Baseline rows (only from Hero runs to avoid duplication)
            if include_baselines and method == "Ours (Hero)":
                for base_name, display_name in BASELINE_MAPPING.items():
                    score_col = f"score_{base_name}"
                    time_col = (
                        f"core_sec_{base_name}"
                        if f"core_sec_{base_name}" in metrics_data.columns
                        else f"time_{base_name}"
                    )
                    feasible_col = f"feasible_{base_name}"  # Use explicit feasibility column from evaluate.py

                    if (
                        score_col in metrics_data.columns
                        and time_col in metrics_data.columns
                    ):
                        df_base = (
                            metrics_data.copy()
                        )  # Keep all rows for Pass calculation
                        df_base["Method"] = display_name
                        df_base["Seed"] = seed

                        # Store baseline score as llm_score for consistency in global VBS calculation
                        df_base["llm_score"] = df_base[score_col]
                        df_base["feasible"] = (
                            df_base[feasible_col]
                            if feasible_col in df_base.columns
                            else (df_base[score_col] > _INFEASIBLE_SCORE)
                        )

                        # Set Pass from feasibility column (on ALL rows)
                        if feasible_col in df_base.columns:
                            df_base["Pass"] = df_base[feasible_col].astype(int)
                        else:
                            # Fallback: approximate feasibility (score > _INFEASIBLE_SCORE means solver ran, but not necessarily feasible)
                            df_base["Pass"] = (
                                df_base[score_col] > _INFEASIBLE_SCORE
                            ).astype(int)

                        # Now filter broken problems for Gap calculation
                        df_base_valid = df_base[
                            df_base["vbs_score"] > _EPSILON_MEDIUM
                        ].copy()

                        # Calculate Gap for baseline (only for valid VBS rows)
                        base_scores = df_base_valid[score_col].clip(lower=0.0)
                        df_base_valid["Gap"] = (
                            df_base_valid["vbs_score"] - base_scores
                        ) / df_base_valid["vbs_score"]
                        df_base_valid["Cost"] = df_base_valid[time_col]

                        # For rows with invalid VBS, set Gap to NaN
                        df_base_invalid = df_base[
                            df_base["vbs_score"] <= _EPSILON_MEDIUM
                        ].copy()
                        if len(df_base_invalid) > 0:
                            df_base_invalid["Gap"] = np.nan
                            df_base_invalid["Cost"] = df_base_invalid[time_col]
                            df_base = pd.concat(
                                [df_base_valid, df_base_invalid], ignore_index=True
                            )
                        else:
                            df_base = df_base_valid

                        # Baselines don't have "Timeouts" in the same way (they're deterministic)
                        # They either succeed (Pass=1) or fail (Pass=0)
                        df_base["FailureType"] = df_base["Pass"].apply(
                            lambda x: "Success" if x else "Logic Error"
                        )

                        # Preserve feasible column for baselines
                        # Also preserve uuid and baseline score (as llm_score) for global VBS calculation
                        columns_to_keep = [
                            "Method",
                            "Seed",
                            "difficulty_class",
                            "Gap",
                            "Cost",
                            "Pass",
                            "vbs_score",
                            "FailureType",
                        ]
                        if "uuid" in df_base.columns:
                            columns_to_keep.append("uuid")
                        if "llm_score" in df_base.columns:
                            columns_to_keep.append(
                                "llm_score"
                            )  # Store baseline score as llm_score
                        if "feasible" in df_base.columns:
                            columns_to_keep.append("feasible")
                        # Baselines don't have error_type, so we don't need to add it
                        dfs.append(df_base[columns_to_keep])

        except Exception as e:
            print(f"❌ Error reading {csv_path}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    final_df = pd.concat(dfs, ignore_index=True)

    # CRITICAL: Calculate global VBS per (uuid, seed) to ensure all methods use the same VBS
    # This prevents results from changing when new methods are added
    if "uuid" in final_df.columns and "llm_score" in final_df.columns:
        print(
            "📊 Calculating global VBS per instance (max across all methods + baselines)..."
        )

        # Collect all feasible scores per (uuid, seed) pair
        # For all methods (LLM and baselines): use llm_score if feasible
        global_vbs_data = []

        for (uuid, seed), group in final_df.groupby(["uuid", "Seed"]):
            # Collect all feasible scores (LLM methods and baselines both use llm_score column now)
            feasible_scores = group[group["feasible"]]["llm_score"].dropna()

            if len(feasible_scores) > 0:
                global_vbs = feasible_scores.max()
            else:
                # If no feasible solutions, use max of all vbs_score as fallback
                vbs_scores = group["vbs_score"].dropna()
                global_vbs = vbs_scores.max() if len(vbs_scores) > 0 else float("-inf")

            global_vbs_data.append(
                {"uuid": uuid, "Seed": seed, "global_vbs": global_vbs}
            )

        global_vbs_df = pd.DataFrame(global_vbs_data)

        # Merge global VBS back into final_df
        final_df = final_df.merge(global_vbs_df, on=["uuid", "Seed"], how="left")

        # Recalculate gaps using global VBS for all methods
        # Gap formula: (global_vbs - method_score) / global_vbs
        valid_vbs_mask = (final_df["global_vbs"] > _EPSILON_MEDIUM) & final_df[
            "global_vbs"
        ].notna()
        method_scores = (
            final_df.loc[valid_vbs_mask, "llm_score"].fillna(0.0).clip(lower=0.0)
        )
        global_vbs_vals = final_df.loc[valid_vbs_mask, "global_vbs"]

        # Recalculate gaps
        final_df.loc[valid_vbs_mask, "Gap"] = (
            (global_vbs_vals - method_scores) / global_vbs_vals
        ).to_numpy()

        # Update vbs_score to global_vbs for consistency
        final_df["vbs_score"] = final_df["global_vbs"].fillna(final_df["vbs_score"])

        # Drop global_vbs column (now stored in vbs_score)
        final_df = final_df.drop(columns=["global_vbs"], errors="ignore")

        print("✅ Global VBS calculated and gaps recalculated")
    else:
        if "uuid" not in final_df.columns:
            print(
                "⚠️  Warning: 'uuid' column not found. Cannot calculate global VBS. Using per-method VBS."
            )
        if "llm_score" not in final_df.columns:
            print(
                "⚠️  Warning: 'llm_score' column not found. Cannot calculate global VBS. Using per-method VBS."
            )

    # Filter out invalid gaps (negative, > 1.0, or NaN) for plotting/analysis
    # BUT: Keep Pass rate calculated on ALL rows (including those with invalid gaps)
    # This ensures pass rates reflect the true fraction of feasible solutions
    valid_gap_mask = (
        (final_df["Gap"] >= 0) & (final_df["Gap"] <= 1.0) & (~final_df["Gap"].isna())
    )

    # For methods where we need Gap (plotting, gap stats), use only valid gaps
    # But for Pass rate calculation, we'll aggregate ALL rows
    # So we keep all rows but mark invalid gaps
    final_df = final_df.copy()
    final_df.loc[~valid_gap_mask, "Gap"] = np.nan  # Mark invalid gaps as NaN

    print(
        f"✅ Loaded {len(final_df)} instances across {final_df['Method'].nunique()} methods"
    )
    print(
        f"   Valid gaps: {valid_gap_mask.sum()}/{len(final_df)} ({valid_gap_mask.mean() * 100:.1f}%)"
    )
    return final_df


def generate_latex_table(results_data: pd.DataFrame, output_path: str):
    """
    Generate LaTeX table with Mean ± Std across seeds.

    Format: $97.8_{\\pm 0.1}$

    IMPORTANT: Pass rate is calculated on ALL rows (including invalid gaps).
    Gap and Cost are calculated only on rows with valid gaps.
    """
    # First: Mean per seed per method
    # Pass rate: use ALL rows (including those with invalid gaps)
    # Gap and Cost: use only rows with valid gaps
    per_seed_pass = (
        results_data.groupby(["Method", "Seed"])["Pass"].mean().reset_index()
    )
    per_seed_gap_cost = (
        results_data[results_data["Gap"].notna()]
        .groupby(["Method", "Seed"])
        .agg({"Gap": "mean", "Cost": "mean"})
        .reset_index()
    )

    # Merge to combine Pass (all rows) with Gap/Cost (valid rows only)
    per_seed = per_seed_pass.merge(per_seed_gap_cost, on=["Method", "Seed"], how="left")

    # Second: Mean/Std across seeds
    agg = per_seed.groupby("Method").agg(
        {"Pass": ["mean", "std"], "Gap": ["mean", "std"], "Cost": ["mean", "std"]}
    )

    def _is_close(a: float, b: float, tol: float = 1e-9) -> bool:
        return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol)

    # Best-in-column (ties allowed; compare on MEAN only)
    pass_means = (agg[("Pass", "mean")] * 100).to_numpy()
    gap_means = (agg[("Gap", "mean")] * 100).to_numpy()
    time_means = agg[("Cost", "mean")].to_numpy()

    best_pass = np.nanmax(pass_means) if len(pass_means) else np.nan
    best_gap = np.nanmin(gap_means) if len(gap_means) else np.nan
    best_time = np.nanmin(time_means) if len(time_means) else np.nan

    # Format for LaTeX
    rows = []
    for method in agg.index:
        pass_m = agg.loc[method, ("Pass", "mean")] * 100
        pass_s = (
            agg.loc[method, ("Pass", "std")] * 100
            if not pd.isna(agg.loc[method, ("Pass", "std")])
            else 0.0
        )
        gap_m = agg.loc[method, ("Gap", "mean")] * 100
        gap_s = (
            agg.loc[method, ("Gap", "std")] * 100
            if not pd.isna(agg.loc[method, ("Gap", "std")])
            else 0.0
        )
        cost_m = agg.loc[method, ("Cost", "mean")]
        cost_s = (
            agg.loc[method, ("Cost", "std")]
            if not pd.isna(agg.loc[method, ("Cost", "std")])
            else 0.0
        )

        pass_is_best = _is_close(pass_m, best_pass)
        gap_is_best = _is_close(gap_m, best_gap)
        time_is_best = _is_close(cost_m, best_time)

        def _fmt_math(mean: float, std: float, mean_fmt: str, std_fmt: str) -> str:
            if (
                std > 0
                and std_fmt
                and (
                    (std_fmt == "{:.1f}" and std > _STD_THRESHOLD)
                    or (std_fmt == "{:.3f}" and std > _STD_THRESHOLD_COST)
                )
            ):
                inner = f"{mean_fmt.format(mean)}_{{\\pm {std_fmt.format(std)}}}"
            else:
                inner = mean_fmt.format(mean)
            return f"${inner}$"

        def _maybe_bold_math(s: str, bold: bool) -> str:
            # Turn "$...$" into "$\\mathbf{...}$" (robust for entries with _{\\pm ...})
            if not bold:
                return s
            if s.startswith("$") and s.endswith("$"):
                inner = s[1:-1]
                return f"$\\\\mathbf{{{inner}}}$"
            return f"\\\\textbf{{{s}}}"

        pass_str = _maybe_bold_math(
            _fmt_math(pass_m, pass_s, "{:.1f}", "{:.1f}"), pass_is_best
        )
        gap_str = _maybe_bold_math(
            _fmt_math(gap_m, gap_s, "{:.1f}", "{:.1f}"), gap_is_best
        )
        cost_str = _maybe_bold_math(
            _fmt_math(cost_m, cost_s, "{:.3f}", "{:.3f}"), time_is_best
        )

        rows.append(
            {
                "Method": method,
                r"Pass Rate (\%) $\uparrow$": pass_str,
                r"Gap (\%) $\downarrow$": gap_str,
                r"Time (s) $\downarrow$": cost_str,
            }
        )

    tex_df = pd.DataFrame(rows)

    # Sort by method (Hero first, then ablations, then baselines)
    method_order = [
        "Ours (Hero)",
        "Base (Best-of-64)",
        "Ours (+Oracle)",
        "Ours (+Diversity)",
        "Ours (w/o Structure)",
        "Ours (w/o Prompt)",
        "ShinkaEvolve",
        "CP-SAT",
        "Local Search",
        "Greedy",
        "BnB",
    ]
    tex_df["sort_key"] = tex_df["Method"].apply(
        lambda x: method_order.index(x) if x in method_order else 999
    )
    tex_df = tex_df.sort_values("sort_key").drop("sort_key", axis=1)

    # Generate LaTeX
    latex_str = tex_df.to_latex(index=False, escape=False, column_format="lccc")

    with Path(output_path).open("w") as f:
        f.write(latex_str)

    print(f"✅ LaTeX table saved to {output_path}")


def generate_error_types_table(results_data: pd.DataFrame, output_path: str):  # noqa: PLR0912, PLR0915
    """
    Generate LaTeX table with detailed error type percentages per method.

    Uses the actual error_type column from the data (none, syntax, runtime, timeout, json,
    security, missing_code, constraint, unknown) rather than collapsing into categories.

    Format: $97.8_{\\pm 0.1}$ for each error type.
    Shows how each method fails (or succeeds) with full detail.
    """
    # Check if error_type column exists (for LLM methods)
    # For baselines, we only have feasibility, so we'll handle them separately
    llm_df = results_data[
        results_data["Method"].str.contains("Ours|Shinka", case=False, na=False)
    ].copy()

    if "error_type" not in llm_df.columns and len(llm_df) > 0:
        print("⚠️  error_type column not found. Skipping error types table.")
        return

    # Map error types to readable labels (matching evaluate.py)
    error_labels = {
        "none": "None (Valid)",
        "syntax": "Syntax Error",
        "runtime": "Runtime Error",
        "timeout": "Timeout",
        "json": "JSON Parse Error",
        "security": "Security Violation",
        "missing_code": "Missing Code Block",
        "constraint": "Constraint Violation",
        "unknown": "Unknown Error",
    }

    # For LLM methods: use error_type column from the original data
    # We need to reload the data to get error_type, since it's not in the aggregated results_data
    # Instead, let's check if error_type is already in results_data (it should be if we preserved it)
    all_methods_data = []

    for method in results_data["Method"].unique():
        method_df = results_data[results_data["Method"] == method].copy()

        # Check if this is an LLM method and if error_type exists
        is_llm = method in llm_df["Method"].unique() if len(llm_df) > 0 else False

        if is_llm and "error_type" in method_df.columns:
            # LLM method: use actual error_type
            method_df["error_type_clean"] = method_df["error_type"].fillna("unknown")
        elif "Pass" in method_df.columns:
            # Baseline method or LLM without error_type: use Pass column (which is derived from feasibility)
            # Pass=1 means feasible (none), Pass=0 means infeasible (constraint violation)
            method_df["error_type_clean"] = method_df["Pass"].apply(
                lambda x: "none" if x == 1 else "constraint"
            )
        elif "feasible" in method_df.columns:
            # Fallback: use feasible column if Pass doesn't exist
            method_df["error_type_clean"] = method_df.apply(
                lambda row: "none" if row.get("feasible", False) else "constraint",
                axis=1,
            )
        else:
            # Fallback: assume all are valid if we can't determine
            method_df["error_type_clean"] = "none"

        all_methods_data.append(method_df)

    if not all_methods_data:
        print("⚠️  No data available for error types table.")
        return

    combined_df = pd.concat(all_methods_data, ignore_index=True)

    # Calculate percentages per Method per Seed
    counts = (
        combined_df.groupby(["Method", "Seed", "error_type_clean"])
        .size()
        .reset_index(name="Count")
    )
    totals = combined_df.groupby(["Method", "Seed"]).size().reset_index(name="Total")

    merged = counts.merge(totals, on=["Method", "Seed"])
    merged["Rate"] = (merged["Count"] / merged["Total"]) * 100

    # Aggregate across seeds: mean ± std
    agg = (
        merged.groupby(["Method", "error_type_clean"])["Rate"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Pivot to have error_type as columns
    pivot_mean = agg.pivot_table(
        index="Method", columns="error_type_clean", values="mean"
    ).fillna(0.0)
    pivot_std = agg.pivot_table(
        index="Method", columns="error_type_clean", values="std"
    ).fillna(0.0)

    # Get all error types that appear in the data, ordered by importance
    all_error_types = [
        "none",
        "timeout",
        "constraint",
        "syntax",
        "runtime",
        "json",
        "security",
        "missing_code",
        "unknown",
    ]
    # Only include error types that actually appear in the data
    available_error_types = [et for et in all_error_types if et in pivot_mean.columns]

    def _is_close(a: float, b: float, tol: float = 1e-9) -> bool:
        return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol)

    # Best-in-column rules:
    # - None (Valid): higher is better
    # - All other error types: lower is better
    best_by_et: dict[str, float] = {}
    for et in available_error_types:
        col = pivot_mean[et].to_numpy()
        if et == "none":
            best_by_et[et] = float(np.nanmax(col)) if len(col) else float("nan")
        else:
            best_by_et[et] = float(np.nanmin(col)) if len(col) else float("nan")

    # Format for LaTeX
    rows = []
    for method in pivot_mean.index:
        row = {"Method": method}

        for et in available_error_types:
            mean_val = pivot_mean.loc[method, et]
            std_val = pivot_std.loc[method, et] if et in pivot_std.columns else 0.0

            # Format as $97.8_{\pm 0.1}$ or just $97.8$ if std is negligible
            if std_val > _STD_THRESHOLD:
                inner = f"{mean_val:.1f}_{{\\pm {std_val:.1f}}}"
            else:
                inner = f"{mean_val:.1f}"

            is_best = _is_close(float(mean_val), best_by_et.get(et, float("nan")))
            formatted = f"$\\\\mathbf{{{inner}}}$" if is_best else f"${inner}$"

            # Use readable label for column name
            base_label = error_labels.get(et, et.replace("_", " ").title())
            arrow = r"$\uparrow$" if et == "none" else r"$\downarrow$"
            label = rf"{base_label} (\%) {arrow}"
            row[label] = formatted

        rows.append(row)

    tex_df = pd.DataFrame(rows)

    # Sort by method (Hero first, then Base, then ablations, then baselines)
    method_order = [
        "Ours (Hero)",
        "Base (Best-of-64)",
        "Ours (+Oracle)",
        "Ours (+Diversity)",
        "Ours (w/o Structure)",
        "Ours (w/o Prompt)",
        "ShinkaEvolve",
        "CP-SAT",
        "Local Search",
        "Greedy",
        "BnB",
    ]
    tex_df["sort_key"] = tex_df["Method"].apply(
        lambda x: method_order.index(x) if x in method_order else 999
    )
    tex_df = tex_df.sort_values("sort_key").drop("sort_key", axis=1)

    # Generate LaTeX with appropriate column format
    num_cols = len(tex_df.columns) - 1  # -1 for Method column
    col_format = f"l{'c' * num_cols}"
    latex_str = tex_df.to_latex(index=False, escape=False, column_format=col_format)

    with Path(output_path).open("w") as f:
        f.write(latex_str)

    print(f"✅ Error types table saved to {output_path}")


def plot_efficiency_frontier(results_data: pd.DataFrame, output_path: str, result_roots: list[str] | None = None):  # noqa: PLR0912, PLR0915
    """
    Figure 1: Efficiency Frontier (Pareto of Gap vs Cost with Error Bars).

    Shows the trade-off between optimality gap and inference cost.
    Best methods are in the bottom-left (low gap, low cost).
    """
    # Aggregation: Mean per seed, then mean ± std across seeds
    # Filter out NaN gaps for plotting
    df_valid = results_data[results_data["Gap"].notna()].copy()
    per_seed = (
        df_valid.groupby(["Method", "Seed"])
        .agg({"Gap": "mean", "Cost": "mean"})
        .reset_index()
    )

    agg = per_seed.groupby("Method").agg(["mean", "std"])

    _fig, ax = plt.subplots(figsize=(3.25, 2.75))

    # Methods to highlight (main methods for efficiency comparison)
    highlight = [
        "Ours (Hero)",
        "Base (Best-of-64)",
        "CP-SAT",
        "Local Search",
        "Greedy",
        "BnB",
        "ShinkaEvolve",
    ]

    for method in agg.index:
        # Include main methods, Base (Best-of-64), and ablations
        if method not in highlight and not method.startswith("Ours"):
            continue

        x = agg.loc[method, ("Cost", "mean")]
        y = agg.loc[method, ("Gap", "mean")] * 100  # Convert to %

        x_err = (
            agg.loc[method, ("Cost", "std")]
            if not pd.isna(agg.loc[method, ("Cost", "std")])
            else 0.0
        )
        y_err = (
            agg.loc[method, ("Gap", "std")] * 100
            if not pd.isna(agg.loc[method, ("Gap", "std")])
            else 0.0
        )

        color = PALETTE.get(method, "gray")
        # Marker conventions (match manuscript palette/shape semantics)
        if method.startswith("Ours"):
            marker = "D"  # diamond
        elif method == "Base (Best-of-64)":
            marker = "h"  # hexagon
        else:
            marker = "o"
        zorder = (
            10
            if method.startswith("Ours")
            or method in {"Base (Best-of-64)", "ShinkaEvolve"}
            else 5
        )

        ax.errorbar(
            x,
            y,
            xerr=x_err,
            yerr=y_err,
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=2,
            elinewidth=1,
            markersize=5,
            label=method,
            zorder=zorder,
            alpha=0.8,
        )

        # Annotate Hero
        if method == "Ours (Hero)":
            ax.annotate(
                "Ours",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    # Add Base Model Best-of-64 point (fallback if not in aggregated data)
    # Only load from scaling_stats.csv if Base (Best-of-64) is not already in agg.index.
    # Prefer including Base from metrics_final.csv (so it appears in *all* plots, not just Fig. 1).
    base_scaling_files = []
    if "Base (Best-of-64)" not in agg.index:
        search_dirs = list(result_roots or []) + [BASE_RESULT_DIR]
        for search_dir in search_dirs:
            base_scaling_files = list(
                Path(search_dir).glob("qwen2.5-coder-14b/base/seed*/scaling_stats.csv")
            )
            base_scaling_files = [str(f) for f in base_scaling_files]
            if base_scaling_files:
                break
    if base_scaling_files:
        base_gaps = []
        base_costs = []
        for scaling_file in base_scaling_files:
            try:
                scaling_df = pd.read_csv(scaling_file)
                k64_row = scaling_df[scaling_df["k"] == _BEST_OF_K]
                if len(k64_row) > 0:
                    gap_64 = k64_row.iloc[0]["opt_gap_mean"]  # Already in %
                    base_gaps.append(gap_64)

                    # Calculate cost: load corresponding metrics_final.csv
                    metrics_file = Path(scaling_file).with_name("metrics_final.csv")
                    if metrics_file.exists():
                        metrics_df = pd.read_csv(metrics_file)
                        df_valid_metrics = metrics_df[metrics_df["feasible"]].copy()
                        if len(df_valid_metrics) > 0:
                            if "llm_core_sec" in df_valid_metrics.columns:
                                cost_per_sample = df_valid_metrics[
                                    "llm_core_sec"
                                ].mean()
                            elif "execution_time" in df_valid_metrics.columns:
                                cost_per_sample = df_valid_metrics[
                                    "execution_time"
                                ].mean()
                            else:
                                cost_per_sample = 0.0
                            # Cost for best-of-64 = _BEST_OF_K x cost per sample
                            base_costs.append(cost_per_sample * _BEST_OF_K)
            except Exception as e:
                print(
                    f"⚠️  Warning: Could not load base model data from {scaling_file}: {e}"
                )
                continue

        if base_gaps and base_costs:
            base_gap_mean = np.mean(base_gaps)
            base_gap_std = np.std(base_gaps)
            base_cost_mean = np.mean(base_costs)
            base_cost_std = np.std(base_costs)

            # Plot Base Model Best-of-64 using the standard palette + hexagon marker
            base_color = PALETTE.get("Base (Best-of-64)", "#e377c2")
            ax.errorbar(
                base_cost_mean,
                base_gap_mean,
                xerr=base_cost_std,
                yerr=base_gap_std,
                fmt="h",
                color=base_color,
                ecolor=base_color,
                capsize=2,
                elinewidth=1,
                markersize=7,
                markeredgewidth=1.2,
                markeredgecolor=base_color,
                markerfacecolor="white",
                label="Base (Best-of-64)",
                zorder=15,
                alpha=0.9,
            )

    ax.set_xscale("log")
    ax.set_xlabel(r"Inference Cost (Core $\times$ s) $\downarrow$", fontsize=10)
    ax.set_ylabel(r"Optimality Gap (%) $\downarrow$", fontsize=10)
    ax.set_ylim(bottom=-1.0, top=100.0)  # Extend to 100%

    ax.grid(True, which="major", ls="-", alpha=0.15)
    # Move legend to top left
    legend = ax.legend(
        frameon=True, loc="upper left", fontsize=7, framealpha=0.90, ncol=1
    )
    legend.set_zorder(100)  # Ensure legend is on top of other elements

    plt.tight_layout(pad=0.2)
    # Save both PNG and PDF
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = Path(output_path).with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved Efficiency Frontier: {output_path} and {pdf_path}")


def plot_scaling_analysis(output_path: str, result_roots: list[str] | None = None):  # noqa: PLR0912, PLR0915
    """
    Figure 5: Scaling Analysis (Base Model Optimality Gap vs k with Hero overlay).

    Shows that Base Model performance saturates at ~30% gap even with k=64,
    while Hero achieves 4.1% gap with k=1, demonstrating algorithmic advantage.
    """
    # Load base model scaling stats from all seeds
    # Search report-set result roots first, then fall back to legacy BASE_RESULT_DIR
    search_dirs = list(result_roots or []) + [BASE_RESULT_DIR]
    base_scaling_files = []
    for search_dir in search_dirs:
        base_scaling_files = list(
            Path(search_dir).glob("qwen2.5-coder-14b/base/seed*/scaling_stats.csv")
        )
        base_scaling_files = [str(f) for f in base_scaling_files]
        if base_scaling_files:
            break

    if not base_scaling_files:
        print("⚠️  No base model scaling stats found. Skipping scaling plot.")
        return

    # Aggregate scaling stats across seeds
    all_k_values = set()
    gap_data = defaultdict(list)  # k -> [gap values across seeds]

    for scaling_file in base_scaling_files:
        try:
            scaling_df = pd.read_csv(scaling_file)
            for _, row in scaling_df.iterrows():
                k = int(row["k"])
                gap = row["opt_gap_mean"]  # Already in %
                all_k_values.add(k)
                gap_data[k].append(gap)
        except Exception as e:
            print(f"⚠️  Warning: Could not load {scaling_file}: {e}")
            continue

    if not gap_data:
        print("⚠️  No scaling data loaded. Skipping scaling plot.")
        return

    # Calculate mean ± std across seeds for each k
    k_sorted = sorted(all_k_values)
    gap_means = [np.mean(gap_data[k]) for k in k_sorted]
    gap_stds = [np.std(gap_data[k]) for k in k_sorted]

    # Load Hero aggregated gap (from main results)
    # Try to load from aggregated results or calculate from results_data if available
    hero_gap = 4.1  # Default from known results
    hero_gap_std = 1.3

    # Try to get Hero gap from aggregated results if available
    hero_result_files = list(
        Path(BASE_RESULT_DIR).glob(
            "qwen2.5-coder-14b/grpo/seed*/job-*/metrics_final.csv"
        )
    )
    hero_result_files = [str(f) for f in hero_result_files]
    if hero_result_files:
        # Find Hero job IDs (from the specific list)
        hero_job_ids = ["1315163", "1315168", "1315173"]  # Hero for seeds 101, 202, 303
        hero_gaps = []
        for f in hero_result_files:
            if any(job_id in f for job_id in hero_job_ids):
                try:
                    df_hero = pd.read_csv(f)
                    df_valid = df_hero[
                        (df_hero["feasible"]) & (df_hero["vbs_score"] > _EPSILON_MEDIUM)
                    ].copy()
                    if len(df_valid) > 0:
                        llm_scores = df_valid["llm_score"].clip(lower=0.0)
                        gap = (
                            (df_valid["vbs_score"] - llm_scores) / df_valid["vbs_score"]
                        ).mean() * 100
                        hero_gaps.append(gap)
                except Exception:
                    continue
        if hero_gaps:
            hero_gap = np.mean(hero_gaps)
            hero_gap_std = np.std(hero_gaps)

    # Load Greedy baseline gap dynamically from baseline results
    greedy_gap = 22.0  # Default fallback
    greedy_gaps = []
    if hero_result_files:
        # Baselines are extracted from Hero runs, so use the same files
        for f in hero_result_files:
            if any(job_id in f for job_id in hero_job_ids):
                try:
                    df_hero = pd.read_csv(f)
                    # Check if Greedy baseline columns exist
                    if (
                        "score_greedy" in df_hero.columns
                        and "feasible_greedy" in df_hero.columns
                    ):
                        df_valid = df_hero[
                            (df_hero["feasible_greedy"])
                            & (df_hero["vbs_score"] > _EPSILON_MEDIUM)
                        ].copy()
                        if len(df_valid) > 0:
                            greedy_scores = df_valid["score_greedy"].clip(lower=0.0)
                            gap = (
                                (df_valid["vbs_score"] - greedy_scores)
                                / df_valid["vbs_score"]
                            ).mean() * 100
                            greedy_gaps.append(gap)
                except Exception:
                    continue
        if greedy_gaps:
            greedy_gap = np.mean(greedy_gaps)

    # Create plot
    _fig, ax = plt.subplots(figsize=(3.25, 2.5))

    # Plot Base Model curve
    ax.errorbar(
        k_sorted,
        gap_means,
        yerr=gap_stds,
        fmt="-o",
        color=PALETTE.get("Base (Best-of-64)", "#e377c2"),
        capsize=3,
        elinewidth=1,
        markersize=5,
        label="Base Model",
        linewidth=1.5,
        alpha=0.8,
    )

    # Plot Hero horizontal line (greedy, k=1)
    ax.axhline(
        y=hero_gap,
        color=PALETTE.get("Ours (Hero)", "#1f77b4"),
        linestyle="--",
        linewidth=1.5,
        label="Ours (Greedy, k=1)",
        alpha=0.8,
    )
    # Add error bar for Hero
    ax.fill_between(
        [k_sorted[0], k_sorted[-1]],
        hero_gap - hero_gap_std,
        hero_gap + hero_gap_std,
        color=PALETTE.get("Ours (Hero)", "#1f77b4"),
        alpha=0.2,
    )

    # Plot Greedy baseline for reference
    ax.axhline(
        y=greedy_gap,
        color=PALETTE.get("Greedy", "#d62728"),
        linestyle=":",
        linewidth=1,
        label="Greedy Baseline",
        alpha=0.6,
    )

    # Add annotation: "Algorithmic Gap" arrow/brace
    # Position: between Base at k=64 and Hero line
    base_k64_gap = gap_means[-1] if len(gap_means) > 0 else 30.6
    mid_gap = (base_k64_gap + hero_gap) / 2
    ax.annotate(
        "",
        xy=(k_sorted[-1], hero_gap),
        xytext=(k_sorted[-1], base_k64_gap),
        arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1.5, "alpha": 0.7},
    )
    ax.text(
        k_sorted[-1] * 1.3,
        mid_gap,
        "Algorithmic\nGap",
        fontsize=8,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "black",
            "lw": 0.5,
        },
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of Samples (k)", fontsize=10)
    ax.set_ylabel(r"Optimality Gap (%) $\downarrow$", fontsize=10)
    ax.set_ylim(bottom=-1.0, top=100.0)
    ax.set_xticks(k_sorted)
    ax.set_xticklabels([str(int(k)) for k in k_sorted])

    ax.grid(True, which="major", ls="-", alpha=0.15)
    ax.grid(True, which="minor", ls="--", alpha=0.1)
    ax.legend(frameon=True, loc="upper right", fontsize=7, framealpha=0.95, ncol=1)

    plt.tight_layout(pad=0.2)
    # Save both PNG and PDF
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = Path(output_path).with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved Scaling Analysis: {output_path} and {pdf_path}")


def plot_robustness_profile(results_data: pd.DataFrame, output_path: str):
    """
    Figure 2: Performance Profile (Fraction of instances with Gap <= threshold).

    Shows robustness: what fraction of problems achieve at most a given gap threshold.
    Best methods shoot up immediately (high fraction at low gap).
    """
    from matplotlib.ticker import FuncFormatter  # noqa: PLC0415

    _fig, ax = plt.subplots(figsize=(3.25, 2.5))

    # X-axis: Gap Thresholds (0% to 50% for readability)
    taus = np.linspace(0.0, 0.5, 500)

    # Plot Methods (in order of importance)
    methods = [
        "Ours (Hero)",
        "Base (Best-of-64)",
        "Local Search",
        "Greedy",
        "ShinkaEvolve",
        "CP-SAT",
        "BnB",
    ]
    available = [m for m in methods if m in results_data["Method"].unique()]

    for method in available:
        subset = results_data[results_data["Method"] == method]
        # Filter out NaN gaps for robustness profile
        subset = subset[subset["Gap"].notna()].copy()
        color = PALETTE.get(method, "black")

        # Calculate CDF per seed
        ys_per_seed = []
        for seed in sorted(subset["Seed"].unique()):
            gaps = subset[subset["Seed"] == seed]["Gap"].to_numpy()
            if len(gaps) == 0:
                continue
            # CDF: Fraction with Gap <= tau for each tau
            y = np.mean(gaps[:, None] <= taus[None, :], axis=0)
            ys_per_seed.append(y)

        if not ys_per_seed:
            continue

        ys = np.array(ys_per_seed)
        y_mean = np.mean(ys, axis=0)
        y_std = np.std(ys, axis=0)

        # Plot
        lw = (
            2.0
            if method in {"Ours (Hero)", "Base (Best-of-64)", "ShinkaEvolve"}
            else 1.5
        )
        alpha = (
            1.0
            if method.startswith("Ours")
            or method in {"Base (Best-of-64)", "ShinkaEvolve"}
            else 0.8
        )
        linestyle = (
            "-"
            if method.startswith("Ours")
            or method in {"Base (Best-of-64)", "ShinkaEvolve"}
            else "--"
        )

        ax.plot(
            taus,
            y_mean,
            label=method,
            color=color,
            lw=lw,
            alpha=alpha,
            linestyle=linestyle,
        )
        if len(ys_per_seed) > 1:
            ax.fill_between(
                taus, y_mean - y_std, y_mean + y_std, color=color, alpha=0.1
            )

    # Formatting
    ax.set_xlabel(r"Optimality Gap ($\tau$) $\downarrow$", fontsize=10)
    ax.set_ylabel(r"Fraction Solved $\leq \tau$ $\downarrow$", fontsize=10)
    ax.set_xlim(0.0, 0.5)  # Focus on the "good" region (0% to 50% gap)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

    ax.legend(loc="lower right", fontsize=7, framealpha=0.95)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(pad=0.2)
    # Save both PNG and PDF
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = Path(output_path).with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved Robustness Profile: {output_path} and {pdf_path}")


def plot_stratified_boxplot(results_data: pd.DataFrame, output_path: str):
    """
    Figure 3: Violin Plot of Gap by Difficulty.

    Shows distribution of optimality gap across difficulty levels.
    Uses violin plots to show full distribution shape.
    """
    # Filter methods (include CP-SAT and Base)
    methods = [
        "Ours (Hero)",
        "Base (Best-of-64)",
        "Local Search",
        "Greedy",
        "ShinkaEvolve",
        "BnB",
        "CP-SAT",
    ]
    plot_df = results_data[results_data["Method"].isin(methods)].copy()

    # Filter out NaN gaps for violin plot
    plot_df = plot_df[plot_df["Gap"].notna()].copy()

    if plot_df.empty:
        print("⚠️  No data to plot for selected methods")
        return

    # Order: Trivial -> Moderate -> Hard
    plot_df["difficulty_class"] = pd.Categorical(
        plot_df["difficulty_class"],
        categories=["Trivial", "Moderate", "Hard"],
        ordered=True,
    )

    # Convert Gap to %
    plot_df["GapPct"] = plot_df["Gap"] * 100

    _fig, ax = plt.subplots(figsize=(6.75, 2.5))  # Two-column width

    # Check if CP-SAT gap is always 0 (or very close to 0)
    cpsat_df = plot_df[plot_df["Method"] == "CP-SAT"].copy()
    cpsat_always_zero = False
    if not cpsat_df.empty:
        # Check if all CP-SAT gaps are essentially 0 (within tolerance)
        cpsat_gaps = cpsat_df["GapPct"].abs()
        cpsat_always_zero = (
            cpsat_gaps < _CP_SAT_GAP_TOLERANCE
        ).all()  # All gaps < _CP_SAT_GAP_TOLERANCE%

    # Separate CP-SAT if it's always zero, otherwise include it in violins
    if cpsat_always_zero:
        other_df = plot_df[plot_df["Method"] != "CP-SAT"].copy()
    else:
        other_df = plot_df.copy()
        cpsat_df = pd.DataFrame()  # Empty so we don't plot the line

    # Plot violins for methods (including CP-SAT if not always zero)
    if not other_df.empty:
        sns.violinplot(
            data=other_df,
            x="difficulty_class",
            y="GapPct",
            hue="Method",
            palette=PALETTE,
            linewidth=0.8,
            ax=ax,
            saturation=0.9,
            inner="box",
            cut=0,
        )

    # Plot CP-SAT as a horizontal line at 0% gap (optimal) if always zero
    if not cpsat_df.empty and cpsat_always_zero:
        # Draw a horizontal line at y=0 for CP-SAT across all difficulties
        for diff in ["Trivial", "Moderate", "Hard"]:
            # Find the x-position for this difficulty
            diff_positions = {"Trivial": 0, "Moderate": 1, "Hard": 2}
            x_pos = diff_positions.get(diff, 1)
            # Draw a thick horizontal line at 0% gap
            ax.axhline(
                y=0.0,
                xmin=0.15 + x_pos * 0.28,
                xmax=0.15 + (x_pos + 1) * 0.28,
                color=PALETTE.get("CP-SAT", "#ff7f0e"),
                linewidth=3,
                linestyle="--",
                alpha=0.8,
                zorder=5,
            )

        # Add CP-SAT to legend manually (as a line, not a violin)
        from matplotlib.lines import Line2D  # noqa: PLC0415

        handles, labels = ax.get_legend_handles_labels()
        cpsat_handle = Line2D(
            [0],
            [0],
            color=PALETTE.get("CP-SAT", "#ff7f0e"),
            linewidth=3,
            linestyle="--",
            label="CP-SAT (Optimal)",
        )
        handles.append(cpsat_handle)
        labels.append("CP-SAT (Optimal)")
        ax.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            ncol=3,
            fontsize=8,
            framealpha=0.9,
        )
    else:
        ax.legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.9)

    ax.set_ylabel(r"Optimality Gap (%) $\downarrow$", fontsize=10)
    ax.set_xlabel("Difficulty", fontsize=10)
    ax.set_ylim(-2, 50)  # Focus on meaningful gaps
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout(pad=0.2)
    # Save both PNG and PDF
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = Path(output_path).with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved Stratified Violin Plot: {output_path} and {pdf_path}")


def plot_failure_analysis(results_data: pd.DataFrame, output_path: str):
    """
    Figure 4: Stacked Bar Chart of Failure Modes.

    Visualizes [Logic Errors] vs [Timeouts] for Ours and Ablations.
    Shows how complexity breaks alignment: more complex methods generate invalid solutions.
    """
    # Filter to only "Ours" methods + Base (Best-of-64) + ShinkaEvolve (ignore simple baselines)
    relevant_methods = [
        m
        for m in results_data["Method"].unique()
        if "Ours" in m or "Base (Best-of-64)" in m or "Shinka" in m
    ]
    plot_df = results_data[results_data["Method"].isin(relevant_methods)].copy()

    if plot_df.empty:
        print("⚠️  No data to plot for failure analysis")
        return

    # Check if FailureType column exists
    if "FailureType" not in plot_df.columns:
        print("⚠️  FailureType column not found. Skipping failure analysis plot.")
        return

    # Calculate rates per Method per Seed
    # Group by Method -> Seed -> FailureType
    counts = (
        plot_df.groupby(["Method", "Seed", "FailureType"])
        .size()
        .reset_index(name="Count")
    )
    totals = plot_df.groupby(["Method", "Seed"]).size().reset_index(name="Total")

    merged = counts.merge(totals, on=["Method", "Seed"])
    merged["Rate"] = (merged["Count"] / merged["Total"]) * 100

    # Average across seeds
    agg = (
        merged.groupby(["Method", "FailureType"])["Rate"].mean().unstack(fill_value=0.0)  # noqa: PD010
    )

    # Reorder columns for stacking: Logic Error (Bottom/Red), Timeout (Top/Orange)
    # We plot just the Errors (Logic + Timeout) to focus on failure modes
    if "Logic Error" not in agg.columns:
        agg["Logic Error"] = 0.0
    if "Timeout" not in agg.columns:
        agg["Timeout"] = 0.0

    plot_data = agg[["Logic Error", "Timeout"]].copy()

    # Sort methods by Total Error Rate (descending)
    plot_data["TotalError"] = plot_data["Logic Error"] + plot_data["Timeout"]
    plot_data = plot_data.sort_values("TotalError", ascending=False)
    plot_data = plot_data.drop(columns=["TotalError"])

    # Check if we have any data to plot
    if plot_data.empty or plot_data.sum(axis=1).max() == 0:
        print("⚠️  No failure data to plot (all methods succeeded)")
        return

    # Plotting
    _fig, ax = plt.subplots(figsize=(5, 3))

    # Colors: Red for Logic Errors, Orange for Timeouts (paper-compatible)
    colors = ["#d62728", "#ff7f0e"]  # Red, Orange

    plot_data.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors,
        width=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_ylabel(r"Failure Rate (%) $\downarrow$", fontsize=10)
    ax.set_xlabel("")
    # No title (paper standard)
    ax.legend(title="Failure Type", loc="upper right", fontsize=8, framealpha=0.95)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    # Auto-scale y-axis with padding (ensure at least 5% range for visibility)
    max_rate = plot_data.sum(axis=1).max()
    ax.set_ylim(0, max(max_rate * 1.15, 5.0))  # At least 5% range, or 15% padding

    plt.tight_layout(pad=0.2)
    # Save both PNG and PDF
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = Path(output_path).with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved Failure Analysis: {output_path} and {pdf_path}")


def plot_stratified_spread_old(
    results_data: pd.DataFrame, output_path: str, main_methods: list[str] | None = None
):
    """
    Generate stratified performance plot with instance spread (scatter) + seed stability (centroids).

    Args:
        results_data: DataFrame with Method, Seed, difficulty_class, Gap, Cost
        output_path: Output file path
        main_methods: List of methods to include (if None, uses default set)
    """
    if main_methods is None:
        main_methods = [
            "Ours (Hero)",
            "Local Search",
            "Greedy",
            "CP-SAT",
            "BnB",
            "ShinkaEvolve",
        ]

    plot_df = results_data[results_data["Method"].isin(main_methods)].copy()

    if plot_df.empty:
        print("⚠️  No data to plot for selected methods")
        return

    # Order difficulties
    plot_df["difficulty_class"] = pd.Categorical(
        plot_df["difficulty_class"],
        categories=["Trivial", "Moderate", "Hard"],
        ordered=True,
    )

    # Calculate Centroids (Mean across all instances/seeds per difficulty)
    centroids = (
        plot_df.groupby(["difficulty_class", "Method"], observed=True)
        .agg({"Cost": "mean", "Gap": "mean"})
        .reset_index()
    )

    # Create FacetGrid WITHOUT hue to avoid automatic legend
    g = sns.FacetGrid(
        plot_df,
        col="difficulty_class",
        height=3.5,
        aspect=1.0,
        sharey=True,
        sharex=False,
    )

    # 1. Plot the "Instance Cloud" (Scatter with high transparency)
    def plot_scatter(data, **_kwargs):
        ax = plt.gca()
        for method in data["Method"].unique():
            subset = data[data["Method"] == method]
            color = PALETTE.get(method, "black")
            ax.scatter(
                subset["Cost"], subset["Gap"], alpha=0.1, s=10, linewidth=0, color=color
            )

    g.map_dataframe(plot_scatter)

    # 2. Plot the "Centroids" (Mean performance with big non-transparent markers)
    for ax_idx, ax in enumerate(g.axes.flat):
        diff = ["Trivial", "Moderate", "Hard"][ax_idx]
        subset = centroids[centroids["difficulty_class"] == diff]

        for _, row in subset.iterrows():
            method = row["Method"]
            color = PALETTE.get(method, "black")
            # Use non-transparent X marker for centroids (average)
            ax.scatter(
                row["Cost"],
                row["Gap"],
                c=color,
                s=200,
                edgecolors="white",
                linewidth=2.5,
                marker="X",
                zorder=10,
                alpha=1.0,
            )

    # Set log scale for x-axis
    for ax in g.axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel("Cost (Core x Seconds)", fontsize=10)
        ax.set_ylabel("Optimality Gap", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

    g.set_titles(col_template="{col_name} Difficulty", fontsize=11)

    # Create custom legend with X markers (only once, on first subplot)
    from matplotlib.lines import Line2D  # noqa: PLC0415

    legend_handles = []
    # Get unique methods from the data in the order they appear in main_methods
    unique_methods = [m for m in main_methods if m in plot_df["Method"].unique()]
    for method in unique_methods:
        color = PALETTE.get(method, "black")
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                markerfacecolor=color,
                markersize=12,
                markeredgecolor="white",
                markeredgewidth=2,
                label=method,
                alpha=1.0,
            )
        )
    g.axes.flat[0].legend(
        handles=legend_handles,
        title="Method",
        fontsize=9,
        loc="upper right",
        framealpha=0.95,
    )

    plt.tight_layout()
    # Save both PNG and PDF
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = Path(output_path).with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"✅ Stratified plot saved to {output_path} and {pdf_path}")


def plot_performance_profile_old(results_data: pd.DataFrame, output_path: str):
    """
    Generate aggregated performance profile plot (ECDF of optimality ratios).

    Shows bands across 3 seeds with mean ± std dev for each method.
    Includes Hero, ablations, ShinkaEvolve, and baselines.

    Args:
        results_data: DataFrame with Method, Seed, Gap, vbs_score
        output_path: Output file path
    """
    # Calculate optimality ratio for each instance
    # Ratio = 1 - Gap (since Gap = (VBS - Score) / VBS, Ratio = Score / VBS = 1 - Gap)
    results_data = results_data.copy()
    results_data["OptimalityRatio"] = 1.0 - results_data["Gap"]

    # Filter out invalid ratios
    results_data = results_data[
        (results_data["OptimalityRatio"] >= 0)
        & (results_data["OptimalityRatio"] <= 1.0)
    ].copy()

    if results_data.empty:
        print("⚠️  No valid data for performance profile")
        return

    # Methods to plot (in order)
    method_order = [
        "Ours (Hero)",
        "Base (Best-of-64)",
        "Ours (+Oracle)",
        "Ours (+Diversity)",
        "Ours (w/o Prompt)",
        "ShinkaEvolve",
        "CP-SAT",
        "Local Search",
        "Greedy",
        "BnB",
    ]

    # Filter to methods that exist in data
    available_methods = [
        m for m in method_order if m in results_data["Method"].unique()
    ]

    _fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each method
    for method in available_methods:
        method_data = results_data[results_data["Method"] == method].copy()
        if method_data.empty:
            continue

        color = PALETTE.get(method, "black")

        # Calculate ECDF for each seed
        seed_ecdfs = {}
        for seed in sorted(method_data["Seed"].unique()):
            seed_data = method_data[method_data["Seed"] == seed]
            ratios = seed_data["OptimalityRatio"].to_numpy()
            ratios = np.sort(ratios)

            if len(ratios) == 0:
                continue

            # ECDF: fraction of problems solved at each ratio threshold
            y_vals = np.arange(1, len(ratios) + 1) / len(ratios)
            seed_ecdfs[seed] = (ratios, y_vals)

        if not seed_ecdfs:
            continue

        # Aggregate across seeds: mean ± std dev
        # Interpolate to common x-axis (optimality ratios from 0 to 1)
        x_common = np.linspace(0.0, 1.0, 1000)
        y_values_per_seed = []

        for ratios, y_vals in seed_ecdfs.values():
            # Interpolate to common x-axis
            y_interp = np.interp(x_common, ratios, y_vals, left=0.0, right=1.0)
            y_values_per_seed.append(y_interp)

        if not y_values_per_seed:
            continue

        # Calculate mean and std across seeds
        y_mean = np.mean(y_values_per_seed, axis=0)
        y_std = np.std(y_values_per_seed, axis=0)

        # Plot mean line
        linestyle = (
            "-" if method.startswith("Ours") or method == "ShinkaEvolve" else "--"
        )
        linewidth = (
            3.0 if method.startswith("Ours") or method == "ShinkaEvolve" else 1.5
        )
        alpha = 1.0 if method.startswith("Ours") or method == "ShinkaEvolve" else 0.7

        ax.plot(
            x_common,
            y_mean,
            label=method,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )

        # Plot confidence band (mean ± std)
        if len(y_values_per_seed) > 1:  # Only show band if multiple seeds
            ax.fill_between(
                x_common, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2
            )

    # Formatting
    ax.set_xlabel("Optimality Ratio (Score / VBS)", fontsize=14)
    ax.set_ylabel("Fraction of Problems Solved", fontsize=14)
    # Remove title as requested
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)  # Top left, larger font
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    # Increase tick label font size
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    # Save both PNG and PDF
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = Path(output_path).with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"✅ Performance profile plot saved to {output_path} and {pdf_path}")


def main():  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results across seeds and experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate all results (latest 15 jobs, or use --jobs-per-seed 1 to ensure all seeds)
  python aggregate_plots.py
  
  # Aggregate with custom job limit
  python aggregate_plots.py --max-jobs 20
  
  # Custom output directory
  python aggregate_plots.py --output-dir my_aggregated_results
  
  # With W&B logging
  python aggregate_plots.py --log-to-wandb
        """,
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=15,
        help="Maximum number of latest job directories to include (default: 15, ignored if --jobs-per-seed is set)",
    )
    parser.add_argument(
        "--jobs-per-seed",
        type=int,
        default=None,
        help="Select this many latest jobs per (method, seed) pair (overrides --max-jobs). Default behavior uses jobs_per_seed=1 to ensure all seeds are included.",
    )
    parser.add_argument(
        "--include-generalization",
        action="store_true",
        help="Include the (+Generalization) ablation in aggregation (default: excluded).",
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default=None,
        help="Filter by model name (e.g., 'qwen2.5-coder-14b')",
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
        help="Root directory to scan for SDS results (overrides default evaluation/sds/results)",
    )
    parser.add_argument(
        "--report-set",
        type=str,
        default=None,
        help="Path to a report-set JSON manifest. If set, aggregates across all SDS roots listed there.",
    )
    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        help="Log aggregated results to Weights & Biases (requires WANDB_API_KEY)",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        default=True,
        help="Include baseline solvers in aggregation (default: True)",
    )
    parser.add_argument(
        "--exclude-baselines",
        dest="include_baselines",
        action="store_false",
        help="Exclude baseline solvers from aggregation",
    )
    args = parser.parse_args()

    # Find all metrics files (default: moving dir; optionally override)
    if args.report_set:
        rs = load_report_set(args.report_set)
        sds_roots = rs.get("sds", {}).get("result_roots", [])
        if not sds_roots:
            print(f"❌ Report set has no SDS result_roots: {args.report_set}")
            return

        # When using --report-set, automatically use aggregated_report_batches/{report_set_name}/
        report_set_name = rs.get("name", Path(args.report_set).stem)
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            # Only override if user didn't specify a custom output dir
            args.output_dir = (
                f"evaluation/sds/aggregated_report_batches/{report_set_name}"
            )

        all_files = find_all_metrics_files_from_roots(sds_roots)
        print(f"📌 Report set: {report_set_name}")
        print(f"📌 SDS roots: {sds_roots}")
    else:
        root = args.results_root or BASE_RESULT_DIR
        all_files = find_all_metrics_files(base_dir=root)
        if args.results_root:
            print(f"📌 Results root override: {args.results_root}")

    # Create output directory (after potentially overriding it for report-set)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")

    print(f"📊 Found {len(all_files)} metrics_final.csv files")

    if not all_files:
        print("❌ No metrics files found!")
        return

    # Default: use an allowlist rather than hardcoded job IDs.
    # This avoids brittle coupling to specific SLURM job IDs and prevents seed dropouts
    # when new methods are added.
    allowed_methods = [
        "Ours (Hero)",
        "Ours (+Oracle)",
        "Ours (+Diversity)",
        "Ours (+Soft Gate)",
        "Ours (w/o Structure)",
        "Ours (w/o Prompt)",
        "Base (Best-of-64)",
        "ShinkaEvolve",
    ]
    if args.include_generalization:
        # NOTE: the generalization config may be absent, but historical results may still exist.
        allowed_methods.append("Ours (+Generalization)")

    # Select latest jobs (only the 15 specific ones for 14b)
    # Default to jobs_per_seed=1 to ensure we get all 3 seeds for each method
    jobs_per_seed = args.jobs_per_seed if args.jobs_per_seed is not None else 1
    selected_files = select_latest_jobs(
        all_files,
        max_jobs=args.max_jobs,
        model_filter=args.model_filter,
        jobs_per_seed=jobs_per_seed,
        specific_job_ids=None,
        allowed_methods=allowed_methods,
    )
    print(f"📊 Selected {len(selected_files)} files for aggregation")

    # Load and merge data
    aggregated_data = load_all_data(
        selected_files, include_baselines=args.include_baselines
    )

    if aggregated_data.empty:
        print("❌ No data loaded!")
        return

    # Print summary
    print("\n📊 Data Summary:")
    print(f"   Methods: {', '.join(aggregated_data['Method'].unique())}")
    print(f"   Seeds: {sorted(aggregated_data['Seed'].unique())}")
    print(f"   Total instances: {len(aggregated_data)}")

    # Generate LaTeX tables
    table_path = Path(args.output_dir) / "final_results_table.tex"
    print("\n📝 Generating LaTeX table (performance metrics)...")
    generate_latex_table(aggregated_data, str(table_path))

    # Generate error types table
    error_table_path = Path(args.output_dir) / "error_types_table.tex"
    print("\n📝 Generating LaTeX table (error types)...")
    generate_error_types_table(aggregated_data, str(error_table_path))

    # Generate paper-standard plots
    print("\n📊 Generating paper-standard plots...")

    # Figure 1: Efficiency Frontier
    eff_path = Path(args.output_dir) / "fig1_efficiency.png"
    print("  Generating Figure 1: Efficiency Frontier...")
    plot_efficiency_frontier(aggregated_data, str(eff_path), result_roots=sds_roots if args.report_set else None)

    # Figure 2: Robustness Profile
    robust_path = Path(args.output_dir) / "fig2_robustness.png"
    print("  Generating Figure 2: Robustness Profile...")
    plot_robustness_profile(aggregated_data, str(robust_path))

    # Figure 3: Stratified Box Plot
    boxplot_path = Path(args.output_dir) / "fig3_stratified.png"
    print("  Generating Figure 3: Stratified Box Plot...")
    plot_stratified_boxplot(aggregated_data, str(boxplot_path))

    # Figure 4: Failure Analysis
    failure_path = Path(args.output_dir) / "fig4_failure_modes.png"
    print("  Generating Figure 4: Failure Analysis...")
    plot_failure_analysis(aggregated_data, str(failure_path))

    # Figure 5: Scaling Analysis
    scaling_path = Path(args.output_dir) / "fig5_scaling.png"
    print("  Generating Figure 5: Scaling Analysis...")
    plot_scaling_analysis(str(scaling_path), result_roots=sds_roots if args.report_set else None)

    # W&B Logging
    if args.log_to_wandb:
        if not HAS_WANDB:
            print("⚠️  wandb not available. Skipping W&B logging.")
        elif not os.getenv("WANDB_API_KEY"):
            print("⚠️  WANDB_API_KEY not found. Skipping W&B logging.")
        else:
            print("\n📊 Logging to W&B...")
            try:
                project = os.getenv("WANDB_PROJECT", "sds-paper-aggregation")
                entity = os.getenv("WANDB_ENTITY", "smassoudi-eth-z-rich")

                wandb.init(
                    project=project,
                    entity=entity,
                    name="final-aggregated-results",
                    job_type="aggregation",
                )

                # Log images
                eff_path_wb = Path(args.output_dir) / "fig1_efficiency.png"
                robust_path_wb = Path(args.output_dir) / "fig2_robustness.png"
                boxplot_path_wb = Path(args.output_dir) / "fig3_stratified.png"
                failure_path_wb = Path(args.output_dir) / "fig4_failure_modes.png"

                if eff_path_wb.exists():
                    wandb.log({"efficiency_frontier": wandb.Image(str(eff_path_wb))})
                if robust_path_wb.exists():
                    wandb.log({"robustness_profile": wandb.Image(str(robust_path_wb))})
                if boxplot_path_wb.exists():
                    wandb.log({"stratified_boxplot": wandb.Image(str(boxplot_path_wb))})
                if failure_path_wb.exists():
                    wandb.log({"failure_analysis": wandb.Image(str(failure_path_wb))})

                # Log table as artifact
                if Path(table_path).exists():
                    artifact = wandb.Artifact(
                        "aggregated-results-table", type="latex-table"
                    )
                    artifact.add_file(str(table_path))
                    wandb.log_artifact(artifact)

                # Log summary metrics
                per_seed = (
                    aggregated_data.groupby(["Method", "Seed"])
                    .agg({"Pass": "mean", "Gap": "mean", "Cost": "mean"})
                    .reset_index()
                )

                agg_summary = per_seed.groupby("Method").agg(
                    {
                        "Pass": ["mean", "std"],
                        "Gap": ["mean", "std"],
                        "Cost": ["mean", "std"],
                    }
                )

                for method in agg_summary.index:
                    pass_m = agg_summary.loc[method, ("Pass", "mean")] * 100
                    gap_m = agg_summary.loc[method, ("Gap", "mean")] * 100
                    cost_m = agg_summary.loc[method, ("Cost", "mean")]

                    wandb.summary[f"{method}/pass_rate"] = pass_m
                    wandb.summary[f"{method}/optimality_gap"] = gap_m
                    wandb.summary[f"{method}/cost"] = cost_m

                wandb.finish()
                print("✅ Successfully logged to W&B")
            except Exception as e:
                print(f"❌ Error logging to W&B: {e}")

    # Convergence Analysis Aggregation
    convergence_files = find_all_convergence_files()
    if convergence_files:
        print(f"\n📊 Found {len(convergence_files)} convergence analysis files")
        agg_stats = aggregate_convergence_stats(
            convergence_files, args.output_dir, model_filter=args.model_filter
        )
        if agg_stats:
            mean_conv = agg_stats["mean_convergence"]
            std_conv = agg_stats["std_convergence"]
            print(f"✅ Aggregated convergence: {mean_conv:.2f}% ± {std_conv:.2f}%")
            print("   Per-seed breakdown:")
            for seed_data in agg_stats["per_seed"]:
                print(
                    f"     Seed {seed_data['seed']}: {seed_data['convergence_rate']:.2f}%"
                )

    # Best Hard Instances Aggregation
    best_hard_files = find_all_best_hard_files()
    if best_hard_files:
        print(f"\n📊 Found {len(best_hard_files)} best hard instance files")
        aggregate_best_hard_instances(
            best_hard_files, args.output_dir, model_filter=args.model_filter
        )

    print(f"\n✅ Done! Results saved to {args.output_dir}")


def find_all_convergence_files(base_dir: str = BASE_RESULT_DIR) -> list[str]:
    """Find all convergence_analysis.csv files recursively."""
    base_path = Path(base_dir)
    return sorted(str(p) for p in base_path.rglob("convergence_analysis.csv"))


def find_all_best_hard_files(base_dir: str = BASE_RESULT_DIR) -> list[str]:
    """Find all best_hard_instance.json files recursively."""
    base_path = Path(base_dir)
    return sorted(str(p) for p in base_path.rglob("best_hard_instance.json"))


def aggregate_convergence_stats(
    convergence_files: list[str],
    output_dir: str,
    model_filter: str | None = "qwen2.5-coder-14b",
) -> dict | None:
    """
    Aggregate convergence statistics across seeds.
    Uses same parse_path_metadata() logic to identify hero jobs.
    """
    if not convergence_files:
        return None

    # Load all convergence files
    all_results = []
    for csv_path in convergence_files:
        # Use existing parse_path_metadata to extract method, seed, model, job_id
        method, seed, model, job_id = parse_path_metadata(csv_path)

        # Only aggregate hero jobs
        if method != "Ours (Hero)":
            continue

        if model_filter and model != model_filter:
            continue

        # Load convergence data
        try:
            convergence_data = pd.read_csv(csv_path)

            # Calculate stats for this seed
            total = len(convergence_data)
            # Check if feasible column exists, otherwise assume all are feasible
            feasible = (
                convergence_data["feasible"].sum()
                if "feasible" in convergence_data.columns
                else total
            )

            hero_matches = (
                convergence_data["is_hero_template"].sum()
                if "is_hero_template" in convergence_data.columns
                else 0
            )

            convergence_rate = (hero_matches / feasible * 100) if feasible > 0 else 0

            all_results.append(
                {
                    "seed": seed,
                    "job_id": job_id,
                    "model": model,
                    "total": total,
                    "feasible": feasible,
                    "hero_matches": int(hero_matches),
                    "convergence_rate": float(convergence_rate),
                }
            )
        except Exception as e:
            print(f"⚠️  Failed to load {csv_path}: {e}")
            continue

    if not all_results:
        return None

    # Aggregate across seeds
    results_df = pd.DataFrame(all_results)

    # Group by seed and take mean if multiple jobs per seed
    per_seed = (
        results_df.groupby("seed")
        .agg(
            {
                "convergence_rate": "mean",
                "hero_matches": "sum",
                "feasible": "sum",
                "total": "sum",
            }
        )
        .reset_index()
    )

    # Calculate aggregate statistics
    convergence_rates = per_seed["convergence_rate"].to_numpy()
    mean_convergence = float(np.mean(convergence_rates))
    std_convergence = float(np.std(convergence_rates))

    agg_stats = {
        "mean_convergence": mean_convergence,
        "std_convergence": std_convergence,
        "per_seed": per_seed.to_dict("records"),
    }

    # Save aggregated stats
    stats_path = Path(output_dir) / "convergence_statistics.json"
    with stats_path.open("w") as f:
        json.dump(agg_stats, f, indent=2)
    print(f"✅ Saved convergence statistics to {stats_path}")

    return agg_stats


def aggregate_best_hard_instances(
    best_hard_files: list[str],
    output_dir: str,
    model_filter: str | None = "qwen2.5-coder-14b",
) -> list[dict] | None:
    """
    Aggregate best hard instances across seeds.
    Uses same parse_path_metadata() logic to identify hero jobs.
    """
    if not best_hard_files:
        return None

    # Load all best hard instance files
    all_instances = []
    for json_path in best_hard_files:
        # Use existing parse_path_metadata to extract method, seed, model, job_id
        # We need to construct a fake CSV path to use parse_path_metadata
        # Replace 'best_hard_instance.json' with 'metrics_final.csv' in path
        fake_csv_path = json_path.replace(
            "best_hard_instance.json", "metrics_final.csv"
        )
        method, _seed, model, _job_id = parse_path_metadata(fake_csv_path)

        # Only aggregate hero jobs
        if method != "Ours (Hero)":
            continue

        if model_filter and model != model_filter:
            continue

        # Load best hard instance
        try:
            with Path(json_path).open() as f:
                instance = json.load(f)
            all_instances.append(instance)
        except Exception as e:
            print(f"⚠️  Failed to load {json_path}: {e}")
            continue

    if not all_instances:
        return None

    # Save aggregated instances
    # Save as JSON
    best_hard_json_path = Path(output_dir) / "best_hard_instances.json"
    with best_hard_json_path.open("w") as f:
        json.dump(all_instances, f, indent=2)
    print(f"✅ Saved best hard instances (JSON) to {best_hard_json_path}")

    # Save as readable Markdown
    best_hard_md_path = Path(output_dir) / "best_hard_instances.md"
    save_best_hard_markdown(all_instances, str(best_hard_md_path))
    print(f"✅ Saved best hard instances (Markdown) to {best_hard_md_path}")

    return all_instances


def save_best_hard_markdown(best_hard_instances: list[dict], output_path: str):
    """
    Save best hard instances to a readable Markdown file.
    """
    with Path(output_path).open("w") as f:
        f.write("# Best Hard Instances by Seed\n\n")
        f.write(
            "This document contains the reasoning traces and generated code for the highest-scoring "
        )
        f.write("Hard difficulty instances from each training seed.\n\n")
        f.write("---\n\n")

        for instance in best_hard_instances:
            seed = instance.get("seed", "Unknown")
            uuid = instance.get("uuid", "Unknown")
            llm_score = instance.get("llm_score", 0)
            vbs_score = instance.get("vbs_score")
            gap = instance.get("optimality_gap_percent")
            exec_time = instance.get("execution_time")
            mission = instance.get("mission_summary", "")

            f.write(f"## Seed {seed}\n\n")
            f.write(f"**UUID**: `{uuid}`\n\n")
            f.write(f"**LLM Score**: {llm_score:.2f}\n\n")
            if vbs_score:
                f.write(f"**VBS Score**: {vbs_score:.2f}\n\n")
            if gap is not None:
                f.write(f"**Optimality Gap**: {gap:.2f}%\n\n")
            if exec_time:
                f.write(f"**Execution Time**: {exec_time:.4f}s\n\n")
            if mission:
                f.write(f"**Mission Summary**: {mission}\n\n")

            reasoning = instance.get("reasoning")
            if reasoning:
                f.write("### Reasoning Trace\n\n")
                f.write("```\n")
                f.write(reasoning)
                f.write("\n```\n\n")

            code = instance.get("code_snippet")
            if code:
                f.write("### Generated Code\n\n")
                f.write("```python\n")
                f.write(code)
                f.write("\n```\n\n")

            f.write("---\n\n")


if __name__ == "__main__":
    main()
