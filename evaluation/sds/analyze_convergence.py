#!/usr/bin/env python3
"""
Analyze algorithmic convergence in LLM-generated code.

This script:
1. Fetches generated code from W&B evaluation runs
2. Performs static analysis to detect Simulated Annealing template
3. Calculates convergence statistics (how many solutions match the Hero template)
4. Saves results to job directories for aggregation

Usage:
    python analyze_convergence.py [--job-dirs DIR ...] [--model MODEL] [--seeds SEED ...]
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Try to import optional dependencies
try:
    from dotenv import load_dotenv

    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

try:
    from wandb import Api

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("⚠️  wandb not available. Install with: pip install wandb")

# Load environment variables (for WANDB_API_KEY)
if HAS_DOTENV:
    load_dotenv()

# Also check cluster file (same as slurm scripts)
if not os.environ.get("WANDB_API_KEY") and HAS_WANDB:
    cluster_key_path = Path("~/llm/wandb_token.txt").expanduser()
    if cluster_key_path.exists():
        try:
            with cluster_key_path.open() as f:
                os.environ["WANDB_API_KEY"] = f.read().strip()
        except Exception as e:
            print(f"⚠️  Failed to load W&B key from cluster file: {e}")

# --- CONFIGURATION ---
BASE_RESULT_DIR = "evaluation/sds/results"

# Numerical constants for comparisons and thresholds
_MIN_TEMPERATURE = 100  # Minimum valid temperature for Simulated Annealing
_MIN_COOLING = 0.8  # Minimum valid cooling rate for SA
_MIN_ITERATIONS = 100  # Minimum valid iterations for SA
_TOP_N_DISTRIBUTIONS = 3  # Number of top values to show in distributions
_WANDB_PATH_PARTS = 3  # Expected number of parts in W&B path (entity/project/run_name)

# Hardcoded job IDs for hero runs (fallback if metadata not found)
# From aggregate_plots.py - the specific 15 job IDs for 14B hero runs
HERO_JOB_IDS_BY_SEED = {
    101: ["1315159", "1315160", "1315161", "1315162", "1315163"],
    202: ["1315164", "1315165", "1315166", "1315167", "1315168"],
    303: ["1315169", "1315170", "1315171", "1315172", "1315173"],
}

# Hero template parameters (exact match required)
HERO_TEMPLATE_PARAMS = {"T": 1000, "cooling": 0.99, "iters": 1000}


def find_hero_job_directories(
    base_dir: str = BASE_RESULT_DIR,
    model_filter: str = "qwen2.5-coder-14b",
    seeds: list[int] | None = None,
) -> list[str]:
    """
    Find hero job directories using experiment_metadata.json.

    Strategy:
    1. Scan for all experiment_metadata.json files
    2. Filter by method_name == "Ours (Hero)" AND config_name == "config_hero"
    3. Filter by model and seeds
    4. Return list of job directory paths
    """
    if seeds is None:
        seeds = [101, 202, 303]
    hero_dirs = []
    base_path = Path(base_dir)

    for metadata_path in base_path.rglob("experiment_metadata.json"):
        try:
            with metadata_path.open() as f:
                metadata = json.load(f)

            # Check if this is a hero config
            if (
                metadata.get("method_name") == "Ours (Hero)"
                and metadata.get("config_name") == "config_hero"
                and metadata.get("model") == model_filter
                and metadata.get("seed") in seeds
            ):
                job_dir = metadata_path.parent
                hero_dirs.append(str(job_dir))
        except Exception:
            continue

    return sorted(hero_dirs)


def find_hero_jobs_fallback(base_dir: str, model: str, seeds: list[int]) -> list[str]:
    """Fallback: construct paths from hardcoded job IDs."""
    hero_dirs = []
    base_path = Path(base_dir)
    for seed in seeds:
        for job_id in HERO_JOB_IDS_BY_SEED.get(seed, []):
            job_dir = base_path / model / "grpo" / f"seed{seed}" / f"job-{job_id}"
            if job_dir.exists():
                hero_dirs.append(str(job_dir))
    return hero_dirs


def find_hero_jobs(
    base_dir: str = BASE_RESULT_DIR,
    model: str = "qwen2.5-coder-14b",
    seeds: list[int] | None = None,
    use_fallback: bool = True,
) -> list[str]:
    """
    Find hero job directories with metadata-first, fallback-to-hardcoded strategy.
    """
    # Try metadata-based detection first
    if seeds is None:
        seeds = [101, 202, 303]
    hero_dirs = find_hero_job_directories(base_dir, model, seeds)

    if hero_dirs:
        print(f"✅ Found {len(hero_dirs)} hero job directories via metadata")
        return hero_dirs

    # Fallback to hardcoded job IDs if metadata not found
    if use_fallback:
        print("⚠️  No hero jobs found via metadata, using hardcoded job IDs")
        hero_dirs = find_hero_jobs_fallback(base_dir, model, seeds)
        if hero_dirs:
            print(f"✅ Found {len(hero_dirs)} hero job directories via fallback")
        else:
            print("❌ No hero job directories found even with fallback")

    return hero_dirs


def verify_hero_job(job_dir: str) -> tuple[bool, dict | None]:
    """
    Verify that a job directory is actually a hero job.
    Returns (is_hero, metadata_dict)
    """
    metadata_path = Path(job_dir) / "experiment_metadata.json"

    if metadata_path.exists():
        try:
            with metadata_path.open() as f:
                metadata = json.load(f)

            is_hero = (
                metadata.get("method_name") == "Ours (Hero)"
                and metadata.get("config_name") == "config_hero"
            )
        except Exception as e:
            print(f"⚠️  Failed to read metadata from {metadata_path}: {e}")
            return False, None
        else:
            return is_hero, metadata

    # If no metadata, check if path matches hero pattern
    if "/grpo/" in job_dir and "job-" in job_dir:
        # Could be hero, but not verified - warn user
        print(f"⚠️  No metadata found for {job_dir}, assuming hero based on path")
        return True, None

    return False, None


def construct_wandb_run_name_from_metadata(metadata: dict) -> str:
    """
    Construct W&B evaluation run name matching legacy conventions.
    Target format: qwen2.5-coder-14b-grpo-sds-seed202-job1315168
    (no -config-hero, no -eval suffix)
    """
    # 1. Normalize Model Name
    model = metadata.get("model", "qwen2.5-coder-14b")
    # Quick normalization without importing evaluate.py
    if "qwen2.5" in model.lower() and "14b" in model.lower():
        normalized_model = "qwen2.5-coder-14b"
    else:
        normalized_model = model.lower().split("/")[-1]

    # 2. Extract Components
    training_scheme = metadata.get("training_scheme", "grpo")
    seed = metadata.get("seed")
    job_id = metadata.get("job_id")

    seed_str = f"seed{seed}" if seed is not None else "seed42"
    job_str = f"job{job_id}" if job_id else ""

    # 3. Handle Scheme - strip 'config' for legacy runs
    # Legacy metadata might have "grpo-config_hero", but the run name is just "grpo"
    scheme_parts = training_scheme.lower().replace("_", "-").split("-")
    base_scheme_parts = []
    for part in scheme_parts:
        if part.startswith("config"):
            break
        base_scheme_parts.append(part)

    base_scheme = "-".join(base_scheme_parts) if base_scheme_parts else "grpo"

    # 4. Construct Name (Legacy runs rarely have '-eval' suffix in the title)
    # Result: qwen2.5-coder-14b-grpo-sds-seed202-job1315168
    if job_str:
        run_name = f"{normalized_model}-{base_scheme}-sds-{seed_str}-{job_str}"
    else:
        run_name = f"{normalized_model}-{base_scheme}-sds-{seed_str}"

    return run_name


def fetch_wandb_table(  # noqa: PLR0911, PLR0912, PLR0915
    run_path: str,
    table_name: str = "all_solutions",
    project: str | None = None,
    entity: str | None = None,
) -> pd.DataFrame | None:
    """
    Fetches artifacts from W&B. Handles auto-generated artifact names (e.g., run-id-all_solutions).
    The artifact name includes a random run ID (e.g., run-j0dlb7vd-all_solutions:v0), so we
    must search through artifacts to find the one containing the table_name.
    """
    if not HAS_WANDB:
        print("❌ wandb not available.")
        return None

    if not os.environ.get("WANDB_API_KEY"):
        print("❌ WANDB_API_KEY not found. Cannot fetch from W&B.")
        return None

    try:
        api = Api()

        # Defaults
        if not entity:
            entity = os.environ.get("WANDB_ENTITY", "smassoudi-eth-z-rich")

        # CRITICAL: Always check 'qwen-coder-sds-rl' for legacy runs, regardless of env var
        projects_to_try = []
        if project:
            projects_to_try.append(project)

        defaults = ["qwen-coder-sds-rl", "shinka-evolve"]
        for p in defaults:
            if p not in projects_to_try:
                projects_to_try.append(p)

        # Clean run name
        if "/" in run_path:
            # Full path: entity/project/run_name
            parts = run_path.split("/")
            if len(parts) == _WANDB_PATH_PARTS:
                entity = parts[0]
                project = parts[1]
                run_name = parts[2]
            else:
                run_name = run_path.split("/")[-1]
        else:
            run_name = run_path

        # Try name variants: "name", "name-eval", "name" (without -eval)
        candidates = [run_name]
        if not run_name.endswith("-eval"):
            candidates.append(f"{run_name}-eval")
        else:
            candidates.append(run_name[:-5])

        run = None

        # --- STEP 1: Find the Run ---
        print(f"🔎 Searching for run '{run_name}'...")
        for proj in projects_to_try:
            if run:
                break

            # Method A: Direct lookup of candidates
            for candidate in candidates:
                try:
                    full_path = f"{entity}/{proj}/{candidate}"
                    run = api.run(full_path)
                    print(f"   ✅ Found run in project '{proj}': {full_path}")
                    break
                except Exception:
                    pass

            # Method B: Search by Job ID (Robust fallback)
            if not run:
                job_match = re.search(r"job(\d+)", run_name)
                if job_match:
                    job_id = job_match.group(1)
                    try:
                        runs = api.runs(
                            f"{entity}/{proj}",
                            filters={
                                "display_name": {"$regex": f"job{job_id}"},
                                "state": "finished",
                            },
                        )
                        for r in runs:
                            if "sds" in r.name:
                                run = r
                                print(
                                    f"   ✅ Found run via Job ID {job_id} in '{proj}': {r.name}"
                                )
                                break
                        if run:
                            break
                    except Exception:
                        continue

        if not run:
            print(f"i  W&B run not found (expected for old jobs): {run_name}")
            return None

        # --- STEP 2: Find the Artifact (The Fix for 'run-j0dlb7vd-all_solutions') ---
        try:
            artifacts = list(run.logged_artifacts())
            table_artifact = None

            # Search strategy: look for 'all_solutions' inside the artifact name
            # The artifact is likely named 'run-<id>-all_solutions:v0'
            for artifact in artifacts:
                if table_name in artifact.name:
                    table_artifact = artifact
                    break

            if not table_artifact:
                print(
                    f"⚠️  Run found, but artifact containing '{table_name}' is missing."
                )
                print(f"   Available artifacts: {[a.name for a in artifacts]}")
                return None

            print(f"📦 Downloading artifact: {table_artifact.name}...")
            artifact_dir = table_artifact.download()

            # Locate the JSON file inside the artifact directory
            # It's usually 'all_solutions.table.json'
            artifact_path = Path(artifact_dir)
            json_files = [f for f in artifact_path.iterdir() if f.suffix == ".json"]

            if not json_files:
                print("⚠️  No .json file found inside artifact folder.")
                return None

            # If multiple JSONs, prefer the one with table_name in it
            target_file = json_files[0]
            for f in json_files:
                if table_name in f.name:
                    target_file = f
                    break

            table_path = target_file

            with table_path.open() as f:
                data = json.load(f)

            solutions_data = pd.DataFrame(data["data"], columns=data["columns"])
            print(f"✅ Loaded {len(solutions_data)} rows from W&B table.")
        except Exception as e:
            print(f"⚠️  Error fetching W&B data: {e}")
            return None
        else:
            return solutions_data

    except Exception as e:
        # This is expected for old jobs where W&B runs may not exist
        # The script will fall back to local CSV files
        error_msg = str(e)
        if "not found" in error_msg.lower():
            # Extract run name from error message (format: <Run entity/project/run-name (not found)>)
            # Try to extract just the run name for cleaner output
            # Note: re is already imported at module level
            match = re.search(r"/([^/\s]+)\s*\(", error_msg)
            if match:
                run_name = match.group(1)
                print(f"i  W&B run not found (expected for old jobs): {run_name}")
            else:
                print(
                    "i  W&B run not found (expected for old jobs, will use local CSV)"
                )
        else:
            print(f"⚠️  Error fetching from W&B: {e}")
        return None


def analyze_code_structure(code: str) -> dict:
    """
    Performs static analysis to detect the 'Hero Template' signature.

    Returns:
        Dictionary with extracted hyperparameters and structural flags
    """
    if not code or not isinstance(code, str):
        return {
            "T": None,
            "cooling": None,
            "iters": None,
            "has_constraint_guard": False,
            "has_metropolis": False,
            "has_imports": False,
        }

    # 1. Regex Extraction for Hyperparameters
    # Handle both "T = 1000" and "temperature = 1000"
    t_match = re.search(r"\b(?:T|temperature)\s*=\s*(\d+)", code, re.IGNORECASE)

    # Handle cooling_rate (allow slight variations like 0.995 vs 0.99)
    cool_match = re.search(r"cooling_rate\s*=\s*([0-9.]+)", code, re.IGNORECASE)

    # Handle iterations - could be "n_iterations", "iterations", "max_iter", or inferred from loop
    iter_match = re.search(
        r"(?:n_?iterations|iterations|max_iter)\s*=\s*(\d+)", code, re.IGNORECASE
    )

    # If no explicit iteration variable, try to infer from "for _ in range(...)" patterns
    if not iter_match:
        # Look for common iteration patterns
        range_matches = re.findall(r"for\s+[^:]+in\s+range\s*\(\s*(\d+)\s*\)", code)
        if range_matches:
            # Take the largest range value (likely the main iteration loop)
            max_range = max(range_matches, key=int)
            iter_match = type(
                "obj", (object,), {"group": lambda _self, _n: max_range}
            )()

    # 2. Structural Component Checks
    has_imports = "import random" in code and "import math" in code

    # The "Constraint Guard" - Rejection sampling inside the loop
    # We look for "while not is_feasible" pattern
    has_constraint_guard = bool(re.search(r"while\s+not\s+is_feasible", code))

    # The Metropolis Acceptance Criterion
    # Checks for exp(delta/T) logic - handle both T and temperature variable names
    has_metropolis = (
        ("math.exp" in code)
        and ("random.random()" in code)
        and (
            "/ T" in code
            or "/ temperature" in code
            or "/T" in code
            or "/temperature" in code
        )
    )

    # 3. Compile Data
    temp_val = int(t_match.group(1)) if t_match else None
    cooling_val = float(cool_match.group(1)) if cool_match else None
    iters_val = int(iter_match.group(1)) if iter_match else None

    return {
        "T": temp_val,
        "cooling": cooling_val,
        "iters": iters_val,
        "has_constraint_guard": has_constraint_guard,
        "has_metropolis": has_metropolis,
        "has_imports": has_imports,
    }


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


def extract_best_hard_instance(df: pd.DataFrame, metadata: dict | None) -> dict | None:
    """
    Extract the best Hard instance (highest llm_score) for this seed.

    Returns:
        Dictionary with uuid, reasoning, code_snippet, and metadata
    """
    if "difficulty_class" not in df.columns:
        return None

    hard_df = df[df["difficulty_class"] == "Hard"].copy()
    if len(hard_df) == 0:
        return None

    # Find the instance with highest llm_score
    best_idx = hard_df["llm_score"].idxmax()
    best_row = hard_df.loc[best_idx]

    # Calculate optimality gap if vbs_score is available
    optimality_gap = None
    if "vbs_score" in best_row and pd.notna(best_row["vbs_score"]):
        vbs_score = best_row["vbs_score"]
        llm_score = best_row["llm_score"]
        if vbs_score > 0:
            optimality_gap = (vbs_score - llm_score) / vbs_score * 100

    result = {
        "seed": metadata.get("seed") if metadata else None,
        "job_id": metadata.get("job_id") if metadata else None,
        "uuid": str(best_row["uuid"]),
        "difficulty_class": "Hard",
        "llm_score": float(best_row["llm_score"]),
        "vbs_score": float(best_row["vbs_score"])
        if "vbs_score" in best_row and pd.notna(best_row["vbs_score"])
        else None,
        "optimality_gap_percent": optimality_gap,
        "execution_time": float(best_row["execution_time"])
        if "execution_time" in best_row and pd.notna(best_row["execution_time"])
        else None,
        "mission_summary": str(best_row["mission_summary"])
        if "mission_summary" in best_row
        else None,
        "reasoning": str(best_row["reasoning"])
        if "reasoning" in best_row and pd.notna(best_row["reasoning"])
        else None,
        "code_snippet": str(best_row["code_snippet"])
        if "code_snippet" in best_row and pd.notna(best_row["code_snippet"])
        else None,
    }

    return result


def check_hero_template(meta: dict) -> bool:
    """
    Check if code matches the Hero Algorithmic Template.

    DEFINITION:
    The "Hero Template" is defined by its STRUCTURE, not specific hyperparameters.
    Different seeds converged to different (but valid) local optima for hyperparameters,
    but all discovered the same algorithmic class: Simulated Annealing with Constraint Guard.

    Core Invariants (The "DNA" of the Hero Strategy):
    1. Constraint Guard: Rejection sampling inside neighbor loop (unique to RL)
    2. Metropolis Criterion: Probabilistic acceptance exp(delta/T)
    3. Annealing Schedule: Valid cooling schedule (T *= cooling)

    We DO NOT enforce specific hyperparameter values (T=1000, cooling=0.99, etc.)
    because different seeds converged to different but valid local optima.
    """
    # 1. Structural Components (The "DNA" - Required)
    # These are the core algorithmic features that define the Hero strategy
    if not (meta["has_constraint_guard"] and meta["has_metropolis"]):
        return False

    # 2. Temperature Check
    # Just check if T exists and is a reasonable starting temp for this problem scale
    # Typical SA uses T in hundreds to thousands range
    if meta["T"] is None or meta["T"] < _MIN_TEMPERATURE:
        return False

    # 3. Cooling Check
    # Check if cooling exists and is a valid decay factor (0 < c < 1)
    # Typical SA range is 0.8 to 0.999
    if meta["cooling"] is None:
        return False
    if not (_MIN_COOLING <= meta["cooling"] < 1.0):
        return False

    # 4. Iteration Check
    # Either explicit iterations or a "while T > X" loop (which we infer)
    # We don't strictly require 'iters' to be detected if the loop structure is there
    # But if we did detect 'iters', ensure it's substantial (not a trivial loop)
    return not (meta["iters"] is not None and meta["iters"] < _MIN_ITERATIONS)


def analyze_job_convergence(  # noqa: PLR0912, PLR0915
    job_dir: str,
    project: str | None = None,
    entity: str | None = None,
    skip_wandb: bool = False,
) -> dict | None:
    """
    Analyze convergence for a single job directory.

    Returns:
        Dictionary with convergence statistics, or None if failed
    """
    print(f"\n📊 Analyzing: {job_dir}")

    # Verify this is a hero job
    is_hero, metadata = verify_hero_job(job_dir)
    if not is_hero:
        print(f"⚠️  Skipping {job_dir}: not a hero job")
        return None

    # Try to fetch from W&B
    metrics_data = None
    if not skip_wandb and HAS_WANDB and metadata:
        # Construct W&B run name
        try:
            run_name = construct_wandb_run_name_from_metadata(metadata)
            metrics_data = fetch_wandb_table(
                run_name, table_name="all_solutions", project=project, entity=entity
            )
        except Exception as e:
            print(f"⚠️  Failed to construct W&B run name: {e}")

    # Fallback: try to load from local metrics_final.csv if available
    if metrics_data is None:
        metrics_path = Path(job_dir) / "metrics_final.csv"
        if metrics_path.exists():
            print(f"📁 Loading from local CSV: {metrics_path}")
            metrics_data = pd.read_csv(metrics_path)
            # Check if code_snippet column exists
            if "code_snippet" not in metrics_data.columns:
                print("⚠️  No 'code_snippet' column in metrics_final.csv")
                return None
        else:
            print(f"❌ No W&B data and no local CSV found for {job_dir}")
            return None

    # Filter for feasible solutions
    feasible_df = (
        metrics_data[metrics_data["feasible"]].copy()
        if "feasible" in metrics_data.columns
        else metrics_data.copy()
    )

    print(f"🔬 Analyzing {len(feasible_df)} feasible solutions...")

    # Apply static analysis
    analysis_results = []

    for idx, row in tqdm(
        feasible_df.iterrows(), total=len(feasible_df), desc="Analyzing code"
    ):
        code = row.get("code_snippet", "")
        if not code:
            continue

        meta = analyze_code_structure(code)

        # Check if matches hero template
        is_hero_template = check_hero_template(meta)
        meta["is_hero_template"] = is_hero_template
        meta["uuid"] = row.get("uuid", idx)
        meta["feasible"] = True  # We already filtered
        analysis_results.append(meta)

    if not analysis_results:
        print("⚠️  No code snippets found to analyze")
        return None

    results_df = pd.DataFrame(analysis_results)

    # Compute statistics
    total = len(results_df)
    hero_count = results_df["is_hero_template"].sum()
    convergence_rate = (hero_count / total) * 100 if total > 0 else 0

    print(f"\n{'=' * 50}")
    print("CONVERGENCE ANALYSIS REPORT")
    print(f"{'=' * 50}")
    print(f"Total Solutions Analyzed: {total}")
    print(f"Matches Hero Template:    {hero_count}")
    print(f"Convergence Rate:         {convergence_rate:.2f}%")
    print(f"{'-' * 50}")

    # Breakdown of components (structural vs parametric)
    print("Component Compliance (Structural):")
    print(
        f" - Constraint Guard: {results_df['has_constraint_guard'].mean() * 100:.1f}%"
    )
    print(f" - Metropolis Logic: {results_df['has_metropolis'].mean() * 100:.1f}%")

    # Hyperparameter validation (flexible - just check they're reasonable)
    temp_ok = results_df["T"].apply(lambda x: x is not None and x >= _MIN_TEMPERATURE)
    print(f" - Valid Temperature (T≥{_MIN_TEMPERATURE}): {temp_ok.mean() * 100:.1f}%")

    cooling_ok = results_df["cooling"].apply(
        lambda x: x is not None and _MIN_COOLING <= x < 1.0
    )
    print(f" - Valid Cooling ({_MIN_COOLING}≤c<1.0): {cooling_ok.mean() * 100:.1f}%")

    iters_ok = results_df["iters"].apply(lambda x: x is None or x >= _MIN_ITERATIONS)
    print(
        f" - Valid Iterations (≥{_MIN_ITERATIONS} or dynamic): {iters_ok.mean() * 100:.1f}%"
    )

    # Show actual hyperparameter distributions (this is the interesting part!)
    print("\nHyperparameter Distributions (Seed-Specific Local Optima):")
    if results_df["T"].notna().any():
        temp_dist = results_df["T"].value_counts().head(_TOP_N_DISTRIBUTIONS)
        print(f" - Temperature: {dict(temp_dist)}")
    if results_df["cooling"].notna().any():
        cooling_dist = results_df["cooling"].value_counts().head(_TOP_N_DISTRIBUTIONS)
        print(f" - Cooling rates: {dict(cooling_dist)}")
    if results_df["iters"].notna().any():
        iters_dist = results_df["iters"].value_counts().head(_TOP_N_DISTRIBUTIONS)
        print(f" - Iterations: {dict(iters_dist)}")
    else:
        print(" - Iterations: Dynamic (while T > threshold)")

    # Save detailed results
    csv_path = Path(job_dir) / "convergence_analysis.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Detailed results saved to {csv_path}")

    # Save summary
    summary = {
        "seed": metadata.get("seed") if metadata else None,
        "job_id": metadata.get("job_id") if metadata else None,
        "total_solutions": total,
        "feasible_solutions": total,  # Already filtered
        "hero_template_matches": int(hero_count),
        "convergence_rate": float(convergence_rate),
        "component_compliance": {
            "constraint_guard": float(results_df["has_constraint_guard"].mean() * 100),
            "metropolis": float(results_df["has_metropolis"].mean() * 100),
            "valid_temperature": float(
                results_df["T"]
                .apply(lambda x: x is not None and x >= _MIN_TEMPERATURE)
                .mean()
                * 100
            ),
            "valid_cooling": float(
                results_df["cooling"]
                .apply(lambda x: x is not None and _MIN_COOLING <= x < 1.0)
                .mean()
                * 100
            ),
            "valid_iterations": float(
                results_df["iters"]
                .apply(lambda x: x is None or x >= _MIN_ITERATIONS)
                .mean()
                * 100
            ),
        },
        "hyperparameter_distributions": {
            "temperature": dict(
                results_df["T"].value_counts().head(_TOP_N_DISTRIBUTIONS).to_dict()
            )
            if results_df["T"].notna().any()
            else {},
            "cooling": dict(
                results_df["cooling"]
                .value_counts()
                .head(_TOP_N_DISTRIBUTIONS)
                .to_dict()
            )
            if results_df["cooling"].notna().any()
            else {},
            "iterations": dict(
                results_df["iters"].value_counts().head(_TOP_N_DISTRIBUTIONS).to_dict()
            )
            if results_df["iters"].notna().any()
            else {},
        },
    }

    summary_path = Path(job_dir) / "convergence_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to {summary_path}")

    # Extract best Hard instance (highest llm_score where difficulty_class == "Hard")
    best_hard = extract_best_hard_instance(feasible_df, metadata)
    if best_hard:
        best_hard_path = Path(job_dir) / "best_hard_instance.json"
        with best_hard_path.open("w") as f:
            json.dump(best_hard, f, indent=2)
        print(f"✅ Best Hard instance saved to {best_hard_path}")
        summary["best_hard_instance"] = {
            "uuid": best_hard["uuid"],
            "llm_score": best_hard["llm_score"],
            "vbs_score": best_hard.get("vbs_score"),
            "optimality_gap": best_hard.get("optimality_gap"),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze algorithmic convergence in LLM-generated code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect hero jobs and analyze
  python analyze_convergence.py
  
  # Analyze specific job directories
  python analyze_convergence.py --job-dirs path/to/job1 path/to/job2
  
  # Custom model and seeds
  python analyze_convergence.py --model qwen2.5-coder-14b --seeds 101 202 303
  
  # Skip W&B and use local CSV only
  python analyze_convergence.py --skip-wandb
        """,
    )

    parser.add_argument(
        "--job-dirs",
        nargs="+",
        default=None,
        help="Specific job directories to analyze (default: auto-detect hero jobs)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-coder-14b",
        help="Model name for hero detection (default: qwen2.5-coder-14b)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[101, 202, 303],
        help="Seeds to analyze (default: 101, 202, 303)",
    )
    parser.add_argument(
        "--skip-fallback",
        action="store_true",
        help="Skip hardcoded job ID fallback if metadata not found",
    )
    parser.add_argument(
        "--skip-wandb",
        action="store_true",
        help="Skip W&B fetch, use local CSV if available",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project (default: from env or 'qwen-coder-sds-rl')",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (default: from env or 'smassoudi-eth-z-rich')",
    )

    args = parser.parse_args()

    # Determine which job directories to analyze
    if args.job_dirs:
        # Manual specification
        job_dirs = [str(Path(d).resolve()) for d in args.job_dirs]
        print(f"📁 Using {len(job_dirs)} manually specified job directories")
    else:
        # Auto-detect hero jobs
        job_dirs = find_hero_jobs(
            base_dir=BASE_RESULT_DIR,
            model=args.model,
            seeds=args.seeds,
            use_fallback=not args.skip_fallback,
        )

        if not job_dirs:
            print("❌ No hero job directories found. Specify --job-dirs manually.")
            return

    # Process each job directory
    summaries = []
    best_hard_instances = []
    for job_dir in job_dirs:
        summary = analyze_job_convergence(
            job_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            skip_wandb=args.skip_wandb,
        )
        if summary:
            summaries.append(summary)
            # Load best hard instance if it was saved
            best_hard_path = Path(job_dir) / "best_hard_instance.json"
            if best_hard_path.exists():
                with best_hard_path.open() as f:
                    best_hard_instances.append(json.load(f))

    # Print aggregate summary
    if summaries:
        print(f"\n{'=' * 50}")
        print(f"AGGREGATE SUMMARY ({len(summaries)} seeds)")
        print(f"{'=' * 50}")

        convergence_rates = [s["convergence_rate"] for s in summaries]
        mean_rate = np.mean(convergence_rates)
        std_rate = np.std(convergence_rates)

        print(f"Mean Convergence Rate: {mean_rate:.2f}% ± {std_rate:.2f}%")
        print("\nPer-seed breakdown:")
        for s in summaries:
            print(
                f"  Seed {s['seed']}: {s['convergence_rate']:.2f}% ({s['hero_template_matches']}/{s['total_solutions']})"
            )

    # Note: Best Hard instances are saved per-job directory
    # Aggregation happens in aggregate_plots.py for consistency

    print(f"\n✅ Done! Analyzed {len(summaries)} job directories.")


if __name__ == "__main__":
    main()
