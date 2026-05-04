import argparse
import contextlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

# Try to import datasets for HuggingFace support
try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Import local utils
try:
    from utils import (
        check_constraint_violations,
        deserialize_mission,
        mission_to_instance,
        run_candidate,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from utils import (
        check_constraint_violations,
        deserialize_mission,
        mission_to_instance,
        run_candidate,
    )

from syndeopt.core.feasibility import feasible
from syndeopt.core.scoring import score
from syndeopt.solvers import get_solver


# --- CUSTOM EXCEPTIONS ---
class DatasetLoadError(RuntimeError):
    """Raised when dataset loading fails."""

    def __init__(self, dataset_name: str, original_error: Exception):
        msg = f"Failed to load dataset {dataset_name}: {original_error}"
        super().__init__(msg)
        self.dataset_name = dataset_name
        self.original_error = original_error


class CodeExtractionError(RuntimeError):
    """Raised when code extraction from dataset fails."""

    def __init__(self):
        msg = "No code solutions extracted from ShinkaEvolve dataset!"
        super().__init__(msg)


# --- CONFIGURATION ---
# Magic value constants
_EPSILON_SMALL = 1e-9
_EPSILON_TINY = 1e-10
_EPSILON_MEDIUM = 1e-6
_EPSILON_EQUALITY = 1e-4
_TRIVIAL_THRESHOLD = 0.01
_MODERATE_THRESHOLD = 0.10
_DEFAULT_SEED = 42
_DEFAULT_SHINKA_SEED = 303

SOLVER_CORES = {
    "greedy": 1,
    "local_search": 1,
    "bnb": 1,
    "cpsat": 8,  # CP-SAT uses multiple cores
}

AVAILABLE_SOLVERS = {
    "greedy": "greedy",
    "local_search": "local_search",
    "cpsat": "cpsat",  # Exact (OR-Tools)
    "bnb": "bnb",  # Exact (Algorithmic)
}

DEFAULT_FIXED_CODE_METHOD_NAME = "Fixed Code"
DEFAULT_FIXED_CODE_LABEL = "fixed-code"


def calculate_true_score(instance, selected_ids):
    if not selected_ids:
        return 0.0
    score = sum(instance.w[i] for i in selected_ids)
    sel_set = set(selected_ids)
    for (i, j), weight in instance.W.items():
        if i in sel_set and j in sel_set:
            score += weight
    return score


def build_dataset_records(dataset_name: str, split: str = "test") -> list[str]:
    """Load an SDS dataset split and convert it into evaluator input records."""
    if not HAS_DATASETS:
        raise ImportError(
            "datasets library not available. Install with: pip install datasets"
        )

    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        raise DatasetLoadError(dataset_name, e) from e

    records = []
    for item in dataset:
        records.append(
            json.dumps(
                {
                    "uuid": item.get("uuid"),
                    "mission": item.get("mission"),
                    "generated_text": "",
                }
            )
        )
    return records


def sanitize_label(value: str | None, default: str) -> str:
    """Make a human-readable label filesystem- and metadata-safe."""
    if not value:
        return default
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or default


def determine_run_type(args) -> str:
    """Return a stable run-type tag for timing summaries."""
    if args.fixed_code_file:
        return "fixed-code"
    if args.best_of_n:
        return "best-of-n"
    if args.shinka_dataset:
        return "shinka-eval"
    return "llm-generate-and-eval"


def infer_method_name(args, extracted_method_name: str | None) -> str | None:
    """Return the preferred method name for metadata and downstream outputs."""
    if args.method_name_override:
        return args.method_name_override
    if args.shinka_dataset:
        return "ShinkaEvolve"
    return extracted_method_name


def write_timing_summary(output_dir: str, summary: dict) -> Path:
    """Persist run timing metadata for downstream aggregation."""
    output_path = Path(output_dir) / "timing_summary.json"
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return output_path


# --- STATISTICAL ANALYSIS CLASS FOR BEST-OF-N ---
class PassAtKAnalyzer:
    """
    Performs Pass@k and BestScore@k analysis using bootstrapping.
    This avoids re-running generation for every k.
    """

    def __init__(self, df, k_values=None):
        if k_values is None:
            k_values = [1, 2, 4, 8, 16, 32, 64]
        self.df = df
        max_samples_per_uuid = (
            df.groupby("uuid").size().max() if "uuid" in df.columns else 1
        )
        self.k_values = [k for k in k_values if k <= max_samples_per_uuid]
        self.results = {}

    def bootstrap_metrics(self, n_bootstraps=500):
        """
        For each k, sample k items per UUID n_bootstraps times and compute metrics.
        """
        print(f"📊 Bootstrapping Pass@k and Best@k analysis (n={n_bootstraps})...")

        grouped = self.df.groupby("uuid")
        metrics_by_k = {k: {"pass_rate": [], "opt_gap": []} for k in self.k_values}

        for k in tqdm(self.k_values, desc="Bootstrapping k"):
            for _ in range(n_bootstraps):
                iter_pass = []
                iter_gap = []

                for _uuid, group in grouped:
                    n_avail = len(group)
                    if n_avail < k:
                        continue  # Skip if not enough samples

                    sample = group.sample(n=k, replace=False)

                    # 1. Did we pass? (At least one feasible solution)
                    is_passed = sample["feasible"].any()
                    iter_pass.append(1 if is_passed else 0)

                    # 2. What is the best score?
                    # Get scores of FEASIBLE solutions only
                    feasible_scores = sample[sample["feasible"]]["llm_score"]

                    if not feasible_scores.empty:
                        best_score = feasible_scores.max()
                    else:
                        best_score = 0.0  # Default for failure

                    # Calculate gap against the row's VBS (use temp_vbs if available, otherwise vbs_score)
                    vbs_col = (
                        "temp_vbs" if "temp_vbs" in sample.columns else "vbs_score"
                    )
                    if vbs_col in sample.columns:
                        vbs = sample.iloc[0][vbs_col]  # VBS is constant for the problem
                        if vbs > _EPSILON_SMALL:
                            gap = (vbs - max(0, best_score)) / vbs
                            iter_gap.append(max(0, gap))  # Clip negative gaps

                if iter_pass:
                    metrics_by_k[k]["pass_rate"].append(np.mean(iter_pass))
                if iter_gap:
                    metrics_by_k[k]["opt_gap"].append(np.mean(iter_gap))

        # Aggregate mean/std
        final_stats = [
            {
                "k": k,
                "pass_rate_mean": np.mean(metrics_by_k[k]["pass_rate"]) * 100,
                "pass_rate_std": np.std(metrics_by_k[k]["pass_rate"]) * 100,
                "opt_gap_mean": np.mean(metrics_by_k[k]["opt_gap"]) * 100
                if metrics_by_k[k]["opt_gap"]
                else 0.0,
                "opt_gap_std": np.std(metrics_by_k[k]["opt_gap"]) * 100
                if metrics_by_k[k]["opt_gap"]
                else 0.0,
            }
            for k in self.k_values
            if metrics_by_k[k]["pass_rate"]
        ]

        return pd.DataFrame(final_stats)

    def plot_scaling_laws(self, stats_df, output_dir):
        """Generates the 'Reasoning Capacity' plot similar to Yue et al."""
        if len(stats_df) == 0:
            print("⚠️  No scaling stats to plot")
            return

        # Plot 1: Optimality Gap vs k (Log Scale)
        plt.figure(figsize=(5, 4))
        plt.errorbar(
            stats_df["k"],
            stats_df["opt_gap_mean"],
            yerr=stats_df["opt_gap_std"],
            fmt="-o",
            color="#d62728",
            capsize=3,
            label="Optimality Gap",
        )

        plt.xscale("log", base=2)
        plt.xlabel("Number of Generations (k)")
        plt.ylabel("Optimality Gap (%)")
        plt.title("Capacity Scaling: Base Model")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.xticks(stats_df["k"], labels=[str(int(k)) for k in stats_df["k"]])
        plt.tight_layout()
        output_path = Path(output_dir)
        plt.savefig(output_path / "scaling_gap_vs_k.png", dpi=300)
        plt.savefig(output_path / "scaling_gap_vs_k.pdf", dpi=300)
        plt.close()

        # Plot 2: Pass Rate vs k
        plt.figure(figsize=(5, 4))
        plt.errorbar(
            stats_df["k"],
            stats_df["pass_rate_mean"],
            yerr=stats_df["pass_rate_std"],
            fmt="-o",
            color="#1f77b4",
            capsize=3,
            label="Pass Rate",
        )

        plt.xscale("log", base=2)
        plt.xlabel("Number of Generations (k)")
        plt.ylabel("Pass Rate (%)")
        plt.title("Coverage Scaling: Base Model")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.xticks(stats_df["k"], labels=[str(int(k)) for k in stats_df["k"]])
        plt.tight_layout()
        plt.savefig(output_path / "scaling_pass_vs_k.png", dpi=300)
        plt.savefig(output_path / "scaling_pass_vs_k.pdf", dpi=300)
        plt.close()


def robust_execution(code, stdin_obj, n_repeats=3):
    """Run code multiple times for stable timing, return best time."""
    # 1. Validation Run
    res = run_candidate(code, stdin_obj)

    if "error" in res or "selection" not in res:
        res["execution_time"] = 0.0
        return res

    times = [res.get("execution_time", 0.0)]

    # 2. Timing Runs
    for _ in range(n_repeats - 1):
        retry_res = run_candidate(code, stdin_obj)
        if "execution_time" in retry_res:
            times.append(retry_res["execution_time"])

    res["execution_time"] = min(times)
    return res


# --- WORKER FUNCTION FOR PARALLELIZATION ---
def evaluate_single_sample(  # noqa: PLR0912, PLR0913, PLR0915
    line,
    idx,
    active_baselines_config,
    time_budget,
    repeats,
    master_seed,
    fixed_code=None,
):
    """
    Process a single line (worker function for parallel execution).

    Args:
        line: JSON line from input file
        idx: Index of the problem (used to derive unique seed)
        active_baselines_config: Dict mapping baseline names to solver keys
        time_budget: Time budget for solvers
        repeats: Number of execution repeats for LLM code
        master_seed: Global seed from args

    Returns:
        Dictionary with evaluation results
    """
    # IMPORT LOCAL SOLVER TO ENSURE PROCESS SAFETY

    # Derive unique seed for this instance (ensures reproducibility per-problem)
    current_seed = master_seed + idx

    record = json.loads(line)
    inst = mission_to_instance(record["mission"])
    mission_dict = deserialize_mission(record["mission"])

    row = {"uuid": record.get("uuid")}

    # 1. Run Baselines
    base_metrics = {}
    best_known_score = -float("inf")

    for name, solver_key in active_baselines_config.items():
        solver = get_solver(solver_key)
        cores_used = SOLVER_CORES.get(name, 1)

        start_b = time.time()
        try:
            # Pass the derived seed for reproducibility
            res = solver.solve(inst, budget_sec=time_budget, seed=current_seed)

            # Validate result (matching bench/runner.py logic)
            # This ensures we catch infeasible solutions and recalculate scores correctly
            is_feasible = feasible(inst, res.mask)

            # [FIX] Use distinct variable name 'solver_score' to avoid shadowing the imported 'score' function
            validated_score = score(inst, res.mask) if is_feasible else float("-inf")

            solver_score = validated_score
            wall_time = res.time_sec if res.time_sec else (time.time() - start_b)
            core_seconds = wall_time * cores_used
            baseline_feasible = is_feasible
        except Exception:
            solver_score = 0.0
            wall_time = 0.0
            core_seconds = 0.0
            baseline_feasible = False

        base_metrics[name] = {
            "score": solver_score,
            "time": wall_time,
            "core_sec": core_seconds,
            "feasible": baseline_feasible,
        }
        row[f"score_{name}"] = solver_score
        row[f"time_{name}"] = wall_time
        row[f"core_sec_{name}"] = core_seconds
        row[f"feasible_{name}"] = (
            baseline_feasible  # Track feasibility for each baseline
        )
        best_known_score = max(best_known_score, solver_score)

    # 2. Run LLM Code
    generated_text = record.get("generated_text", "")
    code_match = None
    reasoning_snippet = ""

    if fixed_code is not None:
        code_match = fixed_code.strip()
    else:
        # Extract reasoning (if present) - supports both <think> and <think> formats
        reasoning_match = re.search(r"<think>(.*?)</think>", generated_text, re.DOTALL)
        if not reasoning_match:
            reasoning_match = re.search(
                r"<think>(.*?)</think>", generated_text, re.DOTALL
            )
        reasoning_snippet = reasoning_match.group(1).strip() if reasoning_match else ""

        match = re.search(r"<code>(.*?)</code>", generated_text, re.DOTALL)
        if match:
            code_match = match.group(1).strip()

    llm_score = float("-inf")
    is_llm_feasible = False
    error_type = "missing_code"
    exec_time = 0.0
    violation_data = {}
    code_snippet = ""

    # Create mission summary string
    mission_summary = f"n_vars={mission_dict.get('n_variables', '?')}, "
    mission_summary += f"cardinality={mission_dict.get('cardinality_bounds', [])}, "
    mission_summary += f"precedence={len(mission_dict.get('precedence', []))}, "
    mission_summary += f"mutex={len(mission_dict.get('mutex', []))}, "
    mission_summary += f"groups={len(mission_dict.get('groups', {}))}, "
    mission_summary += f"interactions={len(mission_dict.get('interactions', {}))}"

    if code_match:
        code = code_match
        code_snippet = code  # Capture for W&B

        # Reconstruct Input
        test_reqs = {
            "n_variables": mission_dict.get("n_variables", 10),
            "cardinality_bounds": mission_dict.get("cardinality_bounds", [2, 8]),
            "precedence": mission_dict.get("precedence", []),
            "mutex": mission_dict.get("mutex", []),
            "groups": mission_dict.get("groups", {}),
        }
        interactions = mission_dict.get("interactions", {})
        weights = mission_dict.get("weights", [1.0] * test_reqs["n_variables"])
        adj = {i: [] for i in range(test_reqs["n_variables"])}
        for k in interactions:
            try:
                u, v = map(int, k.split(","))
                # Bounds check to prevent crashes if K is malformed
                if u < test_reqs["n_variables"] and v < test_reqs["n_variables"]:
                    adj[u].append(v)
                    adj[v].append(u)
            except Exception:
                pass

        stdin_obj = {
            "requirements": {
                **test_reqs,
                "weights": weights,
                "interactions": interactions,
            },
            "catalog": {
                "variables": [
                    {"id": j, "weight": weights[j], "neighbors": adj.get(j, [])}
                    for j in range(test_reqs["n_variables"])
                ],
                "interactions": interactions,
            },
        }

        # Execute
        exec_res = robust_execution(code, stdin_obj, n_repeats=repeats)
        exec_time = exec_res.get("execution_time", 0.0)
        error_type = exec_res.get("error_type", "unknown")

        if "selection" in exec_res:
            sel = exec_res["selection"].get("variables", [])
            violations = check_constraint_violations(inst, sel)

            # Flatten violation details
            violation_data = {
                "viol_cardinality": 1 if violations["cardinality"] else 0,
                "viol_precedence": len(violations["precedence"]),
                "viol_mutex": len(violations["mutex"]),
                "viol_groups": len(violations["groups"]),
            }

            if violations["all_valid"]:
                # [FIX] Preserve negative scores
                llm_score = calculate_true_score(inst, sel)
                is_llm_feasible = True
                error_type = "none"
            else:
                error_type = "constraint"

    if is_llm_feasible:
        best_known_score = max(best_known_score, llm_score)

    # 3. Store Metrics
    row.update(
        {
            "llm_score": llm_score,
            "feasible": is_llm_feasible,
            "error_type": error_type,
            "execution_time": exec_time,
            "llm_core_sec": exec_time * 1,
            "best_known_score": best_known_score,
            "code_snippet": code_snippet,  # Capture full code for W&B
            "reasoning": reasoning_snippet,  # Capture reasoning trace for W&B
            "mission_summary": mission_summary,  # Problem definition summary
            **violation_data,
        }
    )

    # Ratios (kept for legacy support, though VBS is preferred downstream)
    for name, metric in base_metrics.items():
        b_score = metric["score"]
        if b_score > 0:
            row[f"ratio_{name}"] = llm_score / b_score
        elif b_score <= 0:
            # Handle infeasible baseline logic
            row[f"ratio_{name}"] = 1.0 if (llm_score > 0) else 0.0

    # Calculate ratio_vbs locally (optional, Main handles this globally too)
    if best_known_score > 0:
        row["ratio_vbs"] = llm_score / best_known_score
    else:
        row["ratio_vbs"] = 0.0

    return row


def load_shinka_dataset(  # noqa: PLR0912, PLR0913, PLR0915
    dataset_name: str,
    split: str = "train",
    output_file: str | None = None,
    test_dataset: str | None = None,
    seed: int | None = None,
    evaluate_on_own_dataset: bool = False,
    max_test_samples: int | None = None,
) -> str:
    """
    Load ShinkaEvolve dataset from HuggingFace and evaluate on problems.

    This function:
    1. Extracts seed from ShinkaEvolve dataset name (e.g., seed303 from dataset name)
    2. If evaluate_on_own_dataset=True: Uses problems from ShinkaEvolve dataset itself
       If evaluate_on_own_dataset=False: Loads test problems from matching test dataset
    3. Extracts code solutions from ShinkaEvolve dataset
    4. Creates evaluation records pairing problems with ShinkaEvolve code

    Args:
        dataset_name: HuggingFace dataset name (e.g., "SoheylM/ShinkaEvolve-SDS-100-seed303")
        split: ShinkaEvolve dataset split to load (default: "train")
        output_file: Output JSONL file path (default: auto-generated)
        test_dataset: Test dataset to evaluate on (if None, auto-constructs from seed)
        seed: Random seed for sampling test problems (if None, extracts from dataset_name)
        evaluate_on_own_dataset: If True, use problems from ShinkaEvolve dataset itself

    Returns:
        Path to the converted JSONL file
    """
    # Extract seed from dataset name if not provided
    if seed is None:
        seed_match = re.search(r"seed(\d+)", dataset_name)
        if seed_match:
            seed = int(seed_match.group(1))
            print(f"📌 Extracted seed {seed} from ShinkaEvolve dataset name")
        else:
            seed = 303
            print(f"⚠️  Could not extract seed from dataset name, using default: {seed}")

    if output_file is None:
        # Generate output file name
        dataset_safe_name = dataset_name.replace("/", "_").replace("-", "_")
        suffix = "_own_problems" if evaluate_on_own_dataset else ""
        output_file = f"evaluation/sds/{dataset_safe_name}_generations{suffix}.jsonl"

    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load ShinkaEvolve dataset
    print(f"📥 Loading ShinkaEvolve dataset from: {dataset_name} (split: {split})")
    try:
        shinka_dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"⚠️  Error loading dataset with split '{split}': {e}")
        print("   Trying without split...")
        try:
            dataset_dict = load_dataset(dataset_name)
            # Try to find a split
            if "train" in dataset_dict:
                shinka_dataset = dataset_dict["train"]
            elif "test" in dataset_dict:
                shinka_dataset = dataset_dict["test"]
            else:
                # Use the first available split
                split_name = next(iter(dataset_dict.keys()))
                shinka_dataset = dataset_dict[split_name]
                print(f"   Using split: {split_name}")
        except Exception as e2:
            raise DatasetLoadError(dataset_name, e2) from e2

    print(f"✅ Loaded {len(shinka_dataset)} ShinkaEvolve samples")

    # Step 2: Extract problems and code solutions from ShinkaEvolve dataset
    shinka_problems = []  # Problems from ShinkaEvolve dataset (if evaluate_on_own_dataset)
    shinka_codes = []  # Code solutions
    for item in shinka_dataset:
        # Extract problem data if evaluating on own dataset (do this first, before code extraction)
        if evaluate_on_own_dataset:
            mission_data = item.get("mission", {})
            # mission might be a JSON string, parse it if needed
            if isinstance(mission_data, str):
                try:
                    mission_data = json.loads(mission_data)
                except Exception:
                    mission_data = {}
            shinka_problems.append(
                {
                    "uuid": item.get("problem_id")
                    or item.get("uuid")
                    or item.get("id", f"shinka_{len(shinka_problems)}"),
                    "mission": mission_data,
                }
            )

        try:
            # Extract code from ShinkaEvolve format (messages array)
            messages = item.get("messages", [])
            assistant_content = None
            for msg in messages:
                if msg.get("role") == "assistant":
                    assistant_content = msg.get("content", "")
                    break

            if not assistant_content:
                # Fallback: try to get from other fields
                assistant_content = item.get("assistant", "") or item.get("content", "")

            # Extract code from <code> tags within the assistant message
            code_match = re.search(r"<code>(.*?)</code>", assistant_content, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()

                # Fix missing imports - ShinkaEvolve code often uses json/sys but doesn't import them
                code_lower = code.lower()
                needs_json = (
                    "json." in code or "json.load" in code or "json.dumps" in code
                ) and "import json" not in code_lower
                needs_sys = (
                    "sys.stdin" in code or "sys.stdout" in code or "sys." in code
                ) and "import sys" not in code_lower

                if needs_json or needs_sys:
                    imports = []
                    if needs_json:
                        imports.append("import json")
                    if needs_sys:
                        imports.append("import sys")
                    # Add imports at the beginning
                    code = "\n".join(imports) + "\n\n" + code

                shinka_codes.append(code)
            else:
                print(
                    f"⚠️  Warning: No <code> tags found in ShinkaEvolve item {item.get('uuid', 'unknown')}"
                )
                # If evaluating on own dataset, we still need to track that this item has no code
                if not evaluate_on_own_dataset:
                    continue  # Skip if not evaluating on own dataset
        except Exception as e:
            print(f"⚠️  Warning: Failed to extract code from ShinkaEvolve item: {e}")
            if not evaluate_on_own_dataset:
                continue

    if not shinka_codes:
        raise CodeExtractionError()

    print(f"✅ Extracted {len(shinka_codes)} code solutions")

    # Step 3: Get problems to evaluate on
    if evaluate_on_own_dataset:
        # Use problems from ShinkaEvolve dataset itself
        # Match problems to codes by index (they should be in the same order)
        n_pairs = min(len(shinka_problems), len(shinka_codes))
        if len(shinka_problems) != len(shinka_codes):
            print(
                f"⚠️  Warning: Mismatch - {len(shinka_problems)} problems but {len(shinka_codes)} code solutions"
            )
            print(f"   Using {n_pairs} matching pairs")
        test_problems = shinka_problems[:n_pairs]
        # Also trim codes to match
        shinka_codes = shinka_codes[:n_pairs]
        print(
            f"📊 Using {len(test_problems)} problems from ShinkaEvolve dataset itself"
        )
    else:
        # Load test problems from standard test dataset
        if test_dataset is None:
            test_dataset = f"SoheylM/OpenR1-SDS-10k-seed{seed}"
            print(
                f"📌 Using test dataset: {test_dataset} (constructed from seed {seed})"
            )

        print(f"📥 Loading test problems from: {test_dataset} (split: test)")
        try:
            test_dataset_obj = load_dataset(test_dataset, split="test")
        except Exception as e:
            raise DatasetLoadError(test_dataset, e) from e

        print(f"✅ Loaded {len(test_dataset_obj)} test problems")

        n_test = len(test_dataset_obj)

        # Determine how many test samples to use
        if max_test_samples is not None:
            n_sample = min(max_test_samples, n_test)
        else:
            # Default: use number of ShinkaEvolve codes if available, otherwise 100
            n_sample = min(len(shinka_codes) if shinka_codes else 100, n_test)

        if n_test > n_sample:
            # Use first n_sample problems (deterministic, not random sampling)
            # This ensures 1:1 mapping with ShinkaEvolve codes when counts match
            test_problems = [test_dataset_obj[i] for i in range(n_sample)]
            print(f"📊 Using first {n_sample} test problems (deterministic selection)")
        else:
            test_problems = list(test_dataset_obj)
            print(f"📊 Using all {n_test} test problems")

    # Step 4: Create evaluation records - pair problems with ShinkaEvolve code
    print("🔄 Creating evaluation records...")
    converted_count = 0

    # Determine pairing strategy: 1:1 if counts match, otherwise cycle
    use_1to1_mapping = (
        len(test_problems) == len(shinka_codes)
    ) and not evaluate_on_own_dataset
    if use_1to1_mapping:
        print(
            f"📌 Using 1:1 mapping ({len(test_problems)} test problems ↔ {len(shinka_codes)} ShinkaEvolve codes)"
        )
    else:
        print(
            f"📌 Using cycling mapping ({len(test_problems)} test problems, {len(shinka_codes)} ShinkaEvolve codes)"
        )

    with Path(output_file).open("w") as f:
        for i, problem in enumerate(tqdm(test_problems, desc="Creating records")):
            if evaluate_on_own_dataset:
                # Match each code solution to its corresponding problem (by index)
                code_idx = i
                if code_idx >= len(shinka_codes):
                    print(
                        f"⚠️  Warning: Problem {i} has no matching code solution, skipping"
                    )
                    continue
            elif use_1to1_mapping:
                # 1:1 mapping when counts match
                code_idx = i
                if code_idx >= len(shinka_codes):
                    print(
                        f"⚠️  Warning: Problem {i} has no matching code solution, skipping"
                    )
                    continue
            else:
                # Cycle through ShinkaEvolve code solutions for test problems
                code_idx = i % len(shinka_codes)

            code = shinka_codes[code_idx]

            # Wrap code in <code> tags for evaluate.py format
            generated_text = f"<code>\n{code}\n</code>"

            # Create record in format expected by evaluate.py
            record = {
                "uuid": problem.get("uuid") or problem.get("id", f"problem_{i}"),
                "mission": problem.get("mission", {}),
                "generated_text": generated_text,
            }

            f.write(json.dumps(record) + "\n")
            converted_count += 1

    print(f"✅ Created {converted_count} evaluation records: {output_file}")
    print(f"   - Test problems: {len(test_problems)}")
    print(f"   - ShinkaEvolve code solutions: {len(shinka_codes)}")
    return output_file


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name to consistent format for directory structure.
    Examples:
        "Qwen2.5-Coder-7B-Instruct" -> "qwen2.5-coder-7b"
        "qwen2.5-coder-32b" -> "qwen2.5-coder-32b"
        "Qwen2.5-Coder-1.5B-Instruct" -> "qwen2.5-coder-1.5b"
    """
    # Convert to lowercase
    normalized = model_name.lower()
    # Remove common suffixes
    normalized = normalized.replace("-instruct", "")
    # Remove version dots (keep as-is, e.g., "2.5" -> "2.5")
    # Extract size pattern (e.g., "7b", "32b", "1.5b")
    return normalized


def construct_output_dir(  # noqa: PLR0913
    model: str | None = None,
    training_scheme: str | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
    base_dir: str = "evaluation/sds/results",
    job_id: str | None = None,
    shinka_dataset: str | None = None,
    evaluate_on_own_dataset: bool = False,
) -> str:
    """
    Construct hierarchical output directory path.

    If output_dir is explicitly provided, use it (backward compatibility).
    Otherwise, construct from model/training_scheme/seed/job-id or shinka-evolve/dataset/seed/eval-type.

    Args:
        model: Model name (e.g., "qwen2.5-coder-7b")
        training_scheme: Training scheme (e.g., "sft", "grpo", "sft-grpo")
        seed: Seed value (e.g., 101, 202, 303)
        output_dir: Explicit output directory (takes precedence)
        base_dir: Base directory for results
        job_id: Job ID for fine-tuned models (e.g., "12345")
        shinka_dataset: ShinkaEvolve dataset name (e.g., "SoheylM/ShinkaEvolve-SDS-100-seed303")
        evaluate_on_own_dataset: Whether evaluating on own dataset (affects ShinkaEvolve path)

    Returns:
        Constructed output directory path
    """
    # If output_dir is explicitly provided, use it (backward compatibility)
    if output_dir and output_dir != "evaluation/sds/results":
        return output_dir

    # ShinkaEvolve structure: shinka-evolve/{dataset_name}/seed{seed}/{eval_type}/
    if shinka_dataset:
        # Extract dataset name and reformat: "SoheylM/ShinkaEvolve-SDS-100-seed303" -> "ShinkaEvolve-SDS-100-seed303"
        dataset_name = shinka_dataset.split("/")[-1]  # Get last part after /
        # Remove "SoheylM/" prefix if present
        if "/" in shinka_dataset:
            dataset_name = shinka_dataset.split("/")[-1]

        # Extract seed from dataset name if not provided
        if seed is None:
            seed_match = re.search(r"seed(\d+)", dataset_name)
            seed = int(seed_match.group(1)) if seed_match else _DEFAULT_SHINKA_SEED

        seed_str = f"seed{seed}"

        # Evaluation type subfolder
        eval_type = "own-dataset" if evaluate_on_own_dataset else "test"

        return str(
            Path(base_dir) / "shinka-evolve" / dataset_name / seed_str / eval_type
        )

    # Fine-tuned models structure: model/training_scheme/seed/job-{job_id}/
    if model and training_scheme and seed is not None:
        normalized_model = normalize_model_name(model)
        normalized_training = training_scheme.lower().replace("_", "-")
        seed_str = f"seed{seed}"

        # Add job-id subfolder if provided
        if job_id:
            job_str = f"job-{job_id}"
            return str(
                Path(base_dir)
                / normalized_model
                / normalized_training
                / seed_str
                / job_str
            )
        else:
            # No job-id: use seed level (backward compatibility)
            return str(
                Path(base_dir) / normalized_model / normalized_training / seed_str
            )

    # Fallback: if only seed provided, use old format for backward compatibility
    if seed is not None:
        return str(Path(base_dir) / f"results_seed{seed}")

    # Default fallback
    return base_dir


def extract_seed_from_path(path: str) -> int:
    """
    Extract seed from output directory path.
    Handles both old format (results_seed303) and new format (qwen2.5-coder-7b/sft-grpo/seed101).
    """
    # Try new format first: .../seed101/...
    seed_match = re.search(r"seed(\d+)", path)
    if seed_match:
        return int(seed_match.group(1))
    return 42  # Default fallback


def extract_config_and_method_name(
    training_scheme: str | None = None,
) -> tuple[str, str]:
    """
    Extract config name and method name from training scheme.

    Args:
        training_scheme: Training scheme string (e.g., "grpo-config_ablation_oracle")

    Returns:
        Tuple of (config_name, method_name)
        - config_name: e.g., "config_hero", "config_ablation_oracle", None
        - method_name: e.g., "Ours (Hero)", "Ours (+Oracle)", None
    """
    if not training_scheme:
        return None, None

    # Parse training scheme (e.g., "grpo-config_ablation_no_gen" or "sft-grpo-config_ablation_no_gen")
    scheme_parts = training_scheme.lower().replace("_", "-").split("-")

    # Find ablation tag (starts with "config")
    config_name = None
    for part in scheme_parts:
        if part.startswith("config"):
            # Reconstruct full config name (may have been split by hyphens)
            config_idx = scheme_parts.index(part)
            config_parts = scheme_parts[config_idx:]
            config_name = "-".join(config_parts).replace(
                "-", "_"
            )  # Convert back to underscore format
            break

    # Map config name to method name
    if config_name:
        # Remove "config_" prefix if present
        name = config_name.replace("config_", "")

        method_map = {
            "hero": "Ours (Hero)",
            "ablation_oracle": "Ours (+Oracle)",
            "ablation_diversity": "Ours (+Diversity)",
            "ablation_generalization": "Ours (+Generalization)",
            "ablation_soft_gate": "Ours (+Soft Gate)",
            "minimalist": "Ours (w/o Structure)",
            "ablation_prompt": "Ours (w/o Prompt)",
            "ablation_no_oracle": "Ours (No Oracle)",
            "ablation_no_gen": "Ours (No Gen)",
            "ablation_no_diversity": "Ours (No Diversity)",
            "ablation_neutral_prompt": "Ours (Neutral Prompt)",
            "discovery": "Ours (Discovery)",
        }

        method_name = method_map.get(name, f"Ours ({name})")
        return config_name, method_name

    return None, None


def normalize_ablation_tag(config_name: str) -> str:
    """
    Convert config name to readable abbreviation for W&B run names.

    Examples:
        "config_hero" -> "hero"
        "config_ablation_no_gen" -> "no-gen"
        "config_ablation_generalization" -> "gen"
    """
    # Remove "config_" prefix if present
    name = config_name.replace("config_", "")

    # Map to readable abbreviations
    mapping = {
        "hero": "hero",
        "discovery": "discovery",
        "ablation_no_oracle": "no-oracle",
        "ablation_oracle": "oracle",
        "ablation_no_gen": "no-gen",
        "ablation_generalization": "gen",
        "ablation_no_diversity": "no-div",
        "ablation_diversity": "div",
        "ablation_soft_gate": "soft-gate",
        "ablation_neutral_prompt": "neutral-prompt",
        "minimalist": "minimalist",
        "ablation_prompt": "prompt",
    }

    if name in mapping:
        return mapping[name]

    # Fallback: remove "ablation_" prefix and replace underscores with hyphens
    if name.startswith("ablation_"):
        return name.replace("ablation_", "").replace("_", "-")

    return name.replace("_", "-")


def construct_wandb_run_name(  # noqa: PLR0911, PLR0912, PLR0913
    model: str | None = None,
    training_scheme: str | None = None,
    seed: int | None = None,
    job_id: str | None = None,
    shinka_dataset: str | None = None,
    evaluate_on_own_dataset: bool = False,
) -> str:
    """
    Construct W&B run name matching the training script's naming convention.

    For fine-tuned models: {model}-grpo-{ablation}-sds-seed{seed}-job{job_id}
    For ShinkaEvolve: shinka-evolve-{dataset_name}-seed{seed}-{eval_type}

    Args:
        model: Model name (e.g., "qwen2.5-coder-7b")
        training_scheme: Training scheme (e.g., "grpo", "sft-grpo", "grpo-config_ablation_no_gen")
        seed: Seed value
        job_id: Job ID for fine-tuned models
        shinka_dataset: ShinkaEvolve dataset name
        evaluate_on_own_dataset: Whether evaluating on own dataset

    Returns:
        W&B run name string
    """
    if shinka_dataset:
        # ShinkaEvolve: shinka-evolve-{dataset_name}-seed{seed}-{eval_type}
        dataset_name = shinka_dataset.split("/")[
            -1
        ]  # e.g., "ShinkaEvolve-SDS-100-seed303"
        # Remove "seedXXX" from dataset name if present (we'll add it explicitly)
        dataset_base = re.sub(r"-seed\d+", "", dataset_name)
        eval_type = "own-dataset" if evaluate_on_own_dataset else "test"
        return f"shinka-evolve-{dataset_base}-seed{seed}-{eval_type}"
    else:
        # Fine-tuned models: Parse training scheme to extract ablation tag
        normalized_model = normalize_model_name(model) if model else "unknown"
        seed_str = f"seed{seed}" if seed is not None else "seed42"
        job_str = f"job{job_id}" if job_id else ""

        if training_scheme:
            # Parse training scheme (e.g., "grpo-config_ablation_no_gen" or "sft-grpo-config_ablation_no_gen")
            scheme_parts = training_scheme.lower().replace("_", "-").split("-")

            # Find ablation tag (starts with "config")
            ablation_tag = None
            base_scheme = []
            for part in scheme_parts:
                if part.startswith("config"):
                    # Reconstruct full config name (may have been split by hyphens)
                    config_idx = scheme_parts.index(part)
                    # Join remaining parts to reconstruct config name
                    config_parts = scheme_parts[config_idx:]
                    ablation_tag = "-".join(config_parts)
                    break
                else:
                    base_scheme.append(part)

            if ablation_tag:
                # Convert to readable abbreviation
                readable_tag = normalize_ablation_tag(ablation_tag)
                base = "-".join(base_scheme) if base_scheme else "grpo"
                if job_str:
                    return f"{normalized_model}-{base}-{readable_tag}-sds-{seed_str}-{job_str}"
                else:
                    return f"{normalized_model}-{base}-{readable_tag}-sds-{seed_str}"
            else:
                # No ablation tag, use scheme as-is
                normalized_scheme = (
                    "-".join(base_scheme) if base_scheme else training_scheme
                )
                if job_str:
                    return f"{normalized_model}-{normalized_scheme}-sds-{seed_str}-{job_str}"
                else:
                    return f"{normalized_model}-{normalized_scheme}-sds-{seed_str}"
        # No training scheme provided
        elif job_str:
            return f"{normalized_model}-unknown-sds-{seed_str}-{job_str}"
        else:
            return f"{normalized_model}-unknown-sds-{seed_str}"


def log_to_wandb(  # noqa: PLR0912, PLR0913, PLR0915
    output_dir: str,
    model: str | None = None,
    training_scheme: str | None = None,
    seed: int | None = None,
    job_id: str | None = None,
    shinka_dataset: str | None = None,
    evaluate_on_own_dataset: bool = False,
    df: pd.DataFrame = None,
):
    """
    Log evaluation results to Weights & Biases.

    For fine-tuned models: Finds existing run by name and logs to it.
    For ShinkaEvolve: Creates new run with descriptive name.

    Logs:
    - Images: error_distribution.png, plot_pareto_variant2.png, plot_perf_profile.png
    - Artifact: metrics_final.csv
    - Summary metrics: pass_rate, mean_score, etc.

    Args:
        output_dir: Directory containing evaluation results
        model: Model name (for fine-tuned models)
        training_scheme: Training scheme (for fine-tuned models)
        seed: Seed value
        job_id: Job ID (for fine-tuned models)
        shinka_dataset: ShinkaEvolve dataset name
        evaluate_on_own_dataset: Whether evaluating on own dataset
        df: DataFrame with evaluation metrics (optional, for summary metrics)
    """
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        print("⚠️  wandb not available. Install with: pip install wandb")
        return

    # Check if W&B credentials are available
    if not os.environ.get("WANDB_API_KEY"):
        print("⚠️  WANDB_API_KEY not found in environment. Skipping W&B logging.")
        return

    def _infer_eval_tag_from_env_or_path() -> str | None:  # noqa: PLR0911
        """
        Best-effort, backward-compatible method tag inference for eval runs.
        Priority:
        1) WANDB_ABLATION_TAG (set by SLURM scripts; stable)
        2) CHECKPOINT_DIR env var (path contains config_hero / config_ablation_*)
        Returns short tag like: hero/oracle/diversity/prompt/gen/no-gen/no-oracle/no-div.
        """
        tag = os.environ.get("WANDB_ABLATION_TAG", "") or ""
        tag = tag.strip()
        if tag:
            return tag
        ckpt = os.environ.get("CHECKPOINT_DIR", "") or ""
        ckpt_l = ckpt.lower()
        if "config_hero" in ckpt_l:
            return "hero"
        if "config_ablation_oracle" in ckpt_l:
            return "oracle"
        if "config_ablation_diversity" in ckpt_l:
            return "diversity"
        if "config_minimalist" in ckpt_l:
            return "minimalist"
        if "config_ablation_prompt" in ckpt_l:
            return "prompt"
        if "config_ablation_generalization" in ckpt_l:
            return "gen"
        if "config_ablation_soft_gate" in ckpt_l:
            return "soft-gate"
        return None

    # Construct run name (backward compatible)
    legacy_run_name = construct_wandb_run_name(
        model=model,
        training_scheme=training_scheme,
        seed=seed,
        job_id=job_id,
        shinka_dataset=shinka_dataset,
        evaluate_on_own_dataset=evaluate_on_own_dataset,
    )

    run_name = legacy_run_name
    # For new eval runs, prefer a clearer name that includes the ablation tag and ends with -eval.
    # Do NOT attempt to rename historical runs; this only affects the run created/resumed now.
    if not shinka_dataset:
        eval_tag = _infer_eval_tag_from_env_or_path()
        normalized_model = normalize_model_name(model) if model else "unknown"
        seed_str = f"seed{seed}" if seed is not None else "seed42"
        job_str = f"job{job_id}" if job_id else ""
        base = (training_scheme or "grpo").lower()
        if eval_tag and job_str:
            run_name = (
                f"{normalized_model}-{base}-{eval_tag}-sds-{seed_str}-{job_str}-eval"
            )
        elif eval_tag:
            run_name = f"{normalized_model}-{base}-{eval_tag}-sds-{seed_str}-eval"
        else:
            # If we can't infer the tag, still suffix -eval to make it obvious this is an eval run.
            run_name = (
                f"{legacy_run_name}-eval"
                if not legacy_run_name.endswith("-eval")
                else legacy_run_name
            )

    # Get project and entity from environment (same as training)
    project = os.environ.get("WANDB_PROJECT", "qwen-coder-sds-rl")
    entity = os.environ.get("WANDB_ENTITY", "smassoudi-eth-z-rich")
    batch_id = os.environ.get("BATCH_ID", None)

    print("\n📊 Logging evaluation results to W&B...")
    print(f"   Run name: {run_name}")
    print(f"   Project: {project}")
    print(f"   Entity: {entity}")

    try:
        # Initialize W&B run
        # For fine-tuned models: try to resume existing run, otherwise create new
        # For ShinkaEvolve: always create new run
        if shinka_dataset:
            # ShinkaEvolve: create new run
            wandb.init(
                name=run_name,
                project=project,
                entity=entity,
                job_type="evaluation",
                tags=["shinka-evolve", "evaluation"],
            )
        else:
            # Fine-tuned models: try to find existing run by name
            wandb.init(
                name=run_name,
                project=project,
                entity=entity,
                resume="allow",  # Resume if exists, create new if not
                job_type="evaluation",
                tags=["evaluation"],
                group=batch_id,
            )
            # Extra structured metadata for robust future lookup (does not affect legacy runs)
            with contextlib.suppress(Exception):
                wandb.config.update(
                    {
                        "batch_id": batch_id,
                        "eval_tag": os.environ.get("WANDB_ABLATION_TAG", "") or None,
                        "job_id": job_id,
                        "seed": seed,
                        "training_scheme": training_scheme,
                        "model": model,
                        "output_dir": output_dir,
                    },
                    allow_val_change=True,
                )

        # Log images if they exist
        image_files = [
            "error_distribution.png",
            "robustness_profile.png",
            "stratified_boxplot.png",
        ]

        logged_images = []
        output_path = Path(output_dir)
        for img_file in image_files:
            img_path = output_path / img_file
            if img_path.exists():
                try:
                    wandb.log({img_file.replace(".png", ""): wandb.Image(img_path)})
                    logged_images.append(img_file)
                except Exception as e:
                    print(f"   ⚠️  Failed to log {img_file}: {e}")

        if logged_images:
            print(
                f"   ✅ Logged {len(logged_images)} image(s): {', '.join(logged_images)}"
            )

        # Log CSV and LaTeX tables as artifact
        artifact = wandb.Artifact(
            name=f"evaluation-metrics-{run_name}", type="evaluation-results"
        )

        # Add CSV
        csv_path = output_path / "metrics_final.csv"
        if csv_path.exists():
            try:
                artifact.add_file(csv_path)
                print("   ✅ Added metrics_final.csv to artifact")
            except Exception as e:
                print(f"   ⚠️  Failed to add CSV to artifact: {e}")

        # Add LaTeX tables
        tex_files = ["results_table.tex", "results_stratified.tex"]
        for tex_file in tex_files:
            tex_path = output_path / tex_file
            if tex_path.exists():
                try:
                    artifact.add_file(tex_path)
                    print(f"   ✅ Added {tex_file} to artifact")
                except Exception as e:
                    print(f"   ⚠️  Failed to add {tex_file} to artifact: {e}")

        # Log the artifact
        if artifact.manifest.entries:
            try:
                wandb.log_artifact(artifact)
                print("   ✅ Logged evaluation artifacts (CSV + LaTeX tables)")
            except Exception as e:
                print(f"   ⚠️  Failed to log artifact: {e}")

        # Log summary metrics if DataFrame is provided
        if df is not None and not df.empty:
            try:
                feasible_df = df[df["feasible"]]

                summary_metrics = {
                    "evaluation/pass_rate": df["feasible"].mean() * 100,
                    "evaluation/total_samples": len(df),
                    "evaluation/feasible_samples": len(feasible_df),
                }

                if len(feasible_df) > 0:
                    summary_metrics.update(
                        {
                            "evaluation/mean_score": feasible_df["llm_score"].mean(),
                            "evaluation/mean_execution_time": feasible_df[
                                "execution_time"
                            ].mean(),
                        }
                    )

                    # Constraint violation counts
                    violation_cols = [
                        col for col in df.columns if col.startswith("violation_")
                    ]
                    for col in violation_cols:
                        summary_metrics[f"evaluation/{col}"] = df[col].sum()

                wandb.summary.update(summary_metrics)
                print("   ✅ Logged summary metrics")

                # [FIX] Log ALL feasible solutions (not just top 50)
                if "code_snippet" in feasible_df.columns:
                    # Log all feasible solutions with reasoning and mission info
                    table_cols = [
                        "uuid",
                        "mission_summary",  # Problem definition
                        "difficulty_class",
                        "llm_score",
                        "vbs_score",
                        "execution_time",
                        "reasoning",  # Thinking trace
                        "code_snippet",  # Full code
                    ]
                    valid_cols = [c for c in table_cols if c in feasible_df.columns]
                    if valid_cols:
                        # Sort by score for easier browsing (best first)
                        all_solutions = feasible_df.sort_values(
                            by="llm_score", ascending=False
                        )[valid_cols]
                        if not all_solutions.empty:
                            wandb.log(
                                {"all_solutions": wandb.Table(dataframe=all_solutions)}
                            )
                            print(
                                f"   ✅ Logged ALL {len(all_solutions)} feasible solutions to W&B (with reasoning and mission info)"
                            )

                        # Also log top 50 for quick reference
                        top_50 = all_solutions.head(50)
                        if not top_50.empty:
                            wandb.log(
                                {"top_50_solutions": wandb.Table(dataframe=top_50)}
                            )
                            print("   ✅ Logged Top 50 solutions (quick reference)")
            except Exception as e:
                print(f"   ⚠️  Failed to log summary metrics: {e}")

        wandb.finish()
        print(f"   ✅ Successfully logged to W&B run: {run_name}")

    except Exception as e:
        print(f"   ❌ Error logging to W&B: {e}")
        print(f"   Evaluation results are still saved locally in: {output_dir}")
        # Don't fail the evaluation if W&B logging fails


def main():  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(
        description="Evaluate SDS model generations against baseline solvers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # New hierarchical structure (recommended)
  python evaluate.py --model qwen2.5-coder-7b --training-scheme sft-grpo --seed 101
  
  # Evaluate ShinkaEvolve dataset on test problems
  python evaluate.py --shinka-dataset SoheylM/ShinkaEvolve-SDS-100-seed303
  
  # Evaluate ShinkaEvolve dataset on its own problems (verify feasibility)
  python evaluate.py --shinka-dataset SoheylM/ShinkaEvolve-SDS-100-seed303 --evaluate-on-own-dataset
  
  # Evaluate fine-tuned model with job-id (matches training structure)
  python evaluate.py --model qwen2.5-coder-7b --training-scheme sft-grpo --seed 101 --job-id 12345
  
  # With explicit output directory (backward compatible)
  python evaluate.py --output_dir evaluation/sds/results/my_custom_dir
  
  # Plot-only mode
  python evaluate.py --output_dir evaluation/sds/results/qwen2.5-coder-7b/sft-grpo/seed101 --plot-only
        """,
    )
    parser.add_argument(
        "--input_file",
        default="evaluation/sds/generations.jsonl",
        help="Input file with LLM generations (JSONL format)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (if not provided, constructed from model/training/seed)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., 'qwen2.5-coder-7b', 'Qwen2.5-Coder-32B-Instruct')",
    )
    parser.add_argument(
        "--training-scheme",
        type=str,
        default=None,
        choices=["sft", "grpo", "sft-grpo", "sft_grpo", "base", "shinka", "fixed-code"],
        help="Training scheme: 'sft', 'grpo', 'sft-grpo', 'base' (untrained model), or 'shinka' (ShinkaEvolve)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace SDS dataset to evaluate. In fixed-code mode, loads the test split directly.",
    )
    parser.add_argument(
        "--shinka-dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name for ShinkaEvolve dataset (e.g., 'SoheylM/ShinkaEvolve-SDS-100-seed303'). "
        "When provided, loads ShinkaEvolve code solutions and evaluates them on test problems.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="ShinkaEvolve dataset split to load (default: 'train')",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default=None,
        help="Test dataset to evaluate ShinkaEvolve code on (default: auto-constructed from seed, e.g., 'SoheylM/OpenR1-SDS-10k-seed{SEED}')",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Maximum number of test samples to use when evaluating ShinkaEvolve (default: 100, or match number of ShinkaEvolve codes if provided)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate from input file (for faster debugging, default: all samples)",
    )
    parser.add_argument(
        "--evaluate-on-own-dataset",
        action="store_true",
        help="Evaluate ShinkaEvolve code on the problems it was evolved for (from mission column) instead of test problems",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["greedy", "local_search", "bnb"],
        help="List of baselines to run",
    )
    parser.add_argument(
        "--time_budget",
        type=float,
        default=1.0,
        help="Time budget (sec) for baseline solvers",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Number of execution repeats for timing"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for solvers (ensures reproducibility)",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job ID for fine-tuned models (e.g., '12345'). Creates subfolder job-{job_id} to match training structure.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing metrics_final.csv file",
    )
    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        help="Log evaluation results (images, CSV, metrics) to Weights & Biases. "
        "For fine-tuned models, finds existing run by name and logs to it. "
        "For ShinkaEvolve, creates new run. Requires WANDB_API_KEY environment variable.",
    )
    parser.add_argument(
        "--best-of-n",
        action="store_true",
        help="Analyze results as Best-of-N (Pass@k metrics). Use this for Base Model eval with multiple samples per problem. "
        "Automatically detects multiple samples per UUID and runs bootstrapping analysis. "
        "Collapses to best score per UUID BEFORE VBS calculation for fair comparison.",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=500,
        help="Number of bootstrap iterations for Pass@k analysis (default: 500, use 1000-10000 for publication-quality plots)",
    )
    parser.add_argument(
        "--fixed-code-file",
        type=str,
        default=None,
        help="Path to a single Python solver file to run unchanged across every SDS problem.",
    )
    parser.add_argument(
        "--method-name-override",
        type=str,
        default=None,
        help="Override method_name written to experiment metadata and downstream outputs.",
    )
    parser.add_argument(
        "--code-label",
        type=str,
        default=None,
        help="Human-readable label for the fixed code source.",
    )
    parser.add_argument(
        "--code-source-type",
        type=str,
        default=None,
        help="Provenance tag for fixed code runs (for example: frozen-hero, manual-sa).",
    )
    parser.add_argument(
        "--code-source-path",
        type=str,
        default=None,
        help="Original source path recorded in experiment metadata for fixed-code runs.",
    )
    parser.add_argument(
        "--code-source-seed",
        type=int,
        default=None,
        help="Seed associated with the fixed code source, if applicable.",
    )
    args = parser.parse_args()

    # Normalize training scheme (handle both hyphen and underscore)
    if args.training_scheme:
        args.training_scheme = args.training_scheme.replace("_", "-")

    if args.fixed_code_file:
        if not args.training_scheme:
            args.training_scheme = "fixed-code"
        if not args.model:
            args.model = "fixed-code"
        if not args.dataset and not Path(args.input_file).exists():
            print(
                "❌ ERROR: Fixed-code mode requires either --dataset or an existing --input_file"
            )
            sys.exit(1)
        args.code_source_path = args.code_source_path or args.fixed_code_file
        args.code_label = args.code_label or sanitize_label(
            Path(args.fixed_code_file).stem, DEFAULT_FIXED_CODE_LABEL
        )
        args.method_name_override = (
            args.method_name_override or DEFAULT_FIXED_CODE_METHOD_NAME
        )

    # Handle ShinkaEvolve dataset loading
    if args.shinka_dataset:
        if not HAS_DATASETS:
            print(
                "❌ ERROR: datasets library not available. Install with: pip install datasets"
            )
            sys.exit(1)

        print(f"📦 Loading ShinkaEvolve dataset: {args.shinka_dataset}")

        # Extract seed from ShinkaEvolve dataset name if not provided
        # This seed will be used for both test dataset selection and sampling
        if args.seed == _DEFAULT_SEED:  # Only if using default
            seed_match = re.search(r"seed(\d+)", args.shinka_dataset)
            if seed_match:
                args.seed = int(seed_match.group(1))
                print(f"📌 Extracted seed from ShinkaEvolve dataset name: {args.seed}")

        # Set training scheme to "shinka" if not specified
        if not args.training_scheme:
            args.training_scheme = "shinka"
            print(f"📌 Using training scheme: {args.training_scheme}")

        # Set model name if not specified
        if not args.model:
            args.model = "shinka-evolve"
            print(f"📌 Using model name: {args.model}")

        # Add cpsat to baselines if not already present (for fair comparison with LLM evaluation)
        if "cpsat" not in args.baselines:
            args.baselines.append("cpsat")
            print("📌 Added 'cpsat' to baselines for ShinkaEvolve evaluation")

        # Load and convert ShinkaEvolve dataset
        args.input_file = load_shinka_dataset(
            args.shinka_dataset,
            args.dataset_split,
            args.input_file,
            test_dataset=args.test_dataset,
            seed=args.seed,
            evaluate_on_own_dataset=args.evaluate_on_own_dataset,
            max_test_samples=args.max_test_samples,
        )

    # Validate that required args are provided for new structure (only warn, don't fail)
    if (
        not args.plot_only
        and not args.output_dir
        and (not args.model or not args.training_scheme)
    ):
        print("⚠️  WARNING: For hierarchical structure, provide:")
        print("   --model <model_name> --training-scheme <scheme> --seed <seed>")
        print(
            "   Example: --model qwen2.5-coder-7b --training-scheme sft-grpo --seed 101"
        )
        print("   (Using fallback directory structure)")

    # Construct output directory
    args.output_dir = construct_output_dir(
        model=args.model,
        training_scheme=args.training_scheme,
        seed=args.seed,
        output_dir=args.output_dir,
        job_id=args.job_id,
        shinka_dataset=args.shinka_dataset,
        evaluate_on_own_dataset=args.evaluate_on_own_dataset,
    )

    print(f"📁 Output directory: {args.output_dir}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save experiment metadata for aggregation script
    config_name, method_name = extract_config_and_method_name(args.training_scheme)
    config_name, method_name = extract_config_and_method_name(args.training_scheme)
    effective_method_name = infer_method_name(args, method_name)
    metadata = {
        "model": args.model,
        "training_scheme": args.training_scheme,
        "seed": args.seed,
        "job_id": args.job_id,
        "config_name": config_name,
        "method_name": effective_method_name,
        "shinka_dataset": args.shinka_dataset,
        "evaluate_on_own_dataset": args.evaluate_on_own_dataset,
        "dataset": args.dataset,
        "fixed_code_file": args.fixed_code_file,
        "code_source_type": args.code_source_type,
        "code_source_path": args.code_source_path,
        "code_source_seed": args.code_source_seed,
        "code_label": args.code_label,
    }

    metadata_path = Path(args.output_dir) / "experiment_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Saved experiment metadata to {metadata_path}")
    primary_method_name = effective_method_name or "LLM (Ours)"

    # --- PLOT-ONLY MODE ---
    if args.plot_only:
        metrics_file = Path(args.output_dir) / "metrics_final.csv"
        if not metrics_file.exists():
            print(f"❌ ERROR: Metrics file not found: {metrics_file}")
            print("   Run evaluation first to generate metrics_final.csv")
            return

        print(f"📊 Plot-only mode: Reading from {metrics_file}")
        evaluation_data = pd.read_csv(metrics_file)

        # Calculate VBS and difficulty if not present (for backward compatibility)
        if (
            "vbs_score" not in evaluation_data.columns
            or "difficulty_class" not in evaluation_data.columns
        ):
            print(
                "📊 Computing VBS and difficulty classification (missing from CSV)..."
            )

            def calculate_vbs_and_difficulty(row):
                # A. Find VBS (Max of LLM + All Baselines)
                scores = []

                # LLM Score (only if feasible, preserve negative scores)
                if row.get("feasible", False):
                    scores.append(row.get("llm_score", float("-inf")))

                # Baseline Scores (preserve negative scores, filter only -inf)
                for col in row.index:
                    if col.startswith("score_"):
                        score_val = row[col]
                        if score_val > float("-inf"):  # Only exclude -inf (infeasible)
                            scores.append(score_val)

                # VBS is the absolute max found by anyone (can be negative)
                valid_scores = [s for s in scores if s > float("-inf")]
                vbs = float("-inf") if not valid_scores else max(valid_scores)

                # B. Calculate Difficulty (Gap vs Greedy)
                # [FIX] Robust gap formula: (VBS - Greedy) / (|VBS| + epsilon)
                greedy_score = row.get("score_greedy", float("-inf"))
                if greedy_score == float("-inf"):
                    greedy_score = 0.0  # Anchor for gap calc

                if vbs == float("-inf") or vbs <= _EPSILON_SMALL:
                    hardness = 1.0  # All failed
                elif greedy_score <= _EPSILON_SMALL:
                    hardness = 1.0  # Greedy failed
                else:
                    # [FIX] Standard relative gap formula
                    epsilon = 1e-10
                    numerator = vbs - greedy_score
                    denominator = abs(vbs) + epsilon
                    hardness = numerator / denominator

                return pd.Series([vbs, hardness], index=["vbs_score", "hardness"])

            evaluation_data[["vbs_score", "hardness"]] = evaluation_data.apply(
                calculate_vbs_and_difficulty, axis=1
            )

            def classify_diff(h):
                if h < _TRIVIAL_THRESHOLD:
                    return "Trivial"
                if h < _MODERATE_THRESHOLD:
                    return "Moderate"
                return "Hard"

            evaluation_data["difficulty_class"] = evaluation_data["hardness"].apply(
                classify_diff
            )

        evaluation_data[evaluation_data["feasible"]]

        # Infer baselines from CSV columns
        baseline_cols = [
            col for col in evaluation_data.columns if col.startswith("score_")
        ]
        inferred_baselines = [col.replace("score_", "") for col in baseline_cols]
        active_baselines_config = {}
        for b in inferred_baselines:
            if b in AVAILABLE_SOLVERS:
                active_baselines_config[b] = AVAILABLE_SOLVERS[b]

        print(f"⚖️  Detected Baselines: {list(active_baselines_config.keys())}")

        # Extract seed from output_dir (handles both old and new formats)
        args.seed = extract_seed_from_path(args.output_dir)
        print(f"📌 Using seed: {args.seed} (extracted from output_dir)")
    else:
        # Setup Baselines (just config, not actual solvers - those are created in workers)
        active_baselines_config = {}
        for b in args.baselines:
            if b in AVAILABLE_SOLVERS:
                active_baselines_config[b] = AVAILABLE_SOLVERS[b]

        print(f"⚖️  Active Baselines: {list(active_baselines_config.keys())}")

        if args.fixed_code_file and args.dataset:
            print(f"📥 Loading fixed-code dataset: {args.dataset} (split=test)")
            lines = build_dataset_records(args.dataset, split="test")
        else:
            with Path(args.input_file).open() as f:
                lines = [line for line in f if line.strip()]

        # Limit number of samples if --max-samples is provided (for faster debugging)
        if args.max_samples is not None and args.max_samples > 0:
            original_count = len(lines)
            lines = lines[: args.max_samples]
            print(
                f"📊 Limiting evaluation to {len(lines)} samples (from {original_count} total) for faster debugging"
            )

        print(f"🧪 Evaluating {len(lines)} samples...")
        print(f"🚀 Using {args.workers} parallel workers with seed {args.seed}")
        evaluation_start_time = time.time()
        fixed_code = None
        if args.fixed_code_file:
            fixed_code = Path(args.fixed_code_file).read_text()
            print(f"📌 Fixed solver source: {args.fixed_code_file}")

        # Adjust workers if CP-SAT is used (it uses 8 threads internally)
        # This prevents oversubscription: 4 workers x 8 threads = 32 threads competing
        if "cpsat" in active_baselines_config:
            # Moderate throttling: reduce by factor of 2 instead of 8
            # This allows some parallelism while preventing excessive oversubscription
            effective_workers = max(1, args.workers // 2)
            if effective_workers < args.workers:
                print(
                    f"⚠️  CP-SAT detected (uses 8 threads per instance). Reducing concurrency from {args.workers} to {effective_workers} workers to prevent oversubscription."
                )
        else:
            effective_workers = args.workers

        print(f"🚀 Launching {effective_workers} parallel processes.")

        results = []

        # Parallel execution
        if effective_workers <= 1:
            print("🚀 Running evaluation in-process (single-worker mode).")
            iterator = tqdm(enumerate(lines), total=len(lines), desc="Evaluating")
            for i, line in iterator:
                try:
                    results.append(
                        evaluate_single_sample(
                            line,
                            i,
                            active_baselines_config,
                            args.time_budget,
                            args.repeats,
                            args.seed,
                            fixed_code,
                        )
                    )
                except Exception as e:
                    print(f"⚠️  Error in worker: {e}")
        else:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                futures = [
                    executor.submit(
                        evaluate_single_sample,
                        line,
                        i,
                        active_baselines_config,
                        args.time_budget,
                        args.repeats,
                        args.seed,
                        fixed_code,
                    )
                    for i, line in enumerate(lines)
                ]

                for f in tqdm(as_completed(futures), total=len(lines), desc="Evaluating"):
                    try:
                        results.append(f.result())
                    except Exception as e:
                        print(f"⚠️  Error in worker: {e}")

        # --- OUTPUT GENERATION ---
        evaluation_data = pd.DataFrame(results)
        evaluation_wall_clock_seconds = time.time() - evaluation_start_time

        # [NEW] BEST-OF-N ANALYSIS: Check for multiple samples per UUID
        if args.best_of_n and "uuid" in evaluation_data.columns:
            # Check if we have multiple samples per UUID
            uuid_counts = evaluation_data.groupby("uuid").size()
            max_samples = uuid_counts.max()

            if max_samples > 1:
                print(
                    f"🔬 Detected multiple samples per UUID (max: {max_samples}). Running Best-of-N analysis..."
                )

                # STEP 1: Run Scaling Law Analysis on RAW data (before collapsing)
                # We need temporary VBS for gap calculation in bootstrapping
                print(
                    "📊 Step 1: Computing temporary VBS for bootstrapping analysis..."
                )

                def compute_temp_vbs_per_uuid(uuid_group):
                    """Compute VBS across all samples and baselines for a UUID."""
                    scores = []
                    # Collect all LLM scores (feasible only) from all samples
                    for _, row in uuid_group.iterrows():
                        if row.get("feasible", False):
                            scores.append(row.get("llm_score", float("-inf")))
                    # Collect baseline scores (once per UUID, baselines are deterministic)
                    first_row = uuid_group.iloc[0]
                    for col in first_row.index:
                        if col.startswith("score_"):
                            score_val = first_row[col]
                            if score_val > float("-inf"):
                                scores.append(score_val)
                    valid_scores = [s for s in scores if s > float("-inf")]
                    return max(valid_scores) if valid_scores else float("-inf")

                # Compute temporary VBS for bootstrapping (assign to all samples of each UUID)
                vbs_per_uuid = evaluation_data.groupby("uuid").apply(
                    compute_temp_vbs_per_uuid
                )
                evaluation_data["temp_vbs"] = evaluation_data["uuid"].map(vbs_per_uuid)

                # Run bootstrapping analysis on RAW data
                print(
                    f"📊 Step 2: Running bootstrapping analysis (n={args.bootstrap_n})..."
                )
                analyzer = PassAtKAnalyzer(
                    evaluation_data, k_values=[1, 2, 4, 8, 16, 32, 64]
                )
                scaling_stats = analyzer.bootstrap_metrics(
                    n_bootstraps=args.bootstrap_n
                )

                # Save scaling stats
                stats_path = Path(args.output_dir) / "scaling_stats.csv"
                scaling_stats.to_csv(stats_path, index=False)
                print(f"   ✅ Saved scaling stats to {stats_path}")

                # Generate scaling plots
                analyzer.plot_scaling_laws(scaling_stats, args.output_dir)
                print(
                    "   ✅ Generated scaling plots (scaling_gap_vs_k.png/pdf, scaling_pass_vs_k.png/pdf)"
                )

                # STEP 2: Collapse to "Best-of-N" BEFORE VBS calculation
                # This is critical: we want to compare the Base Model's PEAK capability
                # (after 64 tries) as a single "super-agent" against Hero's single attempt
                print(
                    "🔄 Step 3: Collapsing to Best-of-N (picking best feasible score per UUID)..."
                )

                best_rows = []
                grouped = evaluation_data.groupby("uuid")

                for _uuid, group in grouped:
                    # Filter for feasible solutions first
                    feasible_group = group[group["feasible"]]

                    if not feasible_group.empty:
                        # Pick the sample with the HIGHEST score
                        best_idx = feasible_group["llm_score"].idxmax()
                        best_row = evaluation_data.loc[best_idx].copy()
                        best_rows.append(best_row)
                    else:
                        # If no solution worked, pick the first one (it's a fail anyway)
                        best_rows.append(group.iloc[0])

                # OVERWRITE evaluation_data with the collapsed version
                # Remove temporary VBS column (will be recalculated properly below)
                evaluation_data = pd.DataFrame(best_rows).reset_index(drop=True)
                if "temp_vbs" in evaluation_data.columns:
                    evaluation_data = evaluation_data.drop(columns=["temp_vbs"])

                print(
                    f"📉 Data collapsed: {len(results)} raw samples -> {len(evaluation_data)} unique problems (Best-of-N selected)."
                )
                print(
                    "   Note: VBS will now be calculated on the collapsed data for fair comparison."
                )
            else:
                print(
                    "⚠️  --best-of-n specified but only 1 sample per UUID detected. Skipping Best-of-N analysis."
                )

        # [NEW] 1. Calculate Virtual Best Score (VBS) per row
        # This happens AFTER Best-of-N collapsing (if applicable)
        # Now evaluation_data contains only the "champion" sample for Base Model, or single samples for other methods
        print("📊 Computing Virtual Best Solver (VBS) & Difficulty...")

        def calculate_vbs_and_difficulty(row):
            # A. Find VBS (Max of LLM + All Baselines)
            # Compute VBS from scratch (always, since we collapsed before this step)
            scores = []
            # LLM Score (only if feasible, preserve negative scores)
            if row.get("feasible", False):
                scores.append(row.get("llm_score", float("-inf")))
            # Baseline Scores (preserve negative scores, filter only -inf)
            for col in row.index:
                if col.startswith("score_"):
                    score_val = row[col]
                    if score_val > float("-inf"):  # Only exclude -inf (infeasible)
                        scores.append(score_val)
            # VBS is the absolute max found by anyone (can be negative)
            valid_scores = [s for s in scores if s > float("-inf")]
            vbs = float("-inf") if not valid_scores else max(valid_scores)

            # B. Calculate Difficulty (Gap vs Greedy)
            # [FIX] Robust gap formula: (VBS - Greedy) / (|VBS| + epsilon)
            greedy_score = row.get("score_greedy", float("-inf"))
            if greedy_score == float("-inf"):
                greedy_score = 0.0  # Anchor for gap calc

            if vbs == float("-inf") or vbs <= _EPSILON_SMALL:
                # Edge case: Everyone failed. Mark as "Hard" (infeasible or unsolvable)
                hardness = 1.0  # Max difficulty when no one can solve it
            elif greedy_score <= _EPSILON_SMALL:
                # Greedy failed completely. Max Difficulty.
                hardness = 1.0
            else:
                # [FIX] Standard relative gap formula
                epsilon = 1e-10
                numerator = vbs - greedy_score
                denominator = abs(vbs) + epsilon
                hardness = numerator / denominator

            return pd.Series([vbs, hardness], index=["vbs_score", "hardness"])

        # Apply calculation
        evaluation_data[["vbs_score", "hardness"]] = evaluation_data.apply(
            calculate_vbs_and_difficulty, axis=1
        )

        # [NEW] 2. Classify Difficulty Buckets
        def classify_diff(h):
            if h < _TRIVIAL_THRESHOLD:
                return "Trivial"  # <1% gap (Greedy ≈ Optimal)
            if h < _MODERATE_THRESHOLD:
                return "Moderate"  # 1-10% gap
            return "Hard"  # >10% gap (Greedy failed significantly)

        evaluation_data["difficulty_class"] = evaluation_data["hardness"].apply(
            classify_diff
        )

        # Save enriched CSV
        evaluation_data.to_csv(Path(args.output_dir) / "metrics_final.csv", index=False)
        evaluation_data[evaluation_data["feasible"]]

        generation_wall_clock_seconds = float(
            os.environ.get("GENERATION_WALL_CLOCK_SECONDS", "0.0")
        )
        timing_summary = {
            "run_type": determine_run_type(args),
            "method_name": primary_method_name,
            "seed": args.seed,
            "dataset": args.dataset or args.test_dataset or args.shinka_dataset,
            "input_file": args.input_file,
            "num_records_evaluated": len(lines),
            "num_unique_instances": int(
                evaluation_data["uuid"].nunique()
                if "uuid" in evaluation_data.columns
                else len(evaluation_data)
            ),
            "generation_wall_clock_seconds": generation_wall_clock_seconds,
            "evaluation_wall_clock_seconds": evaluation_wall_clock_seconds,
            "temperature": None,
            "n_samples": None,
            "model": args.model,
            "training_scheme": args.training_scheme,
            "job_id": args.job_id,
            "code_source_type": args.code_source_type,
            "code_source_path": args.code_source_path,
            "code_source_seed": args.code_source_seed,
            "code_label": args.code_label,
        }
        timing_path = write_timing_summary(args.output_dir, timing_summary)
        print(f"⏱️  Saved timing summary to {timing_path}")

    # --- CONSOLE SUMMARY ---
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)

    # LLM Feasibility
    print(
        f"\n🤖 {primary_method_name} Pass Rate: {evaluation_data['feasible'].mean():.2%} ({evaluation_data['feasible'].sum()}/{len(evaluation_data)})"
    )

    # Baseline Feasibility
    print("\n⚖️  Baseline Pass Rates:")
    for name in active_baselines_config:
        feasible_col = f"feasible_{name}"
        if feasible_col in evaluation_data.columns:
            pass_rate = evaluation_data[feasible_col].mean()
            pass_count = evaluation_data[feasible_col].sum()
            print(
                f"   {name.replace('_', ' ').title()}: {pass_rate:.2%} ({pass_count}/{len(evaluation_data)})"
            )

    # Check for instances where all solvers fail
    if len(active_baselines_config) > 0:
        all_fail_mask = pd.Series(
            [True] * len(evaluation_data), index=evaluation_data.index
        )
        for name in active_baselines_config:
            feasible_col = f"feasible_{name}"
            if feasible_col in evaluation_data.columns:
                all_fail_mask = all_fail_mask & (~evaluation_data[feasible_col])
        # Also check LLM
        all_fail_mask = all_fail_mask & (~evaluation_data["feasible"])
        all_fail_count = all_fail_mask.sum()
        if all_fail_count > 0:
            print(
                f"\n⚠️  Instances where ALL solvers (including LLM) failed: {all_fail_count}"
            )
            print(
                "   These may be infeasible problem instances or very hard instances."
            )

    print(f"\n🔍 {primary_method_name} Error Distribution:")
    print(evaluation_data["error_type"].value_counts().to_string())

    print(f"\n🚫 {primary_method_name} Constraint Violations (Infeasible Solutions):")
    infeasible_df = evaluation_data[~evaluation_data["feasible"]]
    if len(infeasible_df) > 0:
        for v_col in [
            "viol_cardinality",
            "viol_precedence",
            "viol_mutex",
            "viol_groups",
        ]:
            if v_col in infeasible_df.columns:
                count = (infeasible_df[v_col] > 0).sum()
                print(f"   {v_col}: {count}")
    else:
        print("   (All solutions feasible)")

    # --- 2. PLOTS (paper-standard) ---

    # Set paper-compatible plotting style (matches aggregate_plots.py)
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

    # Universal Palette (matches aggregate_plots.py)
    PALETTE = {  # noqa: N806
        "LLM (Ours)": "#1f77b4",  # Blue
        "Local Search": "#2ca02c",  # Green
        "Greedy": "#d62728",  # Red
        "CP-SAT": "#ff7f0e",  # Orange
        "BnB": "#9467bd",  # Purple
        "Random": "#7f7f7f",  # Gray
    }
    if primary_method_name not in PALETTE:
        PALETTE[primary_method_name] = PALETTE["LLM (Ours)"]

    # Method name mapping for baselines
    BASELINE_MAPPING = {  # noqa: N806
        "greedy": "Greedy",
        "local_search": "Local Search",
        "cpsat": "CP-SAT",
        "bnb": "BnB",
    }

    print("\n📊 Generating paper-standard plots...")

    # Prepare data: Calculate Optimality Gap for all methods
    if len(evaluation_data) > 0 and "vbs_score" in evaluation_data.columns:
        # Calculate gaps for LLM
        evaluation_data["llm_gap"] = evaluation_data.apply(
            lambda row: (
                (
                    row["vbs_score"]
                    - max(0, row["llm_score"] if row.get("feasible", False) else 0.0)
                )
                / row["vbs_score"]
            )
            if row["vbs_score"] > _EPSILON_SMALL
            else np.nan,
            axis=1,
        )

        # Calculate gaps for baselines
        for b_name in active_baselines_config:
            col_name = f"score_{b_name}"
            if col_name in evaluation_data.columns:
                feasible_col = f"feasible_{b_name}"
                if feasible_col in evaluation_data.columns:
                    # Capture loop variables in closure
                    _col_name = col_name
                    _feasible_col = feasible_col

                    def calc_gap(row, cn=_col_name, fc=_feasible_col):
                        if row["vbs_score"] > _EPSILON_SMALL:
                            return (
                                row["vbs_score"] - max(0, row[cn] if row[fc] else 0.0)
                            ) / row["vbs_score"]
                        return np.nan

                    evaluation_data[f"{b_name}_gap"] = evaluation_data.apply(
                        calc_gap, axis=1
                    )
                else:
                    # Capture loop variable in closure
                    _col_name = col_name

                    def calc_gap_fallback(row, cn=_col_name):
                        if row["vbs_score"] > _EPSILON_SMALL:
                            score_val = (
                                row[cn]
                                if np.isfinite(row[cn]) and row[cn] > float("-inf")
                                else 0.0
                            )
                            return (row["vbs_score"] - max(0, score_val)) / row[
                                "vbs_score"
                            ]
                        return np.nan

                    evaluation_data[f"{b_name}_gap"] = evaluation_data.apply(
                        calc_gap_fallback, axis=1
                    )

    # --- Plot 1: Robustness Profile (Fraction Solved vs Optimality Gap) ---
    if len(evaluation_data) > 0 and "vbs_score" in evaluation_data.columns:
        print("  Generating Robustness Profile...")

        _fig, ax = plt.subplots(figsize=(3.25, 2.5))

        # X-axis: Gap Thresholds (0% to 50% for readability)
        taus = np.linspace(0.0, 0.5, 500)

        # Plot Methods (in order of importance)
        methods_to_plot = []
        methods_to_plot.append((primary_method_name, "llm_gap", PALETTE["LLM (Ours)"]))
        for b_name in active_baselines_config:
            gap_col = f"{b_name}_gap"
            if gap_col in evaluation_data.columns:
                method_name = BASELINE_MAPPING.get(
                    b_name, b_name.replace("_", " ").title()
                )
                methods_to_plot.append(
                    (method_name, gap_col, PALETTE.get(method_name, "black"))
                )

        for method_name, gap_col, color in methods_to_plot:
            gaps = evaluation_data[gap_col].dropna().to_numpy()
            if len(gaps) == 0:
                continue

            # CDF: Fraction with Gap <= tau for each tau
            y = np.mean(gaps[:, None] <= taus[None, :], axis=0)

            # Plot styling
            lw = 2.0 if method_name == primary_method_name else 1.5
            alpha = 1.0 if method_name == primary_method_name else 0.8
            linestyle = "-" if method_name == primary_method_name else "--"

            ax.plot(
                taus,
                y,
                label=method_name,
                color=color,
                lw=lw,
                alpha=alpha,
                linestyle=linestyle,
            )

        # Formatting
        ax.set_xlabel(r"Optimality Gap ($\tau$)", fontsize=10)
        ax.set_ylabel(r"Fraction Solved ($g \leq \tau$)", fontsize=10)
        ax.set_xlim(0.0, 0.5)  # Focus on the "good" region (0% to 50% gap)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

        ax.legend(loc="lower right", fontsize=7, framealpha=0.95)
        ax.grid(True, alpha=0.2)

        plt.tight_layout(pad=0.2)
        # Save both PNG and PDF
        robust_path = Path(args.output_dir) / "robustness_profile.png"
        plt.savefig(robust_path, dpi=300, bbox_inches="tight")
        plt.savefig(robust_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close()
        print("   ✅ Saved robustness_profile.png/pdf")

    # --- Plot 2: Stratified Box Plot (Optimality Gap vs Difficulty) ---
    if (
        len(evaluation_data) > 0
        and "difficulty_class" in evaluation_data.columns
        and "vbs_score" in evaluation_data.columns
    ):
        print("  Generating Stratified Box Plot...")

        # Prepare data for box plot
        plot_data = []

        # Add LLM data
        if "llm_gap" in evaluation_data.columns:
            for _idx, row in evaluation_data.iterrows():
                if pd.notna(row.get("llm_gap")) and row.get("difficulty_class") in [
                    "Trivial",
                    "Moderate",
                    "Hard",
                ]:
                    plot_data.append(
                        {
                            "Method": primary_method_name,
                            "GapPct": row["llm_gap"] * 100,
                            "difficulty_class": row["difficulty_class"],
                        }
                    )

        # Add baseline data
        for b_name in active_baselines_config:
            gap_col = f"{b_name}_gap"
            if gap_col in evaluation_data.columns:
                method_name = BASELINE_MAPPING.get(
                    b_name, b_name.replace("_", " ").title()
                )
                for _idx, row in evaluation_data.iterrows():
                    if pd.notna(row.get(gap_col)) and row.get("difficulty_class") in [
                        "Trivial",
                        "Moderate",
                        "Hard",
                    ]:
                        plot_data.append(
                            {
                                "Method": method_name,
                                "GapPct": row[gap_col] * 100,
                                "difficulty_class": row["difficulty_class"],
                            }
                        )

        if plot_data:
            plot_df = pd.DataFrame(plot_data)

            # Order: Trivial -> Moderate -> Hard
            plot_df["difficulty_class"] = pd.Categorical(
                plot_df["difficulty_class"],
                categories=["Trivial", "Moderate", "Hard"],
                ordered=True,
            )

            # Filter methods to plot
            methods = [primary_method_name, "Local Search", "Greedy", "BnB", "CP-SAT"]
            plot_df = plot_df[plot_df["Method"].isin(methods)].copy()

            if not plot_df.empty:
                _fig, ax = plt.subplots(figsize=(6.75, 2.5))  # Two-column width

                # Separate CP-SAT (which is always optimal, gap=0) from other methods
                cpsat_df = plot_df[plot_df["Method"] == "CP-SAT"].copy()
                other_df = plot_df[plot_df["Method"] != "CP-SAT"].copy()

                # Plot boxplots for non-CP-SAT methods
                if not other_df.empty:
                    sns.boxplot(
                        data=other_df,
                        x="difficulty_class",
                        y="GapPct",
                        hue="Method",
                        palette=PALETTE,
                        linewidth=0.8,
                        fliersize=1,
                        ax=ax,
                        saturation=0.9,
                    )

                # Plot CP-SAT as a horizontal line at 0% gap (optimal) with annotation
                if not cpsat_df.empty:
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

                    # Add CP-SAT to legend manually (as a line, not a box)
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

                ax.set_ylabel(r"Optimality Gap (\%)", fontsize=10)
                ax.set_xlabel("Difficulty", fontsize=10)
                ax.set_ylim(-2, 50)  # Focus on meaningful gaps
                ax.grid(True, axis="y", alpha=0.2)

                plt.tight_layout(pad=0.2)
                # Save both PNG and PDF
                boxplot_path = Path(args.output_dir) / "stratified_boxplot.png"
                plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
                plt.savefig(boxplot_path.with_suffix(".pdf"), bbox_inches="tight")
                plt.close()
                print("   ✅ Saved stratified_boxplot.png/pdf")

    # --- Plot 3: Error Distribution (paper-ready, no title) ---
    print("  Generating Error Distribution...")

    # Count error types
    error_counts = evaluation_data["error_type"].value_counts()

    # Map error types to readable labels
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

    # Create readable labels
    labels = [
        error_labels.get(err, err.replace("_", " ").title())
        for err in error_counts.index
    ]

    # Color scheme: Use a professional color palette
    colors = [
        "#2ecc71"
        if err == "none"
        else "#e74c3c"
        if err in ["syntax", "runtime", "constraint"]
        else "#f39c12"
        if err == "timeout"
        else "#3498db"
        for err in error_counts.index
    ]

    _fig, ax = plt.subplots(figsize=(6.75, 2.5))  # Two-column width

    # Create horizontal bar chart for better readability
    ax.barh(
        range(len(error_counts)),
        error_counts.values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels on bars
    for i, (_idx, val) in enumerate(error_counts.items()):
        ax.text(
            val + max(error_counts.values) * 0.01,
            i,
            f"{int(val)}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks(range(len(error_counts)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Count", fontsize=10)
    ax.set_xlim(0, max(error_counts.values) * 1.15)  # Add padding for labels
    ax.grid(True, axis="x", alpha=0.2, linestyle="--")
    ax.invert_yaxis()  # Top to bottom: most common at top

    plt.tight_layout(pad=0.2)
    # Save both PNG and PDF
    error_path = Path(args.output_dir) / "error_distribution.png"
    plt.savefig(error_path, dpi=300, bbox_inches="tight")
    plt.savefig(error_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print("   ✅ Saved error_distribution.png/pdf")

    # --- 3. LATEX TABLE ---
    print("\n📝 Generating LaTeX Table...")

    # [NEW] Stratified Table Generation (VBS-Based)
    # We want columns: Method | Overall Gap | Trivial Gap | Moderate Gap | Hard Gap
    if (
        len(evaluation_data) > 0
        and "difficulty_class" in evaluation_data.columns
        and "vbs_score" in evaluation_data.columns
    ):
        print("📊 Generating stratified difficulty table...")

        def get_gap_stats(method_score_col, subset_df):
            # Filter for valid VBS > 0
            valid = subset_df["vbs_score"] > _EPSILON_SMALL
            if not valid.any():
                return 0.0

            # Calculate Gap: (VBS - Method) / VBS
            # Treat infeasible method scores as 0.0
            vbs = subset_df.loc[valid, "vbs_score"]
            method_scores = subset_df.loc[valid, method_score_col].clip(lower=0.0)

            gaps = (vbs - method_scores) / vbs
            return gaps.mean() * 100

        strat_rows = []

        # 1. Baselines
        for name in active_baselines_config:
            col = f"score_{name}"
            if col not in evaluation_data.columns:
                continue

            row_data = {"Method": name.replace("_", " ").title()}
            row_data["Overall"] = get_gap_stats(col, evaluation_data)
            for diff in ["Trivial", "Moderate", "Hard"]:
                subset = evaluation_data[evaluation_data["difficulty_class"] == diff]
                row_data[diff] = get_gap_stats(col, subset) if len(subset) > 0 else 0.0
            strat_rows.append(row_data)

        # 2. LLM (Ours)
        if "llm_score" in evaluation_data.columns:
            # Create a temporary column that fills NaN/Infeasible with 0
            evaluation_data["llm_score_safe"] = evaluation_data.apply(
                lambda r: r["llm_score"] if r.get("feasible", False) else 0.0, axis=1
            )

            row_data = {"Method": primary_method_name}
            row_data["Overall"] = get_gap_stats("llm_score_safe", evaluation_data)
            for diff in ["Trivial", "Moderate", "Hard"]:
                subset = evaluation_data[evaluation_data["difficulty_class"] == diff]
                row_data[diff] = (
                    get_gap_stats("llm_score_safe", subset) if len(subset) > 0 else 0.0
                )
            strat_rows.append(row_data)

        # Save Stratified Table
        if strat_rows:
            df_strat_table = pd.DataFrame(strat_rows)
            # Reorder columns
            cols = ["Method", "Overall", "Trivial", "Moderate", "Hard"]
            df_strat_table = df_strat_table[cols]

            latex_strat = df_strat_table.to_latex(float_format="%.1f", index=False)
            with (Path(args.output_dir) / "results_stratified.tex").open("w") as f:
                f.write(latex_strat)
            print("   ✅ Saved results_stratified.tex")

    # 1. Determine the "Optimal" reference (VBS - Virtual Best Solver)
    # Use vbs_score which is the max across all solvers (including LLM)
    ref_col = (
        "vbs_score"
        if "vbs_score" in evaluation_data.columns
        else (
            "score_cpsat"
            if "score_cpsat" in evaluation_data.columns
            else "best_known_score"
        )
    )

    # Helper to calc mean gap %: mean( (VBS - Score) / VBS )
    def calc_optimality_gap(score_col, opt_col):
        # Avoid division by zero
        valid = evaluation_data[opt_col] > _EPSILON_MEDIUM
        if not valid.any():
            return 0.0
        gaps = (
            evaluation_data.loc[valid, opt_col]
            - evaluation_data.loc[valid, score_col].clip(lower=0.0)
        ) / evaluation_data.loc[valid, opt_col]
        # Clip negative gaps (in case heuristic accidentally beats VBS)
        gaps = gaps.clip(lower=0.0)
        return gaps.mean() * 100

    raw_rows = []

    # Process Baselines
    for name in active_baselines_config:
        col_name = f"score_{name}"
        if col_name in evaluation_data.columns:
            # Calculate actual pass rate from feasibility column
            feasible_col = f"feasible_{name}"
            if feasible_col in evaluation_data.columns:
                b_pass = evaluation_data[feasible_col].mean() * 100
                # Score: mean of feasible solutions only
                feasible_mask = evaluation_data[feasible_col]
                if feasible_mask.any():
                    b_score = evaluation_data.loc[feasible_mask, col_name].mean()
                else:
                    b_score = 0.0  # No feasible solutions
            else:
                # Fallback: check if score is finite and > -inf
                # A score is feasible if it's not -inf and is finite
                is_finite = evaluation_data[col_name].apply(
                    lambda x: np.isfinite(x) and x > float("-inf")
                )
                b_pass = is_finite.mean() * 100
                # Score: mean of feasible solutions only
                b_score = (
                    evaluation_data.loc[is_finite, col_name].mean()
                    if is_finite.any()
                    else 0.0
                )

            b_time = evaluation_data[f"time_{name}"].mean()
            # Handle core seconds logic
            b_core_sec = (
                evaluation_data[f"core_sec_{name}"].mean()
                if f"core_sec_{name}" in evaluation_data.columns
                else b_time
            )

            # Calculate GAP to Optimality (CP-SAT)
            gap = calc_optimality_gap(col_name, ref_col)

            # Calculate Win Rate vs Greedy (only count feasible solutions)
            if "score_greedy" in evaluation_data.columns:
                # Only compare when both are feasible
                if feasible_col in evaluation_data.columns:
                    both_feasible = evaluation_data[feasible_col] & (
                        evaluation_data["score_greedy"] > float("-inf")
                    )
                    wins = (
                        both_feasible
                        & (evaluation_data[col_name] > evaluation_data["score_greedy"])
                    ).mean() * 100
                else:
                    wins = (
                        evaluation_data[col_name] > evaluation_data["score_greedy"]
                    ).mean() * 100
            else:
                wins = 0.0

            raw_rows.append(
                {
                    "Method": name.replace("_", " ").title(),
                    "Pass": b_pass,
                    "Score": b_score,
                    "Time": b_time,
                    "CoreSec": b_core_sec,
                    "Gap": gap,  # Now this is Gap%, e.g., 5.4
                    "Win": wins,
                }
            )

    # Process LLM
    llm_pass = evaluation_data["feasible"].mean() * 100
    if evaluation_data["feasible"].any():
        llm_feasible = evaluation_data[evaluation_data["feasible"]]
        llm_score = llm_feasible["llm_score"].mean()
        llm_time = llm_feasible["execution_time"].mean()
        llm_core_sec = llm_feasible["llm_core_sec"].mean()

        # Calculate Gap for LLM (treat infeasible as 0.0 for gap calculation)
        # Use vbs_score as reference
        llm_scores_safe = evaluation_data.apply(
            lambda r: r["llm_score"] if r["feasible"] else 0.0, axis=1
        )
        valid = evaluation_data[ref_col] > _EPSILON_MEDIUM
        if valid.any():
            gaps = (
                evaluation_data.loc[valid, ref_col]
                - llm_scores_safe.loc[valid].clip(lower=0.0)
            ) / evaluation_data.loc[valid, ref_col]
            llm_gap = gaps.clip(lower=0.0).mean() * 100
        else:
            llm_gap = 0.0

        if "score_greedy" in evaluation_data.columns:
            # Win rate includes infeasible (they lose)
            llm_wins = (
                (evaluation_data["feasible"])
                & (evaluation_data["llm_score"] > evaluation_data["score_greedy"])
            ).mean() * 100
        else:
            llm_wins = 0.0

        raw_rows.append(
            {
                "Method": primary_method_name,
                "Pass": llm_pass,
                "Score": llm_score,
                "Time": llm_time,
                "CoreSec": llm_core_sec,
                "Gap": llm_gap,
                "Win": llm_wins,
            }
        )
    else:
        raw_rows.append(
            {
                "Method": primary_method_name,
                "Pass": 0.0,
                "Score": -1.0,
                "Time": 999.0,
                "CoreSec": 999.0,
                "Gap": 0.0,
                "Win": 0.0,
            }
        )

    df_table = pd.DataFrame(raw_rows)

    # Formatting helper
    def fmt_best(val, values, is_min=False, fmt="{:.2f}", suffix=""):
        # Handle inf/-inf cases
        if not np.isfinite(val):
            if val == float("-inf"):
                return "-inf"
            elif val == float("inf"):
                return "inf"
            else:
                return "NaN"

        # Filter out non-finite values for best calculation
        finite_values = [v for v in values if np.isfinite(v)]
        if not finite_values:
            # All values are non-finite, just format this one
            s = fmt.format(val) + suffix
            return s

        # Determine best value in column (only finite values)
        best = min(finite_values) if is_min else max(finite_values)
        s = fmt.format(val) + suffix
        # Close enough equality check
        if abs(val - best) < _EPSILON_EQUALITY:
            return f"\\textbf{{{s}}}"
        return s

    if len(df_table) > 0:
        final_rows = []
        for _, row in df_table.iterrows():
            name = (
                f"\\textbf{{{row['Method']}}}"
                if row["Method"] == primary_method_name
                else row["Method"]
            )

            final_rows.append(
                {
                    "Method": name,
                    r"Pass ($\uparrow$)": fmt_best(
                        row["Pass"], df_table["Pass"], False, "{:.1f}", "\\%"
                    ),
                    r"Score ($\uparrow$)": fmt_best(
                        row["Score"], df_table["Score"], False, "{:.1f}"
                    ),
                    r"Time (s) ($\downarrow$)": fmt_best(
                        row["Time"], df_table["Time"], True, "{:.4f}"
                    ),
                    r"Core$\times$s ($\downarrow$)": fmt_best(
                        row["CoreSec"], df_table["CoreSec"], True, "{:.4f}"
                    ),
                    # Gap is MINIMIZED (closer to 0 is better)
                    r"Gap ($\downarrow$)": fmt_best(
                        row["Gap"], df_table["Gap"], True, "{:.2f}", "\\%"
                    ),
                    "Win Rate": f"{row['Win']:.1f}\\%",
                }
            )

        latex_str = pd.DataFrame(final_rows).to_latex(
            index=False, escape=False, column_format="lcccccc"
        )

        # Save
        with (Path(args.output_dir) / "results_table.tex").open("w") as f:
            f.write(latex_str)

    print(f"✅ Done. Results in {args.output_dir}")

    # Log to W&B if requested
    if args.log_to_wandb:
        # df is defined in both plot-only and non-plot-only modes
        log_to_wandb(
            output_dir=args.output_dir,
            model=args.model,
            training_scheme=args.training_scheme,
            seed=args.seed,
            job_id=args.job_id,
            shinka_dataset=args.shinka_dataset,
            evaluate_on_own_dataset=args.evaluate_on_own_dataset,
            df=evaluation_data,  # evaluation_data is available in both modes
        )


if __name__ == "__main__":
    main()
