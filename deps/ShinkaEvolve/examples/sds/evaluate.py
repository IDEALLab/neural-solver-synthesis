"""
Evaluator for Synergistic Dependency Selection (SDS) problem.
Uses ShinkaEvolve's run_shinka_eval pattern.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from shinka.core import run_shinka_eval


def _check_feasibility(selection: list[int], requirements: dict[str, Any]) -> tuple[bool, int]:
    """
    Check if a selection satisfies all feasibility constraints.
    Returns (is_valid, violation_count).
    """
    sel_set = set(selection)
    violations = 0
    
    # A. Cardinality
    L, U = requirements.get("cardinality_bounds", [0, 9999])  # noqa: N806
    if not (L <= len(sel_set) <= U):
        violations += 1
        
    # B. Mutex
    for a, b in requirements.get("mutex", []):
        if a in sel_set and b in sel_set:
            violations += 1
            
    # C. Groups (at most one per group)
    for grp_vars in requirements.get("groups", {}).values():
        if len(set(grp_vars).intersection(sel_set)) > 1:
            violations += 1
    
    # D. Precedence (i -> j means if j selected, i must be selected)
    for i, j in requirements.get("precedence", []):
        if j in sel_set and i not in sel_set:
            violations += 1
            
    return (violations == 0), violations


def _calculate_score(selection: list[int], requirements: dict[str, Any]) -> float:
    """
    Calculate the objective score for a valid selection.
    """
    sel_set = set(selection)
    total = 0.0
    
    # Unary weights
    weights = requirements.get("weights", [])
    for i in selection:
        if i < len(weights):
            total += weights[i]
            
    # Pairwise interactions
    interactions = requirements.get("interactions", {})
    for k, v in interactions.items():
        try:
            u, w = map(int, k.split(","))
            if u in sel_set and w in sel_set:
                total += v
        except Exception:
            pass
            
    return total


def validate_sds(run_output: tuple[str, dict[str, Any]]) -> tuple[bool, str | None]:
    """
    Validates SDS solution output.
    
    Args:
        run_output: Tuple of (stdout_str, problem_data_dict)
    
    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    stdout_str, problem_data = run_output
    requirements = problem_data.get("requirements", {})
    
    if not stdout_str.strip():
        return False, "No output produced"
    
    try:
        out_json = json.loads(stdout_str)
        selection = out_json.get("selection", {}).get("variables", [])
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON output: {e}"
    
    # Check feasibility
    valid, violation_count = _check_feasibility(selection, requirements)
    if not valid:
        return False, f"Feasibility constraint violated (violations: {violation_count})"
    
    return True, None


def get_sds_kwargs(run_index: int) -> dict[str, Any]:  # noqa: ARG001
    """
    Provides keyword arguments for SDS runs.
    The problem data is passed via environment variable SDS_PROBLEM_DATA.
    """
    # Problem data is passed via environment to avoid re-parsing
    problem_data_str = os.environ.get("SDS_PROBLEM_DATA", "{}")
    try:
        problem_data = json.loads(problem_data_str)
    except Exception:
        problem_data = {}
    
    return {"problem_data": problem_data}


def aggregate_sds_metrics(
    results: list[tuple[str, dict[str, Any]]], results_dir: str  # noqa: ARG001
) -> dict[str, Any]:
    """
    Aggregates metrics for SDS evaluation.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}
    
    stdout_str, problem_data = results[0]
    requirements = problem_data.get("requirements", {})
    
    try:
        out_json = json.loads(stdout_str)
        selection = out_json.get("selection", {}).get("variables", [])
    except Exception:
        return {"combined_score": 0.0, "error": "Failed to parse output"}
    
    # Check feasibility
    valid, violation_count = _check_feasibility(selection, requirements)
    if not valid:
        # Soft penalty for infeasible solutions
        return {
            "combined_score": 0.1 / (1 + violation_count),
            "public": {"feasible": False, "violations": violation_count},
            "private": {"selection": selection},
        }
    
    # Calculate score
    raw_score = _calculate_score(selection, requirements)
    
    # Normalize score for fitness (0-1 range)
    # Estimate max score as sum of positive weights + interactions
    max_est = sum(max(0, w) for w in requirements.get("weights", []))
    max_est += sum(max(0, v) for v in requirements.get("interactions", {}).values())
    if max_est == 0:
        max_est = 1.0
    
    fitness = max(0.1, min(1.0, raw_score / max_est))
    
    public_metrics = {
        "feasible": True,
        "selection_size": len(selection),
        "raw_score": float(raw_score),
    }
    
    private_metrics = {
        "selection": selection,
        "raw_score": float(raw_score),
        "normalized_fitness": float(fitness),
    }
    
    return {
        "combined_score": float(fitness),
        "public": public_metrics,
        "private": private_metrics,
    }


# Removed - using run_shinka_eval directly now


def main(program_path: str, results_dir: str):
    """
    Main evaluation function for SDS problem.
    """
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Get problem data from environment
    problem_data_str = os.environ.get("SDS_PROBLEM_DATA", "{}")
    try:
        problem_data = json.loads(problem_data_str)
    except Exception:
        print("Warning: Could not parse SDS_PROBLEM_DATA, using empty dict")
        problem_data = {}
    
    _requirements = problem_data.get("requirements", {})
    _catalog = problem_data.get("catalog", {})
    
    # Use run_shinka_eval pattern
    def _aggregator_with_context(r: list[str]) -> dict[str, Any]:
        # r is a list of return values (JSON strings) from run_sds
        stdout_str = r[0] if r else ""
        return aggregate_sds_metrics([(stdout_str, problem_data)], results_dir)
    
    def _validate_wrapper(run_output: str) -> tuple[bool, str | None]:
        # run_output is the JSON string returned from run_sds
        return validate_sds((run_output, problem_data))
    
    # Get timeout from environment (default 5 seconds for SDS evaluation)
    eval_timeout = float(os.environ.get("SDS_EVAL_TIMEOUT", "5.0"))
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_sds",
        num_runs=1,
        get_experiment_kwargs=lambda _: {"problem_data": problem_data},
        validate_fn=_validate_wrapper,
        aggregate_metrics_fn=_aggregator_with_context,
        timeout=eval_timeout,
    )
    
    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")
    
    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDS evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_sds')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)

