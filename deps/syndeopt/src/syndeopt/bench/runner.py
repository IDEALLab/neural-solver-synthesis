from typing import List, Dict, Any
from time import perf_counter
from ..core.scoring import score
from ..core.feasibility import feasible
from ..solvers.base import get_solver, SolveResult
from ..core.instance import SDSInstance


def run_suite(
    suite: List[tuple],      # list of (name, SDSInstance)
    solver_names: List[str],
    budget_sec: float = 5.0,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Run a benchmark suite.

    Each solver is called with the budget_sec parameter and is expected to
    respect it. The runner measures actual elapsed time and records it.

    Args:
        suite: List of (instance_name, SDSInstance) tuples
        solver_names: List of solver names to run
        budget_sec: Time budget per instance/solver combination
        seed: Random seed

    Returns:
        List of result dictionaries with keys: instance, solver, score, time_sec, feasible, gap, extras
    """
    rows: List[Dict[str, Any]] = []

    for inst_name, inst in suite:
        for sname in solver_names:
            try:
                solver = get_solver(sname)
                start_time = perf_counter()

                res = solver.solve(inst, budget_sec=budget_sec, seed=seed)

                # Measure actual elapsed time (more accurate than solver-reported time)
                elapsed = perf_counter() - start_time
                res.time_sec = elapsed

                # Validate result
                feas = feasible(inst, res.mask)
                true_score = score(inst, res.mask) if feas else float("-inf")

                rows.append({
                    "instance": inst_name,
                    "solver": sname,
                    "score": true_score,
                    "time_sec": res.time_sec,
                    "feasible": feas,
                    "gap": res.gap,
                    "extras": res.extras,
                })
            except RuntimeError:
                # Solver not available (e.g., missing dependencies) - skip silently
                continue
            except Exception as e:
                # Other errors - record as failed
                rows.append({
                    "instance": inst_name,
                    "solver": sname,
                    "score": float("-inf"),
                    "time_sec": 0.0,
                    "feasible": False,
                    "gap": None,
                    "extras": {"error": str(e)},
                })

    return rows
