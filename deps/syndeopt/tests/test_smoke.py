from syndeopt.bench.suites import basic_suite
from syndeopt.solvers import list_solvers, get_solver
from syndeopt.core.scoring import score
from syndeopt.core.feasibility import feasible

def test_smoke():
    suite = basic_suite(seed=1)
    solvers = list(list_solvers().keys())

    # just run a tiny budget to make sure things don't crash
    # Catch RuntimeError for solvers that aren't available
    rows = []
    for inst_name, inst in suite:
        for sname in solvers:
            try:
                solver = get_solver(sname)
                res = solver.solve(inst, budget_sec=0.5, seed=0)
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
                # Skip unavailable solvers (e.g., missing optional dependencies)
                pass

    assert len(rows) > 0
