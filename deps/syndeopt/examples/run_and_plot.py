from syndeopt.bench.suites import basic_suite
from syndeopt.bench.runner import run_suite
from syndeopt.bench.io import save_results_csv
from syndeopt.bench.viz import performance_profile, anytime_curve
from syndeopt.solvers import get_solver

if __name__ == "__main__":
    # 1) Run a basic suite with a few solvers
    suite = basic_suite(seed=0)
    solver_names = ["greedy", "local_search", "cpsat"]

    rows = run_suite(suite, solver_names, budget_sec=5.0, seed=0)
    save_results_csv(rows, "results_basic_suite.csv")

    # 2) Plot performance profile (score-based)
    performance_profile(rows, outfile="perf_profile.png", show=False)

    # 3) Example anytime curve: single instance + cpsat trace
    inst_name, inst = suite[0]
    cpsat = get_solver("cpsat")
    res = cpsat.solve(inst, budget_sec=5.0, seed=0)

    traces = {
        "cpsat": res.trace or [],  # list of (time, score)
    }
    anytime_curve(
        traces,
        outfile=f"anytime_{inst_name}_cpsat.png",
        title=f"Anytime curve on {inst_name}",
        show=False,
    )

    print("Saved perf_profile.png and anytime_*_cpsat.png")
