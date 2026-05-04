"""
NeurIPS-style benchmark script for SYNDEOPT.

- Generates a cross-regime suite of SDS/QUBO instances
- Runs a selected set of solvers
- Saves:
    - results_neurips.csv       (raw results)
    - perf_profile_neurips.png  (score-based performance profile)
    - bar_norm_score_overall.png
    - bar_norm_score_family_<family>.png
    - summary_by_solver.csv
    - summary_by_solver.tex
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from syndeopt.bench.io import save_results_csv
from syndeopt.bench.runner import run_suite
from syndeopt.bench.viz import performance_profile
from syndeopt.gen import (
    make_decomposable_instance,
    make_dense_deceptive_instance,
    make_greedy_easy_instance,
    make_local_optima_instance,
    make_maxcut_qubo_instance,
    make_planted_qubo_instance,
    make_random_qubo_instance,
    make_tree_showcase_instance,
)
from syndeopt.solvers import list_solvers

# ---------------------------------------------------------------------
# 1. Experiment configuration
# ---------------------------------------------------------------------

FAMILIES: list[tuple[str, Callable[..., Any], dict[str, Any]]] = [
    ("greedy_easy", make_greedy_easy_instance, {"n": 12, "card": (5, 5)}),
    ("local_optima", make_local_optima_instance, {"n": 18, "card": (5, 5)}),
    ("tree_showcase", make_tree_showcase_instance, {"n": 14, "card": (4, 10)}),
    ("decomposable", make_decomposable_instance, {"n": 18, "card": (6, 6)}),
    ("dense_deceptive", make_dense_deceptive_instance, {"n": 20, "card": (7, 11)}),
    ("qubo_random", make_random_qubo_instance, {"n": 20, "card": (0, 20)}),
    ("qubo_planted", make_planted_qubo_instance, {"n": 20, "card": (0, 20)}),
    ("qubo_maxcut", make_maxcut_qubo_instance, {"n": 20, "card": (0, 20), "edge_prob": 0.4}),
]

INSTANCES_PER_FAMILY = 10
BASE_SEED = 0
BEST_SCORE_THRESHOLD = 0.999  # Threshold for considering a score as "best"

# Pick a subset of solvers for the main paper-style benchmark.
# We'll intersect this list with the actually registered solvers.
PREFERRED_SOLVERS = [
    "greedy",
    "local_search",
    "cpsat",
    "bnb",  # built-in branch-and-bound (no external dependencies)
]


# ---------------------------------------------------------------------
# 2. Build full suite: many instances across families / seeds
# ---------------------------------------------------------------------

def build_neurips_suite() -> list[tuple[str, Any]]:
    suite: list[tuple[str, Any]] = []
    for family_idx, (family_name, gen_fn, kwargs) in enumerate(FAMILIES):
        for k in range(INSTANCES_PER_FAMILY):
            seed = BASE_SEED + family_idx * 1000 + k
            inst = gen_fn(seed=seed, **kwargs)
            inst_name = f"{family_name}_s{seed}_i{k}"
            suite.append((inst_name, inst))
    return suite


def infer_family_from_name(inst_name: str) -> str:
    # By construction, instance names start with family_name + "_"
    return inst_name.split("_")[0]


# ---------------------------------------------------------------------
# 3. Run benchmark
# ---------------------------------------------------------------------

def main():  # noqa: PLR0915
    # 3.1 Build suite
    suite = build_neurips_suite()
    print(f"Built suite with {len(suite)} instances.")
    print("Families:", sorted({infer_family_from_name(name) for name, _ in suite}))

    # 3.2 Solvers
    registered = list(list_solvers().keys())
    solvers = [s for s in PREFERRED_SOLVERS if s in registered]
    print("Registered solvers:", registered)
    print("Using solvers:", solvers)

    # 3.3 Run experiments
    rows = run_suite(
        suite=suite,
        solver_names=solvers,
        budget_sec=1.0,  # time budget per instance/solver
        seed=0,
    )
    save_results_csv(rows, "results_neurips.csv")
    print(f"Saved results_neurips.csv with {len(rows)} rows")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(rows)
    # add 'family' column
    df["family"] = df["instance"].apply(infer_family_from_name)

    # -----------------------------------------------------------------
    # 4. Performance profile (overall)
    # -----------------------------------------------------------------
    print("Plotting overall performance profile...")
    performance_profile(rows, outfile="perf_profile_neurips.png", show=False)
    print("Saved perf_profile_neurips.png")

    # -----------------------------------------------------------------
    # 5. Normalized score: overall & per family
    # -----------------------------------------------------------------
    print("Computing normalized scores...")

    # per-instance best score
    best_scores = df.groupby("instance")["score"].transform("max")

    def norm_score(row):
        s = float(row["score"])
        best = float(row["best_score"])
        # If no solver did better than 0, treat them equally (norm=1)
        if best <= 0.0 or not math.isfinite(best):
            return 1.0 if math.isfinite(s) else 0.0
        if not row.get("feasible", True) or not math.isfinite(s):
            return 0.0
        return s / best

    df["best_score"] = best_scores
    df["norm_score"] = df.apply(norm_score, axis=1)

    # Overall mean normalized score per solver
    overall = (
        df.groupby("solver")
        .agg(
            mean_norm_score=("norm_score", "mean"),
            frac_best=("norm_score", lambda x: (x >= BEST_SCORE_THRESHOLD).mean()),
            mean_time=("time_sec", "mean"),
        )
        .reset_index()
        .sort_values("mean_norm_score", ascending=False)
    )

    overall.to_csv("summary_by_solver.csv", index=False)
    try:
        overall.to_latex("summary_by_solver.tex", index=False, float_format="%.3f")
    except Exception:
        print("Could not write summary_by_solver.tex (LaTeX not needed, skipping).")

    print("Saved summary_by_solver.csv and summary_by_solver.tex")

    # -----------------------------------------------------------------
    # 6. Bar plots: overall & per family
    # -----------------------------------------------------------------

    # 6.1 Overall mean normalized score
    plt.figure(figsize=(6, 4))
    x = range(len(overall))
    plt.bar(x, overall["mean_norm_score"])
    plt.xticks(x, overall["solver"], rotation=45, ha="right")
    plt.ylabel("Mean normalized score")
    plt.ylim(0, 1.05)
    plt.title("Overall mean normalized score (all families)")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.savefig("bar_norm_score_overall.png", dpi=200)
    plt.close()
    print("Saved bar_norm_score_overall.png")

    # 6.2 Per family bar plots
    families = sorted(df["family"].unique())
    for fam in families:
        sub = df[df["family"] == fam]
        fam_stats = (
            sub.groupby("solver")
            .agg(
                mean_norm_score=("norm_score", "mean"),
                frac_best=("norm_score", lambda x: (x >= BEST_SCORE_THRESHOLD).mean()),
                mean_time=("time_sec", "mean"),
            )
            .reset_index()
            .sort_values("mean_norm_score", ascending=False)
        )

        plt.figure(figsize=(6, 4))
        xs = range(len(fam_stats))
        plt.bar(xs, fam_stats["mean_norm_score"])
        plt.xticks(xs, fam_stats["solver"], rotation=45, ha="right")
        plt.ylabel("Mean normalized score")
        plt.ylim(0, 1.05)
        plt.title(f"Mean normalized score per solver (family: {fam})")
        plt.tight_layout()
        plt.grid(axis="y", linestyle="--", alpha=0.3)

        outname = f"bar_norm_score_family_{fam}.png"
        plt.savefig(outname, dpi=200)
        plt.close()
        print(f"Saved {outname}")

    print("Done. NeurIPS-style benchmark artifacts ready:")
    print("  - results_neurips.csv")
    print("  - perf_profile_neurips.png")
    print("  - bar_norm_score_overall.png")
    print("  - bar_norm_score_family_<family>.png")
    print("  - summary_by_solver.csv / .tex")


if __name__ == "__main__":
    main()
