from __future__ import annotations

from typing import List, Dict, Any, Optional, Iterable
import math

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1) Performance profile (Dolan–Moré style) over score
# ---------------------------------------------------------------------

def performance_profile(
    rows: List[Dict[str, Any]],
    outfile: Optional[str] = None,
    show: bool = False,
    min_tau: float = 1.0,
    max_tau: float = 3.0,
    num_points: int = 200,
) -> None:
    """
    Plot a Dolan–Moré-style performance profile over *score*.

    For each instance i and solver s with score f_{i,s}, define
        best_i = max_s f_{i,s}
        r_{i,s} = best_i / f_{i,s}

    (for a maximization problem; if f_{i,s} <= 0 or infeasible, we treat r_{i,s}=+inf)

    The performance profile of solver s is:
        rho_s(tau) = fraction of instances i such that r_{i,s} <= tau.

    Parameters
    ----------
    rows : list of dict
        Output of bench.runner.run_suite.
    outfile : str or None
        If given, save the figure to this path.
    show : bool
        If True, call plt.show().
    min_tau, max_tau : float
        Range of tau on the x-axis.
    num_points : int
        Number of tau points to evaluate.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows provided for performance_profile.")

    # group by instance to find best score
    by_inst = df.groupby("instance")["score"]
    best_scores = by_inst.max()

    # build a table of ratios r_{i,s} = best_i / score_{i,s}
    # for infeasible / -inf score we set ratio = +inf
    ratios = []
    for _, row in df.iterrows():
        inst = row["instance"]
        solver = row["solver"]
        s = float(row["score"])
        best = float(best_scores[inst])

        if (not row.get("feasible", True)) or not math.isfinite(s) or s <= 0.0:
            r = float("inf")
        else:
            # if best is <= 0, treat everything equally (ratio = 1)
            if best <= 0.0:
                r = 1.0
            else:
                r = best / s

        ratios.append({"instance": inst, "solver": solver, "ratio": r})

    r_df = pd.DataFrame(ratios)
    solvers = sorted(r_df["solver"].unique())
    instances = sorted(r_df["instance"].unique())
    n_instances = len(instances)

    taus = [min_tau + (max_tau - min_tau) * i / (num_points - 1)
            for i in range(num_points)]

    plt.figure(figsize=(7, 5))

    for solver in solvers:
        sub = r_df[r_df["solver"] == solver]
        # map instance -> ratio
        ratio_vals = {row["instance"]: row["ratio"] for _, row in sub.iterrows()}

        y_vals: List[float] = []
        for tau in taus:
            count = 0
            for inst in instances:
                r = ratio_vals.get(inst, float("inf"))
                if r <= tau:
                    count += 1
            rho = count / n_instances
            y_vals.append(rho)

        plt.plot(taus, y_vals, label=solver)

    plt.xlabel(r"Performance ratio $\tau$ (score)")
    plt.ylabel(r"Fraction of instances with $r \leq \tau$")
    plt.title("Performance profile (score-based)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    if outfile is not None:
        plt.tight_layout()
        plt.savefig(outfile, dpi=200)

    if show:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------
# 2) Anytime curve (best-so-far score over time)
# ---------------------------------------------------------------------

def anytime_curve(
    traces: Dict[str, List[tuple]],  # solver_name -> list of (time, score) tuples
    outfile: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot anytime curves showing best-so-far score over time for one or more solvers.

    Parameters
    ----------
    traces : dict
        Dictionary mapping solver names to lists of (time, score) tuples.
        Each tuple represents a solution found at a specific time.
    outfile : str or None
        If given, save the figure to this path.
    title : str or None
        Plot title. If None, uses a default title.
    show : bool
        If True, call plt.show().
    """
    if not traces:
        raise ValueError("No traces provided for anytime_curve.")

    plt.figure(figsize=(8, 5))

    for solver_name, trace in traces.items():
        if not trace:
            continue

        # Sort by time to ensure chronological order
        sorted_trace = sorted(trace, key=lambda x: x[0])
        times = [t for t, _ in sorted_trace]
        scores = [s for _, s in sorted_trace]

        # Compute best-so-far (cumulative maximum for maximization)
        best_so_far = []
        current_best = float("-inf")
        for score in scores:
            if score > current_best:
                current_best = score
            best_so_far.append(current_best)

        plt.plot(times, best_so_far, label=solver_name, marker="o", markersize=3)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Best-so-far score")
    plt.title(title or "Anytime curve (best-so-far score over time)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    if outfile is not None:
        plt.tight_layout()
        plt.savefig(outfile, dpi=200)

    if show:
        plt.show()

    plt.close()
