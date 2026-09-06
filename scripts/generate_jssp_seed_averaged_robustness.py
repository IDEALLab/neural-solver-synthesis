#!/usr/bin/env python3
"""Generate an SDS-style seed-averaged JSSP robustness plot with error bands."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE: dict[str, str] = {
    "LLM (Ours)": "#1f77b4",
    "SPT": "#17becf",
    "LPT": "#bcbd22",
    "MWKR": "#8c564b",
    "MOPR": "#e377c2",
    "Local Search": "#2ca02c",
    "OR-Tools": "#ff7f0e",
}

METHOD_ORDER = [
    "LLM (Ours)",
    "LPT",
    "Local Search",
    "MOPR",
    "MWKR",
    "OR-Tools",
    "SPT",
]


def _set_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8,
            "figure.figsize": (3.25, 2.5),
            "figure.dpi": 300,
        }
    )


def _load_seed_tables(base: Path) -> dict[int, pd.DataFrame]:
    paths = {
        101: base / "seed101/local_bundle_jssp_1733491_1735634/per_domain/jssp/method_gap_long_detailed.csv",
        202: base / "seed202/local_bundle_jssp_1736228_1736229/per_domain/jssp/method_gap_long_detailed.csv",
        303: base / "seed303/local_bundle_jssp_1736230_1736231/per_domain/jssp/method_gap_long_detailed.csv",
    }

    out: dict[int, pd.DataFrame] = {}
    for seed, path in paths.items():
        df = pd.read_csv(path)
        if "gap_clipped" not in df.columns or "method" not in df.columns:
            raise ValueError(f"Missing required columns in {path}")
        out[seed] = df
    return out


def generate_plot(results_root: Path, output_paths: list[Path]) -> None:
    _set_plot_style()
    taus = np.linspace(0.0, 0.5, 500)
    per_seed = _load_seed_tables(results_root)

    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    for method in METHOD_ORDER:
        ys_per_seed: list[np.ndarray] = []
        for seed in sorted(per_seed):
            sdf = per_seed[seed]
            gaps = sdf.loc[sdf["method"] == method, "gap_clipped"].to_numpy(dtype=float)
            if len(gaps) == 0:
                continue
            ys_per_seed.append(np.mean(gaps[:, None] <= taus[None, :], axis=0))

        if not ys_per_seed:
            continue

        ys = np.array(ys_per_seed)
        y_mean = np.mean(ys, axis=0)
        y_std = np.std(ys, axis=0)
        color = PALETTE.get(method, "black")

        is_ours = method == "LLM (Ours)"
        lw = 2.0 if is_ours else 1.4
        ls = "-" if is_ours else "--"
        alpha = 1.0 if is_ours else 0.85

        ax.plot(taus, y_mean, label=method, color=color, lw=lw, alpha=alpha, linestyle=ls)
        if len(ys_per_seed) > 1:
            ax.fill_between(
                taus,
                np.clip(y_mean - y_std, 0.0, 1.0),
                np.clip(y_mean + y_std, 0.0, 1.0),
                color=color,
                alpha=0.10,
            )

    ax.set_xlabel(r"Optimality Gap ($\tau$)")
    ax.set_ylabel(r"Fraction Solved ($g \leq \tau$)")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8, ncol=1)
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 1.01)
    plt.tight_layout(pad=0.2)

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root containing the three frozen seed evaluation bundles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        action="append",
        required=True,
        help="PNG output path; repeat to write the figure to multiple locations.",
    )
    args = parser.parse_args()
    generate_plot(args.results_root, args.output)


if __name__ == "__main__":
    main()
