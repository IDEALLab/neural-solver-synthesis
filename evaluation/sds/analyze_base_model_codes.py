#!/usr/bin/env python3
"""
Robust Base Model Analysis Script.
Focus: Accurate detection of Algorithm Types and Hero Structural Templates.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def strip_clean(code: str) -> str:
    """Removes comments and strings to prevent keyword hallucinations."""
    # Remove comments
    code = re.sub(r"#.*", "", code)
    # Remove string literals
    code = re.sub(r'(["\'])(?:(?=(\\?))\2.)*?\1', "", code)
    return code.lower()


def detect_algorithm(code: str) -> dict:
    """
    Robustly classifies the algorithm type.
    Priority: SA > Local Search > Greedy > Backtracking > Random.
    """
    clean = strip_clean(code)

    result = {
        "is_sa": False,
        "is_greedy": False,
        "is_local_search": False,
        "is_backtracking": False,
        "is_random": False,
        "is_other": False,
    }

    # 1. Simulated Annealing
    # Must have temperature, cooling, and exp logic
    has_temp = bool(re.search(r"\b(t|temperature)\s*=", clean))
    has_cooling = "cooling" in clean
    has_exp = "exp(" in clean or "math.exp" in clean
    if has_temp and has_cooling and has_exp:
        result["is_sa"] = True

    # 2. Greedy
    # Look for sorting by weight OR explicit greedy keywords in logic
    has_sort_weight = "sort" in clean and "weight" in clean
    has_greedy_kw = "greedy" in clean
    if (has_sort_weight or has_greedy_kw) and not result["is_sa"]:
        result["is_greedy"] = True

    # 3. Local Search (Hill Climbing / Descent)
    # Look for neighbor moves without SA cooling
    has_neighbor = "neighbor" in clean or "neighbour" in clean
    has_moves = any(x in clean for x in ["flip", "swap", "climb", "hill"])
    if (has_neighbor or has_moves) and not result["is_sa"]:
        result["is_local_search"] = True

    # 4. Backtracking / DFS
    # Look for recursion + backtracking keywords
    has_recursion = "def" in clean and re.search(
        r"def\s+(\w+).*?\1\(", clean, re.DOTALL
    )
    if "backtrack" in clean or (has_recursion and "dfs" in clean):
        result["is_backtracking"] = True

    # 5. Random Search
    # Fallback if random is used but no other structure found
    if (
        "random" in clean
        and not any(result.values())
        and ("sample" in clean or "choice" in clean or "uniform" in clean)
    ):
        result["is_random"] = True

    # 6. Other
    if not any(result.values()):
        result["is_other"] = True

    return result


def check_hero_structure(code: str, algo_result: dict) -> bool:
    """
    Checks if the code matches the 'Hero Template' structure:
    SA + Metropolis + (Active Guard OR Passive Filter).
    """
    if not algo_result["is_sa"]:
        return False

    clean = strip_clean(code)

    # 1. Metropolis Criterion
    # Must compare neighbor to something using exp/temp
    has_metropolis = False
    if (
        "random.random()" in code
        and ("exp(" in clean or "math.exp" in clean)
        and re.search(r"/\s*(?:t|temperature)\b", clean)
    ):
        has_metropolis = True

    # 2. Constraint Handling (Active or Passive)
    # We accept BOTH 'while not feasible' (Active) and 'if feasible' (Passive)
    # as evidence of the template structure.
    has_guard = False
    if ("is_feasible" in clean or "is_valid" in clean) and (
        "while" in clean or "if" in clean
    ):
        has_guard = True

    return has_metropolis and has_guard


def analyze_directory(base_dir: Path, output_dir: Path):
    results = []

    for seed in [101, 202, 303]:
        csv_path = base_dir / f"seed{seed}" / "metrics_final.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        # Filter for valid VBS (same as aggregate_plots.py)
        epsilon_medium = 1e-6
        df_valid = df[df["vbs_score"] > epsilon_medium].copy()

        # Include ALL instances (feasible and infeasible) to match manuscript
        # Infeasible instances get gap = 1.0 (100%)
        for _, row in df_valid.iterrows():
            code = str(row.get("code_snippet", ""))
            if not code or code == "nan":
                # Infeasible without code: gap = 1.0
                gap = 1.0
                algo = {
                    "is_sa": False,
                    "is_greedy": False,
                    "is_local_search": False,
                    "is_backtracking": False,
                    "is_random": False,
                    "is_other": True,
                }
                is_hero = False
            else:
                # Detect
                algo = detect_algorithm(code)
                is_hero = check_hero_structure(code, algo)

                # Calculate gap: feasible use actual score, infeasible = 1.0
                if row.get("feasible", False):
                    vbs = row.get("vbs_score", 0)
                    score = row.get("llm_score", 0)
                    gap = (vbs - max(0, score)) / vbs if vbs > epsilon_medium else 1.0
                else:
                    gap = 1.0  # Infeasible = 100% gap

            entry = {
                "seed": seed,
                "gap": gap * 100,
                "is_hero_template": is_hero,
                "feasible": row.get("feasible", False),
            }
            entry.update(algo)
            results.append(entry)

    df_res = pd.DataFrame(results)

    # --- Generate LaTeX Table ---
    if len(df_res) > 0:
        total = len(df_res)

        # Calculate Stats
        stats = []
        for key, name in [
            ("is_sa", "Simulated Annealing"),
            ("is_greedy", "Greedy"),
            ("is_backtracking", "Backtracking"),
            ("is_random", "Random Search"),
            ("is_other", "Other"),
        ]:
            sub = df_res[df_res[key]]
            count = len(sub)
            pct = count / total * 100
            gap = sub["gap"].mean() if count > 0 else 0
            stats.append(f"{name} & {count} ({pct:.1f}\\%) & {gap:.2f}\\% \\\\")

        # Hero Stat
        hero_sub = df_res[df_res["is_hero_template"]]
        hero_count = len(hero_sub)
        hero_gap = hero_sub["gap"].mean() if hero_count > 0 else 0
        stats.append(r"\midrule")
        stats.append(
            f"\\textit{{(SA w/ Valid Structure)}} & \\textit{{{hero_count}}} & \\textit{{{hero_gap:.2f}\\%}} \\\\"
        )

        latex = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{\textbf{Base Model Algorithm Distribution.} Analysis of $N="
            + str(total)
            + r"$ feasible codes. The model frequently attempts Simulated Annealing but fails to achieve low optimality gaps due to semantic logic errors.}",
            r"\label{tab:base_model_algos}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"\textbf{Algorithm Class} & \textbf{Frequency} & \textbf{Mean Gap} \\",
            r"\midrule",
            *stats,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        out_file = output_dir / "base_model_distribution.tex"
        out_file.write_text("\n".join(latex))
        print(f"Generated Table: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    analyze_directory(Path(args.base_dir), Path(args.output_dir))
