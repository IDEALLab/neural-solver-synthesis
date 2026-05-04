#!/usr/bin/env python3
"""
Generate SDS (Synergistic Dependency Selection) dataset for LLM training.

UPDATED VERSION: optimized for "Reasoning Hardness" (Frustration > Greedy).

Usage:
    python gen_sds_dataset.py --num 1000 --mode random
"""

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Add syndeopt to path
_workspace_root = Path(__file__).resolve().parent.parent
_syndeopt_path = _workspace_root / "deps" / "syndeopt" / "src"
if str(_syndeopt_path) not in sys.path:
    sys.path.insert(0, str(_syndeopt_path))

from syndeopt.core.instance import CardBounds, SDSInstance  # noqa: E402
from syndeopt.gen import (  # noqa: E402
    make_decomposable_instance,
    make_dense_deceptive_instance,
    make_dense_instance,
    make_greedy_easy_instance,
    make_local_optima_instance,
    make_maxcut_qubo_instance,
    make_planted_qubo_instance,
    make_random_qubo_instance,
    make_structural_trap_instance,
    make_tree_showcase_instance,
)

# Solver import is lazy - only imported when compute_optimal=True
# This ensures no solver dependencies are loaded during normal data generation

# Why the following template is sound:
# 1. Data leakage: The promtp does NOT contain the answer
# 2. Memorization: The prompt does NOT contain the graph topology
# 3. Shortcut learning: The model can NOT ignore stdin and just use the prompt numbers (generalization reward prevents it)

# SDS Prompt Template (Updated for Algorithm Discovery)
SDS_SM_TEMPLATE = """
Output exactly two top-level blocks, in this order, with nothing else:
<think>... step-by-step reasoning ...</think>
<code>... a Python program ...</code>

Task: Write a high-performance, white-box Python solver for the Synergistic Dependency Selection (SDS) optimization problem.

Context: You are given a set of variables with individual values and pairwise interactions (synergies/penalties). 
You must select a subset of variables that maximizes the total value while respecting various constraints (precedence, mutual exclusion, groups, cardinality bounds).

I/O contract for <code>:
- Read ONE JSON object from stdin with keys {{"requirements": {{...}}, "catalog": {{...}}}} (schema below).
- Print ONE JSON object to stdout: {{"selection": {{"variables": [0, 2, 5, 7, ...]}}}}
- Must be FEASIBLE (respect all constraints).
- **Stochastic algorithms (randomness) are allowed** if useful.
- Use Python builtins (including `random`), `math`, `numpy`. Do NOT use external solvers like OR-Tools.

Implement in the code block:
- Define a function: def solve_sds():
  - Parse input (graph, constraints).
  - Implement a logic to search for the best possible solution.
  - Print the best found feasible selection to stdout.

Stdin JSON schema:
{{
  "requirements": {{
    "n_variables": <integer>,  // Total number of variables (0 to n_variables-1)
    "cardinality_bounds": [<min>, <max>],  // Minimum and maximum number of variables to select
    "precedence": [[<i>, <j>], ...],  // DAG constraints: if i is selected, j must be selected (i -> j)
    "mutex": [[<a>, <b>], ...],  // Mutual exclusion: at most one of a or b can be selected
    "groups": {{"<group_id>": [<var_ids>], ...}}  // Group constraints: at most one variable per group
  }},
  "catalog": {{
    "variables": [
      {{"id": <int>, "weight": <float>, "neighbors": [<neighbor_ids>]}},
      // ... exactly n_variables entries, one for each variable id from 0 to n_variables-1
    ],
    "interactions": {{
      "<i>,<j>": <float>,  // Pairwise interaction weight between variables i and j (only present if interaction exists)
      // ... sparse dictionary: only pairs with non-zero interactions are included
    }}
  }}
}}

IMPORTANT: 
- The "variables" array contains exactly n_variables entries (one for each variable id from 0 to n_variables-1).
- The "interactions" dictionary is SPARSE: it only contains entries for pairs that have a pairwise interaction weight.
- Variable "neighbors" lists all variables that have interactions with this variable.
- All variable indices are 0-indexed.

Problem instance summary (for reference - your code must read full data from stdin):
- Variables: {n_variables} (indices 0 to {n_variables_minus_one})
- Cardinality bounds: [{min_card}, {max_card}]
- Precedence constraints: {precedence_count} pairs
- Mutex constraints: {mutex_count} pairs
- Group constraints: {groups_count} groups
- Pairwise interactions: {interactions_count} pairs
"""


def write_jsonl(path, records):
    with Path(path).open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


def split3(seq, n_train, n_val):
    return seq[:n_train], seq[n_train : n_train + n_val], seq[n_train + n_val :]


def make_random_sds_instance(
    n: int | None = None, seed: int | None = None
) -> SDSInstance:
    """
    Generate a HARD random SDS instance.
    Unlike standard random generators, this scales interactions significantly
    higher than weights to ensure Greedy heuristics fail.
    """
    rng = random.Random(seed) if seed is not None else random

    # N-Value: Sweet spot for reasoning (15-35)
    if n is None:
        n = rng.randint(15, 35)

    # Unary Weights: Weak signals (-2 to 2)
    w = [rng.uniform(-2.0, 2.0) for _ in range(n)]

    # Pairwise Interactions: STRONG signals (-15 to 15)
    # High density (0.4 - 0.7) to create complex dependency webs
    interaction_density = rng.uniform(0.4, 0.7)
    W = {}  # noqa: N806
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < interaction_density:
                W[(i, j)] = rng.uniform(-15.0, 15.0)

    # Precedence: valid DAG
    precedence = []
    _PRECEDENCE_PROB = 0.5  # noqa: N806
    if n > 1 and rng.random() < _PRECEDENCE_PROB:
        ordering = list(range(n))
        rng.shuffle(ordering)
        num_precedence = rng.randint(0, min(n, 10))
        for _ in range(num_precedence):
            i_idx = rng.randint(0, n - 2)
            j_idx = rng.randint(i_idx + 1, n - 1)
            i, j = ordering[i_idx], ordering[j_idx]
            if (i, j) not in precedence:
                precedence.append((i, j))

    # Mutex: Constraint Frustration
    mutex = []
    _MUTEX_PROB = 0.6  # noqa: N806
    if n > 1 and rng.random() < _MUTEX_PROB:
        num_mutex = rng.randint(0, min(n // 2, 8))
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_pairs)
        mutex = all_pairs[:num_mutex]

    # Groups
    groups = {}
    _GROUPS_MIN_N = 2  # noqa: N806
    _GROUPS_PROB = 0.5  # noqa: N806
    if n > _GROUPS_MIN_N and rng.random() < _GROUPS_PROB:
        num_groups = rng.randint(1, min(n // 3, 5))
        remaining_vars = list(range(n))
        rng.shuffle(remaining_vars)
        group_size = len(remaining_vars) // num_groups
        for gid in range(num_groups):
            start_idx = gid * group_size
            end_idx = (
                start_idx + group_size if gid < num_groups - 1 else len(remaining_vars)
            )
            groups[gid] = remaining_vars[start_idx:end_idx]

    # Cardinality
    min_card = max(0, rng.randint(0, n // 3))
    max_card = rng.randint(min_card + 1, n)
    card = CardBounds(L=min_card, U=max_card)

    return SDSInstance(
        n=n, w=w, W=W, precedence=precedence, mutex=mutex, groups=groups, card=card
    )


def sds_sample(  # noqa: PLR0912, PLR0915
    mode: str, n_problems: int, seed: int | None = None, compute_optimal: bool = False
) -> list[dict[str, Any]]:
    """
    Generate SDS problem instances with TUNED HARDNESS parameters.
    Ensures W >> w so Greedy fails and Reasoning is required.
    """
    if seed is not None:
        random.seed(seed)

    problems = []

    # Weighted probabilities for problem types
    # Heavily favor "Deceptive/Dense" over "Tree/Easy"
    # [FIX] Added structural_trap (15%) to ensure constraint-dominant instances
    type_weights = {
        "tree": 0.05,  # Too easy
        "greedy_easy": 0.05,  # Sanity check only
        "decomposable": 0.1,  # Moderate
        "local_optima": 0.1,  # Trap testing
        "dense": 0.20,  # HARD (Core) - reduced from 0.25
        "bnb_showcase": 0.15,  # VERY HARD (Core)
        "structural_trap": 0.15,  # [NEW] Constraint-dominant (kills buggy solvers)
        "planted_qubo": 0.1,  # Pattern matching
        "maxcut_qubo": 0.05,  # Frustrated loops - reduced from 0.1
        "random_sds": 0.05,  # General chaos - reduced from 0.1
    }

    modes = list(type_weights.keys())
    probs = [type_weights[m] for m in modes]

    for i in tqdm(range(n_problems), desc=f"Generating {mode} SDS problems"):
        current_seed = seed + i if seed is not None else None

        # 1. Select Type
        if mode == "random":
            ptype = random.choices(modes, weights=probs, k=1)[0]
        elif mode in modes:
            ptype = mode
        else:
            ptype = "dense"  # Default to hard problem

        # 2. Generate Instance with "Torture" Parameters
        # N is kept in [15, 30] range for ideal reasoning difficulty

        if ptype == "tree":
            n = random.randint(15, 30)
            card = (n // 4, n // 2 + 5)
            # High pairwise scale to make edge selection critical
            inst = make_tree_showcase_instance(
                n=n, card=card, weight_scale=5.0, pair_scale=15.0, seed=current_seed
            )

        elif ptype == "greedy_easy":
            n = random.randint(10, 20)
            card = (max(2, n // 3), min(n, n // 2 + 2))
            inst = make_greedy_easy_instance(n=n, card=card, seed=current_seed)

        elif ptype == "local_optima":
            n = random.randint(15, 25)
            card = (max(3, n // 3), min(n, n // 2 + 3))
            inst = make_local_optima_instance(n=n, card=card, seed=current_seed)

        elif ptype == "decomposable":
            n = random.randint(20, 30)
            card = (max(3, n // 4), min(n, n // 2 + 5))
            inst = make_decomposable_instance(
                n=n, card=card, clusters=4, seed=current_seed
            )

        elif ptype == "dense":
            # HARD: Weak unaries (2.0), Massive interactions (20.0)
            n = random.randint(15, 25)
            card = (max(3, n // 3), min(n, n // 2 + 4))
            inst = make_dense_instance(
                n=n,
                card=card,
                weight_scale=2.0,
                pair_scale=20.0,
                pos_pair_frac=0.4,  # Mixed signs = Frustration
                neg_pair_frac=0.4,
                seed=current_seed,
            )

        elif ptype == "bnb_showcase":
            # VERY HARD: Almost zero unary signal, pure interaction logic
            n = random.randint(18, 25)
            card = (max(4, n // 3), min(n, n // 2 + 4))
            inst = make_dense_deceptive_instance(
                n=n,
                card=card,
                weight_scale=1.0,
                pair_scale=25.0,
                pos_pair_frac=0.5,
                neg_pair_frac=0.5,
                seed=current_seed,
            )

        elif ptype == "planted_qubo":
            n = random.randint(20, 30)
            card = (max(0, n // 4), min(n, n // 2 + 5))
            inst = make_planted_qubo_instance(
                n=n,
                card=card,
                signal_strength=5.0,
                noise_scale=2.0,  # Noise masks the pattern
                seed=current_seed,
            )

        elif ptype == "maxcut_qubo":
            n = random.randint(15, 25)
            card = (max(0, n // 4), min(n, n // 2 + 5))
            inst = make_maxcut_qubo_instance(
                n=n, card=card, edge_prob=0.6, weight_scale=10.0, seed=current_seed
            )

        elif ptype == "random_qubo":
            n = random.randint(15, 25)
            card = (max(0, n // 4), min(n, n // 2 + 5))
            inst = make_random_qubo_instance(
                n=n,
                card=card,
                diag_scale=2.0,
                offdiag_scale=10.0,  # Strong interactions
                density=0.6,
                seed=current_seed,
            )

        elif ptype == "structural_trap":
            # [NEW] Structural trap: Chain of pain (precedence-dominant)
            n = random.randint(18, 28)
            chain_length = random.randint(4, 7)  # Chain length varies
            inst = make_structural_trap_instance(
                n=n,
                chain_length=chain_length,
                bait_reward=100.0,  # High reward at leaf
                trap_penalty=-10.0,  # Penalty for each parent
                seed=current_seed,
            )

        else:  # random_sds
            # Scale check: N=50-100 to verify algorithmic scaling (kills brute force / slow code)
            # Matches generalization reward for consistency
            n = random.randint(50, 100)  # Large N range for runtime complexity filter
            inst = make_random_sds_instance(n=n, seed=current_seed)

        # 3. Convert
        problem = _instance_to_problem(inst, ptype, i, compute_optimal)
        problems.append(problem)

    return problems


def _instance_to_problem(
    inst: SDSInstance, problem_type: str, idx: int, compute_optimal: bool = False
) -> dict[str, Any]:
    """Convert SDSInstance to problem dictionary format."""
    requirements = {
        "n_variables": inst.n,
        "cardinality_bounds": [inst.card.L, inst.card.U],
        "precedence": inst.precedence,
        "mutex": inst.mutex,
        "groups": inst.groups,
        "weights": inst.w,
        "interactions": {f"{i},{j}": inst.W[(i, j)] for (i, j) in inst.W},
    }

    problem = {
        "uuid": f"sds_{problem_type}_{idx:06d}",
        "domain": "sds",
        "problem_type": problem_type,
        "mission": requirements,
        "requirements": requirements,
        "catalog": {
            "variables": [
                {"id": j, "weight": inst.w[j], "neighbors": list(inst.adj[j])}
                for j in range(inst.n)
            ],
            "interactions": {f"{i},{j}": inst.W[(i, j)] for (i, j) in inst.W},
        },
        "target": {"optimal_solution": None, "optimal_score": None},
    }

    if compute_optimal:
        from syndeopt.solvers import get_solver  # noqa: PLC0415

        best_score, best_solution = -float("inf"), None

        # We need powerful solvers for these harder problems
        solver_names = ["greedy", "local_search", "bnb"]

        # Try CP-SAT if available (best for these hard instances)
        with contextlib.suppress(BaseException):
            solver_names.append("cpsat")

        for solver_name in solver_names:
            try:
                solver = get_solver(solver_name)
                # Give more time for hard instances
                _LARGE_INSTANCE_THRESHOLD = 20  # noqa: N806
                budget = 5.0 if inst.n > _LARGE_INSTANCE_THRESHOLD else 2.0
                result = solver.solve(inst, budget_sec=budget, seed=0)
                if result.score > best_score:
                    best_score = result.score
                    best_solution = result.mask
            except Exception:
                continue

        if best_solution is not None:
            selected_vars = [j for j in range(inst.n) if (best_solution >> j) & 1]
            problem["target"]["optimal_solution"] = selected_vars
            problem["target"]["optimal_score"] = best_score

    return problem


def sds_render_prompt(prob: dict[str, Any]) -> dict[str, Any]:
    """Render SDS problem as a text prompt."""
    req = prob["requirements"]
    cat = prob["catalog"]

    format_dict = {
        "n_variables": req["n_variables"],
        "n_variables_minus_one": req["n_variables"] - 1,
        "min_card": req["cardinality_bounds"][0],
        "max_card": req["cardinality_bounds"][1],
        "precedence_count": len(req["precedence"]),
        "mutex_count": len(req["mutex"]),
        "groups_count": len(req["groups"]),
        "interactions_count": len(cat["interactions"]),
    }

    template = SDS_SM_TEMPLATE
    for key, value in format_dict.items():
        template = template.replace(f"{{{key}}}", str(value))

    return {
        "uuid": prob["uuid"],
        "problem": template.strip(),
        "mission": prob["mission"],
        "domain": "sds",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate HARD SDS dataset for LLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--num", "-n", type=int, default=1000)
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=[
            "random",
            "tree",
            "dense",
            "decomposable",
            "greedy_easy",
            "local_optima",
            "bnb_showcase",
            "structural_trap",
            "planted_qubo",
            "maxcut_qubo",
            "random_qubo",
            "random_sds",
        ],
    )
    parser.add_argument("--compute-optimal", action="store_true")
    parser.add_argument("--out-prefix", type=str, default="sds")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    _RATIO_TOLERANCE = 1e-6  # noqa: N806
    if (
        abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0)
        > _RATIO_TOLERANCE
    ):
        parser.error("Ratios must sum to 1.0")

    random.seed(args.seed)

    print(
        f"🎯 Generating {args.num} HARD SDS problems (mode: {args.mode}, seed: {args.seed})"
    )
    if args.compute_optimal:
        print("   ⚠️  Computing optimal solutions (this may take longer)")
    problems = sds_sample(
        args.mode, args.num, seed=args.seed, compute_optimal=args.compute_optimal
    )

    n_train = int(args.num * args.train_ratio)
    n_val = int(args.num * args.val_ratio)
    train, val, test = split3(problems, n_train, n_val)

    print(f"📊 Dataset splits: Train {len(train)}, Val {len(val)}, Test {len(test)}")

    write_jsonl(f"{args.out_prefix}_problems_train.jsonl", train)
    write_jsonl(f"{args.out_prefix}_problems_val.jsonl", val)
    write_jsonl(f"{args.out_prefix}_problems_test.jsonl", test)

    print("🎨 Rendering prompts...")

    def render_many(seq):
        for p in seq:
            yield sds_render_prompt(p)

    write_jsonl(
        f"{args.out_prefix}_prompts_train.jsonl",
        list(tqdm(render_many(train), desc="render train", total=len(train))),
    )
    write_jsonl(
        f"{args.out_prefix}_prompts_val.jsonl",
        list(tqdm(render_many(val), desc="render val", total=len(val))),
    )
    write_jsonl(
        f"{args.out_prefix}_prompts_test.jsonl",
        list(tqdm(render_many(test), desc="render test", total=len(test))),
    )

    print("✅ Done!")


if __name__ == "__main__":
    main()
