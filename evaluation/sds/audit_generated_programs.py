"""Audit unique generated programs with the repository's frozen SDS taxonomy."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.sds.analyze_sa_failure_modes import (
    canonicalize_code,
    extract_code,
    has_best_tracking,
    has_feasibility_guard,
    has_two_way_neighbor_logic,
)

ALGORITHM_PRIORITY = (
    ("is_sa", "simulated_annealing"),
    ("is_backtracking", "backtracking"),
    ("is_local_search", "local_search"),
    ("is_greedy", "greedy"),
    ("is_random", "random_search"),
    ("is_other", "other"),
)


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def assigned_name(target: ast.AST) -> str | None:
    return target.id if isinstance(target, ast.Name) else None


def assignment_values(tree: ast.AST) -> dict[str, list[ast.AST]]:
    values: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = assigned_name(target)
                if name:
                    values.setdefault(name, []).append(node.value)
        elif isinstance(node, ast.AnnAssign):
            name = assigned_name(node.target)
            if name and node.value is not None:
                values.setdefault(name, []).append(node.value)
    return values


def expanded_identifiers(
    node: ast.AST,
    assignments: dict[str, list[ast.AST]],
    seen: set[str] | None = None,
) -> set[str]:
    seen = set() if seen is None else seen
    identifiers = {
        child.id.lower() for child in ast.walk(node) if isinstance(child, ast.Name)
    }
    for name in tuple(identifiers):
        if name in seen:
            continue
        seen.add(name)
        for value in assignments.get(name, []):
            identifiers.update(expanded_identifiers(value, assignments, seen))
    return identifiers


def has_exp_call(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call) and dotted_name(child.func).lower().endswith("exp")
        for child in ast.walk(node)
    )


def detect_algorithm_ast(tree: ast.AST) -> dict[str, bool]:
    identifiers = {
        child.id.lower() for child in ast.walk(tree) if isinstance(child, ast.Name)
    }
    identifiers.update(
        child.name.lower()
        for child in ast.walk(tree)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    identifiers.update(
        child.attr.lower() for child in ast.walk(tree) if isinstance(child, ast.Attribute)
    )
    calls = {
        dotted_name(child.func).lower()
        for child in ast.walk(tree)
        if isinstance(child, ast.Call)
    }
    has_temperature = any(name == "t" or "temp" in name for name in identifiers)
    has_cooling = any("cool" in name for name in identifiers) or any(
        isinstance(child, ast.AugAssign)
        and isinstance(child.op, ast.Mult)
        and "temp" in (assigned_name(child.target) or "").lower()
        for child in ast.walk(tree)
    )
    is_sa = has_temperature and has_cooling and has_exp_call(tree)

    function_names = {
        child.name.lower()
        for child in ast.walk(tree)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    is_backtracking = any(
        "backtrack" in name or name == "dfs" for name in function_names
    )
    mutation_calls = {
        name.rsplit(".", 1)[-1]
        for name in calls
        if name.rsplit(".", 1)[-1] in {"add", "append", "discard", "pop", "remove"}
    }
    objective_tokens = ("cost", "fitness", "objective", "score", "value")
    assignments = assignment_values(tree)
    has_objective_comparison = any(
        isinstance(child, (ast.If, ast.While))
        and any(isinstance(part, ast.Compare) for part in ast.walk(child.test))
        and any(
            token in name
            for name in expanded_identifiers(child.test, assignments)
            for token in objective_tokens
        )
        for child in ast.walk(tree)
    )
    has_local_name = any(
        token in name
        for name in identifiers | function_names
        for token in ("climb", "flip", "local_search", "neighbor", "neighbour", "swap")
    )
    has_state_dependent_candidate = any(
        isinstance(child, ast.Assign)
        and any(
            (assigned_name(target) or "").lower().startswith(
                ("candidate", "neighbor", "neighbour", "new")
            )
            for target in child.targets
        )
        and any(
            token in name
            for name in expanded_identifiers(child.value, assignments)
            for token in ("current", "selection", "solution", "state")
        )
        for child in ast.walk(tree)
    )
    is_local_search = bool(
        not is_sa
        and not is_backtracking
        and (
            has_local_name
            or (mutation_calls and has_objective_comparison)
            or (has_state_dependent_candidate and has_objective_comparison)
        )
    )
    has_sort = any(name.endswith(".sort") or name == "sorted" for name in calls)
    has_weight = any("weight" in name for name in identifiers)
    is_greedy = bool(
        not is_sa
        and not is_backtracking
        and not is_local_search
        and (any("greedy" in name for name in identifiers) or (has_sort and has_weight))
    )
    is_random = bool(
        not is_sa
        and not is_backtracking
        and not is_local_search
        and not is_greedy
        and any(name.startswith("random.") for name in calls)
    )
    is_other = not any((is_sa, is_backtracking, is_local_search, is_greedy, is_random))
    return {
        "is_sa": is_sa,
        "is_greedy": is_greedy,
        "is_local_search": is_local_search,
        "is_backtracking": is_backtracking,
        "is_random": is_random,
        "is_other": is_other,
    }


def classify_sa_acceptance_ast(tree: ast.AST) -> str:
    assignments = assignment_values(tree)
    acceptance_identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While)) and has_exp_call(node.test):
            acceptance_identifiers.update(
                expanded_identifiers(node.test, assignments)
            )
    saw_best = any("best" in name or "global" in name for name in acceptance_identifiers)
    saw_current = any(
        "current" in name or name.startswith("curr")
        for name in acceptance_identifiers
    )
    if saw_best and saw_current:
        return "mixed"
    if saw_best:
        return "best_bug"
    if saw_current:
        return "current_ok"
    return "unresolved"


def classify_program(code: str) -> dict[str, Any]:
    canonical = canonicalize_code(code)
    code_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    try:
        tree = ast.parse(canonical)
    except SyntaxError:
        return {
            "code_sha256": code_hash,
            "algorithm_family": "malformed",
            "sa_acceptance": "not_sa",
            "has_feasibility_guard": False,
            "has_best_tracking": False,
            "has_two_way_neighbor_logic": False,
            "canonical_code": canonical,
        }
    detections = detect_algorithm_ast(tree)
    family = next(label for key, label in ALGORITHM_PRIORITY if detections[key])
    is_sa = family == "simulated_annealing"
    return {
        "code_sha256": code_hash,
        "algorithm_family": family,
        "sa_acceptance": classify_sa_acceptance_ast(tree) if is_sa else "not_sa",
        "has_feasibility_guard": has_feasibility_guard(canonical),
        "has_best_tracking": has_best_tracking(canonical),
        "has_two_way_neighbor_logic": has_two_way_neighbor_logic(canonical),
        "canonical_code": canonical,
    }


def audit_generations(input_path: Path, seed: int, condition: str) -> dict[str, Any]:
    if input_path.suffix == ".csv":
        with input_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = [
            json.loads(line) for line in input_path.read_text().splitlines() if line
        ]
    unique: dict[str, dict[str, Any]] = {}
    row_audits: list[dict[str, Any]] = []
    missing_code = 0
    for row in rows:
        code = (
            row.get("generated_code")
            or row.get("code_snippet")
            or extract_code(row.get("generated_text", ""))
        )
        if not code:
            missing_code += 1
            audit = {
                "code_sha256": None,
                "algorithm_family": "malformed",
                "sa_acceptance": "not_sa",
                "has_feasibility_guard": False,
                "has_best_tracking": False,
                "has_two_way_neighbor_logic": False,
            }
        else:
            audit = classify_program(code)
            unique.setdefault(audit["code_sha256"], audit)
        row_audits.append(
            {
                "seed": seed,
                "uuid": row.get("uuid"),
                "condition": condition,
                **{
                    key: value
                    for key, value in audit.items()
                    if key != "canonical_code"
                },
            }
        )

    families = Counter(row["algorithm_family"] for row in row_audits)
    acceptances = Counter(row["sa_acceptance"] for row in row_audits)
    return {
        "summary": {
            "seed": seed,
            "condition": condition,
            "row_count": len(rows),
            "unique_uuid_count": len({row.get("uuid") for row in rows}),
            "unique_code_count": len(unique),
            "missing_code_count": missing_code,
            "algorithm_family_counts": dict(sorted(families.items())),
            "sa_acceptance_counts": dict(sorted(acceptances.items())),
        },
        "rows": row_audits,
        "unique": [unique[key] for key in sorted(unique)],
    }


def write_audit(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2) + "\n"
    )
    with (output_dir / "row_audit.jsonl").open("w") as handle:
        for row in result["rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "unique_programs.jsonl").open("w") as handle:
        for row in result["unique"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_audit(
        audit_generations(args.input, args.seed, args.condition), args.output_dir
    )


if __name__ == "__main__":
    main()
