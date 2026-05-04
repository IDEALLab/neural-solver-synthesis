"""Tests for the fixed-code SDS evaluation workflow."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def make_simple_mission():
    return {
        "n_variables": 3,
        "cardinality_bounds": [1, 2],
        "precedence": [],
        "mutex": [],
        "groups": {},
        "weights": [1.0, 2.0, 3.0],
        "interactions": {},
    }


def make_fixed_solver(path: Path):
    path.write_text(
        "import json\nimport sys\n\n"
        "payload = json.load(sys.stdin)\n"
        "print(json.dumps({'selection': {'variables': [2]}}))\n"
    )


def test_extract_frozen_solver_selects_first_valid_uuid(tmp_path):
    from evaluation.sds.extract_frozen_solver import extract_solver

    csv_path = tmp_path / "metrics_final.csv"
    pd.DataFrame(
        [
            {"uuid": "b_uuid", "feasible": True, "error_type": "none", "code_snippet": "print('b')"},
            {"uuid": "a_uuid", "feasible": True, "error_type": "none", "code_snippet": "print('a')"},
            {"uuid": "c_uuid", "feasible": False, "error_type": "constraint", "code_snippet": "print('c')"},
        ]
    ).to_csv(csv_path, index=False)

    code, metadata = extract_solver(csv_path)
    assert code == "print('a')"
    assert metadata["selected_uuid"] == "a_uuid"


def test_manual_sa_solver_returns_feasible_selection():
    try:
        from evaluation.sds.utils import check_constraint_violations, mission_to_instance, run_candidate
    except ImportError as e:
        pytest.skip(f"Could not import SDS utilities: {e}")

    solver_path = (
        Path(__file__).resolve().parents[2]
        / "evaluation"
        / "sds"
        / "manual_solvers"
        / "constraint_aware_sa.py"
    )
    mission = make_simple_mission()
    stdin_obj = {
        "requirements": {
            **mission,
            "groups": {},
        },
        "catalog": {
            "variables": [
                {"id": idx, "weight": weight, "neighbors": []}
                for idx, weight in enumerate(mission["weights"])
            ],
            "interactions": {},
        },
    }

    result = run_candidate(solver_path.read_text(), stdin_obj, timeout=2.0)
    assert "selection" in result
    selection = result["selection"]["variables"]
    violations = check_constraint_violations(mission_to_instance(mission), selection)
    assert violations["all_valid"]


def test_fixed_code_eval_writes_metadata_and_timing(tmp_path, monkeypatch):
    try:
        from evaluation.sds import evaluate
    except ImportError as e:
        pytest.skip(f"Could not import evaluation.sds.evaluate: {e}")

    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps({"uuid": "uuid-1", "mission": make_simple_mission(), "generated_text": ""})
        + "\n"
    )
    solver_file = tmp_path / "fixed_solver.py"
    make_fixed_solver(solver_file)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--input_file",
            str(input_file),
            "--output_dir",
            str(output_dir),
            "--model",
            "fixed-code",
            "--training-scheme",
            "fixed-code",
            "--seed",
            "101",
            "--fixed-code-file",
            str(solver_file),
            "--method-name-override",
            "Frozen Hero",
            "--code-label",
            "frozen-hero",
            "--code-source-type",
            "frozen-hero",
            "--code-source-seed",
            "101",
            "--baselines",
            "greedy",
            "--workers",
            "1",
            "--repeats",
            "1",
        ],
    )

    evaluate.main()

    metadata = json.loads((output_dir / "experiment_metadata.json").read_text())
    timing = json.loads((output_dir / "timing_summary.json").read_text())
    metrics = pd.read_csv(output_dir / "metrics_final.csv")

    assert metadata["method_name"] == "Frozen Hero"
    assert metadata["code_source_type"] == "frozen-hero"
    assert timing["run_type"] == "fixed-code"
    assert timing["seed"] == 101
    assert "code_snippet" in metrics.columns
    assert metrics.loc[0, "code_snippet"].strip()


def test_runtime_aggregation_errors_when_timing_missing(tmp_path):
    from evaluation.sds.aggregate_fixed_code_results import collect_timing_rows

    with pytest.raises(RuntimeError, match="No timing_summary.json files found"):
        collect_timing_rows(tmp_path, runtime_seed=101)
