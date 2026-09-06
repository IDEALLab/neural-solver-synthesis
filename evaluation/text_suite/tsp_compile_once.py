"""Compile-once selection, freezing, and TSPLIB gates for N26-E12."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from evaluation.sds.utils import run_candidate

REPO_ROOT = Path(__file__).resolve().parents[2]
OPEN_R1_SRC = REPO_ROOT / "deps" / "open-r1" / "src"

if str(OPEN_R1_SRC) not in sys.path:
    sys.path.insert(0, str(OPEN_R1_SRC))

from open_r1.simulators.tsp_simulator import (  # noqa: E402
    TSPSimulator,
    one_tree_lower_bound,
)

SEEDS = (101, 202, 303)
CONDITIONS = ("trained", "base64")
TSPLIB_NAMES = (
    "berlin52",
    "eil51",
    "eil76",
    "kroA100",
    "kroB100",
    "kroC100",
    "kroD100",
    "kroE100",
    "pr76",
    "rat99",
    "rd100",
    "st70",
)
PROTOCOL_IDS = ("N26-E12-v1", "N26-E12-v3")
WHOLE_PAYLOAD_FENCE_RE = re.compile(
    r"\A```(?:python|py)?[ \t]*\n(?P<code>.*?)(?:\n)?```[ \t]*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonicalize_code(code: str) -> str:
    normalized = str(code or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.splitlines()).strip() + "\n"


def canonicalize_generated_code(code: str) -> str:
    """Normalize code and unwrap one Markdown fence enclosing the whole payload."""
    normalized = canonicalize_code(code).strip()
    match = WHOLE_PAYLOAD_FENCE_RE.fullmatch(normalized)
    if match:
        normalized = match.group("code")
    return canonicalize_code(normalized)


def load_protocol(path: Path, *, require_launch_ready: bool = False) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("protocol_id") not in PROTOCOL_IDS:
        raise ValueError("unexpected TSP protocol ID")
    if protocol["model"] != {
        "repo": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "revision": "aedcc2d42b622764e023cf882b6652e646b95671",
        "initialization": "direct_from_base_no_sft",
    }:
        raise ValueError("TSP base-model or zero-SFT contract drift")
    training = protocol["training"]
    expected_training = {
        "instances_per_seed": 10_000,
        "minimum_cities": 20,
        "maximum_cities": 100,
        "families": [
            "uniform",
            "clustered",
            "grid_jittered",
            "radial",
            "anisotropic",
        ],
        "nodes": 3,
        "gh200_per_node": 4,
        "mode": "grpo_cold",
        "sft_checkpoint": None,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"TSP training protocol drift: {key}")
    if protocol["protocol_id"] == "N26-E12-v3":
        expected_full_epoch = {
            "max_steps": -1,
            "epochs": 1,
            "expected_optimizer_steps": 2_500,
            "checkpoint_interval_steps": 30,
            "smoke_split_counts": {
                "train": 4,
                "validation": 1,
                "test": 1,
            },
            "production_step_override": None,
        }
        for key, expected in expected_full_epoch.items():
            if training.get(key) != expected:
                raise ValueError(f"TSP full-epoch protocol drift: {key}")
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(training.get("smoke_dataset_revision", ""))
        ):
            raise ValueError("TSP smoke dataset revision is not frozen")
    if protocol["reward"] != {
        "hard_feasibility_gate": True,
        "format_weight": 0.10,
        "execution_weight": 0.20,
        "quality_weight": 0.70,
        "quality_reference": "deterministic minimum one-tree lower bound rooted at city 0",
        "algorithm_token_shaping": False,
    }:
        raise ValueError("TSP reward protocol drift")
    selection = protocol["selection"]
    expected_selection = {
        "synthesis_prompt_count": 64,
        "development_instance_count": 40,
        "disjoint": True,
        "temperature": 0.0,
        "completions_per_prompt": 1,
        "timeout_seconds": 5.0,
        "failure_gap_penalty": 10.0,
        "test_access_before_global_freeze": bool(
            protocol.get("post_result_markdown_normalization_correction")
        ),
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ValueError(f"TSP selection protocol drift: {key}")
    benchmark = protocol["benchmark"]
    if tuple(benchmark["instances"]) != TSPLIB_NAMES:
        raise ValueError("TSPLIB instance set drift")
    if benchmark["repetitions"] != 5 or benchmark["time_limit_seconds"] != 5.0:
        raise ValueError("TSPLIB trial protocol drift")
    if benchmark["methods"] != [
        "trained",
        "base64",
        "nearest_neighbor",
        "insertion",
        "two_opt",
        "ortools_single_worker",
    ]:
        raise ValueError("TSPLIB comparison set drift")
    if set(protocol["seeds"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("TSP protocol must contain all three seeds")
    if require_launch_ready:
        for seed, row in protocol["seeds"].items():
            revision = str(row.get("dataset_revision", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", revision):
                raise ValueError(f"seed {seed} dataset revision is not frozen")
        verify_protocol_assets(path, REPO_ROOT)
    return protocol


def verify_protocol_assets(protocol_path: Path, repo_root: Path) -> dict[str, str]:
    protocol = load_protocol(protocol_path)
    verified = {}
    for row in protocol["assets"]:
        relative = row["path"]
        expected = row["sha256"]
        asset = repo_root / relative
        if not asset.is_file():
            raise ValueError(f"missing frozen protocol asset: {relative}")
        actual = sha256_file(asset)
        if actual != expected:
            raise ValueError(
                f"protocol asset drift for {relative}: {actual} != {expected}"
            )
        verified[relative] = actual
    open_r1_expected = protocol["submodules"]["deps/open-r1"]
    open_r1_actual = subprocess.check_output(
        ["git", "-C", str(repo_root / "deps/open-r1"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if open_r1_actual != open_r1_expected:
        raise ValueError("OpenR1 submodule commit drift")
    return verified


def rank_development_rows(
    uuids: Iterable[str],
    salt: str,
    synthesis_count: int = 64,
    selection_count: int = 40,
) -> tuple[list[int], list[int]]:
    values = list(uuids)
    if len(values) != len(set(values)):
        raise ValueError("TSP development UUIDs are not unique")
    ranked = sorted(
        range(len(values)),
        key=lambda index: (sha256_text(f"{salt}|{values[index]}"), values[index]),
    )
    if len(ranked) < synthesis_count + selection_count:
        raise ValueError("insufficient TSP validation rows")
    return ranked[:synthesis_count], ranked[
        synthesis_count : synthesis_count + selection_count
    ]


def _load_local_split(dataset_root: Path, split: str, revision: str):
    from datasets import load_from_disk

    manifest_path = dataset_root / "n26_dataset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset_revision") != revision:
        raise ValueError("local TSP dataset revision drift")
    dataset = load_from_disk(str(dataset_root))
    if split not in dataset:
        raise ValueError(f"local TSP dataset split missing: {split}")
    return dataset[split]


def freeze_development(
    protocol_path: Path, seed: int, dataset_root: Path, output: Path
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path, require_launch_ready=True)
    seed_row = protocol["seeds"][str(seed)]
    split = protocol["selection"]["split"]
    dataset = _load_local_split(dataset_root, split, seed_row["dataset_revision"])
    uuids = [str(value) for value in dataset["uuid"]]
    synthesis, development = rank_development_rows(
        uuids,
        protocol["selection"]["salt"],
        protocol["selection"]["synthesis_prompt_count"],
        protocol["selection"]["development_instance_count"],
    )
    result = {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "dataset_root": str(dataset_root),
        "dataset_revision": seed_row["dataset_revision"],
        "split": split,
        "dataset_uuid_sha256": sha256_text("\n".join(uuids)),
        "synthesis_indices": synthesis,
        "synthesis_uuids": [uuids[index] for index in synthesis],
        "selection_indices": development,
        "selection_uuids": [uuids[index] for index in development],
        "external_test_accessed": False,
    }
    if set(result["synthesis_uuids"]) & set(result["selection_uuids"]):
        raise ValueError("TSP synthesis and development sets overlap")
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def assert_generation_provenance(
    protocol: dict[str, Any],
    seed: int,
    condition: str,
    development: dict[str, Any],
    generations: list[dict[str, Any]],
    model_revision: str,
) -> None:
    expected = {
        "dataset_revision": protocol["seeds"][str(seed)]["dataset_revision"],
        "model_revision": model_revision,
        "split": protocol["selection"]["split"],
        "temperature": protocol["selection"]["temperature"],
        "seed": seed,
    }
    if [str(row.get("uuid")) for row in generations] != development["synthesis_uuids"]:
        raise ValueError("TSP generation UUID order drift")
    for row in generations:
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"{condition} generation provenance drift: {key}")
        if row.get("domain") != "tsp":
            raise ValueError("non-TSP generation in E12")


def prepare_selection(
    protocol_path: Path,
    seed: int,
    condition: str,
    dataset_root: Path,
    development_path: Path,
    generations_path: Path,
    model_revision: str,
    inputs_path: Path,
    candidates_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path, require_launch_ready=True)
    if condition not in CONDITIONS:
        raise ValueError(f"invalid E12 condition: {condition}")
    development = json.loads(development_path.read_text())
    generations = [
        json.loads(line) for line in generations_path.read_text().splitlines() if line
    ]
    if len(generations) != protocol["selection"]["synthesis_prompt_count"]:
        raise ValueError(
            "E12 must generate exactly 64 completions per condition and seed"
        )
    assert_generation_provenance(
        protocol, seed, condition, development, generations, model_revision
    )
    unique: dict[str, str] = {}
    normalized_fenced_completion_count = 0
    for row in generations:
        raw_code = canonicalize_code(row.get("generated_code", ""))
        code = canonicalize_generated_code(raw_code)
        normalized_fenced_completion_count += code != raw_code
        if code.strip():
            unique.setdefault(sha256_text(code), code)
    if not unique:
        raise ValueError("E12 produced no code candidates")
    candidates = [
        {
            "candidate_id": index,
            "code_sha256": code_hash,
            "generated_code": unique[code_hash],
        }
        for index, code_hash in enumerate(sorted(unique))
    ]
    candidate_manifest = {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "condition": condition,
        "completion_count": len(generations),
        "unique_candidate_count": len(candidates),
        "model_revision": model_revision,
        "code_normalization": "strip_one_whole_payload_markdown_fence_v1",
        "normalized_fenced_completion_count": normalized_fenced_completion_count,
        "candidates": candidates,
        "external_test_accessed": False,
    }
    candidates_path.write_text(json.dumps(candidate_manifest, indent=2) + "\n")
    dataset = _load_local_split(
        dataset_root,
        development["split"],
        protocol["seeds"][str(seed)]["dataset_revision"],
    ).select(development["selection_indices"])
    with inputs_path.open("w") as handle:
        for candidate in candidates:
            for selection_rank, row in enumerate(dataset):
                handle.write(
                    json.dumps(
                        {
                            "protocol_id": protocol["protocol_id"],
                            "seed": seed,
                            "condition": condition,
                            "uuid": row["uuid"],
                            "family": row["family"],
                            "mission": row["mission"],
                            "candidate_id": candidate["candidate_id"],
                            "candidate_sha256": candidate["code_sha256"],
                            "generated_code": candidate["generated_code"],
                            "selection_rank": selection_rank,
                        }
                    )
                    + "\n"
                )
    return candidate_manifest


def _strict_tour(result: Any) -> Any:
    if not isinstance(result, dict) or set(result) not in (
        {"selection"},
        {"selection", "execution_time"},
    ):
        return None
    selection = result["selection"]
    if not isinstance(selection, dict) or set(selection) != {"tour"}:
        return None
    return selection["tour"]


def _evaluate_selection_row(payload: tuple[dict[str, Any], float]) -> dict[str, Any]:
    row, timeout = payload
    mission = row["mission"]
    if isinstance(mission, str):
        mission = json.loads(mission)
    mission = dict(mission)
    mission["random_seed"] = int(
        sha256_text(
            f"{row['seed']}|{row['condition']}|{row['candidate_sha256']}|{row['uuid']}"
        )[:8],
        16,
    )
    started = time.perf_counter()
    result = run_candidate(
        row["generated_code"],
        {"requirements": mission, "catalog": {"cities": mission["cities"]}},
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    tour = _strict_tour(result)
    simulator = TSPSimulator()
    feasible = simulator.validate_design(tour, mission)
    simulation = simulator.simulate(tour, mission) if feasible else {}
    feasible = feasible and bool(simulation.get("feasible", False))
    lower_bound = one_tree_lower_bound(mission["cities"])
    tour_length = float(simulation.get("tour_length", float("inf")))
    gap = (
        max(0.0, (tour_length - lower_bound) / lower_bound)
        if feasible and lower_bound > 0
        else float("inf")
    )
    return {
        "seed": row["seed"],
        "condition": row["condition"],
        "uuid": row["uuid"],
        "family": row["family"],
        "n_cities": mission["n_cities"],
        "candidate_id": row["candidate_id"],
        "candidate_sha256": row["candidate_sha256"],
        "feasible": feasible,
        "tour_length": tour_length,
        "one_tree_lower_bound": lower_bound,
        "bounded_gap": gap,
        "execution_time": elapsed,
        "error": result.get("error", "")
        if isinstance(result, dict)
        else "invalid result",
    }


def evaluate_selection(
    protocol_path: Path, inputs_path: Path, output_path: Path, workers: int
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path, require_launch_ready=True)
    rows = [json.loads(line) for line in inputs_path.read_text().splitlines() if line]
    timeout = float(protocol["selection"]["timeout_seconds"])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(_evaluate_selection_row, ((row, timeout) for row in rows))
        )
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return {"row_count": len(results), "workers": workers}


def select_solver(
    protocol_path: Path,
    seed: int,
    condition: str,
    metrics_path: Path,
    candidates_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path, require_launch_ready=True)
    candidate_manifest = json.loads(candidates_path.read_text())
    candidates = {
        int(row["candidate_id"]): row for row in candidate_manifest["candidates"]
    }
    with metrics_path.open(newline="") as handle:
        metrics = list(csv.DictReader(handle))
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        grouped[int(row["candidate_id"])].append(row)
    expected = protocol["selection"]["development_instance_count"]
    penalty = protocol["selection"]["failure_gap_penalty"]
    summaries = []
    for candidate_id, candidate in candidates.items():
        rows = grouped.get(candidate_id, [])
        if len(rows) != expected or len({row["uuid"] for row in rows}) != expected:
            raise ValueError(f"candidate {candidate_id} lacks exact 40-row coverage")
        feasible = [row["feasible"].lower() == "true" for row in rows]
        penalized = [
            float(row["bounded_gap"]) if ok else penalty
            for row, ok in zip(rows, feasible, strict=True)
        ]
        summaries.append(
            {
                "candidate_id": candidate_id,
                "code_sha256": candidate["code_sha256"],
                "pass_rate": sum(feasible) / len(feasible),
                "failure_penalized_mean_bounded_gap": statistics.fmean(penalized),
            }
        )
    selected = min(
        summaries,
        key=lambda row: (
            -row["pass_rate"],
            row["failure_penalized_mean_bounded_gap"],
            row["code_sha256"],
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    solver = output_dir / "selected_solver.py"
    solver.write_text(candidates[int(selected["candidate_id"])]["generated_code"])
    result = {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "condition": condition,
        "selection_order": "feasibility desc, failure-penalized one-tree bounded gap asc, SHA-256 asc",
        "selected": selected,
        "solver_sha256": sha256_file(solver),
        "candidate_summaries": summaries,
        "external_test_accessed": False,
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def audit_prompt_leakage(texts: Iterable[str], protocol: dict[str, Any]) -> None:
    optima = [str(value) for value in protocol["benchmark"]["optima"].values()]
    for text in texts:
        normalized = str(text).lower()
        leaked_names = [name for name in TSPLIB_NAMES if name.lower() in normalized]
        leaked_optima = [
            value
            for value in optima
            if re.search(rf"(?<!\d){re.escape(value)}(?!\d)", normalized)
        ]
        if leaked_names or leaked_optima:
            raise ValueError(
                f"external benchmark leakage detected: names={leaked_names}, optima={leaked_optima}"
            )


def freeze_all(
    protocol_path: Path, selection_root: Path, source_commit: str, output: Path
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path, require_launch_ready=True)
    if output.exists():
        raise ValueError(f"refusing to overwrite global freeze: {output}")
    selected = {}
    leakage_texts = []
    for condition in CONDITIONS:
        selected[condition] = {}
        for seed in SEEDS:
            root = selection_root / condition / f"seed{seed}"
            summary_path = root / "selected" / "selection_summary.json"
            solver_path = root / "selected" / "selected_solver.py"
            summary = json.loads(summary_path.read_text())
            solver_hash = sha256_file(solver_path)
            if summary["solver_sha256"] != solver_hash:
                raise ValueError("selection summary and solver hash disagree")
            generations = root / "candidate_generations.jsonl"
            leakage_texts.extend(
                value
                for row in (
                    json.loads(line)
                    for line in generations.read_text().splitlines()
                    if line
                )
                for value in (
                    row.get("problem", ""),
                    row.get("completion", ""),
                    row.get("generated_code", ""),
                )
            )
            leakage_texts.append(solver_path.read_text())
            selected[condition][str(seed)] = {
                "solver_path": str(solver_path),
                "solver_sha256": solver_hash,
                "selection_summary_sha256": sha256_file(summary_path),
                "model_revision": json.loads(
                    (root / "candidate_manifest.json").read_text()
                )["model_revision"],
            }
    audit_prompt_leakage(leakage_texts, protocol)
    result = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "source_commit": source_commit,
        "created_unix": time.time(),
        "selected": selected,
        "all_six_hashes_frozen": sum(len(rows) for rows in selected.values()) == 6,
        "prompt_leakage_audit": "pass",
        "external_test_accessed": False,
    }
    correction = protocol.get("post_result_markdown_normalization_correction")
    if correction:
        result["post_test_mechanical_correction"] = True
        result["external_test_outcomes_known_before_correction"] = bool(
            correction["test_outcomes_observed_before_correction"]
        )
        result["external_test_used_for_corrected_selection"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def bind_existing_tsplib_for_correction(
    protocol_path: Path,
    global_freeze_path: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Bind immutable prior TSPLIB bytes to a transparent post-result correction."""
    protocol = load_protocol(protocol_path, require_launch_ready=True)
    correction = protocol.get("post_result_markdown_normalization_correction")
    if not correction or not correction.get("test_outcomes_observed_before_correction"):
        raise ValueError("protocol does not declare a post-result correction")
    freeze = json.loads(global_freeze_path.read_text())
    validate_global_freeze(protocol, freeze)
    if not freeze.get("post_test_mechanical_correction"):
        raise ValueError("global freeze lacks correction provenance")
    if freeze.get("external_test_used_for_corrected_selection") is not False:
        raise ValueError("corrected selection must remain development-only")
    if output_root.exists():
        raise ValueError(f"refusing to overwrite correction benchmark bytes: {output_root}")

    source_provenance_path = source_root / "provenance.json"
    if sha256_file(source_provenance_path) != correction["original_external_provenance_sha256"]:
        raise ValueError("original TSPLIB provenance drift")
    source_provenance = json.loads(source_provenance_path.read_text())
    if source_provenance["global_freeze_sha256"] != correction["original_global_freeze_sha256"]:
        raise ValueError("original TSPLIB provenance is not bound to the reported freeze")

    output_root.mkdir(parents=True)
    copied_rows = []
    for row in source_provenance["instances"]:
        source = source_root / row["path"]
        if sha256_file(source) != row["raw_sha256"]:
            raise ValueError(f"original TSPLIB bytes drift: {row['name']}")
        target = output_root / row["path"]
        shutil.copy2(source, target)
        if sha256_file(target) != row["raw_sha256"]:
            raise ValueError(f"copied TSPLIB bytes drift: {row['name']}")
        copied_rows.append(row)

    result = {
        **source_provenance,
        "created_unix": time.time(),
        "created_after_global_freeze": time.time() >= float(freeze["created_unix"]),
        "global_freeze_sha256": sha256_file(global_freeze_path),
        "instances": copied_rows,
        "post_test_mechanical_correction": True,
        "test_outcomes_observed_before_corrected_freeze": True,
        "external_test_used_for_corrected_selection": False,
        "original_external_provenance_sha256": sha256_file(source_provenance_path),
        "original_global_freeze_sha256": source_provenance["global_freeze_sha256"],
        "raw_instance_bytes_reused_without_change": True,
    }
    (output_root / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def hash_model_tree(model_root: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash every regular model artifact for an immutable local model revision."""
    if not model_root.is_dir():
        raise ValueError(f"trained model directory is missing: {model_root}")
    files = []
    # Hash the final inference artifact only; checkpoint-* directories contain
    # redundant training snapshots and are retained separately as provenance.
    for path in sorted(item for item in model_root.iterdir() if item.is_file()):
        relative = path.relative_to(model_root).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files or not any(row["path"].endswith(".safetensors") for row in files):
        raise ValueError("trained model tree does not contain safetensors weights")
    revision = sha256_text(json.dumps(files, sort_keys=True, separators=(",", ":")))
    return revision, files


def record_training(
    protocol_path: Path,
    seed: int,
    dataset_root: Path,
    model_root: Path,
    source_commit: str,
    container_sha256: str,
    slurm_job_id: str,
    smoke: bool,
    output: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path, require_launch_ready=True)
    dataset_manifest = json.loads(
        (dataset_root / "n26_dataset_manifest.json").read_text()
    )
    expected_revision = (
        protocol["training"]["smoke_dataset_revision"]
        if smoke and protocol["protocol_id"] == "N26-E12-v3"
        else protocol["seeds"][str(seed)]["dataset_revision"]
    )
    if dataset_manifest.get("dataset_revision") != expected_revision:
        raise ValueError("training dataset revision drift")
    production_counts = {
        "train": 10_000,
        "validation": 1_000,
        "test": 1_000,
    }
    expected_counts = (
        protocol["training"].get("smoke_split_counts", production_counts)
        if smoke
        else production_counts
    )
    if dataset_manifest.get("counts") != expected_counts:
        raise ValueError("E12 training dataset does not contain the frozen split sizes")
    if any(model_root.glob("adapter_model*")):
        raise ValueError("E12 forbids SFT/adapter initialization artifacts")
    training_results_path = model_root / "train_results.json"
    if not training_results_path.is_file():
        raise ValueError("completed training result is missing")
    training_results = json.loads(training_results_path.read_text())
    expected_train_samples = int(expected_counts["train"])
    if int(training_results.get("train_samples", -1)) != expected_train_samples:
        raise ValueError(
            f"E12 training did not use exactly {expected_train_samples} rows"
        )
    trainer_state_path = model_root / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise ValueError("completed training is missing trainer_state.json")
    trainer_state = json.loads(trainer_state_path.read_text())
    global_step = int(trainer_state.get("global_step", 0))
    expected_steps = (
        1
        if smoke
        else int(protocol["training"].get("expected_optimizer_steps", global_step))
    )
    invalid_step = global_step != expected_steps
    if invalid_step:
        raise ValueError(
            f"E12 training must complete exactly {expected_steps} optimizer steps"
        )
    log_history = trainer_state.get("log_history", [])
    losses = [
        float(row["loss"])
        for row in log_history
        if "loss" in row and math.isfinite(float(row["loss"]))
    ]
    gradient_norms = [
        float(row["grad_norm"])
        for row in log_history
        if "grad_norm" in row and math.isfinite(float(row["grad_norm"]))
    ]
    if not losses or not gradient_norms or max(gradient_norms) <= 0:
        raise ValueError("E12 training lacks evidence of a real GRPO parameter update")
    revision, files = hash_model_tree(model_root)
    result = {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "smoke": smoke,
        "mode": "grpo_cold",
        "sft_checkpoint": None,
        "base_model": protocol["model"],
        "dataset_revision": expected_revision,
        "model_root": str(model_root),
        "model_revision": revision,
        "model_files": files,
        "train_samples": int(training_results.get("train_samples", -1)),
        "global_step": global_step,
        "finite_logged_losses": losses,
        "positive_gradient_norm_max": max(gradient_norms),
        "real_grpo_update_verified": True,
        "source_commit": source_commit,
        "open_r1_commit": protocol["submodules"]["deps/open-r1"],
        "container_sha256": container_sha256,
        "slurm_job_id": slurm_job_id,
        "status": "complete",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_tsplib(path: Path, expected_name: str | None = None) -> dict[str, Any]:
    headers: dict[str, str] = {}
    coordinates: list[dict[str, int]] = []
    in_coordinates = False
    saw_eof = False
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "NODE_COORD_SECTION":
            in_coordinates = True
            continue
        if line == "EOF":
            saw_eof = True
            break
        if in_coordinates:
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"malformed TSPLIB coordinate row: {line}")
            node_id, x, y = fields
            coordinates.append({"id": int(node_id) - 1, "x": float(x), "y": float(y)})
        else:
            match = re.fullmatch(r"([A-Z_]+)\s*:\s*(.+)", line)
            if not match:
                raise ValueError(f"malformed TSPLIB header: {line}")
            headers[match.group(1)] = match.group(2).strip()
    required = {"NAME", "TYPE", "DIMENSION", "EDGE_WEIGHT_TYPE"}
    if not required.issubset(headers) or not in_coordinates or not saw_eof:
        raise ValueError("incomplete TSPLIB EUC_2D file")
    if headers["TYPE"] != "TSP" or headers["EDGE_WEIGHT_TYPE"] != "EUC_2D":
        raise ValueError("only symmetric TSPLIB EUC_2D instances are allowed")
    if expected_name is not None and headers["NAME"].lower() != expected_name.lower():
        raise ValueError("TSPLIB instance name does not match the predeclared name")
    dimension = int(headers["DIMENSION"])
    if len(coordinates) != dimension:
        raise ValueError("TSPLIB coordinate count does not match DIMENSION")
    if [row["id"] for row in coordinates] != list(range(dimension)):
        raise ValueError(
            "TSPLIB node IDs must be contiguous and one-based in the source"
        )
    if len({(row["x"], row["y"]) for row in coordinates}) != dimension:
        raise ValueError("TSPLIB instance contains duplicate integer coordinates")
    return {
        "name": headers["NAME"],
        "n_cities": dimension,
        "cities": coordinates,
        "edge_weight_type": "EUC_2D",
        "time_limit_sec": 5.0,
    }


def fetch_tsplib(
    protocol_path: Path,
    global_freeze_path: Path,
    output_root: Path,
    accept_external_terms: bool,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path, require_launch_ready=True)
    freeze = json.loads(global_freeze_path.read_text())
    validate_global_freeze(protocol, freeze)
    if not accept_external_terms:
        raise ValueError("explicit --accept-external-terms is required")
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in TSPLIB_NAMES:
        url = protocol["benchmark"]["urls"][name]
        compressed = urllib.request.urlopen(url, timeout=60).read()  # noqa: S310
        raw = gzip.decompress(compressed) if url.endswith(".gz") else compressed
        path = output_root / f"{name}.tsp"
        path.write_bytes(raw)
        mission = parse_tsplib(path, name)
        rows.append(
            {
                "name": name,
                "url": url,
                "compressed_sha256": sha256_bytes(compressed),
                "raw_sha256": sha256_bytes(raw),
                "canonical_instance_sha256": sha256_text(
                    json.dumps(mission, sort_keys=True, separators=(",", ":"))
                ),
                "dimension": mission["n_cities"],
                "path": path.name,
            }
        )
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "created_unix": time.time(),
        "created_after_global_freeze": time.time() >= float(freeze["created_unix"]),
        "upstream": protocol["benchmark"]["upstream"],
        "license_notice": protocol["benchmark"]["license_notice"],
        "redistribution": "raw TSPLIB files remain external on capstor and are never committed or uploaded",
        "global_freeze_sha256": sha256_file(global_freeze_path),
        "instances": rows,
    }
    (output_root / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def pretest_gate(
    protocol_path: Path,
    global_freeze_path: Path,
    external_root: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path, require_launch_ready=True)
    freeze = json.loads(global_freeze_path.read_text())
    validate_global_freeze(protocol, freeze)
    provenance_path = external_root / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("global_freeze_sha256") != sha256_file(global_freeze_path):
        raise ValueError("external provenance is not bound to this global freeze")
    if not provenance.get("created_after_global_freeze"):
        raise ValueError("TSPLIB data was opened before global solver freeze")
    if [row["name"] for row in provenance["instances"]] != list(TSPLIB_NAMES):
        raise ValueError("external provenance instance order drift")
    external_hashes = set()
    for row in provenance["instances"]:
        path = external_root / row["path"]
        if sha256_file(path) != row["raw_sha256"]:
            raise ValueError(f"external TSPLIB hash drift: {row['name']}")
        mission = parse_tsplib(path, row["name"])
        canonical = sha256_text(
            json.dumps(mission, sort_keys=True, separators=(",", ":"))
        )
        if canonical != row["canonical_instance_sha256"]:
            raise ValueError(f"canonical TSPLIB hash drift: {row['name']}")
        external_hashes.add(canonical)
    for condition in CONDITIONS:
        for seed in SEEDS:
            selected = freeze["selected"][condition][str(seed)]
            solver = Path(selected["solver_path"])
            if sha256_file(solver) != selected["solver_sha256"]:
                raise ValueError("frozen solver hash drift before test")
            audit_prompt_leakage([solver.read_text()], protocol)
    return {
        "status": "pass",
        "six_solver_hashes": True,
        "external_instance_count": len(external_hashes),
        "prompt_leakage": "pass",
        "benchmark_may_run": True,
    }


def validate_global_freeze(
    protocol: dict[str, Any], freeze: dict[str, Any]
) -> dict[str, str]:
    """Verify all six immutable solvers before any external benchmark access."""
    if freeze.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("global freeze protocol drift")
    if not freeze.get("all_six_hashes_frozen"):
        raise ValueError("all six solvers must be frozen before TSPLIB access")
    if freeze.get("external_test_accessed") is not False:
        raise ValueError("external benchmark was already accessed")
    if freeze.get("prompt_leakage_audit") != "pass":
        raise ValueError("global freeze lacks a passing leakage audit")
    selected = freeze.get("selected")
    if not isinstance(selected, dict) or set(selected) != set(CONDITIONS):
        raise ValueError("global freeze does not contain both E12 conditions")
    verified: dict[str, str] = {}
    for condition in CONDITIONS:
        rows = selected[condition]
        if not isinstance(rows, dict) or set(rows) != {str(seed) for seed in SEEDS}:
            raise ValueError(f"global freeze does not contain three {condition} seeds")
        for seed in SEEDS:
            row = rows[str(seed)]
            solver = Path(row["solver_path"])
            expected = str(row.get("solver_sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("global freeze contains an invalid solver SHA-256")
            if not solver.is_file() or sha256_file(solver) != expected:
                raise ValueError("frozen solver hash drift before TSPLIB access")
            audit_prompt_leakage([solver.read_text()], protocol)
            verified[f"{condition}/seed{seed}"] = expected
    if len(verified) != 6:
        raise ValueError("global freeze must verify exactly six solver hashes")
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-protocol")
    verify.add_argument("--protocol", type=Path, required=True)
    verify.add_argument("--repo-root", type=Path, required=True)
    verify.add_argument("--launch-ready", action="store_true")

    freeze = commands.add_parser("freeze-development")
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--seed", type=int, required=True)
    freeze.add_argument("--dataset-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    prepare = commands.add_parser("prepare-selection")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--condition", choices=CONDITIONS, required=True)
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--development", type=Path, required=True)
    prepare.add_argument("--generations", type=Path, required=True)
    prepare.add_argument("--model-revision", required=True)
    prepare.add_argument("--inputs", type=Path, required=True)
    prepare.add_argument("--candidates", type=Path, required=True)

    evaluate = commands.add_parser("evaluate-selection")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--inputs", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--workers", type=int, default=64)

    select = commands.add_parser("select")
    select.add_argument("--protocol", type=Path, required=True)
    select.add_argument("--seed", type=int, required=True)
    select.add_argument("--condition", choices=CONDITIONS, required=True)
    select.add_argument("--metrics", type=Path, required=True)
    select.add_argument("--candidates", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)

    freeze_global = commands.add_parser("freeze-all")
    freeze_global.add_argument("--protocol", type=Path, required=True)
    freeze_global.add_argument("--selection-root", type=Path, required=True)
    freeze_global.add_argument("--source-commit", required=True)
    freeze_global.add_argument("--output", type=Path, required=True)

    training = commands.add_parser("record-training")
    training.add_argument("--protocol", type=Path, required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--dataset-root", type=Path, required=True)
    training.add_argument("--model-root", type=Path, required=True)
    training.add_argument("--source-commit", required=True)
    training.add_argument("--container-sha256", required=True)
    training.add_argument("--slurm-job-id", required=True)
    training.add_argument("--smoke", action="store_true")
    training.add_argument("--output", type=Path, required=True)

    fetch = commands.add_parser("fetch-tsplib")
    fetch.add_argument("--protocol", type=Path, required=True)
    fetch.add_argument("--global-freeze", type=Path, required=True)
    fetch.add_argument("--output-root", type=Path, required=True)
    fetch.add_argument("--accept-external-terms", action="store_true")

    bind = commands.add_parser("bind-existing-tsplib-for-correction")
    bind.add_argument("--protocol", type=Path, required=True)
    bind.add_argument("--global-freeze", type=Path, required=True)
    bind.add_argument("--source-root", type=Path, required=True)
    bind.add_argument("--output-root", type=Path, required=True)

    gate = commands.add_parser("pretest-gate")
    gate.add_argument("--protocol", type=Path, required=True)
    gate.add_argument("--global-freeze", type=Path, required=True)
    gate.add_argument("--external-root", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "verify-protocol":
        protocol = load_protocol(args.protocol, require_launch_ready=args.launch_ready)
        result = (
            verify_protocol_assets(args.protocol, args.repo_root)
            if args.launch_ready
            else {"protocol_id": protocol["protocol_id"], "status": "prelaunch-valid"}
        )
    elif args.command == "freeze-development":
        result = freeze_development(
            args.protocol, args.seed, args.dataset_root, args.output
        )
    elif args.command == "prepare-selection":
        result = prepare_selection(
            args.protocol,
            args.seed,
            args.condition,
            args.dataset_root,
            args.development,
            args.generations,
            args.model_revision,
            args.inputs,
            args.candidates,
        )
    elif args.command == "evaluate-selection":
        result = evaluate_selection(
            args.protocol, args.inputs, args.output, args.workers
        )
    elif args.command == "select":
        result = select_solver(
            args.protocol,
            args.seed,
            args.condition,
            args.metrics,
            args.candidates,
            args.output_dir,
        )
    elif args.command == "freeze-all":
        result = freeze_all(
            args.protocol, args.selection_root, args.source_commit, args.output
        )
    elif args.command == "record-training":
        result = record_training(
            args.protocol,
            args.seed,
            args.dataset_root,
            args.model_root,
            args.source_commit,
            args.container_sha256,
            args.slurm_job_id,
            args.smoke,
            args.output,
        )
    elif args.command == "fetch-tsplib":
        result = fetch_tsplib(
            args.protocol,
            args.global_freeze,
            args.output_root,
            args.accept_external_terms,
        )
    elif args.command == "bind-existing-tsplib-for-correction":
        result = bind_existing_tsplib_for_correction(
            args.protocol,
            args.global_freeze,
            args.source_root,
            args.output_root,
        )
    else:
        result = pretest_gate(args.protocol, args.global_freeze, args.external_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
