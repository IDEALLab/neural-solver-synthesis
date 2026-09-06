"""Run a frozen-model iterative SDS repair baseline without test feedback."""

# Optional cluster dependencies are imported only in the paths that need them,
# and protocol failures intentionally carry the violated invariant.
# ruff: noqa: PLC0415, PLR0915, TRY003

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import re
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from evaluation.sds.utils import (
    check_constraint_violations,
    deserialize_mission,
    mission_to_instance,
    run_candidate,
)

CODE_TAG_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
FAMILY_RE = re.compile(r"_[0-9]+$")
EPSILON = 1e-9


class ProtocolError(ValueError):
    """Raised when a frozen experimental invariant is violated."""


@dataclass(frozen=True)
class Candidate:
    completion_index: int
    round_index: int
    generated_text: str
    code: str | None
    code_sha256: str | None
    parent_code_sha256: str | None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def family_from_uuid(uuid: str) -> str:
    return FAMILY_RE.sub("", uuid)


def mission_sha256(mission: Any) -> str:
    return sha256_text(canonical_json(deserialize_mission(mission)))


def uuid_set_sha256(uuids: set[str]) -> str:
    return sha256_text("\n".join(sorted(uuids)) + "\n")


def deduplicate_rows_by_mission(
    rows: list[dict[str, Any]], salt: str
) -> list[dict[str, Any]]:
    """Retain one deterministically ranked UUID for each canonical mission."""
    selected: dict[str, tuple[str, dict[str, Any]]] = {}
    for row in rows:
        mission_hash = mission_sha256(row["mission"])
        rank = sha256_text(f"{salt}:{row['uuid']}:{mission_hash}")
        incumbent = selected.get(mission_hash)
        if incumbent is None or rank < incumbent[0]:
            selected[mission_hash] = (rank, row)
    return [entry[1] for entry in sorted(selected.values(), key=lambda entry: entry[0])]


def extract_code(generated_text: str) -> str | None:
    match = CODE_TAG_RE.search(generated_text or "")
    if not match:
        match = CODE_FENCE_RE.search(generated_text or "")
    if not match:
        return None
    code = match.group(1).replace("\r\n", "\n").replace("\r", "\n")
    code = "\n".join(line.rstrip() for line in code.splitlines()).strip()
    return f"{code}\n" if code else None


def completion_schedule(protocol: dict[str, Any]) -> list[int]:
    schedule = [int(protocol["initial_completions"])]
    schedule.extend(
        [int(protocol["completions_per_repair_round"])] * int(protocol["repair_rounds"])
    )
    if sum(schedule) != int(protocol["total_completion_budget"]):
        raise ProtocolError(
            f"completion schedule totals {sum(schedule)}, expected "
            f"{protocol['total_completion_budget']}"
        )
    return schedule


def select_balanced_rows(
    rows: list[dict[str, Any]],
    cases_per_family: int,
    salt: str,
    total_cases: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        uuid = str(row["uuid"])
        if uuid in seen:
            raise ProtocolError(f"duplicate validation UUID: {uuid}")
        seen.add(uuid)
        grouped[family_from_uuid(uuid)].append(row)

    ranked_by_family: dict[str, list[dict[str, Any]]] = {}
    selected: list[dict[str, Any]] = []
    for family in sorted(grouped):
        family_rows = grouped[family]
        if len(family_rows) < cases_per_family:
            raise ProtocolError(
                f"family {family} has {len(family_rows)} rows; need {cases_per_family}"
            )
        ranked_by_family[family] = sorted(
            family_rows,
            key=lambda row: sha256_text(
                f"{salt}:{row['uuid']}:{mission_sha256(row['mission'])}"
            ),
        )
        selected.extend(ranked_by_family[family][:cases_per_family])

    if total_cases is not None:
        remaining = total_cases - len(selected)
        if remaining < 0 or remaining > len(ranked_by_family):
            raise ProtocolError(
                f"cannot select {total_cases} near-balanced rows from "
                f"{len(ranked_by_family)} families at base depth {cases_per_family}"
            )
        extra_families = sorted(
            ranked_by_family,
            key=lambda family: sha256_text(f"{salt}:extra-family:{family}"),
        )[:remaining]
        for family in extra_families:
            family_rows = ranked_by_family[family]
            if len(family_rows) <= cases_per_family:
                raise ProtocolError(
                    f"family {family} lacks a row for near-balanced fill"
                )
            selected.append(family_rows[cases_per_family])
    return sorted(
        selected, key=lambda row: (family_from_uuid(str(row["uuid"])), str(row["uuid"]))
    )


def select_initial_prompt_rows(
    rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[family_from_uuid(str(row["uuid"]))].append(row)
    interleaved: list[dict[str, Any]] = []
    depth = 0
    while len(interleaved) < len(rows):
        added = False
        for family in sorted(grouped):
            family_rows = sorted(grouped[family], key=lambda row: str(row["uuid"]))
            if depth < len(family_rows):
                interleaved.append(family_rows[depth])
                added = True
        if not added:
            break
        depth += 1
    if count > len(interleaved):
        raise ProtocolError(f"requested {count} initial prompts from {len(rows)} rows")
    return interleaved[:count]


def assert_disjoint(
    dev_uuids: set[str], test_uuids_by_seed: dict[int, set[str]]
) -> None:
    for seed, test_uuids in sorted(test_uuids_by_seed.items()):
        overlap = sorted(dev_uuids & test_uuids)
        if overlap:
            raise ProtocolError(
                f"development/test UUID overlap for test seed {seed}: {overlap[:5]}"
            )


def assert_content_disjoint(
    development_hashes: set[str], test_hashes_by_seed: dict[int, set[str]]
) -> None:
    """Reject development instances whose canonical missions occur in any test set."""
    for seed, test_hashes in sorted(test_hashes_by_seed.items()):
        overlap = sorted(development_hashes & test_hashes)
        if overlap:
            raise ProtocolError(
                f"development/test mission-content overlap for test seed {seed}: "
                f"{overlap[:5]}"
            )


def load_protocol(path: Path) -> dict[str, Any]:  # noqa: PLR0912
    protocol = json.loads(path.read_text())
    content_policy = protocol.get("development_content_policy")
    if content_policy not in {None, "exclude_all_test_missions_and_deduplicate"}:
        raise ProtocolError(f"unknown development content policy: {content_policy}")
    completion_schedule(protocol)
    expected_families = int(protocol.get("expected_development_families", 10))
    base_cases = int(protocol["development_cases_per_family"]) * expected_families
    total_cases = int(protocol["expected_development_cases"])
    if not base_cases <= total_cases <= base_cases + expected_families:
        raise ProtocolError(
            "expected_development_cases does not match the declared near-balanced design"
        )
    test_evaluation = protocol.get("test_evaluation")
    if content_policy and not isinstance(test_evaluation, dict):
        raise ProtocolError(
            "content-clean protocol must freeze test evaluation settings"
        )
    if content_policy:
        expected_controller = {
            "initial_completions": 8,
            "repair_rounds": 7,
            "completions_per_repair_round": 8,
            "total_completion_budget": 64,
            "development_cases_per_family": 4,
            "expected_development_cases": 40,
            "expected_development_families": 9,
        }
        for key, expected_value in expected_controller.items():
            if protocol.get(key) != expected_value:
                raise ProtocolError(
                    f"content-clean full protocol requires {key}={expected_value!r}"
                )
        expected_max_model_len = (
            32768
            if protocol.get("protocol_id")
            in {
                "N26-E4-v9-content-clean-native-context",
                "N26-E4-v10-content-clean-bounded-diagnostics",
                "N26-E4-v11-content-clean-context-budgeted",
            }
            else 16384
        )
        expected_sampling = {
            "temperature": 0.6,
            "max_tokens": 4096,
            "max_model_len": expected_max_model_len,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": True,
        }
        if protocol.get("sampling") != expected_sampling:
            raise ProtocolError("content-clean full protocol sampling settings drifted")
        expected_development_execution = {
            "timeout_seconds": 5.0,
            "workers": 40,
            "diagnostic_cases": 6,
            "infeasible_gap_penalty": 1.0,
        }
        if protocol.get("development_execution") != expected_development_execution:
            raise ProtocolError(
                "content-clean full protocol development workload drifted"
            )
        expected_reference = {
            "solver": "OR-Tools CP-SAT",
            "ortools_version": "9.14.6206",
            "max_time_seconds": 600.0,
            "num_search_workers": 1,
        }
        observed_reference = protocol.get("development_reference", {})
        for key, expected_value in expected_reference.items():
            if observed_reference.get(key) != expected_value:
                raise ProtocolError(
                    f"content-clean full protocol development reference drifted: {key}"
                )
        expected_diagnostics = {
            "include_full_requirements": True,
            "include_candidate_selection": True,
            "candidate_selection_max_raw_ids": (
                128
                if protocol.get("protocol_id")
                in {
                    "N26-E4-v10-content-clean-bounded-diagnostics",
                    "N26-E4-v11-content-clean-context-budgeted",
                }
                else None
            ),
            "include_compact_all_benchmark_feasible": True,
            "exclude_proved_infeasible": True,
        }
        if protocol.get("protocol_id") == "N26-E4-v11-content-clean-context-budgeted":
            expected_diagnostics.update(
                {
                    "full_requirements_context_policy": "tokenizer_greedy_fit_rotating",
                    "target_full_requirement_cases": 6,
                    "minimum_full_requirement_cases": 1,
                }
            )
        if expected_diagnostics["candidate_selection_max_raw_ids"] is None:
            expected_diagnostics.pop("candidate_selection_max_raw_ids")
        if protocol.get("diagnostic_payload") != expected_diagnostics:
            raise ProtocolError(
                "content-clean protocol must provide full fair diagnostic payloads"
            )
    if test_evaluation:
        expected_test = {
            "rows_per_seed": int(protocol["expected_split_count"]),
            "time_budget_seconds": 5.0,
            "execution_repeats": 3,
            "workers": 64,
            "classical_baselines": "none",
            "quality_uses_first_execution": True,
            "join_keys": ["seed", "uuid"],
        }
        for key, expected_value in expected_test.items():
            if test_evaluation.get(key) != expected_value:
                raise ProtocolError(
                    f"test_evaluation {key}={test_evaluation.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
    return protocol


def freeze_manifest(protocol_path: Path, output_path: Path) -> None:
    from datasets import load_dataset

    protocol = load_protocol(protocol_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output_path}")

    test_uuids_by_seed: dict[int, set[str]] = {}
    test_mission_hashes_by_seed: dict[int, set[str]] = {}
    test_hashes: dict[str, str] = {}
    test_content_hashes: dict[str, str] = {}
    for seed_text, dataset_name in protocol["datasets"].items():
        seed = int(seed_text)
        revision = protocol["dataset_revisions"][seed_text]
        test_rows = load_dataset(
            dataset_name, split=protocol["test_split"], revision=revision
        )
        if len(test_rows) != int(protocol["expected_split_count"]):
            raise ProtocolError(
                f"unexpected test count for seed {seed}: {len(test_rows)}"
            )
        uuids = {str(uuid) for uuid in test_rows["uuid"]}
        if len(uuids) != len(test_rows):
            raise ProtocolError(f"duplicate test UUID for seed {seed}")
        test_uuids_by_seed[seed] = uuids
        test_hashes[seed_text] = uuid_set_sha256(uuids)
        mission_hashes = {mission_sha256(row["mission"]) for row in test_rows}
        test_mission_hashes_by_seed[seed] = mission_hashes
        test_content_hashes[seed_text] = uuid_set_sha256(mission_hashes)

    excluded_test_content = set().union(*test_mission_hashes_by_seed.values())

    frozen_seeds: dict[str, Any] = {}
    for seed_text, dataset_name in protocol["datasets"].items():
        seed = int(seed_text)
        revision = protocol["dataset_revisions"][seed_text]
        dev_rows = load_dataset(
            dataset_name, split=protocol["development_split"], revision=revision
        )
        if len(dev_rows) != int(protocol["expected_split_count"]):
            raise ProtocolError(
                f"unexpected validation count for seed {seed}: {len(dev_rows)}"
            )
        candidate_rows = list(dev_rows)
        if protocol.get("development_content_policy") == (
            "exclude_all_test_missions_and_deduplicate"
        ):
            candidate_rows = [
                row
                for row in candidate_rows
                if mission_sha256(row["mission"]) not in excluded_test_content
            ]
            candidate_rows = deduplicate_rows_by_mission(
                candidate_rows, f"{protocol['manifest_salt']}:deduplicate:seed{seed}"
            )
        selected = select_balanced_rows(
            candidate_rows,
            int(protocol["development_cases_per_family"]),
            f"{protocol['manifest_salt']}:seed{seed}",
            int(protocol["expected_development_cases"]),
        )
        selected_families = {family_from_uuid(str(row["uuid"])) for row in selected}
        if len(selected_families) != int(
            protocol.get("expected_development_families", 10)
        ):
            raise ProtocolError(
                f"balanced selection found {len(selected_families)} families"
            )
        if len(selected) != int(protocol["expected_development_cases"]):
            raise ProtocolError(
                f"balanced selection produced {len(selected)} development rows"
            )
        assert_disjoint({str(row["uuid"]) for row in selected}, test_uuids_by_seed)
        if protocol.get("development_content_policy"):
            selected_hashes = {mission_sha256(row["mission"]) for row in selected}
            if len(selected_hashes) != len(selected):
                raise ProtocolError("development manifest contains duplicate missions")
            assert_content_disjoint(selected_hashes, test_mission_hashes_by_seed)
        frozen_seeds[seed_text] = {
            "dataset": dataset_name,
            "dataset_revision": revision,
            "split": protocol["development_split"],
            "rows": [
                {
                    "uuid": str(row["uuid"]),
                    "family": family_from_uuid(str(row["uuid"])),
                    "mission_sha256": mission_sha256(row["mission"]),
                }
                for row in selected
            ],
        }

    expected_families = int(protocol.get("expected_development_families", 10))
    base_count = int(protocol["development_cases_per_family"]) * expected_families
    extra_count = int(protocol["expected_development_cases"]) - base_count
    selection_rule = "four lowest salted SHA-256 ranks per available UUID family"
    if extra_count:
        selection_rule += (
            f", then one additional row from {extra_count} families chosen by a "
            "separate salted family rank"
        )
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_text(protocol_path.read_text()),
        "selection_rule": selection_rule,
        "test_uuid_set_sha256": test_hashes,
        "test_mission_content_set_sha256": test_content_hashes,
        "development_content_policy": protocol.get("development_content_policy"),
        "expected_development_families": protocol.get(
            "expected_development_families", 10
        ),
        "seeds": frozen_seeds,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")


def load_frozen_rows(
    protocol: dict[str, Any], manifest: dict[str, Any], seed: int
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    seed_text = str(seed)
    if manifest["protocol_id"] != protocol["protocol_id"]:
        raise ProtocolError("manifest/protocol ID mismatch")
    frozen = manifest["seeds"][seed_text]
    dataset_name = protocol["datasets"][seed_text]
    revision = protocol["dataset_revisions"][seed_text]
    if frozen["dataset"] != dataset_name:
        raise ProtocolError("manifest dataset mismatch")
    if frozen["dataset_revision"] != revision:
        raise ProtocolError("manifest dataset revision mismatch")

    validation = load_dataset(
        dataset_name, split=protocol["development_split"], revision=revision
    )
    by_uuid = {str(row["uuid"]): row for row in validation}
    selected: list[dict[str, Any]] = []
    for frozen_row in frozen["rows"]:
        uuid = frozen_row["uuid"]
        if uuid not in by_uuid:
            raise ProtocolError(f"frozen validation UUID missing: {uuid}")
        row = dict(by_uuid[uuid])
        actual_sha = mission_sha256(row["mission"])
        if actual_sha != frozen_row["mission_sha256"]:
            raise ProtocolError(f"mission drift for validation UUID: {uuid}")
        selected.append(row)

    if len(selected) != int(protocol["expected_development_cases"]):
        raise ProtocolError(f"unexpected frozen development count: {len(selected)}")

    dev_uuids = {str(row["uuid"]) for row in selected}
    test_uuids_by_seed: dict[int, set[str]] = {}
    for test_seed_text, test_dataset in protocol["datasets"].items():
        test_rows = load_dataset(
            test_dataset,
            split=protocol["test_split"],
            revision=protocol["dataset_revisions"][test_seed_text],
        )
        test_uuids = {str(uuid) for uuid in test_rows["uuid"]}
        expected_hash = manifest["test_uuid_set_sha256"][test_seed_text]
        if uuid_set_sha256(test_uuids) != expected_hash:
            raise ProtocolError(f"test UUID set drift for seed {test_seed_text}")
        test_uuids_by_seed[int(test_seed_text)] = test_uuids
    assert_disjoint(dev_uuids, test_uuids_by_seed)
    if protocol.get("development_content_policy"):
        dev_hashes = {mission_sha256(row["mission"]) for row in selected}
        if len(dev_hashes) != len(selected):
            raise ProtocolError("frozen development rows contain duplicate missions")
        test_mission_hashes_by_seed = {
            int(test_seed_text): {mission_sha256(row["mission"]) for row in test_rows}
            for test_seed_text, test_dataset in protocol["datasets"].items()
            for test_rows in [
                load_dataset(
                    test_dataset,
                    split=protocol["test_split"],
                    revision=protocol["dataset_revisions"][test_seed_text],
                )
            ]
        }
        assert_content_disjoint(dev_hashes, test_mission_hashes_by_seed)
    return selected


def calculate_score(instance: Any, selected_ids: list[int]) -> float:
    total = sum(float(instance.w[index]) for index in selected_ids)
    selected = set(selected_ids)
    total += sum(
        float(weight)
        for (left, right), weight in instance.W.items()
        if left in selected and right in selected
    )
    return total


def normalize_selected_ids(value: Any, n_variables: int) -> list[int] | None:
    """Normalize SDS set membership while rejecting malformed or out-of-range IDs."""
    if not isinstance(value, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) for index in value
    ):
        return None
    if any(index < 0 or index >= n_variables for index in value):
        return None
    return list(dict.fromkeys(value))


def certify_dev_row(payload: tuple[dict[str, Any], float]) -> dict[str, Any]:
    from evaluation.sds.certify_optima import solve_instance

    row, max_time_seconds = payload
    instance = mission_to_instance(row["mission"])
    result = solve_instance(
        instance,
        max_time_sec=max_time_seconds,
        solver_seed=0,
        num_workers=1,
    )
    return {"uuid": str(row["uuid"]), **result}


def certify_dev_rows(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    reference = protocol["development_reference"]
    payloads = [(row, float(reference["max_time_seconds"])) for row in rows]
    results: list[dict[str, Any]] = []
    workers = min(len(rows), int(protocol["development_execution"]["workers"]))
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [executor.submit(certify_dev_row, payload) for payload in payloads]
        results.extend(future.result() for future in as_completed(futures))
    results.sort(key=lambda row: row["uuid"])
    return results


def certify_development(
    protocol_path: Path, manifest_path: Path, seed: int, output_path: Path
) -> None:
    import ortools

    protocol = load_protocol(protocol_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol_sha256") != sha256_text(protocol_path.read_text()):
        raise ProtocolError("frozen manifest was not generated from this protocol")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite references: {output_path}")
    expected_version = str(protocol["development_reference"]["ortools_version"])
    if ortools.__version__ != expected_version:
        raise ProtocolError(
            f"OR-Tools version {ortools.__version__} does not match {expected_version}"
        )
    rows = load_frozen_rows(protocol, manifest, seed)
    references = certify_dev_rows(rows, protocol)
    for reference in references:
        reference["ortools_version"] = ortools.__version__
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, references)


def load_development_references(
    path: Path, rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    references = [json.loads(line) for line in path.read_text().splitlines() if line]
    by_uuid = {str(row["uuid"]): row for row in references}
    expected_uuids = {str(row["uuid"]) for row in rows}
    if len(by_uuid) != len(references):
        raise ProtocolError("duplicate UUID in development references")
    if set(by_uuid) != expected_uuids:
        raise ProtocolError("development reference UUID set mismatch")
    versions = {str(row.get("ortools_version")) for row in references}
    expected_version = str(protocol["development_reference"]["ortools_version"])
    if versions != {expected_version}:
        raise ProtocolError("development reference OR-Tools version mismatch")
    references.sort(key=lambda row: str(row["uuid"]))
    return references, {
        uuid: float(row["best_bound"]) if row.get("best_bound") is not None else None
        for uuid, row in by_uuid.items()
    }


def build_stdin_obj(mission: Any) -> dict[str, Any]:
    mission_dict = deserialize_mission(mission)
    n_variables = int(mission_dict.get("n_variables", 0))
    interactions = mission_dict.get("interactions", {})
    weights = mission_dict.get("weights", [0.0] * n_variables)
    adjacency = {index: [] for index in range(n_variables)}
    for pair in interactions:
        with contextlib.suppress(ValueError):
            left, right = map(int, pair.split(","))
            if left in adjacency and right in adjacency:
                adjacency[left].append(right)
                adjacency[right].append(left)
    requirements = {
        "n_variables": n_variables,
        "cardinality_bounds": mission_dict.get("cardinality_bounds", [0, n_variables]),
        "precedence": mission_dict.get("precedence", []),
        "mutex": mission_dict.get("mutex", []),
        "groups": mission_dict.get("groups", {}),
        "weights": weights,
        "interactions": interactions,
    }
    return {
        "requirements": requirements,
        "catalog": {
            "variables": [
                {
                    "id": index,
                    "weight": weights[index],
                    "neighbors": adjacency[index],
                }
                for index in range(n_variables)
            ],
            "interactions": interactions,
        },
    }


def evaluate_one(
    payload: tuple[Candidate, dict[str, Any], float | None, str, float, float],
) -> dict[str, Any]:
    (
        candidate,
        row,
        reference_score,
        reference_status,
        timeout_seconds,
        infeasible_penalty,
    ) = payload
    uuid = str(row["uuid"])
    benchmark_feasible = reference_status != "INFEASIBLE"
    if candidate.code is None:
        return {
            "completion_index": candidate.completion_index,
            "code_sha256": None,
            "uuid": uuid,
            "feasible": False,
            "benchmark_feasible": benchmark_feasible,
            "reference_status": reference_status,
            "score": 0.0,
            "reference_score": reference_score,
            "penalized_gap": (
                infeasible_penalty if reference_score is not None else None
            ),
            "execution_time": 0.0,
            "error_type": "no_code",
            "error": "No code block was extracted.",
        }

    mission = deserialize_mission(row["mission"])
    instance = mission_to_instance(mission)
    result = run_candidate(
        candidate.code, build_stdin_obj(mission), timeout=timeout_seconds
    )
    base = {
        "completion_index": candidate.completion_index,
        "code_sha256": candidate.code_sha256,
        "uuid": uuid,
        "reference_score": reference_score,
        "benchmark_feasible": benchmark_feasible,
        "reference_status": reference_status,
        "execution_time": float(result.get("execution_time", 0.0)),
    }
    if "selection" not in result:
        return {
            **base,
            "feasible": False,
            "score": 0.0,
            "penalized_gap": (
                infeasible_penalty if reference_score is not None else None
            ),
            "error_type": result.get("error_type", "unknown"),
            "error": str(result.get("error", "missing selection"))[:500],
        }

    selection = result.get("selection", {})
    raw_selected_ids = (
        selection.get("variables", []) if isinstance(selection, dict) else []
    )
    selected_ids = normalize_selected_ids(raw_selected_ids, instance.n)
    if selected_ids is None:
        return {
            **base,
            "feasible": False,
            "score": 0.0,
            "penalized_gap": (
                infeasible_penalty if reference_score is not None else None
            ),
            "error_type": "invalid_selection",
            "error": "selection must contain in-range integer IDs",
        }
    selection_counts = {
        "raw_selection_count": len(raw_selected_ids),
        "unique_selection_count": len(selected_ids),
        "duplicate_selection_count": len(raw_selected_ids) - len(selected_ids),
    }
    violations = check_constraint_violations(instance, selected_ids)
    if not violations["all_valid"]:
        return {
            **base,
            **selection_counts,
            "selected_ids": selected_ids,
            "feasible": False,
            "score": 0.0,
            "penalized_gap": (
                infeasible_penalty if reference_score is not None else None
            ),
            "error_type": "constraint",
            "violations": violations,
        }
    score = calculate_score(instance, selected_ids)
    gap = None
    if reference_score is not None:
        gap = (
            max(0.0, (reference_score - max(0.0, score)) / reference_score)
            if reference_score > EPSILON
            else 0.0
        )
    return {
        **base,
        **selection_counts,
        "selected_ids": selected_ids,
        "feasible": True,
        "score": score,
        "penalized_gap": gap,
        "error_type": "none",
    }


def summarize_candidate(
    candidate: Candidate, evaluations: list[dict[str, Any]]
) -> dict[str, Any]:
    benchmark_rows = [
        row for row in evaluations if bool(row.get("benchmark_feasible", True))
    ]
    proved_infeasible = [
        row for row in evaluations if not bool(row.get("benchmark_feasible", True))
    ]
    if not benchmark_rows:
        raise ProtocolError("candidate evaluation lacks benchmark-feasible rows")
    feasible = sum(bool(row["feasible"]) for row in benchmark_rows)
    count = len(benchmark_rows)
    bounded_gaps = [
        float(row["penalized_gap"])
        for row in benchmark_rows
        if row["penalized_gap"] is not None
    ]
    return {
        **asdict(candidate),
        "generated_text": None,
        "code": None,
        "development_evaluations": len(evaluations),
        "benchmark_feasible_development_evaluations": count,
        "proved_infeasible_development_evaluations": len(proved_infeasible),
        "false_feasible_on_proved_infeasible": sum(
            bool(row["feasible"]) for row in proved_infeasible
        ),
        "feasible_count": feasible,
        "pass_rate": feasible / count,
        "bounded_reference_evaluations": len(bounded_gaps),
        "mean_penalized_gap": (
            sum(bounded_gaps) / len(bounded_gaps) if bounded_gaps else float("inf")
        ),
        "mean_execution_time": sum(
            float(row["execution_time"]) for row in benchmark_rows
        )
        / count,
    }


def candidate_rank_key(summary: dict[str, Any]) -> tuple[float, float, int]:
    return (
        -float(summary["pass_rate"]),
        float(summary["mean_penalized_gap"]),
        int(summary["completion_index"]),
    )


def diagnostic_rows(
    evaluations: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    evaluations = [
        row for row in evaluations if bool(row.get("benchmark_feasible", True))
    ]
    failures = sorted(
        (row for row in evaluations if not row["feasible"]),
        key=lambda row: (str(row["error_type"]), str(row["uuid"])),
    )
    feasible = sorted(
        (row for row in evaluations if row["feasible"]),
        key=lambda row: (
            (
                -float(row["penalized_gap"])
                if row["penalized_gap"] is not None
                else float("inf")
            ),
            str(row["uuid"]),
        ),
    )

    def diverse_prefix(
        ordered: list[dict[str, Any]], count: int
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        selected_ids: set[int] = set()
        seen_families: set[str] = set()
        for index, row in enumerate(ordered):
            family = family_from_uuid(str(row["uuid"]))
            if family not in seen_families:
                selected.append(row)
                selected_ids.add(index)
                seen_families.add(family)
                if len(selected) == count:
                    return selected
        for index, row in enumerate(ordered):
            if index not in selected_ids:
                selected.append(row)
                if len(selected) == count:
                    break
        return selected

    selected = diverse_prefix(failures, limit)
    if len(selected) < limit:
        existing_families = {family_from_uuid(str(row["uuid"])) for row in selected}
        remaining = [
            row
            for row in feasible
            if family_from_uuid(str(row["uuid"])) not in existing_families
        ]
        remaining.extend(
            row
            for row in feasible
            if family_from_uuid(str(row["uuid"])) in existing_families
        )
        selected.extend(diverse_prefix(remaining, limit - len(selected)))
    return selected[:limit]


def compact_diagnostic_line(row: dict[str, Any]) -> str:
    family = family_from_uuid(str(row["uuid"]))
    selection_count = row.get("unique_selection_count")
    if selection_count is None and isinstance(row.get("selected_ids"), list):
        selection_count = len(row["selected_ids"])
    if row["feasible"]:
        gap = (
            f"{100.0 * float(row['penalized_gap']):.2f}%"
            if row.get("penalized_gap") is not None
            else "unresolved"
        )
        detail = f"feasible score={float(row['score']):.6g} gap={gap}"
    else:
        error = str(row.get("error", row.get("violations", "")))[:160]
        detail = f"failed {row['error_type']} detail={error}"
    return f"- {row['uuid']} [{family}]: {detail}; selected_count={selection_count}"


def compact_selection_diagnostic(
    selected_ids: Any, *, n_variables: int, max_raw_ids: int
) -> dict[str, Any]:
    if max_raw_ids <= 0:
        raise ProtocolError("candidate selection diagnostic limit must be positive")
    if not isinstance(selected_ids, list):
        return {
            "raw_type": type(selected_ids).__name__,
            "raw_repr": repr(selected_ids)[:512],
        }
    integer_ids = [
        value
        for value in selected_ids
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    unique_integer_ids = list(dict.fromkeys(integer_ids))
    return {
        "raw_count": len(selected_ids),
        "unique_integer_count": len(unique_integer_ids),
        "duplicate_integer_occurrences": len(integer_ids) - len(unique_integer_ids),
        "boolean_count": sum(isinstance(value, bool) for value in selected_ids),
        "non_integer_count": sum(not isinstance(value, int) for value in selected_ids),
        "out_of_range_integer_count": sum(
            value < 0 or value >= n_variables for value in integer_ids
        ),
        "raw_prefix": selected_ids[:max_raw_ids],
        "raw_ids_truncated": max(0, len(selected_ids) - max_raw_ids),
    }


def format_feedback(
    summary: dict[str, Any],
    evaluations: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    limit: int,
    *,
    include_full_requirements: bool | None = None,
) -> str:
    rows_by_uuid = {str(row["uuid"]): row for row in development_rows}
    diagnostic_payload = protocol.get("diagnostic_payload", {})
    if include_full_requirements is None:
        include_full_requirements = bool(
            diagnostic_payload.get("include_full_requirements")
        )
    lines = [
        "DEVELOPMENT-ONLY EXECUTION SUMMARY",
        f"feasible: {summary['feasible_count']}/"
        f"{summary['benchmark_feasible_development_evaluations']} "
        f"({100.0 * summary['pass_rate']:.1f}%)",
        f"proved-infeasible rows excluded: "
        f"{summary['proved_infeasible_development_evaluations']}",
        f"bounded-reference failure-penalized gap: "
        f"{100.0 * summary['mean_penalized_gap']:.2f}% over "
        f"{summary['bounded_reference_evaluations']}/"
        f"{summary['benchmark_feasible_development_evaluations']} "
        "benchmark-feasible cases",
        f"mean execution time: {summary['mean_execution_time']:.4f} s",
    ]
    benchmark_evaluations = sorted(
        (row for row in evaluations if bool(row.get("benchmark_feasible", True))),
        key=lambda row: str(row["uuid"]),
    )
    if diagnostic_payload.get("include_compact_all_benchmark_feasible"):
        lines.append("All benchmark-feasible development cases (compact):")
        lines.extend(compact_diagnostic_line(row) for row in benchmark_evaluations)
    lines.append("Failure-first, family-diverse detailed cases:")
    for row in diagnostic_rows(evaluations, limit):
        if row["feasible"]:
            if row["reference_score"] is None:
                detail = f"score={row['score']:.6g}, reference_upper=unresolved"
            else:
                detail = (
                    f"score={row['score']:.6g}, "
                    f"reference_upper={row['reference_score']:.6g}, "
                    f"gap={100.0 * row['penalized_gap']:.2f}%"
                )
        else:
            detail = f"error_type={row['error_type']}, detail={str(row.get('error', row.get('violations', '')))[:240]}"
        lines.append(f"- {row['uuid']}: {detail}")
        if diagnostic_payload.get("include_candidate_selection"):
            development = rows_by_uuid[str(row["uuid"])]
            requirements = build_stdin_obj(development["mission"])["requirements"]
            max_raw_ids = int(
                diagnostic_payload.get("candidate_selection_max_raw_ids", 10**9)
            )
            lines.append(
                "  candidate_selection_diagnostic="
                + json.dumps(
                    compact_selection_diagnostic(
                        row.get("selected_ids"),
                        n_variables=int(requirements["n_variables"]),
                        max_raw_ids=max_raw_ids,
                    ),
                    separators=(",", ":"),
                )
            )
        if include_full_requirements:
            development = rows_by_uuid[str(row["uuid"])]
            requirements = build_stdin_obj(development["mission"])["requirements"]
            lines.append(
                "  development_requirements="
                + json.dumps(requirements, sort_keys=True, separators=(",", ":"))
            )
    return "\n".join(lines)


def make_candidate(
    completion_index: int,
    round_index: int,
    generated_text: str,
    parent_code_sha256: str | None,
) -> Candidate:
    code = extract_code(generated_text)
    return Candidate(
        completion_index=completion_index,
        round_index=round_index,
        generated_text=generated_text,
        code=code,
        code_sha256=sha256_text(code) if code else None,
        parent_code_sha256=parent_code_sha256,
    )


def evaluate_candidates(
    candidates: list[Candidate],
    rows: list[dict[str, Any]],
    references: dict[str, float | None],
    protocol: dict[str, Any],
    reference_statuses: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    execution = protocol["development_execution"]
    reference_statuses = reference_statuses or {
        str(row["uuid"]): "FEASIBLE" for row in rows
    }
    payloads = [
        (
            candidate,
            row,
            references[str(row["uuid"])],
            reference_statuses[str(row["uuid"])],
            float(execution["timeout_seconds"]),
            float(execution["infeasible_gap_penalty"]),
        )
        for candidate in candidates
        for row in rows
    ]
    by_candidate: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with ProcessPoolExecutor(
        max_workers=int(execution["workers"]),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = [executor.submit(evaluate_one, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            by_candidate[int(result["completion_index"])].append(result)
    summaries = []
    for candidate in candidates:
        evaluations = sorted(
            by_candidate[candidate.completion_index], key=lambda row: row["uuid"]
        )
        if len(evaluations) != len(rows):
            raise ProtocolError("candidate development evaluation count mismatch")
        by_candidate[candidate.completion_index] = evaluations
        summaries.append(summarize_candidate(candidate, evaluations))
    return summaries, by_candidate


def load_system_prompt(path: Path) -> str:
    import yaml

    config = yaml.safe_load(path.read_text())
    prompt = str(config.get("system_prompt", "")).strip()
    if not prompt:
        raise ProtocolError(f"system prompt missing from {path}")
    return prompt


def format_chat(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def prompt_token_count(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def assert_prompt_budget(
    tokenizer: Any, prompt: str, max_model_len: int, max_tokens: int
) -> int:
    count = prompt_token_count(tokenizer, prompt)
    available = max_model_len - max_tokens
    if count > available:
        raise ProtocolError(
            f"repair prompt has {count} tokens but only {available} are available"
        )
    return count


def build_budgeted_repair_user_prompt(
    system_prompt: str,
    incumbent_code: str,
    summary: dict[str, Any],
    evaluations: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    round_index: int,
    input_token_limit: int,
    count_messages: Callable[[str, str], int],
) -> tuple[str, int, list[str]]:
    """Pack exact diagnostics deterministically without truncating a requirement."""
    diagnostic_payload = protocol["diagnostic_payload"]
    base_feedback = format_feedback(
        summary,
        evaluations,
        development_rows,
        protocol,
        int(protocol["development_execution"]["diagnostic_cases"]),
        include_full_requirements=False,
    )
    base_user_prompt = (
        f"{protocol['repair_instruction']}\n\n"
        f"INCUMBENT CODE\n<code>\n{incumbent_code}</code>\n\n{base_feedback}"
    )
    base_tokens = count_messages(system_prompt, base_user_prompt)
    if base_tokens > input_token_limit:
        raise ProtocolError(
            f"repair prompt has {base_tokens} tokens but only "
            f"{input_token_limit} are available"
        )
    if not diagnostic_payload.get("include_full_requirements"):
        return base_user_prompt, base_tokens, []

    rows_by_uuid = {str(row["uuid"]): row for row in development_rows}
    ordered = diagnostic_rows(evaluations, len(development_rows))
    target = int(diagnostic_payload["target_full_requirement_cases"])
    minimum = int(diagnostic_payload["minimum_full_requirement_cases"])
    offset = ((round_index - 1) * target) % len(ordered)
    rotated = ordered[offset:] + ordered[:offset]
    blocks: list[str] = []
    included_uuids: list[str] = []
    prompt = base_user_prompt
    prompt_tokens = base_tokens
    for evaluation in rotated:
        uuid = str(evaluation["uuid"])
        requirements = build_stdin_obj(rows_by_uuid[uuid]["mission"])["requirements"]
        block = f"- {uuid} exact_development_requirements=" + json.dumps(
            requirements, sort_keys=True, separators=(",", ":")
        )
        candidate_blocks = [*blocks, block]
        candidate_user_prompt = (
            f"{base_user_prompt}\n\n"
            "Context-budgeted exact development requirements "
            "(complete payloads, never truncated):\n" + "\n".join(candidate_blocks)
        )
        candidate_tokens = count_messages(system_prompt, candidate_user_prompt)
        if candidate_tokens <= input_token_limit:
            blocks = candidate_blocks
            included_uuids.append(uuid)
            prompt = candidate_user_prompt
            prompt_tokens = candidate_tokens
            if len(blocks) == target:
                break
    if len(blocks) < minimum:
        raise ProtocolError(
            "repair feedback cannot fit the minimum complete requirement payload"
        )
    return prompt, prompt_tokens, included_uuids


def build_budgeted_repair_prompt(
    tokenizer: Any,
    system_prompt: str,
    incumbent_code: str,
    summary: dict[str, Any],
    evaluations: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    round_index: int,
) -> tuple[str, int, list[str]]:
    """Build a native chat-template repair prompt for the local vLLM controller."""
    sampling = protocol["sampling"]

    def count_messages(system: str, user: str) -> int:
        return prompt_token_count(tokenizer, format_chat(tokenizer, system, user))

    user_prompt, prompt_tokens, included_uuids = build_budgeted_repair_user_prompt(
        system_prompt,
        incumbent_code,
        summary,
        evaluations,
        development_rows,
        protocol,
        round_index,
        int(sampling["max_model_len"]) - int(sampling["max_tokens"]),
        count_messages,
    )
    return (
        format_chat(tokenizer, system_prompt, user_prompt),
        prompt_tokens,
        included_uuids,
    )


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, text=True).strip()

    return {
        "top_level_commit": run("git", "rev-parse", "HEAD"),
        "submodules": run("git", "submodule", "status", "--recursive").splitlines(),
        "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=all")),
    }


def resolved_model_commit(model: Any) -> str | None:
    """Return the cached Transformers commit exposed by vLLM, when available."""
    engine = getattr(model, "llm_engine", None)
    model_config = getattr(engine, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    commit = getattr(hf_config, "_commit_hash", None)
    return str(commit) if commit else None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_repair(
    protocol_path: Path,
    manifest_path: Path,
    development_references_path: Path,
    seed: int,
    output_dir: Path,
) -> None:
    from vllm import LLM, SamplingParams

    started = time.time()
    protocol = load_protocol(protocol_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol_sha256") != sha256_text(protocol_path.read_text()):
        raise ProtocolError("frozen manifest was not generated from this protocol")
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse immutable output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    provenance = git_provenance()
    if provenance["dirty"]:
        raise ProtocolError("source checkout is dirty")
    rows = load_frozen_rows(protocol, manifest, seed)
    references_raw, references = load_development_references(
        development_references_path, rows, protocol
    )
    reference_statuses = {
        str(row["uuid"]): str(row["status"]) for row in references_raw
    }

    system_prompt_path = Path(protocol["system_prompt_config"])
    system_prompt = load_system_prompt(system_prompt_path)
    sampling = protocol["sampling"]
    model = LLM(
        model=protocol["model"],
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=bool(sampling["enforce_eager"]),
        gpu_memory_utilization=float(sampling["gpu_memory_utilization"]),
        max_model_len=int(sampling["max_model_len"]),
    )
    model_commit = resolved_model_commit(model)
    tokenizer = model.get_tokenizer()

    all_candidates: list[Candidate] = []
    all_summaries: list[dict[str, Any]] = []
    all_evaluations: dict[int, list[dict[str, Any]]] = {}
    generation_records: list[dict[str, Any]] = []
    completion_index = 0
    schedule = completion_schedule(protocol)

    initial_rows = select_initial_prompt_rows(rows, schedule[0])
    initial_prompts = [
        format_chat(tokenizer, system_prompt, str(row["problem"]))
        for row in initial_rows
    ]
    initial_prompt_tokens = [
        assert_prompt_budget(
            tokenizer,
            prompt,
            int(sampling["max_model_len"]),
            int(sampling["max_tokens"]),
        )
        for prompt in initial_prompts
    ]
    initial_params = SamplingParams(
        temperature=float(sampling["temperature"]),
        max_tokens=int(sampling["max_tokens"]),
        n=1,
        seed=seed * 1000,
    )
    generated = model.generate(initial_prompts, initial_params)
    round_candidates = []
    for prompt, prompt_tokens, output in zip(
        initial_prompts, initial_prompt_tokens, generated, strict=True
    ):
        text = output.outputs[0].text
        candidate = make_candidate(completion_index, 0, text, None)
        round_candidates.append(candidate)
        generation_records.append(
            {
                "completion_index": completion_index,
                "round_index": 0,
                "parent_code_sha256": None,
                "prompt_sha256": sha256_text(prompt),
                "prompt_token_count": prompt_tokens,
                "prompt": prompt,
                "generated_text": text,
                "code_sha256": candidate.code_sha256,
            }
        )
        completion_index += 1

    summaries, evaluations = evaluate_candidates(
        round_candidates, rows, references, protocol, reference_statuses
    )
    all_candidates.extend(round_candidates)
    all_summaries.extend(summaries)
    all_evaluations.update(evaluations)

    for round_index, round_count in enumerate(schedule[1:], start=1):
        incumbent_summary = min(all_summaries, key=candidate_rank_key)
        incumbent = next(
            candidate
            for candidate in all_candidates
            if candidate.completion_index == incumbent_summary["completion_index"]
        )
        repair_prompt, repair_prompt_tokens, exact_requirement_uuids = (
            build_budgeted_repair_prompt(
                tokenizer,
                system_prompt,
                incumbent.code or "",
                incumbent_summary,
                all_evaluations[incumbent.completion_index],
                rows,
                protocol,
                round_index,
            )
        )
        repair_params = SamplingParams(
            temperature=float(sampling["temperature"]),
            max_tokens=int(sampling["max_tokens"]),
            n=round_count,
            seed=seed * 1000 + round_index,
        )
        output = model.generate([repair_prompt], repair_params)[0]
        if len(output.outputs) != round_count:
            raise ProtocolError("vLLM completion count mismatch")
        round_candidates = []
        for completion in output.outputs:
            candidate = make_candidate(
                completion_index,
                round_index,
                completion.text,
                incumbent.code_sha256,
            )
            round_candidates.append(candidate)
            generation_records.append(
                {
                    "completion_index": completion_index,
                    "round_index": round_index,
                    "parent_code_sha256": incumbent.code_sha256,
                    "prompt_sha256": sha256_text(repair_prompt),
                    "prompt_token_count": repair_prompt_tokens,
                    "exact_requirement_uuids": exact_requirement_uuids,
                    "exact_requirement_count": len(exact_requirement_uuids),
                    "prompt": repair_prompt,
                    "generated_text": completion.text,
                    "code_sha256": candidate.code_sha256,
                }
            )
            completion_index += 1
        summaries, evaluations = evaluate_candidates(
            round_candidates, rows, references, protocol, reference_statuses
        )
        all_candidates.extend(round_candidates)
        all_summaries.extend(summaries)
        all_evaluations.update(evaluations)

    if completion_index != int(protocol["total_completion_budget"]):
        raise ProtocolError(f"produced {completion_index} completions")
    selected_summary = min(all_summaries, key=candidate_rank_key)
    selected = next(
        candidate
        for candidate in all_candidates
        if candidate.completion_index == selected_summary["completion_index"]
    )
    if selected.code is None:
        raise ProtocolError("selected candidate has no code")

    (output_dir / "selected_solver.py").write_text(selected.code)
    write_jsonl(output_dir / "generations.jsonl", generation_records)
    write_jsonl(
        output_dir / "candidate_summaries.jsonl",
        sorted(all_summaries, key=lambda row: row["completion_index"]),
    )
    write_jsonl(
        output_dir / "development_evaluations.jsonl",
        [row for index in sorted(all_evaluations) for row in all_evaluations[index]],
    )
    run_manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_text(protocol_path.read_text()),
        "frozen_manifest_sha256": sha256_text(manifest_path.read_text()),
        "development_references_sha256": sha256_text(
            development_references_path.read_text()
        ),
        "seed": seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": protocol["model"],
        "resolved_model_commit": model_commit,
        "system_prompt_sha256": sha256_text(system_prompt),
        "development_uuids": [str(row["uuid"]) for row in rows],
        "development_mission_sha256": {
            str(row["uuid"]): mission_sha256(row["mission"]) for row in rows
        },
        "completion_count": completion_index,
        "development_execution_count": completion_index * len(rows),
        "development_reference_certified_count": sum(
            bool(row["certified_optimal"]) for row in references_raw
        ),
        "development_reference_certified_rate": sum(
            bool(row["certified_optimal"]) for row in references_raw
        )
        / len(references_raw),
        "development_reference_status_counts": {
            status: sum(str(row["status"]) == status for row in references_raw)
            for status in sorted({str(row["status"]) for row in references_raw})
        },
        "development_reference_objective_bound_count": sum(
            row.get("objective") is not None and row.get("best_bound") is not None
            for row in references_raw
        ),
        "development_reference_objective_bound_rate": sum(
            row.get("objective") is not None and row.get("best_bound") is not None
            for row in references_raw
        )
        / len(references_raw),
        "development_reference_ortools_version": str(
            references_raw[0]["ortools_version"]
        ),
        "development_reference_max_time_seconds": float(
            protocol["development_reference"]["max_time_seconds"]
        ),
        "selected_completion_index": selected.completion_index,
        "selected_code_sha256": selected.code_sha256,
        "selected_summary": selected_summary,
        "synthesis_wall_clock_seconds": time.time() - started,
        "provenance": provenance,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )


def rescore_immutable_candidates(  # noqa: PLR0913
    protocol_path: Path,
    manifest_path: Path,
    development_references_path: Path,
    source_synthesis_dir: Path,
    seed: int,
    output_dir: Path,
) -> None:
    """Re-evaluate an immutable generated pool after evaluator-only corrections."""
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse immutable output directory: {output_dir}"
        )
    protocol = load_protocol(protocol_path)
    manifest = json.loads(manifest_path.read_text())
    source_manifest_path = source_synthesis_dir / "run_manifest.json"
    generations_path = source_synthesis_dir / "generations.jsonl"
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ProtocolError("source run/protocol ID mismatch")
    if source_manifest.get("protocol_sha256") != sha256_text(protocol_path.read_text()):
        raise ProtocolError("source run/protocol hash mismatch")

    rows = load_frozen_rows(protocol, manifest, seed)
    references_raw, references = load_development_references(
        development_references_path, rows, protocol
    )
    reference_statuses = {
        str(row["uuid"]): str(row["status"]) for row in references_raw
    }
    records = [
        json.loads(line) for line in generations_path.read_text().splitlines() if line
    ]
    records.sort(key=lambda row: int(row["completion_index"]))
    expected = int(protocol["total_completion_budget"])
    if [int(row["completion_index"]) for row in records] != list(range(expected)):
        raise ProtocolError("immutable generation pool is incomplete or reordered")

    candidates = []
    for record in records:
        candidate = make_candidate(
            int(record["completion_index"]),
            int(record["round_index"]),
            str(record["generated_text"]),
            record.get("parent_code_sha256"),
        )
        if candidate.code_sha256 != record.get("code_sha256"):
            raise ProtocolError(
                f"immutable code hash mismatch at completion {candidate.completion_index}"
            )
        candidates.append(candidate)

    summaries, evaluations = evaluate_candidates(
        candidates, rows, references, protocol, reference_statuses
    )
    selected_summary = min(summaries, key=candidate_rank_key)
    selected = next(
        candidate
        for candidate in candidates
        if candidate.completion_index == selected_summary["completion_index"]
    )
    if selected.code is None:
        raise ProtocolError("corrected selection has no code")

    output_dir.mkdir(parents=True)
    (output_dir / "selected_solver.py").write_text(selected.code)
    write_jsonl(
        output_dir / "candidate_summaries.jsonl",
        sorted(summaries, key=lambda row: row["completion_index"]),
    )
    write_jsonl(
        output_dir / "development_evaluations.jsonl",
        [row for index in sorted(evaluations) for row in evaluations[index]],
    )
    correction = {
        "protocol_id": protocol["protocol_id"],
        "correction_id": "N26-E4-unique-selection-rescore-v1",
        "seed": seed,
        "source_synthesis_dir": str(source_synthesis_dir),
        "source_run_manifest_sha256": sha256_text(source_manifest_path.read_text()),
        "source_generations_sha256": sha256_text(generations_path.read_text()),
        "completion_count": len(candidates),
        "development_execution_count": len(candidates) * len(rows),
        "original_selected_completion_index": int(
            source_manifest["selected_completion_index"]
        ),
        "corrected_selected_completion_index": selected.completion_index,
        "selection_changed": selected.completion_index
        != int(source_manifest["selected_completion_index"]),
        "selected_code_sha256": selected.code_sha256,
        "selected_summary": selected_summary,
        "scoring_semantics": "SDS variables are set-valued; repeated IDs are normalized before feasibility and objective scoring.",
        "provenance": git_provenance(),
    }
    (output_dir / "correction_manifest.json").write_text(
        json.dumps(correction, indent=2) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-manifest")
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--development-references", type=Path, required=True)
    run.add_argument("--seed", type=int, choices=[101, 202, 303], required=True)
    run.add_argument("--output-dir", type=Path, required=True)

    certify = subparsers.add_parser("certify-development")
    certify.add_argument("--protocol", type=Path, required=True)
    certify.add_argument("--manifest", type=Path, required=True)
    certify.add_argument("--seed", type=int, choices=[101, 202, 303], required=True)
    certify.add_argument("--output", type=Path, required=True)

    rescore = subparsers.add_parser("rescore-immutable")
    rescore.add_argument("--protocol", type=Path, required=True)
    rescore.add_argument("--manifest", type=Path, required=True)
    rescore.add_argument("--development-references", type=Path, required=True)
    rescore.add_argument("--source-synthesis-dir", type=Path, required=True)
    rescore.add_argument("--seed", type=int, choices=[101, 202, 303], required=True)
    rescore.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze-manifest":
        freeze_manifest(args.protocol, args.output)
    elif args.command == "certify-development":
        certify_development(args.protocol, args.manifest, args.seed, args.output)
    elif args.command == "run":
        run_repair(
            args.protocol,
            args.manifest,
            args.development_references,
            args.seed,
            args.output_dir,
        )
    else:
        rescore_immutable_candidates(
            args.protocol,
            args.manifest,
            args.development_references,
            args.source_synthesis_dir,
            args.seed,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
