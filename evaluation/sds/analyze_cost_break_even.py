#!/usr/bin/env python3
"""Create an auditable end-to-end cost and break-even report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any


class InputHashError(RuntimeError):
    def __init__(self, path: Path, actual: str, expected: str):
        super().__init__(f"SHA-256 mismatch for {path}: {actual} != {expected}")


class MissingMethodError(RuntimeError):
    def __init__(self, method: str):
        super().__init__(f"missing method in solver cost input: {method}")


class NonpositiveSavingsError(ValueError):
    def __init__(self):
        super().__init__("marginal savings must be positive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_json(repo_root: Path, spec: dict[str, str]) -> Any:
    path = repo_root / spec["path"]
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise InputHashError(path, actual, spec["sha256"])
    return json.loads(path.read_text())


def method_mean(rows: list[dict[str, Any]], method: str, field: str) -> float:
    values = [float(row[field]) for row in rows if row["Method"] == method]
    if not values:
        raise MissingMethodError(method)
    return mean(values)


def ceiling_break_even(upfront_seconds: float, marginal_seconds: float) -> int:
    if marginal_seconds <= 0:
        raise NonpositiveSavingsError
    return math.ceil(upfront_seconds / marginal_seconds)


def phase_row(
    method: str,
    seed: int | str,
    phase: str,
    resource: str,
    value: float | None,
    unit: str,
    status: str,
    source: str,
    notes: str = "",
) -> dict[str, Any]:
    """Build one unit-preserving resource-ledger row."""
    return {
        "method": method,
        "seed": seed,
        "phase": phase,
        "resource": resource,
        "value": value,
        "unit": unit,
        "status": status,
        "source": source,
        "notes": notes,
    }


def compute_break_even_v2(
    training_gpu_seconds: dict[int, float],
    hero_generation_1000_central_gpu_seconds: dict[int, float],
    hero_generation_1000_conservative_gpu_seconds: dict[int, float],
    base64_generation_gpu_seconds_per_instance: dict[int, float],
) -> dict[str, Any]:
    """Compute like-for-like GPU break-even thresholds and sensitivity ranges."""
    seeds = sorted(training_gpu_seconds)
    inputs = (
        hero_generation_1000_central_gpu_seconds,
        hero_generation_1000_conservative_gpu_seconds,
        base64_generation_gpu_seconds_per_instance,
    )
    if any(set(seeds) != set(values) for values in inputs):
        raise ValueError("break-even inputs must contain identical seed sets")

    per_seed = []
    for seed in seeds:
        central_upfront = (
            training_gpu_seconds[seed] + hero_generation_1000_central_gpu_seconds[seed]
        )
        conservative_upfront = (
            training_gpu_seconds[seed]
            + hero_generation_1000_conservative_gpu_seconds[seed]
        )
        marginal = base64_generation_gpu_seconds_per_instance[seed]
        per_seed.append(
            {
                "seed": seed,
                "central_upfront_allocated_gpu_seconds": central_upfront,
                "conservative_upfront_allocated_gpu_seconds": conservative_upfront,
                "hero_generation_1000_central_gpu_seconds": hero_generation_1000_central_gpu_seconds[
                    seed
                ],
                "hero_generation_1000_conservative_gpu_seconds": hero_generation_1000_conservative_gpu_seconds[
                    seed
                ],
                "base64_generation_gpu_seconds_per_instance": marginal,
                "central_instances": ceiling_break_even(central_upfront, marginal),
                "conservative_instances": ceiling_break_even(
                    conservative_upfront, marginal
                ),
            }
        )

    campaign_central_upfront = sum(training_gpu_seconds.values()) + sum(
        hero_generation_1000_central_gpu_seconds.values()
    )
    campaign_conservative_upfront = sum(training_gpu_seconds.values()) + sum(
        hero_generation_1000_conservative_gpu_seconds.values()
    )
    campaign_sensitivity = [
        {
            "base_throughput_seed": seed,
            "central_instances": ceiling_break_even(
                campaign_central_upfront,
                base64_generation_gpu_seconds_per_instance[seed],
            ),
            "conservative_instances": ceiling_break_even(
                campaign_conservative_upfront,
                base64_generation_gpu_seconds_per_instance[seed],
            ),
        }
        for seed in seeds
    ]
    one_policy_central = [row["central_instances"] for row in per_seed]
    one_policy_conservative = [row["conservative_instances"] for row in per_seed]
    campaign_central = [row["central_instances"] for row in campaign_sensitivity]
    campaign_conservative = [
        row["conservative_instances"] for row in campaign_sensitivity
    ]
    return {
        "equation": "ceil((allocated training GPU-s + charged Hero artifact-generation GPU-s) / Base-64 generation GPU-s per deployment)",
        "per_seed_one_policy": per_seed,
        "one_policy_central_sensitivity_interval_instances": [
            min(one_policy_central),
            max(one_policy_central),
        ],
        "one_policy_conservative_sensitivity_interval_instances": [
            min(one_policy_conservative),
            max(one_policy_conservative),
        ],
        "three_seed_campaign_central_upfront_allocated_gpu_seconds": campaign_central_upfront,
        "three_seed_campaign_conservative_upfront_allocated_gpu_seconds": campaign_conservative_upfront,
        "three_seed_campaign_by_base_throughput": campaign_sensitivity,
        "three_seed_campaign_central_sensitivity_interval_instances": [
            min(campaign_central),
            max(campaign_central),
        ],
        "three_seed_campaign_conservative_sensitivity_interval_instances": [
            min(campaign_conservative),
            max(campaign_conservative),
        ],
        "central_generation_policy": "Seed 101 is measured over the full 1,000-output run; seeds 202/303 use that full-run timing as a reconstructed proxy because the 32-prompt calibration includes fixed startup overhead.",
        "conservative_generation_policy": "For seeds 202/303 only, the startup-inclusive 32-prompt Hero wall is scaled linearly to 1,000 outputs as a deliberately conservative upper sensitivity bound.",
        "exclusions": [
            "CPU execution seconds are not added to GPU seconds.",
            "Wall time is not treated as GPU utilization.",
            "No monetary conversion is made without a hardware price ratio.",
        ],
    }


def compute_break_even_full(
    training_gpu_seconds: dict[int, float],
    hero_generation_gpu_seconds_per_instance: dict[int, float],
    base64_generation_gpu_seconds_per_instance: dict[int, float],
    frozen_compile_generation_gpu_seconds: dict[int, float],
    generation_allocation_multipliers: dict[int, int] | None = None,
    frozen_compile_allocation_multipliers: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Compute ordinary and compile-once GPU break-even without mixing regimes."""
    seeds = sorted(training_gpu_seconds)
    inputs = (
        hero_generation_gpu_seconds_per_instance,
        base64_generation_gpu_seconds_per_instance,
        frozen_compile_generation_gpu_seconds,
    )
    if any(set(seeds) != set(values) for values in inputs):
        raise ValueError("break-even inputs must contain identical seed sets")

    per_seed = []
    for seed in seeds:
        training = float(training_gpu_seconds[seed])
        hero_generation = float(hero_generation_gpu_seconds_per_instance[seed])
        base_generation = float(base64_generation_gpu_seconds_per_instance[seed])
        compile_generation = float(frozen_compile_generation_gpu_seconds[seed])
        ordinary_savings = base_generation - hero_generation
        per_seed.append(
            {
                "seed": seed,
                "training_allocated_gpu_seconds": training,
                "hero_generation_gpu_seconds_per_instance": hero_generation,
                "base64_generation_gpu_seconds_per_instance": base_generation,
                "ordinary_hero_marginal_gpu_seconds_saved_per_instance": ordinary_savings,
                "ordinary_hero_break_even_instances": ceiling_break_even(
                    training, ordinary_savings
                ),
                "frozen_compile_generation_gpu_seconds": compile_generation,
                "frozen_compile_upfront_gpu_seconds": training + compile_generation,
                "frozen_hero_break_even_instances": ceiling_break_even(
                    training + compile_generation, base_generation
                ),
            }
        )

    campaign_training = sum(float(value) for value in training_gpu_seconds.values())
    campaign_compile = sum(
        float(value) for value in frozen_compile_generation_gpu_seconds.values()
    )
    campaign = []
    for seed in seeds:
        hero_generation = float(hero_generation_gpu_seconds_per_instance[seed])
        base_generation = float(base64_generation_gpu_seconds_per_instance[seed])
        campaign.append(
            {
                "deployment_throughput_seed": seed,
                "ordinary_hero_break_even_instances": ceiling_break_even(
                    campaign_training, base_generation - hero_generation
                ),
                "frozen_hero_break_even_instances": ceiling_break_even(
                    campaign_training + campaign_compile, base_generation
                ),
            }
        )

    def interval(field: str, rows: list[dict[str, Any]]) -> list[int]:
        values = [int(row[field]) for row in rows]
        return [min(values), max(values)]

    result = {
        "ordinary_hero_equation": "ceil(training allocated GPU-s / (Base-64 generation GPU-s per instance - Hero generation GPU-s per instance))",
        "frozen_hero_equation": "ceil((training allocated GPU-s + 64-completion validation synthesis GPU-s) / Base-64 generation GPU-s per instance)",
        "per_seed_one_policy": per_seed,
        "one_policy_ordinary_hero_interval_instances": interval(
            "ordinary_hero_break_even_instances", per_seed
        ),
        "one_policy_frozen_hero_interval_instances": interval(
            "frozen_hero_break_even_instances", per_seed
        ),
        "three_seed_campaign_training_allocated_gpu_seconds": campaign_training,
        "three_seed_campaign_compile_generation_gpu_seconds": campaign_compile,
        "three_seed_campaign_by_deployment_throughput": campaign,
        "three_seed_campaign_ordinary_hero_interval_instances": interval(
            "ordinary_hero_break_even_instances", campaign
        ),
        "three_seed_campaign_frozen_hero_interval_instances": interval(
            "frozen_hero_break_even_instances", campaign
        ),
        "unit_policy": [
            "CPU execution seconds are not added to GPU seconds.",
            "Slurm allocation and active one-GPU generation are reported separately.",
            "No monetary conversion is made without an explicit hardware price ratio.",
        ],
    }
    if generation_allocation_multipliers is not None:
        if frozen_compile_allocation_multipliers is None:
            raise ValueError("compile allocation multipliers are required")
        if set(seeds) != set(generation_allocation_multipliers) or set(seeds) != set(
            frozen_compile_allocation_multipliers
        ):
            raise ValueError("allocation multipliers must contain identical seed sets")
        allocated_per_seed = []
        for seed in seeds:
            generation_multiplier = int(generation_allocation_multipliers[seed])
            compile_multiplier = int(frozen_compile_allocation_multipliers[seed])
            if generation_multiplier < 1 or compile_multiplier < 1:
                raise ValueError("allocation multipliers must be positive")
            training = float(training_gpu_seconds[seed])
            hero = float(hero_generation_gpu_seconds_per_instance[seed])
            base = float(base64_generation_gpu_seconds_per_instance[seed])
            compile_generation = float(frozen_compile_generation_gpu_seconds[seed])
            allocated_per_seed.append(
                {
                    "seed": seed,
                    "generation_allocated_gpus": generation_multiplier,
                    "compile_allocated_gpus": compile_multiplier,
                    "ordinary_hero_break_even_instances": ceiling_break_even(
                        training, (base - hero) * generation_multiplier
                    ),
                    "frozen_hero_break_even_instances": ceiling_break_even(
                        training + compile_generation * compile_multiplier,
                        base * generation_multiplier,
                    ),
                }
            )
        allocated_campaign_compile = sum(
            float(frozen_compile_generation_gpu_seconds[seed])
            * int(frozen_compile_allocation_multipliers[seed])
            for seed in seeds
        )
        allocated_campaign = []
        for seed in seeds:
            multiplier = int(generation_allocation_multipliers[seed])
            hero = float(hero_generation_gpu_seconds_per_instance[seed])
            base = float(base64_generation_gpu_seconds_per_instance[seed])
            allocated_campaign.append(
                {
                    "deployment_throughput_seed": seed,
                    "generation_allocated_gpus": multiplier,
                    "ordinary_hero_break_even_instances": ceiling_break_even(
                        campaign_training, (base - hero) * multiplier
                    ),
                    "frozen_hero_break_even_instances": ceiling_break_even(
                        campaign_training + allocated_campaign_compile,
                        base * multiplier,
                    ),
                }
            )
        result["allocated_node_sensitivity"] = {
            "ordinary_hero_equation": "ceil(training allocated GPU-s / ((Base-64 active seconds - Hero active seconds) * allocated generation GPUs))",
            "frozen_hero_equation": "ceil((training allocated GPU-s + compile active seconds * allocated compile GPUs) / (Base-64 active seconds * allocated generation GPUs))",
            "per_seed_one_policy": allocated_per_seed,
            "one_policy_ordinary_hero_interval_instances": interval(
                "ordinary_hero_break_even_instances", allocated_per_seed
            ),
            "one_policy_frozen_hero_interval_instances": interval(
                "frozen_hero_break_even_instances", allocated_per_seed
            ),
            "three_seed_campaign_by_deployment_throughput": allocated_campaign,
            "three_seed_campaign_ordinary_hero_interval_instances": interval(
                "ordinary_hero_break_even_instances", allocated_campaign
            ),
            "three_seed_campaign_frozen_hero_interval_instances": interval(
                "frozen_hero_break_even_instances", allocated_campaign
            ),
            "classification": "scheduler-allocation sensitivity; active generation remains one GPU",
        }
    return result


def git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def portable_path(path: Path, repo_root: Path) -> str:
    """Prefer repository-relative provenance while supporting external artifacts."""
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def analyze(manifest: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    timings = {
        name: load_verified_json(repo_root, spec)
        for name, spec in manifest["timing_inputs"].items()
    }
    solver_rows = load_verified_json(repo_root, manifest["solver_cost_input"])
    training_jobs = manifest["training_scheduler_source"]["jobs"]

    training_rows = []
    for job in training_jobs:
        gpu_seconds = float(job["elapsed_seconds"] * job["allocated_gpus"])
        cpu_seconds = float(job["elapsed_seconds"] * job["allocated_cpus"])
        training_rows.append(
            {
                "seed": job["seed"],
                "job_id": job["job_id"],
                "state": job["state"],
                "elapsed_seconds": job["elapsed_seconds"],
                "allocated_gpus": job["allocated_gpus"],
                "allocated_cpus": job["allocated_cpus"],
                "gpu_hours": gpu_seconds / 3600.0,
                "coallocated_cpu_hours": cpu_seconds / 3600.0,
            }
        )

    hero_timing = timings["hero_seed101"]
    base_timing = timings["base64_seed101"]
    n_instances = int(base_timing["num_unique_instances"])
    hero_generation_seconds = float(hero_timing["generation_wall_clock_seconds"])
    hero_job_seconds = hero_generation_seconds + float(
        hero_timing["evaluation_wall_clock_seconds"]
    )
    base_generation_per_instance = (
        float(base_timing["generation_wall_clock_seconds"]) / n_instances
    )
    base_job_per_instance = (
        float(base_timing["generation_wall_clock_seconds"])
        + float(base_timing["evaluation_wall_clock_seconds"])
    ) / n_instances

    mean_training_gpu_seconds = mean(row["gpu_hours"] for row in training_rows) * 3600
    campaign_training_gpu_seconds = (
        sum(row["gpu_hours"] for row in training_rows) * 3600
    )

    hero_execution = method_mean(solver_rows, "Ours (Hero)", "Cost")
    frozen_execution = method_mean(solver_rows, "Frozen Hero", "Cost")
    base_execution = method_mean(solver_rows, "Base (Best-of-64)", "Cost")

    generation_only_upfront = mean_training_gpu_seconds + hero_generation_seconds
    allocated_job_upfront = mean_training_gpu_seconds + hero_job_seconds
    campaign_lower_bound_upfront = (
        campaign_training_gpu_seconds + hero_generation_seconds
    )

    frozen_eval_wall = [
        float(timings[f"frozen_hero_seed{seed}"]["evaluation_wall_clock_seconds"])
        for seed in (101, 202, 303)
    ]

    return {
        "protocol_id": manifest["protocol_id"],
        "analysis_commit": git_commit(repo_root),
        "training": {
            "per_seed": training_rows,
            "mean_gpu_hours_per_policy": mean(
                row["gpu_hours"] for row in training_rows
            ),
            "campaign_gpu_hours_three_seeds": sum(
                row["gpu_hours"] for row in training_rows
            ),
            "campaign_coallocated_cpu_hours": sum(
                row["coallocated_cpu_hours"] for row in training_rows
            ),
        },
        "measured_inference": {
            "instances": n_instances,
            "hero_1000_output_generation_gpu_wall_seconds": hero_generation_seconds,
            "hero_generation_and_evaluation_job_wall_seconds": hero_job_seconds,
            "base64_generation_gpu_wall_seconds_per_instance": base_generation_per_instance,
            "base64_generation_and_evaluation_job_wall_seconds_per_instance": base_job_per_instance,
            "hero_execution_seconds_per_instance": hero_execution,
            "frozen_hero_execution_seconds_per_instance": frozen_execution,
            "base64_candidate_execution_seconds_per_instance": base_execution,
            "post_generation_speedup_base64_over_hero": base_execution / hero_execution,
            "post_generation_speedup_base64_over_frozen_hero": base_execution
            / frozen_execution,
            "frozen_hero_full_harness_wall_seconds_mean": mean(frozen_eval_wall),
            "frozen_hero_full_harness_note": "The fixed-code harness reruns candidates and classical baselines; this wall time is not deployment solver latency.",
        },
        "break_even": {
            "single_policy_generation_only_instances": ceiling_break_even(
                generation_only_upfront, base_generation_per_instance
            ),
            "single_policy_allocated_job_wall_instances": ceiling_break_even(
                allocated_job_upfront, base_job_per_instance
            ),
            "three_seed_campaign_generation_only_lower_bound_instances": ceiling_break_even(
                campaign_lower_bound_upfront, base_generation_per_instance
            ),
            "generation_only_upfront_gpu_hours": generation_only_upfront / 3600.0,
            "allocated_job_upfront_gpu_hours": allocated_job_upfront / 3600.0,
            "cpu_execution_seconds_saved_per_instance_after_freeze": base_execution
            - frozen_execution,
            "interpretation": "The two single-policy thresholds are accounting bounds, not one universal number. They do not monetize or add CPU and GPU seconds.",
        },
        "shinka_evolve": manifest["shinka_evolve"],
        "limitations": [
            "Only seed 101 has a recovered raw Hero/Base-64 generation timing pair; extrapolation to other seeds is not presented as measured variance.",
            "Allocated training CPU-hours are not CPU utilization and are not folded into the GPU break-even thresholds.",
            "The full fixed-code harness wall time includes repeats and classical baselines, so per-instance candidate execution is the deployment metric.",
            "ShinkaEvolve synthesis wall time and token-level billing records are unavailable; its approximate API cost cannot be converted into the hardware break-even calculation.",
        ],
    }


def _rows_for_method_seed(
    rows: list[dict[str, Any]], method: str, seed: int
) -> dict[str, Any]:
    matches = [
        row for row in rows if row["Method"] == method and int(row["Seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {method} row for seed {seed}, got {len(matches)}"
        )
    return matches[0]


def _runtime_row(
    rows: list[dict[str, Any]], method: str, seed: int
) -> dict[str, Any] | None:
    matches = [
        row for row in rows if row["Method"] == method and int(row["Seed"]) == seed
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate runtime rows for {method}, seed {seed}")
    return matches[0] if matches else None


def _load_calibrations(
    calibration_root: Path, result_manifest: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    if result_manifest.get("protocol_id") != "N26-E3-v2":
        raise ValueError("invalid generation-calibration result protocol")
    if result_manifest.get("status") != "complete":
        raise ValueError("generation-calibration result is not complete")
    if result_manifest.get("efficacy_excluded") is not True:
        raise ValueError("generation-calibration result is not efficacy-excluded")

    for relative, expected in result_manifest["files_sha256"].items():
        path = calibration_root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise InputHashError(path, actual, expected)

    calibrations: dict[int, dict[str, Any]] = {}
    for seed in (202, 303):
        path = calibration_root / f"seed{seed}" / "calibration.json"
        if not path.exists():
            raise FileNotFoundError(f"missing frozen calibration: {path}")
        payload = json.loads(path.read_text())
        if (
            payload.get("protocol_id") != "N26-E3-v2"
            or int(payload.get("seed", -1)) != seed
        ):
            raise ValueError(f"invalid calibration identity: {path}")
        if payload.get("efficacy_excluded") is not True:
            raise ValueError(f"calibration is not marked efficacy-excluded: {path}")
        if int(payload.get("row_count", -1)) != 32:
            raise ValueError(f"unexpected calibration row count: {path}")
        scheduler = result_manifest["scheduler"][str(seed)]
        if str(payload.get("slurm_job_id")) != str(scheduler["job_id"]):
            raise ValueError(f"scheduler job mismatch: {path}")
        calibrations[seed] = payload
    return calibrations


def analyze_v2(
    protocol: dict[str, Any], repo_root: Path, calibration_root: Path
) -> dict[str, Any]:
    """Create the complete AC-facing phase ledger without mixing resource units."""
    if protocol.get("protocol_id") != "N26-E3-v2":
        raise ValueError("analyze_v2 requires the N26-E3-v2 protocol")

    inputs = {
        name: load_verified_json(repo_root, spec)
        for name, spec in protocol["input_files"].items()
        if spec["path"].endswith(".json")
    }
    source_manifest = inputs["source_manifest"]
    canonical_rows = inputs["canonical_summary"]
    runtime_rows = inputs["runtime_summary"]
    adaptive_rows = inputs["adaptive_repair"]["rows"]
    adaptive_protocol_id = str(inputs["adaptive_repair"]["protocol_id"])
    wandb_runs = {int(row["seed"]): row for row in inputs["wandb_training"]["runs"]}
    result_manifest_path = repo_root / protocol["generation_calibration_result"]
    calibration_root_source = portable_path(calibration_root, repo_root)
    calibration_result = json.loads(result_manifest_path.read_text())
    calibrations = _load_calibrations(calibration_root, calibration_result)
    training_jobs = {
        int(row["seed"]): row
        for row in source_manifest["training_scheduler_source"]["jobs"]
    }
    adaptive_by_seed = {int(row["seed"]): row for row in adaptive_rows}

    phase_rows: list[dict[str, Any]] = []
    training_gpu_seconds: dict[int, float] = {}
    hero_generation_1000_central: dict[int, float] = {}
    hero_generation_1000_conservative: dict[int, float] = {}
    base_generation_per_instance: dict[int, float] = {}

    historical_hero = _runtime_row(runtime_rows, "Ours (Hero)", 101)
    if historical_hero is None:
        raise ValueError("missing seed-101 full Hero generation timing")
    historical_hero_1000_seconds = float(historical_hero["GenerationWallClock"])

    for seed in (101, 202, 303):
        job = training_jobs[seed]
        allocated_gpu_seconds = float(job["elapsed_seconds"] * job["allocated_gpus"])
        training_gpu_seconds[seed] = allocated_gpu_seconds
        phase_rows.append(
            phase_row(
                "Hero",
                seed,
                "training",
                "allocated_gh200",
                allocated_gpu_seconds / 3600.0,
                "GH200-hours",
                "measured",
                f"Slurm sacct job {job['job_id']}",
                "Allocated capacity; not measured GPU utilization.",
            )
        )
        phase_rows.append(
            phase_row(
                "Hero",
                seed,
                "training",
                "wall_clock",
                float(wandb_runs[seed]["runtime_seconds"]),
                "seconds",
                "reconstructed",
                f"W&B run {wandb_runs[seed]['run_id']}",
                "Original run wall clock; reported separately from allocated GPU-hours.",
            )
        )
        phase_rows.append(
            phase_row(
                "Hero",
                seed,
                "checkpoint_production",
                "wall_clock",
                None,
                "seconds",
                "included_in_training",
                f"Slurm job {job['job_id']} / {job['selected_checkpoint']}",
                "The selected checkpoint was produced inside the charged training run; checkpoint-only wall time was not logged and is not added again.",
            )
        )
        phase_rows.append(
            phase_row(
                "Hero",
                seed,
                "development_execution",
                "cpu",
                None,
                "CPU-seconds",
                "unavailable",
                f"Slurm job {job['job_id']}",
                "Training-time reward executions are included in the allocated training job, but separate CPU utilization timing was not logged.",
            )
        )

        if seed == 101:
            base_runtime = _runtime_row(runtime_rows, "Base (Best-of-64)", seed)
            assert base_runtime is not None
            hero_central_seconds = historical_hero_1000_seconds
            hero_conservative_seconds = historical_hero_1000_seconds
            base_seconds_per_instance = (
                float(base_runtime["GenerationWallClock"]) / 1000.0
            )
            hero_status = "measured"
            base_status = "measured"
            hero_source = historical_hero["Path"]
            base_source = base_runtime["Path"]
        else:
            calibration = calibrations[seed]
            rows = int(calibration["row_count"])
            hero_central_seconds = historical_hero_1000_seconds
            hero_conservative_seconds = (
                float(calibration["hero_generation_wall_seconds"]) / rows * 1000.0
            )
            base_seconds_per_instance = (
                float(calibration["base64_generation_wall_seconds"]) / rows
            )
            hero_status = "reconstructed_proxy"
            base_status = "calibrated"
            hero_source = str(
                Path(calibration_root_source) / f"seed{seed}" / "calibration.json"
            )
            base_source = hero_source

        hero_generation_1000_central[seed] = hero_central_seconds
        hero_generation_1000_conservative[seed] = hero_conservative_seconds
        base_generation_per_instance[seed] = base_seconds_per_instance
        phase_rows.extend(
            [
                phase_row(
                    "Hero",
                    seed,
                    "generation",
                    "one_gpu",
                    hero_central_seconds,
                    "GPU-seconds per 1,000 outputs",
                    hero_status,
                    hero_source,
                    "Seed 101 is a measured full run. Seeds 202/303 use that full-run timing as a central proxy; their startup-inclusive 32-prompt calibrations are reported separately and efficacy-excluded.",
                ),
                phase_row(
                    "Frozen Hero",
                    seed,
                    "selection",
                    "wall_clock",
                    None,
                    "seconds",
                    "unavailable",
                    "evaluation/sds/extract_frozen_solver.py",
                    "Deterministic first-valid-row extraction from the charged Hero artifact; historical extraction wall time was not logged.",
                ),
                phase_row(
                    "Base Best-of-64",
                    seed,
                    "generation",
                    "one_gpu",
                    base_seconds_per_instance,
                    "GPU-seconds per deployment instance",
                    base_status,
                    base_source,
                    "One deployment instance contains 64 completions; seeds 202/303 are startup-inclusive contemporary calibration measurements.",
                ),
            ]
        )

        if seed != 101:
            calibration = calibrations[seed]
            scheduler = calibration_result["scheduler"][str(seed)]
            phase_rows.extend(
                [
                    phase_row(
                        "Hero",
                        seed,
                        "generation_calibration",
                        "one_gpu",
                        float(calibration["hero_generation_wall_seconds"]),
                        "GPU-seconds per 32 outputs",
                        "calibrated",
                        hero_source,
                        "Startup-inclusive batch wall; not linearly extrapolated for the central estimate.",
                    ),
                    phase_row(
                        "Hero",
                        seed,
                        "generation_sensitivity_upper",
                        "one_gpu",
                        hero_conservative_seconds,
                        "GPU-seconds per 1,000 outputs",
                        "calibrated_upper_bound",
                        hero_source,
                        "Linear scaling of the startup-inclusive 32-prompt wall; deliberately conservative, not a measured full run.",
                    ),
                    phase_row(
                        "Calibration only",
                        seed,
                        "generation_calibration",
                        "allocated_gh200",
                        float(scheduler["elapsed_seconds"])
                        * int(scheduler["allocated_gpus"])
                        / 3600.0,
                        "GH200-hours",
                        "measured",
                        f"Slurm sacct job {scheduler['job_id']}",
                        "Scheduler allocated four GH200s; tensor-parallel generation used one active GPU. Efficacy excluded.",
                    ),
                    phase_row(
                        "Calibration only",
                        seed,
                        "generation_calibration",
                        "active_generation_gpus",
                        float(
                            calibration_result["measurement_scope"][
                                "active_generation_gpus"
                            ]
                        ),
                        "GH200 count",
                        "reconstructed",
                        portable_path(result_manifest_path, repo_root),
                        "Active GPU count is reported separately from the four-GPU scheduler allocation.",
                    ),
                ]
            )

        for method, canonical_name, phase in (
            ("Hero", "Ours (Hero)", "final_execution"),
            ("Frozen Hero", "Frozen Hero", "final_execution"),
            ("Base Best-of-64", "Base (Best-of-64)", "selection"),
            (
                "Adaptive Repair",
                "Adaptive Repair (64 completions)",
                "final_execution",
            ),
            ("ShinkaEvolve", "ShinkaEvolve", "final_execution"),
        ):
            canonical = _rows_for_method_seed(canonical_rows, canonical_name, seed)
            is_base_selection = method == "Base Best-of-64"
            phase_rows.append(
                phase_row(
                    method,
                    seed,
                    phase,
                    "cpu",
                    float(canonical["Cost"]),
                    "CPU-seconds per deployment instance",
                    "measured",
                    "experiments/neurips2026/cost_inputs_v2/canonical_summary_by_seed.json",
                    (
                        "Cumulative execution and selection across the 64 candidates; generation is separate."
                        if is_base_selection
                        else "Selected-program execution only; generation is separate."
                    ),
                )
            )
            if is_base_selection:
                phase_rows.append(
                    phase_row(
                        method,
                        seed,
                        "final_execution",
                        "cpu",
                        None,
                        "CPU-seconds per deployment instance",
                        "included_in_selection",
                        "experiments/neurips2026/cost_inputs_v2/canonical_summary_by_seed.json",
                        "Best-of-64 returns the selected candidate result during its per-instance selection run; no second execution is charged.",
                    )
                )

        adaptive = adaptive_by_seed[seed]
        adaptive_job = int(adaptive["slurm_job_id"])
        adaptive_elapsed = float(
            protocol["adaptive_repair_allocations"][str(seed)]["elapsed_seconds"]
        )
        adaptive_gpus = int(
            protocol["adaptive_repair_allocations"][str(seed)]["allocated_gpus"]
        )
        phase_rows.extend(
            [
                phase_row(
                    "Adaptive Repair",
                    seed,
                    "generation",
                    "one_gpu",
                    float(adaptive["synthesis_wall_seconds"]),
                    "GPU-seconds",
                    "measured",
                    f"{adaptive_protocol_id} job {adaptive_job}",
                    "In-process vLLM synthesis wall for exactly 64 completions.",
                ),
                phase_row(
                    "Adaptive Repair",
                    seed,
                    "development_execution",
                    "cpu",
                    float(adaptive["development_execution_cpu_seconds"]),
                    "CPU-seconds",
                    "measured",
                    f"{adaptive_protocol_id} job {adaptive_job}",
                    f"{adaptive['development_execution_count']} frozen-development executions.",
                ),
                phase_row(
                    "Adaptive Repair",
                    seed,
                    "selection",
                    "wall_clock",
                    float(adaptive["end_to_end_job_step_seconds"]),
                    "seconds",
                    "measured",
                    f"{adaptive_protocol_id} job {adaptive_job}",
                    "End-to-end controller step wall includes generation, execution feedback, repair orchestration, and selection; do not add to component rows.",
                ),
                phase_row(
                    "Adaptive Repair",
                    seed,
                    "final_execution",
                    "wall_clock",
                    float(adaptive["final_test_harness_wall_seconds"]),
                    "seconds per 1,000-row test harness",
                    "measured",
                    f"{adaptive_protocol_id} job {adaptive_job}",
                    "Harness wall is reported separately from selected-program CPU execution.",
                ),
                phase_row(
                    "Adaptive Repair",
                    seed,
                    "generation_and_selection_allocation",
                    "allocated_gh200",
                    adaptive_elapsed * adaptive_gpus / 3600.0,
                    "GH200-hours",
                    "measured",
                    f"Slurm sacct job {adaptive_job}",
                    "Four-GPU job allocation; one GPU hosted vLLM. Reported separately from one-GPU synthesis wall.",
                ),
            ]
        )

        for phase in ("generation", "development_execution", "selection"):
            phase_rows.append(
                phase_row(
                    "ShinkaEvolve",
                    seed,
                    phase,
                    "external_api_or_cpu",
                    None,
                    "unavailable",
                    "unavailable",
                    "docs/technical-reports/SHINKA_TRACE_GENERATION_REPORT.md",
                    "Token billing and synthesis/selection timing cannot be recovered and are not imputed.",
                )
            )

    break_even = compute_break_even_v2(
        training_gpu_seconds,
        hero_generation_1000_central,
        hero_generation_1000_conservative,
        base_generation_per_instance,
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "analysis_commit": git_commit(repo_root),
        "calibration_root": calibration_root_source,
        "phase_rows": phase_rows,
        "break_even": break_even,
        "provenance": {
            "input_hashes": {
                name: spec["sha256"] for name, spec in protocol["input_files"].items()
            },
            "calibration_sha256": {
                str(seed): sha256_file(
                    calibration_root / f"seed{seed}" / "calibration.json"
                )
                for seed in (202, 303)
            },
            "calibration_result_manifest": {
                "path": portable_path(result_manifest_path, repo_root),
                "sha256": sha256_file(result_manifest_path),
                "verified_files": calibration_result["files_sha256"],
            },
            "classification": {
                "measured": "direct timing or scheduler record from the original run",
                "reconstructed": "metadata recovered from an original system of record",
                "calibrated": "contemporary frozen-prompt throughput measurement, efficacy-excluded",
                "calibrated_upper_bound": "linear scaling of a startup-inclusive short calibration, used only as a conservative sensitivity bound",
                "reconstructed_proxy": "an original full-run measurement reused centrally where only a startup-inclusive short calibration is available",
                "included_in_training": "charged by the enclosing training allocation and not added again",
                "included_in_selection": "charged by the enclosing candidate-selection measurement and not added again",
                "unavailable": "not recovered and not imputed",
            },
            "unit_policy": protocol["units"]["combination_policy"],
        },
    }


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "cost_break_even.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    training_rows = result["training"]["per_seed"]
    with (output_dir / "training_allocations.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(training_rows[0]))
        writer.writeheader()
        writer.writerows(training_rows)

    measured = result["measured_inference"]
    break_even = result["break_even"]
    lines = [
        "# N26-E3 End-to-End Cost and Break-Even",
        "",
        f"Analysis commit: `{result['analysis_commit']}`",
        "",
        "## Measured resource accounting",
        "",
        f"- One Hero policy used {result['training']['mean_gpu_hours_per_policy']:.2f} allocated GH200-hours on average (12 GPUs for about four hours).",
        f"- The complete three-seed training campaign used {result['training']['campaign_gpu_hours_three_seeds']:.2f} allocated GH200-hours.",
        f"- The recovered Hero generation run produced 1,000 programs in {measured['hero_1000_output_generation_gpu_wall_seconds']:.1f} one-GPU seconds.",
        f"- Base-64 generation used {measured['base64_generation_gpu_wall_seconds_per_instance']:.3f} one-GPU seconds per deployment instance.",
        f"- Candidate execution used {measured['base64_candidate_execution_seconds_per_instance']:.3f} CPU seconds per Base-64 instance versus {measured['frozen_hero_execution_seconds_per_instance']:.3f} for Frozen Hero.",
        f"- The submitted post-generation comparison is {measured['post_generation_speedup_base64_over_hero']:.1f}x for Base-64 versus ordinary Hero; compile-once Frozen Hero is {measured['post_generation_speedup_base64_over_frozen_hero']:.1f}x faster than Base-64 candidate execution.",
        "",
        "## Break-even",
        "",
        f"- Generation-only accounting: {break_even['single_policy_generation_only_instances']:,} deployment instances for one trained policy.",
        f"- Allocated one-GPU job-wall accounting: {break_even['single_policy_allocated_job_wall_instances']:,} deployment instances for one trained policy.",
        f"- If all three training seeds are charged to one deployed policy, the generation-only lower bound is {break_even['three_seed_campaign_generation_only_lower_bound_instances']:,} instances.",
        "",
        "These are transparent accounting bounds. CPU and GPU seconds are not added or monetized without an explicit hardware price ratio.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
    ]
    (output_dir / "cost_break_even.md").write_text("\n".join(lines))


def write_report_v2(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "cost_break_even.json").write_text(
        json.dumps(result["break_even"], indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(result["provenance"], indent=2, sort_keys=True) + "\n"
    )
    rows = result["phase_rows"]
    with (output_dir / "per_seed_costs.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    one_low, one_high = result["break_even"][
        "one_policy_central_sensitivity_interval_instances"
    ]
    one_upper_low, one_upper_high = result["break_even"][
        "one_policy_conservative_sensitivity_interval_instances"
    ]
    campaign_low, campaign_high = result["break_even"][
        "three_seed_campaign_central_sensitivity_interval_instances"
    ]
    campaign_upper_low, campaign_upper_high = result["break_even"][
        "three_seed_campaign_conservative_sensitivity_interval_instances"
    ]
    lines = [
        "# N26-E3-v2 Complete Cost and Break-Even Accounting",
        "",
        f"Analysis commit: `{result['analysis_commit']}`",
        "",
        "## Break-even",
        "",
        f"- Central one-policy GPU accounting ranges from {one_low:,} to {one_high:,} repeated deployments across the three measured/calibrated Base-64 throughput seeds.",
        f"- Conservatively scaling the startup-inclusive short Hero calibrations gives {one_upper_low:,} to {one_upper_high:,} deployments.",
        f"- Charging the complete three-seed research campaign gives a central range of {campaign_low:,} to {campaign_high:,} deployments and a conservative range of {campaign_upper_low:,} to {campaign_upper_high:,}.",
        "- The equation charges allocated training plus the full 1,000-output Hero artifact used for deterministic solver extraction, then compares this only with one-GPU Base-64 generation per deployment.",
        "- CPU execution, GPU allocation, one-GPU generation, and wall time remain separate resource classes.",
        "",
        "## Evidence status",
        "",
        "- Original Slurm, W&B, generation, evaluation, and adaptive-repair records are measured or reconstructed per row.",
        "- Seeds 202/303 Base-64 throughput is a labelled contemporary calibration on 32 frozen prompts. Their Hero batch timings are reported directly, while the measured seed-101 full run is the central 1,000-output proxy and linear scaling is only a conservative upper bound.",
        "- The calibration jobs allocated four GH200s while one GPU performed tensor-parallel generation; both quantities are exposed and the generated programs are excluded from efficacy results.",
        "- Historical Frozen Hero extraction time and ShinkaEvolve synthesis, selection, and token billing are unavailable and are not imputed.",
        "",
        "The full phase-by-seed ledger is `per_seed_costs.csv`; exact hashes and evidence definitions are in `provenance.json`.",
        "",
    ]
    (output_dir / "cost_break_even.md").write_text("\n".join(lines))


def _verify_result_files(root: Path, result_manifest: dict[str, Any]) -> None:
    for relative, expected in result_manifest["files_sha256"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing registered evidence: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise InputHashError(path, actual, expected)


def _validate_scheduler_records(
    result_manifest: dict[str, Any], expected_seeds: set[int]
) -> dict[str, dict[str, Any]]:
    scheduler = result_manifest.get("scheduler")
    if not isinstance(scheduler, dict):
        raise ValueError("result manifest is missing scheduler records")
    actual_seeds = {int(seed) for seed in scheduler}
    if actual_seeds != expected_seeds:
        raise ValueError(
            f"scheduler seeds must be {sorted(expected_seeds)}, got {sorted(actual_seeds)}"
        )
    for seed_text, record in scheduler.items():
        seed = int(seed_text)
        if record.get("state") != "COMPLETED":
            raise ValueError(f"scheduler job for seed {seed} is not COMPLETED")
        if not str(record.get("job_id", "")).strip():
            raise ValueError(f"scheduler job for seed {seed} has no job ID")
        if float(record.get("elapsed_seconds", 0)) <= 0:
            raise ValueError(f"scheduler job for seed {seed} has no elapsed time")
        if int(record.get("allocated_gpus", 0)) <= 0:
            raise ValueError(f"scheduler job for seed {seed} has no GPU allocation")
    return scheduler


def _load_full_calibrations(
    root: Path, result_manifest: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    if result_manifest.get("protocol_id") != "N26-E3-v3-full-throughput-result":
        raise ValueError("invalid full-throughput result protocol")
    if result_manifest.get("status") != "complete":
        raise ValueError("full-throughput calibration is not complete")
    if result_manifest.get("efficacy_excluded") is not True:
        raise ValueError("full-throughput generations are not efficacy-excluded")
    scheduler_records = _validate_scheduler_records(result_manifest, {202, 303})
    _verify_result_files(root, result_manifest)

    expected_rows = int(result_manifest["expected_row_count"])
    if expected_rows != 1000:
        raise ValueError(
            f"full-throughput calibration must contain 1000 rows, got {expected_rows}"
        )
    calibrations: dict[int, dict[str, Any]] = {}
    for seed_text, scheduler in scheduler_records.items():
        seed = int(seed_text)
        path = root / f"seed{seed}/calibration.json"
        relative = str(path.relative_to(root))
        if relative not in result_manifest["files_sha256"]:
            raise ValueError(f"unregistered full calibration: {relative}")
        payload = json.loads(path.read_text())
        expected = {
            "protocol_id": "N26-E3-v2",
            "seed": seed,
            "row_count": expected_rows,
            "hero_completion_count": expected_rows,
            "base_completion_count": expected_rows * 64,
            "calibration_scope": "full-test-throughput",
            "efficacy_excluded": True,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"full calibration {path} has {key}={payload.get(key)!r}"
                )
        if str(payload.get("slurm_job_id")) != str(scheduler["job_id"]):
            raise ValueError(f"scheduler job mismatch: {path}")
        if int(payload.get("active_generation_gpus", -1)) != 1:
            raise ValueError(f"unexpected active GPU count: {path}")
        for field in ("hero_generation_wall_seconds", "base64_generation_wall_seconds"):
            if float(payload.get(field, 0)) <= 0:
                raise ValueError(f"full calibration {path} has nonpositive {field}")
        calibrations[seed] = payload
    return calibrations


def _load_frozen_compile_costs(
    root: Path,
    result_manifest: dict[str, Any],
    *,
    expected_registration_protocol_id: str,
    expected_timing_protocol_id: str,
) -> dict[int, dict[str, Any]]:
    if result_manifest.get("protocol_id") != expected_registration_protocol_id:
        raise ValueError("invalid Frozen Hero cost registration")
    if result_manifest.get("status") != "complete":
        raise ValueError("Frozen Hero cost registration is not complete")
    if result_manifest.get("timing_protocol_id") != expected_timing_protocol_id:
        raise ValueError(
            "Frozen Hero result does not declare the expected timing protocol"
        )
    scheduler_records = _validate_scheduler_records(result_manifest, {101, 202, 303})
    _verify_result_files(root, result_manifest)

    result: dict[int, dict[str, Any]] = {}
    for seed_text, scheduler in scheduler_records.items():
        seed = int(seed_text)
        seed_root = root / f"seed{seed}"
        required_paths = {
            "timing": seed_root / "timing_accounting.json",
            "references": seed_root / "development_references.jsonl",
            "evaluations": seed_root
            / "selection/candidate_development_evaluations.jsonl",
            "test_metrics": seed_root
            / f"test/fixed-code/frozen-hero/seed{seed}/metrics_final.csv",
        }
        for path in required_paths.values():
            relative = str(path.relative_to(root))
            if relative not in result_manifest["files_sha256"]:
                raise ValueError(f"unregistered Frozen Hero evidence: {relative}")

        timing = json.loads(required_paths["timing"].read_text())
        if timing.get("protocol_id") != expected_timing_protocol_id:
            raise ValueError(f"invalid Frozen Hero timing protocol for seed {seed}")
        if int(timing.get("synthesis_completions", -1)) != 64:
            raise ValueError(
                f"Frozen Hero seed {seed} did not synthesize 64 completions"
            )
        if int(timing.get("active_generation_gpus", -1)) != 1:
            raise ValueError(f"Frozen Hero seed {seed} active GPU count is not one")
        if str(timing.get("slurm_job_id")) != str(scheduler["job_id"]):
            raise ValueError(f"Frozen Hero scheduler mismatch for seed {seed}")
        if timing.get("allocated_node_gpus") is not None and int(
            timing["allocated_node_gpus"]
        ) != int(scheduler["allocated_gpus"]):
            raise ValueError(f"Frozen Hero GPU allocation mismatch for seed {seed}")
        if timing.get("test_outcomes_accessed_after_freeze") is not True:
            raise ValueError(
                f"Frozen Hero seed {seed} lacks freeze-before-test evidence"
            )
        for field in (
            "generation_gpu_wall_seconds",
            "development_selection_wall_seconds",
            "test_evaluation_wall_seconds",
        ):
            if float(timing.get(field, 0)) <= 0:
                raise ValueError(f"Frozen Hero seed {seed} has nonpositive {field}")

        references = [
            json.loads(line)
            for line in required_paths["references"].read_text().splitlines()
            if line
        ]
        evaluations = [
            json.loads(line)
            for line in required_paths["evaluations"].read_text().splitlines()
            if line
        ]
        with required_paths["test_metrics"].open(newline="") as stream:
            test_metrics = list(csv.DictReader(stream))
        reference_uuids = [str(row["uuid"]) for row in references]
        if len(reference_uuids) != 40 or len(set(reference_uuids)) != 40:
            raise ValueError(f"Frozen Hero seed {seed} must have 40 unique references")
        evaluations_by_code: dict[str, list[dict[str, Any]]] = {}
        for row in evaluations:
            evaluations_by_code.setdefault(str(row["code_sha256"]), []).append(row)
        if not evaluations_by_code or len(evaluations_by_code) > 64:
            raise ValueError(f"Frozen Hero seed {seed} has invalid unique-code count")
        expected_uuid_set = set(reference_uuids)
        for code_hash, rows in evaluations_by_code.items():
            evaluation_uuids = [str(row["uuid"]) for row in rows]
            if (
                len(evaluation_uuids) != 40
                or set(evaluation_uuids) != expected_uuid_set
            ):
                raise ValueError(
                    f"Frozen Hero seed {seed} code {code_hash} lacks a complete development block"
                )
        if (
            len(test_metrics) != 1000
            or len({str(row["uuid"]) for row in test_metrics}) != 1000
        ):
            raise ValueError(f"Frozen Hero seed {seed} lacks 1,000 unique test rows")
        test_latencies = [float(row["execution_time"]) for row in test_metrics]
        if any(value < 0 for value in test_latencies):
            raise ValueError(f"Frozen Hero seed {seed} has negative test latency")
        result[seed] = {
            "generation_gpu_seconds": float(timing["generation_gpu_wall_seconds"]),
            "development_reference_cpu_seconds": sum(
                float(row["wall_time_sec"]) for row in references
            ),
            "development_execution_cpu_seconds": sum(
                float(row["execution_time"]) for row in evaluations
            ),
            "development_selection_wall_seconds": float(
                timing["development_selection_wall_seconds"]
            ),
            "test_harness_wall_seconds": float(timing["test_evaluation_wall_seconds"]),
            "test_best_of_three_latency_mean_seconds": mean(test_latencies),
            "test_best_of_three_latency_sum_seconds": sum(test_latencies),
            "test_rows": len(test_metrics),
            "candidate_evaluations": len(evaluations),
            "reference_rows": len(references),
            "scheduler": scheduler,
        }
    return result


def _load_adaptive_controller_costs(
    root: Path,
    result_manifest: dict[str, Any],
    *,
    expected_registration_protocol_id: str,
    expected_timing_protocol_id: str,
) -> dict[int, dict[str, Any]]:
    """Load hash-registered controller, development, and final-test costs."""
    if result_manifest.get("protocol_id") != expected_registration_protocol_id:
        raise ValueError("invalid Adaptive Repair cost registration")
    if result_manifest.get("status") != "complete":
        raise ValueError("Adaptive Repair cost registration is not complete")
    if result_manifest.get("timing_protocol_id") != expected_timing_protocol_id:
        raise ValueError(
            "Adaptive Repair result does not declare the expected timing protocol"
        )
    scheduler_records = _validate_scheduler_records(result_manifest, {101, 202, 303})
    _verify_result_files(root, result_manifest)

    result: dict[int, dict[str, Any]] = {}
    for seed_text, scheduler in scheduler_records.items():
        seed = int(seed_text)
        seed_root = root / f"seed{seed}"
        required_paths = {
            "timing": seed_root / "timing_summary.json",
            "references": seed_root / "development_references.jsonl",
            "run_manifest": seed_root / "synthesis/run_manifest.json",
            "evaluations": seed_root / "synthesis/development_evaluations.jsonl",
            "test_metrics": seed_root / "test-evaluation/metrics_final.csv",
        }
        for path in required_paths.values():
            relative = str(path.relative_to(root))
            if relative not in result_manifest["files_sha256"]:
                raise ValueError(f"unregistered Adaptive Repair evidence: {relative}")

        timing = json.loads(required_paths["timing"].read_text())
        run_manifest = json.loads(required_paths["run_manifest"].read_text())
        expected = {
            "protocol_id": expected_timing_protocol_id,
            "seed": seed,
            "completion_count": 64,
            "development_execution_count": 2560,
            "test_rows": 1000,
            "test_execution_repeats": 3,
            "test_classical_baselines": "none",
        }
        for key, value in expected.items():
            if timing.get(key) != value:
                raise ValueError(f"Adaptive Repair seed {seed} timing {key} drifted")
        if str(run_manifest.get("slurm_job_id")) != str(scheduler["job_id"]):
            raise ValueError(f"Adaptive Repair scheduler mismatch for seed {seed}")
        if run_manifest.get("protocol_id") != expected_timing_protocol_id:
            raise ValueError(f"Adaptive Repair seed {seed} run protocol drifted")
        if int(run_manifest.get("completion_count", -1)) != 64:
            raise ValueError(
                f"Adaptive Repair seed {seed} did not generate 64 completions"
            )
        for field in (
            "synthesis_wall_clock_seconds",
            "synthesis_phase_wall_clock_seconds",
            "final_test_harness_wall_seconds",
            "end_to_end_job_step_seconds",
        ):
            if float(timing.get(field, 0)) <= 0:
                raise ValueError(f"Adaptive Repair seed {seed} has nonpositive {field}")

        references = [
            json.loads(line)
            for line in required_paths["references"].read_text().splitlines()
            if line
        ]
        evaluations = [
            json.loads(line)
            for line in required_paths["evaluations"].read_text().splitlines()
            if line
        ]
        if len(references) != 40 or len({str(row["uuid"]) for row in references}) != 40:
            raise ValueError(f"Adaptive Repair seed {seed} must have 40 references")
        by_completion: dict[int, set[str]] = {}
        for row in evaluations:
            by_completion.setdefault(int(row["completion_index"]), set()).add(
                str(row["uuid"])
            )
        expected_uuids = {str(row["uuid"]) for row in references}
        if set(by_completion) != set(range(64)) or any(
            uuids != expected_uuids for uuids in by_completion.values()
        ):
            raise ValueError(
                f"Adaptive Repair seed {seed} lacks a complete development block"
            )
        with required_paths["test_metrics"].open(newline="") as stream:
            test_metrics = list(csv.DictReader(stream))
        if (
            len(test_metrics) != 1000
            or len({str(row["uuid"]) for row in test_metrics}) != 1000
        ):
            raise ValueError(
                f"Adaptive Repair seed {seed} lacks 1,000 unique test rows"
            )
        test_latencies = [float(row["execution_time"]) for row in test_metrics]
        if any(value < 0 for value in test_latencies):
            raise ValueError(f"Adaptive Repair seed {seed} has negative test latency")
        result[seed] = {
            "controller_wall_seconds": float(
                timing["synthesis_phase_wall_clock_seconds"]
            ),
            "model_controller_wall_seconds": float(
                timing["synthesis_wall_clock_seconds"]
            ),
            "development_reference_cpu_seconds": sum(
                float(row["wall_time_sec"]) for row in references
            ),
            "development_execution_cpu_seconds": sum(
                float(row["execution_time"]) for row in evaluations
            ),
            "test_harness_wall_seconds": float(
                timing["final_test_harness_wall_seconds"]
            ),
            "test_best_of_three_latency_mean_seconds": mean(test_latencies),
            "test_best_of_three_latency_sum_seconds": sum(test_latencies),
            "test_rows": len(test_metrics),
            "scheduler": scheduler,
        }
    return result


def analyze_v3(
    protocol: dict[str, Any],
    repo_root: Path,
    calibration_root: Path,
    adaptive_repair_root: Path,
    frozen_hero_root: Path,
) -> dict[str, Any]:
    """Build the final measured cost ledger and corrected break-even regimes."""
    if protocol.get("protocol_id") != "N26-E3-v3":
        raise ValueError("analyze_v3 requires the N26-E3-v3 protocol")
    inputs = {
        name: load_verified_json(repo_root, spec)
        for name, spec in protocol["input_files"].items()
    }
    source_manifest = inputs["source_manifest"]
    corrected_rows = inputs["corrected_summary"]
    wandb_runs = {int(row["seed"]): row for row in inputs["wandb_training"]["runs"]}
    training_jobs = {
        int(row["seed"]): row
        for row in source_manifest["training_scheduler_source"]["jobs"]
    }
    hero_seed101_timing = load_verified_json(
        repo_root, source_manifest["timing_inputs"]["hero_seed101"]
    )
    base_seed101_timing = load_verified_json(
        repo_root, source_manifest["timing_inputs"]["base64_seed101"]
    )

    full_manifest_spec = protocol["full_generation_result"]
    full_manifest_path = repo_root / full_manifest_spec["path"]
    full_manifest = load_verified_json(repo_root, full_manifest_spec)
    calibrations = _load_full_calibrations(calibration_root, full_manifest)
    frozen_manifest_spec = protocol["frozen_hero_cost_result"]
    frozen_manifest_path = repo_root / frozen_manifest_spec["path"]
    frozen_manifest = load_verified_json(repo_root, frozen_manifest_spec)
    frozen_costs = _load_frozen_compile_costs(
        frozen_hero_root,
        frozen_manifest,
        expected_registration_protocol_id=protocol[
            "frozen_hero_cost_registration_protocol_id"
        ],
        expected_timing_protocol_id=protocol["frozen_hero_timing_protocol_id"],
    )
    adaptive_manifest_spec = protocol["adaptive_repair_cost_result"]
    adaptive_manifest_path = repo_root / adaptive_manifest_spec["path"]
    adaptive_manifest = load_verified_json(repo_root, adaptive_manifest_spec)
    adaptive_costs = _load_adaptive_controller_costs(
        adaptive_repair_root,
        adaptive_manifest,
        expected_registration_protocol_id=protocol[
            "adaptive_repair_cost_registration_protocol_id"
        ],
        expected_timing_protocol_id=protocol["adaptive_repair_timing_protocol_id"],
    )

    phase_rows: list[dict[str, Any]] = []
    training_gpu_seconds: dict[int, float] = {}
    hero_generation_per_instance: dict[int, float] = {}
    base_generation_per_instance: dict[int, float] = {}
    frozen_compile_generation: dict[int, float] = {}
    generation_allocation_multipliers: dict[int, int] = {}
    frozen_compile_allocation_multipliers: dict[int, int] = {}

    if (
        int(hero_seed101_timing.get("num_records_evaluated", -1)) != 1000
        or int(base_seed101_timing.get("num_records_evaluated", -1)) != 64000
        or int(base_seed101_timing.get("num_unique_instances", -1)) != 1000
    ):
        raise ValueError("seed-101 generation evidence is not the complete workload")

    for seed in (101, 202, 303):
        training_job = training_jobs[seed]
        allocated_training = float(
            training_job["elapsed_seconds"] * training_job["allocated_gpus"]
        )
        training_gpu_seconds[seed] = allocated_training
        phase_rows.extend(
            [
                phase_row(
                    "Hero",
                    seed,
                    "training",
                    "allocated_gh200",
                    allocated_training / 3600.0,
                    "GH200-hours",
                    "measured",
                    f"Slurm sacct job {training_job['job_id']}",
                    "Allocated capacity, not GPU utilization.",
                ),
                phase_row(
                    "Hero",
                    seed,
                    "training",
                    "wall_clock",
                    float(wandb_runs[seed]["runtime_seconds"]),
                    "seconds",
                    "reconstructed",
                    f"W&B run {wandb_runs[seed]['run_id']}",
                ),
                phase_row(
                    "Hero",
                    seed,
                    "checkpoint_production",
                    "wall_clock",
                    None,
                    "seconds",
                    "included_in_training",
                    f"Slurm job {training_job['job_id']}",
                    "Checkpoint production occurred inside the charged training run.",
                ),
            ]
        )

        if seed == 101:
            hero_full_seconds = float(
                hero_seed101_timing["generation_wall_clock_seconds"]
            )
            base_full_seconds = float(
                base_seed101_timing["generation_wall_clock_seconds"]
            )
            generation_source = source_manifest["timing_inputs"]["hero_seed101"]["path"]
            base_source = source_manifest["timing_inputs"]["base64_seed101"]["path"]
            generation_allocation_multipliers[seed] = int(
                protocol["deployment_allocation_sensitivity"][
                    "seed101_generation_allocated_gpus"
                ]
            )
        else:
            calibration = calibrations[seed]
            hero_full_seconds = float(calibration["hero_generation_wall_seconds"])
            base_full_seconds = float(calibration["base64_generation_wall_seconds"])
            generation_source = str(calibration_root / f"seed{seed}/calibration.json")
            base_source = generation_source
            generation_scheduler = full_manifest["scheduler"][str(seed)]
            generation_allocation_multipliers[seed] = int(
                generation_scheduler["allocated_gpus"]
            )
            phase_rows.append(
                phase_row(
                    "Throughput calibration",
                    seed,
                    "full_generation_calibration_job",
                    "allocated_gh200",
                    float(generation_scheduler["elapsed_seconds"])
                    * int(generation_scheduler["allocated_gpus"])
                    / 3600.0,
                    "GH200-hours",
                    "measured",
                    f"Slurm sacct job {generation_scheduler['job_id']}",
                    "Efficacy-excluded measurement overhead; "
                    f"one GPU was active and {generation_scheduler['allocated_gpus']} "
                    "GPU(s) were allocated.",
                )
            )
        hero_generation_per_instance[seed] = hero_full_seconds / 1000.0
        base_generation_per_instance[seed] = base_full_seconds / 1000.0
        phase_rows.extend(
            [
                phase_row(
                    "Hero",
                    seed,
                    "generation",
                    "one_gpu",
                    hero_full_seconds,
                    "GPU-seconds per 1,000 deployment instances",
                    "measured",
                    generation_source,
                    "One deterministic completion per prompt over the full frozen test prompt set.",
                ),
                phase_row(
                    "Base Best-of-64",
                    seed,
                    "generation",
                    "one_gpu",
                    base_generation_per_instance[seed],
                    "GPU-seconds per deployment instance",
                    "measured",
                    base_source,
                    "64 completions per prompt; calibration generations are efficacy-excluded.",
                ),
            ]
        )

        frozen = frozen_costs[seed]
        frozen_compile_generation[seed] = float(frozen["generation_gpu_seconds"])
        scheduler = frozen["scheduler"]
        frozen_compile_allocation_multipliers[seed] = int(scheduler["allocated_gpus"])
        phase_rows.extend(
            [
                phase_row(
                    "Frozen Hero",
                    seed,
                    "generation",
                    "one_gpu",
                    frozen_compile_generation[seed],
                    "GPU-seconds per 64-completion compile",
                    "measured",
                    str(frozen_hero_root / f"seed{seed}/timing_accounting.json"),
                ),
                phase_row(
                    "Frozen Hero",
                    seed,
                    "development_reference",
                    "cpu",
                    float(frozen["development_reference_cpu_seconds"]),
                    "CPU-seconds",
                    "measured",
                    str(frozen_hero_root / f"seed{seed}/development_references.jsonl"),
                    f"Sum across {frozen['reference_rows']} CP-SAT reference rows executed in parallel.",
                ),
                phase_row(
                    "Frozen Hero",
                    seed,
                    "development_execution",
                    "cpu",
                    float(frozen["development_execution_cpu_seconds"]),
                    "CPU-seconds",
                    "measured",
                    str(
                        frozen_hero_root
                        / f"seed{seed}/selection/candidate_development_evaluations.jsonl"
                    ),
                    f"Sum across {frozen['candidate_evaluations']} unique-candidate executions.",
                ),
                phase_row(
                    "Frozen Hero",
                    seed,
                    "selection",
                    "wall_clock",
                    float(frozen["development_selection_wall_seconds"]),
                    "seconds",
                    "measured",
                    str(frozen_hero_root / f"seed{seed}/timing_accounting.json"),
                    "Parallel candidate execution and lexicographic selection wall; reference construction is separate.",
                ),
                phase_row(
                    "Frozen Hero",
                    seed,
                    "compile_job",
                    "allocated_gh200",
                    float(scheduler["elapsed_seconds"])
                    * int(scheduler["allocated_gpus"])
                    / 3600.0,
                    "GH200-hours",
                    "measured",
                    f"Slurm sacct job {scheduler['job_id']}",
                    "Billed allocation; active one-GPU synthesis is reported separately.",
                ),
                phase_row(
                    "Frozen Hero",
                    seed,
                    "final_execution",
                    "subprocess_wall",
                    float(frozen["test_best_of_three_latency_mean_seconds"]),
                    "best-of-three seconds per deployment instance",
                    "measured",
                    str(
                        frozen_hero_root
                        / f"seed{seed}/test/fixed-code/frozen-hero/seed{seed}/metrics_final.csv"
                    ),
                    "The evaluator records the fastest of three subprocess wall times; this is benchmark latency, not total CPU utilization.",
                ),
                phase_row(
                    "Frozen Hero",
                    seed,
                    "final_test_harness",
                    "wall_clock",
                    float(frozen["test_harness_wall_seconds"]),
                    "seconds per 1,000 instances with three repeats",
                    "measured",
                    str(frozen_hero_root / f"seed{seed}/timing_accounting.json"),
                    "End-to-end parallel evaluation wall time; reported separately from per-instance latency.",
                ),
            ]
        )

        canonical_names = {
            "Hero": "Ours (Hero)",
            "Base Best-of-64": "Base (Best-of-64)",
            "ShinkaEvolve": "ShinkaEvolve",
        }
        for method, canonical_name in canonical_names.items():
            row = _rows_for_method_seed(corrected_rows, canonical_name, seed)
            phase_rows.append(
                phase_row(
                    method,
                    seed,
                    "final_execution" if method != "Base Best-of-64" else "selection",
                    "subprocess_wall",
                    float(row["Cost"]),
                    "seconds per deployment instance",
                    "measured",
                    protocol["input_files"]["corrected_summary"]["path"],
                    "Candidate-program subprocess latency; generation is separate."
                    if method != "Base Best-of-64"
                    else "Cumulative candidate subprocess wall and selection over 64 candidates; generation is separate.",
                )
            )

        adaptive = adaptive_costs[seed]
        adaptive_scheduler = adaptive["scheduler"]
        phase_rows.extend(
            [
                phase_row(
                    "Adaptive Repair",
                    seed,
                    "generation_repair_selection",
                    "one_gpu_controller_process_wall",
                    float(adaptive["controller_wall_seconds"]),
                    "seconds",
                    "measured",
                    str(adaptive_repair_root / f"seed{seed}/timing_summary.json"),
                    "Includes model load, 64 completions, feedback orchestration, development execution wall, and selection; pure generation is not separable and is not claimed.",
                ),
                phase_row(
                    "Adaptive Repair",
                    seed,
                    "development_reference",
                    "cpu",
                    float(adaptive["development_reference_cpu_seconds"]),
                    "CPU-seconds",
                    "measured",
                    str(
                        adaptive_repair_root
                        / f"seed{seed}/development_references.jsonl"
                    ),
                    "Sum of single-worker CP-SAT reference wall times; references execute in parallel.",
                ),
                phase_row(
                    "Adaptive Repair",
                    seed,
                    "development_execution",
                    "cpu",
                    float(adaptive["development_execution_cpu_seconds"]),
                    "CPU-seconds",
                    "measured",
                    str(
                        adaptive_repair_root
                        / f"seed{seed}/synthesis/development_evaluations.jsonl"
                    ),
                    "Sum across all 2,560 development executions.",
                ),
                phase_row(
                    "Adaptive Repair",
                    seed,
                    "final_execution",
                    "subprocess_wall",
                    float(adaptive["test_best_of_three_latency_mean_seconds"]),
                    "best-of-three seconds per deployment instance",
                    "measured",
                    str(
                        adaptive_repair_root
                        / f"seed{seed}/test-evaluation/metrics_final.csv"
                    ),
                    "The evaluator records the fastest of three subprocess wall times; this is benchmark latency, not total CPU utilization.",
                ),
                phase_row(
                    "Adaptive Repair",
                    seed,
                    "final_test_harness",
                    "wall_clock",
                    float(adaptive["test_harness_wall_seconds"]),
                    "seconds per 1,000 instances with three repeats",
                    "measured",
                    str(adaptive_repair_root / f"seed{seed}/timing_summary.json"),
                    "End-to-end parallel evaluation wall time; reported separately from per-instance latency.",
                ),
                phase_row(
                    "Adaptive Repair",
                    seed,
                    "full_job",
                    "allocated_gh200",
                    float(adaptive_scheduler["elapsed_seconds"])
                    * int(adaptive_scheduler["allocated_gpus"])
                    / 3600.0,
                    "GH200-hours",
                    "measured",
                    f"Slurm sacct job {adaptive_scheduler['job_id']} ({protocol['adaptive_repair_timing_protocol_id']})",
                    "Includes environment setup, reference construction, controller, and final test harness.",
                ),
            ]
        )

        for phase in ("generation", "development_execution", "selection"):
            phase_rows.append(
                phase_row(
                    "ShinkaEvolve",
                    seed,
                    phase,
                    "external_api_or_cpu",
                    None,
                    "unavailable",
                    "unavailable",
                    "docs/technical-reports/SHINKA_TRACE_GENERATION_REPORT.md",
                    "Synthesis wall, token billing, and selection timing were not recoverable and are not imputed.",
                )
            )

    break_even = compute_break_even_full(
        training_gpu_seconds,
        hero_generation_per_instance,
        base_generation_per_instance,
        frozen_compile_generation,
        generation_allocation_multipliers,
        frozen_compile_allocation_multipliers,
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "analysis_commit": git_commit(repo_root),
        "phase_rows": phase_rows,
        "break_even": break_even,
        "provenance": {
            "input_hashes": {
                name: spec["sha256"] for name, spec in protocol["input_files"].items()
            },
            "full_generation_result": {
                "path": portable_path(full_manifest_path, repo_root),
                "sha256": sha256_file(full_manifest_path),
                "artifact_root": portable_path(calibration_root, repo_root),
            },
            "frozen_hero_cost_result": {
                "path": portable_path(frozen_manifest_path, repo_root),
                "sha256": sha256_file(frozen_manifest_path),
                "artifact_root": portable_path(frozen_hero_root, repo_root),
            },
            "adaptive_repair_cost_result": {
                "path": portable_path(adaptive_manifest_path, repo_root),
                "sha256": sha256_file(adaptive_manifest_path),
                "artifact_root": portable_path(adaptive_repair_root, repo_root),
            },
            "seed101_generation_timing": {
                "hero": source_manifest["timing_inputs"]["hero_seed101"],
                "base64": source_manifest["timing_inputs"]["base64_seed101"],
            },
            "classification": {
                "measured": "direct timing, execution sum, or Slurm record",
                "reconstructed": "metadata recovered from an original system of record",
                "included_in_training": "charged by training and not added again",
                "unavailable": "not recovered and not imputed",
            },
            "unit_policy": protocol["units"]["combination_policy"],
        },
    }


def write_report_v3(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "cost_break_even.json").write_text(
        json.dumps(result["break_even"], indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(result["provenance"], indent=2, sort_keys=True) + "\n"
    )
    rows = result["phase_rows"]
    with (output_dir / "per_seed_costs.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    values = result["break_even"]
    one_hero = values["one_policy_ordinary_hero_interval_instances"]
    one_frozen = values["one_policy_frozen_hero_interval_instances"]
    campaign_hero = values["three_seed_campaign_ordinary_hero_interval_instances"]
    campaign_frozen = values["three_seed_campaign_frozen_hero_interval_instances"]
    allocated = values["allocated_node_sensitivity"]
    lines = [
        "# N26-E3-v3 Full Cost and Break-Even Accounting",
        "",
        f"Analysis commit: `{result['analysis_commit']}`",
        "",
        "## GPU break-even",
        "",
        f"- One-policy ordinary Hero: {one_hero[0]:,}--{one_hero[1]:,} deployment instances.",
        f"- One-policy compile-once Frozen Hero: {one_frozen[0]:,}--{one_frozen[1]:,} deployment instances.",
        f"- Full three-seed campaign, ordinary deployment: {campaign_hero[0]:,}--{campaign_hero[1]:,} instances.",
        f"- Full three-seed campaign, compile-once deployment: {campaign_frozen[0]:,}--{campaign_frozen[1]:,} instances.",
        f"- Scheduler-allocation sensitivity, one policy: ordinary {allocated['one_policy_ordinary_hero_interval_instances'][0]:,}--{allocated['one_policy_ordinary_hero_interval_instances'][1]:,}; compile-once {allocated['one_policy_frozen_hero_interval_instances'][0]:,}--{allocated['one_policy_frozen_hero_interval_instances'][1]:,} instances.",
        f"- Scheduler-allocation sensitivity, full campaign: ordinary {allocated['three_seed_campaign_ordinary_hero_interval_instances'][0]:,}--{allocated['three_seed_campaign_ordinary_hero_interval_instances'][1]:,}; compile-once {allocated['three_seed_campaign_frozen_hero_interval_instances'][0]:,}--{allocated['three_seed_campaign_frozen_hero_interval_instances'][1]:,} instances.",
        "",
        "Ordinary Hero retains one completion per deployment; Frozen Hero charges the 64-completion validation synthesis once. CPU execution, active GPU generation, billed allocation, and wall time remain separate. ShinkaEvolve synthesis and billing remain unavailable.",
        "",
        "The phase-by-seed ledger is `per_seed_costs.csv`; hashes and evidence definitions are in `provenance.json`.",
        "",
    ]
    (output_dir / "cost_break_even.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/neurips2026/cost_accounting_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-root",
        type=Path,
        help="Registered generation-calibration directory containing seed202/303.",
    )
    parser.add_argument(
        "--adaptive-repair-root",
        type=Path,
        help="Registered content-clean Adaptive Repair artifact directory.",
    )
    parser.add_argument(
        "--frozen-hero-root",
        type=Path,
        help="Registered leakage-free Frozen Hero construction directory.",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest = json.loads((repo_root / args.manifest).read_text())
    if manifest.get("protocol_id") == "N26-E3-v3":
        if (
            args.calibration_root is None
            or args.adaptive_repair_root is None
            or args.frozen_hero_root is None
        ):
            raise ValueError(
                "--calibration-root, --adaptive-repair-root, and --frozen-hero-root "
                "are required for N26-E3-v3"
            )
        result = analyze_v3(
            manifest,
            repo_root,
            args.calibration_root.resolve(),
            args.adaptive_repair_root.resolve(),
            args.frozen_hero_root.resolve(),
        )
        write_report_v3(result, args.output_dir)
    elif manifest.get("protocol_id") == "N26-E3-v2":
        if args.calibration_root is None:
            raise ValueError("--calibration-root is required for N26-E3-v2")
        result = analyze_v2(manifest, repo_root, args.calibration_root.resolve())
        write_report_v2(result, args.output_dir)
    else:
        result = analyze(manifest, repo_root)
        write_report(result, args.output_dir)
    print(json.dumps(result["break_even"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
