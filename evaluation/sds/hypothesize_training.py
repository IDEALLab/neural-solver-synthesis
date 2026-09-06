"""Validation helpers for the frozen N26-E7 training-time prompt ablation."""

# ruff: noqa: PLC0415, PLR0911, PLR0915, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


RUNTIME_IDENTITY_FIELDS = frozenset(
    {
        "hub_model_id",
        "logging_dir",
        "output_dir",
        "run_name",
        "vllm_server_host",
    }
)


def normalize_runtime_value(value: Any) -> Any:
    """Convert serialized TrainingArguments values into stable JSON data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return normalize_runtime_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): normalize_runtime_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_runtime_value(item) for item in value]
    if isinstance(value, set):
        normalized = [normalize_runtime_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if hasattr(value, "to_dict"):
        return normalize_runtime_value(value.to_dict())
    if hasattr(value, "__dict__"):
        return normalize_runtime_value(vars(value))
    return repr(value)


def compare_runtime_arguments(
    control: dict[str, Any],
    ablation: dict[str, Any],
    allowed_fields: frozenset[str] = RUNTIME_IDENTITY_FIELDS,
) -> dict[str, Any]:
    """Reject any runtime-argument drift outside explicit run-identity fields."""
    control_normalized = normalize_runtime_value(control)
    ablation_normalized = normalize_runtime_value(ablation)
    fields = sorted(set(control_normalized) | set(ablation_normalized))
    differences = {
        field: {
            "control": control_normalized.get(field),
            "ablation": ablation_normalized.get(field),
        }
        for field in fields
        if control_normalized.get(field) != ablation_normalized.get(field)
    }
    scientific_differences = sorted(set(differences) - set(allowed_fields))
    if scientific_differences:
        raise ValueError(
            "runtime TrainingArguments differ outside run identity: "
            f"{scientific_differences}"
        )
    return {
        "status": "validated",
        "argument_count": len(fields),
        "allowed_fields": sorted(allowed_fields),
        "observed_allowed_differences": differences,
        "observed_identity_differences": {
            field: value
            for field, value in differences.items()
            if field in RUNTIME_IDENTITY_FIELDS
        },
        "observed_operational_differences": {
            field: value
            for field, value in differences.items()
            if field not in RUNTIME_IDENTITY_FIELDS
        },
        "scientific_differences": [],
    }


def load_training_arguments(path: Path) -> dict[str, Any]:
    import torch

    arguments = torch.load(path, map_location="cpu", weights_only=False)
    if not hasattr(arguments, "__dict__"):
        raise ValueError(f"unexpected TrainingArguments payload: {type(arguments)!r}")
    return vars(arguments)


def validate_runtime_resume_provenance(
    control: dict[str, Any],
    ablation: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate a protocol-declared continuation without weakening other pairs."""
    runtime_validation = protocol.get("runtime_validation", {})
    resume_policy = runtime_validation.get("resume_from_checkpoint")
    if resume_policy is None:
        return None

    allowed = set(runtime_validation.get("allowed_operational_differences", []))
    if allowed != {"resume_from_checkpoint"}:
        raise ValueError(
            "resume policy requires exactly the resume_from_checkpoint allowance"
        )

    observed_control = normalize_runtime_value(control.get("resume_from_checkpoint"))
    observed_ablation = normalize_runtime_value(
        ablation.get("resume_from_checkpoint")
    )
    expected_control = resume_policy["control"]
    expected_ablation = resume_policy["ablation"]
    expected_path = expected_ablation["path"]
    if observed_control != expected_control:
        raise ValueError(
            f"control resume provenance: {observed_control!r} != {expected_control!r}"
        )
    if observed_ablation != expected_path:
        raise ValueError(
            f"ablation resume provenance: {observed_ablation!r} != {expected_path!r}"
        )

    checkpoint = Path(expected_path)
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise ValueError(f"resume trainer state is missing: {state_path}")
    state = load_json(state_path)
    expected_step = int(expected_ablation["global_step"])
    expected_horizon = int(expected_ablation["scheduler_max_steps"])
    if checkpoint.name != f"checkpoint-{expected_step}":
        raise ValueError("resume checkpoint name does not match its declared step")
    if int(state.get("global_step", -1)) != expected_step:
        raise ValueError(
            f"resume global step: {state.get('global_step')} != {expected_step}"
        )
    if int(state.get("max_steps", -1)) != expected_horizon:
        raise ValueError(
            f"resume scheduler horizon: {state.get('max_steps')} != {expected_horizon}"
        )

    shard_count = int(expected_ablation["distributed_shards"])
    state_dir = checkpoint / f"global_step{expected_step}"
    counts = {
        "optimizer": len(list(state_dir.glob("*optim_states.pt"))),
        "model_state": len(list(state_dir.glob("*model_states.pt"))),
        "rng": len(list(checkpoint.glob("rng_state_*.pth"))),
    }
    wrong_counts = {
        name: count for name, count in counts.items() if count != shard_count
    }
    if wrong_counts:
        raise ValueError(
            f"resume checkpoint has incomplete distributed state: {wrong_counts}"
        )
    for name in ("scheduler.pt", "latest", "model.safetensors.index.json"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"resume checkpoint is missing {name}")

    return {
        "status": "validated",
        "control_resume_from_checkpoint": observed_control,
        "ablation_resume_from_checkpoint": observed_ablation,
        "global_step": expected_step,
        "scheduler_max_steps": expected_horizon,
        "distributed_state_counts": counts,
        "trainer_state_sha256": sha256_file(state_path),
        "scheduler_sha256": sha256_file(checkpoint / "scheduler.pt"),
    }


def validate_runtime_prompt_change(
    control_prompt: str, ablation_prompt: str, protocol_path: Path
) -> dict[str, Any]:
    """Require the serialized runtime prompts to encode the frozen one-line edit."""
    protocol = load_json(protocol_path)
    control_prompt = control_prompt.strip()
    ablation_prompt = ablation_prompt.strip()
    prompt_protocol = protocol.get("configs", protocol)
    instruction = prompt_protocol["removed_instruction"]
    if control_prompt.count(instruction.rstrip("\n")) != 1:
        raise ValueError("runtime control prompt lacks the frozen instruction")
    if control_prompt.replace(instruction, "") != ablation_prompt:
        raise ValueError(
            "runtime ablation prompt has changes beyond the frozen instruction"
        )
    hashes = {
        "control_prompt_sha256": sha256_text(control_prompt),
        "ablation_prompt_sha256": sha256_text(ablation_prompt),
    }
    for key, actual in hashes.items():
        expected = prompt_protocol[key]
        if actual != expected:
            raise ValueError(f"runtime {key}: {actual} != {expected}")
    return {"status": "validated", **hashes}


def validate_runtime_arguments(
    control_path: Path, ablation_path: Path, protocol_path: Path
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    control = load_training_arguments(control_path)
    ablation = load_training_arguments(ablation_path)
    prompt_validation = validate_runtime_prompt_change(
        str(control.pop("system_prompt")),
        str(ablation.pop("system_prompt")),
        protocol_path,
    )
    runtime_validation = protocol.get("runtime_validation", {})
    allowed_operational = frozenset(
        runtime_validation.get("allowed_operational_differences", [])
    )
    if allowed_operational - {"resume_from_checkpoint"}:
        raise ValueError(
            "unsupported operational runtime allowances: "
            f"{sorted(allowed_operational - {'resume_from_checkpoint'})}"
        )
    result = compare_runtime_arguments(
        control,
        ablation,
        allowed_fields=RUNTIME_IDENTITY_FIELDS | allowed_operational,
    )
    resume_provenance = validate_runtime_resume_provenance(
        control,
        ablation,
        protocol,
    )
    return {
        **result,
        "intended_prompt_difference": prompt_validation,
        "resume_provenance": resume_provenance,
        "control_training_args_sha256": sha256_file(control_path),
        "ablation_training_args_sha256": sha256_file(ablation_path),
    }


EVALUATION_METADATA_FILES = (
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def checkpoint_artifact_manifest(
    checkpoint: Path, *, min_model_bytes: int = 25_000_000_000
) -> dict[str, Any]:
    """Hash the complete inference artifact named by its safetensors index."""
    index_path = checkpoint / "model.safetensors.index.json"
    index = load_json(index_path)
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise ValueError(f"checkpoint has no indexed model shards: {checkpoint}")
    required = [
        *EVALUATION_METADATA_FILES,
        "model.safetensors.index.json",
        "trainer_state.json",
        "training_args.bin",
        *shard_names,
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"checkpoint is incomplete: {missing}")
    model_bytes = sum((checkpoint / name).stat().st_size for name in shard_names)
    if model_bytes < min_model_bytes:
        raise ValueError(
            f"indexed model has {model_bytes} bytes, expected at least {min_model_bytes}"
        )
    return {
        "checkpoint": str(checkpoint),
        "model_shards": shard_names,
        "model_shard_count": len(shard_names),
        "model_bytes": model_bytes,
        "index_metadata": index.get("metadata", {}),
        "weight_map_sha256": sha256_text(
            json.dumps(index["weight_map"], sort_keys=True, separators=(",", ":"))
        ),
        "files_sha256": {name: sha256_file(checkpoint / name) for name in required},
        "optimizer_and_rng_state_excluded": True,
    }


def validate_checkpoint_pair(
    control_checkpoint: Path,
    ablation_checkpoint: Path,
    *,
    protocol_path: Path,
    min_model_bytes: int = 25_000_000_000,
) -> dict[str, Any]:
    """Validate runtime equivalence and hash both complete inference artifacts."""
    runtime = validate_runtime_arguments(
        control_checkpoint / "training_args.bin",
        ablation_checkpoint / "training_args.bin",
        protocol_path,
    )
    control = checkpoint_artifact_manifest(
        control_checkpoint, min_model_bytes=min_model_bytes
    )
    ablation = checkpoint_artifact_manifest(
        ablation_checkpoint, min_model_bytes=min_model_bytes
    )
    metadata_mismatches = [
        name
        for name in EVALUATION_METADATA_FILES
        if control["files_sha256"][name] != ablation["files_sha256"][name]
    ]
    if metadata_mismatches:
        raise ValueError(
            "control/ablation inference metadata differs: " f"{metadata_mismatches}"
        )
    if control["weight_map_sha256"] != ablation["weight_map_sha256"]:
        raise ValueError("control/ablation model tensor layouts differ")
    return {
        "status": "validated",
        "runtime_arguments": runtime,
        "control": control,
        "ablation": ablation,
        "matching_inference_metadata": list(EVALUATION_METADATA_FILES),
        "expected_weight_difference": True,
    }


def validate_prompt_only_change(
    protocol_path: Path, control_path: Path, ablation_path: Path
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    control = yaml.safe_load(control_path.read_text())
    ablation = yaml.safe_load(ablation_path.read_text())

    control_prompt = control.pop("system_prompt").strip()
    ablation_prompt = ablation.pop("system_prompt").strip()
    if control != ablation:
        changed = sorted(
            key
            for key in set(control) | set(ablation)
            if control.get(key) != ablation.get(key)
        )
        raise ValueError(f"non-prompt configuration changes: {changed}")

    instruction = protocol["removed_instruction"]
    if control_prompt.count(instruction.rstrip("\n")) != 1:
        raise ValueError(
            "control prompt must contain the frozen instruction exactly once"
        )
    expected_ablation = control_prompt.replace(instruction, "")
    if ablation_prompt != expected_ablation:
        raise ValueError(
            "ablation prompt is not the control prompt minus the frozen instruction"
        )

    checks = {
        "control_config_sha256": sha256_file(control_path),
        "ablation_config_sha256": sha256_file(ablation_path),
        "control_prompt_sha256": sha256_text(control_prompt),
        "ablation_prompt_sha256": sha256_text(ablation_prompt),
    }
    for key, actual in checks.items():
        if actual != protocol[key]:
            raise ValueError(f"{key}: {actual} != {protocol[key]}")
    return {"protocol_id": protocol["protocol_id"], **checks, "status": "validated"}


def validate_target(protocol_path: Path, seed: int, max_steps: int) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    seed_text = str(seed)
    if seed_text not in protocol["targets"]:
        raise ValueError(f"unsupported seed: {seed}")
    expected = int(protocol["targets"][seed_text]["max_steps"])
    if max_steps != expected:
        raise ValueError(f"seed {seed} target: {max_steps} != {expected}")
    return {"protocol_id": protocol["protocol_id"], "seed": seed, "max_steps": expected}


def validate_checkpoint(
    protocol_path: Path, seed: int, checkpoint: Path
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    target = int(protocol["targets"][str(seed)]["max_steps"])
    expected_name = f"checkpoint-{target}"
    if checkpoint.name != expected_name:
        raise ValueError(f"checkpoint name: {checkpoint.name} != {expected_name}")
    state_path = checkpoint / "trainer_state.json"
    if not state_path.exists():
        raise ValueError(f"missing trainer state: {state_path}")
    state = load_json(state_path)
    if int(state.get("global_step", -1)) != target:
        raise ValueError(f"global step: {state.get('global_step')} != {target}")
    return {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "checkpoint": str(checkpoint),
        "global_step": target,
        "trainer_state_sha256": sha256_file(state_path),
        "status": "validated",
    }


def validate_historical_environment(
    protocol_path: Path, open_r1_root: Path, edf_path: Path
) -> dict[str, Any]:
    """Validate the exact historical stack and one-line prompt intervention."""
    protocol = load_json(protocol_path)
    configs = protocol["configs"]
    control_path = open_r1_root / configs["control"]
    ablation_path = open_r1_root / configs["ablation"]
    observed = {
        "control_sha256": sha256_file(control_path),
        "ablation_sha256": sha256_file(ablation_path),
    }
    for key, actual in observed.items():
        if actual != configs[key]:
            raise ValueError(f"historical {key}: {actual} != {configs[key]}")
    control = control_path.read_bytes()
    ablation = ablation_path.read_bytes()
    removed = configs["removed_instruction"].encode()
    if control.count(removed) != 1:
        raise ValueError("historical control lacks the unique removed instruction")
    if control.replace(b"  " + removed, b"", 1) != ablation:
        raise ValueError("historical configs differ beyond one exact prompt line")
    if sha256_file(edf_path) != protocol["edf"]["sha256"]:
        raise ValueError("historical EDF hash drift")
    commit = subprocess.check_output(
        ["git", "-C", str(open_r1_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != protocol["open_r1"]["execution_commit"]:
        raise ValueError("historical OpenR1 execution commit drift")
    dirty = subprocess.check_output(
        ["git", "-C", str(open_r1_root), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise ValueError("historical OpenR1 checkout is dirty")
    return {
        "protocol_id": protocol["protocol_id"],
        "open_r1_commit": commit,
        "edf_sha256": protocol["edf"]["sha256"],
        **observed,
        "status": "validated",
    }


def validate_historical_checkpoint(
    protocol_path: Path, seed: int, checkpoint: Path
) -> dict[str, Any]:
    """Reject a stopped checkpoint if its scheduler no longer matches history."""
    protocol = load_json(protocol_path)
    if str(seed) not in protocol["targets"]:
        raise ValueError(f"unsupported historical seed: {seed}")
    target = protocol["targets"][str(seed)]
    stop_step = int(target["stop_step"])
    expected_name = f"checkpoint-{stop_step}"
    if checkpoint.name != expected_name:
        raise ValueError(f"checkpoint name: {checkpoint.name} != {expected_name}")
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise ValueError(f"missing trainer state: {state_path}")
    state = load_json(state_path)
    if int(state.get("global_step", -1)) != stop_step:
        raise ValueError(f"global step: {state.get('global_step')} != {stop_step}")
    horizon = int(protocol["scheduler"]["max_steps"])
    if int(state.get("max_steps", -1)) != horizon:
        raise ValueError(
            f"scheduler horizon: {state.get('max_steps')} != {horizon}"
        )
    learning_rates = [
        float(row["learning_rate"])
        for row in state.get("log_history", [])
        if int(row.get("step", -1)) == stop_step and "learning_rate" in row
    ]
    if not learning_rates:
        raise ValueError("target-step learning rate is missing from trainer state")
    observed_lr = learning_rates[-1]
    expected_lr = float(target["expected_learning_rate"])
    tolerance = float(protocol["scheduler"]["expected_learning_rate_tolerance"])
    if abs(observed_lr - expected_lr) > tolerance:
        raise ValueError(
            f"target learning rate: {observed_lr} != {expected_lr} +/- {tolerance}"
        )
    return {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "checkpoint": str(checkpoint),
        "global_step": stop_step,
        "scheduler_max_steps": horizon,
        "learning_rate": observed_lr,
        "expected_learning_rate": expected_lr,
        "learning_rate_tolerance": tolerance,
        "trainer_state_sha256": sha256_file(state_path),
        "status": "validated",
    }


def validate_extended_checkpoint(
    protocol_path: Path, seed: int, step: int, checkpoint: Path
) -> dict[str, Any]:
    """Validate one predeclared checkpoint in the extended prompt trajectory."""
    protocol = load_json(protocol_path)
    seed_text = str(seed)
    if seed_text not in protocol["resumes"]:
        raise ValueError(f"unsupported extended-training seed: {seed}")
    allowed_steps = {int(value) for value in protocol["evaluation"]["analysis_checkpoints"]}
    if step not in allowed_steps:
        raise ValueError(f"checkpoint {step} is not predeclared: {sorted(allowed_steps)}")
    expected_name = f"checkpoint-{step}"
    if checkpoint.name != expected_name:
        raise ValueError(f"checkpoint name: {checkpoint.name} != {expected_name}")
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise ValueError(f"missing trainer state: {state_path}")
    state = load_json(state_path)
    if int(state.get("global_step", -1)) != step:
        raise ValueError(f"global step: {state.get('global_step')} != {step}")
    training = protocol["training"]
    horizon = int(training["scheduler_horizon_steps"])
    if int(state.get("max_steps", -1)) != horizon:
        raise ValueError(f"scheduler horizon: {state.get('max_steps')} != {horizon}")
    learning_rates = [
        float(row["learning_rate"])
        for row in state.get("log_history", [])
        if int(row.get("step", -1)) == step and "learning_rate" in row
    ]
    if not learning_rates:
        raise ValueError("target-step learning rate is missing from trainer state")
    observed_lr = learning_rates[-1]
    expected_lr = float(training["expected_learning_rates"][str(step)])
    tolerance = float(training["expected_learning_rate_tolerance"])
    if abs(observed_lr - expected_lr) > tolerance:
        raise ValueError(
            f"target learning rate: {observed_lr} != {expected_lr} +/- {tolerance}"
        )
    return {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "checkpoint": str(checkpoint),
        "global_step": step,
        "scheduler_max_steps": horizon,
        "learning_rate": observed_lr,
        "expected_learning_rate": expected_lr,
        "learning_rate_tolerance": tolerance,
        "trainer_state_sha256": sha256_file(state_path),
        "inference_artifact": checkpoint_artifact_manifest(checkpoint),
        "status": "validated",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config")
    config_parser.add_argument("--protocol", type=Path, required=True)
    config_parser.add_argument("--control", type=Path, required=True)
    config_parser.add_argument("--ablation", type=Path, required=True)

    target_parser = subparsers.add_parser("target")
    target_parser.add_argument("--protocol", type=Path, required=True)
    target_parser.add_argument("--seed", type=int, required=True)
    target_parser.add_argument("--max-steps", type=int, required=True)

    checkpoint_parser = subparsers.add_parser("checkpoint")
    checkpoint_parser.add_argument("--protocol", type=Path, required=True)
    checkpoint_parser.add_argument("--seed", type=int, required=True)
    checkpoint_parser.add_argument("--checkpoint", type=Path, required=True)
    checkpoint_parser.add_argument("--output", type=Path)

    runtime_parser = subparsers.add_parser("runtime-args")
    runtime_parser.add_argument("--control", type=Path, required=True)
    runtime_parser.add_argument("--ablation", type=Path, required=True)
    runtime_parser.add_argument("--protocol", type=Path, required=True)
    runtime_parser.add_argument("--output", type=Path)

    pair_parser = subparsers.add_parser("checkpoint-pair")
    pair_parser.add_argument("--control", type=Path, required=True)
    pair_parser.add_argument("--ablation", type=Path, required=True)
    pair_parser.add_argument("--protocol", type=Path, required=True)
    pair_parser.add_argument("--output", type=Path, required=True)
    pair_parser.add_argument("--min-model-bytes", type=int, default=25_000_000_000)

    historical_environment_parser = subparsers.add_parser("historical-environment")
    historical_environment_parser.add_argument("--protocol", type=Path, required=True)
    historical_environment_parser.add_argument("--open-r1-root", type=Path, required=True)
    historical_environment_parser.add_argument("--edf", type=Path, required=True)
    historical_environment_parser.add_argument("--output", type=Path)

    historical_checkpoint_parser = subparsers.add_parser("historical-checkpoint")
    historical_checkpoint_parser.add_argument("--protocol", type=Path, required=True)
    historical_checkpoint_parser.add_argument("--seed", type=int, required=True)
    historical_checkpoint_parser.add_argument("--checkpoint", type=Path, required=True)
    historical_checkpoint_parser.add_argument("--output", type=Path)

    extended_checkpoint_parser = subparsers.add_parser("extended-checkpoint")
    extended_checkpoint_parser.add_argument("--protocol", type=Path, required=True)
    extended_checkpoint_parser.add_argument("--seed", type=int, required=True)
    extended_checkpoint_parser.add_argument("--step", type=int, required=True)
    extended_checkpoint_parser.add_argument("--checkpoint", type=Path, required=True)
    extended_checkpoint_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "historical-environment":
        result = validate_historical_environment(
            args.protocol, args.open_r1_root, args.edf
        )
    elif args.command == "historical-checkpoint":
        result = validate_historical_checkpoint(
            args.protocol, args.seed, args.checkpoint
        )
    elif args.command == "extended-checkpoint":
        result = validate_extended_checkpoint(
            args.protocol, args.seed, args.step, args.checkpoint
        )
    elif args.command == "config":
        result = validate_prompt_only_change(args.protocol, args.control, args.ablation)
    elif args.command == "target":
        result = validate_target(args.protocol, args.seed, args.max_steps)
    elif args.command == "runtime-args":
        result = validate_runtime_arguments(args.control, args.ablation, args.protocol)
    elif args.command == "checkpoint-pair":
        result = validate_checkpoint_pair(
            args.control,
            args.ablation,
            protocol_path=args.protocol,
            min_model_bytes=args.min_model_bytes,
        )
    else:
        result = validate_checkpoint(args.protocol, args.seed, args.checkpoint)
    payload = json.dumps(result, indent=2) + "\n"
    if getattr(args, "output", None):
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
