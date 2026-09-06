import argparse
import json
import os
import re
from pathlib import Path

import yaml
from datasets import load_dataset
from vllm import LLM, SamplingParams

from evaluation.sds.prompt_variants import (
    CANONICAL_VARIANT,
    PROMPT_VARIANTS,
    PromptVariantError,
    apply_prompt_variant,
    sha256_text,
)

# Magic value constants
_MIN_MODEL_NAME_PARTS = 3


# --- CUSTOM EXCEPTIONS ---
class CheckpointNotFoundError(FileNotFoundError):
    """Raised when checkpoint directory or checkpoints are not found."""

    def __init__(self, checkpoint_dir: str, message: str | None = None):
        if message is None:
            msg = f"Checkpoint directory does not exist: {checkpoint_dir}"
        else:
            msg = message
        super().__init__(msg)
        self.checkpoint_dir = checkpoint_dir


class ModelPathNotFoundError(FileNotFoundError):
    """Raised when model path does not exist."""

    def __init__(self, model_path: str):
        msg = f"Model path does not exist: {model_path}"
        super().__init__(msg)
        self.model_path = model_path


class ModelDiscoveryError(ValueError):
    """Raised when model path cannot be discovered."""

    def __init__(self):
        msg = (
            "Must provide either --checkpoint_dir or (--model, --training-scheme, --seed) "
            "to discover model path"
        )
        super().__init__(msg)


class DatasetLoadError(RuntimeError):
    """Raised when dataset loading fails."""

    def __init__(self, dataset_name: str, original_error: Exception):
        msg = f"Failed to load dataset {dataset_name}: {original_error}"
        super().__init__(msg)
        self.dataset_name = dataset_name
        self.original_error = original_error


def format_prompt(item, tokenizer, system_prompt=None):
    """Reconstructs the exact prompt format used during training."""
    # Use provided system prompt or default (matching config_hero.yaml)
    if system_prompt is None:
        system_prompt = (
            "You are an expert engineering reasoning engine specialized in combinatorial optimization. Your task is to write a high-performance Python script to solve Synergistic Dependency Selection (SDS) problems.\n\n"
            "FORMATTING RULES:\n"
            "1. Begin with a <think> block.\n"
            "2. End with a single <code> block containing the JSON-processing Python script.\n"
            "3. No other text is allowed outside these blocks.\n\n"
            "THINKING GUIDELINES:\n"
            "Inside the <think> block, you must engage in a rigorous algorithm design process:\n"
            "- **Deconstruct**: Analyze the objective landscape. Acknowledge that simple heuristics (like greedy selection) will likely get stuck in local optima due to negative interaction weights and complex constraints.\n"
            '- **Hypothesize**: Propose a search strategy capable of exploring the solution space effectively. Consider how to balance "exploitation" (improving a good solution) with "exploration" (escaping bad local optima).\n'
            '- **Critique**: Question your approach. "Does my algorithm just pick the next best item? If so, it will fail on deceptive landscapes. How do I add lookahead, backtracking, or iterative improvement?"\n'
            "- **Simulate**: Mentally dry-run the logic to ensure constraints (mutex, precedence) are strictly satisfied during the search.\n"
            "- **Finalize**: Verify I/O requirements.\n\n"
            "GOAL:\n"
            "Your code must aim for **Global Optimality**, while being feasible. You must write a self-contained solver (no external black-box libraries like OR-Tools) that intelligently searches for the best possible score under constraints."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": item["problem"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def normalize_model_name_for_checkpoint(model_name: str) -> str:
    """
    Normalize model name to match checkpoint directory naming convention.
    Examples:
        "qwen2.5-coder-7b" -> "Qwen2.5-Coder-7B-Instruct"
        "qwen2.5-coder-32b" -> "Qwen2.5-Coder-32B-Instruct"
        "qwen2.5-coder-1.5b" -> "Qwen2.5-Coder-1.5B-Instruct"
    """
    # Convert to title case and handle special cases
    parts = model_name.lower().split("-")
    if len(parts) >= _MIN_MODEL_NAME_PARTS and parts[0] == "qwen" and parts[1] == "2.5":
        size = parts[2].upper()  # "7b" -> "7B", "32b" -> "32B"
        return f"Qwen2.5-Coder-{size}-Instruct"
    # Fallback: capitalize each part
    return "-".join(p.capitalize() for p in parts)


def extract_model_size(model_name: str) -> str:
    """
    Extract model size (e.g., "7B", "14B", "32B") from model name.

    Examples:
        "qwen2.5-coder-7b" -> "7B"
        "Qwen2.5-Coder-14B-Instruct" -> "14B"
        "qwen2.5-coder-32b" -> "32B"
    """
    # Normalize to lowercase for parsing
    model_lower = model_name.lower()

    # Try to extract size pattern (e.g., "7b", "14b", "32b", "1.5b")
    # Pattern: digits followed by 'b' (case insensitive)
    size_match = re.search(r"(\d+(?:\.\d+)?)b", model_lower)
    if size_match:
        size_str = size_match.group(1)
        # Convert to uppercase format (e.g., "7" -> "7B", "1.5" -> "1.5B")
        return (
            f"{size_str.upper()}B" if not size_str.endswith("B") else size_str.upper()
        )

    # Fallback: try to extract from normalized checkpoint name
    normalized = normalize_model_name_for_checkpoint(model_name)
    match = re.search(r"Qwen2\.5-Coder-(\d+(?:\.\d+)?B)-Instruct", normalized)
    if match:
        return match.group(1)

    return None


def auto_detect_hero_config(
    model_name: str | None = None, model_path: str | None = None
) -> str:
    """
    Auto-detect the Hero config file path based on model name or path.

    Args:
        model_name: Model name (e.g., "qwen2.5-coder-7b", "Qwen2.5-Coder-14B-Instruct")
        model_path: Model path (e.g., "Qwen/Qwen2.5-Coder-14B-Instruct" or checkpoint path)

    Returns:
        Path to config_hero.yaml file, or None if cannot be determined
    """
    # Try to extract size from model_name first
    size = None
    if model_name:
        size = extract_model_size(model_name)

    # If not found, try from model_path
    if not size and model_path:
        # Extract model name from path (handle both HF identifiers and local paths)
        path_parts = model_path.split("/")
        if len(path_parts) > 1:
            # HF identifier: "Qwen/Qwen2.5-Coder-14B-Instruct"
            potential_model = path_parts[-1]
            size = extract_model_size(potential_model)
        else:
            # Local path: try to extract from directory name
            size = extract_model_size(model_path)

    if not size:
        return None

    # Construct config path: deps/open-r1/recipes/Qwen2.5-Coder-{SIZE}-Instruct/grpo/config_hero.yaml
    config_path = Path(
        f"deps/open-r1/recipes/Qwen2.5-Coder-{size}-Instruct/grpo/config_hero.yaml"
    )

    # Check if file exists
    if config_path.exists():
        return str(config_path)

    return None


def construct_checkpoint_dir_name(
    model: str, training_scheme: str, seed: int, base_dir: str | None = None
) -> str:
    """
    Construct checkpoint directory name based on model/training-scheme/seed.

    Matches the naming convention used in training scripts:
    - Qwen2.5-Coder-7B-Instruct-GRPO-SDS-OPT-seed101
    - Qwen2.5-Coder-7B-Instruct-SFT-GRPO-seed101
    - Qwen2.5-1.5B-Instruct-SDS-SFT-GRPO-seed101

    Args:
        model: Model name (e.g., "qwen2.5-coder-7b")
        training_scheme: Training scheme (e.g., "sft", "grpo", "sft-grpo")
        seed: Seed value (e.g., 101, 202, 303)
        base_dir: Base checkpoint directory. Defaults to `SDS_CHECKPOINT_ROOT`
            when set, otherwise `./checkpoints`.

    Returns:
        Full path to checkpoint directory
    """
    if base_dir is None:
        base_dir = Path(os.environ.get("SDS_CHECKPOINT_ROOT", "checkpoints"))
    else:
        base_dir = Path(base_dir)

    # Normalize model name to match checkpoint naming
    normalized_model = normalize_model_name_for_checkpoint(model)

    # Map training scheme to checkpoint suffix
    # Training scripts use: "GRPO-SDS-OPT", "SFT-GRPO", "SFT", etc.
    scheme_map = {"sft": "SFT", "grpo": "GRPO-SDS-OPT", "sft-grpo": "SFT-GRPO"}
    scheme_suffix = scheme_map.get(
        training_scheme.lower().replace("_", "-"), training_scheme.upper()
    )

    # Construct directory name
    checkpoint_dir = f"{normalized_model}-{scheme_suffix}-seed{seed}"

    return str(base_dir / checkpoint_dir)


def discover_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Discover the latest checkpoint in a checkpoint directory.

    Looks for checkpoint-* directories and returns the one with highest number.
    If no checkpoints found but config.json exists, returns the base directory.

    Args:
        checkpoint_dir: Base checkpoint directory

    Returns:
        Path to latest checkpoint or base directory if no checkpoints
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise CheckpointNotFoundError(checkpoint_dir)

    # Find all checkpoint-* directories
    checkpoints = list(checkpoint_path.glob("checkpoint-*"))

    if not checkpoints:
        # No checkpoint-* directories, check if base directory has config.json
        config_file = checkpoint_path / "config.json"
        if config_file.exists():
            return str(checkpoint_path)
        raise CheckpointNotFoundError(
            checkpoint_dir, f"No checkpoints found in {checkpoint_dir}"
        )

    # Sort by checkpoint number (extract number from "checkpoint-123")
    def get_checkpoint_num(path):
        basename = path.name
        try:
            return int(basename.split("-")[1])
        except (IndexError, ValueError):
            return -1

    latest = max(checkpoints, key=get_checkpoint_num)
    return str(latest)


def discover_model_path(
    model: str | None = None,
    training_scheme: str | None = None,
    seed: int | None = None,
    checkpoint_dir: str | None = None,
    base_checkpoint_dir: str | None = None,
) -> str:
    """
    Discover model checkpoint path from model/training-scheme/seed or explicit checkpoint_dir.

    Args:
        model: Model name (e.g., "qwen2.5-coder-7b")
        training_scheme: Training scheme (e.g., "sft", "grpo", "sft-grpo")
        seed: Seed value (e.g., 101, 202, 303)
        checkpoint_dir: Explicit checkpoint directory (takes precedence)
        base_checkpoint_dir: Base directory for checkpoints

    Returns:
        Path to latest checkpoint
    """
    # If explicit checkpoint_dir provided, use it
    if checkpoint_dir:
        return discover_latest_checkpoint(checkpoint_dir)

    # Otherwise, construct from model/training-scheme/seed
    if model and training_scheme and seed is not None:
        constructed_dir = construct_checkpoint_dir_name(
            model, training_scheme, seed, base_checkpoint_dir
        )
        return discover_latest_checkpoint(constructed_dir)

    raise ModelDiscoveryError()


def main():  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(
        description="Generate SDS solutions from trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explicit model path (original method)
  python generate.py --model_path /path/to/checkpoint-100
  
  # Auto-discover from model/training/seed (new method)
  python generate.py --model qwen2.5-coder-7b --training-scheme sft-grpo --seed 101
  
  # Explicit checkpoint directory (auto-finds latest checkpoint)
  python generate.py --checkpoint_dir /path/to/Qwen2.5-Coder-7B-Instruct-GRPO-SDS-OPT-seed101
        """,
    )
    # Model path discovery (mutually exclusive groups)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Explicit path to checkpoint (original method)",
    )
    model_group.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint directory (auto-finds latest checkpoint)",
    )

    # Model discovery arguments (required if using auto-discovery)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for auto-discovery (e.g., 'qwen2.5-coder-7b')",
    )
    parser.add_argument(
        "--training-scheme",
        type=str,
        default=None,
        choices=["sft", "grpo", "sft-grpo", "sft_grpo"],
        help="Training scheme for auto-discovery",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for auto-discovery (e.g., 101, 202, 303)",
    )
    parser.add_argument(
        "--base_checkpoint_dir",
        type=str,
        default=None,
        help="Base checkpoint directory; defaults to SDS_CHECKPOINT_ROOT or ./checkpoints",
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help="Optional immutable Hugging Face model revision.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="SoheylM/OpenR1-SDS-10k-seed303",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default=None,
        help="Optional immutable Hugging Face dataset revision.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split to generate from (default: test).",
    )
    parser.add_argument(
        "--uuid_list",
        type=str,
        default=None,
        help=(
            "Optional newline-delimited UUID allowlist. Rows are emitted in the "
            "listed order and every UUID must occur exactly once in the split."
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation/sds/generations.jsonl",
        help="Output file for generations",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of samples per prompt (1 for greedy, >1 for Pass@K)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional deterministic prefix length for throughput calibration.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to YAML config file to load system_prompt from",
    )
    parser.add_argument(
        "--prompt_variant",
        choices=PROMPT_VARIANTS,
        default=CANONICAL_VARIANT,
        help="Predeclared system-prompt perturbation.",
    )
    parser.add_argument(
        "--provenance_file",
        type=str,
        default=None,
        help="Optional JSON path for model, dataset, and prompt provenance.",
    )
    args = parser.parse_args()

    # Discover model path if not explicitly provided
    if args.model_path:
        model_path = args.model_path
    else:
        try:
            model_path = discover_model_path(
                model=args.model,
                training_scheme=args.training_scheme,
                seed=args.seed,
                checkpoint_dir=args.checkpoint_dir,
                base_checkpoint_dir=args.base_checkpoint_dir,
            )
            print(f"🔍 Auto-discovered model path: {model_path}")
        except (ValueError, FileNotFoundError) as e:
            parser.error(f"Failed to discover model path: {e}")

    # Create output dir
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Load system prompt from config (explicit or auto-detected)
    system_prompt = None
    config_file_path = args.config_file

    # Auto-detect Hero config if not explicitly provided
    if not config_file_path:
        config_file_path = auto_detect_hero_config(
            model_name=args.model, model_path=model_path
        )
        if config_file_path:
            print(f"🔍 Auto-detected Hero config: {config_file_path}")

    # Load system prompt from config if available
    if config_file_path and Path(config_file_path).exists():
        with Path(config_file_path).open() as f:
            config = yaml.safe_load(f)
            system_prompt = config.get("system_prompt", None)
            if system_prompt:
                # Strip leading/trailing whitespace (YAML multi-line strings can have extra newlines)
                system_prompt = system_prompt.strip()
                print(f"✅ Loaded system prompt from {config_file_path}")
                print(f"   System prompt length: {len(system_prompt)} characters")
                print(f"   First 100 chars: {system_prompt[:100]}...")
            else:
                print(
                    f"⚠️  Warning: 'system_prompt' key not found in {config_file_path}"
                )
                print("   Using default system prompt")
    elif config_file_path:
        print(f"⚠️  Warning: Config file not found: {config_file_path}")
        print("   Using default system prompt")
    else:
        print("i  No config file provided, using default system prompt")

    canonical_system_prompt = system_prompt
    if args.prompt_variant != CANONICAL_VARIANT and canonical_system_prompt is None:
        parser.error("Non-canonical prompt variants require an existing config_file")
    if canonical_system_prompt is not None:
        try:
            system_prompt = apply_prompt_variant(
                canonical_system_prompt, args.prompt_variant
            )
        except PromptVariantError as exc:
            parser.error(str(exc))
        print(f"Prompt variant: {args.prompt_variant}")
        print(f"Canonical prompt SHA-256: {sha256_text(canonical_system_prompt)}")
        print(f"Effective prompt SHA-256: {sha256_text(system_prompt)}")

    if args.provenance_file:
        provenance_path = Path(args.provenance_file)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance = {
            "schema_version": 1,
            "model_path": model_path,
            "model_revision": args.model_revision,
            "dataset": args.dataset,
            "dataset_revision": args.dataset_revision,
            "dataset_split": args.dataset_split,
            "uuid_list": args.uuid_list,
            "uuid_list_sha256": (
                sha256_text(Path(args.uuid_list).read_text())
                if args.uuid_list
                else None
            ),
            "config_file": config_file_path,
            "config_file_sha256": (
                sha256_text(Path(config_file_path).read_text())
                if config_file_path and Path(config_file_path).exists()
                else None
            ),
            "prompt_variant": args.prompt_variant,
            "canonical_system_prompt_sha256": (
                sha256_text(canonical_system_prompt)
                if canonical_system_prompt is not None
                else None
            ),
            "effective_system_prompt_sha256": (
                sha256_text(system_prompt) if system_prompt is not None else None
            ),
            "temperature": args.temperature,
            "n_samples": args.n_samples,
            "max_samples": args.max_samples,
        }
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"Saved generation provenance to {provenance_path}")

    print(f"Loading vLLM Model: {model_path}")
    # Check if it's a local path (exists on filesystem) or HuggingFace identifier
    # vLLM supports both: local paths and HF identifiers like "Qwen/Qwen2.5-Coder-7B-Instruct"
    model_path_obj = Path(model_path)
    is_hf_identifier = "/" in model_path and not model_path_obj.exists()
    if not is_hf_identifier and not model_path_obj.exists():
        raise ModelPathNotFoundError(model_path)

    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 8192,
    }
    if args.model_revision:
        llm_kwargs["revision"] = args.model_revision
    llm = LLM(
        **llm_kwargs,
    )
    tokenizer = llm.get_tokenizer()

    print(f"Loading Dataset: {args.dataset}")
    try:
        ds = load_dataset(
            args.dataset, split=args.dataset_split, revision=args.dataset_revision
        )
    except Exception as e:
        raise DatasetLoadError(args.dataset, e) from e
    if args.uuid_list:
        requested = [
            line.strip()
            for line in Path(args.uuid_list).read_text().splitlines()
            if line.strip()
        ]
        if len(requested) != len(set(requested)):
            parser.error("--uuid_list contains duplicate UUIDs")
        by_uuid = {}
        for index, uuid in enumerate(ds["uuid"]):
            uuid = str(uuid)
            if uuid in by_uuid:
                parser.error(f"dataset split contains duplicate UUID: {uuid}")
            by_uuid[uuid] = index
        missing = [uuid for uuid in requested if uuid not in by_uuid]
        if missing:
            parser.error(f"--uuid_list UUIDs missing from split: {missing[:5]}")
        ds = ds.select([by_uuid[uuid] for uuid in requested])
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print("Formatting Prompts...")
    prompts = [format_prompt(item, tokenizer, system_prompt) for item in ds]

    print(f"Running Inference on {len(prompts)} samples...")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=4096,
        n=args.n_samples,
    )

    outputs = llm.generate(prompts, sampling_params)

    print(f"Saving to {args.output_file}...")
    with Path(args.output_file).open("w") as f:
        for ds_item, output in zip(ds, outputs, strict=False):
            for i, comp in enumerate(output.outputs):
                record = {
                    "uuid": ds_item.get("uuid"),
                    "mission": ds_item.get("mission"),  # Ground truth
                    "prompt": output.prompt,
                    "generated_text": comp.text,
                    "sample_idx": i,
                    "dataset_revision": args.dataset_revision,
                    "model_revision": args.model_revision,
                    "prompt_variant": args.prompt_variant,
                    "system_prompt_sha256": (
                        sha256_text(system_prompt) if system_prompt is not None else None
                    ),
                }
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
