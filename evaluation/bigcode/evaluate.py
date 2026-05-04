#!/usr/bin/env python3
"""
Thin wrapper around BigCode evaluation harness with W&B logging.

This script:
1. Calls the BigCode harness (via accelerate launch)
2. Reads output metrics_*.json files
3. Logs results to W&B (similar to SDS evaluate.py)
4. Saves experiment_metadata.json for aggregation
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None


def normalize_model_name(model: str) -> str:
    """Normalize model name for W&B run naming."""
    return model.lower().replace("/", "-").replace("_", "-").replace(".", "")


def _infer_eval_tag_from_env_or_path() -> str | None:  # noqa: PLR0911
    """Infer method tag from environment or checkpoint path."""
    tag = os.environ.get("WANDB_ABLATION_TAG", "").strip()
    if tag:
        return tag
    ckpt = os.environ.get("CHECKPOINT_DIR", "").lower()
    if "config_hero" in ckpt:
        return "hero"
    if "config_ablation_oracle" in ckpt:
        return "oracle"
    if "config_ablation_diversity" in ckpt:
        return "diversity"
    if "config_minimalist" in ckpt:
        return "minimalist"
    if "config_ablation_prompt" in ckpt:
        return "prompt"
    if "config_ablation_generalization" in ckpt:
        return "gen"
    return None


def construct_wandb_run_name(
    model: str,
    seed: int | None,
    job_id: str | None,
    eval_tag: str | None = None,
) -> str:
    """Construct W&B run name for BigCode evaluation."""
    normalized_model = normalize_model_name(model)
    seed_str = f"seed{seed}" if seed is not None else "seed42"
    job_str = f"job{job_id}" if job_id else ""
    base = "grpo"

    if eval_tag and job_str:
        return f"{normalized_model}-{base}-{eval_tag}-bigcode-{seed_str}-{job_str}-eval"
    elif eval_tag:
        return f"{normalized_model}-{base}-{eval_tag}-bigcode-{seed_str}-eval"
    elif job_str:
        return f"{normalized_model}-{base}-bigcode-{seed_str}-{job_str}-eval"
    else:
        return f"{normalized_model}-{base}-bigcode-{seed_str}-eval"


def method_from_config_name(cfg: str) -> str:
    """Map config name to method name."""
    name = cfg.replace("config_", "").replace("_", "-")
    method_map = {
        "hero": "Ours (Hero)",
        "ablation-oracle": "Ours (+Oracle)",
        "ablation-diversity": "Ours (+Diversity)",
        "minimalist": "Ours (w/o Structure)",
        "ablation-generalization": "Ours (+Generalization)",
        "ablation-prompt": "Ours (w/o Prompt)",
    }
    return method_map.get(name, f"Ours ({name})")


def save_experiment_metadata(
    output_dir: str,
    model: str,
    seed: int,
    job_id: str | None,
    training_scheme: str | None = None,
) -> dict:
    """Save experiment_metadata.json for aggregation."""
    config_name = None
    method_name = None

    if training_scheme:
        parts = training_scheme.lower().replace("_", "-").split("-")
        for i, part in enumerate(parts):
            if part.startswith("config"):
                config_name = "-".join(parts[i:]).replace("-", "_")
                break
        if config_name:
            method_name = method_from_config_name(config_name)

    if not config_name:
        ablation_tag = os.environ.get("ABLATION_TAG", "").strip()
        if ablation_tag:
            config_name = ablation_tag
            method_name = method_from_config_name(config_name)

    if (
        not method_name
        and (os.environ.get("IS_HF_MODEL", "false") or "").lower() == "true"
    ):
        method_name = "Base"
        config_name = "base"

    metadata = {
        "model": model,
        "training_scheme": training_scheme or "",
        "seed": seed,
        "job_id": job_id or "",
        "config_name": config_name,
        "method_name": method_name,
    }

    metadata_path = Path(output_dir) / "experiment_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"📝 Saved experiment metadata to {metadata_path}")
    return metadata


def load_metrics(output_dir: str, tasks: list[str]) -> dict:
    """Load metrics from all task JSON files."""
    all_metrics = {}
    output_path = Path(output_dir)
    for task in tasks:
        metrics_path = output_path / f"metrics_{task}.json"
        if metrics_path.exists():
            with metrics_path.open() as f:
                task_metrics = json.load(f)
                all_metrics.update(task_metrics)
        else:
            print(f"⚠️  Metrics file not found: {metrics_path}")
    return all_metrics


def log_to_wandb(  # noqa: PLR0912, PLR0913
    output_dir: str,
    model: str,
    seed: int,
    job_id: str | None,
    tasks: list[str],
    metadata: dict,
):
    """Log BigCode evaluation results to W&B."""
    if wandb is None:
        print("⚠️  wandb not available. Install with: pip install wandb")
        return

    if not os.environ.get("WANDB_API_KEY"):
        print("⚠️  WANDB_API_KEY not found. Skipping W&B logging.")
        return

    eval_tag = _infer_eval_tag_from_env_or_path()
    run_name = construct_wandb_run_name(model, seed, job_id, eval_tag)

    project = os.environ.get("WANDB_PROJECT", "qwen-coder-bigcode-rl")
    entity = os.environ.get("WANDB_ENTITY", "smassoudi-eth-z-rich")
    batch_id = os.environ.get("BATCH_ID", None)

    print("\n📊 Logging evaluation results to W&B...")
    print(f"   Run name: {run_name}")
    print(f"   Project: {project}")
    print(f"   Entity: {entity}")

    try:
        wandb.init(
            name=run_name,
            project=project,
            entity=entity,
            resume="allow",
            job_type="evaluation",
            tags=["evaluation", "bigcode"],
            group=batch_id,
        )

        # Log config
        wandb.config.update(
            {
                "batch_id": batch_id,
                "eval_tag": eval_tag,
                "job_id": job_id,
                "seed": seed,
                "model": model,
                "training_scheme": metadata.get("training_scheme"),
                "output_dir": output_dir,
            },
            allow_val_change=True,
        )

        # Load and log metrics
        metrics = load_metrics(output_dir, tasks)
        if metrics:
            summary_metrics = {}
            for task in tasks:
                task_data = metrics.get(task, {})
                if isinstance(task_data, dict):
                    # Extract pass@1, pass@10, etc.
                    for key, value in task_data.items():
                        if isinstance(value, int | float):
                            summary_metrics[f"bigcode/{task}/{key}"] = value

            if summary_metrics:
                wandb.summary.update(summary_metrics)
                print(f"   ✅ Logged {len(summary_metrics)} summary metrics")

        # Log artifacts (metrics JSON files)
        artifact = wandb.Artifact(
            name=f"bigcode-metrics-{run_name}",
            type="evaluation-results",
        )

        output_path = Path(output_dir)
        for task in tasks:
            metrics_path = output_path / f"metrics_{task}.json"
            if metrics_path.exists():
                artifact.add_file(str(metrics_path))
                print(f"   ✅ Added metrics_{task}.json to artifact")

            generations_path = output_path / f"generations_{task}.json"
            if generations_path.exists():
                artifact.add_file(str(generations_path))
                print(f"   ✅ Added generations_{task}.json to artifact")

        if artifact.manifest.entries:
            wandb.log_artifact(artifact)
            print("   ✅ Logged evaluation artifacts")

        wandb.finish()
        print(f"   ✅ Successfully logged to W&B run: {run_name}")

    except Exception as e:
        print(f"   ❌ Error logging to W&B: {e}")
        print(f"   Evaluation results are still saved locally in: {output_dir}")


def main():  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="BigCode evaluation wrapper with W&B logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # BigCode harness arguments (passed through)
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help="Tasks to evaluate (e.g., humaneval mbpp)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--n_samples", type=int, default=1, help="Number of samples per problem"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling (default: greedy decoding)",
    )
    parser.add_argument(
        "--max_length_generation", type=int, default=2048, help="Max generation length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (must be 1 for greedy decoding)",
    )
    parser.add_argument(
        "--precision", type=str, default="bf16", help="Precision (bf16, fp16, fp32)"
    )
    parser.add_argument(
        "--allow_code_execution", action="store_true", help="Allow code execution"
    )
    parser.add_argument(
        "--save_generations", action="store_true", help="Save generations to file"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for metrics and generations",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--job-id", type=str, default=None, help="Job ID for fine-tuned models"
    )
    parser.add_argument(
        "--training-scheme",
        type=str,
        default=None,
        help="Training scheme (for metadata)",
    )

    # W&B arguments
    parser.add_argument(
        "--log-to-wandb", action="store_true", help="Log results to Weights & Biases"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")

    # Run harness for each task (BigCode harness processes one task at a time)
    print(f"\n🚀 Running BigCode harness for tasks: {', '.join(args.tasks)}")
    for task in args.tasks:
        print(f"\n{'=' * 60}")
        print(f"📊 Evaluating task: {task}")
        print(f"{'=' * 60}")

        # Construct harness command for this task
        harness_cmd = [
            "accelerate",
            "launch",
            "deps/bigcode-evaluation-harness/main.py",
            "--model",
            args.model,
            "--tasks",
            task,  # Single task per call
            "--n_samples",
            str(args.n_samples),
            "--max_length_generation",
            str(args.max_length_generation),
            "--batch_size",
            str(args.batch_size),
            "--precision",
            args.precision,
            "--save_generations_path",
            str(Path(args.output_dir) / f"generations_{task}.json"),
            "--metric_output_path",
            str(Path(args.output_dir) / f"metrics_{task}.json"),
            "--use_auth_token",
        ]

        # Handle temperature: if 0.0, use greedy decoding (don't pass temperature)
        # Otherwise, pass temperature and enable sampling
        if args.temperature > 0.0:
            harness_cmd.extend(["--temperature", str(args.temperature)])
            if args.do_sample:
                harness_cmd.append("--do_sample")
        else:
            # Greedy decoding: explicitly set do_sample=False (or omit, as False is default)
            # Don't pass --temperature when it's 0.0 (transformers rejects it)
            pass  # Default is greedy (do_sample=False)
        if args.allow_code_execution:
            harness_cmd.append("--allow_code_execution")
        if args.save_generations:
            harness_cmd.append("--save_generations")

        try:
            subprocess.run(
                harness_cmd,
                check=True,
                capture_output=False,  # Show output in real-time
            )
            print(f"✅ Task {task} complete!")
            output_path = Path(args.output_dir)
            print(f"   Generations: {output_path / f'generations_{task}.json'}")
            print(f"   Metrics:     {output_path / f'metrics_{task}.json'}")
        except subprocess.CalledProcessError:
            print(f"❌ ERROR: Harness failed for task {task}")
            print(f"   Command: {' '.join(harness_cmd)}")
            sys.exit(1)

    print(f"\n✅ Pipeline complete! All artifacts saved in: {args.output_dir}")

    # Save experiment metadata
    metadata = save_experiment_metadata(
        output_dir=args.output_dir,
        model=args.model,
        seed=args.seed,
        job_id=args.job_id,
        training_scheme=args.training_scheme,
    )

    # Log to W&B if requested
    if args.log_to_wandb:
        log_to_wandb(
            output_dir=args.output_dir,
            model=args.model,
            seed=args.seed,
            job_id=args.job_id,
            tasks=args.tasks,
            metadata=metadata,
        )

    # Print summary
    print("\n📊 Summary of Pass@1 scores:")
    metrics = load_metrics(args.output_dir, args.tasks)
    for task in args.tasks:
        task_data = metrics.get(task, {})
        if isinstance(task_data, dict):
            pass_at_1 = task_data.get("pass@1", "N/A")
            print(f"   {task}: Pass@1 = {pass_at_1}")


if __name__ == "__main__":
    main()
