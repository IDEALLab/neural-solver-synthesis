#!/usr/bin/env python3
"""Generate code samples for the mixed text-suite (SDS + JSSP + CVRP)."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import yaml
from datasets import load_dataset, load_from_disk

# Allow running as a script from arbitrary working directories.
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

def extract_code_block(text: str) -> str:
    """Extract code inside the last <code> block, or return stripped text."""
    if not isinstance(text, str):
        return ""
    matches = re.findall(r"<code>(.*?)</code>", text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def _format_prompt(problem_text: str, tokenizer, system_prompt: str | None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": problem_text})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mixed-suite code candidates with vLLM.")
    parser.add_argument("--model_path", required=True, help="Model path or HF ID")
    parser.add_argument(
        "--dataset",
        default="SoheylM/OpenR1-TextProblems-10k-seed101",
        help="HF dataset ID for mixed text problems",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--dataset_revision", default=None, help="Immutable HF dataset revision")
    parser.add_argument("--output_file", required=True, help="Output JSONL path")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable vLLM CUDA-graph/torch.compile execution without changing decoding.",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--model_revision", default=None, help="Immutable HF model revision")
    parser.add_argument(
        "--provenance_model_revision",
        default="",
        help="Recorded model identity for local checkpoints; defaults to model_revision.",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="Optional sample cap for smoke runs")
    parser.add_argument("--system_prompt", default="", help="Optional system prompt override")
    parser.add_argument("--config_file", default="", help="Optional YAML config containing system_prompt")
    parser.add_argument("--indices_manifest", default="", help="Optional JSON manifest containing row indices")
    parser.add_argument("--indices_key", default="synthesis_indices", help="Manifest field containing row indices")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # Local import so unit tests can import this module without vLLM installed.
    from vllm import LLM, SamplingParams

    dataset_path = Path(args.dataset)
    if dataset_path.is_dir():
        manifest_path = dataset_path / "n26_dataset_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"local dataset manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("dataset_revision") != args.dataset_revision:
            raise ValueError("local dataset revision does not match the pinned revision")
        dataset = load_from_disk(str(dataset_path))[args.split]
    else:
        dataset = load_dataset(args.dataset, split=args.split, revision=args.dataset_revision)
    if args.indices_manifest:
        manifest = json.loads(Path(args.indices_manifest).read_text())
        indices = [int(value) for value in manifest[args.indices_key]]
        if len(indices) != len(set(indices)):
            raise ValueError("indices manifest contains duplicates")
        dataset = dataset.select(indices)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    system_prompt = args.system_prompt or None
    if args.config_file:
        config = yaml.safe_load(Path(args.config_file).read_text())
        config_prompt = config.get("system_prompt")
        if not isinstance(config_prompt, str) or not config_prompt.strip():
            raise ValueError("config_file does not contain a non-empty system_prompt")
        if system_prompt is not None:
            raise ValueError("use either --system_prompt or --config_file, not both")
        system_prompt = config_prompt.strip()

    llm_kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    if args.model_revision:
        llm_kwargs["revision"] = args.model_revision
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    prompts = [_format_prompt(row["problem"], tokenizer, system_prompt) for row in dataset]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        n=args.n_samples,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as handle:
        for prompt_idx, (row, output) in enumerate(zip(dataset, outputs, strict=False)):
            mission = row.get("mission")
            for sample_idx, candidate in enumerate(output.outputs):
                completion = candidate.text
                generated_code = extract_code_block(completion)
                record = {
                    "dataset": args.dataset,
                    "dataset_revision": args.dataset_revision,
                    "model_revision": args.provenance_model_revision or args.model_revision,
                    "split": args.split,
                    "uuid": row.get("uuid"),
                    "domain": row.get("domain", "sds"),
                    "mission": mission,
                    "prompt_id": prompt_idx,
                    "sample_idx": sample_idx,
                    "problem": row.get("problem", ""),
                    "completion": completion,
                    "generated_code": generated_code,
                    "temperature": args.temperature,
                    "seed": args.seed,
                }
                handle.write(json.dumps(record) + "\n")

    print(f"Saved {len(dataset) * args.n_samples} generations to {output_path}")


if __name__ == "__main__":
    main()
