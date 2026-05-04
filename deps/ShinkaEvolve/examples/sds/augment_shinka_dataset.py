#!/usr/bin/env python3
"""
Augment ShinkaEvolve SDS datasets from 100 to 1000 samples.

This script:
1. Downloads existing ShinkaEvolve-SDS-100-seed{xxx} datasets from HuggingFace
2. For each original sample, creates 10 variations:
   - Same code (copied from original)
   - Paraphrased thinking trace (using GPT-5-mini)
   - New problem prompt (generated using same logic as original dataset)
3. Pushes expanded datasets to both SoheylM and IDEALLab organizations
"""

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from huggingface_hub import HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HfApi = None
    HF_HUB_AVAILABLE = False

_MIN_MESSAGES_COUNT = 2

# Calculate paths: script is in deps/ShinkaEvolve/examples/sds/
# Project root is 4 levels up
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent.parent.parent

# Add project root to Python path for imports (like ShinkaEvolve tutorial)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import problem generation functions
try:
    from data.gen_sds_dataset import (
        _instance_to_problem,
        make_dense_instance,
        make_tree_showcase_instance,
        sds_render_prompt,
    )
except ImportError:
    gen_sds_path = project_root / "data" / "gen_sds_dataset.py"
    if not gen_sds_path.exists():
        print("❌ Could not import gen_sds_dataset.py.")
        print(f"   Script dir: {script_dir}")
        print(f"   Project root: {project_root}")
        print(f"   Looking for: {gen_sds_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("gen_sds_dataset", gen_sds_path)
    if spec is None or spec.loader is None:
        print("❌ Could not load gen_sds_dataset.py via importlib.")
        print(f"   Looking for: {gen_sds_path}")
        sys.exit(1)

    gen_sds_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_sds_module)
    _instance_to_problem = gen_sds_module._instance_to_problem
    make_dense_instance = gen_sds_module.make_dense_instance
    make_tree_showcase_instance = gen_sds_module.make_tree_showcase_instance
    sds_render_prompt = gen_sds_module.sds_render_prompt

# Import ShinkaEvolve utilities (already in path from project_root)
from shinka.annotation.trace_generator import TraceGenerator  # noqa: E402

# Try to import tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def get_deterministic_seed(*args) -> int:
    """
    Creates a unique 32-bit integer seed from any number of arguments.
    Ensures reproducibility across runs.
    """
    seed_str = "_".join(str(arg) for arg in args)
    hash_bytes = hashlib.sha256(seed_str.encode('utf-8')).digest()
    return int.from_bytes(hash_bytes[:4], byteorder='big')


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken if available, otherwise use approximation."""
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    return len(text) // 4


def extract_code_from_assistant(content: str) -> str:
    """Extract code from <code> block in assistant message."""
    code_match = re.search(r"<code>(.*?)</code>", content, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return ""


def extract_trace_from_assistant(content: str) -> str:
    """Extract thinking trace from assistant message (handles both <think> and <think> formats)."""
    # Try <think> first (format used in run_sds_pipeline.py)
    trace_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if trace_match:
        return trace_match.group(1).strip()
    # Fallback to <think> format (if dataset uses different format)
    trace_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if trace_match:
        return trace_match.group(1).strip()
    return ""


def paraphrase_trace(
    tracer: TraceGenerator,
    original_trace: str,
    problem_prompt: str,
    code: str,
    seed: int,  # noqa: ARG001
) -> str:
    """
    Paraphrase the original thinking trace using GPT-5-mini.
    
    Uses a custom system prompt that asks for paraphrasing while maintaining
    the same reasoning structure and correctness.
    """
    system_prompt = (
        "You are an expert algorithmist explaining a Python solution for a "
        "combinatorial optimization problem (SDS). Your task is to PARAPHRASE the "
        "given thinking trace, expressing the same reasoning and strategy in different words. "
        "Maintain the same logical flow and correctness, but use different phrasing, "
        "sentence structure, and word choices. Do NOT change the core reasoning or strategy. "
        "Output ONLY the paraphrased thinking trace content, without any XML tags or markdown formatting."
    )
    
    user_prompt = (
        f"ORIGINAL THINKING TRACE:\n{original_trace}\n\n"
        f"PROBLEM:\n{problem_prompt}\n\n"
        f"CODE:\n{code}\n\n"
        f"Paraphrase the thinking trace above, maintaining the same reasoning but using different words:"
    )
    
    # Note: OpenAI API calls are inherently stochastic, but we use the same prompt structure
    # for reproducibility. The seed parameter is not directly supported by gpt-5-mini,
    # but we include it in the prompt for consistency tracking.
    try:
        # Use the TraceGenerator's client but with custom prompt
        models_without_temp_support = ["gpt-5-mini", "gpt-5-nano"]
        kwargs = {
            "model": tracer.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        if tracer.model not in models_without_temp_support:
            kwargs["temperature"] = 0.7  # Higher temperature for more variation
        
        response = tracer.client.chat.completions.create(**kwargs)
        paraphrased = response.choices[0].message.content.strip()
        
        # Filter out failed generations
        if "Analysis failed." in paraphrased or not paraphrased:
            print("   ⚠️  Paraphrase generation failed, using original trace")
            return original_trace
        else:
            return paraphrased
    except Exception as e:
        print(f"   ⚠️  Error paraphrasing trace: {e}, using original trace")
        return original_trace


def generate_new_problem(
    original_type: str,
    original_index: int,
    variation_index: int,
    master_seed: int,
) -> dict[str, Any]:
    """
    Generate a new problem instance matching the original type (dense or tree).
    
    Uses deterministic seeding for reproducibility.
    """
    # Create deterministic seed for this variation
    problem_seed = get_deterministic_seed(master_seed, original_index, variation_index)
    rng = random.Random(problem_seed)
    
    # Generate problem size (matching original pipeline logic)
    n_size = rng.randint(15, 25)
    
    # Calculate cardinality bounds (matching original pipeline)
    card = (max(3, n_size//3), min(n_size, n_size//2 + 4))
    
    if original_type == "dense":
        # HARD: Weak unaries (2.0), Massive interactions (20.0)
        # Mixed signs (0.4/0.4) = Frustration (greedy fails)
        inst = make_dense_instance(
            n=n_size,
            card=card,
            weight_scale=2.0,
            pair_scale=20.0,
            pos_pair_frac=0.4,
            neg_pair_frac=0.4,
            seed=problem_seed
        )
    else:  # tree
        # HARD: Very weak unaries (1.0), Very strong interactions (25.0)
        inst = make_tree_showcase_instance(
            n=n_size,
            card=card,
            weight_scale=1.0,
            pair_scale=25.0,
            seed=problem_seed
        )
    
    # Convert to problem dict format
    problem_dict = _instance_to_problem(inst, original_type, original_index * 10 + variation_index)
    return problem_dict


def augment_single_sample(
    original_sample: dict[str, Any],
    original_index: int,
    master_seed: int,
    api_key: str,
    num_variations: int = 10,
) -> tuple[int, list[dict[str, Any]]]:
    """
    Create augmented variations for a single original sample.
    
    Args:
        original_sample: Original dataset sample
        original_index: Index of the original sample
        master_seed: Master seed for reproducibility
        api_key: OpenAI API key for creating tracer
        num_variations: Number of variations to create (default: 10)
    
    Returns:
        Tuple of (original_index, list of augmented samples)
    """
    # Create tracer per worker (thread-safe)
    tracer = TraceGenerator(api_key=api_key, model="gpt-5-mini")
    
    augmented_samples = []
    
    # Extract original data
    messages = original_sample.get("messages", [])
    if len(messages) < _MIN_MESSAGES_COUNT:
        print(f"   ⚠️  Sample {original_index} has invalid messages format, skipping")
        return (original_index, [])
    
    assistant_content = messages[1].get("content", "")
    original_code = extract_code_from_assistant(assistant_content)
    original_trace = extract_trace_from_assistant(assistant_content)
    
    if not original_code:
        print(f"   ⚠️  Sample {original_index} has no code block, skipping")
        return (original_index, [])
    
    if not original_trace:
        print(f"   ⚠️  Sample {original_index} has no thinking trace, skipping")
        return (original_index, [])
    
    original_score = original_sample.get("score", 0.0)
    original_type = original_sample.get("type", "dense")
    
    # Generate variations
    for var_idx in range(num_variations):
        # Generate new problem
        new_problem_dict = generate_new_problem(
            original_type=original_type,
            original_index=original_index,
            variation_index=var_idx,
            master_seed=master_seed,
        )
        
        # Render new problem prompt
        prompt_data = sds_render_prompt(new_problem_dict)
        new_problem_prompt = prompt_data["problem"]
        
        # Paraphrase thinking trace
        # Note: We use the master_seed for trace generation, but OpenAI API is stochastic
        # The seed is included for tracking purposes
        paraphrased_trace = paraphrase_trace(
            tracer=tracer,
            original_trace=original_trace,
            problem_prompt=new_problem_prompt,
            code=original_code,
            seed=get_deterministic_seed(master_seed, original_index, var_idx, "trace"),
        )
        
        # Format new assistant content (matching original format from run_sds_pipeline.py)
        new_assistant_content = (
            f"<think>\n{paraphrased_trace}\n</think>\n\n"
            f"<code>\n{original_code}\n</code>"
        )
        
        # Calculate token counts
        user_tokens = count_tokens(new_problem_prompt)
        assistant_tokens = count_tokens(new_assistant_content)
        total_tokens = user_tokens + assistant_tokens
        
        # Create augmented sample
        augmented_sample = {
            "problem_id": prompt_data.get("uuid", f"problem_{original_index}_{var_idx}"),
            "messages": [
                {"role": "user", "content": new_problem_prompt},
                {"role": "assistant", "content": new_assistant_content}
            ],
            "score": float(original_score),  # Same score (same code)
            "type": original_type,  # Same type
            "mission": json.dumps(new_problem_dict.get("mission", {})),  # New mission
            "index": original_index * 10 + var_idx,  # New index
            "num_tokens": total_tokens,
            "num_tokens_user": user_tokens,
            "num_tokens_assistant": assistant_tokens,
        }
        
        augmented_samples.append(augmented_sample)
    
    return (original_index, augmented_samples)


def main():  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(
        description="Augment ShinkaEvolve SDS datasets from 100 to 1000 samples"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[101, 202, 303],
        help="Seeds to process (default: 101 202 303)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API Key (optional if in env)"
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=10,
        help="Number of variations per original sample (default: 10)"
    )
    parser.add_argument(
        "--skip_push",
        action="store_true",
        help="Skip pushing to HuggingFace (only generate locally)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for augmentation. "
             "If not specified, defaults to number of CPU cores. "
             "Set to 1 for sequential processing."
    )
    parser.add_argument(
        "--source-prefix",
        type=str,
        default="SoheylM/ShinkaEvolve-SDS-100",
        help="Source HF dataset prefix without -seed suffix "
             "(default: SoheylM/ShinkaEvolve-SDS-100)"
    )
    parser.add_argument(
        "--target-prefix",
        type=str,
        default="SoheylM/ShinkaEvolve-SDS-1000",
        help="Target HF dataset prefix without -seed suffix "
             "(default: SoheylM/ShinkaEvolve-SDS-1000)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for local augmented backups (default: script dir)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables from .env file in project root
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        # Fallback: try to find .env in current directory or parent directories
        load_dotenv()
    
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API Key required for trace paraphrasing")  # noqa: TRY003
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and not args.skip_push:
        print("⚠️  HF_TOKEN not found. Use --skip_push to skip uploading, or set HF_TOKEN")
        return
    
    # Determine number of workers
    if args.workers is None:
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = max(1, args.workers)
    
    backup_root = Path(args.output_dir) if args.output_dir else script_dir
    backup_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print(f"\n{'='*80}")
        print(f"Processing seed {seed}")
        print(f"{'='*80}")

        # Download original dataset
        original_dataset_name = f"{args.source_prefix}-seed{seed}"
        print(f"📥 Downloading {original_dataset_name}...")
        try:
            original_dataset = load_dataset(original_dataset_name, split="train")
        except Exception as e:
            print(f"❌ Failed to download {original_dataset_name}: {e}")
            continue
        
        print(f"✅ Downloaded {len(original_dataset)} samples")
        
        # Augment samples
        print(f"🔄 Augmenting samples (creating {args.num_variations} variations each)...")
        
        # Convert dataset to list for parallel processing
        original_samples_list = list(original_dataset)
        
        if num_workers == 1:
            # Sequential processing (easier debugging)
            print("🔄 Running in sequential mode (workers=1)")
            augmented_samples = []
            for i, original_sample in enumerate(tqdm(original_samples_list, desc="Augmenting")):
                _, variations = augment_single_sample(
                    original_sample=original_sample,
                    original_index=i,
                    master_seed=seed,
                    api_key=api_key,
                    num_variations=args.num_variations,
                )
                augmented_samples.extend(variations)
        else:
            # Parallel processing
            print(f"⚡ Running in parallel mode ({num_workers} workers)")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all samples
                future_to_index = {
                    executor.submit(
                        augment_single_sample,
                        original_sample=original_samples_list[i],
                        original_index=i,
                        master_seed=seed,
                        api_key=api_key,
                        num_variations=args.num_variations,
                    ): i
                    for i in range(len(original_samples_list))
                }
                
                # Collect results as they complete (with progress bar)
                completed_results = {}
                with tqdm(total=len(original_samples_list), desc="Augmenting") as pbar:
                    for future in as_completed(future_to_index):
                        i = future_to_index[future]
                        try:
                            orig_idx, variations = future.result()
                            if variations:
                                completed_results[orig_idx] = variations
                        except Exception as e:
                            print(f"   ❌ Sample {i} failed with exception: {e}")
                        finally:
                            pbar.update(1)
                
                # Sort by original index to maintain deterministic order
                augmented_samples = []
                for orig_idx in sorted(completed_results.keys()):
                    augmented_samples.extend(completed_results[orig_idx])
        
        print(f"✅ Generated {len(augmented_samples)} augmented samples")
        
        if len(augmented_samples) == 0:
            print(f"⚠️  No samples generated for seed {seed}, skipping")
            continue
        
        # Create dataset
        augmented_dataset = Dataset.from_list(augmented_samples)
        
        # Save locally as temporary backup (save in script directory)
        output_file = backup_root / f"shinka_augmented_seed{seed}.jsonl"
        print(f"💾 Saving local backup to {output_file}...")
        with output_file.open("w") as f:
            f.writelines(json.dumps(sample) + "\n" for sample in augmented_samples)
        print(f"✅ Saved {len(augmented_samples)} samples to {output_file}")

        run_config = {
            "dataset_variant": "shinka-sds-1000-v2" if "-v2" in args.target_prefix else "shinka-sds-1000",
            "seed": seed,
            "source_dataset": original_dataset_name,
            "target_dataset": f"{args.target_prefix}-seed{seed}",
            "num_source_samples": len(original_dataset),
            "num_variations": args.num_variations,
            "num_output_samples": len(augmented_samples),
            "trace_model": "gpt-5-mini",
            "output_backup_file": str(output_file),
        }
        run_config_file = backup_root / f"shinka_augmented_seed{seed}_run_config.json"
        with run_config_file.open("w") as f:
            json.dump(run_config, f, indent=2)
        print(f"💾 Saved augmentation config to {run_config_file}")
        
        # Push to HuggingFace
        push_successful = True
        if not args.skip_push:
            repos_to_push = [f"{args.target_prefix}-seed{seed}"]
            
            for repo_name in repos_to_push:
                print(f"📤 Pushing to {repo_name}...")
                try:
                    augmented_dataset.push_to_hub(
                        repo_name,
                        token=hf_token,
                        private=True,
                    )
                    print(f"✅ Successfully pushed to {repo_name}")
                    if HF_HUB_AVAILABLE:
                        api = HfApi(token=hf_token)
                        api.upload_file(
                            path_or_fileobj=str(run_config_file),
                            path_in_repo="run_config.json",
                            repo_id=repo_name,
                            repo_type="dataset",
                            commit_message=f"Add augmentation config for seed {seed}"
                        )
                        print(f"✅ Successfully uploaded run_config.json to {repo_name}")
                except Exception as e:
                    print(f"❌ Error pushing to {repo_name}: {e}")
                    push_successful = False
        else:
            print("⏭️  Skipping push to HuggingFace (--skip_push)")
            push_successful = False  # Keep file if skip_push is used
        
        # Clean up local file after successful push
        if push_successful and not args.skip_push and output_file.exists():
            output_file.unlink()
            print(f"🗑️  Cleaned up local backup file: {output_file}")
    
    print(f"\n{'='*80}")
    print("✅ Augmentation complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
