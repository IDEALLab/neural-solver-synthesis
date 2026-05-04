"""
Pipeline for generating high-quality SFT datasets for SDS problems using ShinkaEvolve.

This script:
1. Generates SDS problem instances
2. Runs ShinkaEvolve evolution for each problem
3. Extracts best solutions from the database
4. Generates reasoning traces
5. Formats data for student model training
"""

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

_MIN_FITNESS_THRESHOLD = 0.01
_KILO_THRESHOLD = 1000

import tqdm  # noqa: E402
from datasets import Dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Try to import huggingface_hub for config file upload
try:
    from huggingface_hub import HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    HfApi = None

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Calculate paths: script is in deps/ShinkaEvolve/examples/sds/
# Project root is 4 levels up
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent.parent.parent

# Add project root to Python path for imports (like ShinkaEvolve tutorial)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ShinkaEvolve imports
from shinka.annotation.trace_generator import TraceGenerator  # noqa: E402
from shinka.core import EvolutionConfig, EvolutionRunner  # noqa: E402
from shinka.database import DatabaseConfig, ProgramDatabase  # noqa: E402
from shinka.launch import LocalJobConfig  # noqa: E402

# Import problem generation from data directory
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


def create_sds_evolution_config(  # noqa: PLR0913
    problem_data: dict,  # noqa: ARG001
    num_generations: int = 10,
    results_dir: str | None = None,
    script_dir: Path | None = None,
    max_parallel_jobs: int = 1,
    max_patch_attempts: int = 3,
    max_patch_resamples: int = 3,
    patch_types: list | None = None,
    patch_type_probs: list | None = None,
    llm_models: list | None = None,
    llm_temperatures: list | None = None,
    llm_max_tokens: int = 8192,
) -> EvolutionConfig:
    """Create EvolutionConfig for SDS problem."""

    search_task_sys_msg = """You are an expert algorithmist specializing in combinatorial optimization, particularly the Synergistic Dependency Selection (SDS) problem.

The SDS problem requires selecting a subset of variables that:
1. Satisfies cardinality bounds (min/max number of items)
2. Respects mutex constraints (cannot select both items in a mutex pair)
3. Respects group constraints (at most one item per group)
4. Satisfies precedence constraints (if j is selected, i must be selected)
5. Maximizes the objective: sum of selected item weights + pairwise interaction values

The code should read JSON from stdin with "requirements" and "catalog", and output JSON to stdout with "selection" containing "variables" list.

Deceptive interactions can make purely greedy orderings unreliable, so prefer robust search procedures that still respect the execution-time budget.

Be creative and find efficient solutions that maximize the objective while respecting all constraints. The code must execute very quickly (within seconds) - prioritize time-efficient algorithms and avoid computationally expensive operations.
"""
    
    # Set defaults if not provided
    if patch_types is None:
        patch_types = ["diff", "full", "cross"]
    if patch_type_probs is None:
        patch_type_probs = [0.6, 0.3, 0.1]
    if llm_models is None:
        llm_models = ["gpt-5-nano", "gpt-5-mini"]
    if llm_temperatures is None:
        llm_temperatures = [0.0, 0.5, 0.7]
    
    return EvolutionConfig(
        task_sys_msg=search_task_sys_msg,
        patch_types=patch_types,
        patch_type_probs=patch_type_probs,
        num_generations=num_generations,
        max_parallel_jobs=max_parallel_jobs,
        max_patch_resamples=max_patch_resamples,
        max_patch_attempts=max_patch_attempts,
        job_type="local",
        language="python",
        llm_models=llm_models,
          llm_kwargs={
            "temperatures": llm_temperatures,
            "max_tokens": llm_max_tokens,
        },
        embedding_model="text-embedding-3-small",
        init_program_path=str(script_dir / "initial.py") if script_dir else "initial.py",
        results_dir=results_dir,
    )


def create_sds_database_config(  # noqa: PLR0913
    num_islands: int = 2,
    archive_size: int = 20,
    parent_selection_strategy: str = "weighted",
    parent_selection_lambda: float = 10.0,
    elite_selection_ratio: float = 0.3,
    num_archive_inspirations: int = 4,
    num_top_k_inspirations: int = 2,
    migration_interval: int = 10,
    migration_rate: float = 0.1,
    island_elitism: bool = True,
) -> DatabaseConfig:
    """Create DatabaseConfig for SDS evolution."""
    return DatabaseConfig(
        db_path="evolution_db.sqlite",
        num_islands=num_islands,
        archive_size=archive_size,
        elite_selection_ratio=elite_selection_ratio,
        num_archive_inspirations=num_archive_inspirations,
        num_top_k_inspirations=num_top_k_inspirations,
        migration_interval=migration_interval,
        migration_rate=migration_rate,
        island_elitism=island_elitism,
        parent_selection_strategy=parent_selection_strategy,
        parent_selection_lambda=parent_selection_lambda,
    )


def get_deterministic_seed(*args):
    """
    Creates a unique 32-bit integer seed from any number of arguments.
    Ensures no overlap between runs (e.g., Seed 101 vs 202).
    
    This matches the implementation in rewards_unified_v2.py for consistency.
    
    Args:
        *args: Any number of arguments to combine into a seed
        
    Returns:
        int: 32-bit integer seed
    """
    # Combine all args into a single unique string
    seed_str = "_".join(str(arg) for arg in args)
    # Hash it (SHA256 is robust enough to prevent collision)
    hash_bytes = hashlib.sha256(seed_str.encode('utf-8')).digest()
    # Convert to integer and clip to 32-bit (standard for numpy/random)
    return int.from_bytes(hash_bytes[:4], byteorder='big')


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken if available, otherwise use approximation.
    
    Args:
        text: Text to count tokens for
    
    Returns:
        Approximate token count
    """
    if TIKTOKEN_AVAILABLE:
        try:
            # Use cl100k_base encoding (GPT-4, GPT-3.5-turbo)
            # This is a reasonable approximation for most modern tokenizers
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fall back to approximation if encoding fails
            pass
    
    # Simple approximation: ~4 characters per token (common for English)
    # This is a rough estimate but better than nothing
    return len(text) // 4


def process_single_problem(  # noqa: PLR0913, PLR0915
    i: int,
    master_seed: int,
    num_generations: int,
    script_dir: Path,
    output_dir: Path,
    api_key: str,
    # Evolution config parameters
    max_parallel_jobs: int = 1,
    max_patch_attempts: int = 3,
    max_patch_resamples: int = 3,
    patch_types: list | None = None,
    patch_type_probs: list | None = None,
    llm_models: list | None = None,
    llm_temperatures: list | None = None,
    llm_max_tokens: int = 8192,
    # Database config parameters
    num_islands: int = 2,
    archive_size: int = 20,
    parent_selection_strategy: str = "weighted",
    parent_selection_lambda: float = 10.0,
    elite_selection_ratio: float = 0.3,
    num_archive_inspirations: int = 4,
    num_top_k_inspirations: int = 2,
    migration_interval: int = 10,
    migration_rate: float = 0.1,
    island_elitism: bool = True,
) -> dict:
    """
    Process a single SDS problem through ShinkaEvolve evolution.
    
    This function is designed to be called in parallel across multiple problems.
    Each problem gets its own isolated environment (results directory, database, etc.).
    
    Args:
        i: Problem index (0-based)
        master_seed: Master seed for reproducibility
        num_generations: Number of evolution generations
        script_dir: Path to script directory (for initial.py, evaluate.py)
        output_dir: Base output directory
        api_key: OpenAI API key for trace generation
    
    Returns:
        dict: Dataset record if successful, None if failed
    """
    try:
        # Calculate deterministic seed for this problem using hash-based seeding
        # This ensures:
        # 1. Problem i always gets the same seed regardless of execution order
        # 2. No overlap between different master seeds (e.g., seed 101 problem 102 vs seed 202 problem 1)
        # 3. Matches the approach used in generalization reward for consistency
        if master_seed is not None:
            problem_gen_seed = get_deterministic_seed(master_seed, i)
        else:
            problem_gen_seed = i
        
        # Use the seed to create a deterministic RNG for problem size selection
        rng_for_size = random.Random(problem_gen_seed)
        
        # A. Generate a Problem Instance (deterministic when seed is set)
        # Use same HARD parameters as GRPO dataset generation for consistency
        ptype = "dense" if i % 2 == 0 else "tree"
        n_size = rng_for_size.randint(15, 25)  # Deterministic when seed is set
        
        # Calculate cardinality bounds (matching GRPO dataset generation)
        if ptype == "dense":
            card = (max(3, n_size//3), min(n_size, n_size//2 + 4))
            # HARD: Weak unaries (2.0), Massive interactions (20.0)
            # Mixed signs (0.4/0.4) = Frustration (greedy fails)
            inst = make_dense_instance(
                n=n_size,
                card=card,
                weight_scale=2.0,    # Weak unaries (default is 8.0)
                pair_scale=20.0,    # Strong interactions (default is 6.0)
                pos_pair_frac=0.4,  # Mixed signs = Frustration
                neg_pair_frac=0.4,
                seed=problem_gen_seed
            )
        else:  # tree
            card = (max(3, n_size//3), min(n_size, n_size//2 + 4))
            # HARD: Very weak unaries (1.0), Very strong interactions (25.0)
            inst = make_tree_showcase_instance(
                n=n_size,
                card=card,
                weight_scale=1.0,   # Very weak unaries
                pair_scale=25.0,    # Very strong interactions
                seed=problem_gen_seed
            )

        problem_dict = _instance_to_problem(inst, ptype, i)
        prompt_data = sds_render_prompt(problem_dict)
        
        # B. Set up evolution for this problem
        # Create unique results directory for this problem
        problem_results_dir = output_dir / f"problem_{i}"
        problem_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variable for evaluate.py to use (isolated per worker)
        os.environ["SDS_PROBLEM_DATA"] = json.dumps(problem_dict)
        
        # Create configs with absolute paths
        # Ensure results_dir is absolute so it works regardless of working directory
        evo_config = create_sds_evolution_config(
            problem_dict,
            num_generations=num_generations,
            results_dir=str(problem_results_dir.resolve()),
            script_dir=script_dir,
            max_parallel_jobs=max_parallel_jobs,
            max_patch_attempts=max_patch_attempts,
            max_patch_resamples=max_patch_resamples,
            patch_types=patch_types,
            patch_type_probs=patch_type_probs,
            llm_models=llm_models,
            llm_temperatures=llm_temperatures,
            llm_max_tokens=llm_max_tokens,
        )
        db_config = create_sds_database_config(
            num_islands=num_islands,
            archive_size=archive_size,
            parent_selection_strategy=parent_selection_strategy,
            parent_selection_lambda=parent_selection_lambda,
            elite_selection_ratio=elite_selection_ratio,
            num_archive_inspirations=num_archive_inspirations,
            num_top_k_inspirations=num_top_k_inspirations,
            migration_interval=migration_interval,
            migration_rate=migration_rate,
            island_elitism=island_elitism,
        )
        # EvolutionRunner will prepend results_dir, so just use filename
        db_config.db_path = "evolution_db.sqlite"
        
        job_config = LocalJobConfig(
            eval_program_path=str(script_dir / "evaluate.py")
        )
        
        # C. Run Shinka Evolution
        # Change to script directory so relative paths in evaluate.py work correctly
        # Note: Evolution uses natural randomness to explore diverse solutions
        # and avoid mode collapse. Problem generation is seeded for reproducibility.
        original_cwd = Path.cwd()
        try:
            os.chdir(script_dir)  # Change to script directory for evolution
            evo_runner = EvolutionRunner(
                evo_config=evo_config,
                job_config=job_config,
                db_config=db_config,
                verbose=False,  # Less verbose for batch processing
            )
            evo_runner.run()
        except Exception as e:
            print(f"   ⚠️ Evolution failed for problem {i}: {e}")
            return None
        finally:
            os.chdir(original_cwd)  # Restore original working directory
        
        # D. Extract best solution from database
        # EvolutionRunner creates db at results_dir/db_path (see runner.py line 140)
        # Since results_dir is absolute and db_config.db_path is "evolution_db.sqlite",
        # the final path is: results_dir / "evolution_db.sqlite"
        db_path = problem_results_dir.resolve() / "evolution_db.sqlite"
        
        # Verify the path exists (EvolutionRunner should have created it)
        if not db_path.exists():
            print(f"   ⚠️ Database not found at {db_path} for problem {i}")
            return None
        
        best_code, best_fitness = extract_best_code_from_database(str(db_path))
        
        if not best_code or best_fitness <= _MIN_FITNESS_THRESHOLD:
            print(f"   ⚠️ Skipping {i}: No valid solution found (fitness: {best_fitness:.4f})")
            return None
        
        # E. Extract only the EVOLVE-BLOCK content (remove markers and fixed code)
        # The best_code from database contains the full file, but we only want the evolved part
        from shinka.llm import extract_between  # noqa: PLC0415
        evolved_code = extract_between(best_code, "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END", False)
        if not evolved_code or evolved_code == "none":
            # Fallback: if extraction fails, use the full code (shouldn't happen)
            evolved_code = best_code
            print(f"   ⚠️ Warning: Could not extract EVOLVE-BLOCK content for problem {i}, using full code")
        
        # F. Generate Thinking Trace
        # Create trace generator per worker (thread-safe)
        tracer = TraceGenerator(api_key=api_key, model="gpt-5-mini")
        trace = tracer.generate(prompt_data["problem"], evolved_code)
        
        # G. Filter out failed trace generations
        if "Analysis failed." in trace or trace.strip() == "Analysis failed.":
            print(f"   ⚠️ Skipping {i}: Trace generation failed")
            return None
        
        # H. Format Entry for Student Training
        full_content = (
            f"<think>\n{trace}\n</think>\n\n"
            f"<code>\n{evolved_code}\n</code>"
        )
        
        # Calculate token counts
        user_tokens = count_tokens(prompt_data["problem"])
        assistant_tokens = count_tokens(full_content)
        total_tokens = user_tokens + assistant_tokens
        
        return {
            "problem_id": prompt_data.get("uuid", f"problem_{i}"),
            "messages": [
                {"role": "user", "content": prompt_data["problem"]},
                {"role": "assistant", "content": full_content}
            ],
            "score": float(best_fitness),
            "type": ptype,
            "mission": json.dumps(problem_dict.get("mission", {})),
            "index": i,  # Keep track of original index for sorting
            "num_tokens": total_tokens,  # Total tokens (user + assistant)
            "num_tokens_user": user_tokens,  # User prompt tokens
            "num_tokens_assistant": assistant_tokens,  # Assistant response tokens
        }
    except Exception as e:
        print(f"   ❌ Error processing problem {i}: {e}")
        traceback.print_exc()
        return None


def extract_best_code_from_database(db_path: str) -> tuple[str, float]:
    """
    Extract the best solution code and score from the evolution database.
    
    Args:
        db_path: Absolute path to the database file
    
    Returns:
        (code: str, combined_score: float)
    """
    # Ensure db_path is absolute
    db_path_abs = Path(db_path).resolve()
    if not db_path_abs.exists():
        raise FileNotFoundError(f"Database file not found: {db_path_abs}")  # noqa: TRY003
    
    db_config = DatabaseConfig(db_path=str(db_path_abs))
    db = ProgramDatabase(config=db_config, read_only=True)
    
    best_program = db.get_best_program()
    if not best_program:
        return None, 0.0
    
    return best_program.code, best_program.combined_score or 0.0


def main():  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(
        description="Generate SFT dataset for SDS using ShinkaEvolve"
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Number of SFT samples to generate"
    )
    parser.add_argument(
        "--generations", type=int, default=10,
        help="Evolution generations per problem"
    )
    parser.add_argument(
        "--push_to", type=str, default=None,
        help="HuggingFace repo ID (e.g., 'Org/Dataset'). If not provided, will auto-generate "
             "names based on sample count and seed: SoheylM/ShinkaEvolve-SDS-{N}k-seed{seed} and "
             "IDEALLab/ShinkaEvolve-SDS-{N}k-seed{seed}"
    )
    parser.add_argument(
        "--api_key", type=str,
        help="OpenAI API Key (optional if in env)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results. If not specified, will be created in script directory with timestamp."
    )
    parser.add_argument(
        "--eval_timeout", type=float, default=5.0,
        help="Timeout in seconds for code execution during evaluation (default: 5.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Master seed for reproducibility. Seeds problem generation deterministically. "
             "Evolution uses natural randomness to ensure diversity and avoid mode collapse."
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers for problem processing. "
             "If not specified, defaults to number of CPU cores. "
             "Set to 1 for sequential processing."
    )
    
    # Evolution config arguments
    parser.add_argument(
        "--max_parallel_jobs", type=int, default=1,
        help="Max parallel jobs within each problem's evolution (default: 1)"
    )
    parser.add_argument(
        "--max_patch_attempts", type=int, default=3,
        help="Max attempts to generate valid patches (default: 3)"
    )
    parser.add_argument(
        "--max_patch_resamples", type=int, default=3,
        help="Max resamples if patch fails validation (default: 3)"
    )
    parser.add_argument(
        "--patch_types", type=str, nargs="+", default=["diff", "full", "cross"],
        help="Patch types: diff, full, cross (default: diff full cross)"
    )
    parser.add_argument(
        "--patch_type_probs", type=float, nargs="+", default=[0.6, 0.3, 0.1],
        help="Probabilities for each patch type (default: 0.6 0.3 0.1)"
    )
    parser.add_argument(
        "--llm_models", type=str, nargs="+", default=["gpt-5-nano", "gpt-5-mini"],
        help="LLM models for mutations (default: gpt-5-nano gpt-5-mini)"
    )
    parser.add_argument(
        "--llm_temperatures", type=float, nargs="+", default=[0.0, 0.5, 0.7],
        help="LLM temperatures (default: 0.0 0.5 0.7)"
    )
    parser.add_argument(
        "--llm_max_tokens", type=int, default=8192,
        help="Max tokens for LLM generation (default: 8192)"
    )
    
    # Database config arguments
    parser.add_argument(
        "--num_islands", type=int, default=2,
        help="Number of evolutionary islands (default: 2)"
    )
    parser.add_argument(
        "--archive_size", type=int, default=20,
        help="Size of elite solution archive per island (default: 20)"
    )
    parser.add_argument(
        "--parent_selection_strategy", type=str, default="weighted",
        choices=["weighted", "power_law", "beam_search", "uniform"],
        help="Parent selection strategy (default: weighted)"
    )
    parser.add_argument(
        "--parent_selection_lambda", type=float, default=10.0,
        help="Lambda for weighted parent selection (default: 10.0)"
    )
    parser.add_argument(
        "--elite_selection_ratio", type=float, default=0.3,
        help="Fraction of elite programs for inspiration (default: 0.3)"
    )
    parser.add_argument(
        "--num_archive_inspirations", type=int, default=4,
        help="Number of programs drawn from archive (default: 4)"
    )
    parser.add_argument(
        "--num_top_k_inspirations", type=int, default=2,
        help="Number of top programs from current generation (default: 2)"
    )
    parser.add_argument(
        "--migration_interval", type=int, default=10,
        help="Generations between island migrations (default: 10)"
    )
    parser.add_argument(
        "--migration_rate", type=float, default=0.1,
        help="Fraction of population to migrate (default: 0.1)"
    )
    parser.add_argument(
        "--no_island_elitism", dest="island_elitism", action="store_false",
        help="Disable island elitism (default: enabled)"
    )
    
    args = parser.parse_args()
    
    # Set default for island_elitism if not explicitly set
    if not hasattr(args, 'island_elitism'):
        args.island_elitism = True
    
    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API Key required for SFT Data Generation")  # noqa: TRY003
    
    # Check for HF_TOKEN if pushing to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    
    # Set up seeding for reproducibility
    # Strategy: Seed problem generation deterministically, but let evolution
    # use natural randomness to avoid mode collapse while maintaining problem reproducibility
    master_seed = args.seed
    if master_seed is not None:
        # Seed Python's random for problem generation
        random.seed(master_seed)
        print(f"🌱 Using master seed: {master_seed} (problem generation seeded, evolution uses natural randomness)")
    else:
        print("⚠️  No seed provided - results will not be reproducible")
    
    # Create output directory with timestamp if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = script_dir / f"sds_dataset_output_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run configuration to config file
    config_data = {
        "prompt_variant": "neutral-v2",
        "seed": master_seed,
        "samples": args.samples,
        "generations": args.generations,
        "eval_timeout": args.eval_timeout,
        "workers": args.workers if args.workers is not None else "auto",
        "output_dir": str(output_dir),
        "timestamp": datetime.now().isoformat(),
        # Evolution config
        "max_parallel_jobs": args.max_parallel_jobs,
        "max_patch_attempts": args.max_patch_attempts,
        "max_patch_resamples": args.max_patch_resamples,
        "patch_types": args.patch_types,
        "patch_type_probs": args.patch_type_probs,
        "llm_models": args.llm_models,
        "llm_temperatures": args.llm_temperatures,
        "llm_max_tokens": args.llm_max_tokens,
        # Database config
        "num_islands": args.num_islands,
        "archive_size": args.archive_size,
        "parent_selection_strategy": args.parent_selection_strategy,
        "parent_selection_lambda": args.parent_selection_lambda,
        "elite_selection_ratio": args.elite_selection_ratio,
        "num_archive_inspirations": args.num_archive_inspirations,
        "num_top_k_inspirations": args.num_top_k_inspirations,
        "migration_interval": args.migration_interval,
        "migration_rate": args.migration_rate,
        "island_elitism": args.island_elitism,
        "push_to": args.push_to,
    }
    config_file = output_dir / "run_config.json"
    with Path(config_file).open("w") as f:
        json.dump(config_data, f, indent=2)
    print(f"💾 Saved run configuration to {config_file}")
    
    # Set eval timeout in environment for evaluate.py to pick up
    os.environ["SDS_EVAL_TIMEOUT"] = str(args.eval_timeout)
    
    dataset_records = []
    
    # Determine number of workers
    if args.workers is None:
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = max(1, args.workers)
    
    print(f"🚀 Starting Pipeline: {args.samples} samples")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚡ Using {num_workers} parallel workers")
    print(f"⏱️  Evaluation timeout: {args.eval_timeout}s per code execution")
    
    # Process problems in parallel
    if num_workers == 1:
        # Sequential processing (easier debugging)
        print("🔄 Running in sequential mode (workers=1)")
        for i in tqdm.tqdm(range(args.samples), desc="Generating samples"):
            result = process_single_problem(
                i=i,
                master_seed=master_seed,
                num_generations=args.generations,
                script_dir=script_dir,
                output_dir=output_dir,
                api_key=api_key,
                max_parallel_jobs=args.max_parallel_jobs,
                max_patch_attempts=args.max_patch_attempts,
                max_patch_resamples=args.max_patch_resamples,
                patch_types=args.patch_types,
                patch_type_probs=args.patch_type_probs,
                llm_models=args.llm_models,
                llm_temperatures=args.llm_temperatures,
                llm_max_tokens=args.llm_max_tokens,
                num_islands=args.num_islands,
                archive_size=args.archive_size,
                parent_selection_strategy=args.parent_selection_strategy,
                parent_selection_lambda=args.parent_selection_lambda,
                elite_selection_ratio=args.elite_selection_ratio,
                num_archive_inspirations=args.num_archive_inspirations,
                num_top_k_inspirations=args.num_top_k_inspirations,
                migration_interval=args.migration_interval,
                migration_rate=args.migration_rate,
                island_elitism=args.island_elitism,
            )
            if result is not None:
                dataset_records.append(result)
    else:
        # Parallel processing
        print(f"⚡ Running in parallel mode ({num_workers} workers)")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all problems
            future_to_index = {
                executor.submit(
                    process_single_problem,
                    i=i,
                    master_seed=master_seed,
                    num_generations=args.generations,
                    script_dir=script_dir,
                    output_dir=output_dir,
                    api_key=api_key,
                    max_parallel_jobs=args.max_parallel_jobs,
                    max_patch_attempts=args.max_patch_attempts,
                    max_patch_resamples=args.max_patch_resamples,
                    patch_types=args.patch_types,
                    patch_type_probs=args.patch_type_probs,
                    llm_models=args.llm_models,
                    llm_temperatures=args.llm_temperatures,
                    llm_max_tokens=args.llm_max_tokens,
                    num_islands=args.num_islands,
                    archive_size=args.archive_size,
                    parent_selection_strategy=args.parent_selection_strategy,
                    parent_selection_lambda=args.parent_selection_lambda,
                    elite_selection_ratio=args.elite_selection_ratio,
                    num_archive_inspirations=args.num_archive_inspirations,
                    num_top_k_inspirations=args.num_top_k_inspirations,
                    migration_interval=args.migration_interval,
                    migration_rate=args.migration_rate,
                    island_elitism=args.island_elitism,
                ): i
                for i in range(args.samples)
            }
            
            # Collect results as they complete (with progress bar)
            completed_results = {}
            with tqdm.tqdm(total=args.samples, desc="Generating samples") as pbar:
                for future in as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        result = future.result()
                        if result is not None:
                            completed_results[i] = result
                    except Exception as e:
                        print(f"   ❌ Problem {i} failed with exception: {e}")
                    finally:
                        pbar.update(1)
            
            # Sort by original index to maintain deterministic order
            dataset_records = [
                completed_results[i] 
                for i in sorted(completed_results.keys())
            ]
    
    print(f"\n✅ Generated {len(dataset_records)} high-quality samples out of {args.samples} attempted.")
    
    # Filter out any entries with failed traces (safety check)
    filtered_records = []
    for record in dataset_records:
        assistant_content = record.get("messages", [{}])[1].get("content", "") if len(record.get("messages", [])) > 1 else ""
        if "Analysis failed." in assistant_content:
            print(f"⚠️  Filtering out problem {record.get('index', 'unknown')}: Contains 'Analysis failed.' in trace")
            continue
        filtered_records.append(record)
    
    if len(filtered_records) < len(dataset_records):
        print(f"📊 Filtered {len(dataset_records) - len(filtered_records)} entries with failed traces")
        dataset_records = filtered_records
    
    # G. Save or Push Dataset
    if len(dataset_records) > 0:
        ds = Dataset.from_list(dataset_records)
        
        # Determine repo names using same logic as push_sds_to_hf.py
        if args.push_to:
            # Use provided repo name
            repos_to_push = [args.push_to]
        else:
            # Auto-generate repo names based on sample count and seed
            total_samples = len(dataset_records)
            # Format sample count for repo name (e.g., 10000 -> "10k", 5000 -> "5k")
            if total_samples >= _KILO_THRESHOLD:
                sample_suffix = f"{total_samples // 1000}k"
            else:
                sample_suffix = str(total_samples)
            
            # Use seed from args if available, otherwise use master_seed
            seed_value = args.seed if args.seed is not None else master_seed if master_seed is not None else 0
            
            # Create descriptive repo names for both organizations (matching push_sds_to_hf.py style)
            soheylm_repo = f"SoheylM/ShinkaEvolve-SDS-{sample_suffix}-seed{seed_value}"
            ideallab_repo = f"IDEALLab/ShinkaEvolve-SDS-{sample_suffix}-seed{seed_value}"
            repos_to_push = [soheylm_repo, ideallab_repo]
        
        # Push to HuggingFace
        if not hf_token:
            print("⚠️  HF_TOKEN not found in environment. Cannot push to HuggingFace.")
            print("   Saving dataset locally instead...")
            repos_to_push = []  # Skip pushing
        
        success_count = 0
        for repo_name in repos_to_push:
            print(f"📤 Pushing to HuggingFace: {repo_name}...")
            try:
                # Push the dataset (this creates/updates the repository)
                ds.push_to_hub(repo_name, token=hf_token, private=True)
                print(f"✅ Successfully pushed dataset to {repo_name}")
                
                # Upload the config file as a separate file in the repository
                # This doesn't affect the dataset structure - it's just metadata
                # The config file will appear in the repository root, separate from the dataset files
                if HF_HUB_AVAILABLE and config_file.exists():
                    try:
                        api = HfApi(token=hf_token)
                        api.upload_file(
                            path_or_fileobj=str(config_file),
                            path_in_repo="run_config.json",
                            repo_id=repo_name,
                            repo_type="dataset",
                            commit_message=f"Add run configuration for {len(dataset_records)} samples"
                        )
                        print(f"✅ Successfully uploaded run_config.json to {repo_name}")
                    except Exception as config_error:
                        print(f"⚠️  Warning: Could not upload config file to {repo_name}: {config_error}")
                        print("   Dataset was pushed successfully, but config file upload failed")
                elif not HF_HUB_AVAILABLE:
                    print("⚠️  Warning: huggingface_hub not available, skipping config file upload")
                
                success_count += 1
            except Exception as e:
                print(f"❌ Error pushing to {repo_name}: {e}")
        
        if success_count > 0:
            print(f"✅ Successfully pushed to {success_count}/{len(repos_to_push)} repositories")
        else:
            print("❌ Failed to push to any repository")
        
        # Always save locally as backup
        output_file = output_dir / "sft_dataset.jsonl"
        with Path(output_file).open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in dataset_records)
        print(f"💾 Saved local backup to {output_file}")
        
        # Also save as JSON for easy inspection
        json_file = output_dir / "sft_dataset.json"
        with Path(json_file).open("w") as f:
            json.dump(dataset_records, f, indent=2)
        print(f"💾 Also saved to {json_file}")
    else:
        print("⚠️  No dataset records to save or push")


if __name__ == "__main__":
    main()
