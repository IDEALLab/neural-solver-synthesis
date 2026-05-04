# SDS (Synergistic Dependency Selection) Example

This example demonstrates how to use ShinkaEvolve for the SDS combinatorial optimization problem and generate high-quality SFT datasets.

## Files

- **`initial.py`**: Starting solution with `EVOLVE-BLOCK` markers. The `solve_sds()` function will be evolved.
- **`evaluate.py`**: Evaluation script using ShinkaEvolve's `run_shinka_eval` pattern. Validates feasibility and calculates fitness.
- **`run_evo.py`**: Standard evolution runner for solving a single SDS problem (same pattern as other examples).
- **`run_sds_pipeline.py`**: **Optional** - Specialized pipeline for generating SFT training datasets. Only needed if you want to create datasets with reasoning traces for model training.

## Usage

### Standard Evolution (Single Problem)

For solving a single SDS problem, use `run_evo.py` just like other ShinkaEvolve examples:

```bash
# Set problem data via environment variable
export SDS_PROBLEM_DATA='{"requirements": {...}, "catalog": {...}}'

# Run evolution
cd examples/sds
python run_evo.py
```

This follows the standard ShinkaEvolve pattern: `initial.py` + `evaluate.py` + `run_evo.py`.

### Dataset Generation (SFT Training Data)

**Use this to generate SFT training datasets.** The pipeline script automates:
- Generating multiple problem instances with hash-based seeding for reproducibility
- Running evolution for each problem in parallel
- Extracting best solutions from the evolution database
- Generating reasoning traces using LLM
- Formatting training data with token counts
- Pushing to Hugging Face Hub (optional)

#### Basic Usage

```bash
# From examples/sds directory
cd examples/sds
python run_sds_pipeline.py \
    --samples 1000 \
    --generations 10 \
    --seed 303 \
    --workers 10 \
    --eval_timeout 5.0

# Or from project root
python deps/ShinkaEvolve/examples/sds/run_sds_pipeline.py \
    --samples 1000 \
    --generations 10 \
    --seed 303
```

#### Arguments

- **`--samples`** (default: 100): Number of SFT samples to generate
- **`--generations`** (default: 10): Evolution generations per problem
- **`--seed`** (optional): Master seed for reproducibility. Seeds problem generation deterministically using hash-based seeding. Evolution uses natural randomness to ensure diversity and avoid mode collapse.
- **`--workers`** (default: CPU count): Number of parallel workers for problem processing. Set to 1 for sequential processing.
- **`--eval_timeout`** (default: 5.0): Timeout in seconds for code execution during evaluation. Prevents runaway processes from slow LLM-generated code. Use `null` or omit for no timeout (not recommended for large runs).
- **`--output_dir`** (optional): Output directory for results. If not specified, creates a timestamped directory: `sds_dataset_output_YYYYMMDDHHMMSS` in the script directory.
- **`--push_to`** (optional): HuggingFace repo ID (e.g., `YourOrg/SDS-Dataset`). If not provided, auto-generates names based on sample count and seed (configured in script)
- **`--api_key`** (optional): OpenAI API Key. Can also be set via `OPENAI_API_KEY` environment variable.

#### Output Structure

Each run creates a timestamped output directory containing:

```
sds_dataset_output_20251205084713/
├── run_config.json          # Run parameters (seed, samples, generations, etc.)
├── sft_dataset.jsonl        # Training dataset (JSONL format)
├── sft_dataset.json         # Training dataset (JSON format)
└── problem_0/              # Per-problem evolution results
    ├── evolution_db.sqlite
    ├── gen_0/
    ├── gen_1/
    └── ...
```

#### Reproducibility

The pipeline uses a **hash-based seeding strategy**:
- **Problem generation**: Deterministically seeded using `hash(master_seed, problem_index)` to ensure unique, reproducible problems across different master seeds
- **Evolution**: Uses natural randomness to avoid mode collapse while maintaining problem reproducibility

This ensures:
- Same `(seed, problem_index)` always generates the same problem
- Different seeds generate different problems (no overlap)
- Evolution maintains diversity through natural randomness

#### Hugging Face Integration

The generated dataset includes:
- Problem instances with full requirements and catalog
- Evolved solutions with reasoning traces
- Token counts (`num_tokens`, `num_tokens_user`, `num_tokens_assistant`) using `tiktoken`

To push to Hugging Face, set `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=your_token_here
python run_sds_pipeline.py --samples 1000 --seed 303 --push_to YourOrg/SDS-Dataset
```

If `--push_to` is not provided, the script auto-generates repository names based on configured organizations (see script source for details).

## Integration with ShinkaEvolve

This example properly integrates with ShinkaEvolve's architecture:

- ✅ Uses `EvolutionRunner` with `EvolutionConfig`, `JobConfig`, `DatabaseConfig`
- ✅ Follows the `evaluate.py` pattern with `run_shinka_eval` and timeout support
- ✅ Uses `EVOLVE-BLOCK` markers in `initial.py`
- ✅ Leverages database, islands, archives, and all ShinkaEvolve features
- ✅ Extracts best solutions from the database after evolution
- ✅ Parallel problem processing with configurable workers
- ✅ Hash-based seeding for reproducible problem generation
- ✅ Automatic config file generation for run tracking
- ✅ Token counting for training data preparation

## Problem Format

The SDS problem expects:
- **Input** (stdin or parameter): JSON with `requirements` and `catalog`
- **Output** (stdout or return): JSON with `selection.variables` list

The evaluator validates:
- Cardinality bounds
- Mutex constraints
- Group constraints
- Precedence constraints
- Objective maximization (weights + interactions)

**Evaluation Timeout**: Code execution during evaluation is subject to a timeout (default 5.0s, configurable via `--eval_timeout`). This prevents runaway processes from slow or infinite-loop code generated by the LLM. The timeout is enforced via `SDS_EVAL_TIMEOUT` environment variable, which is read by `evaluate.py`.

**Note**: The timeout feature requires a modification to the core ShinkaEvolve file `shinka/core/wrap_eval.py` to add a `timeout` parameter to `run_shinka_eval()`. This modification uses Unix signal-based timeout handling (`signal.SIGALRM`) to interrupt long-running code execution.

