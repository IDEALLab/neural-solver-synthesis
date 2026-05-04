# Scripts Directory

This directory contains **active scripts** for training, evaluation, and result generation. Legacy scripts are archived in `scripts/archive/`.

## Active Scripts

### Training Scripts

- **`launch_14b_grpo_experiments.sh`** - Batch launcher for 14B GRPO training experiments
  - Launches Hero + 4 ablations (Oracle, Diversity, w/o Structure, w/o Prompt) across 3 seeds
  - All 5 configurations (Hero + 4 ablations) are included in main results
  - Usage: `./scripts/launch_14b_grpo_experiments.sh [--seed SEED]`
  
- **`train_capstor_unified_sds_qwen_coder.slurm`** - Multi-node training on Capstor cluster
- **`train_unified_sds_qwen_coder.slurm`** - Single-node training

### Evaluation Scripts

- **`evaluate_14b_grpo_experiments.sh`** - Batch evaluator for SDS evaluation
  - Evaluates latest checkpoints for all 5 configs (Hero + 4 ablations) across 3 seeds
  - All configurations are included in the main results report set
  - Usage: `./scripts/evaluate_14b_grpo_experiments.sh [--seed SEED] [--batch-id BATCH_ID]`
  
- **`evaluate_14b_grpo_bigcode.sh`** - Batch evaluator for BigCode evaluation (HumanEval, MBPP)
  - Usage: `./scripts/evaluate_14b_grpo_bigcode.sh [--seed SEED] [--batch-id BATCH_ID]`

- **`evaluate_base_model_sds.sh`** - Batch evaluator for Base model Best-of-64 SDS evaluation
  - Generates 64 samples per problem (temperature=0.6) with bootstrap Pass@k analysis
  - Usage: `./scripts/evaluate_base_model_sds.sh [--seed SEED] [--batch-id BATCH_ID]`

- **`evaluate_shinka_baseline.sh`** - Batch evaluator for ShinkaEvolve baseline SDS evaluation
  - Runs locally (no GPU/SLURM needed), evaluates evolved codes against test problems
  - Usage: `./scripts/evaluate_shinka_baseline.sh [--seed SEED] [--batch-id BATCH_ID] [--no-wandb]`

- **`eval_capstor_sds_pipeline.slurm`** - Multi-node SDS evaluation on Capstor
- **`eval_capstor_bigcode.slurm`** - BigCode evaluation on Capstor
- **`eval_capstor_universal_search.slurm`** - Universal solver search on Capstor
- **`eval_sds_pipeline.slurm`** - Single-node SDS evaluation
- **`eval_sds_plot_only.slurm`** - Plot generation only (no evaluation)

### Result Generation

- **`generate_paper_results.sh`** - One-command result generation from report set
  - Runs convergence analysis, aggregates SDS/BigCode results, generates all plots/tables
  - Usage: `./scripts/generate_paper_results.sh [report_set.json]`
  - Default: `experiments/report_sets/paper_main_results_v1.json`

### Universal Solver Search

- **`run_universal_search_base.sh`** - Runs adaptive tournament search on Base model generations
  - Finds universal solver from 64k Base model code samples

### Utility Scripts

- **`push_all_submodules.sh`** - Pushes all git submodules
- **`update_submodules.sh`** - Updates git submodules
- Some helper scripts in the development repo also operate on excluded manuscript submodules.

## Archived Scripts

**Location**: `scripts/archive/`

The `archive/` subdirectory contains legacy scripts from previous experiments and domains:

- **Old training scripts**: legacy domain/model experiments, 7B/32B variants, and SFT experiments
- **Old evaluation scripts**: Domain-specific evaluation pipelines
- **Migration scripts**: One-time data migration utilities (e.g., `migrate_minimalist_to_struct_feas.py`)
- **Experimental protocols**: `EXPERIMENTAL_PROTOCOL.md` (historical documentation)

These scripts are kept for reference but are **not actively used** in the current SDS-focused workflow.

## Script Organization

- **`.sh` files**: Bash scripts for batch operations and utilities
- **`.slurm` files**: SLURM job scripts for cluster execution
- **`.py` files**: Python scripts (currently only in archive)

## Usage Patterns

### Typical Workflow

1. **Training**: `./scripts/launch_14b_grpo_experiments.sh`
2. **Base Model SDS**: `./scripts/evaluate_base_model_sds.sh --batch-id YYYYMMDD_baselines-v1`
3. **ShinkaEvolve**: `./scripts/evaluate_shinka_baseline.sh --batch-id YYYYMMDD_baselines-v1`
4. **RL Models SDS**: `./scripts/evaluate_14b_grpo_experiments.sh --batch-id YYYYMMDD_desc-v1`
5. **BigCode**: `./scripts/evaluate_14b_grpo_bigcode.sh --batch-id YYYYMMDD_desc-v1`
6. **Universal Search**: `./scripts/run_universal_search_base.sh --base-root ... --batch-id ...` (after step 2)
7. **Aggregation**: `./scripts/generate_paper_results.sh experiments/report_sets/paper_main_results_v1.json`

Steps 2-5 can run in parallel. Step 6 depends on step 2 completing.

For the full end-to-end reproduction guide, see [`docs/REPRODUCTION.md`](../docs/REPRODUCTION.md).

### Cluster Execution

Most `.slurm` scripts are designed to run on HPC clusters. They handle:
- Container mounting
- Environment setup
- W&B logging
- Batch ID management

See individual script headers for cluster-specific requirements.
