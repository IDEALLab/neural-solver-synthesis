#!/usr/bin/env python3
"""
Example runner for SDS evolution using ShinkaEvolve.
This can be used for single problem evolution.
For batch dataset generation, use run_sds_pipeline.py
"""

import os

from shinka.core import EvolutionConfig, EvolutionRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Job configuration
job_config = LocalJobConfig(eval_program_path="evaluate.py")

# Database configuration
db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=40,
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    migration_interval=10,
    migration_rate=0.1,
    island_elitism=True,
    parent_selection_strategy="weighted",
    parent_selection_lambda=10.0,
)

# Task system message for SDS
search_task_sys_msg = """You are an expert algorithmist specializing in combinatorial optimization, particularly the Synergistic Dependency Selection (SDS) problem.

The SDS problem requires selecting a subset of variables that:
1. Satisfies cardinality bounds (min/max number of items)
2. Respects mutex constraints (cannot select both items in a mutex pair)
3. Respects group constraints (at most one item per group)
4. Satisfies precedence constraints (if j is selected, i must be selected)
5. Maximizes the objective: sum of selected item weights + pairwise interaction values

Key strategies to explore:
1. Greedy selection with constraint-aware ordering
2. Dynamic programming for small instances
3. Constraint satisfaction with backtracking
4. Weighted scoring that accounts for interactions
5. Precedence-aware selection ordering
6. Group-aware selection that maximizes diversity

The code should read JSON from stdin with "requirements" and "catalog", and output JSON to stdout with "selection" containing "variables" list.

Be creative and find efficient solutions that maximize the objective while respecting all constraints."""

# Evolution configuration
evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=20,
    max_parallel_jobs=2,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=["gpt-4o"],
      llm_kwargs={
        "temperatures": [0.0, 0.5, 0.7],
        "max_tokens": 8192,
    },
    embedding_model="text-embedding-3-small",
    init_program_path="initial.py",
    results_dir=None,  # Auto-generated
)


def main():
    """
    Run evolution for SDS problem.
    Problem data should be set via SDS_PROBLEM_DATA environment variable.
    """
    # Check for problem data
    problem_data_str = os.environ.get("SDS_PROBLEM_DATA")
    if not problem_data_str:
        print("Warning: SDS_PROBLEM_DATA not set. Using empty problem.")
        print("Set it like: export SDS_PROBLEM_DATA='{\"requirements\": {...}, \"catalog\": {...}}'")
    
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    main()

