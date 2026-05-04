# Universal Solver Search (SDS)

## Executive Summary

The **Universal Solver Search** is an adaptive tournament-based method to identify a single, high-quality Python code program from a large pool of generated solutions (e.g., 64,000 samples from Base Model Best-of-64 evaluation). The goal is to find a "universal solver" that:

1. **Feasibility**: Satisfies all constraints on all test instances
2. **Optimality**: Achieves low optimality gap across the test set
3. **Efficiency**: Completes execution within a strict timeout (default: 5 seconds per instance)

This search operates **without any new LLM inference**—it only evaluates existing generated code samples, making it computationally efficient compared to generating new solutions.

---

## 1. Problem Motivation

### Context

During Base Model (Best-of-64) evaluation, we generate **64 independent code samples per problem** (1000 problems × 64 samples = **64,000 total generations**). These are saved in `generations.jsonl` files (~1.1GB per seed).

### Challenge

While the "Best-of-64" strategy (selecting the best solution per problem) achieves good aggregate performance, it requires:
- **64× test-time compute** (generating 64 samples per problem)
- **64× storage** (saving all generations)
- **Per-problem selection** (choosing the best of 64 for each instance)

### Goal

Find a **single universal code program** that can be reused across all problems, eliminating the need for per-problem sampling while maintaining competitive performance.

---

## 2. Methodology

### 2.1 Overview

The search uses a **two-stage adaptive tournament**:

1. **Stage 1 (Tournament)**: Evaluate all unique candidate codes on a discriminative subset of missions (default: 30 missions)
2. **Stage 2 (Verification)**: Evaluate top survivors on the full test set (1000 missions)

### 2.2 Code Deduplication

**Input**: `generations.jsonl` files containing 64,000 raw code samples

**Process**:
1. **Extract code blocks**: Use regex pattern `<code>\s*(.*?)\s*</code>` (case-insensitive, dotall mode) to extract code from generated text
   - If no `<code>` tag is found, the candidate is skipped (not included in the pool)
2. **Canonicalize**: Normalize code for stable hashing:
   - Normalize line endings: `\r\n` and `\r` → `\n`
   - Strip trailing whitespace from each line
   - Add final newline
   - **Note**: We intentionally do NOT perform AST normalization, as generated code may be syntactically invalid
3. **Hash**: Compute SHA-256 hash of canonicalized code, take first 16 hexadecimal characters as unique identifier
4. **Deduplicate**: Map unique hash → canonicalized code (if same hash appears multiple times, keep first occurrence)

**Output**: Unique code hash → code mapping

**Result**: Typically reduces 64,000 samples to **~5,000–15,000 unique codes** (depending on diversity). In practice, we observed ~64k unique codes per seed, indicating high diversity in Base Model generations.

### 2.3 Tournament Mission Selection

The tournament uses a **discriminative subset** of missions (default: 30) selected from the full test set using the Base Model's `metrics_final.csv` from the **same seed**:

**Selection Algorithm**:
1. **Load Base Model metrics**: Read `metrics_final.csv` from the corresponding Base Model evaluation (same seed)
2. **Compute optimality gaps**: 
   - If `Gap` column exists, use it directly (convert from percentage if >1.01)
   - Otherwise, compute from `vbs_score` and `llm_score`: `gap = (vbs_score - llm_score) / vbs_score` (only for feasible solutions)
3. **Select Hard missions (60%)**: 
   - Sort all missions by optimality gap (descending)
   - Take top `floor(0.6 * n_missions)` missions with highest gaps
4. **Select Random missions (40%)**: 
   - From remaining missions, randomly sample `n_missions - num_hard` missions
   - Uses deterministic RNG seeded with the search seed (default: same as `--seed`)

**Rationale**: This strategy prioritizes missions where the Base Model struggled, making the tournament more selective and efficient at filtering out weak candidates early.

### 2.4 Evaluation Criteria

For each candidate code on each mission:

1. **Feasibility**: 
   - Code must execute without syntax/runtime errors
   - Code must return valid JSON with `"selection"` containing `"variables"` list
   - Selection must satisfy all SDS constraints: cardinality bounds, precedence, mutex pairs, group constraints
   - Constraint checking uses `check_constraint_violations()` from `evaluation/sds/utils.py` (same as main evaluation pipeline)

2. **Timeout**: 
   - Execution must complete within the timeout (default: 5.0 seconds)
   - Timeout is enforced by `run_candidate()` sandbox execution
   - A candidate is considered timed out if `execution_time > timeout + 1e-6` or `error_type == "timeout"`

3. **Optimality Gap**: 
   - **VBS Reference**: Per-UUID VBS score is read from Base Model's `metrics_final.csv` (same seed)
   - **Candidate Score**: Objective value computed using `calculate_true_score()` (sum of selected item weights + pairwise interaction values)
   - **Gap Calculation**: `gap = (VBS - candidate_score) / VBS` if `VBS > 1e-6`, otherwise `gap = None` (not computed for degenerate instances)
   - **Edge Cases**: 
     - If candidate is infeasible, `candidate_score = 0.0`
     - If `candidate_score < 0`, it's clipped to `0.0` before gap calculation
     - If `gap < 0` (shouldn't happen), it's set to `0.0`

**Universal Criterion**: A candidate is "universal" only if it is:
- ✅ Feasible on **all** evaluated missions
- ✅ Completes within timeout on **all** evaluated missions
- ✅ Has low mean optimality gap (computed only on missions with valid VBS)

### 2.5 Tournament Rounds

#### Round 1: Tournament Evaluation

- **Input**: All unique candidates from deduplication
- **Missions**: Tournament subset (default: 30 missions, selected as described in Section 2.3)
- **Evaluation**: Evaluate each candidate on all tournament missions in parallel (using `ProcessPoolExecutor`)
- **Metrics Computed**: 
  - Per-candidate summary: `evals`, `feasible`, `timeouts`, `ok`, `universal_ok`, `mean_gap`, `max_gap`, `mean_time`
  - **Important**: `mean_gap` and `max_gap` are computed **only on the 30 tournament missions**
- **Filtering**: Keep only candidates where `universal_ok = True` (feasible + no timeout on **all** tournament missions)
- **Ranking**: Sort filtered candidates by:
  1. `universal_ok` (descending: True first)
  2. `mean_gap` (ascending: lower is better)
  3. `max_gap` (ascending: lower is better)
- **Selection**: Keep top `--keep-per-round` candidates (default: 500)
- **Early Exit**: If no candidates pass the filter (`strict.empty`), exit with status `"no_survivors"` (no Round 2)

#### Round 2: Full Verification

- **Input**: Top survivors from Round 1 (default: top 10 by ranking)
- **Missions**: **Full test set** (all 1000 missions)
- **Evaluation**: Evaluate each survivor on all 1000 missions in parallel
- **Metrics Computed**: 
  - Same structure as Round 1, but **`mean_gap` and `max_gap` are computed on all 1000 missions**
  - This is the final performance metric reported in `final_topK.csv`
- **Selection**: The candidate with:
  - `universal_ok = True` (feasible + no timeout on all 1000 missions)
  - Lowest `mean_gap` (tie-break by `max_gap`)
- **Output**: 
  - If universal winner found: Status `"winner_found"`, save code to `winner_code.py`
  - If no universal winner: Status `"no_universal_winner"`, but still report best candidates in `final_topK.csv`

### 2.6 Output

The search produces:

- **`winner_code.py`**: The single universal solver code (if found)
- **`result.json`**: Status (`winner_found`, `no_universal_winner`, `no_survivors`)
- **`tournament_evals.csv`**: Detailed evaluation results from Round 1
- **`final_evals.csv`**: Detailed evaluation results from Round 2
- **`tournament_summary.csv`**: Per-candidate summary from Round 1
- **`final_topK.csv`**: Per-candidate summary from Round 2
- **`candidates_dedup.json`**: Deduplicated candidate pool (for reproducibility)
- **`tournament_config.json`**: Full configuration and tournament mission UUIDs

---

## 3. Implementation Details

### 3.1 Code Location

- **Main Script**: `evaluation/sds/universal_solver_search.py`
- **SLURM Wrapper**: `scripts/eval_capstor_universal_search.slurm`
- **Batch Launcher**: `scripts/run_universal_search_base.sh`

### 3.2 Dependencies

- **Evaluation Utilities**: Reuses `evaluation/sds/utils.py`:
  - `run_candidate()`: Sandboxed code execution with timeout
  - `check_constraint_violations()`: SDS constraint validation
  - `mission_to_instance()`: Convert mission dict to syndeopt instance
- **Test Dataset**: Loads from HuggingFace (configured dataset name with seed)
- **VBS Scores**: Reads from `metrics_final.csv` (from Base Model evaluation)

### 3.3 Execution Semantics

The search uses **identical evaluation semantics** as `evaluation/sds/evaluate.py`:

- **`stdin_obj` Construction**: Uses `build_stdin_obj()` function that reconstructs the exact JSON payload format:
  - `requirements`: Contains `n_variables`, `cardinality_bounds`, `precedence`, `mutex`, `groups`, `weights`, `interactions`
  - `catalog`: Contains `variables` (with `id`, `weight`, `neighbors`) and `interactions`
  - This matches the format expected by the generated code (same as training/evaluation)
- **Sandbox Execution**: Uses `run_candidate()` from `evaluation/sds/utils.py`:
  - Executes code in isolated subprocess with timeout
  - Captures stdout/stderr
  - Returns execution time, error type, and parsed JSON result
- **Constraint Checking**: Uses `check_constraint_violations()` from `evaluation/sds/utils.py`:
  - Validates cardinality bounds (min/max items)
  - Validates precedence constraints (if j selected, i must be selected)
  - Validates mutex constraints (cannot select both items in a pair)
  - Validates group constraints (at most one item per group)
- **Objective Score Calculation**: Uses `calculate_true_score()`:
  - Sum of selected item weights: `sum(weights[i] for i in selection)`
  - Plus pairwise interactions: `sum(interactions[(i,j)] for i,j in selection)`
  - Preserves negative scores (does not clip to zero)
- **VBS Reference**: 
  - Reads per-UUID VBS scores from Base Model's `metrics_final.csv` (same seed)
  - VBS is the maximum score achieved across all methods (Base Model + baselines) for that specific UUID
  - Gap is computed as `(VBS - candidate_score) / VBS` for each mission independently

This ensures fair comparison with other evaluation results, as the universal solver is evaluated under identical conditions.

---

## 4. Running Universal Search

### 4.1 Prerequisites

**On Cluster**:

1. **Base Model generations**: Must have run Base Model (Best-of-64) evaluation for seeds 101, 202, 303
2. **File locations**: `generations.jsonl` files must exist (typically ~1.1GB each)
3. **Metrics CSV**: `metrics_final.csv` from the same Base Model evaluation run

**File Locations** (typical cluster paths):

```
/capstor/scratch/{CLUSTER}/{USER}/llm-finetuning/evaluation/sds/results/qwen2.5-coder-14b/base/seed{101,202,303}/generations.jsonl
/capstor/scratch/{CLUSTER}/{USER}/llm-finetuning/evaluation/sds/results/qwen2.5-coder-14b/base/seed{101,202,303}/metrics_final.csv
```

**Note**: These files are **NOT** in the git repository (too large: ~1.1GB each). They exist only on the cluster filesystem.

**Recovery note (2026-05-02)**:
- The raw Base generation files were re-verified to exist in the private HF datasets `SoheylM/OpenR1-SDS-Base-Generations-seed{101,202,303}`.
- Each repo contains `generations.jsonl`, and authenticated access works from Clariden using the token file at `$HOME/llm/hf_token.txt`.
- This matters because the universal-search candidate pool (`191,699` unique extracted codes) can be rebuilt from those HF-backed raw files even if some local scratch copies have been cleaned up.
- At the time of verification, a direct scratch copy was still present for `seed101` at `CLUSTER_ROOT/llm-finetuning-baseline-evaluation/evaluation/sds/results_batches/20260326_baseline-eval-v1/qwen2.5-coder-14b/base/seed101/generations.jsonl`, while `seed202` and `seed303` needed HF recovery.
- For future raw audits, prefer the HF-backed `generations.jsonl` files plus the code extraction logic in `evaluation/sds/universal_solver_search.py` (`extract_code` + `canonicalize_code`) so the audited population matches the universal-search source of truth.

### 4.2 Recommended: Batch Launcher (All Seeds)

**On Cluster**:

```bash
cd /capstor/scratch/{CLUSTER}/{USER}/llm-finetuning

./scripts/run_universal_search_base.sh \
  --base-root /capstor/scratch/{CLUSTER}/{USER}/llm-finetuning/evaluation/sds/results/qwen2.5-coder-14b/base \
  --batch-id 20260120_minimal-feas-v1
```

This will:
- Submit 3 SLURM jobs (one per seed: 101, 202, 303)
- Auto-detect `generations.jsonl` and `metrics_final.csv` paths
- Write outputs to `evaluation/sds/universal_search_batches/20260120_minimal-feas-v1/seed{101,202,303}/`

### 4.3 Manual: Single Seed

**On Cluster**:

```bash
cd /capstor/scratch/{CLUSTER}/{USER}/llm-finetuning

sbatch scripts/eval_capstor_universal_search.slurm \
  --seed 101 \
  --generations-jsonl /capstor/scratch/{CLUSTER}/{USER}/llm-finetuning/evaluation/sds/results/qwen2.5-coder-14b/base/seed101/generations.jsonl \
  --metrics-csv /capstor/scratch/{CLUSTER}/{USER}/llm-finetuning/evaluation/sds/results/qwen2.5-coder-14b/base/seed101/metrics_final.csv \
  --batch-id 20260120_minimal-feas-v1
```

### 4.4 Advanced Options

**Custom Tournament Parameters**:

```bash
sbatch scripts/eval_capstor_universal_search.slurm \
  --seed 101 \
  --generations-jsonl <path> \
  --metrics-csv <path> \
  --batch-id <BATCH_ID> \
  --timeout 10.0 \
  --workers 32 \
  --tournament-missions 50 \
  --survivors 5 \
  --keep-per-round 1000
```

**Parameters**:
- `--timeout`: Per-instance execution timeout (default: 5.0 seconds)
- `--workers`: Parallel evaluation workers (default: 64)
- `--tournament-missions`: Number of missions in Round 1 (default: 30)
- `--survivors`: Number of top candidates verified in Round 2 (default: 10)
- `--keep-per-round`: Max candidates kept after Round 1 filtering (default: 500)

---

## 5. Output Structure

### 5.1 Directory Layout

```
evaluation/sds/universal_search_batches/<BATCH_ID>/seed{101,202,303}/
├── winner_code.py              # Universal solver code (if found)
├── result.json                  # Search status and summary
├── tournament_config.json       # Full configuration
├── candidates_dedup.json        # Deduplicated candidate pool
├── tournament_evals.csv         # Round 1: All candidates on tournament missions
├── tournament_summary.csv       # Round 1: Per-candidate summary
├── final_evals.csv              # Round 2: Top survivors on full test set
└── final_topK.csv               # Round 2: Per-candidate summary
```

### 5.2 Key Output Files

#### `result.json`

```json
{
  "status": "winner_found",
  "seed": 101,
  "n_candidates": 12450,
  "n_tournament_survivors": 342,
  "n_final_evaluated": 10,
  "winner_hash": "a3f2b1c4d5e6f7g8"
}
```

**Status values**:
- `winner_found`: Universal solver found (feasible + no timeout on all 1000 missions)
- `no_universal_winner`: Top survivors failed on some missions in Round 2
- `no_survivors`: No candidates passed Round 1 (all failed on tournament missions)

#### `tournament_summary.csv`

Columns:
- `code_hash`: Unique code identifier
- `evals`: Number of missions evaluated
- `feasible`: Number of feasible solutions
- `timeouts`: Number of timeout failures
- `ok`: Number of successful (feasible + no timeout) solutions
- `universal_ok`: Boolean (feasible + no timeout on **all** evaluated missions)
- `mean_gap`: Mean optimality gap (lower is better)
- `max_gap`: Maximum optimality gap
- `mean_time`: Mean execution time

#### `final_topK.csv`

Same structure as `tournament_summary.csv`, but for Round 2 (full test set evaluation).

**Important**: The `mean_gap` and `max_gap` values in `final_topK.csv` are computed on **all 1000 test missions**, not just the 30 tournament missions. This makes them directly comparable to other evaluation results (Hero, Base Best-of-64, etc.) which are also evaluated on the full 1000-mission test set.

**Example interpretation**:
- `evals: 1000` means the candidate was evaluated on all test missions
- `feasible: 984` means 984 missions were solved feasibly
- `timeouts: 16` means 16 missions exceeded the timeout
- `mean_gap: 0.2404` means average optimality gap of 24.04% across all 1000 missions (where VBS is valid)
- `universal_ok: False` means the candidate is not universal (failed on some missions)

---

## 6. Cluster Resource Requirements

### 6.1 SLURM Configuration

**Default** (`scripts/eval_capstor_universal_search.slurm`):
- **Partition**: `normal`
- **Time**: 6 hours
- **Nodes**: 1
- **CPUs**: 64
- **Memory**: Default (typically sufficient for code evaluation)

### 6.2 Typical Runtime

- **Round 1 (Tournament)**: ~30–60 minutes (evaluating ~10k candidates on 30 missions)
- **Round 2 (Verification)**: ~10–20 minutes (evaluating 10 survivors on 1000 missions)
- **Total**: ~1–2 hours per seed

### 6.3 Storage

- **Input**: `generations.jsonl` files (~1.1GB each, 3 seeds = ~3.3GB total)
- **Output**: ~10–50MB per seed (CSV files, JSON configs, winner code)

---

## 7. Integration with Evaluation Pipeline

### 7.1 Relationship to Base Model Evaluation

Universal search **depends on** Base Model (Best-of-64) evaluation:

1. **Input**: `generations.jsonl` from Base Model evaluation
2. **VBS Reference**: `metrics_final.csv` from Base Model evaluation
3. **Test Dataset**: Same seed as Base Model evaluation

### 7.2 Aggregating Results Across Seeds

After running universal search for all seeds (101, 202, 303), you can aggregate the results using the dedicated aggregation script:

**Script**: `evaluation/sds/aggregate_universal_search.py`

**Usage**:
```bash
conda activate llm-finetuning
python evaluation/sds/aggregate_universal_search.py --batch-id 20260120_minimal-feas-v1
```

**Output**: Creates `evaluation/sds/universal_search_batches/<BATCH_ID>/aggregated/` containing:
- `aggregated_summary.json`: Summary statistics across all seeds
- `aggregated_report.md`: Human-readable markdown report
- `combined_topK.csv`: Combined `final_topK.csv` from all seeds

**Summary Statistics**:
- Total unique candidates across all seeds
- Total tournament survivors
- Total final evaluated
- Universal winners found (0/3 in our case)
- Best candidate performance (mean gap, max gap, feasible rate, timeout rate)
- Per-seed best mean gap

**Interpretation**: The aggregation helps answer:
- Did any seed find a universal winner? (Typically: No)
- What's the best single-code performance across all seeds?
- How does it compare to Base Best-of-64 aggregate performance?

### 7.3 Relationship to Main Aggregation

Universal search results are **not** automatically aggregated into the main SDS results tables/plots (`evaluation/sds/aggregate_plots.py`). They are a separate analysis to answer the question: *"Can we find a single universal code that matches Best-of-64 performance?"*

If you want to include the universal solver in the main efficiency frontier plot, you would need to:
1. Extract `winner_code.py` from the search results (or use the best candidate from `final_topK.csv` if no universal winner)
2. Evaluate it on the test set using `evaluation/sds/evaluate.py` (to get proper cost metrics)
3. Add it to the aggregation pipeline (`evaluation/sds/aggregate_plots.py`)

---

## 8. Troubleshooting

### 8.1 "No survivors found"

**Symptom**: `result.json` shows `"status": "no_survivors"`

**Possible causes**:
- Tournament missions are too strict (all candidates fail)
- Timeout too short (default: 5s)
- Base Model generations are low quality

**Solutions**:
- Increase `--tournament-missions` (more coverage)
- Increase `--timeout` (allow slower codes)
- Check Base Model evaluation quality

### 8.2 "No universal winner"

**Symptom**: `result.json` shows `"status": "no_universal_winner"`

**Possible causes**:
- Top candidates fail on some test instances
- Timeout too strict for some hard instances

**Solutions**:
- Check `final_topK.csv` to see which missions failed
- Increase `--timeout` if failures are due to slow execution
- Consider using the best candidate anyway (even if not "universal")

### 8.3 File not found errors

**Symptom**: `ERROR: no generations.jsonl files matched`

**Solutions**:
- Verify `--base-root` path is correct
- Check that `generations.jsonl` exists (may be in `job-*/` subfolder)
- Use absolute paths on cluster

### 8.4 Out of memory

**Symptom**: Job fails with memory errors

**Solutions**:
- Reduce `--workers` (fewer parallel evaluations)
- Reduce `--keep-per-round` (fewer candidates in memory)
- Process candidates in batches (modify script)

---

## 9. Future Work

### 9.1 Potential Improvements

1. **Multi-round tournaments**: Progressive elimination over multiple rounds
2. **Ensemble selection**: Combine top-K codes instead of single winner
3. **Code mutation**: Apply lightweight mutations to top candidates
4. **Difficulty-aware selection**: Prioritize codes that solve hard instances

### 9.2 Integration Ideas

1. **Automatic aggregation**: Include universal solver in main results tables
2. **W&B logging**: Log tournament progress and results
3. **Visualization**: Plot tournament elimination curves

---

## 10. References

- **Main Script**: `evaluation/sds/universal_solver_search.py`
- **Aggregation Script**: `evaluation/sds/aggregate_universal_search.py`
- **SLURM Wrapper**: `scripts/eval_capstor_universal_search.slurm`
- **Batch Launcher**: `scripts/run_universal_search_base.sh`
- **Evaluation Utilities**: `evaluation/sds/utils.py`
- **Base Model Evaluation**: See [PASS_AT_K_EVALUATION_REPORT.md](PASS_AT_K_EVALUATION_REPORT.md)

---

## Appendix: Quick Reference

### Find Base Model generations on cluster

```bash
find /capstor/scratch/{CLUSTER}/{USER} -path '*qwen2.5-coder-14b*base*' -name generations.jsonl
```

### Run universal search (all seeds)

```bash
./scripts/run_universal_search_base.sh \
  --base-root /capstor/scratch/{CLUSTER}/{USER}/llm-finetuning/evaluation/sds/results/qwen2.5-coder-14b/base \
  --batch-id <BATCH_ID>
```

### Check results

```bash
cat evaluation/sds/universal_search_batches/<BATCH_ID>/seed101/result.json
cat evaluation/sds/universal_search_batches/<BATCH_ID>/seed101/final_topK.csv
```

### Aggregate results across seeds

```bash
conda activate llm-finetuning
python evaluation/sds/aggregate_universal_search.py --batch-id <BATCH_ID>
cat evaluation/sds/universal_search_batches/<BATCH_ID>/aggregated/aggregated_report.md
```
