# SDS Evaluation Pipeline: Technical Report

## Executive Summary

This report describes the complete evaluation pipeline for assessing model performance on the Synergistic Dependency Selection (SDS) problem. We detail the evaluation workflow, metrics computation, statistical analysis (Pass@k), visualization generation, and aggregation across seeds and experiments. The pipeline supports evaluation of base models, fine-tuned models, and ShinkaEvolve-generated solutions.

---

## 1. Overview

### 1.1 Purpose

The evaluation pipeline provides comprehensive assessment of SDS problem-solving capabilities:

1. **Feasibility Validation**: Checks constraint satisfaction (cardinality, mutex, groups, precedence)
2. **Quality Metrics**: Computes optimality gaps against Virtual Best Solver (VBS)
3. **Scalability Analysis**: Pass@k analysis for best-of-N sampling
4. **Robustness Profiling**: Performance across difficulty levels
5. **Comparative Analysis**: Comparison against baseline solvers (Greedy, Local Search, BnB, CP-SAT)

### 1.2 Evaluation Modes

**Mode 1: Base Model Evaluation**
- Generates N=64 completions per problem
- Performs Pass@k bootstrap analysis
- Evaluates scaling laws (capacity vs coverage)

**Mode 2: Fine-Tuned Model Evaluation**
- Single generation per problem (deterministic)
- Direct comparison with baselines
- Stratified analysis by difficulty

**Mode 3: ShinkaEvolve Evaluation**
- Evaluates evolved code solutions
- Can test on test dataset or own dataset
- Validates solution quality

---

## 1.3 Experiment Management (Batches + Report Sets)

To ensure **paper reproducibility** and avoid the “latest job takes over” failure mode, we separate:

- **Moving results (default / legacy)**: `evaluation/sds/results/`
- **Frozen batch results**: `evaluation/sds/results_batches/<BATCH_ID>/`
- **Explicit report sets**: `experiments/report_sets/<name>.json`

### Batch-aware evaluation outputs

Evaluation jobs can write outputs under a specific batch by passing:

```bash
./scripts/evaluate_14b_grpo_experiments.sh --batch-id <BATCH_ID>
```

When `--batch-id` is omitted, evaluation outputs go to the legacy moving directory. This is backward compatible with historical runs.

### Paper-safe aggregation via report sets

Aggregation supports explicit report sets so that tables/plots are generated from a fixed set of frozen roots:

```bash
python evaluation/sds/aggregate_plots.py --report-set experiments/report_sets/paper_main_results_v1.json --model-filter qwen2.5-coder-14b
```

### Baseline-only batches (recommended)

Base (Best-of-64) and ShinkaEvolve are often reused across many RL variants. To prevent accidental leakage of old RL jobs into a new revision, we recommend keeping a **baseline-only batch** containing only:

- `Base (Best-of-64)` (SDS)
- `ShinkaEvolve` (SDS)

and composing revisions by including:

- new RL batch root (Hero/ablations)
- baseline-only batch root (Base/Shinka only)

This guarantees old RL jobs are not aggregated.

---

## 2. Evaluation Workflow

### 2.1 Input Format

**JSONL File Structure**:
```json
{
  "uuid": "problem_000001",
  "mission": {
    "n_variables": 20,
    "cardinality_bounds": [5, 14],
    "precedence": [[0, 5], [2, 8]],
    "mutex": [[1, 3], [4, 7]],
    "groups": {"group_0": [0, 1, 2]},
    "weights": [1.2, -0.8, ...],
    "interactions": {"0,5": 12.3, "1,3": -8.7, ...}
  },
  "generated_text": "<think>...\n</think>\n\n<code>...\n</code>"
}
```

### 2.2 Evaluation Process

**Step 1: Load and Parse**
```python
# Load JSONL file
records = [json.loads(line) for line in open(input_file)]

# Extract problem instance
instance = mission_to_instance(record['mission'])

# Extract code from generated_text
code = extract_block(generated_text, "code")
```

**Step 2: Run Baseline Solvers**
```python
for solver_name in ["greedy", "local_search", "bnb", "cpsat"]:
    solver = get_solver(solver_name)
    result = solver.solve(instance, budget_sec=time_budget, seed=seed)
    
    # Validate feasibility
    is_feasible = feasible(instance, result.mask)
    score = score(instance, result.mask) if is_feasible else float("-inf")
    
    metrics[solver_name] = {
        "score": score,
        "time": result.time_sec,
        "feasible": is_feasible
    }
```

**Step 3: Execute LLM Code**
```python
# Reconstruct input JSON
stdin_obj = {
    "requirements": {
        "n_variables": n_vars,
        "cardinality_bounds": [L, U],
        "precedence": precedence,
        "mutex": mutex,
        "groups": groups,
        "weights": weights,
        "interactions": interactions
    },
    "catalog": {
        "variables": [...],
        "interactions": {...}
    }
}

# Execute with timeout (5.0s default)
result = run_candidate(code, stdin_obj, timeout=5.0)

# Extract selection
selection = result["selection"]["variables"]

# Validate constraints
violations = check_constraint_violations(instance, selection)

# Calculate score
llm_score = calculate_true_score(instance, selection) if violations["all_valid"] else float("-inf")
```

**Step 4: Compute Metrics**
```python
# Virtual Best Solver (VBS)
vbs_score = max(
    llm_score if is_feasible else float("-inf"),
    max(solver_scores.values())
)

# Optimality Gap
if vbs_score > 1e-9:
    gap = (vbs_score - max(0, llm_score)) / vbs_score
else:
    gap = np.nan  # Invalid VBS

# Pass Rate (binary: feasible or not)
pass_rate = 1 if is_feasible else 0

# Cost (core-seconds)
cost = execution_time * num_cores
```

### 2.3 Parallel Processing

**Worker Pool**:
```python
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = [
        executor.submit(evaluate_single_sample, line, idx, ...)
        for idx, line in enumerate(input_lines)
    ]
    
    results = [future.result() for future in as_completed(futures)]
```

**Isolation**:
- Each worker processes problems independently
- Solver instances created per worker (process-safe)
- No shared state between workers

---

## 3. Metrics Computation

### 3.1 Core Metrics

**Feasibility**:
- **Binary**: `feasible = True` if all constraints satisfied
- **Violation Count**: Number of constraint violations
  - Cardinality: `|selection| < L` or `|selection| > U`
  - Precedence: Missing dependencies
  - Mutex: Conflicting pairs selected
  - Groups: Multiple items from same group

**Score**:
```python
score = sum(weights[selected]) + sum(interactions[pairs])
```
- **Preserves Negative Scores**: Important for optimization landscape
- **Infeasible Solutions**: Score = `-inf`

**Optimality Gap**:
```python
gap = (VBS - MethodScore) / VBS
```
- **Range**: [0.0, 1.0] (0% = optimal, 100% = worst)
- **VBS**: Maximum score across all methods (LLM + baselines)
- **Handles Negative Scores**: Clips to 0.0 for gap calculation
- **Infeasible Solutions**: Assigned gap = 1.0 (maximum penalty)
- **Reporting**: Mean gaps in tables are calculated only on feasible solutions to decouple validity failure from optimization quality. However, infeasible solutions (gap = 1.0) are included in robustness profiles and stratified boxplots to reflect true system reliability.

**Cost**:
```python
cost = execution_time * num_cores
```
- **Units**: Core-seconds
- **LLM**: `execution_time * 1` (single-threaded)
- **Baselines**: `wall_time * cores_used` (multi-threaded for CP-SAT)

### 3.2 Difficulty Classification

The evaluation pipeline classifies problem instances into three difficulty categories (**Trivial**, **Moderate**, **Hard**) based on how well a simple greedy heuristic performs relative to the best-known solution. This classification enables stratified analysis to understand how different methods perform across varying problem complexity.

#### 3.2.1 Virtual Best Solver (VBS) Calculation

The **Virtual Best Solver (VBS)** represents the best score achieved by any solver (LLM or baseline) on a given problem instance. There are two levels of VBS calculation:

**Per-Experiment VBS (Individual Evaluation)**:
```python
# Collect all feasible scores for this experiment
scores = []
if llm_feasible:
    scores.append(llm_score)
for baseline_name in baselines:
    if baseline_feasible[baseline_name]:
        scores.append(baseline_score[baseline_name])

# VBS is the maximum score found
vbs = max(scores) if scores else float('-inf')
```

**Global VBS (Aggregation)**:
When aggregating results across multiple methods and experiments, we compute a **global VBS** per problem instance to ensure fair comparison:

```python
# After loading all data from all methods
for (uuid, seed), group in final_df.groupby(['uuid', 'Seed']):
    # Collect all feasible scores across ALL methods (LLM + baselines)
    feasible_scores = group[group['feasible'] == True]['llm_score'].dropna()
    
    if len(feasible_scores) > 0:
        global_vbs = feasible_scores.max()
        # Recalculate gaps for all methods using this global VBS
        for method in group['Method'].unique():
            method_df = group[group['Method'] == method]
            method_df['Gap'] = (global_vbs - method_df['llm_score']) / global_vbs
```

**Key Points**:
- Only **feasible** solutions are considered (infeasible solutions are excluded)
- VBS can be negative (if all solutions have negative objective values)
- If all solvers fail, VBS is set to `-inf` and the instance is classified as "Hard"
- **Global VBS ensures consistency**: When comparing methods, all methods use the same VBS (the maximum across all methods), preventing methods from appearing artificially optimal simply because they found a better solution than deterministic baselines but worse than the true optimum
- **Critical for Base (Best-of-64)**: The global VBS includes the union of all $N=64$ Base model samples, ensuring that the Base model's optimality gap is calculated against the true best solution found across all methods, not just against deterministic baselines

#### 3.2.2 Hardness Metric Calculation

The **hardness** metric measures the relative gap between the greedy baseline and the VBS:

```python
# Standard case: both VBS and Greedy are feasible
epsilon = 1e-10
numerator = vbs - greedy_score
denominator = abs(vbs) + epsilon
hardness = numerator / denominator
```

**Edge Cases**:
- **All solvers fail**: `hardness = 1.0` (maximum difficulty)
- **Greedy fails but VBS succeeds**: `hardness = 1.0` (greedy completely failed)
- **VBS ≤ 0**: Uses `abs(vbs)` in denominator to handle negative objectives

**Interpretation**:
- `hardness = 0.0`: Greedy achieves the optimal solution (VBS)
- `hardness = 0.05`: Greedy is 5% worse than optimal
- `hardness = 1.0`: Greedy failed completely or all solvers failed

#### 3.2.3 Difficulty Classification Thresholds

Instances are classified into three categories based on hardness:

```python
def classify_diff(hardness):
    if hardness < 0.01:
        return "Trivial"   # <1% gap (Greedy ≈ Optimal)
    elif hardness < 0.10:
        return "Moderate"  # 1-10% gap
    else:
        return "Hard"       # ≥10% gap (Greedy failed significantly)
```

**Categories**:

1. **Trivial** (`hardness < 0.01`):
   - Greedy heuristic performs nearly optimally (<1% gap from VBS)
   - These instances are easy to solve and don't require sophisticated search strategies
   - Example: Simple instances where greedy selection naturally finds good solutions

2. **Moderate** (`0.01 ≤ hardness < 0.10`):
   - Greedy heuristic has a moderate gap (1-10%) from optimal
   - These instances benefit from improved search strategies but are not extremely challenging
   - Example: Instances with some local optima that can be escaped with basic local search

3. **Hard** (`hardness ≥ 0.10`):
   - Greedy heuristic fails significantly (≥10% gap) or completely fails
   - These instances have deceptive landscapes that trap greedy algorithms
   - Example: Instances with strong pairwise interactions where greedy selection gets stuck in poor local optima

**Special Case**: Instances where all solvers fail (including greedy) are automatically classified as "Hard" (`hardness = 1.0`), as they represent unsolvable or extremely challenging problems.

#### 3.2.4 Usage in Stratified Analysis

The difficulty classification is used to:

1. **Stratified Box Plots**: Visualize optimality gap distributions across difficulty levels (see Section 5.3)
2. **Performance Breakdown**: Understand how methods perform on easy vs. hard instances
3. **Difficulty Transfer Assessment**: Evaluate whether learned strategies transfer from easy to hard problems

**Example**: The manuscript's Figure~\ref{fig:stratified} shows that greedy heuristics collapse on Hard instances (median gap >30%), while the RL-trained policy maintains a median gap <5%, demonstrating that learned search strategies generalize to unseen structural complexity.

### 3.3 Error Classification

**Error Types**:
- **`none`**: Valid solution (feasible)
- **`syntax`**: Python syntax error
- **`runtime`**: Runtime exception during execution
- **`timeout`**: Execution exceeded 5.0s limit
- **`json`**: Invalid JSON output format
- **`security`**: Security violation (restricted imports)
- **`missing_code`**: No `<code>` block found
- **`constraint`**: Constraint violation (infeasible solution)
- **`unknown`**: Unclassified error

**Failure Types** (for analysis):
- **Success**: Feasible solution
- **Timeout**: `error_type == 'timeout'` or `execution_time > 4.9s`
- **Logic Error**: All other failures (syntax, constraint, runtime, etc.)

---

## 4. Pass@k Analysis

### 4.1 Purpose

Pass@k analysis answers: **"If we only use k out of N samples, how well does the model perform?"**

This is critical for:
- Understanding scaling laws (capacity vs coverage)
- Resource planning (how many samples needed)
- Comparing with code generation benchmarks

### 4.2 Bootstrap Procedure

**For each k ∈ {1, 2, 4, 8, 16, 32, 64}**:

1. **Bootstrap Iterations** (n=500):
   ```python
   for bootstrap_iter in range(500):
       for problem_uuid in problems:
           # Sample k solutions (without replacement)
           sample = group.sample(n=k, replace=False)
           
           # Pass: At least one feasible?
           is_passed = sample['feasible'].any()
           
           # Best Score: Maximum of feasible solutions
           feasible_scores = sample[sample['feasible']]['llm_score']
           best_score = feasible_scores.max() if not feasible_scores.empty else 0.0
           
           # Optimality Gap
           gap = (VBS - best_score) / VBS if VBS > 0 else 1.0
   ```

2. **Aggregation**:
   ```python
   # After 500 iterations, compute statistics
   pass_rate_mean = np.mean(bootstrap_pass_rates) * 100
   pass_rate_std = np.std(bootstrap_pass_rates) * 100
   
   opt_gap_mean = np.mean(bootstrap_gaps) * 100
   opt_gap_std = np.std(bootstrap_gaps) * 100
   ```

### 4.3 Scaling Plots

**Plot 1: Optimality Gap vs k** (`scaling_gap_vs_k.png`):
- X-axis: Number of generations (k), log scale
- Y-axis: Optimality gap (%), with error bars
- Shows: How gap decreases with more samples

**Plot 2: Pass Rate vs k** (`scaling_pass_vs_k.png`):
- X-axis: Number of generations (k), log scale
- Y-axis: Pass rate (%), with error bars
- Shows: How coverage increases with more samples

**Interpretation**:
- **Steep curves**: Model benefits significantly from multiple samples
- **Flat curves**: Model is consistent (single sample sufficient)
- **Error bars**: Bootstrap confidence intervals

---

## 5. Visualization

### 5.1 Individual Evaluation Plots

**Plot 1: Robustness Profile** (`robustness_profile.png`)

**Purpose**: Shows fraction of problems solved within a given optimality gap threshold

**Method**:
```python
# For each gap threshold tau (0% to 50%)
taus = np.linspace(0.0, 0.5, 500)

# For each method
for method in methods:
    gaps = df[df['Method'] == method]['llm_gap'].values
    
    # CDF: Fraction with gap <= tau
    y = np.mean(gaps[:, None] <= taus[None, :], axis=0)
    
    plt.plot(taus, y, label=method)
```

**Interpretation**:
- **Higher curves**: Better performance (more problems solved at low gap)
- **Steep initial rise**: Many problems solved near-optimally
- **Plateau**: Hard problems remain unsolved
- **Note**: Infeasible solutions are included with gap = 1.0, affecting the fraction solved at high gap thresholds

**Plot 2: Stratified Box Plot** (`stratified_boxplot.png`)

**Purpose**: Shows distribution of optimality gaps across difficulty levels

**Note**: Infeasible solutions are included with gap = 1.0, appearing as outliers or affecting the distribution

**Structure**:
- X-axis: Difficulty (Trivial, Moderate, Hard)
- Y-axis: Optimality gap (%)
- Box plots: Median, quartiles, outliers per method

**Special Handling**:
- **CP-SAT**: Plotted as horizontal line at 0% (always optimal)
- **Other methods**: Standard box plots

**Interpretation**:
- **Lower boxes**: Better performance
- **Wide boxes**: High variance (inconsistent)
- **Outliers**: Exceptional cases (very good or very bad)

**Plot 3: Error Distribution** (`error_distribution.png`)

**Purpose**: Visualizes failure modes

**Format**: Horizontal bar chart
- Y-axis: Error types (None, Syntax, Runtime, Timeout, Constraint, etc.)
- X-axis: Count of occurrences
- Colors: Green (valid), Red (errors), Orange (timeout)

**Interpretation**:
- **Large "None" bar**: High success rate
- **Large "Constraint" bar**: Many infeasible solutions
- **Large "Timeout" bar**: Code too slow

### 5.2 Aggregate Plots (Cross-Seed)

**Plot 1: Efficiency Frontier** (`fig1_efficiency.png`)

**Purpose**: Pareto frontier of optimality gap vs inference cost

**Method**:
```python
# Aggregate across seeds: mean ± std
per_seed = df.groupby(['Method', 'Seed']).agg({
    'Gap': 'mean',
    'Cost': 'mean'
})

agg = per_seed.groupby('Method').agg(['mean', 'std'])

# Plot with error bars
plt.errorbar(agg[('Cost', 'mean')], agg[('Gap', 'mean')] * 100,
             xerr=agg[('Cost', 'std')], yerr=agg[('Gap', 'std')] * 100)
```

**Axes**:
- X-axis: Inference cost (core-seconds), log scale
- Y-axis: Optimality gap (%), linear scale

**Interpretation**:
- **Bottom-left**: Best (low gap, low cost)
- **Trade-off curve**: Pareto frontier
- **Error bars**: Seed-to-seed variation

**Plot 2: Robustness Profile** (`fig2_robustness.png`)

**Purpose**: Same as individual plot, but aggregated across seeds

**Enhancement**: Includes shaded regions (mean ± std across seeds)

**Plot 3: Stratified Box Plot** (`fig3_stratified.png`)

**Purpose**: Same as individual plot, but aggregated across seeds

**Enhancement**: Shows all seeds combined (broader distribution)

**Plot 4: Failure Analysis** (`fig4_failure.png`)

**Purpose**: Stacked bar chart of failure modes

**Structure**:
- X-axis: Methods (Hero, ablations, ShinkaEvolve)
- Y-axis: Failure rate (%)
- Stacked bars: Logic Error (red) + Timeout (orange)

**Interpretation**:
- **Tall bars**: High failure rate
- **Red dominance**: Logic errors (syntax, constraints)
- **Orange dominance**: Timeouts (slow code)

---

## 6. LaTeX Tables

### 6.1 Performance Table (`final_results_table.tex`)

**Structure**:
```
Method | Overall Gap | Trivial Gap | Moderate Gap | Hard Gap | Pass Rate
```

**Format**: Mean ± Std across seeds

**Example**:
```latex
\begin{tabular}{lccccc}
\toprule
Method & Overall & Trivial & Moderate & Hard & Pass Rate \\
\midrule
Ours (Hero) & $12.3_{\pm 0.5}$ & $2.1_{\pm 0.1}$ & $15.6_{\pm 0.8}$ & $28.9_{\pm 1.2}$ & $97.8_{\pm 0.1}$ \\
CP-SAT & $0.0$ & $0.0$ & $0.0$ & $0.0$ & $100.0$ \\
\bottomrule
\end{tabular}
```

### 6.2 Error Types Table (`error_types_table.tex`)

**Structure**:
```
Method | None | Timeout | Constraint | Syntax | Runtime | ...
```

**Format**: Percentage of each error type (mean ± std)

**Purpose**: Detailed failure mode analysis

### 6.3 Stratified Table (`results_stratified.tex`)

**Structure**: Same as performance table, but per-difficulty breakdown

**Purpose**: Understand performance across difficulty levels

---

## 7. Aggregation Pipeline

### 7.1 File Discovery

**Process**:
```python
# Recursively find all metrics_final.csv files
pattern = "evaluation/sds/results/**/metrics_final.csv"
files = glob.glob(pattern, recursive=True)
```

**Path Parsing**:
```python
# Extract metadata from path
# e.g., evaluation/sds/results/qwen2.5-coder-14b/grpo-config_hero/seed101/job-12345/metrics_final.csv

def parse_path_metadata(path):
    # Extract method (Hero, ablation, ShinkaEvolve)
    # Extract seed (101, 202, 303)
    # Extract model (qwen2.5-coder-14b)
    # Extract job_id (12345)
    return method, seed, model, job_id
```

### 7.2 Data Loading

**Process**:
```python
for csv_path in files:
    df = pd.read_csv(csv_path)
    method, seed, model, job_id = parse_path_metadata(csv_path)
    
    # Add metadata columns
    df['Method'] = method
    df['Seed'] = seed
    df['Model'] = model
    df['JobID'] = job_id
    
    # Calculate metrics if missing
    if 'vbs_score' not in df.columns:
        df['vbs_score'] = calculate_vbs(df)
        df['difficulty_class'] = classify_difficulty(df)
    
    # Calculate gap
    df['Gap'] = (df['vbs_score'] - df['llm_score']) / df['vbs_score']
    
    all_dfs.append(df)
```

### 7.3 Aggregation

**Step 1: Global VBS Calculation**:
Before aggregating, we compute a global VBS per problem instance to ensure all methods are compared fairly:

```python
# After concatenating all dataframes from all methods
final_df = pd.concat(all_dfs, ignore_index=True)

# Calculate global VBS per (uuid, seed) pair
for (uuid, seed), group in final_df.groupby(['uuid', 'Seed']):
    # Collect all feasible scores across ALL methods
    feasible_scores = group[group['feasible'] == True]['llm_score'].dropna()
    
    if len(feasible_scores) > 0:
        global_vbs = feasible_scores.max()
        # Update VBS for all rows in this group
        final_df.loc[group.index, 'vbs_score'] = global_vbs
        # Recalculate gaps using global VBS
        final_df.loc[group.index, 'Gap'] = (
            (global_vbs - final_df.loc[group.index, 'llm_score'].clip(lower=0.0)) 
            / global_vbs
        )
```

**Why Global VBS is Critical**:
- **Prevents artificial optimality**: Without global VBS, a method might appear optimal simply because it found a better solution than deterministic baselines, even if another method found an even better solution
- **Fair comparison**: All methods are compared against the same reference (the true best solution found across all methods)
- **Base (Best-of-64) accuracy**: The global VBS includes the union of all $N=64$ Base model samples, ensuring accurate gap calculation

**Step 2: Aggregation Across Seeds**:
```python
# First: Mean per seed per method
per_seed_pass = df.groupby(['Method', 'Seed'])['Pass'].mean().reset_index()
per_seed_gap_cost = df[df['Gap'].notna()].groupby(['Method', 'Seed']).agg({
    'Gap': 'mean',
    'Cost': 'mean'
}).reset_index()

# Second: Mean/Std across seeds
agg = per_seed_gap_cost.groupby('Method').agg({
    'Gap': ['mean', 'std'],
    'Cost': ['mean', 'std']
})

# Format: mean ± std
gap_mean = agg[('Gap', 'mean')] * 100
gap_std = agg[('Gap', 'std')] * 100
```

**Stratified Aggregation**:
```python
# Group by Method and difficulty_class
strat_agg = df.groupby(['Method', 'difficulty_class']).agg({
    'Gap': ['mean', 'std'],
    'Pass': 'mean'
})
```

### 7.4 Job Selection

**Critical Requirement**: We must ensure that **all 3 seeds (101, 202, 303) are included for each method** to maintain statistical validity. The aggregation pipeline uses `jobs_per_seed=1` by default to guarantee this.

**ShinkaEvolve Special Handling**:
- ShinkaEvolve files are filtered to only include those matching the `ShinkaEvolve-SDS-1000-seed{seed}/seed{seed}/test/metrics_final.csv` pattern
- This ensures only the correct 1000-row files are included, excluding other evaluation files (e.g., `SDS-100`, `own-dataset`, `shinka/`)
- **Bug Fix**: Previously, the aggregation logic was including all ShinkaEvolve files for seed 303 (4 files instead of 1), leading to 1300 rows instead of 1000. This was fixed by filtering files to only include those matching the `ShinkaEvolve-SDS-1000` pattern in the path.

**Job Selection Strategy**:
```python
# For each (method, seed, model) combination
experiment_groups = defaultdict(list)

for file in files:
    method, seed, model, job_id = parse_path_metadata(file)
    key = (method, seed, model)
    experiment_groups[key].append((file, job_id))

# Select jobs per (method, seed) pair
for key, job_files in experiment_groups.items():
    job_files.sort(key=lambda x: x[1], reverse=True)  # Sort by job_id
    
    if jobs_per_seed is not None:
        # Select N latest jobs per (method, seed) - ensures all seeds included
        for file_path, job_id in job_files[:jobs_per_seed]:
            selected_files.append((file_path, method, seed, model, job_id))
    else:
        # Default: take latest job (but may exclude some seeds if max_jobs limit hit)
        selected_files.append((job_files[0][0], method, seed, model, job_files[0][1]))
```

**Default Behavior (jobs_per_seed=1)**:
The aggregation script defaults to `jobs_per_seed=1` to ensure all 3 seeds are included for each method:

```python
# In aggregate_plots.py main()
jobs_per_seed = args.jobs_per_seed if args.jobs_per_seed is not None else 1
selected_files = select_latest_jobs(
    all_files,
    max_jobs=args.max_jobs,
    jobs_per_seed=jobs_per_seed,  # Default: 1 (ensures all seeds)
    specific_job_ids=SPECIFIC_14B_JOB_IDS
)
```

**Why This Matters**:
- **Without `jobs_per_seed=1`**: If `max_jobs` is too small (e.g., 15), the selection might exclude some seeds (e.g., seed 101 with oldest modification time), leading to incorrect aggregated results (e.g., mean execution time calculated from only 2 seeds instead of 3)
- **With `jobs_per_seed=1`**: Guarantees exactly 1 job per (method, seed) pair, ensuring all 3 seeds are included regardless of `max_jobs` limit
- **Result**: Accurate aggregation with all 3 seeds for all methods (Hero, 4 ablations, Base, ShinkaEvolve)

**Specific Job IDs** (for 14B experiments):
```python
SPECIFIC_14B_JOB_IDS = [
    "1315159", "1315160", "1315161", "1315162", "1315163",  # Seed 101
    "1315164", "1315165", "1315166", "1315167", "1315168",  # Seed 202
    "1315169", "1315170", "1315171", "1315172", "1315173"   # Seed 303
]

# Filter to only these jobs
selected_files = [f for f in files if job_id in SPECIFIC_14B_JOB_IDS]
```

**Verification**:
After selection, verify that all methods have all 3 seeds:
```python
# Check seed coverage
for method in methods:
    seeds = df[df['Method'] == method]['Seed'].unique()
    assert len(seeds) == 3, f"{method} missing seeds! Found: {seeds}"
```

---

## 8. Output Structure

### 8.1 Individual Evaluation Output

```
evaluation/sds/results/qwen2.5-coder-14b/grpo-config_hero/seed101/job-12345/
├── generations.jsonl              # Raw evaluation records
├── metrics_final.csv              # Aggregated metrics per problem
├── results_summary.json            # Summary statistics
├── pass_at_k_analysis.json        # Pass@k bootstrap results
├── baseline_comparison.json        # Baseline solver metrics
├── scaling_gap_vs_k.png/pdf       # Gap scaling plot
├── scaling_pass_vs_k.png/pdf      # Pass rate scaling plot
├── robustness_profile.png/pdf     # Robustness CDF
├── stratified_boxplot.png/pdf     # Difficulty-stratified box plot
├── error_distribution.png/pdf      # Error type distribution
└── results_stratified.tex         # LaTeX table (stratified)
```

### 8.2 Aggregate Output

```
evaluation/sds/aggregated_report/
├── final_results_table.tex         # Main performance table
├── error_types_table.tex           # Error analysis table
├── fig1_efficiency.png/pdf         # Efficiency frontier
├── fig2_robustness.png/pdf         # Robustness profile
├── fig3_stratified.png/pdf         # Stratified box plot
└── fig4_failure.png/pdf            # Failure analysis
```

### 8.3 CSV Structure

**metrics_final.csv**:
```csv
uuid,llm_score,feasible,error_type,execution_time,vbs_score,difficulty_class,
score_greedy,score_local_search,score_cpsat,score_bnb,
time_greedy,time_local_search,time_cpsat,time_bnb,
feasible_greedy,feasible_local_search,feasible_cpsat,feasible_bnb,
llm_gap,greedy_gap,local_search_gap,cpsat_gap,bnb_gap
```

**Key Columns**:
- `uuid`: Problem identifier
- `llm_score`: LLM solution score
- `feasible`: Binary feasibility flag
- `vbs_score`: Virtual Best Solver score
- `difficulty_class`: Trivial/Moderate/Hard
- `*_gap`: Optimality gap for each method

---

## 9. Running Evaluations

### 9.1 Base Model Evaluation

```bash
# Generate 64 completions per problem
python evaluation/sds/generate.py \
    --model qwen2.5-coder-14b \
    --output_file evaluation/sds/generations.jsonl \
    --num_samples 64 \
    --seed 101

# Evaluate with Pass@k analysis
python evaluation/sds/evaluate.py \
    --input_file evaluation/sds/generations.jsonl \
    --output_dir evaluation/sds/results/qwen2.5-coder-14b/base/seed101 \
    --baselines greedy local_search bnb cpsat \
    --best-of-n
```

#### 9.1.1 Base Model Performance Characteristics

**Observed Performance:**
- **Pass Rate**: ~85.9% (vs. Hero: 99.8%)
- **Mean Optimality Gap**: ~19.0% (vs. Hero: 4.1%)
- **Best-of-64 Strategy**: Despite generating 64 samples per problem and selecting the best feasible solution, the base model significantly underperforms compared to fine-tuned models.

**Why Base Model Performs Poorly:**

1. **Low Feasibility Rate**: Only ~85.9% of generated solutions satisfy all constraints. Even with 64 attempts, many problems have no feasible solutions, resulting in automatic failure.

2. **Poor Solution Quality**: Among feasible solutions, the base model's code quality is substantially lower than fine-tuned models. The untrained model lacks the specialized reasoning patterns learned during GRPO training, leading to:
   - Suboptimal algorithmic choices (e.g., naive greedy heuristics instead of meta-heuristics)
   - Inefficient search strategies
   - Poor constraint handling

3. **Limited Capacity**: While the base model possesses latent capacity (as evidenced by occasional high-quality solutions), it cannot reliably access this capacity without training. The best-of-64 strategy helps but cannot compensate for the fundamental lack of specialized knowledge.

**CP-SAT Gap Discrepancy in Base Model Evaluation:**

An interesting observation is that CP-SAT's optimality gap appears slightly worse in base model evaluation plots (~0.98% ± 6.52%) compared to hero/ablation evaluations (~0.01% ± 0.14%). This discrepancy is **not a bug** but rather a consequence of how VBS is calculated:

1. **VBS Calculation**: VBS = max(LLM_best_score, CP-SAT_score, other_baselines)

2. **Base Model Occasionally Beats CP-SAT**: In base model evaluation, the best-of-64 strategy occasionally finds solutions that outperform CP-SAT on ~9.3% of problems. When this happens:
   - VBS = base_model_best_score > CP-SAT_score
   - CP-SAT gap = (VBS - CP-SAT) / VBS > 0

3. **Hero Evaluation**: In hero/ablation evaluations, the single-generation models rarely beat CP-SAT (only ~0.9% of problems), so CP-SAT is almost always the VBS, resulting in gap ≈ 0.

**Conclusion**: The base model's poor performance (19% gap, 85.9% pass rate) is real and expected for an untrained model. The best-of-64 strategy provides some benefit but cannot overcome the fundamental limitations of an untrained model. The CP-SAT gap discrepancy reflects the base model's occasional success rather than an evaluation bug.

### 9.2 Fine-Tuned Model Evaluation

```bash
# Evaluate checkpoint
python evaluation/sds/evaluate.py \
    --model qwen2.5-coder-14b \
    --training-scheme grpo-config_hero \
    --seed 101 \
    --job-id 12345 \
    --baselines greedy local_search bnb cpsat
```

**Automatic Discovery**:
- Finds checkpoint: `/workspace/checkpoints/.../job-12345/checkpoint-90/`
- Generates completions: 1 per problem (deterministic)
- Saves to: `evaluation/sds/results/qwen2.5-coder-14b/grpo-config_hero/seed101/job-12345/`

### 9.3 ShinkaEvolve Evaluation

**On Test Dataset**:
```bash
python evaluation/sds/evaluate.py \
    --shinka-dataset {ORG}/ShinkaEvolve-SDS-100-seed303 \
    --seed 303
```

**On Own Dataset**:
```bash
python evaluation/sds/evaluate.py \
    --shinka-dataset {ORG}/ShinkaEvolve-SDS-100-seed303 \
    --evaluate-on-own-dataset
```

### 9.4 Plot-Only Mode

```bash
# Regenerate plots from existing results
python evaluation/sds/evaluate.py \
    --output_dir evaluation/sds/results/qwen2.5-coder-14b/grpo-config_hero/seed101/job-12345 \
    --plot-only
```

### 9.5 Aggregation

```bash
# Aggregate across all experiments
python evaluation/sds/aggregate_plots.py \
    --output-dir evaluation/sds/aggregated_report \
    --max-jobs 15 \
    --include-baselines
```

**Options**:
- `--max-jobs`: Maximum experiments to include (ignored if `--jobs-per-seed` is set)
- `--model-filter`: Filter by model (e.g., `qwen2.5-coder-14b`)
- `--jobs-per-seed`: Select N latest jobs per (method, seed) pair. **Default: 1** (ensures all 3 seeds are included for each method)
- `--include-baselines`: Include baseline solver results

**Important**: The script defaults to `jobs_per_seed=1` to ensure all 3 seeds (101, 202, 303) are included for each method. This prevents incorrect aggregation when `max_jobs` is too small and would otherwise exclude some seeds.

---

## 10. Running on Capstor Cluster

### 10.1 Individual Evaluation

```bash
# SSH to cluster
ssh capstor

# Navigate to project
cd /workspace/llm-finetuning

# Submit evaluation job
sbatch scripts/eval_capstor_sds_pipeline.slurm \
    --model qwen2.5-coder-14b \
    --training-scheme grpo-config_hero \
    --seed 101 \
    --job-id 12345
```

### 10.2 Batch Evaluation

```bash
# Evaluate all base, hero, and ablations
./scripts/evaluate_14b_grpo_experiments.sh
```

### 10.3 Plot Generation

```bash
# Generate plots from existing results
sbatch scripts/eval_sds_plot_only.slurm \
    --seed 101 \
    --output-dir evaluation/sds/results/qwen2.5-coder-14b/grpo-config_hero/seed101/job-12345
```

### 10.4 Check Results

```bash
# View summary
cat evaluation/sds/results/.../results_summary.json

# View Pass@k analysis
cat evaluation/sds/results/.../pass_at_k_analysis.json

# View plots
ls evaluation/sds/results/.../*.png
```

---

## 11. Statistical Methods

### 11.1 Bootstrap Resampling

**Purpose**: Estimate confidence intervals for Pass@k metrics

**Procedure**:
1. Sample k solutions per problem (without replacement)
2. Compute metrics (pass rate, gap)
3. Repeat 500 times
4. Compute mean ± std across bootstrap iterations

**Advantages**:
- No assumptions about distribution
- Handles correlation within problems
- Provides confidence intervals

### 11.2 Aggregation Across Seeds

**Mean ± Std**:
```python
# Per seed
per_seed = df.groupby(['Method', 'Seed']).agg({
    'Gap': 'mean',
    'Pass': 'mean'
})

# Across seeds
agg = per_seed.groupby('Method').agg(['mean', 'std'])

# Format: mean ± std
result = f"{agg[('Gap', 'mean')]:.1f} ± {agg[('Gap', 'std')]:.1f}"
```

**Interpretation**:
- **Low std**: Consistent across seeds (robust)
- **High std**: High variance (seed-dependent)

### 11.3 Stratified Analysis

**Purpose**: Understand performance across difficulty levels

**Method**:
```python
# Group by difficulty
for diff in ["Trivial", "Moderate", "Hard"]:
    subset = df[df['difficulty_class'] == diff]
    gap_mean = subset['Gap'].mean() * 100
    pass_rate = subset['Pass'].mean() * 100
```

**Benefits**:
- Identifies where methods excel/fail
- Reveals difficulty-specific patterns
- Guides problem generation

---

## 12. Plot Styling

### 12.1 Paper-Compatible Style

**Configuration**:
```python
plt.rcParams.update({
    "text.usetex": False,  # Set True if LaTeX installed
    "font.family": "serif",
    "font.serif": ["Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.figsize": (3.25, 2.5),  # 1-column width
    "figure.dpi": 300
})
```

**Color Palette**:
```python
PALETTE = {
    "LLM (Ours)": "#1f77b4",      # Blue
    "Local Search": "#2ca02c",     # Green
    "Greedy": "#d62728",           # Red
    "CP-SAT": "#ff7f0e",           # Orange
    "BnB": "#9467bd",              # Purple
    "Random": "#7f7f7f"            # Gray
}
```

### 12.2 File Formats

**Output Formats**:
- **PNG**: 300 DPI (for presentations, web)
- **PDF**: Vector format (for papers, LaTeX)

**Naming Convention**:
- Individual: `{plot_name}.png/pdf`
- Aggregate: `fig{N}_{plot_name}.png/pdf`

---

## 13. Best Practices

### 13.1 Evaluation

1. **Consistent Baselines**: Always include CP-SAT (optimal reference)
2. **Multiple Seeds**: Evaluate on 3 seeds (101, 202, 303) for robustness
3. **Full Test Set**: Use complete test set (1000 problems) for statistical power
4. **Timeout Settings**: Use 5.0s timeout (matches training)

### 13.2 Analysis

1. **Global VBS Calculation**: Always use global VBS (maximum across all methods) when aggregating results to ensure fair comparison. This prevents methods from appearing artificially optimal.
2. **Gap Filtering**: Filter invalid VBS (≤ 1e-6) for gap calculations
3. **Pass Rate**: Calculate on ALL problems (including invalid VBS)
4. **Stratification**: Always analyze by difficulty class
5. **Seed Coverage**: Always verify that all 3 seeds (101, 202, 303) are included for each method before aggregation. Use `jobs_per_seed=1` to guarantee this.

### 13.3 Visualization

1. **Error Bars**: Always include (bootstrap std or seed std)
2. **Log Scales**: Use for cost (wide range) and k values (scaling)
3. **Consistent Colors**: Use same palette across all plots
4. **No Titles**: paper standard (captions in the manuscript)

---

## 14. Troubleshooting

### 14.1 Common Issues

**Problem**: No feasible solutions
- **Cause**: Model generates invalid code or constraint violations
- **Fix**: Check error distribution, review code generation

**Problem**: VBS = 0 or negative
- **Cause**: All methods failed or problem has negative optimal score
- **Fix**: Filter invalid VBS (≤ 1e-6) for gap calculations

**Problem**: High variance across seeds
- **Cause**: Non-deterministic generation or small test set
- **Fix**: Use larger test set, check generation settings

**Problem**: Missing plots
- **Cause**: Insufficient data or missing columns
- **Fix**: Check `metrics_final.csv` structure, verify data loading

### 14.2 Debugging

**Check Data**:
```python
# Load and inspect
df = pd.read_csv("metrics_final.csv")
print(df.columns)
print(df.describe())
print(df['feasible'].value_counts())
```

**Verify Metrics**:
```python
# Check VBS calculation
vbs = df[['llm_score', 'score_greedy', 'score_cpsat']].max(axis=1)
print(f"VBS range: {vbs.min()} to {vbs.max()}")

# Check gaps
gaps = (df['vbs_score'] - df['llm_score']) / df['vbs_score']
print(f"Gap range: {gaps.min()} to {gaps.max()}")
```

---

## 15. Conclusion

The SDS evaluation pipeline provides comprehensive assessment of model performance through:

1. **Robust Metrics**: Feasibility, optimality gaps, cost analysis
2. **Statistical Rigor**: Bootstrap resampling, seed aggregation
3. **Rich Visualizations**: Scaling laws, robustness profiles, failure analysis
4. **Comparative Analysis**: Baseline comparisons, stratified breakdowns
5. **Reproducibility**: Deterministic evaluation, consistent baselines

The resulting plots and tables enable thorough understanding of model capabilities, limitations, and areas for improvement.

---

## Appendix A: File Structure

```
evaluation/sds/
├── evaluate.py              # Main evaluation script
├── generate.py               # Generation script (base models)
├── aggregate_plots.py        # Aggregation and cross-seed plots
├── utils.py                 # Utility functions (run_candidate, etc.)
├── results/                 # Individual evaluation results
│   └── qwen2.5-coder-14b/
│       ├── base/
│       ├── grpo-config_hero/
│       └── ...
└── aggregated_report/       # Cross-experiment aggregation
    ├── final_results_table.tex
    └── fig*.png/pdf
```

## Appendix B: Key Functions

**evaluate.py**:
- `evaluate_single_sample()`: Worker function for parallel evaluation
- `PassAtKAnalyzer`: Bootstrap Pass@k analysis
- `plot_scaling_laws()`: Scaling plots
- `main()`: Main evaluation workflow

**aggregate_plots.py**:
- `find_all_metrics_files()`: File discovery
- `load_all_data()`: Data loading and merging
- `plot_efficiency_frontier()`: Efficiency plot
- `plot_robustness_profile()`: Robustness CDF
- `plot_stratified_boxplot()`: Difficulty box plot
- `plot_failure_analysis()`: Failure modes
- `generate_latex_table()`: LaTeX table generation

## Appendix C: References

- Evaluation Script: `evaluation/sds/evaluate.py`
- Aggregation Script: `evaluation/sds/aggregate_plots.py`
- Pass@k Report: `docs/technical-reports/PASS_AT_K_EVALUATION_REPORT.md`
- SLURM Scripts: `scripts/eval_capstor_sds_pipeline.slurm`, `scripts/eval_sds_plot_only.slurm`
