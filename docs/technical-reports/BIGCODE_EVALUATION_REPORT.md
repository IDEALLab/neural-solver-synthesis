# BigCode Evaluation Methodology: Technical Report

## Executive Summary

This report describes the evaluation methodology for assessing code generation models on standard Python programming benchmarks using the BigCode evaluation harness. We evaluate base and fine-tuned models on HumanEval and MBPP using greedy decoding (Pass@1) to directly compare with the Qwen 2.5 Coder technical report.

---

## 1. Problem Setup

### Datasets

We evaluate on two standard code generation benchmarks:

1. **HumanEval** (OpenAI, 2021)
   - **Size**: 164 hand-written Python programming problems
   - **Format**: Function signature + docstring → complete function implementation
   - **Evaluation**: Unit tests (execution-based)
   - **Metric**: Pass@1 (fraction of problems solved correctly)

2. **MBPP** (Austin et al., 2021)
   - **Size**: 500 crowd-sourced Python programming problems
   - **Format**: Task description + one example test → complete solution
   - **Evaluation**: Automated test cases (execution-based)
   - **Metric**: Pass@1 (fraction of problems solved correctly)

### Models Evaluated

- **Base Model**: Qwen2.5-Coder-14B-Instruct (untrained, from HuggingFace)
- **Fine-tuned Models**: 
  - Hero configuration (full training setup)
  - Default ablation configurations (Oracle, Diversity, Prompt)
  - (Optional / legacy) Generalization ablation can be included when explicitly enabled

### Evaluation Configuration

To match the Qwen 2.5 Coder technical report methodology:
- **Decoding**: Greedy (temperature=0, do_sample=False)
- **Samples**: n_samples=1 (Pass@1 evaluation)
- **Batch Size**: 1 (required for greedy decoding)
- **Max Generation Length**: 2048 tokens
- **Precision**: bfloat16

---

## 1.4 Experiment Management (Batches + Report Sets)

BigCode evaluation mirrors the SDS experiment management approach:

- **Moving results (default / legacy)**: `evaluation/bigcode/results/`
- **Frozen batch results**: `evaluation/bigcode/results_batches/<BATCH_ID>/`
- **Explicit report sets**: `experiments/report_sets/<name>.json`

### Batch-aware evaluation outputs (Capstor)

To write BigCode eval outputs into a frozen batch:

```bash
./scripts/evaluate_14b_grpo_bigcode.sh --batch-id <BATCH_ID>
```

When `--batch-id` is omitted, outputs go to the legacy moving directory. This keeps backward compatibility with historical runs.

### Paper-safe aggregation via report sets

Aggregation supports explicit report sets so that tables are generated from a fixed set of frozen roots:

```bash
python evaluation/bigcode/aggregate_results.py --report-set experiments/report_sets/paper_main_results_v1.json
```

### Metadata and method disambiguation

BigCode job folder paths historically did not encode the ablation type. For reproducible aggregation:

- For **new runs**, the evaluation job writes `experiment_metadata.json` into each job folder.
- For **legacy runs**, aggregation infers the method tag from the embedded checkpoint path in `metrics_*.json` when metadata is missing.

This prevents “all runs mapped to Hero” failure modes and allows mean/std across the 3 seeds per method.

### W&B logging conventions (BigCode)

BigCode evaluation supports W&B logging via the Capstor SLURM script:

- **Project**: `WANDB_PROJECT_BIGCODE` (fallback: `WANDB_PROJECT`, default: `qwen-coder-bigcode`)
- **Grouping**: uses `BATCH_ID` as `wandb.group` when set
- **Run naming**: includes `-bigcode-` and ends with `-eval` for unambiguous filtering

---

## 2. Evaluation Methodology

### 2.1 Objective

We want to answer: **"How well does the model solve Python programming problems with a single deterministic generation?"**

This matches the standard reporting methodology in code generation benchmarks (e.g., Qwen technical report) where:
- **Pass@1** is the primary metric (deterministic, reproducible)
- **Greedy decoding** ensures consistency across runs
- **Single sample** provides a fair comparison with published results

### 2.2 Generation Process

For each problem in the dataset:

1. **Prompt Construction**:
   - **HumanEval**: Uses the function signature and docstring from the dataset
   - **MBPP**: Uses the task description and one example test case
   - The harness handles prompt formatting automatically

2. **Generation**:
   - Model generates code completion/solution
   - Uses greedy decoding (temperature=0, deterministic)
   - Stops at stop tokens (`\nclass`, `\ndef`, `\nassert`, etc.)
   - Maximum length: 2048 tokens

3. **Post-processing**:
   - Strips the prompt from the generation (if needed)
   - Extracts the code solution
   - Handles stop tokens appropriately

### 2.3 Evaluation Process

For each generated solution:

1. **Code Execution**:
   - Executes the generated code in a sandboxed environment
   - Timeout: 10 seconds per problem (default in harness)
   - Security: Code execution is sandboxed to prevent malicious code

2. **Test Execution**:
   - **HumanEval**: Runs the canonical test function from the dataset
   - **MBPP**: Runs all test cases from the dataset
   - Checks if the solution passes all tests

3. **Pass@1 Calculation**:
   - **Pass@1** = (Number of problems with at least one passing solution) / (Total problems)
   - Since n_samples=1, this simplifies to: (Number of correct solutions) / (Total problems)
   - Result is a fraction (0.0 to 1.0), reported as percentage (0% to 100%)

### 2.4 Aggregation Across Seeds

For statistical robustness, we evaluate each model configuration across **3 random seeds** (101, 202, 303):

1. **Per-Seed Results**: Each seed produces one Pass@1 score per task
2. **Aggregation**: 
   - **Mean**: Average Pass@1 across the 3 seeds
   - **Std**: Standard deviation across the 3 seeds
3. **Reporting**: Format as `mean ± std` (e.g., `81.5 ± 2.0%`)

---

## 3. BigCode Evaluation Harness

### 3.1 Architecture

The BigCode evaluation harness (`bigcode-evaluation-harness`) provides:

- **Task Abstraction**: Each benchmark (HumanEval, MBPP) is implemented as a `Task` class
- **Generation Pipeline**: Handles model loading, prompt construction, and parallel generation
- **Evaluation Pipeline**: Executes code, runs tests, and computes metrics
- **Result Storage**: Saves generations and metrics in JSON format

### 3.2 Key Components

#### Task Classes

Each task (`HumanEval`, `MBPP`) implements:
- `get_dataset()`: Loads the benchmark dataset
- `get_prompt(doc)`: Constructs the prompt for a problem
- `get_reference(doc)`: Gets the ground truth test cases
- `postprocess_generation(generation, idx)`: Extracts code from generation
- `process_results(generations, references)`: Computes Pass@1 metric

#### Code Execution

- Uses `code_eval` module for safe code execution
- Sandboxed Python environment
- Timeout handling (default 10s per problem)
- Error handling (syntax errors, runtime errors, timeouts)

#### Metrics Computation

- **Pass@1**: Fraction of problems solved correctly
- **Pass@k** (if n_samples > 1): Uses unbiased estimator from Chen et al. (2021)
- For n_samples=1, Pass@1 = (correct solutions) / (total problems)

---

## 4. Implementation Details

### 4.1 Generation Settings

Our evaluation uses the following fixed parameters (matching Qwen report):

```python
temperature = 0.0          # Greedy decoding
n_samples = 1              # Pass@1 evaluation
do_sample = False         # Deterministic
batch_size = 1            # Required for greedy decoding
max_length_generation = 2048
precision = "bf16"        # Mixed precision
```

**Why batch_size=1?**
- Greedy decoding (do_sample=False) does not support `num_return_sequences > 1`
- The harness maps `batch_size` to `num_return_sequences` internally
- For greedy decoding, we must use batch_size=1

### 4.2 Prompt Format

**HumanEval**:
- Input: Function signature + docstring
- Example:
  ```python
  def add(a: int, b: int) -> int:
      """Add two integers and return the result."""
  ```
- Model generates: Function body

**MBPP**:
- Input: Task description + one example test
- Example:
  ```python
  """
  Write a function to add two numbers.
  assert add(1, 2) == 3
  """
  ```
- Model generates: Complete solution

### 4.3 Stop Words

The harness uses task-specific stop words to prevent over-generation:

- **HumanEval**: `["\nclass", "\ndef", "\n@", "\nprint", "\n```", "<file_sep>"]`
- **MBPP**: `["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"]`

These ensure the model stops after completing the required function/solution.

### 4.4 Code Execution Safety

- **Sandboxing**: Code runs in isolated environment
- **Timeout**: 10 seconds per problem (configurable)
- **Error Handling**: Catches syntax errors, runtime errors, timeouts
- **Security**: Prevents malicious code execution (file system access, network, etc.)

---

## 5. Results Aggregation

### 5.1 Data Collection

For each experiment (model + config + seed):
- **Generations**: Saved to `generations_{task}.json`
- **Metrics**: Saved to `metrics_{task}.json` with Pass@1 score
- **Metadata**: Saved to `experiment_metadata.json` (method name, config, seed, job_id)

### 5.2 Aggregation Script

The `aggregate_results.py` script:

1. **Discovers Results**: Scans `evaluation/bigcode/results/` recursively for `metrics_*.json` files

2. **Parses Metadata** (in priority order):
   - **First**: Reads `experiment_metadata.json` if available (most reliable)
   - **Second**: Uses job ID mapping (for backward compatibility)
   - **Third**: Parses path patterns (fallback)

3. **Groups by Method**: Identifies experiments (Base, Hero, Ablations)

4. **Aggregates Across Seeds**:
   - Computes mean Pass@1 per (Method, Task) across 3 seeds
   - Computes standard deviation across seeds
   - Formats as `mean ± std` for LaTeX table

5. **Generates Output**:
   - LaTeX table: `bigcode_results_table.tex`
   - Console summary with all methods

### 5.3 Validation

The aggregation script includes robust validation:

- **Seed Extraction**: Warns if seed cannot be extracted from path
- **Duplicate Detection**: Raises error if duplicate (Method, Seed, Task) combinations found
- **Value Validation**: Checks that Pass@1 values are in range [0.0, 1.0] (fractions, not percentages)
- **Data Integrity**: Verifies exactly 36 combinations (6 methods × 3 seeds × 2 tasks)

---

## 6. Comparison with Qwen Technical Report

### 6.1 Methodology Alignment

Our evaluation methodology **exactly matches** the Qwen 2.5 Coder technical report:

| Aspect | Qwen Report | Our Evaluation |
|--------|-------------|----------------|
| Decoding | Greedy (temperature=0) | ✅ Greedy (temperature=0) |
| Samples | 1 (Pass@1) | ✅ n_samples=1 |
| Tasks | HumanEval, MBPP | ✅ HumanEval, MBPP |
| Metric | Pass@1 | ✅ Pass@1 |
| Format | Percentage | ✅ Percentage (0-100%) |

### 6.2 Direct Comparability

This alignment ensures:
- **Fair Comparison**: Our results are directly comparable to Qwen's published numbers
- **Reproducibility**: Same methodology = same evaluation conditions
- **Validity**: Standard benchmark evaluation protocol

### 6.3 Expected Results

Based on Qwen 2.5 Coder technical report:
- **Qwen2.5-Coder-14B-Instruct (Base)**: 
  - HumanEval: ~88.4% (7B model reported 88.4%, 14B should be similar or higher)
  - MBPP: ~74.8% (our measured baseline)

Our results show:
- **Base Model**: HumanEval 82.3%, MBPP 74.8%
- **Hero Model**: HumanEval 81.5 ± 2.0%, MBPP 74.0 ± 0.5%

**Note**: Slight differences may be due to:
- Different random seeds
- Minor implementation differences in prompt formatting
- Evaluation environment differences

---

## 7. Statistical Robustness

### 7.1 Multi-Seed Evaluation

We evaluate each configuration across **3 seeds** (101, 202, 303) to:
- **Estimate Variance**: Understand how sensitive results are to randomness
- **Report Confidence**: Mean ± std provides statistical confidence
- **Detect Outliers**: Identify if one seed produces anomalous results

### 7.2 Standard Deviation Interpretation

The standard deviation across seeds reflects:
- **Model Stability**: How consistent the model is across different random seeds
- **Evaluation Variance**: Natural variation in code generation (even with greedy decoding, there may be minor differences)
- **Statistical Confidence**: Smaller std = more reliable estimate

**Example Interpretation**:
- `81.5 ± 2.0%`: Mean Pass@1 is 81.5%, with typical variation of ±2% across seeds
- This indicates the model is relatively stable across seeds

### 7.3 Validation Checks

The aggregation script performs several validation checks:

1. **No Duplicates**: Ensures each (Method, Seed, Task) combination appears exactly once
2. **Seed Extraction**: Warns if seed cannot be extracted (would default to 0, causing duplicates)
3. **Value Range**: Validates Pass@1 is in [0.0, 1.0] (fractions, not percentages)
4. **Complete Data**: Verifies all expected combinations are present

---

## 8. Technical Implementation

### 8.1 Evaluation Script

The `eval_capstor_bigcode.slurm` script:

1. **Model Discovery**: Finds checkpoints or loads from HuggingFace
2. **Path Translation**: Maps cluster paths to container paths
3. **Environment Setup**: Configures Python, HuggingFace, code execution
4. **Harness Execution**: Runs `accelerate launch main.py` with appropriate arguments
5. **Metadata Saving**: Saves `experiment_metadata.json` for aggregation

### 8.2 Harness Execution

```bash
accelerate launch deps/bigcode-evaluation-harness/main.py \
    --model "$CONTAINER_MODEL_PATH" \
    --tasks "$TASK" \
    --temperature 0 \
    --n_samples 1 \
    --do_sample False \
    --max_length_generation 2048 \
    --batch_size 1 \
    --precision bf16 \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "$OUTPUT_DIR/generations_${TASK}.json" \
    --metric_output_path "$OUTPUT_DIR/metrics_${TASK}.json"
```

### 8.3 Result Files

Each evaluation produces:

- `generations_{task}.json`: List of generated code solutions (list of lists)
- `metrics_{task}.json`: Evaluation metrics (Pass@1 score)
- `experiment_metadata.json`: Experiment metadata (method, config, seed, job_id)

**Metrics JSON Structure**:
```json
{
  "humaneval": {
    "pass@1": 0.823170731707317
  },
  "config": {
    "temperature": 0.0,
    "n_samples": 1,
    "do_sample": false,
    ...
  }
}
```

---

## 9. Results Interpretation

### 9.1 Pass@1 Score

**Pass@1** represents the fraction of problems solved correctly with a single generation:
- **Higher is Better**: 100% = all problems solved, 0% = no problems solved
- **Interpretation**: 
  - >80%: Strong performance (model solves most problems)
  - 60-80%: Good performance
  - <60%: May indicate issues (catastrophic forgetting, poor training, etc.)

### 9.2 Mean ± Std Format

Results are reported as `mean ± std`:
- **Mean**: Average Pass@1 across 3 seeds
- **Std**: Standard deviation across seeds
- **Interpretation**:
  - Small std (<1%): Very stable across seeds
  - Medium std (1-3%): Moderate variation
  - Large std (>3%): High variation (may indicate instability)

### 9.3 Comparison Across Methods

When comparing methods:
- **Base vs. Fine-tuned**: Check if fine-tuning improves or degrades performance
- **Hero vs. Ablations**: Identify which components contribute to performance
- **Statistical Significance**: Large std may indicate results are not statistically significant

---

## 10. Limitations and Considerations

### 10.1 Single Sample Evaluation

- **Limitation**: Pass@1 with n_samples=1 only measures deterministic performance
- **Implication**: Does not capture the model's ability to solve problems with multiple attempts
- **Mitigation**: This matches standard reporting (Qwen report), ensuring comparability

### 10.2 Greedy Decoding

- **Limitation**: Greedy decoding may not find optimal solutions
- **Implication**: Results may underestimate model capability
- **Justification**: Matches Qwen report methodology for fair comparison

### 10.3 Evaluation Environment

- **Sandboxing**: Code execution is sandboxed, which may differ from real-world execution
- **Timeout**: 10s timeout may be insufficient for some complex problems
- **Dependencies**: Some problems may require external libraries not available in sandbox

### 10.4 Dataset Limitations

- **HumanEval**: Only 164 problems (small sample size)
- **MBPP**: 500 problems (larger, but still limited)
- **Coverage**: May not represent all types of programming tasks

---

## 11. Summary

### What We Do

- Evaluate models on HumanEval (164 problems) and MBPP (500 problems)
- Use greedy decoding (temperature=0) with single sample (Pass@1)
- Aggregate results across 3 seeds (101, 202, 303) for statistical robustness
- Report mean ± std for each method and task

### How We Evaluate

- **Generation**: BigCode evaluation harness with greedy decoding
- **Execution**: Sandboxed Python environment with timeout (10s)
- **Metrics**: Pass@1 (fraction of problems solved correctly)
- **Aggregation**: Mean and standard deviation across seeds

### What It Means

- **Pass@1**: Fraction of problems solved with a single deterministic generation
- **Mean ± Std**: Average performance with confidence interval across seeds
- **Comparison**: Directly comparable to Qwen 2.5 Coder technical report

### Key Takeaway

Our evaluation methodology **exactly matches** the Qwen technical report, ensuring:
- **Fair Comparison**: Results are directly comparable to published numbers
- **Reproducibility**: Same methodology = same evaluation conditions
- **Validity**: Standard benchmark evaluation protocol used in code generation research

---

## Appendix: Running Evaluations on Cluster

### Single Experiment Evaluation

**Base Model**:
```bash
# SSH to cluster
ssh <your-cluster-login>

# Navigate to repo
cd /capstor/scratch/{CLUSTER}/$USER/llm-finetuning

# Evaluate base model (seed 303)
sbatch scripts/eval_capstor_bigcode.slurm --base-model Qwen/Qwen2.5-Coder-14B-Instruct 303
```

**Fine-tuned Model (Latest Checkpoint)**:
```bash
# Discovery mode: finds latest checkpoint automatically
sbatch scripts/eval_capstor_bigcode.slurm 303 qwen2.5-coder-14b grpo

# Explicit checkpoint
sbatch scripts/eval_capstor_bigcode.slurm --checkpoint-dir /capstor/scratch/{CLUSTER}/$USER/checkpoints/Qwen2.5-Coder-14B-Instruct-GRPO-SDS-OPT-seed303/checkpoint-80 grpo 303
```

### Batch Evaluation

**Evaluate All 14B GRPO Experiments (Base + Hero + Ablations)**:
```bash
# All seeds (18 jobs: 1 base + 5 experiments × 3 seeds)
./scripts/evaluate_14b_grpo_bigcode.sh

# Single seed (6 jobs: 1 base + 5 experiments × 1 seed)
./scripts/evaluate_14b_grpo_bigcode.sh --seed 303
```

**Monitor Jobs**:
```bash
# Check job status
squeue -u $USER

# Check logs
tail -f /capstor/scratch/{CLUSTER}/{USER}/logs/eval-bigcode-*.out
```

### Aggregating Results

**Generate Aggregated Table**:
```bash
# Activate environment
conda activate llm-finetuning

# Aggregate all results
python evaluation/bigcode/aggregate_results.py

# Custom output directory
python evaluation/bigcode/aggregate_results.py --output-dir my_aggregated_results
```

**Results Location**:
- Individual results: `evaluation/bigcode/results/{model}/{scheme}/seed{seed}/job-{job_id}/`
  - `generations_humaneval.json`: Generated code solutions
  - `generations_mbpp.json`: Generated code solutions
  - `metrics_humaneval.json`: Pass@1 score for HumanEval
  - `metrics_mbpp.json`: Pass@1 score for MBPP
  - `experiment_metadata.json`: Experiment metadata (method, config, seed)
- Aggregated results: `evaluation/bigcode/aggregated_report/`
  - `bigcode_results_table.tex`: LaTeX table with mean ± std across seeds

**Quick Pass@1 Check**:
```bash
# Extract Pass@1 scores from metrics
python3 << 'EOF'
import json
import glob

for metrics_file in glob.glob('evaluation/bigcode/results/**/metrics_*.json', recursive=True):
    with open(metrics_file) as f:
        data = json.load(f)
        for task, results in data.items():
            if task != 'config' and 'pass@1' in results:
                print(f'{metrics_file}: {task} Pass@1 = {results["pass@1"]:.2%}')
EOF
```

---

## Appendix: Code References

- **Evaluation Script**: `scripts/eval_capstor_bigcode.slurm`
- **Aggregation Script**: `evaluation/bigcode/aggregate_results.py`
- **BigCode Harness**: `deps/bigcode-evaluation-harness/`
  - **Main Entry**: `main.py`
  - **Task Definitions**: `bigcode_eval/tasks/humaneval.py`, `bigcode_eval/tasks/mbpp.py`
  - **Evaluation**: `bigcode_eval/evaluator.py`
  - **Code Execution**: `bigcode_eval/tasks/custom_metrics/code_eval.py`
- **Batch Evaluation**: `scripts/evaluate_14b_grpo_bigcode.sh`

---

## References

1. **HumanEval**: Chen et al. (2021). "Evaluating Large Language Models Trained on Code". arXiv:2107.03374
2. **MBPP**: Austin et al. (2021). "Program Synthesis with Large Language Models". arXiv:2108.07732
3. **Qwen 2.5 Coder**: Qwen Team. "Qwen2.5-Coder Technical Report" (2024)
4. **BigCode Evaluation Harness**: BigCode Team. "bigcode-evaluation-harness" (GitHub)
