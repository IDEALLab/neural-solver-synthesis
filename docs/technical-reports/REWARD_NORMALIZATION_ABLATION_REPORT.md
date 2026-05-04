# Reward Normalization Ablation Report

Status as of 2026-03-29.

## Executive Summary

This ablation was designed as a narrow test of whether Hero remains strong when only the SDS nominal reward normalization heuristic is changed.

What was held fixed:

- reward stack structure
- reward weights: `0.10 / 0.20 / 0.70`
- system prompt
- dataset family and seed matching
- training budget
- SDS evaluation pipeline

What was changed:

- only the SDS nominal reward normalization heuristic inside `open-r1`

Final 3-seed treatment results:

| Seed | Train Job | Eval Job | Control Pass | Treatment Pass | Delta Pass | Control Gap | Treatment Gap | Delta Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 1736090 | 1738057 | 97.80% | 57.50% | -40.30 pts | 3.50% | 42.95% | +39.45 pts |
| 202 | 1742835 | 1742836 | 98.00% | 55.20% | -42.80 pts | 5.58% | 45.51% | +39.94 pts |
| 303 | 1738649 | 1738650 | 97.70% | 98.50% | +0.80 pts | 3.15% | 4.75% | +1.60 pts |

Treatment aggregate across 3 seeds:

- pass rate: `70.40% +/- 24.36%`
- mean gap: `31.07% +/- 22.83%`
- aggregate error counts: `none=2112`, `constraint=829`, `timeout=59`

Control aggregate across 3 seeds:

- pass rate: `97.83% +/- 0.15%`
- mean gap: `4.08% +/- 1.31%`
- dominant failure mode: timeout only

Bottom line:

- this ablation does not support a robustness claim
- the normalization change induces large cross-seed variability
- the main degradation mode is feasibility and constraint satisfaction, not always optimization quality once a solution is feasible

## Goal

The goal was to determine whether current Hero performance is robust to a change in the nominal reward normalization heuristic, without changing reward weights or any other major part of the recipe.

The intended scientific comparison was:

- control: current minimalist Hero
- treatment: same recipe, but with an alternate normalization rule for the SDS nominal reward

This experiment was kept standalone on purpose. It was not wired into `aggregate_plots.py`, report sets, or the main paper result-generation pipeline.

## Exact Treatment Definition

Control:

- config: `deps/open-r1/recipes/Qwen2.5-Coder-14B-Instruct/grpo/config_hero.yaml`
- nominal reward path: default SDS normalization heuristic

Treatment:

- config: `deps/open-r1/recipes/Qwen2.5-Coder-14B-Instruct/grpo/config_ablation_reward_normalization.yaml`
- nominal reward function: `unified_nominal_reward_topk_interaction_bound`
- normalization rule:
  - keep `max_weight_contribution = sum(top_U_positive_weights)`
  - replace the interaction estimate with `sum(top_max_pairs_positive_interactions)`
  - define `max_pairs = min(num_positive_interactions, U * (U - 1) // 2)`

No changes were made to `syndeopt`.

## Implementation Details

### `open-r1` Changes

The experiment was implemented as a branch-specific treatment path rather than by mutating Hero in place.

Files changed:

- `deps/open-r1/src/open_r1/simulators/sds_simulator.py`
- `deps/open-r1/src/open_r1/rewards_unified_v2.py`
- `deps/open-r1/src/open_r1/rewards.py`
- `deps/open-r1/recipes/Qwen2.5-Coder-14B-Instruct/grpo/config_ablation_reward_normalization.yaml`
- `deps/open-r1/tests/test_sds_reward_normalization.py`

What was added:

- in `sds_simulator.py`
  - `_get_normalization_variant()`
  - `_get_max_interaction_contribution(...)`
  - support for two variants:
    - `avg_positive_times_max_pairs` for the control path
    - `topk_positive_interactions` for the treatment path
  - `_calculate_reward(...)` now delegates interaction normalization estimation through that variant-aware helper
- in `rewards_unified_v2.py`
  - `_TOPK_POSITIVE_INTERACTION_CONFIG`
  - `_unified_nominal_reward_with_simulator_config(...)`
  - `unified_nominal_reward_topk_interaction_bound(...)`
- in `rewards.py`
  - reward registry entry for `unified_nominal_reward_topk_interaction_bound`
- in the ablation config
  - the reward stack remains format + execution + nominal
  - the nominal function is swapped to `unified_nominal_reward_topk_interaction_bound`
  - reward weights remain unchanged
- in tests
  - focused coverage was added for the normalization-path behavior

Implementation references:

- `deps/open-r1/src/open_r1/simulators/sds_simulator.py`
  - `_get_normalization_variant`: line `296`
  - `_get_max_interaction_contribution`: line `303`
  - `_calculate_reward`: line `325`
- `deps/open-r1/src/open_r1/rewards_unified_v2.py`
  - `_TOPK_POSITIVE_INTERACTION_CONFIG`: line `109`
  - `unified_nominal_reward_topk_interaction_bound`: line `1112`
  - `_unified_nominal_reward_with_simulator_config`: line `1128`
- `deps/open-r1/src/open_r1/rewards.py`
  - import and registry wiring for `unified_nominal_reward_topk_interaction_bound`: lines `38`, `644`
- `deps/open-r1/recipes/Qwen2.5-Coder-14B-Instruct/grpo/config_ablation_reward_normalization.yaml`
  - reward functions: lines `109-112`
  - reward weights: line `114`

### Cluster and Launcher Changes

This experiment also required branch-specific Clariden setup and a few infrastructure fixes that were discovered while bringing up the first runs.

Files changed:

- `edf_files/gh200-llm-sds-training-reward-normalization-daints.toml`
- `scripts/train_capstor_unified_sds_qwen_coder.slurm`
- `scripts/eval_capstor_sds_pipeline.slurm`

What changed operationally:

- created a branch-specific EDF file for Clariden
- created and synced the branch-specific scratch checkout:
  - `CLUSTER_ROOT/llm-finetuning-reward-normalization`
- added `EDF_ENVIRONMENT` override support to the train and eval launchers
- fixed the multi-node master worker IP probe to use explicit `/bin/bash`
- exported `WANDB_API_KEY` and `HF_TOKEN` into the container-side shell
- fixed explicit-checkpoint evaluation by exporting `MODEL_PATH` into the container-side shell before path translation
- kept training and evaluation under the branch-specific EDF and scratch checkout

Important script references:

- `scripts/train_capstor_unified_sds_qwen_coder.slurm`
  - `EDF_ENVIRONMENT` support: line `365`
  - `WANDB_API_KEY` export: line `369`
  - fixed master worker IP probe: line `428`
  - explicit `/bin/bash` container launch: line `463`
- `scripts/eval_capstor_sds_pipeline.slurm`
  - checkpoint / model path handling: lines `201`, `234`, `343`, `357`, `364`, `421`
  - `WANDB_API_KEY` export: line `437`
  - `EDF_ENVIRONMENT` support: line `438`
  - explicit `/bin/bash` container launch: line `441`
  - container-side `MODEL_PATH` export: line `450`

## Validation Before Launch

Code-level validation completed locally:

- `PYTHONPATH=deps/open-r1/src:deps/syndeopt/src pytest deps/open-r1/tests/test_sds_reward_normalization.py -q`
  - result: passed
- shell syntax checks on the modified Slurm scripts
  - result: passed

Operational validation completed on Clariden:

- branch-specific scratch checkout existed and pointed to `codex/reward-normalization-ablation`
- `deps/open-r1` inside that checkout pointed to `codex/reward-normalization-ablation`
- branch-specific EDF file existed in `~/.edf`

## Metric Computation and Verification Procedure

This section documents exactly how the reported numbers were computed and re-verified.

### Metric Semantics

The reported pass rate and mean gap follow the repository's own evaluation and aggregation logic, not an ad hoc post-processing rule.

Source-of-truth code:

- `evaluation/sds/evaluate.py`
  - LLM pass computation: lines `2443-2444`
  - LLM gap computation: lines `2454-2462`
- `evaluation/sds/aggregate_plots.py`
  - pass computed from `feasible` on all rows: lines `462-470`
  - gap computed with infeasible rows treated as zero score: lines `474-479`
  - invalid gaps filtered after construction: lines `733-741`
  - per-seed then across-seed aggregation: lines `764-781`

The exact semantics are:

- `pass_rate`
  - computed on all rows as `mean(feasible)`
- `gap`
  - for each row with `vbs_score > 1e-6`
  - define `effective_llm_score = llm_score` if `feasible == True`, else `0.0`
  - compute `gap = (vbs_score - max(effective_llm_score, 0.0)) / vbs_score`
  - keep the row only if the resulting gap is finite and in `[0, 1]`
  - average those row-level gaps
- `valid_gap_count`
  - number of rows surviving that valid-gap filter
- `error_counts`
  - raw counts from the `error_type` column in `metrics_final.csv`

Important consequence:

- infeasible rows are not excluded from mean gap
- they contribute as `100%` gap when `vbs_score > 0`

This matters materially for seed `202`. An earlier draft of this report mistakenly quoted a feasible-only mean gap for treatment seed `202`. That was incorrect under the repo's actual metric semantics and has been corrected here.

### Control Selection Procedure

The control comparison uses the frozen paper-safe Hero runs from:

- `experiments/report_sets/paper_main_results_v1.json`

That manifest explicitly records the canonical Hero SDS jobs:

- seed `101`: `1315163`
- seed `202`: `1315168`
- seed `303`: `1315173`

Those are also reflected in:

- `docs/technical-reports/EXPERIMENT_STRUCTURE_SUMMARY.md`

So the control numbers are not based on a subjective latest-run pick; they come from the frozen main-results manifest.

### Raw Artifact Locations Used

Control metrics CSVs:

- `evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed101/job-1315163/metrics_final.csv`
- `evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed202/job-1315168/metrics_final.csv`
- `evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed303/job-1315173/metrics_final.csv`

Treatment metrics CSVs on Clariden:

- `CLUSTER_ROOT/llm-finetuning-reward-normalization/evaluation/sds/results_batches/20260327_reward-normalization-v1/qwen2.5-coder-14b/grpo/seed101/job-1736090/metrics_final.csv`
- `CLUSTER_ROOT/llm-finetuning-reward-normalization/evaluation/sds/results_batches/20260327_reward-normalization-v1/qwen2.5-coder-14b/grpo/seed202/job-1742835/metrics_final.csv`
- `CLUSTER_ROOT/llm-finetuning-reward-normalization/evaluation/sds/results_batches/20260327_reward-normalization-v1/qwen2.5-coder-14b/grpo/seed303/job-1738649/metrics_final.csv`

### Verification Commands Used

Control recomputation from local frozen artifacts:

```bash
python3 - <<'PY'
import csv, math
from pathlib import Path
jobs = {
  101: Path("evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed101/job-1315163/metrics_final.csv"),
  202: Path("evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed202/job-1315168/metrics_final.csv"),
  303: Path("evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed303/job-1315173/metrics_final.csv"),
}
for seed, p in jobs.items():
    rows = list(csv.DictReader(p.open()))
    pass_rate = sum(str(r["feasible"]).lower() == "true" for r in rows) / len(rows)
    gaps = []
    for r in rows:
        v = float(r["vbs_score"])
        if v > 1e-6:
            llm = float(r["llm_score"]) if str(r["feasible"]).lower() == "true" else 0.0
            gap = (v - max(0.0, llm)) / v
            if 0 <= gap <= 1 and not math.isnan(gap):
                gaps.append(gap)
    print(seed, pass_rate, sum(gaps) / len(gaps), len(gaps))
PY
```

Treatment recomputation from Clariden artifacts:

```bash
ssh clariden 'python3 - <<'"'"'PY'"'"'
import csv, math
jobs = {
  101: "CLUSTER_ROOT/llm-finetuning-reward-normalization/evaluation/sds/results_batches/20260327_reward-normalization-v1/qwen2.5-coder-14b/grpo/seed101/job-1736090/metrics_final.csv",
  202: "CLUSTER_ROOT/llm-finetuning-reward-normalization/evaluation/sds/results_batches/20260327_reward-normalization-v1/qwen2.5-coder-14b/grpo/seed202/job-1742835/metrics_final.csv",
  303: "CLUSTER_ROOT/llm-finetuning-reward-normalization/evaluation/sds/results_batches/20260327_reward-normalization-v1/qwen2.5-coder-14b/grpo/seed303/job-1738649/metrics_final.csv",
}
for seed, p in jobs.items():
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    pass_rate = sum(str(r["feasible"]).lower() == "true" for r in rows) / len(rows)
    gaps = []
    for r in rows:
        v = float(r["vbs_score"])
        if v > 1e-6:
            llm = float(r["llm_score"]) if str(r["feasible"]).lower() == "true" else 0.0
            gap = (v - max(0.0, llm)) / v
            if 0 <= gap <= 1 and not math.isnan(gap):
                gaps.append(gap)
    print(seed, pass_rate, sum(gaps) / len(gaps), len(gaps))
PY'
```

These recomputations matched the control numbers already in the frozen report set and corrected the treatment seed `202` gap to the proper repository-consistent value.

## Control Reference

Hero control runs used for comparison:

| Seed | Batch | Job | Metrics File | Pass Rate | Mean Gap |
| --- | --- | --- | --- | ---: | ---: |
| 101 | `20251230_struct-feas-v1` | `1315163` | `evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed101/job-1315163/metrics_final.csv` | 97.80% | 3.50% |
| 202 | `20251230_struct-feas-v1` | `1315168` | `evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed202/job-1315168/metrics_final.csv` | 98.00% | 5.58% |
| 303 | `20251230_struct-feas-v1` | `1315173` | `evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed303/job-1315173/metrics_final.csv` | 97.70% | 3.15% |

Control aggregate across 3 seeds:

- pass rate: `97.83% +/- 0.15%`
- mean gap: `4.08% +/- 1.31%`

Note:

- the mean gap above follows the evaluation pipeline convention and is averaged only over valid finite feasible cases

## Treatment Run Ledger

Evaluation batch ID for all treatment runs:

- `20260327_reward-normalization-v1`

### Seed 101

Final successful path:

- train job: `1736090`
- eval job: `1738057`
- training outcome: reached `checkpoint-90`, then hit the 4-hour Slurm limit
- evaluated checkpoint:
  - `CLUSTER_ROOT/checkpoints/Qwen2.5-Coder-14B-Instruct-GRPO-SDS-seed101-config_ablation_reward_normalization/job-1736090/checkpoint-90`
- result directory:
  - `CLUSTER_ROOT/llm-finetuning-reward-normalization/evaluation/sds/results_batches/20260327_reward-normalization-v1/qwen2.5-coder-14b/grpo/seed101/job-1736090`
- W&B run:
  - `uzrvnaqx`

Earlier failed attempts and fixes:

- `1735949`
  - failure: launcher could not resolve worker IP because the probe used plain `bash` in `srun`
  - fix: explicit `/bin/bash` and improved master worker IP resolution
- `1736007`
  - failure: `WANDB_API_KEY` was not reaching the container-side shell
  - fix: export credentials into the container shell explicitly
- `1736091`
  - failure mode: dependent eval never started because training timed out instead of exiting `afterok`
  - mitigation: manual evaluation from `checkpoint-90`
- `1737865`
  - failure: explicit-checkpoint evaluation failed because `MODEL_PATH` was not exported into the container-side shell
  - fix: export `MODEL_PATH` before container-side path translation in the eval wrapper

### Seed 202

First attempt:

- train job: `1738647`
- eval job: `1738648`
- outcome:
  - training failed after about `2m40s`
  - no `checkpoint-90` was produced
  - dependent eval failed because there was no checkpoint to evaluate
- classification:
  - operational failure, not a scientific datapoint

Successful rerun:

- train job: `1742835`
- eval job: `1742836`
- training outcome: reached `checkpoint-90`, then hit the 4-hour Slurm limit
- evaluated checkpoint:
  - `CLUSTER_ROOT/checkpoints/Qwen2.5-Coder-14B-Instruct-GRPO-SDS-seed202-config_ablation_reward_normalization/job-1742835/checkpoint-90`
- result directory:
  - `CLUSTER_ROOT/llm-finetuning-reward-normalization/evaluation/sds/results_batches/20260327_reward-normalization-v1/qwen2.5-coder-14b/grpo/seed202/job-1742835`

### Seed 303

Successful path:

- train job: `1738649`
- eval job: `1738650`
- training outcome: reached `checkpoint-90`, then hit the 4-hour Slurm limit
- evaluated checkpoint:
  - `CLUSTER_ROOT/checkpoints/Qwen2.5-Coder-14B-Instruct-GRPO-SDS-seed303-config_ablation_reward_normalization/job-1738649/checkpoint-90`
- result directory:
  - `CLUSTER_ROOT/llm-finetuning-reward-normalization/evaluation/sds/results_batches/20260327_reward-normalization-v1/qwen2.5-coder-14b/grpo/seed303/job-1738649`

## Final Results

### Per-Seed Treatment Results

| Seed | Pass Rate | Mean Gap | Valid Gap Count | Error Counts |
| --- | ---: | ---: | ---: | --- |
| 101 | 57.50% | 42.95% | 984 / 1000 | `none=575`, `constraint=409`, `timeout=16` |
| 202 | 55.20% | 45.51% | 980 / 1000 | `none=552`, `constraint=420`, `timeout=28` |
| 303 | 98.50% | 4.75% | 985 / 1000 | `none=985`, `timeout=15` |

### Control vs Treatment Comparison

| Seed | Control Pass | Treatment Pass | Delta Pass | Control Gap | Treatment Gap | Delta Gap | Control Errors | Treatment Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 101 | 97.80% | 57.50% | -40.30 pts | 3.50% | 42.95% | +39.45 pts | `none=978`, `timeout=22` | `none=575`, `constraint=409`, `timeout=16` |
| 202 | 98.00% | 55.20% | -42.80 pts | 5.58% | 45.51% | +39.94 pts | `none=980`, `timeout=20` | `none=552`, `constraint=420`, `timeout=28` |
| 303 | 97.70% | 98.50% | +0.80 pts | 3.15% | 4.75% | +1.60 pts | `none=977`, `timeout=23` | `none=985`, `timeout=15` |

### Aggregate Comparison

Control aggregate:

- pass rate mean: `97.83%`
- pass rate std: `0.15%`
- mean gap mean: `4.08%`
- mean gap std: `1.31%`

Treatment aggregate:

- pass rate mean: `70.40%`
- pass rate std: `24.36%`
- mean gap mean: `31.07%`
- mean gap std: `22.83%`

Average treatment minus control:

- pass rate delta mean: `-27.43` percentage points
- mean gap delta mean: `+27.00` percentage points

Aggregate treatment failure profile across all 3000 evaluated instances:

- `none=2112`
- `constraint=829`
- `timeout=59`

## Analysis

### What the Results Clearly Show

This experiment does not support the claim that Hero is robust to this normalization change.

The strongest evidence is:

- seed `101` degrades sharply in both pass rate and mean gap
- seed `202` also degrades sharply in both pass rate and mean gap under the repository's actual metric semantics
- seed `303` remains close to Hero, which means the treatment is not uniformly catastrophic

The overall pattern is therefore:

- high seed sensitivity
- substantial instability in feasibility
- non-uniform effect on optimization quality conditional on feasibility

### Feasibility vs Optimization Quality

The most informative pattern is now:

- seed `101` degrades sharply in pass rate and mean gap
- seed `202` also degrades sharply in pass rate and mean gap under the repo's true metric semantics
- seed `303` remains near Hero

This indicates that the ablation is not just hurting feasibility while leaving aggregate solution quality untouched. Under the repository's actual gap metric, seeds `101` and `202` both show large quality degradation once infeasible outputs are correctly counted as `100%` gap.

That interpretation is consistent with:

- large `constraint` error counts on seeds `101` and `202`
- large positive gap deltas on those same seeds

So the most likely behavioral effect is:

- the alternate normalization weakens or distorts the training signal that helps the model consistently produce feasible, high-quality solutions

### What This Means for the Reviewer Question

The result is unfavorable for a robustness story and favorable to the reviewer concern.

The fairest scientific framing is:

- the nominal reward normalization heuristic is a consequential part of the method
- current Hero performance is sensitive to that design choice
- the ablation reveals substantial cross-seed instability, especially in constraint satisfaction

What should not be claimed:

- not that the entire approach is invalid
- not that the ablation uniformly destroys optimization quality
- not that one seed alone explains the whole behavior

What is justified by the full 3-seed result:

- the method is not robust to this normalization change
- the original normalization heuristic appears to be doing meaningful work
- the primary degradation mode is reduced feasibility reliability

## Operational Assessment

This is a valid scientific comparison rather than an infrastructure artifact.

Reasons:

- the treatment code path was implemented explicitly and separately
- the focused normalization tests passed locally
- all three scientific seeds produced evaluated treatment results
- all successful evaluations used complete `checkpoint-90` directories
- the seed `202` first failure was isolated as an operational issue and replaced by a successful rerun before inclusion in the final comparison

## Artifacts

Primary branch and setup:

- top-level branch: `codex/reward-normalization-ablation`
- `open-r1` branch: `codex/reward-normalization-ablation`
- Clariden EDF:
  - `~/.edf/gh200-llm-sds-training-reward-normalization-daints.toml`

Primary output batch:

- `20260327_reward-normalization-v1`

Treatment result directories:

- seed `101`
  - `CLUSTER_ROOT/llm-finetuning-reward-normalization/evaluation/sds/results_batches/20260327_reward-normalization-v1/qwen2.5-coder-14b/grpo/seed101/job-1736090`
- seed `202`
  - `CLUSTER_ROOT/llm-finetuning-reward-normalization/evaluation/sds/results_batches/20260327_reward-normalization-v1/qwen2.5-coder-14b/grpo/seed202/job-1742835`
- seed `303`
  - `CLUSTER_ROOT/llm-finetuning-reward-normalization/evaluation/sds/results_batches/20260327_reward-normalization-v1/qwen2.5-coder-14b/grpo/seed303/job-1738649`

Each result directory contains:

- `metrics_final.csv`
- `experiment_metadata.json`
- `generations.jsonl`
- `results_table.tex`
- `results_stratified.tex`
- robustness and error-distribution plots

## Final Interpretation

Final bottom line:

- changing only the nominal reward normalization heuristic materially changes behavior
- the original Hero result is sensitive to this choice
- the dominant instability is feasibility and constraint satisfaction
- the ablation therefore weakens any claim that normalization is merely a harmless implementation detail
