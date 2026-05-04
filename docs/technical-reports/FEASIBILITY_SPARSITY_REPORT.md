# Feasibility Sparsity Analysis Report

## Executive Summary

This report measures feasibility density during Hero-style SDS GRPO training at the true 64-sample group level. The underlying question is whether the SDS nominal reward is effectively too sparse to drive learning because, within a GRPO group of 64 sampled completions, feasible samples might be rare or absent.

We first attempted to recover this statistic from existing logs and artifacts. That recovery path failed because historic artifacts did not preserve a stable, unique SDS mission identity that would let us reconstruct the exact 64-sample groups with confidence. We therefore instrumented `deps/open-r1`, validated the logging on Clariden, and then launched fresh 4-hour runs for seeds `101`, `202`, and `303`.

Across the three production runs, feasibility was not sparse at the group level. Over `1,284` reconstructed GRPO groups of `64` completions each:

- `85.83%` of groups contained at least one feasible completion
- the average group contained `37.94` feasible completions
- the pooled feasible completion rate was `59.27%`

Even in the early training tercile, `88.76%` of groups already contained at least one feasible sample, with `23.32` feasible samples on average. The main learning-dynamics change over training is therefore not the first appearance of feasibility, but the increasing density of feasible samples within each group.

## 1. Reviewer Question This Report Answers

Reviewer `ABks` W6 asks, in effect:

- what fraction of GRPO groups of 64 samples contain at least one feasible SDS solution?
- how does that fraction evolve during training?
- if the fraction is low, does that imply the effective nominal learning signal is sparse?

The exact target statistic is group-level, not just average reward. It is not enough to report mean feasibility reward or mean nominal reward. We need to know whether the 64 samples generated for a prompt contain feasible candidates, because GRPO compares samples within the group.

## 2. Success Criteria

The branch dossier defined the following success criteria:

1. fraction of 64-sample GRPO groups with at least one feasible sample
2. average number of feasible samples per group
3. early / middle / late training view, or equivalent progression view
4. short interpretation
5. explicit statement of whether the answer came from historic recovery or fresh instrumentation

This report satisfies all five.

## 3. Outcome Summary

### 3.1 Main Result

Across all three fresh instrumented production runs:

- groups reconstructed: `1,284`
- group size: `64`
- fraction of groups with any feasible sample: `0.8582554517133957`
- mean feasible samples per group: `37.93535825545171`
- feasible completion rate: `0.592739972741433`

Rounded for prose:

- `85.83%` of groups had at least one feasible sample
- average feasible count was `37.94 / 64`
- feasible completion rate was `59.27%`

### 3.2 Early / Middle / Late View

Using per-seed terciles of `reward_call_index`:

| Stage | Groups | Frac. groups with any feasible | Mean feasible count / 64 | Feasible completion rate |
| --- | ---: | ---: | ---: | ---: |
| Early | 427 | 0.8876 | 23.32 | 0.3644 |
| Middle | 428 | 0.8248 | 39.48 | 0.6168 |
| Late | 429 | 0.8625 | 50.94 | 0.7960 |

### 3.3 Bottom-Line Interpretation

The nominal reward is not sparse at the GRPO-group level in these runs. A typical 64-sample group usually already contains feasible samples, even early in training. What becomes denser over training is not merely the chance of seeing one feasible sample, but the number of feasible samples inside the group.

## 4. Chronology of How We Got Here

This section is intentionally detailed, because the purpose of this document is not just to present the final numbers, but to explain exactly how those numbers were produced and why we trust them.

### 4.1 Initial Plan

The original plan followed the branch dossier:

1. try to recover the exact statistic from existing artifacts
2. if exact recovery fails, instrument training
3. validate the instrumentation on short Clariden jobs
4. run fresh production jobs
5. aggregate offline and write a directly reusable report

### 4.2 Recovery Audit Failed

We first investigated whether existing artifacts already contained enough information.

The recovery audit looked at:

- W&B histories
- W&B completion tables
- existing Clariden-side outputs
- logged SDS prompts and mission-dependent reward behavior

The failure mode was subtle but important:

- SDS prompts in the training dataset were not unique identifiers for SDS missions
- a concrete probe found one logged prompt that matched `68` cached training examples
- none of those candidate missions reproduced the logged nominal reward exactly
- historic metadata did not preserve a code snapshot or stable mission identifier sufficient to reconstruct the exact training groups

Conclusion:

- recovery from old logs was not exact enough for appendix use
- fresh instrumentation was required

### 4.3 First Instrumentation Attempt Was Insufficient

The first logging implementation wrote feasibility summaries directly during reward computation. That revealed a distributed-systems issue:

- the logs showed `group_size = 8`
- the intended GRPO group size is `64`
- `step` was `null`

This happened because each training rank only sees its local shard of generations, not the full global 64-way group.

This first attempt was useful because it surfaced the real constraint:

- the correct solution was not to summarize globally inside one rank
- the correct solution was to log rank-local shards losslessly and reconstruct the true group offline

### 4.4 Logging Design Was Refined in Three Steps

The logging path then evolved through three concrete fixes:

1. rank-sharded feasibility logging
   - each rank writes its own feasibility shard
   - each record includes distributed metadata and a monotonic `reward_call_index`
2. dataset-identity-preserving raw generation logging
   - each raw completion is logged with stable SDS instance identity
   - prompt text is no longer the sole key
3. logging reliability fixes
   - instrumentation failures are explicitly written to a dedicated error directory
   - raw traces and feasibility summaries share one logging context per reward invocation

These fixes were implemented in `deps/open-r1` and then validated in live Clariden jobs before launching the final production runs.

### 4.5 Validation Runs Before Production

We ran multiple short jobs before the full 4-hour runs.

Important validation milestones:

- early short jobs surfaced the local-8-versus-global-64 bug
- later short jobs validated rank-sharded logging
- validation job `1742998` confirmed:
  - raw generation traces were being written
  - feasibility summary shards were being written
  - `problem_uuid` was present
  - `reward_call_index` aligned between raw traces and feasibility summaries
  - 64-sample groups could be reconstructed cleanly across all 8 ranks

That validation result is what made the final three-seed production run scientifically defensible.

## 5. Exact Files Involved

This section lists the exact files that were part of the final measurement path.

### 5.1 Branch Planning / Scope

- `docs/FEASIBILITY_SPARSITY_BRANCH_DOSSIER.md`

This file defined the question, stop conditions, and expected deliverables.

### 5.2 Training Launcher

- `scripts/train_capstor_unified_sds_qwen_coder.slurm`

This is the existing cluster training script used for multi-node SDS GRPO training. We intentionally reused the existing script rather than inventing a new launcher, because it already handled:

- multi-node layout
- vLLM server startup
- Accelerate launch
- Hugging Face authentication
- W&B authentication
- cache directories
- checkpoint paths
- output directory naming

### 5.3 Clariden Container Configuration

- `edf_files/gh200-llm-sds-training-feasibility-sparsity-daints.toml`

This EDF file mounts the dedicated Clariden checkout and routes `/workspace/logs` to capstor scratch so the JSONL instrumentation outputs persist outside the container.

### 5.4 GRPO Training Configuration

- `deps/open-r1/recipes/Qwen2.5-Coder-14B-Instruct/grpo/config_hero.yaml`

This config defines the Hero-style GRPO setup used in the production runs, including:

- `num_generations: 64`
- `per_device_train_batch_size: 8`
- `gradient_accumulation_steps: 4`
- `use_vllm: true`
- `dataset_name: SoheylM/OpenR1-SDS-10k-seed101` as the default config value
- reward stack:
  - `unified_format_reward`
  - `unified_code_execution_reward_no_oracle`
  - `unified_nominal_reward`

In the actual production runs, the per-seed dataset name was overridden from the launcher via `--dataset_name SoheylM/OpenR1-SDS-10k-seed{SEED}`.

### 5.5 Dataset Identity Propagation

- `deps/open-r1/src/open_r1/grpo.py`

This file was modified so each dataset example carries stable identity fields into training:

- `problem_uuid`
- `problem_prompt_hash`
- `problem_mission_hash`

The purpose of this change was to avoid relying on prompt text as the only join key.

### 5.6 Reward-Side Instrumentation

- `deps/open-r1/src/open_r1/rewards_unified_v2.py`

This file contains the instrumentation that actually wrote the logs used in the final analysis.

Key functions:

- `_build_logging_context`
  - creates shared per-invocation logging metadata
- `_log_group_feasibility_stats`
  - writes rank-local feasibility summaries
- `_log_generation_traces`
  - writes raw per-completion traces
- `_log_instrumentation_error`
  - writes exception information if instrumentation fails

The reward function `unified_code_execution_reward_no_oracle(...)` calls both logging paths after reward computation, using a shared logging context.

### 5.7 Regression / Unit Tests

- `deps/open-r1/tests/test_feasibility_logging.py`

This file verifies:

- rank-sharded feasibility logs include `reward_call_index` and `local_group_size`
- `reward_call_index` increments between invocations
- raw generation traces include `problem_uuid` and completion text
- shared logging context keeps feasibility summaries and raw traces aligned on `reward_call_index`

### 5.8 Offline Aggregation

- `scripts/analyze_feasibility_sparsity.py`

This script reconstructs 64-sample global groups from rank-sharded logs and writes the checked-in small artifact bundle:

- `analysis/feasibility_sparsity/summary.json`
- `analysis/feasibility_sparsity/per_seed_summary.csv`
- `analysis/feasibility_sparsity/stage_pooled_summary.csv`
- `analysis/feasibility_sparsity/progress_bins.csv`

## 6. Clariden Environment and Execution Setup

### 6.1 Dedicated Clariden Checkout

The branch used a dedicated Clariden checkout:

- `/capstor/scratch/cscs/$USER/llm-finetuning-feasibility-sparsity`

This was intentional. We did not reuse an unrelated scratch checkout because:

- the branch modifies a submodule (`deps/open-r1`)
- we needed a branch-specific EDF config
- we wanted a clean mapping from cluster outputs to this branch’s purpose

### 6.2 Container Setup

The EDF file used:

- image:
  - `CLUSTER_ROOT/containers/gh200-llm-sds-training.sqsh`
- code mounts:
  - `CLUSTER_ROOT/llm-finetuning-feasibility-sparsity:/workspace/llm-finetuning`
  - `CLUSTER_ROOT/llm-finetuning-feasibility-sparsity/deps/open-r1:/workspace/open-r1`
  - `CLUSTER_ROOT/llm-finetuning-feasibility-sparsity/deps/syndeopt:/workspace/syndeopt`
- logs mount:
  - `CLUSTER_ROOT/logs:/workspace/logs`
- models/checkpoints/cache mounts:
  - `CLUSTER_ROOT/models:/workspace/models`
  - `CLUSTER_ROOT/checkpoints:/workspace/checkpoints`
  - `CLUSTER_ROOT/hf_datasets_cache:CLUSTER_ROOT/capstor`

This mount design matters because the instrumentation writes to `/workspace/logs/...`, which resolves to a persistent capstor scratch directory.

### 6.3 Multi-Node Layout

For the 14B GRPO runs, the script uses a 3-node layout:

- node 0: vLLM server
- nodes 1 and 2: training workers

Training is distributed across 8 GPUs total on the worker nodes, while vLLM serves generations separately.

### 6.4 Existing Script Logic We Reused

The launcher already handled:

- `HF_TOKEN` and `HF_HUB_TOKEN` loading from `$HOME/llm/hf_token.txt`
- `WANDB_API_KEY` loading from `$HOME/llm/wandb_token.txt`
- W&B naming and output directories
- dataset cache clearing
- `accelerate launch` setup
- `trl vllm-serve` startup
- host/container path translation

This is important because it means the final result was obtained through the project’s existing execution path, not through a one-off experimental script.

## 7. Exact Training Method Used for the Final Measurement

### 7.1 Training Objective

We measured feasibility during the actual Hero-style SDS GRPO training path, not in a post hoc evaluation-only script.

The training configuration was:

- model: `Qwen/Qwen2.5-Coder-14B-Instruct`
- mode: `grpo_cold`
- seeds: `101`, `202`, `303`
- dataset family: `SoheylM/OpenR1-SDS-10k-seed{SEED}`
- reward stack:
  - `unified_format_reward`
  - `unified_code_execution_reward_no_oracle`
  - `unified_nominal_reward`

### 7.2 Why the Group Size Is 64

The GRPO config sets:

- `num_generations: 64`
- `per_device_train_batch_size: 8`

With 8 training GPUs total, each reward invocation corresponds to one prompt whose 64 generations are distributed as 8 local generations per rank across 8 ranks.

That is why:

- the true analysis unit is a 64-sample global group
- each rank-local shard has size 8
- offline reconstruction across 8 rank files is required

### 7.3 Production Jobs

The final reported numbers come from these fresh Clariden jobs:

- seed `101`: job `1743423`
- seed `202`: job `1743424`
- seed `303`: job `1743425`

All three were submitted as 4-hour GRPO jobs using the existing SDS launcher.

Operationally, the production submission pattern was:

```bash
sbatch --time=04:00:00 --nodes=3 --ntasks=3 \
  scripts/train_capstor_unified_sds_qwen_coder.slurm \
  --mode grpo_cold \
  --model 14B \
  --seed 101 \
  --dataset-name SoheylM/OpenR1-SDS-10k-seed101 \
  --config deps/open-r1/recipes/Qwen2.5-Coder-14B-Instruct/grpo/config_hero.yaml
```

and analogously for seeds `202` and `303`.

### 7.4 Validation Jobs

Short validation jobs were run first to make sure the instrumentation produced the expected files and schemas. The final important validation run was:

- validation job `1742998`

This run confirmed:

- rank-sharded feasibility summaries
- rank-sharded raw generation traces
- aligned `reward_call_index`
- valid `problem_uuid`, `mission_hash`, `prompt_hash` grouping

## 8. Logging Method in Detail

This section explains exactly what was logged and how the logged records were used.

### 8.1 Feasibility Summary Logs

Written by:

- `_log_group_feasibility_stats(...)`

Directory:

- `/workspace/logs/feasibility_sparsity/<run_id>/rankXXXXX.jsonl`

Each record includes:

- `trainer_global_step`
- `reward_call_index`
- `rank`
- `local_rank`
- `world_size`
- `local_group_ordinal`
- `mission_hash`
- `prompt_hash`
- `local_group_size`
- `feasible_count_in_group`
- `has_any_feasible_in_group`
- `feasible_fraction_in_group`

Important nuance:

- these are rank-local shard summaries, not yet the global 64-sample group

### 8.2 Raw Generation Trace Logs

Written by:

- `_log_generation_traces(...)`

Directory:

- `/workspace/logs/feasibility_generation_traces/<run_id>/rankXXXXX.jsonl`

Each record includes:

- `trainer_global_step`
- `reward_call_index`
- `rank`
- `local_rank`
- `world_size`
- `local_group_ordinal`
- `sample_ordinal_in_group`
- `problem_uuid`
- `mission_hash`
- `prompt_hash`
- `completion_sha256`
- `completion_text`
- `reward`
- `exact_feasible`

These raw traces are the lossless record of what was generated and whether it was feasible.

### 8.3 Why We Logged Both

We logged both summary shards and raw traces for complementary reasons:

- summary shards make aggregation cheap and explicit
- raw traces make the aggregation auditable
- raw traces let us validate that the summary counts are correct

This dual-logging design is part of why the final result is trustworthy.

Although rank-0 also logs compact shard-level metrics to W&B for observability, the final reported statistics in this report do not come from W&B. The source of truth is the rank-sharded JSONL output on Clariden plus the offline reconstruction script.

### 8.4 Instrumentation Error Logging

Any instrumentation exceptions are written to:

- `/workspace/logs/feasibility_instrumentation_errors`

For the final three production jobs, no instrumentation error files were emitted.

## 9. Offline Aggregation Method

### 9.1 Why Offline Aggregation Was Necessary

Because each rank sees only its local shard of the group, no single rank directly observes the entire 64-sample set. The final statistic therefore had to be reconstructed offline.

### 9.2 Reconstruction Logic

The aggregation script:

- reads all `rank*.jsonl` files for one job
- groups records by `reward_call_index`
- sums local `feasible_count_in_group` values across ranks
- sums local `local_group_size` values across ranks
- checks identity consistency through `mission_hash` and `prompt_hash`
- uses raw generation traces to verify:
  - total sample count is exactly 64
  - all 8 ranks are represented
  - there is exactly one `problem_uuid`, one `mission_hash`, and one `prompt_hash`
  - feasible counts derived from raw traces match the summary counts

### 9.3 Integrity Checks

The final aggregated artifact reports:

- `integrity_issue_count = 0`

This means that across all three production jobs:

- every reconstructed group had size 64
- every group had all 8 shards
- group identity was internally consistent
- trace-derived feasibility counts matched summary-derived feasibility counts

### 9.4 Stage and Progress Definitions

Two progression views are reported.

Per-seed early/middle/late stages:

- each seed’s groups are sorted by `reward_call_index`
- they are split into terciles

Normalized progress bins:

- pooled groups are assigned a normalized position using:
  - `progress_fraction = reward_call_index / max_reward_call_index`
- 10 pooled bins are then formed from `[0.0, 0.1)` through `[0.9, 1.0]`

### 9.5 Important Caveat About Step Number

The final analysis uses `reward_call_index` as the progression coordinate, not a guaranteed optimizer-step field.

Why:

- earlier instrumentation attempts showed that `trainer_global_step` was not reliably populated in this reward path
- `reward_call_index` is monotonic and aligned between the two log streams
- it is therefore sufficient for within-run trajectory analysis

This matters because the report’s progression view should be read as:

- early / middle / late reward-hook progression

not as:

- exact optimizer-step indices

That caveat does not affect the main reviewer question about whether groups are usually empty or not.

## 10. Exact Production Artifact Counts

For the final production runs:

- seed `101` / job `1743423`
  - trace lines: `26,880`
  - summary lines: `3,360`
  - reconstructed groups: `420`
- seed `202` / job `1743424`
  - trace lines: `30,656`
  - summary lines: `3,832`
  - reconstructed groups: `479`
- seed `303` / job `1743425`
  - trace lines: `24,640`
  - summary lines: `3,080`
  - reconstructed groups: `385`

These counts are internally consistent:

- `trace lines = groups * 64`
- `summary lines = groups * 8`

## 11. Final Results in Detail

### 11.1 Pooled Results Across All Seeds

Over `1,284` reconstructed 64-sample groups:

- fraction of groups with at least one feasible sample:
  - `0.8582554517133957`
- mean feasible samples per group:
  - `37.93535825545171`
- feasible completion rate:
  - `0.592739972741433`

Rounded:

- `85.83%` of groups contain at least one feasible sample
- average feasible count is `37.94 / 64`
- feasible completion rate is `59.27%`

### 11.2 Pooled Early / Middle / Late Results

| Stage | Groups | Frac. groups with any feasible | Mean feasible count / 64 | Feasible completion rate |
| --- | ---: | ---: | ---: | ---: |
| Early | 427 | 0.8875878220140515 | 23.320843091334893 | 0.3643881733021077 |
| Middle | 428 | 0.8247663551401869 | 39.47663551401869 | 0.616822429906542 |
| Late | 429 | 0.8624708624708625 | 50.94405594405595 | 0.7960008741258742 |

Key takeaways:

- already in early training, groups are usually not empty of feasible samples
- from early to late, the mean feasible count rises from `23.32` to `50.94`
- the pooled feasible completion rate rises from `36.44%` to `79.60%`

### 11.3 Normalized Progress Bins

| Bin | Groups | Frac. groups with any feasible | Mean feasible count / 64 | Feasible completion rate |
| --- | ---: | ---: | ---: | ---: |
| 0.0-0.1 | 129 | 0.8449612403100775 | 10.0 | 0.15625 |
| 0.1-0.2 | 128 | 0.9296875 | 24.1953125 | 0.3780517578125 |
| 0.2-0.3 | 129 | 0.875968992248062 | 31.24031007751938 | 0.4881298449612403 |
| 0.3-0.4 | 128 | 0.9296875 | 37.4140625 | 0.5845947265625 |
| 0.4-0.5 | 127 | 0.7952755905511811 | 39.31496062992126 | 0.6142962598425197 |
| 0.5-0.6 | 129 | 0.7364341085271318 | 36.95348837209303 | 0.5773982558139535 |
| 0.6-0.7 | 128 | 0.8828125 | 46.4453125 | 0.7257080078125 |
| 0.7-0.8 | 129 | 0.8682170542635659 | 50.27131782945737 | 0.7854893410852714 |
| 0.8-0.9 | 128 | 0.8984375 | 53.1796875 | 0.8309326171875 |
| 0.9-1.0 | 129 | 0.8217054263565892 | 50.434108527131784 | 0.7880329457364341 |

The exact fraction of groups with any feasible sample is not perfectly monotonic bin-to-bin, but it remains high throughout. The clearer monotonic trend is in the density of feasibility inside each group.

### 11.4 Per-Seed Results

| Seed | Job | Groups | Frac. groups with any feasible | Mean feasible count / 64 | Feasible completion rate |
| --- | --- | ---: | ---: | ---: | ---: |
| 101 | 1743423 | 420 | 0.8904761904761904 | 40.75 | 0.63671875 |
| 202 | 1743424 | 479 | 0.7661795407098121 | 31.878914405010438 | 0.4981080375782881 |
| 303 | 1743425 | 385 | 0.9376623376623376 | 42.4 | 0.6625 |

Per-seed early / middle / late values are stored in:

- `analysis/feasibility_sparsity/per_seed_summary.csv`

### 11.5 Seed-Specific Interpretation

Seeds `101` and `303` are especially strong. Seed `202` is clearly weaker, but even that seed does not support the idea that GRPO groups are usually empty of feasible samples. The weakest seed still has:

- `76.62%` of groups with at least one feasible sample overall
- `87.42%` of groups with at least one feasible sample in the early tercile

That is not a sparse-signal regime in the reviewer’s feared sense.

## 12. How To Interpret These Results With Respect To the Reviewer

This section is written for human understanding rather than just record-keeping.

### 12.1 What the Reviewer Was Worried About

The reviewer’s concern can be paraphrased like this:

- GRPO only gets useful nominal learning signal from feasible samples
- if most 64-sample groups have zero feasible samples, then the nominal signal is effectively sparse
- if the nominal signal is sparse, then the training success might depend mainly on auxiliary shaping signals rather than the nominal SDS objective

That is a reasonable concern in principle.

### 12.2 What Our Measurement Actually Shows

Our measurement shows that, in these Hero-style runs:

- most groups are not empty of feasible samples
- this is already true early in training
- over time, groups become more densely populated with feasible samples

The key implication is:

- the nominal SDS objective is available to GRPO much more often than the reviewer suspected

### 12.3 Why “At Least One Feasible Sample” Matters

If a 64-sample group contains at least one feasible sample, then the nominal objective is not absent from that comparison set. That does not mean every sample is useful, but it does mean the group is not completely blind to feasibility-conditioned reward.

Since `85.83%` of groups contain at least one feasible sample overall, and `88.76%` do so already in the early tercile, the nominal path is usually present.

### 12.4 Why “Mean Feasible Count” Matters Even More

The stronger signal is actually the average number of feasible samples per group:

- early: `23.32 / 64`
- middle: `39.48 / 64`
- late: `50.94 / 64`

This means the relevant progression is not:

- from zero feasible samples to one feasible sample

but rather:

- from a moderately dense feasible subset to a very dense feasible subset

That is much less compatible with a “vanishingly sparse nominal signal” story.

### 12.5 What We Should and Should Not Claim

What we can claim confidently:

- the nominal SDS signal is not sparse at the 64-sample group level in these runs
- groups with zero feasible samples are the minority, not the norm
- the density of feasible samples rises substantially during training

What we should not overclaim:

- this does not prove auxiliary rewards are unimportant
- this does not prove the same density profile would hold for every model, dataset, or training recipe
- this does not prove monotonicity at every small timescale

The correct reviewer-facing claim is narrower and stronger:

- in the exact Hero-style SDS training path studied here, the reviewer’s feared “mostly empty GRPO groups” regime does not occur

## 13. Limitations and Scope

This report should be read with the following scope limitations in mind:

- the production runs were 4-hour runs, not necessarily full-epoch completions
- progression is indexed by `reward_call_index`, not guaranteed optimizer-step metadata
- the result is about the measured Hero-style SDS configuration, not every possible ablation
- the report answers group-level feasibility sparsity, not all possible reward-quality questions

These limitations do not undermine the core conclusion about whether groups are usually empty of feasible samples.

## 14. Appendix-Ready Interpretation

The nominal reward is not sparse at the GRPO-group level in these Hero-style SDS runs. Across three fresh instrumented seeds, `85.83%` of 64-sample groups contained at least one feasible sample, and the average group contained `37.94` feasible samples. Even in the early training tercile, `88.76%` of groups had at least one feasible sample, with `23.32` feasible samples on average. The main learning-dynamics change is therefore not the appearance of the first feasible sample, but the densification of feasible samples within each group as training progresses.

## 15. Appendix-Ready Paragraph

We directly measured feasibility density during GRPO by instrumenting fresh Hero-style SDS training runs on Clariden and reconstructing the true 64-sample groups across distributed ranks. Across 3 seeds (`1,284` total groups), `85.83%` of groups contained at least one feasible sample, with an average of `37.94` feasible samples per group. Even in the early training tercile, `88.76%` of groups already contained at least one feasible sample, with `23.32` feasible samples on average. This indicates that the nominal SDS signal is not sparse at the group level in the way the reviewer feared; what changes over training is primarily the density of feasible samples within a group, which rises substantially over time.

## 16. Reproducibility Pointers

Primary checked-in artifacts:

- `analysis/feasibility_sparsity/summary.json`
- `analysis/feasibility_sparsity/per_seed_summary.csv`
- `analysis/feasibility_sparsity/stage_pooled_summary.csv`
- `analysis/feasibility_sparsity/progress_bins.csv`
- `scripts/analyze_feasibility_sparsity.py`

Primary raw-output locations on Clariden:

- `CLUSTER_ROOT/logs/feasibility_sparsity/1743423/`
- `CLUSTER_ROOT/logs/feasibility_sparsity/1743424/`
- `CLUSTER_ROOT/logs/feasibility_sparsity/1743425/`
- `CLUSTER_ROOT/logs/feasibility_generation_traces/1743423/`
- `CLUSTER_ROOT/logs/feasibility_generation_traces/1743424/`
- `CLUSTER_ROOT/logs/feasibility_generation_traces/1743425/`

The checked-in artifact bundle is the small, reproducible summary. The Clariden directories are the full source-of-truth raw logs from which the summary was computed.
