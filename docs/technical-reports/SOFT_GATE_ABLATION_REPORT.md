# Soft-Gate SDS Ablation Report

## Scope

This report tracks the SDS ablation that replaces Hero's hard nominal feasibility gate with a soft violation penalty while keeping the rest of the training recipe fixed.

## Exact Reward Rule

- Reward function: `unified_soft_nominal_reward`
- Config: `config_ablation_soft_gate.yaml`
- Domain: SDS only
- Formula:
  - `normalized_score = normalize_sds_score(raw_score, requirements)`
  - `soft_reward = clamp(normalized_score - 0.15 * constraint_violations, 0.0, 1.0)`
- Fixed design choice:
  - only the nominal reward changes
  - format reward, execution reward, prompt, dataset, and evaluation protocol stay Hero-matched

## Smoke Result

- Batch ID: `20260326_soft-gate-smoke-v1`
- Status: `COMPLETED`
- Cluster workspace: `CLUSTER_ROOT/llm-finetuning-soft-reward-ablation`
- EDF environment: `gh200-llm-sds-training-soft-reward-ablation-daints`
- Training job:
  - SLURM job `1727303`
  - command: `bash scripts/launch_14b_grpo_experiments.sh --seed 101 --config config_ablation_soft_gate.yaml --time-limit 02:00:00 --edf-env gh200-llm-sds-training-soft-reward-ablation-daints`
  - terminal state: `TIMEOUT` at the intended 2h budget
  - checkpoint produced: `CLUSTER_ROOT/checkpoints/Qwen2.5-Coder-14B-Instruct-GRPO-SDS-seed101-config_ablation_soft_gate/job-1727303/checkpoint-30`
- SDS evaluation:
  - SLURM job `1728673`
  - command: `bash scripts/evaluate_14b_grpo_experiments.sh --seed 101 --config config_ablation_soft_gate --batch-id 20260326_soft-gate-smoke-v1 --edf-env gh200-llm-sds-training-soft-reward-ablation-daints`
  - terminal state: `COMPLETED`
  - result root: `CLUSTER_ROOT/llm-finetuning-soft-reward-ablation/evaluation/sds/results_batches/20260326_soft-gate-smoke-v1/qwen2.5-coder-14b/grpo/seed101/job-1727303`
- Smoke headline metrics:
  - Feasibility / pass rate: `55.50%` (`555/1000`)
  - Error distribution: `none=555`, `constraint=427`, `timeout=17`, `runtime=1`
  - Constraint violations: `precedence=427`, `mutex=1`, `cardinality=0`, `groups=0`
- W&B:
  - training: [qwen2.5-coder-14b-grpo-soft-gate-sds-seed101-job1727303](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/4v6tpyca)
  - evaluation: [qwen2.5-coder-14b-grpo-soft-gate-sds-seed101-job1727303-eval](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/k42mm0vq)
- Notes:
  - The smoke run was stable enough to justify the full seed101 rollout.
  - The smoke checkpoint already showed that the new reward path trains and evaluates end to end without infrastructure issues.

## Seed101 Result

- Batch ID: `20260326_soft-gate-v1`
- Status: `COMPLETED`
- Training job:
  - SLURM job `1729098`
  - command: `bash scripts/launch_14b_grpo_experiments.sh --seed 101 --config config_ablation_soft_gate.yaml --edf-env gh200-llm-sds-training-soft-reward-ablation-daints`
  - terminal state: `TIMEOUT` at the intended 4h budget
  - checkpoints produced: `checkpoint-30`, `checkpoint-60`, `checkpoint-90`
- Evaluation:
  - original dependent eval submitter `1729099` was canceled because `afterok` did not fire after a walltime timeout
  - manual eval job `1730496`
  - command: `bash scripts/evaluate_14b_grpo_experiments.sh --seed 101 --config config_ablation_soft_gate --batch-id 20260326_soft-gate-v1 --edf-env gh200-llm-sds-training-soft-reward-ablation-daints`
  - terminal state: `COMPLETED`
  - evaluated checkpoint: `CLUSTER_ROOT/checkpoints/Qwen2.5-Coder-14B-Instruct-GRPO-SDS-seed101-config_ablation_soft_gate/job-1729098/checkpoint-90`
  - result root: `CLUSTER_ROOT/llm-finetuning-soft-reward-ablation/evaluation/sds/results_batches/20260326_soft-gate-v1/qwen2.5-coder-14b/grpo/seed101/job-1729098`
- SDS headline metrics vs Hero:
  - Soft gate feasibility / pass rate: `56.80%` (`568/1000`)
  - Hero seed101 comparator feasibility / pass rate: `97.80%` (`978/1000`)
  - Delta vs Hero: `-41.00` percentage points
  - Shared-VBS mean optimality gap:
    - Hero: `3.52%`
    - Soft gate: `44.94%`
    - delta: `+41.42` percentage points
  - Main soft-gate failure mode: infeasible outputs, dominated by precedence violations
  - Soft-gate error distribution: `none=568`, `constraint=414`, `timeout=18`
  - Soft-gate violation totals: `precedence=583`, `groups=9`, `cardinality=0`, `mutex=0`
- Early-learning artifact:
  - training W&B run: [qwen2.5-coder-14b-grpo-soft-gate-sds-seed101-job1729098](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/2ep79w1x)
  - eval W&B run: [qwen2.5-coder-14b-grpo-soft-gate-sds-seed101-job1729098-eval](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/j1yl1lnr)
  - Hero eval comparator W&B run: [qwen2.5-coder-14b-grpo-hero-sds-seed101-job1315163-eval](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/6r83o2ge)
  - qualitative training note:
    - the soft-gate run did not collapse; later training logs showed much higher nominal reward than at startup, but this did not translate into Hero-level final feasibility

## Multi-Seed Result Summary

- Included seeds: `101`, `202`, `303`
- Frozen result root: `evaluation/sds/results_batches/20260326_soft-gate-v1`
- Aggregated report set: `experiments/report_sets/paper_soft_gate_v1.json`
- Summary:
  - Soft gate seed results:
    - seed101: `56.80%`
    - seed202: `55.20%`
    - seed303: `60.20%`
  - Hero seed results:
    - seed101: `97.80%`
    - seed202: `98.00%`
    - seed303: `97.70%`
  - 3-seed feasibility average:
    - soft gate: `57.40%`
    - Hero: `97.83%`
    - delta: `-40.43` percentage points
  - Shared-VBS mean optimality gap by seed:
    - seed101: Hero `3.52%`, Soft gate `44.94%`, delta `+41.42` points
    - seed202: Hero `5.59%`, Soft gate `45.31%`, delta `+39.72` points
    - seed303: Hero `3.17%`, Soft gate `40.24%`, delta `+37.07` points
  - 3-seed mean optimality gap:
    - Hero: `4.09%`
    - Soft gate: `43.50%`
    - delta: `+39.40` percentage points
  - Interpretation:
    - The negative result is not limited to seed101. Across all three seeds, the soft-gate variant underperforms Hero on both SDS feasibility and mean optimality gap, with the dominant failure mode remaining constraint violations, especially precedence violations.

## Gap Computation Note

- Gap definition:
  - `gap = max(0, (VBS - score) / VBS)`
  - infeasible solutions are treated as `score = 0`, giving `100%` gap when `VBS > 0`
- For the appendix comparison above, we used a shared per-problem VBS for each seed:
  - `VBS = max(greedy, local_search, cpsat, bnb, Hero, Soft Gate)`
  - this avoids comparing Hero and Soft Gate against different denominators when one of them beats the baselines on a subset of instances
- The resulting shared-VBS gap numbers are therefore directly comparable between Hero and Soft Gate.
- Practical implication:
  - the much worse mean gap for Soft Gate is driven in large part by its much higher infeasibility rate, because each infeasible output receives a worst-case `100%` gap whenever a feasible VBS exists for that problem.

## Comparator Reference

- Canonical Hero seed101 evaluation root:
  - `REPO_ROOT/evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed101/job-1315163`
- Canonical Hero seed202 evaluation root:
  - `REPO_ROOT/evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed202/job-1315168`
- Canonical Hero seed303 evaluation root:
  - `REPO_ROOT/evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed303/job-1315173`
- Canonical Hero metrics:
  - [seed101 metrics_final.csv](evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed101/job-1315163/metrics_final.csv)
  - [seed202 metrics_final.csv](evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed202/job-1315168/metrics_final.csv)
  - [seed303 metrics_final.csv](evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed303/job-1315173/metrics_final.csv)

## Run Ledger

- W&B project:
  - [qwen-coder-sds-rl](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl)
- Smoke training:
  - job `1727303`
  - checkpoint `checkpoint-30`
  - W&B run: [qwen2.5-coder-14b-grpo-soft-gate-sds-seed101-job1727303](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/4v6tpyca)
- Smoke evaluation:
  - job `1728673`
  - batch `20260326_soft-gate-smoke-v1`
  - W&B run: [qwen2.5-coder-14b-grpo-soft-gate-sds-seed101-job1727303-eval](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/k42mm0vq)
- Full training:
  - job `1729098`
  - checkpoints `checkpoint-30`, `checkpoint-60`, `checkpoint-90`
  - W&B run: [qwen2.5-coder-14b-grpo-soft-gate-sds-seed101-job1729098](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/2ep79w1x)
- Stale dependent eval submitter:
  - job `1729099`
  - canceled because the `afterok` dependency was never satisfied after training timed out at walltime
- Full evaluation:
  - job `1730496`
  - batch `20260326_soft-gate-v1`
  - W&B run: [qwen2.5-coder-14b-grpo-soft-gate-sds-seed101-job1729098-eval](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/j1yl1lnr)
- Seed202 training:
  - job `1730693`
  - terminal state: `TIMEOUT` at the intended 4h budget
  - checkpoint used for eval: `checkpoint-90`
  - result root: `CLUSTER_ROOT/llm-finetuning-soft-reward-ablation/evaluation/sds/results_batches/20260326_soft-gate-v1/qwen2.5-coder-14b/grpo/seed202/job-1730693`
  - W&B run: [qwen2.5-coder-14b-grpo-soft-gate-sds-seed202-job1730693](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/4n83o6bc)
- Seed202 eval submitter:
  - job `1730696`
  - dependency mode: `afterany:1730693`
  - terminal state: `COMPLETED`
- Seed202 evaluation:
  - job `1732288`
  - batch `20260326_soft-gate-v1`
  - pass rate: `55.20%` (`552/1000`)
  - error distribution: `none=552`, `constraint=414`, `timeout=34`
  - precedence violations: `582`
  - W&B run: [qwen2.5-coder-14b-grpo-soft-gate-sds-seed202-job1730693-eval](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/p27j3n9a)
- Seed303 training:
  - job `1730694`
  - terminal state: `TIMEOUT` at the intended 4h budget
  - checkpoint used for eval: `checkpoint-90`
  - result root: `CLUSTER_ROOT/llm-finetuning-soft-reward-ablation/evaluation/sds/results_batches/20260326_soft-gate-v1/qwen2.5-coder-14b/grpo/seed303/job-1730694`
  - W&B run: [qwen2.5-coder-14b-grpo-soft-gate-sds-seed303-job1730694](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/583bsdn4)
- Seed303 eval submitter:
  - job `1730695`
  - dependency mode: `afterany:1730694`
  - terminal state: `COMPLETED`
- Seed303 evaluation:
  - job `1732287`
  - batch `20260326_soft-gate-v1`
  - pass rate: `60.20%` (`602/1000`)
  - error distribution: `none=602`, `constraint=372`, `timeout=26`
  - precedence violations: `521`
  - W&B run: [qwen2.5-coder-14b-grpo-soft-gate-sds-seed303-job1730694-eval](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/p37jqrqw)
- Hero seed101 comparator evaluation:
  - W&B run: [qwen2.5-coder-14b-grpo-hero-sds-seed101-job1315163-eval](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/6r83o2ge)
- Hero seed202 comparator training run:
  - W&B run: [qwen2.5-coder-14b-grpo-hero-sds-seed202-job1315168](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/0c9zvpvd)
- Hero seed303 comparator training run:
  - W&B run: [qwen2.5-coder-14b-grpo-hero-sds-seed303-job1315173](https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl/runs/ssc0gjsy)

## Interpretation

Replacing the hard nominal feasibility gate with a soft penalty on simulator-reported violations degraded SDS performance relative to Hero across all completed SDS seeds. Over seeds `101/202/303`, the soft-gate variant reached `56.80% / 55.20% / 60.20%` feasibility versus canonical Hero's `97.80% / 98.00% / 97.70%`, for a 3-seed average of `57.40%` versus `97.83%` (delta `-40.43` points). Using a shared per-problem VBS across baselines plus both compared methods, the mean optimality gap also worsened from `4.09%` for Hero to `43.50%` for Soft Gate (delta `+39.40` points). The dominant failure mode remained constraint violations, especially precedence violations, which suggests that the hard gate is not merely making the reward sparse: it is providing an important feasibility-first learning signal that the soft-gate variant weakens.

## Standalone Summary

We implemented the requested SDS ablation in which Hero's hard nominal feasibility gate was replaced by a soft violation-penalized nominal reward, while keeping the prompt, model, data, execution reward, format reward, and evaluation protocol fixed. Concretely, we used `R_soft = clamp(normalize(score) - 0.15 * constraint_violations, 0, 1)` and evaluated the resulting `config_ablation_soft_gate` model on the same SDS benchmark.

This soft-gate variant did not improve performance. Across seeds `101/202/303`, its feasibility rates were `56.8%`, `55.2%`, and `60.2%`, compared with canonical Hero's `97.8%`, `98.0%`, and `97.7%`, respectively. The 3-seed average therefore drops from `97.83%` for Hero to `57.40%` for the soft-gate ablation. Using a shared per-problem VBS across baselines plus both compared methods, the mean optimality gap also worsens from `4.09%` for Hero to `43.50%` for Soft Gate. This gap degradation is driven in large part by the much higher infeasibility rate under Soft Gate, since infeasible outputs are assigned worst-case `100%` gap whenever a feasible VBS exists. The dominant failure mode remains infeasibility, especially precedence violations. Our interpretation is that the hard feasibility gate is not simply an overly harsh source of reward sparsity; rather, it appears to provide an important feasibility-first learning signal for SDS. In other words, softening the gate gives partial credit to infeasible but superficially high-scoring candidates, and this weakens the model's incentive to internalize constraint satisfaction.
