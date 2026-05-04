# Paper Release Manifest

This document defines the canonical code-release state for the paper "Beyond Inference-Time Search: Reinforcement Learning Synthesizes Reusable Solvers".

## 1. Scope

This release promotes the paper-backed revision work into one coherent reproducibility surface:

- refreshed SDS main comparison bundle
- frozen-solver / fixed-code validation
- manually specified constraint-aware SA baseline
- refreshed neutral-prompt ShinkaEvolve comparison
- soft-gate ablation
- reward-normalization sensitivity
- feasibility-sparsity analysis
- timeout analysis

CVRP is intentionally excluded from this release.

## 2. Canonical submodule pins

- `deps/open-r1`
  - commit: `0fd3d1b0be4cb009fb8a1279ca6f2e52c4019d2f`
  - role: unified SDS reward/training stack with soft-gate, normalization, and feasibility logging support
- `deps/ShinkaEvolve`
  - commit: `202269eb9adcb788e047470721c2cf91216fec89`
  - role: neutral-prompt SDS fairness rerun used by the final paper

## 3. Main paper bundles

### 3.1 Main manifest

- manifest:
  - `experiments/report_sets/paper_public_main_v1.json`

### 3.2 Checked-in outputs

- SDS:
  - `evaluation/sds/aggregated_report_batches/paper_public_main_v1/`
- BigCode:
  - `evaluation/bigcode/aggregated_report_batches/paper_public_main_v1/`

### 3.3 Paper-facing mapping

- Figure 1-5 SDS bundle
  - source manifest: `experiments/report_sets/paper_public_main_v1.json`
  - checked-in outputs: `evaluation/sds/aggregated_report_batches/paper_public_main_v1/`
  - release snapshot note: `docs/NEURIPS_2026_CODE_RELEASE_SNAPSHOT.md`
- BigCode table
  - source manifest: `experiments/report_sets/paper_public_main_v1.json`
  - checked-in output: `evaluation/bigcode/aggregated_report_batches/paper_public_main_v1/bigcode_results_table.tex`

## 4. Appendix and supporting evidence

### 4.1 Fixed-code / runtime bundle

- path:
  - `evaluation/sds/aggregated_report_batches/20260326_baseline-eval-v1/`
- used for:
  - frozen compile-once validation
  - manual SA baseline comparison
  - representative runtime accounting

### 4.2 Soft-gate ablation

- manifest:
  - `experiments/report_sets/paper_soft_gate_v1.json`
- report:
  - `docs/technical-reports/SOFT_GATE_ABLATION_REPORT.md`

### 4.3 Reward-normalization sensitivity

- report:
  - `docs/technical-reports/REWARD_NORMALIZATION_ABLATION_REPORT.md`
- machine-readable summary:
  - `docs/technical-reports/REWARD_NORMALIZATION_ABLATION_SUMMARY.json`

### 4.4 Feasibility sparsity

- report:
  - `docs/technical-reports/FEASIBILITY_SPARSITY_REPORT.md`
- checked-in summaries:
  - `analysis/feasibility_sparsity/summary.json`
  - `analysis/feasibility_sparsity/per_seed_summary.csv`
  - `analysis/feasibility_sparsity/progress_bins.csv`
  - `analysis/feasibility_sparsity/stage_pooled_summary.csv`

### 4.5 Timeout analysis

- report:
  - `docs/technical-reports/TIMEOUT_FAILURE_ANALYSIS_REPORT.md`

### 4.6 Appendix manifest

- manifest:
  - `experiments/report_sets/paper_public_appendix_v1.json`

## 5. Additional-domain evidence

The paper includes JSSP evidence, and this release keeps the companion artifact identifiers explicit in `docs/release_artifact_inventory.json` so the additional-domain evidence remains easy to trace alongside the SDS core release.

## 6. Default user journey

The default public documentation path for this release is:

1. `README.md`
2. `docs/REPRODUCTION.md`
3. `docs/technical-reports/README.md`

Internal review-response archives are intentionally omitted from this standalone code release.
