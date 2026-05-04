# Technical Reports

These reports provide the detailed methodology and supporting evidence behind the paper release surface.

If you are new to the repo, start with:

1. `../release_manifest.md`
2. `../REPRODUCTION.md`
3. the reports below as needed

## Release-critical supporting reports

- `PROBLEM_TRANSFER_PLAYBOOK.md`
  - how to adapt the solver-synthesis recipe from SDS to a new domain such as JSSP or CVRP
- `SOFT_GATE_ABLATION_REPORT.md`
  - SDS soft-gate appendix ablation
- `REWARD_NORMALIZATION_ABLATION_REPORT.md`
  - SDS nominal-normalization sensitivity study
- `REWARD_NORMALIZATION_ABLATION_SUMMARY.json`
  - machine-readable summary for the normalization study
- `FEASIBILITY_SPARSITY_REPORT.md`
  - exact 64-sample GRPO feasibility-density measurement
- `TIMEOUT_FAILURE_ANALYSIS_REPORT.md`
  - SDS timeout concentration analysis

## Core pipeline reports

- `EVALUATION_PIPELINE_REPORT.md`
  - SDS / BigCode evaluation methodology
- `EXPERIMENT_MANAGEMENT_REPORT.md`
  - frozen batch/report-set conventions
- `PASS_AT_K_EVALUATION_REPORT.md`
  - base-model scaling and pass@k analysis
- `BIGCODE_EVALUATION_REPORT.md`
  - HumanEval / MBPP evaluation flow
- `HERO_ABLATION_TRAINING_REPORT.md`
  - default SDS training stack
- `SYNDEOPT_INTEGRATION_REPORT.md`
  - SDS simulator/problem-library integration
- `ALGORITHMIC_CONVERGENCE_ANALYSIS_REPORT.md`
  - convergence/code-family analysis
- `UNIVERSAL_SOLVER_SEARCH_REPORT.md`
  - universal-code appendix analysis

## Notes on archival material

Some historical notes still exist elsewhere in the repository history, but the default public documentation path should now flow through the release manifest and the public report-set manifests rather than through review-oriented artifacts.
