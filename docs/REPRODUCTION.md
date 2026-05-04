# Paper Reproduction Guide

This guide is organized around the canonical manifests for the paper release.

## Reproduction levels

| Level | Goal | Requires private frozen result roots? | Typical command |
| --- | --- | --- | --- |
| 1 | Verify the checked-in paper bundles | No | Inspect `evaluation/*/aggregated_report_batches/paper_public_main_v1/` |
| 2 | Rebuild the main paper figures/tables from frozen SDS/BigCode roots | Yes | `./scripts/generate_paper_results.sh` |
| 3 | Validate appendix/supporting evidence | Partly | `./scripts/generate_paper_appendix_results.sh` |
| 4 | Full retraining / reevaluation | Yes (cluster + credentials) | Domain- and cluster-specific scripts in `scripts/` and `deps/open-r1/recipes/` |

## Where each validation runs

### MacBook-local validation

The following can and should be validated from a normal local checkout:

- `./scripts/validate_paper_release.sh`
- manifest and inventory parsing
- checked-in bundle presence
- SDS / BigCode / open-r1 unit tests
- appendix/supporting-evidence validation
- shell syntax for release scripts and cluster launchers

This is the first validation pass because it checks that the branch is coherent as shipped.

If you want that validator to also rerun the full main-paper aggregation path, use:

```bash
./scripts/validate_paper_release.sh --run-main-regen
```

That stronger check will rewrite the checked-in aggregate outputs, so it is best run from a clean worktree.

### Clariden / cluster validation

The following are not honestly MacBook validations:

- `sbatch` / `srun` smoke tests for the SDS evaluation launchers
- EDF environment selection
- Capstor checkpoint and dataset path assumptions
- full regeneration runs that depend on frozen result roots not present in the local checkout
- any retraining or reevaluation using GH200 resources

In other words:

- the release branch can be structurally and functionally validated locally
- the cluster launch path still needs at least one Clariden smoke pass before calling the whole public stack fully exercised

## Canonical manifests

- Main paper:
  - `experiments/report_sets/paper_public_main_v1.json`
- Appendix / supporting evidence:
  - `experiments/report_sets/paper_public_appendix_v1.json`
- Release mapping:
  - `docs/release_manifest.md`
- Artifact inventory:
  - `docs/release_artifact_inventory.json`

## Level 1: Verify the checked-in paper bundles

This level does not require Hugging Face, W&B, or cluster access.

Inspect the checked-in outputs:

- SDS paper bundle:
  - `evaluation/sds/aggregated_report_batches/paper_public_main_v1/`
- BigCode paper bundle:
  - `evaluation/bigcode/aggregated_report_batches/paper_public_main_v1/`
- Fixed-code / runtime bundle:
  - `evaluation/sds/aggregated_report_batches/20260326_baseline-eval-v1/`

Then compare those against the release snapshot note:

- release snapshot note:
  - `docs/NEURIPS_2026_CODE_RELEASE_SNAPSHOT.md`

## Level 2: Rebuild the main paper bundles

This level assumes you have synced the frozen evaluation roots listed in `experiments/report_sets/paper_public_main_v1.json` into the expected local locations.

```bash
./scripts/generate_paper_results.sh
```

This regenerates:

- SDS plots/tables into `evaluation/sds/aggregated_report_batches/paper_public_main_v1/`
- BigCode table into `evaluation/bigcode/aggregated_report_batches/paper_public_main_v1/`
- Universal-solver appendix aggregate if the referenced batch is present

Important note:

- the public branch ships the final aggregated paper bundles
- it does **not** ship all large private/raw evaluation roots
- those roots remain inventoried in `docs/release_artifact_inventory.json` and are intended to be published/synced in a dedicated artifact-publication pass
- if those roots live only on Clariden or shared storage, this level should be treated as a cluster-backed validation rather than a pure local validation

## Level 3: Validate appendix and supporting evidence

```bash
./scripts/generate_paper_appendix_results.sh
```

This helper does two things:

- validates the checked-in appendix/supporting summaries:
  - soft-gate report
  - reward-normalization summary
  - feasibility-sparsity summaries
  - timeout analysis report
  - fixed-code baseline/runtime bundle
- conditionally regenerates the SDS soft-gate aggregate if the required frozen result roots are present locally

Supporting evidence currently lives in:

- `docs/technical-reports/SOFT_GATE_ABLATION_REPORT.md`
- `docs/technical-reports/REWARD_NORMALIZATION_ABLATION_REPORT.md`
- `docs/technical-reports/REWARD_NORMALIZATION_ABLATION_SUMMARY.json`
- `docs/technical-reports/FEASIBILITY_SPARSITY_REPORT.md`
- `analysis/feasibility_sparsity/`
- `docs/technical-reports/TIMEOUT_FAILURE_ANALYSIS_REPORT.md`
- `evaluation/sds/aggregated_report_batches/20260326_baseline-eval-v1/`

## Level 4: Full retraining / reevaluation

This level is the full research stack:

- cluster/container access
- private datasets/checkpoints or regeneration from scratch
- Hugging Face tokens
- W&B credentials

The training/evaluation stack is split across:

- top-level launch scripts in `scripts/`
- GRPO configs in `deps/open-r1/recipes/Qwen2.5-Coder-14B-Instruct/grpo/`
- SDS evaluation scripts in `evaluation/sds/`
- companion additional-domain artifacts inventoried in `docs/release_manifest.md`

## Recommended validation order

1. Run `./scripts/validate_paper_release.sh` locally.
2. If the frozen SDS/BigCode roots are available locally, let that script run `./scripts/generate_paper_results.sh`; otherwise, sync those roots or switch to Clariden for that step.
3. On Clariden, smoke-test the launchers that the public release depends on:
   - `scripts/eval_capstor_sds_fixed_code.slurm`
   - `scripts/eval_capstor_sds_pipeline.slurm`
   - `scripts/evaluate_baseline_evidence.sh`
4. Only after those pass should Level 4 retraining or full reruns be considered validated.

## Additional-domain note: JSSP

The paper includes JSSP evidence, and this release keeps the companion artifact identifiers explicit in `docs/release_artifact_inventory.json`.
