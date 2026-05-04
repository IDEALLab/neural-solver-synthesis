# llm-finetuning

Code release for the paper "Beyond Inference-Time Search: Reinforcement Learning Synthesizes Reusable Solvers".

This repository packages the paper-backed experiments as first-class reproducibility surfaces. The main benchmark is Synergistic Dependency Selection (SDS), with additional-domain evidence on the Job Shop Scheduling Problem (JSSP) inventoried in the release manifest.

## Quick start

```bash
# Local environment
./setup_dev.sh --dev
conda activate llm-finetuning

# Validate everything that is honestly checkable from a local checkout
./scripts/validate_paper_release.sh

# Recreate the main paper bundles from the canonical public manifest
./scripts/generate_paper_results.sh

# Validate appendix/supporting evidence and regenerate soft-gate plots when
# the corresponding frozen SDS result roots are available locally
./scripts/generate_paper_appendix_results.sh
```

## Canonical public entrypoints

- Main paper manifest:
  - `experiments/report_sets/paper_public_main_v1.json`
- Appendix / supporting-evidence manifest:
  - `experiments/report_sets/paper_public_appendix_v1.json`
- Human-readable release manifest:
  - `docs/release_manifest.md`
- Machine-readable artifact inventory:
  - `docs/release_artifact_inventory.json`

The checked-in paper bundles currently live at:

- SDS:
  - `evaluation/sds/aggregated_report_batches/paper_public_main_v1/`
- BigCode:
  - `evaluation/bigcode/aggregated_report_batches/paper_public_main_v1/`
- Fixed-code / runtime audit bundle:
  - `evaluation/sds/aggregated_report_batches/20260326_baseline-eval-v1/`

## What is included in this release

The public release path now includes:

- refreshed SDS baseline package with the neutral-prompt ShinkaEvolve rerun
- fixed-code SDS evaluation support for frozen-solver validation
- manually specified constraint-aware simulated annealing baseline
- soft-gate SDS ablation support
- reward-normalization ablation support
- feasibility-sparsity logging + summary artifacts
- timeout analysis report
- paper-aligned manifests, figures, tables, and release docs

CVRP remains intentionally out of scope for this release.

## Reproducibility model

This repository now has two complementary reproducibility modes:

1. **Paper-bundle verification**
   - inspect the checked-in aggregated outputs that match the final manuscript
   - use `docs/release_manifest.md` to map every paper-facing number to its source bundle

2. **Frozen-artifact regeneration**
   - regenerate the paper figures/tables from the canonical manifests once the frozen private result roots have been synced locally
   - regenerate the appendix-only SDS analyses through the dedicated appendix manifest and helper script

The repo intentionally keeps the main SDS comparison frame separate from the late diagnostic ablations so the virtual-best-solver denominator for the headline figures remains stable.

## Local vs cluster validation

Most of the public-release surface can be validated on a MacBook:

- checked-in bundle presence
- manifest and inventory integrity
- shell syntax
- SDS / BigCode / open-r1 tests
- appendix/supporting-evidence validation

The remaining cluster-only checks are:

- `sbatch` / `srun` launcher smoke tests
- EDF environment resolution
- Capstor checkpoint and dataset path assumptions
- any retraining or reevaluation that depends on GH200 resources or private frozen roots

Use `./scripts/validate_paper_release.sh` for the local portion first, then use `docs/REPRODUCTION.md` for the remaining cluster-only checks.

If you want the validator itself to exercise the full main-paper regeneration path, run:

```bash
./scripts/validate_paper_release.sh --run-main-regen
```

## Repository structure

```text
llm-finetuning/
├── evaluation/                    # SDS + BigCode evaluation and aggregation
├── analysis/feasibility_sparsity/ # Checked-in feasibility-density summaries
├── experiments/report_sets/       # Canonical public manifests
├── docs/                          # Release manifest, reproduction guide, technical reports
├── scripts/                       # Training/evaluation launchers and reproducibility helpers
├── deps/                          # Pinned companion dependency trees
└── tests/                         # Top-level validation tests
```

## Default documentation path

If you are trying to reproduce the paper, start here:

1. `docs/release_manifest.md`
2. `docs/REPRODUCTION.md`
3. `docs/technical-reports/README.md`

Internal review-response archives are intentionally omitted from this standalone code release.
