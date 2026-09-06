# Neural Solver Synthesis

Code release for the paper "Beyond Inference-Time Search: Reinforcement Learning Synthesizes Reusable Solvers".

This repository packages the paper-backed experiments as first-class reproducibility surfaces. The main benchmark is Synergistic Dependency Selection (SDS), with additional-domain evidence on the Job Shop Scheduling Problem (JSSP) and a bounded Traveling Salesperson Problem (TSP) extension trained with RL directly from the base model without an SFT stage.

## Quick start

```bash
# Local environment
./setup_dev.sh --dev
conda activate llm-finetuning

# Validate everything that is honestly checkable from a local checkout
./scripts/validate_paper_release.sh

# Validate the compact final-evidence package directly
python scripts/validate_neurips2026_public_evidence.py
```

## Canonical public entrypoints

- Versioned GitHub release:
  - https://github.com/IDEALLab/neural-solver-synthesis/releases/tag/neurips2026-evidence-v1.0.0
- Immutable compact evidence:
  - https://huggingface.co/datasets/IDEALLab/Neural-Solver-Synthesis-Final-Evidence-v1/tree/c34da924fbb4f0645e24b6061678bc206dd630a0
- Curated W&B report:
  - https://wandb.ai/neural-solver-synthesis/neurips2026-evidence-v2/reports/Neural-Solver-Synthesis:-Final-Evidence--VmlldzoxNzg4MDQ3Nw==
- Main paper manifest:
  - `experiments/report_sets/paper_public_main_v1.json`
- Appendix / supporting-evidence manifest:
  - `experiments/report_sets/paper_public_appendix_v1.json`
- Human-readable release manifest:
  - `docs/release_manifest.md`
- Machine-readable artifact inventory:
  - `docs/release_artifact_inventory.json`
- Final evidence map:
  - `docs/EVIDENCE_MAP.md`
  - `docs/final_evidence_index.json`

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
- paper-aligned manifests, figures, tables, and release docs
- certified SDS references and duplicate-safe Base Best-of-64 results
- same-model and hosted adaptive-repair controls
- input-disjoint universal search, end-to-end cost accounting, and prompt sensitivity
- compile-once JSSP and bounded TSP evaluations, including adverse outcomes

CVRP remains intentionally out of scope for this release.

## Reproducibility model

This repository now has two complementary reproducibility modes:

1. **Paper-bundle verification**
   - inspect the checked-in aggregated outputs that match the final manuscript
   - use `docs/release_manifest.md` to map every paper-facing number to its source bundle

2. **Frozen-artifact regeneration**
   - download the immutable large artifacts referenced by the evidence index
   - regenerate paper figures and tables from the canonical report manifests

The repo intentionally keeps the main SDS comparison frame separate from the late diagnostic ablations so the virtual-best-solver denominator for the headline figures remains stable.

## Validation scope

Most of the public-release surface can be validated on a MacBook:

- checked-in bundle presence
- manifest and inventory integrity
- shell syntax
- SDS / BigCode / open-r1 tests
- compact final-evidence checksum and claim validation

Full regeneration additionally requires:

- the large model, generation, and evaluation artifacts linked from the evidence index
- a compatible GPU environment for model inference or retraining
- explicit local paths supplied by the user rather than embedded infrastructure paths

Use `./scripts/validate_paper_release.sh` for the local portion first, then use
`docs/REPRODUCTION.md` for the artifact-backed regeneration path.

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
├── docs/                          # Evidence map, release manifest, and reproduction guide
├── scripts/                       # Portable aggregation and validation helpers
├── deps/                          # Pinned companion dependency trees
└── tests/                         # Top-level validation tests
```

## Default documentation path

If you are trying to reproduce the paper, start here:

1. `docs/release_manifest.md`
2. `docs/REPRODUCTION.md`
3. `docs/LICENSING.md`

Private correspondence and internal publication records are intentionally omitted from this standalone code release.
