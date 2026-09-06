# Paper Reproduction Guide

This guide separates checks that run from a normal clone from workflows that
require large immutable model and evaluation artifacts.

## Reproduction levels

| Level | Goal | Additional artifacts required? | Entry point |
| --- | --- | --- | --- |
| 1 | Verify compact final evidence | No | `python scripts/validate_neurips2026_public_evidence.py` |
| 2 | Run focused evaluator tests | No | `./scripts/validate_paper_release.sh` |
| 3 | Regenerate aggregate tables and plots | Yes | `./scripts/generate_paper_results.sh` |
| 4 | Repeat inference or training | Yes, plus suitable compute | Domain-specific code under `evaluation/` and `deps/open-r1/` |

## Canonical entry points

- Human-readable claim map: `docs/EVIDENCE_MAP.md`
- Machine-readable claim index: `docs/final_evidence_index.json`
- Compact evidence package: `artifacts/neurips2026/`
- Main report manifest: `experiments/report_sets/paper_public_main_v1.json`
- Appendix report manifest: `experiments/report_sets/paper_public_appendix_v1.json`
- Artifact inventory: `docs/release_artifact_inventory.json`
- Immutable evidence snapshot:
  `IDEALLab/Neural-Solver-Synthesis-Final-Evidence-v1@c34da924fbb4f0645e24b6061678bc206dd630a0`
- Public visualization:
  `https://wandb.ai/neural-solver-synthesis/neurips2026-evidence-v2/reports/Neural-Solver-Synthesis:-Final-Evidence--VmlldzoxNzg4MDQ3Nw==`

## Level 1: compact evidence

The compact package covers certified SDS references, duplicate-safe Base
Best-of-64 evaluation, adaptive-repair controls, input-disjoint universal
search, cost accounting, prompt sensitivity, JSSP, and bounded TSP evidence.

```bash
python scripts/validate_neurips2026_public_evidence.py
pytest -q tests/release/test_neurips2026_public_evidence.py
```

These checks verify package membership, SHA-256 values, cross-file claim values,
certified intervals, frozen solver hashes, and the absence of private publication
material.

## Level 2: evaluator tests

```bash
./scripts/validate_paper_release.sh
```

The release validator parses the manifests, checks the committed SDS and
BigCode bundles, runs focused evaluator tests, and validates the compact evidence
package. Optional dependencies may cause explicitly reported test skips.

## Level 3: aggregate regeneration

The Git repository contains final compact outputs but does not duplicate every
large generation or evaluation pool. Download the immutable revisions recorded
in `docs/final_evidence_index.json`, place them at paths of your choice, and
update a local copy of the report manifest to point to those roots.

```bash
./scripts/generate_paper_results.sh path/to/local_report_manifest.json
```

Do not change seed membership, joins, or selection rules. SDS joins use
`(seed, uuid)`. Corrected TSP rows and post-result limitations must remain
visible when regenerating reports.

## Level 4: inference and training

The model-training stack is under `deps/open-r1/`; evaluation and aggregation
code is under `evaluation/`. Repeating the complete study requires the published
datasets and checkpoints, compatible accelerator hardware, and explicit local
storage configuration. The release intentionally contains no scheduler account,
private mount point, credential, or site-specific environment file.

The public evidence should be treated as the frozen target for any repetition:

1. Preserve model and dataset revisions.
2. Preserve seeds, prompts, selection budgets, and development/test separation.
3. Freeze selected programs before opening test outcomes.
4. Report adverse outcomes, unresolved bounds, and post-result corrections.
5. Compare regenerated aggregates with the checksums and values in the evidence
   index.

## Additional domains

JSSP is a compile-once deployment evaluation using JSSP-trained policies. TSP
is a bounded direct-from-base RL training test without an SFT stage. The TSP
quality/stability gate failed, and native 2-opt and OR-Tools baselines remained
stronger. See `docs/EVIDENCE_MAP.md` for the exact scope and limitations.

## Licensing

See `docs/LICENSING.md` before redistributing code, models, datasets, or standard
benchmark files.
