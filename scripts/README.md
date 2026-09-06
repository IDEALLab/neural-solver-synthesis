# Release Scripts

This directory contains the portable helpers distributed with the paper release.
Cluster launchers and internal job-management scripts are intentionally omitted.

## Validation

- `validate_paper_release.sh` checks manifests, checked-in bundles, focused unit
  tests, dependency tests, and the compact final-evidence package.
- `validate_neurips2026_public_evidence.py` verifies package membership,
  checksums, claim mappings, certified intervals, solver hashes, and privacy
  rules.

```bash
./scripts/validate_paper_release.sh
python scripts/validate_neurips2026_public_evidence.py
```

## Aggregation

- `generate_paper_results.sh` regenerates the main SDS and BigCode bundles from
  the frozen roots declared in `experiments/report_sets/paper_public_main_v1.json`.
- `generate_paper_appendix_results.sh` validates or regenerates supporting
  analyses when their larger source bundles are available.
- `analyze_feasibility_sparsity.py` aggregates feasibility-density diagnostics.
- `generate_jssp_seed_averaged_robustness.py` recreates the JSSP performance
  profile from explicitly supplied frozen result and output paths.

Large inputs are not duplicated in Git. Obtain the immutable artifact revisions
listed in `docs/final_evidence_index.json` before requesting full regeneration.
