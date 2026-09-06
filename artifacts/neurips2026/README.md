# Final NeurIPS 2026 Evidence

This directory contains the compact, sanitized evidence underlying the final
paper. Values are generated from frozen result files rather than retyped.

## Contents

- `sds/`: certified full-method summaries, per-seed results, paired comparisons,
  certified reference validation, and the Hero-excluded VBS.
- `base_best_of_64/`: duplicate-selection audit of immutable Base generations.
- `adaptive_repair/`: same-model and hosted execution-feedback protocols.
- `universal_search/`: input-disjoint selection from the 191,699-program pool.
- `cost/`: end-to-end GPU/CPU accounting and break-even calculations.
- `hypothesize/`: inference/training sensitivity and all reported later checkpoints.
- `jssp/`: compile-once standard-instance evaluation.
- `tsp/`: direct-from-base RL boundary evaluation and all corrected trial rows.
- `solvers/`: frozen selected solver programs available in this compact release.

The TSP primary gate failed, and its corrected parser evaluation is disclosed as
post-result rather than test-blind confirmation. The hosted repair comparison is
not token-, dollar-, latency-, or training-compute-matched. These results do not
support a claim that reinforcement learning is uniquely necessary.

`checksums.sha256` records every file in this directory except itself. The public
claim map is `docs/final_evidence_index.json`.
