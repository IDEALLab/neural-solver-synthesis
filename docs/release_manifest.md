# Paper Release Manifest

This document defines the public code and evidence surface for "Beyond
Inference-Time Search: Reinforcement Learning Synthesizes Reusable Solvers."

## Scope

The release contains:

- certified SDS references with exact optima, bound intervals, and transparent
  handling of proved-infeasible instances;
- duplicate-safe Base Best-of-64 scoring and a Hero-excluded virtual best solver;
- same-model and hosted adaptive-repair controls;
- input-disjoint universal search and end-to-end cost accounting;
- inference-time and training-time `Hypothesize` sensitivity;
- compile-once JSSP evidence and bounded TSP boundary evidence;
- focused evaluation, aggregation, privacy, and consistency tests.

CVRP is outside the release scope. The evidence does not claim that RL is
uniquely necessary, that transfer is tuning-free, or that generated solvers
dominate native solvers.

## Dependency snapshots

The export vendors exact dependency snapshots rather than public submodule
metadata:

| Dependency | Commit | Role |
| --- | --- | --- |
| `deps/open-r1` | `fc26a663ee5d290a971f1172e673d6bc39ca0870` | Training and reward stack |
| `deps/syndeopt` | `d5bbbb8ebe5350db9fd07ce23bac766d9fc6f825` | SDS instances and simulator |
| `deps/ShinkaEvolve` | `202269eb9adcb788e047470721c2cf91216fec89` | Neutral-prompt evolutionary baseline |
| `deps/bigcode-evaluation-harness` | `b89ac82afbe9e945d44db8776d3b3fc56bf87c5b` | Frozen code-evaluation support |

## Compact final evidence

- Claim index: `docs/final_evidence_index.json`
- Human map: `docs/EVIDENCE_MAP.md`
- Checksummed package: `artifacts/neurips2026/`
- Validator: `scripts/validate_neurips2026_public_evidence.py`
- GitHub release:
  `https://github.com/IDEALLab/neural-solver-synthesis/releases/tag/neurips2026-evidence-v1.0.0`
- Hugging Face evidence revision:
  `IDEALLab/Neural-Solver-Synthesis-Final-Evidence-v1@c34da924fbb4f0645e24b6061678bc206dd630a0`
- W&B report:
  `https://wandb.ai/neural-solver-synthesis/neurips2026-evidence-v2/reports/Neural-Solver-Synthesis:-Final-Evidence--VmlldzoxNzg4MDQ3Nw==`

The package contains the final SDS, adaptive-repair, cost, universal-search,
prompt-sensitivity, JSSP, and TSP summaries. TSP trial rows and available frozen
solver programs are included. The same-model adaptive-repair programs remain
identified by immutable SHA-256 values because their historical source files
were unavailable in the compact publication workspace.

## Paper bundles

- SDS: `evaluation/sds/aggregated_report_batches/paper_public_main_v1/`
- BigCode: `evaluation/bigcode/aggregated_report_batches/paper_public_main_v1/`
- Fixed-code/runtime evidence:
  `evaluation/sds/aggregated_report_batches/20260326_baseline-eval-v1/`
- Main report manifest: `experiments/report_sets/paper_public_main_v1.json`
- Appendix report manifest: `experiments/report_sets/paper_public_appendix_v1.json`

Large candidate pools, checkpoints, and raw evaluation roots are published as
separate immutable artifacts rather than duplicated in Git.

## Corrections and limitations

- Base Best-of-64 is scored over unique selected IDs.
- Universal search is input-disjoint and joins strictly on `(seed, uuid)`.
- The hosted repair control is not token-, dollar-, latency-, or
  training-compute-matched to the open-model control.
- Its final allocation followed an earlier truncated attempt.
- The corrected TSP parser was applied after outcomes were observed, so TSP is
  reported as boundary evidence rather than blind confirmation.
- The TSP primary quality/stability gate failed, and native baselines remained
  stronger.

## Validation

```bash
python scripts/validate_neurips2026_public_evidence.py
./scripts/validate_paper_release.sh
```

See `docs/REPRODUCTION.md` for regeneration levels and `docs/LICENSING.md` for
reuse terms.
