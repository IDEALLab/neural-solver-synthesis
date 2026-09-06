# Final Evidence Map

This page maps the paper's final scientific claims to the compact public result
files in `artifacts/neurips2026/`. Exact values, uncertainty definitions, sample
counts, checksums, and immutable remote revisions are recorded in
`docs/final_evidence_index.json`.

## Public Surfaces

- [GitHub release](https://github.com/IDEALLab/neural-solver-synthesis/releases/tag/neurips2026-evidence-v1.0.0)
- [Immutable Hugging Face evidence](https://huggingface.co/datasets/IDEALLab/Neural-Solver-Synthesis-Final-Evidence-v1/tree/c34da924fbb4f0645e24b6061678bc206dd630a0)
- [Curated W&B report](https://wandb.ai/neural-solver-synthesis/neurips2026-evidence-v2/reports/Neural-Solver-Synthesis:-Final-Evidence--VmlldzoxNzg4MDQ3Nw==)

The W&B report is an aggregated visualization surface. The checksum-verified
files in GitHub and Hugging Face remain the canonical result store.

## Policy Checkpoints

The six newly released checkpoints contain inference files only. Each model
repository includes `files.sha256.json`; all listed files and repository cards
were verified through anonymous downloads at the immutable revisions below.

| Condition | Seed | Immutable Hugging Face revision |
| --- | ---: | --- |
| TSP RL policy | 101 | [`3241ef5e`](https://huggingface.co/IDEALLab/Qwen2.5-Coder-14B-Instruct-GRPO-TSP-Hero-seed101/tree/3241ef5e1de219b083283324481d6bc604b6227a) |
| TSP RL policy | 202 | [`0290cbfb`](https://huggingface.co/IDEALLab/Qwen2.5-Coder-14B-Instruct-GRPO-TSP-Hero-seed202/tree/0290cbfb7e9bf45239711cf8885c326272b9dbb7) |
| TSP RL policy | 303 | [`1c1b2c23`](https://huggingface.co/IDEALLab/Qwen2.5-Coder-14B-Instruct-GRPO-TSP-Hero-seed303/tree/1c1b2c23c6ed277d5f6fd75522d840dc09b15450) |
| SDS without `Hypothesize`, step 90 | 101 | [`1271c838`](https://huggingface.co/IDEALLab/Qwen2.5-Coder-14B-Instruct-GRPO-SDS-NoHypothesize-step90-seed101/tree/1271c838d73f9f919df66fbf1ba87068204fad9d) |
| SDS without `Hypothesize`, step 90 | 202 | [`c914264f`](https://huggingface.co/IDEALLab/Qwen2.5-Coder-14B-Instruct-GRPO-SDS-NoHypothesize-step90-seed202/tree/c914264f1193a246b98e2211c884b32f68ecde68) |
| SDS without `Hypothesize`, step 90 | 303 | [`c06552a0`](https://huggingface.co/IDEALLab/Qwen2.5-Coder-14B-Instruct-GRPO-SDS-NoHypothesize-step90-seed303/tree/c06552a00c43a2e53f54c4029dc2d65f3ce3ea92) |

They are also appended to the public
[Neural Solver Synthesis collection](https://huggingface.co/collections/IDEALLab/neural-solver-synthesis-698b3e0434677fd2d2f5b3cf).

## SDS

The certified SDS package reports exact optima where proved, explicit intervals
for unresolved feasible rows, and transparent handling of 51 independently
reproved infeasible rows. All comparisons join strictly on `(seed, uuid)`.

| Evidence | Public file | Scope |
| --- | --- | --- |
| Hero and all frozen methods | `artifacts/neurips2026/sds/certified_summary.json` | Content-clean, three-seed certified evaluation |
| Per-seed results | `artifacts/neurips2026/sds/certified_summary_by_seed.json` | Counts, pass rates, gaps, intervals, and infeasible handling |
| Base Best-of-64 audit | `artifacts/neurips2026/base_best_of_64/selection_audit.json` | Unique-ID rescoring of immutable generations |
| Same-model adaptive repair | `artifacts/neurips2026/adaptive_repair/same_model_protocol_summary.json` | Qwen 14B, 64 completions, real execution feedback |
| Hosted adaptive repair | `artifacts/neurips2026/adaptive_repair/hosted_protocol_summary.json` | GPT-5.4-mini, medium reasoning, 64 completions |
| Universal search | `artifacts/neurips2026/universal_search/protocol_summary.json` | Input-disjoint selection from 191,699 unique programs |
| Cost boundary | `artifacts/neurips2026/cost/cost_break_even.json` | Separate GPU, CPU, wall-time, and break-even accounting |

The hosted control is not token-, dollar-, latency-, or training-compute-matched,
and its final allocation followed an earlier truncated attempt. It demonstrates
that a stronger deployment-time reasoner can be a viable alternative; it does
not erase the amortization result for the smaller open policy. The same-model
control tests one execution-feedback controller and does not exhaust non-RL
search. None of these results supports unique necessity of reinforcement learning.

## Prompt Sensitivity

| Evidence | Public file | Scope |
| --- | --- | --- |
| Inference-time removal | `artifacts/neurips2026/hypothesize/inference_time_summary.json` | Removes exactly one instruction from frozen policies |
| Training-time step 90 | `artifacts/neurips2026/hypothesize/training_step90_summary.json` | Historical stack with matched control |
| Later checkpoints | `artifacts/neurips2026/hypothesize/extended_trajectory_summary.json` | Post hoc steps 120, 150, and 180, all reported |
| Program families | `artifacts/neurips2026/hypothesize/program_audit_summary.json` | Frozen taxonomy and code-deduplicated audit |

The instruction is not load-bearing at frozen-policy inference. At training step
90 it improves sample efficiency and stability, with substantial seed
heterogeneity. Later recovery is non-monotonic, so the evidence does not imply
inaccessible knowledge or universal prompt necessity.

## Additional Domains

| Evidence | Public file | Scope |
| --- | --- | --- |
| JSSP | `artifacts/neurips2026/jssp/summary.json` | 750 compile-once trials on 50 standard instances |
| TSP | `artifacts/neurips2026/tsp/summary.json` | 180 trained-policy trials plus Base and native baselines |
| TSP trial rows | `artifacts/neurips2026/tsp/per_trial.csv` | Corrected per-instance outcomes |
| Frozen programs | `artifacts/neurips2026/solvers/` | Hosted-repair and TSP selected solver source |

JSSP is a within-family deployment test using JSSP-trained policies. The TSP
policies were trained with RL directly from the base model without an SFT stage.
All three trained TSP seeds are feasible, but the primary quality/stability gate
fails and native 2-opt and OR-Tools remain stronger. The corrected TSP evaluation
uses a post-result parser correction and is boundary evidence, not blind
confirmation.

## Verification

```bash
python scripts/validate_neurips2026_public_evidence.py
pytest -q tests/release/test_neurips2026_public_evidence.py
```

The Qwen same-model adaptive-repair programs are identified by their frozen
SHA-256 values in the protocol summary; their historical source files were not
available in the final compact publication workspace. No program is reconstructed
from observed test outcomes.
