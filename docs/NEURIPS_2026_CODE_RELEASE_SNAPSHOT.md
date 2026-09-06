# NeurIPS 2026 Code Release Snapshot

This note records the versioned public surfaces for the code release and the
anonymized NeurIPS supplementary bundle.

## Release anchor

- The public Git commit and neutral release tag identify the exact released tree.
- Immutable Hugging Face revisions identify large artifacts outside Git.
- The evidence index records checksums for every compact result file.

## Paper

- Title:
  - `Beyond Inference-Time Search: Reinforcement Learning Synthesizes Reusable Solvers`

## Artifact surfaces pinned at this snapshot

- Hugging Face collection:
  - `https://huggingface.co/collections/IDEALLab/neural-solver-synthesis`
- W&B final-paper project:
  - `https://wandb.ai/neural-solver-synthesis/qwen-coder-sds-rl?nw=01nxi1s0ex1`

## Submodule pins relevant to the release export

- `deps/open-r1`
  - `fc26a663ee5d290a971f1172e673d6bc39ca0870`
- `deps/syndeopt`
  - `d5bbbb8ebe5350db9fd07ce23bac766d9fc6f825`
- `deps/ShinkaEvolve`
  - `202269eb9adcb788e047470721c2cf91216fec89`
- `deps/bigcode-evaluation-harness`
  - `b89ac82afbe9e945d44db8776d3b3fc56bf87c5b`

## Export products

The export pipeline emits two products from the same frozen scientific state:

1. `public-release/`
   - a clean fresh git repository intended for the eventual public code release
2. `neurips-anon/`
   - an anonymized supplementary tree
3. `dist/llm-finetuning-neurips2026-anonymized.zip`
   - the zip derived from `neurips-anon/`

Both exports share the same scientific/reproducibility baseline. They differ
only in release-facing rewrites and anonymization redactions.
