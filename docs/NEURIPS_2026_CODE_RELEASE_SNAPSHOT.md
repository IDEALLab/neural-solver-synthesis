# NeurIPS 2026 Code Release Snapshot

This note records the canonical release snapshot used to derive both the public
code-release export and the anonymized NeurIPS supplementary bundle.

## Baseline

- Branch:
  - `release snapshot`
- Annotated tag:
  - `neurips2026-submission`
- Commit:
  - `882b8199b93229cc92942bd2161fb3266e52ed77`

## Paper

- Title:
  - `Beyond Inference-Time Search: Reinforcement Learning Synthesizes Reusable Solvers`

## Artifact surfaces pinned at this snapshot

- Hugging Face collection:
  - `https://huggingface.co/collections/SoheylM/neural-solver-synthesis-698b3b24b714db59dde6bf02`
- W&B final-paper workspace:
  - `https://wandb.ai/smassoudi-eth-z-rich/qwen-coder-sds-rl?nw=httg3nl3fo8`

## Submodule pins relevant to the release export

- `deps/open-r1`
  - `6ee5e32f4af11fb2f5d95fe8ecfca3ea8fbd6e0e`
- `deps/syndeopt`
  - `d5bbbb8b01b2149a615fd0f4981ebf9f3b3e1d2f`
- `deps/ShinkaEvolve`
  - `202269eb9adcb788e047470721c2cf91216fec89`
- `deps/bigcode-evaluation-harness`
  - `b89ac8226700f9f1fb0f93834ca2ef8de1a0f5ff`

## Export products

The export pipeline emits two products from this exact source tag:

1. `public-release/`
   - a clean fresh git repository intended for the eventual public code release
2. `neurips-anon/`
   - an anonymized supplementary tree
3. `dist/llm-finetuning-neurips2026-anonymized.zip`
   - the zip derived from `neurips-anon/`

Both exports share the same scientific/reproducibility baseline. They differ
only in release-facing rewrites and anonymization redactions.
