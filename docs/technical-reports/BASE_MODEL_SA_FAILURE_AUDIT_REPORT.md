# Base Model SA Failure Audit

## Summary

This note documents a fresh raw-generation audit of the Base Best-of-64 SDS code pool, aligned to the same unique-code population used by universal search.

- **Raw source**: private HF datasets `SoheylM/OpenR1-SDS-Base-Generations-seed{101,202,303}`
- **Authenticated access path**: Clariden token at `$HOME/llm/hf_token.txt`
- **Population**: `192,000` raw generations across three seeds
- **Extractable code blocks**: `191,699`
- **Unique extracted codes**: `191,699`

The exact code extraction semantics were matched to [universal_solver_search.py](evaluation/sds/universal_solver_search.py): extract code from `<code>...</code>` and canonicalize whitespace before deduplication.

## SA-Like Subset

Applying the same SA-like structural heuristic to the unique raw code pool yields:

- **SA-like unique codes**: `41,903` (`21.86%` of the unique raw code pool)

Within that SA-like subset:

- **Current-state acceptance detected**: `25,517` (`60.90%`)
- **Global-best acceptance bug detected**: `12,050` (`28.76%`)
- **Mixed acceptance signals**: `843` (`2.01%`)
- **Acceptance unresolved by heuristic**: `3,493` (`8.34%`)

This confirms that the global-best bug is a substantial recurrent failure mode, but not the only reason SA-like retrieved templates fail to become strong reusable solvers.

## Coarse Failure Taxonomy

We then split the entire SA-like subset into exclusive, coarse structural buckets:

| Bucket | Count | % of SA-like |
|---|---:|---:|
| `best_bug` | 12,050 | 28.76% |
| `ambiguous_acceptance` | 4,336 | 10.35% |
| `current_ok_no_guard` | 1,597 | 3.81% |
| `current_ok_no_best_tracking` | 12,538 | 29.92% |
| `current_ok_guarded_but_weak_moves` | 4,281 | 10.22% |
| `current_ok_structurally_complete` | 7,101 | 16.95% |

Interpretation:

- The largest non-`best_bug` bucket is **`current_ok_no_best_tracking`**.
- Even after removing the global-best bug, the majority of SA-like codes are still structurally incomplete as reusable solvers.

## Non-Bug Remainder

If we remove the `best_bug` bucket, the remaining SA-like population is:

- **Non-`best_bug` SA-like codes**: `29,853`

Breakdown within that remainder:

- `ambiguous_acceptance`: `4,336` (`14.52%` of remainder)
- `current_ok_no_guard`: `1,597` (`5.35%`)
- `current_ok_no_best_tracking`: `12,538` (`42.00%`)
- `current_ok_guarded_but_weak_moves`: `4,281` (`14.34%`)
- `current_ok_structurally_complete`: `7,101` (`23.79%`)

This is the key answer to “why does the rest of SA still do poorly?”:

- only about **one quarter** of the non-`best_bug` SA-like codes appear structurally complete under this coarse static audit
- the rest are dominated by missing best-solution tracking, ambiguous acceptance logic, or weak / one-sided neighborhood structure

## Important Caveat

These buckets are **structural heuristics**, not a complete semantic proof of solver quality.

In particular:

- `current_ok_structurally_complete` does **not** mean “good solver”
- it only means the code appears to have the three core ingredients this audit checked:
  - current-state-style acceptance
  - explicit feasibility guard
  - explicit best-solution tracking
  - plus a non-degenerate two-way neighborhood heuristic

So this audit supports a measured claim:

> Base-model retrieval often reaches SA-like syntax, but much of the retrieved pool remains operationally incomplete. The global-best bug is one important recurrent failure mode, yet the larger pattern is broader semantic incompleteness rather than a single isolated bug.

## Implementation

- **Script**: [analyze_sa_failure_modes.py](evaluation/sds/analyze_sa_failure_modes.py)
- **Remote execution environment**: Clariden `python3` with HF auth
- **Date of audit**: `2026-05-02`
