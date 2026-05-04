# Hero Timeout Failure Analysis

Date: 2026-03-29

Status: supporting analysis note for the final paper appendix package

Purpose: document what is currently supported about the `2.2%` Hero timeout cases on SDS, using the canonical three-seed Hero evaluation artifacts already used in the paper.

## 1. Source Data

Canonical Hero metrics were read from:

- `evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed101/job-1315163/metrics_final.csv`
- `evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed202/job-1315168/metrics_final.csv`
- `evaluation/sds/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed303/job-1315173/metrics_final.csv`

All `3,000` SDS evaluation rows were included.

Timeout rows were identified by:

- `error_type == "timeout"`

Per-instance structural features were parsed from `mission_summary`:

- `n_vars`
- `cardinality_bounds`
- `precedence`
- `mutex`
- `groups`
- `interactions`

## 2. High-Level Timeout Counts

Across the three canonical Hero seeds:

- total rows: `3,000`
- timeout rows: `65`

Difficulty breakdown of timeout rows:

- `57 / 65` on `Hard`
- `8 / 65` on `Moderate`
- `0 / 65` on `Trivial`

This already supports the claim that Hero timeouts concentrate on the hardest SDS strata.

## 3. Descriptive Comparison: Timeout vs Non-timeout

### 3.1 `n_vars`

- timeout rows:
  - mean `74.34`
  - median `75`
  - min `50`
  - max `100`
- non-timeout rows:
  - mean `22.82`
  - median `22`

### 3.2 `interactions`

- timeout rows:
  - mean `1518.69`
  - median `1527`
  - min `556`
  - max `3151`
- non-timeout rows:
  - mean `153.82`
  - median `102`

### 3.3 Other fields

- `card_hi`
  - timeout mean `47.06`
  - non-timeout mean `16.46`
- `card_lo`
  - timeout mean `14.14`
  - non-timeout mean `6.24`
- `precedence`
  - timeout mean `2.88`
  - non-timeout mean `2.73`
- `mutex`
  - timeout mean `2.66`
  - non-timeout mean `1.34`
- `groups`
  - timeout mean `2.63`
  - non-timeout mean `0.57`

The dominant visible pattern is size and interaction density, not precedence depth.

## 4. Statistical Tests

Because timeout rows are rare and the feature distributions are non-Gaussian, the main comparisons use:

- two-sided Mann-Whitney U tests for timeout vs non-timeout rows
- Holm correction across the tested features
- Fisher exact tests for enrichment claims

### 4.1 Mann-Whitney U results

| Feature | Timeout mean | Non-timeout mean | Raw p-value | Holm-adjusted p | Rank-biserial effect |
|---|---:|---:|---:|---:|---:|
| `n_vars` | `74.34` | `22.82` | `1.056e-41` | `6.337e-41` | `0.977` |
| `interactions` | `1518.69` | `153.82` | `1.393e-41` | `6.964e-41` | `0.977` |
| `card_hi` | `47.06` | `16.46` | `1.007e-33` | `4.027e-33` | `0.873` |
| `card_lo` | `14.14` | `6.24` | `2.321e-20` | `6.963e-20` | `0.651` |
| `groups` | `2.63` | `0.57` | `1.597e-48` | `1.118e-47` | `0.692` |
| `precedence` | `2.88` | `2.73` | `6.339e-01` | `6.339e-01` | `-0.033` |
| `mutex` | `2.66` | `1.34` | `2.890e-02` | `5.781e-02` | `0.146` |

Interpretation:

- `n_vars` and `interactions` are both extremely strongly associated with timeout status.
- `precedence` is not significant.
- `mutex` is weak and does not survive Holm correction.
- `groups` and cardinality-range size also correlate, but the clearest summary is still size + interaction density.

### 4.2 Fisher exact tests

#### Hard-instance enrichment

Contingency table:

- timeout rows:
  - `Hard = 57`
  - `not-Hard = 8`
- non-timeout rows:
  - `Hard = 1570`
  - `not-Hard = 1365`

Result:

- odds ratio `6.19`
- one-sided Fisher p-value `5.98e-09`

So timeout rows are significantly enriched for `Hard` instances.

#### Top-decile size/density enrichment

For `n_vars >= 90th percentile` of all SDS rows:

- threshold: `28`
- timeout vs non-timeout table:
  - `[[65, 0], [320, 2615]]`
- Fisher p-value: `7.17e-61`

For `interactions >= 90th percentile` of all SDS rows:

- threshold: `300`
- timeout vs non-timeout table:
  - `[[65, 0], [273, 2662]]`
- Fisher p-value: `6.44e-65`

Thus all `65` timeout rows fall in the top decile of both instance size and interaction count.

## 5. Seed-Level Timeout Descriptives

### seed101

- timeout rows: `22`
- `n_vars` mean `79.32`
- `interactions` mean `1674.64`
- difficulty: `18 Hard`, `4 Moderate`

### seed202

- timeout rows: `20`
- `n_vars` mean `75.05`
- `interactions` mean `1606.90`
- difficulty: `20 Hard`

### seed303

- timeout rows: `23`
- `n_vars` mean `68.96`
- `interactions` mean `1292.83`
- difficulty: `19 Hard`, `4 Moderate`

The same high-size / high-density pattern appears in each seed.

## 6. What This Supports

This analysis supports the following claims:

1. Hero timeouts are not random.
2. They are significantly concentrated on `Hard` SDS instances.
3. More specifically, they are strongly associated with very large, interaction-dense SDS instances.
4. Precedence depth does not appear to be the main explanatory factor.

## 7. What This Does Not Support

This analysis does **not** identify the exact inner-loop mechanism of timeout.

The timeout rows in `metrics_final.csv` do not contain completed selections or detailed internal solver traces, so this note cannot distinguish among:

- expensive neighborhood search on large dense instances
- repeated repair attempts
- long SA schedules
- any other specific control-flow bottleneck

So the strongest defensible conclusion is:

- timeout failures reflect computational difficulty on large, interaction-dense SDS instances,
- not a specifically isolated retry-loop pathology.

## 8. Suggested Appendix Sentence

One compact wording option:

> Across the 3,000 canonical SDS evaluation rows, timeout cases are significantly concentrated on large, interaction-dense instances: timeout rows have mean `74.3` variables vs `22.8` for non-timeout rows and mean `1518.7` interactions vs `153.8`, with Mann-Whitney `p < 1e-40` for both; all 65 timeout cases fall in the top decile of both `n_vars` and interaction count.
