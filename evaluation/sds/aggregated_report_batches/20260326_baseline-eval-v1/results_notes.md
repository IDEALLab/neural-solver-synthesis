# Baseline Evaluation Notes

- Quality summary (`baseline_summary.csv`) is aggregated across all three seeds: 101, 202, and 303. Per-seed values are in `baseline_summary_by_seed.csv`.
- Frozen Hero selection rule: filter canonical Hero `metrics_final.csv` rows to `feasible == True` and `error_type == "none"`, sort by `uuid`, and take the first row.
- Failure semantics: a non-pass means the method did not return a feasible solution within the fixed evaluation budget. It should not be described as a reviewer-confusing "violation" unless the returned program actually violated constraints.
- Frozen Hero mean pass rate: 97.9%.
- Frozen Hero mean optimality gap: 4.34%.
- Hand-written SA mean pass rate: 95.6%.
- Hand-written SA mean optimality gap: 5.32%.
- Runtime audit across all available seeds is in `runtime_summary_by_seed.csv` and `runtime_summary_available_seeds.csv`.
- Runtime summary uses fresh seed101 timing reruns for Ours Hero, Base Best-of-64, ShinkaEvolve, Frozen Hero, and Hand-written SA.
