#!/usr/bin/env python3
"""
Aggregate universal search results across seeds.

This script:
1. Scans universal_search_batches/{batch_id}/seed*/ directories
2. Aggregates result.json and final_topK.csv across seeds
3. Generates a summary report comparing to Base Best-of-64 performance
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_result_json(result_path: str) -> dict | None:
    """Load result.json from a seed directory."""
    try:
        with Path(result_path).open() as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load {result_path}: {e}")
        return None


def load_final_topk(csv_path: str) -> pd.DataFrame | None:
    """Load final_topK.csv from a seed directory."""
    try:
        topk_data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️  Failed to load {csv_path}: {e}")
        return None
    else:
        return topk_data


def aggregate_universal_search(batch_id: str, output_dir: str | None = None):  # noqa: PLR0915
    """
    Aggregate universal search results for a given batch.

    Args:
        batch_id: Batch ID (e.g., "20251230_struct-feas-v1")
        output_dir: Output directory (default: evaluation/sds/universal_search_batches/{batch_id}/aggregated)
    """
    base_dir = Path(f"evaluation/sds/universal_search_batches/{batch_id}")

    if not base_dir.exists():
        print(f"❌ Batch directory not found: {base_dir}")
        return

    output_dir = base_dir / "aggregated" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Aggregating universal search results from: {base_dir}")
    print(f"📁 Output directory: {output_dir}")

    # Find all seed directories
    seed_dirs = sorted(
        [d.name for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("seed")]
    )

    if not seed_dirs:
        print("❌ No seed directories found!")
        return

    print(f"📊 Found {len(seed_dirs)} seed directories: {', '.join(seed_dirs)}")

    # Load results from each seed
    all_results = []
    all_topk_dfs = []

    for seed_dir in seed_dirs:
        seed_path = base_dir / seed_dir
        seed = seed_dir.replace("seed", "")

        result_json_path = seed_path / "result.json"
        final_topk_path = seed_path / "final_topK.csv"

        result = load_result_json(str(result_json_path))
        if result:
            result["seed"] = int(seed)
            all_results.append(result)

        topk_df = load_final_topk(str(final_topk_path))
        if topk_df is not None and not topk_df.empty:
            topk_df["seed"] = int(seed)
            all_topk_dfs.append(topk_df)

    if not all_results:
        print("❌ No result.json files found!")
        return

    # Aggregate summary statistics
    summary = {
        "batch_id": batch_id,
        "n_seeds": len(all_results),
        "seeds": [r["seed"] for r in all_results],
        "total_unique_candidates": sum(r.get("n_candidates", 0) for r in all_results),
        "total_tournament_survivors": sum(
            r.get("n_tournament_survivors", 0) for r in all_results
        ),
        "total_final_evaluated": sum(
            r.get("n_final_evaluated", 0) for r in all_results
        ),
        "statuses": [r.get("status", "unknown") for r in all_results],
        "universal_winners_found": sum(
            1 for r in all_results if r.get("status") == "winner_found"
        ),
    }

    # Aggregate top-K results
    if all_topk_dfs:
        combined_topk = pd.concat(all_topk_dfs, ignore_index=True)

        # Best candidate across all seeds (by mean gap)
        best_candidate = combined_topk.sort_values("mean_gap").head(1)

        summary["best_mean_gap"] = (
            float(best_candidate["mean_gap"].iloc[0])
            if not best_candidate.empty
            else None
        )
        summary["best_max_gap"] = (
            float(best_candidate["max_gap"].iloc[0])
            if not best_candidate.empty
            else None
        )
        summary["best_feasible_rate"] = (
            float(best_candidate["feasible"].iloc[0] / best_candidate["evals"].iloc[0])
            if not best_candidate.empty
            else None
        )
        summary["best_timeout_rate"] = (
            float(best_candidate["timeouts"].iloc[0] / best_candidate["evals"].iloc[0])
            if not best_candidate.empty
            else None
        )

        # Per-seed best
        per_seed_best = (
            combined_topk.sort_values("mean_gap").groupby("seed").first().reset_index()
        )

        summary["per_seed_best_mean_gap"] = {
            int(row["seed"]): float(row["mean_gap"])
            for _, row in per_seed_best.iterrows()
        }

        # Save combined top-K
        combined_topk.to_csv(output_dir / "combined_topK.csv", index=False)
        print(f"✅ Saved combined top-K: {len(combined_topk)} candidates")

    # Save summary
    summary_path = output_dir / "aggregated_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved aggregated summary: {summary_path}")

    # Generate markdown report
    report_path = output_dir / "aggregated_report.md"
    with report_path.open("w") as f:
        f.write("# Universal Search Aggregation Report\n\n")
        f.write(f"**Batch ID**: `{batch_id}`\n\n")
        f.write("## Summary\n\n")
        f.write(
            f"- **Seeds Processed**: {summary['n_seeds']} ({', '.join(map(str, summary['seeds']))})\n"
        )
        f.write(
            f"- **Total Unique Candidates**: {summary['total_unique_candidates']:,}\n"
        )
        f.write(
            f"- **Total Tournament Survivors**: {summary['total_tournament_survivors']}\n"
        )
        f.write(f"- **Total Final Evaluated**: {summary['total_final_evaluated']}\n")
        f.write(
            f"- **Universal Winners Found**: {summary['universal_winners_found']}/{summary['n_seeds']}\n"
        )
        f.write(f"- **Statuses**: {', '.join(set(summary['statuses']))}\n\n")

        if summary.get("best_mean_gap") is not None:
            f.write("## Best Candidate Performance\n\n")
            f.write(f"- **Mean Optimality Gap**: {summary['best_mean_gap']:.2%}\n")
            f.write(f"- **Max Optimality Gap**: {summary['best_max_gap']:.2%}\n")
            f.write(f"- **Feasible Rate**: {summary['best_feasible_rate']:.2%}\n")
            f.write(f"- **Timeout Rate**: {summary['best_timeout_rate']:.2%}\n\n")

        if summary.get("per_seed_best_mean_gap"):
            f.write("## Per-Seed Best Mean Gap\n\n")
            for seed, gap in summary["per_seed_best_mean_gap"].items():
                f.write(f"- **Seed {seed}**: {gap:.2%}\n")
            f.write("\n")

        f.write("## Conclusion\n\n")
        if summary["universal_winners_found"] == 0:
            f.write("No universal winners were found across any seed. ")
            f.write(
                "This indicates that even the best single code from 64k samples cannot achieve "
            )
            f.write(
                "perfect feasibility and zero timeouts across all 1000 test instances. "
            )
            f.write(
                "This supports the argument that Best-of-64 sampling is necessary for the Base Model.\n"
            )
        else:
            f.write(
                f"Universal winners were found in {summary['universal_winners_found']} seed(s).\n"
            )

    print(f"✅ Saved markdown report: {report_path}")

    print("\n✅ Aggregation complete!")
    print(f"   Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate universal search results across seeds"
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        required=True,
        help="Batch ID (e.g., '20251230_struct-feas-v1')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: evaluation/sds/universal_search_batches/{batch_id}/aggregated)",
    )
    args = parser.parse_args()

    aggregate_universal_search(args.batch_id, args.output_dir)


if __name__ == "__main__":
    main()
