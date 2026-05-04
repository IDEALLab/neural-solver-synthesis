"""
Universal Solver Search (SDS)
=============================

Goal: Given an existing pool of generated solutions (e.g., Base Best-of-64),
search for a *single* code program that generalizes across the full SDS test set
under a strict per-instance timeout (default: 5s).

This script is designed to be:
- Batch-aware (via the caller choosing an output directory under results_batches/)
- Legacy compatible (can run directly on existing generations.jsonl files)
- Consistent with evaluation semantics in evaluation/sds/evaluate.py:
  - Uses the same stdin_obj construction
  - Uses evaluation/sds/utils.run_candidate sandbox + timeout
  - Computes feasibility via syndeopt instance checks (utils.check_constraint_violations)
  - Computes objective using the same "true score" function
  - Computes gap against per-uuid vbs_score read from an existing metrics_final.csv

Typical usage (inside repo env):
  python evaluation/sds/universal_solver_search.py \
    --seed 101 \
    --generations-jsonl /path/to/generations.jsonl \
    --metrics-csv /path/to/metrics_final.csv \
    --output-dir evaluation/sds/universal_search_batches/<BATCH_ID>/qwen2.5-coder-14b/base/seed101
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# HF datasets support (matches evaluate.py)
try:
    from datasets import load_dataset  # type: ignore[import-untyped]

    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

# Local SDS utilities (sandboxed execution + instance conversion)
try:
    from utils import (  # type: ignore[import-untyped]
        check_constraint_violations,
        deserialize_mission,
        mission_to_instance,
        run_candidate,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from utils import (  # type: ignore[import-untyped]
        check_constraint_violations,
        deserialize_mission,
        mission_to_instance,
        run_candidate,
    )


CODE_BLOCK_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE)

# Magic value constants
_EPSILON_SMALL = 1e-6
_PERCENT_THRESHOLD = 1.01
_TIMEOUT_TOLERANCE = 1e-6


# --- CUSTOM EXCEPTIONS ---
class DatasetsNotInstalledError(RuntimeError):
    """Raised when datasets library is not installed."""

    def __init__(self):
        msg = "datasets is not installed; cannot load HF test split."
        super().__init__(msg)


class MetricsCSVError(ValueError):
    """Raised when metrics CSV is missing required columns."""

    def __init__(self, csv_path: str, message: str):
        msg = f"{message}: {csv_path}"
        super().__init__(msg)
        self.csv_path = csv_path


class MetricsCSVNotFoundError(FileNotFoundError):
    """Raised when metrics CSV cannot be found or inferred."""

    def __init__(self):
        msg = "metrics_final.csv not provided and could not be inferred next to generations.jsonl"
        super().__init__(msg)


def extract_code(generated_text: str) -> str | None:
    m = CODE_BLOCK_RE.search(generated_text or "")
    if not m:
        return None
    code = m.group(1).strip()
    return code or None


def canonicalize_code(code: str) -> str:
    # Keep it simple and stable: normalize line endings + strip trailing whitespace.
    # (We intentionally do NOT attempt AST normalization; generated code may be invalid.)
    code = (code or "").replace("\r\n", "\n").replace("\r", "\n")
    code = "\n".join([ln.rstrip() for ln in code.split("\n")]).strip() + "\n"
    return code


def hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def calculate_true_score(inst, selected_ids: list[int]) -> float:
    # Matches evaluation/sds/evaluate.py semantics (preserve negative scores).
    if not selected_ids:
        return 0.0
    s = 0.0
    for i in selected_ids:
        with contextlib.suppress(Exception):
            # Out of bounds selections are infeasible anyway; treat as 0 contribution.
            s += inst.w[i]
    sel_set = set(selected_ids)
    for (i, j), weight in inst.W.items():
        if i in sel_set and j in sel_set:
            s += weight
    return float(s)


def build_stdin_obj(mission_dict: dict) -> dict:
    """
    Reconstruct stdin payload exactly like evaluation/sds/evaluate.py does.
    """
    test_reqs = {
        "n_variables": mission_dict.get("n_variables", 10),
        "cardinality_bounds": mission_dict.get("cardinality_bounds", [2, 8]),
        "precedence": mission_dict.get("precedence", []),
        "mutex": mission_dict.get("mutex", []),
        "groups": mission_dict.get("groups", {}),
    }
    interactions = mission_dict.get("interactions", {})
    weights = mission_dict.get("weights", [1.0] * test_reqs["n_variables"])
    adj = {i: [] for i in range(test_reqs["n_variables"])}
    for k in interactions:
        try:
            u, v = map(int, k.split(","))
            if u < test_reqs["n_variables"] and v < test_reqs["n_variables"]:
                adj[u].append(v)
                adj[v].append(u)
        except Exception:
            pass
    return {
        "requirements": {**test_reqs, "weights": weights, "interactions": interactions},
        "catalog": {
            "variables": [
                {"id": j, "weight": weights[j], "neighbors": adj.get(j, [])}
                for j in range(test_reqs["n_variables"])
            ],
            "interactions": interactions,
        },
    }


@dataclass(frozen=True)
class Mission:
    uuid: str
    mission_dict: dict
    vbs_score: float


def load_test_missions(seed: int) -> list[Mission]:
    if not HAS_DATASETS:
        raise DatasetsNotInstalledError()
    ds_name = f"SoheylM/OpenR1-SDS-10k-seed{seed}"
    ds = load_dataset(ds_name, split="test")  # type: ignore[assignment]
    missions: list[Mission] = []
    for row in ds:
        uuid = row.get("uuid")
        mission_dict = deserialize_mission(row.get("mission"))
        missions.append(
            Mission(uuid=uuid, mission_dict=mission_dict, vbs_score=float("nan"))
        )
    return missions


def load_vbs_scores(metrics_csv: str) -> dict[str, float]:
    df = pd.read_csv(metrics_csv)
    if "uuid" not in df.columns:
        raise MetricsCSVError(metrics_csv, "metrics CSV missing 'uuid' column")
    # Prefer explicit vbs_score if present; otherwise fall back to best_known_score.
    if "vbs_score" in df.columns:
        col = "vbs_score"
    elif "best_known_score" in df.columns:
        col = "best_known_score"
    else:
        raise MetricsCSVError(
            metrics_csv, "metrics CSV missing 'vbs_score' or 'best_known_score'"
        )
    out = {}
    for _, r in df.iterrows():
        out[str(r["uuid"])] = float(r[col])
    return out


def collect_candidate_codes(generation_files: list[str]) -> dict[str, str]:
    """
    Returns dict: code_hash -> canonical_code
    """
    candidates: dict[str, str] = {}
    total = 0
    skipped_no_code = 0
    for fp in generation_files:
        with Path(fp).open() as f:
            for line in f:
                total += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                code = extract_code(rec.get("generated_text", ""))
                if not code:
                    skipped_no_code += 1
                    continue
                code = canonicalize_code(code)
                h = hash_code(code)
                if h not in candidates:
                    candidates[h] = code
    return candidates


def eval_single(code_hash: str, code: str, mission: Mission, timeout_s: float) -> dict:
    """
    Evaluate a single candidate code on a single mission.
    Hard constraint: must finish <= timeout_s and be feasible.
    """
    inst = mission_to_instance(mission.mission_dict)
    stdin_obj = build_stdin_obj(mission.mission_dict)

    t0 = time.time()
    res = run_candidate(code, stdin_obj, timeout=timeout_s)
    exec_time = float(res.get("execution_time", time.time() - t0))
    err_type = res.get("error_type", "unknown") if "error" in res else "none"

    feasible = False
    llm_score = 0.0
    if "selection" in res:
        try:
            sel = res["selection"].get("variables", [])
        except Exception:
            sel = []
        violations = check_constraint_violations(inst, sel)
        if violations.get("all_valid", False):
            feasible = True
            llm_score = calculate_true_score(inst, sel)
        else:
            err_type = "constraint"
            feasible = False

    vbs = float(mission.vbs_score)
    gap = None
    if vbs > _EPSILON_SMALL:
        safe_score = llm_score if feasible else 0.0
        gap = float((vbs - max(0.0, safe_score)) / vbs)
        if gap < 0:
            gap = 0.0

    timed_out = (err_type == "timeout") or (exec_time > timeout_s + _TIMEOUT_TOLERANCE)
    return {
        "code_hash": code_hash,
        "uuid": mission.uuid,
        "feasible": bool(feasible),
        "timeout": bool(timed_out),
        "error_type": err_type,
        "execution_time": exec_time,
        "llm_score": float(llm_score),
        "vbs_score": vbs,
        "gap": gap,
    }


def pick_tournament_uuids(
    base_metrics_df: pd.DataFrame,
    n_missions: int,
    rng: np.random.RandomState,
) -> list[str]:
    """
    Choose a discriminative subset of missions using the base run's collapsed metrics.
    Heuristic: mix of "hard" (high gap) and random coverage.
    """
    if "uuid" not in base_metrics_df.columns:
        raise MetricsCSVError("", "metrics_final.csv missing uuid column")
    # Prefer Gap column if present; else approximate from vbs_score and llm_score.
    if "Gap" in base_metrics_df.columns:
        gap_col = "Gap"
        gaps = base_metrics_df[gap_col].astype(float)
        # Gap might be percent in some tables; treat >1 as percent.
        if gaps.max() > _PERCENT_THRESHOLD:
            gaps = gaps / 100.0
    elif (
        ("vbs_score" in base_metrics_df.columns)
        and ("llm_score" in base_metrics_df.columns)
        and ("feasible" in base_metrics_df.columns)
    ):
        v = base_metrics_df["vbs_score"].astype(float)
        s = base_metrics_df.apply(
            lambda r: float(r["llm_score"]) if bool(r["feasible"]) else 0.0, axis=1
        )
        gaps = (v - s.clip(lower=0.0)) / v.replace(0.0, np.nan)
        gaps = gaps.fillna(0.0)
    else:
        uuids = base_metrics_df["uuid"].astype(str).tolist()
        rng.shuffle(uuids)
        return uuids[:n_missions]

    tournament_metrics_df = base_metrics_df.copy()
    tournament_metrics_df["_gap"] = gaps
    tournament_metrics_df["_uuid"] = tournament_metrics_df["uuid"].astype(str)

    hard_n = max(1, int(0.6 * n_missions))
    rand_n = n_missions - hard_n

    hard = (
        tournament_metrics_df.sort_values("_gap", ascending=False)
        .head(min(hard_n, len(tournament_metrics_df)))["_uuid"]
        .tolist()
    )
    remaining = tournament_metrics_df[~tournament_metrics_df["_uuid"].isin(hard)][
        "_uuid"
    ].tolist()
    rng.shuffle(remaining)
    rand = remaining[: min(rand_n, len(remaining))]
    picked = hard + rand
    # Ensure exact length
    return picked[:n_missions]


def evaluate_codes_on_missions(  # noqa: PLR0913
    candidates: dict[str, str],
    missions_by_uuid: dict[str, Mission],
    uuids: list[str],
    timeout_s: float,
    workers: int,
    max_evals: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate candidate codes on the specified missions.
    """
    rows: list[dict] = []
    submitted = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for code_hash, code in candidates.items():
            for uuid in uuids:
                if uuid not in missions_by_uuid:
                    continue
                if max_evals is not None and submitted >= max_evals:
                    break
                futures.append(
                    ex.submit(
                        eval_single, code_hash, code, missions_by_uuid[uuid], timeout_s
                    )
                )
                submitted += 1
            if max_evals is not None and submitted >= max_evals:
                break

        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                # Record as generic runtime failure; keep tournament progressing.
                rows.append(
                    {
                        "code_hash": "unknown",
                        "uuid": "unknown",
                        "feasible": False,
                        "timeout": False,
                        "error_type": f"worker:{e}",
                        "execution_time": 0.0,
                        "llm_score": 0.0,
                        "vbs_score": 0.0,
                        "gap": None,
                    }
                )

    return pd.DataFrame(rows)


def summarize_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize per-candidate performance on the evaluated missions.
    Strict rule: any timeout or infeasible counts as failure for "universal" selection.
    """
    if df.empty:
        return pd.DataFrame([])
    grouped = df.groupby("code_hash")
    out_rows = []
    for h, g in grouped:
        n = len(g)
        n_timeout = int(g["timeout"].sum())
        n_feas = int(g["feasible"].sum())
        n_ok = int(
            ((g["feasible"]) & (~g["timeout"]) & (g["error_type"] == "none")).sum()
        )
        # Universal criterion: must be feasible and within timeout on all missions evaluated.
        universal_ok = (n_timeout == 0) and (n_feas == n)
        gaps = g["gap"].dropna().astype(float)
        mean_gap = float(gaps.mean()) if len(gaps) else float("inf")
        max_gap = float(gaps.max()) if len(gaps) else float("inf")
        out_rows.append(
            {
                "code_hash": h,
                "evals": n,
                "feasible": n_feas,
                "timeouts": n_timeout,
                "ok": n_ok,
                "universal_ok": universal_ok,
                "mean_gap": mean_gap,
                "max_gap": max_gap,
                "mean_time": float(g["execution_time"].mean()) if n else float("inf"),
            }
        )
    return pd.DataFrame(out_rows).sort_values(
        ["universal_ok", "mean_gap"], ascending=[False, True]
    )


def main():  # noqa: PLR0912, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=[101, 202, 303])
    ap.add_argument(
        "--generations-jsonl",
        type=str,
        required=True,
        action="append",
        help="Path to generations.jsonl (can be repeated).",
    )
    ap.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Path to metrics_final.csv from the same run (used for vbs_score per uuid). If omitted, will try to infer from generations.jsonl directory.",
    )
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--tournament-missions", type=int, default=30)
    ap.add_argument("--survivors", type=int, default=10)
    ap.add_argument(
        "--keep-per-round",
        type=int,
        default=500,
        help="Upper bound on survivors carried between rounds (after filtering timeouts/infeasible).",
    )
    ap.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on number of unique codes considered (for debugging).",
    )
    ap.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="Seed for deterministic selection of tournament missions (defaults to --seed).",
    )

    args = ap.parse_args()

    # Validate generations_jsonl list
    if not args.generations_jsonl:
        ap.error("--generations-jsonl must be provided at least once")
    # Filter out empty strings (can happen if glob expansion fails)
    args.generations_jsonl = [f for f in args.generations_jsonl if f and f.strip()]
    if not args.generations_jsonl:
        ap.error(
            "--generations-jsonl arguments are empty (glob expansion may have failed)"
        )
    # Validate all files exist
    for fp in args.generations_jsonl:
        if not Path(fp).exists():
            ap.error(f"generations.jsonl file does not exist: {fp}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load test missions
    missions = load_test_missions(args.seed)

    # Load VBS scores from metrics CSV
    metrics_csv = args.metrics_csv
    if metrics_csv is None:
        # Infer from the first generations file directory
        base_dir = Path(args.generations_jsonl[0]).parent
        cand = base_dir / "metrics_final.csv"
        if cand.exists():
            metrics_csv = str(cand)
        else:
            raise MetricsCSVNotFoundError()

    vbs_by_uuid = load_vbs_scores(metrics_csv)
    missions_by_uuid: dict[str, Mission] = {}
    missing_vbs = 0
    for m in missions:
        v = vbs_by_uuid.get(m.uuid, float("nan"))
        if not (v > -float("inf")) or np.isnan(v):
            missing_vbs += 1
        missions_by_uuid[m.uuid] = Mission(
            uuid=m.uuid,
            mission_dict=m.mission_dict,
            vbs_score=float(v) if not np.isnan(v) else 0.0,
        )

    # Candidate codes
    candidates = collect_candidate_codes(args.generations_jsonl)
    if args.max_candidates is not None and args.max_candidates > 0:
        # Deterministic truncation by sorted hash
        keys = sorted(candidates.keys())[: args.max_candidates]
        candidates = {k: candidates[k] for k in keys}

    # Save candidate pool for reproducibility
    output_path = Path(args.output_dir)
    with (output_path / "candidates_dedup.json").open("w") as f:
        json.dump(
            {
                "seed": args.seed,
                "n_candidates": len(candidates),
                "candidates": candidates,
            },
            f,
        )

    # Tournament mission selection based on base metrics
    base_df = pd.read_csv(metrics_csv)
    rng = np.random.RandomState(
        args.rng_seed if args.rng_seed is not None else args.seed
    )
    tour_uuids = pick_tournament_uuids(base_df, args.tournament_missions, rng)

    config = {
        "seed": args.seed,
        "rng_seed": args.rng_seed if args.rng_seed is not None else args.seed,
        "timeout": args.timeout,
        "workers": args.workers,
        "tournament_missions": args.tournament_missions,
        "tournament_uuids": tour_uuids,
        "survivors": args.survivors,
        "keep_per_round": args.keep_per_round,
        "metrics_csv": metrics_csv,
        "generations_jsonl": args.generations_jsonl,
        "n_candidates": len(candidates),
    }
    with (output_path / "tournament_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    # Round 1: evaluate all candidates on tournament missions
    df_eval = evaluate_codes_on_missions(
        candidates=candidates,
        missions_by_uuid=missions_by_uuid,
        uuids=tour_uuids,
        timeout_s=args.timeout,
        workers=args.workers,
    )
    df_eval.to_csv(output_path / "tournament_evals.csv", index=False)
    summary = summarize_candidates(df_eval)
    summary.to_csv(output_path / "tournament_summary.csv", index=False)

    # Filter: must be feasible + not timeout on all tournament missions
    strict = summary[summary["universal_ok"]].copy()
    strict = strict.sort_values(["mean_gap", "max_gap"], ascending=[True, True])
    strict = strict.head(args.keep_per_round)

    if strict.empty:
        # No universal candidate found even on the tournament subset.
        with (output_path / "result.json").open("w") as f:
            json.dump(
                {
                    "status": "no_survivors",
                    "seed": args.seed,
                    "n_candidates": len(candidates),
                    "tournament_missions": len(tour_uuids),
                },
                f,
                indent=2,
            )
        return

    # Final verification on all 1000 missions for top S survivors
    top_hashes = strict["code_hash"].tolist()[: args.survivors]
    survivors = {h: candidates[h] for h in top_hashes if h in candidates}
    all_uuids = [m.uuid for m in missions]

    df_final = evaluate_codes_on_missions(
        candidates=survivors,
        missions_by_uuid=missions_by_uuid,
        uuids=all_uuids,
        timeout_s=args.timeout,
        workers=args.workers,
    )
    df_final.to_csv(output_path / "final_evals.csv", index=False)
    final_summary = summarize_candidates(df_final)
    final_summary.to_csv(output_path / "final_topK.csv", index=False)

    winner = (
        final_summary[final_summary["universal_ok"]]
        .sort_values(["mean_gap", "max_gap"], ascending=[True, True])
        .head(1)
    )
    if winner.empty:
        status = "no_universal_winner"
        winner_hash = None
    else:
        status = "winner_found"
        winner_hash = str(winner.iloc[0]["code_hash"])
        with (output_path / "winner_code.py").open("w") as f:
            f.write(candidates[winner_hash])

    with (output_path / "result.json").open("w") as f:
        json.dump(
            {
                "status": status,
                "seed": args.seed,
                "n_candidates": len(candidates),
                "n_tournament_survivors": len(strict),
                "n_final_evaluated": len(survivors),
                "winner_hash": winner_hash,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
