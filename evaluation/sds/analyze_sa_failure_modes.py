#!/usr/bin/env python3
"""
Analyze failure modes within the raw Base-model SA-like code pool.

This script intentionally works on the same raw `generations.jsonl` source that
feeds universal search, so its audited population can be aligned with the
191,699 unique-code result used in the manuscript.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


CODE_BLOCK_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE)


def extract_code(generated_text):
    match = CODE_BLOCK_RE.search(generated_text or "")
    if not match:
        return None
    code = match.group(1).strip()
    return code or None


def canonicalize_code(code):
    code = (code or "").replace("\r\n", "\n").replace("\r", "\n")
    code = "\n".join(line.rstrip() for line in code.split("\n")).strip() + "\n"
    return code


def strip_clean(code):
    code = re.sub(r"#.*", "", code)
    code = re.sub(r'(["\'])(?:(?=(\\?))\2.)*?\1', "", code)
    return code.lower()


def detect_algorithm(code):
    clean = strip_clean(code)
    result = {
        "is_sa": False,
        "is_greedy": False,
        "is_local_search": False,
        "is_backtracking": False,
        "is_random": False,
        "is_other": False,
    }

    has_temp = bool(re.search(r"\b(t|temperature)\s*=", clean))
    has_cooling = "cooling" in clean
    has_exp = "exp(" in clean or "math.exp" in clean
    if has_temp and has_cooling and has_exp:
        result["is_sa"] = True

    has_sort_weight = "sort" in clean and "weight" in clean
    has_greedy_kw = "greedy" in clean
    if (has_sort_weight or has_greedy_kw) and not result["is_sa"]:
        result["is_greedy"] = True

    has_neighbor = "neighbor" in clean or "neighbour" in clean
    has_moves = any(x in clean for x in ["flip", "swap", "climb", "hill"])
    if (has_neighbor or has_moves) and not result["is_sa"]:
        result["is_local_search"] = True

    has_recursion = "def" in clean and re.search(
        r"def\s+(\w+).*?\1\(", clean, re.DOTALL
    )
    if "backtrack" in clean or (has_recursion and "dfs" in clean):
        result["is_backtracking"] = True

    if (
        "random" in clean
        and not any(result.values())
        and any(k in clean for k in ["sample", "choice", "uniform"])
    ):
        result["is_random"] = True

    if not any(result.values()):
        result["is_other"] = True

    return result


def classify_sa_acceptance(code):
    clean = strip_clean(code)
    lines = clean.splitlines()

    current_pat = re.compile(
        r"(?:delta|d)\s*=\s*[^\n=]+-\s*(?:current_[a-z_]+|curr[a-z_]*|currentvalue|currentscore)|-\s*(?:current_[a-z_]+|curr[a-z_]*|currentvalue|currentscore)"
    )
    best_pat = re.compile(
        r"(?:delta|d)\s*=\s*[^\n=]+-\s*(?:best_[a-z_]+|global_best[a-z_]*|bestvalue|bestscore)|-\s*(?:best_[a-z_]+|global_best[a-z_]*|bestvalue|bestscore)"
    )

    saw_current = False
    saw_best = False

    for i, line in enumerate(lines):
        if "exp(" in line or "math.exp" in line:
            window = "\n".join(lines[max(0, i - 2) : i + 2])
            if best_pat.search(window):
                saw_best = True
            if current_pat.search(window):
                saw_current = True
            if ("best_" in line or "global_best" in line) and not (
                "if current_" in line and "best_" in line
            ):
                saw_best = True
            if "current_" in line or "curr" in line:
                saw_current = True

    if saw_best and saw_current:
        return "mixed"
    if saw_best:
        return "best_bug"
    if saw_current:
        return "current_ok"
    return "unresolved"


def has_feasibility_guard(code):
    clean = strip_clean(code)
    if not ("is_feasible" in clean or "is_valid" in clean):
        return False

    guard_patterns = [
        r"while\s+not\s+is_feasible[a-z_]*",
        r"while\s+not\s+is_valid[a-z_]*",
        r"if\s+not\s+is_feasible[a-z_]*\([^\n]*\)\s*:\s*continue",
        r"if\s+not\s+is_valid[a-z_]*\([^\n]*\)\s*:\s*continue",
        r"if\s+is_feasible[a-z_]*\(",
        r"if\s+is_valid[a-z_]*\(",
        r"elif\s+is_feasible[a-z_]*\(",
        r"elif\s+is_valid[a-z_]*\(",
    ]
    return any(re.search(pat, clean) for pat in guard_patterns)


def has_best_tracking(code):
    clean = strip_clean(code)
    init_pat = re.compile(
        r"(best_(?:selection|solution|state|variables)\s*=\s*current_(?:selection|solution|state|variables))|(best_(?:value|score)\s*=\s*current_(?:value|score))"
    )
    update_pat = re.compile(
        r"if\s+(?:current_|new_|candidate_|neighbor_)[a-z_]*?(?:value|score)\s*>\s*best_(?:value|score)"
    )
    assign_pat = re.compile(
        r"best_(?:selection|solution|state|variables)\s*=\s*(?:current_|new_|candidate_|neighbor_)[a-z_]*(?:selection|solution|state|variables)|best_(?:value|score)\s*=\s*(?:current_|new_|candidate_|neighbor_)[a-z_]*(?:value|score)"
    )
    return bool(init_pat.search(clean) and update_pat.search(clean) and assign_pat.search(clean))


def has_two_way_neighbor_logic(code):
    clean = strip_clean(code)
    has_swap_or_flip = any(tok in clean for tok in ["swap", "flip"])
    has_add = any(tok in clean for tok in ["append(", ".add(", " + [", "candidate = random.choice([i for i in range", "not in new_selection"])
    has_remove = any(tok in clean for tok in [".remove(", ".pop(", ".discard(", "remove a variable", "len(new_selection) > min_card"])
    return has_swap_or_flip or (has_add and has_remove)


def extract_budget_features(code):
    clean = strip_clean(code)
    temp_match = re.search(r"\b(?:t|temperature)\s*=\s*([0-9]+(?:\.[0-9]+)?)", clean)
    cooling_match = re.search(r"cooling(?:_rate)?\s*=\s*([0-9]+(?:\.[0-9]+)?)", clean)
    iter_match = re.search(r"for\s+_\s+in\s+range\((\d+)\)", clean)

    return {
        "temperature": float(temp_match.group(1)) if temp_match else None,
        "cooling": float(cooling_match.group(1)) if cooling_match else None,
        "iterations": int(iter_match.group(1)) if iter_match else None,
    }


def classify_remainder_bucket(code):
    acceptance = classify_sa_acceptance(code)
    guard = has_feasibility_guard(code)
    best_tracking = has_best_tracking(code)
    two_way_moves = has_two_way_neighbor_logic(code)
    budget = extract_budget_features(code)

    if acceptance == "best_bug":
        bucket = "best_bug"
    elif acceptance in {"mixed", "unresolved"}:
        bucket = "ambiguous_acceptance"
    elif not guard:
        bucket = "current_ok_no_guard"
    elif not best_tracking:
        bucket = "current_ok_no_best_tracking"
    elif not two_way_moves:
        bucket = "current_ok_guarded_but_weak_moves"
    else:
        bucket = "current_ok_structurally_complete"

    features = {
        "acceptance": acceptance,
        "has_guard": guard,
        "has_best_tracking": best_tracking,
        "has_two_way_moves": two_way_moves,
        **budget,
    }
    return bucket, features


def stream_hf_generations(org, prefix, seed, token):
    url = (
        f"https://huggingface.co/datasets/{org}/{prefix}-seed{seed}/resolve/main/"
        "generations.jsonl?download=true"
    )
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req) as response:
        for line in response:
            yield json.loads(line)


def analyze(
    *,
    org: str,
    prefix: str,
    seeds,
    hf_token_path,
    sample_per_bucket,
):
    token = Path(hf_token_path).read_text().strip()

    seen = set()
    raw_total = 0
    with_code = 0
    sa_total = 0

    bucket_counts = Counter()
    examples = defaultdict(list)
    budget_summaries = defaultdict(
        lambda: {"temperature": [], "cooling": [], "iterations": []}
    )
    per_seed = {}

    rng = random.Random(0)

    for seed in seeds:
        seed_raw = 0
        seed_with_code = 0
        seed_sa = 0
        for rec in stream_hf_generations(org, prefix, seed, token):
            seed_raw += 1
            raw_total += 1
            code = extract_code(rec.get("generated_text", ""))
            if not code:
                continue
            seed_with_code += 1
            with_code += 1
            code = canonicalize_code(code)
            code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
            if code_hash in seen:
                continue
            seen.add(code_hash)

            algo = detect_algorithm(code)
            if not algo["is_sa"]:
                continue

            seed_sa += 1
            sa_total += 1
            bucket, features = classify_remainder_bucket(code)
            bucket_counts[bucket] += 1

            for key in ["temperature", "cooling", "iterations"]:
                value = features[key]
                if value is not None:
                    budget_summaries[bucket][key].append(value)

            current_examples = examples[bucket]
            record = {
                "seed": seed,
                "uuid": rec.get("uuid"),
                "code_hash": code_hash[:16],
                "features": features,
                "code_excerpt": "\n".join(code.splitlines()[:40]),
            }
            if len(current_examples) < sample_per_bucket:
                current_examples.append(record)
            else:
                replace_idx = rng.randrange(len(current_examples) + 1)
                if replace_idx < len(current_examples):
                    current_examples[replace_idx] = record

        per_seed[seed] = {
            "raw_total": seed_raw,
            "with_code": seed_with_code,
            "sa_like_unique": seed_sa,
        }
        print(
            f"seed{seed}: raw={seed_raw} with_code={seed_with_code} sa_like_unique={seed_sa}",
            file=sys.stderr,
        )

    bucket_percent = {
        bucket: (count / sa_total * 100.0 if sa_total else 0.0)
        for bucket, count in sorted(bucket_counts.items())
    }

    budget_stats = {}
    for bucket, by_key in budget_summaries.items():
        budget_stats[bucket] = {}
        for key, values in by_key.items():
            if values:
                budget_stats[bucket][key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
            else:
                budget_stats[bucket][key] = {
                    "mean": None,
                    "min": None,
                    "max": None,
                    "count": 0,
                }

    return {
        "raw_total": raw_total,
        "with_code": with_code,
        "unique_sa_total": sa_total,
        "per_seed": per_seed,
        "bucket_counts": dict(bucket_counts),
        "bucket_percent": bucket_percent,
        "budget_stats": budget_stats,
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", default="SoheylM")
    parser.add_argument("--prefix", default="OpenR1-SDS-Base-Generations")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument(
        "--hf-token-path",
        default=os.path.expanduser("~/llm/hf_token.txt"),
    )
    parser.add_argument("--sample-per-bucket", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    result = analyze(
        org=args.org,
        prefix=args.prefix,
        seeds=args.seeds,
        hf_token_path=args.hf_token_path,
        sample_per_bucket=args.sample_per_bucket,
    )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
