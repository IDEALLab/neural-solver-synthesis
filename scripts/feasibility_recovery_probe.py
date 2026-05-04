#!/usr/bin/env python3
"""
Probe whether a W&B SDS training completion can be matched back to a cached
dataset example and re-evaluated exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from datasets import Dataset, load_dataset


def gql(token: str, query: str, variables: dict[str, str]) -> dict:
    payload = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def fetch_completion_table(token: str, entity: str, project: str, run: str) -> tuple[str, dict]:
    query = """
query ($entity: String!, $project: String!, $run: String!) {
  project(name: $project, entityName: $entity) {
    run(name: $run) {
      files(first: 500) {
        edges {
          node {
            name
            directUrl
          }
        }
      }
    }
  }
}
"""
    res = gql(token, query, {"entity": entity, "project": project, "run": run})
    files = [edge["node"] for edge in res["data"]["project"]["run"]["files"]["edges"]]
    comp_files = sorted(
        (
            f
            for f in files
            if f["name"].startswith("media/table/completions_") and f["name"].endswith(".table.json")
        ),
        key=lambda item: item["name"],
    )
    if not comp_files:
        raise RuntimeError("No completion tables found for run")

    target = comp_files[0]
    req = urllib.request.Request(target["directUrl"], headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        table = json.load(resp)
    return target["name"], table


def reconstruct_requirements(mission_raw: str) -> dict:
    from open_r1.rewards_unified_v2 import _deserialize_mission

    mission = _deserialize_mission(mission_raw)
    interactions = mission.get("interactions", {})
    weights = mission.get("weights", [])
    n_vars = mission.get("n_variables", 10)
    adj = {i: [] for i in range(n_vars)}
    for key in interactions:
        try:
            u, v = map(int, key.split(","))
        except ValueError:
            continue
        if u < n_vars:
            adj[u].append(v)
        if v < n_vars:
            adj[v].append(u)

    return {
        "requirements": {**mission, "weights": weights, "interactions": interactions},
        "catalog": {
            "variables": [
                {"id": j, "weight": weights[j] if j < len(weights) else 1.0, "neighbors": adj.get(j, [])}
                for j in range(n_vars)
            ],
            "interactions": interactions,
        },
    }


def load_cached_split(dataset_name: str, split: str, cache_root: str) -> Dataset:
    owner, repo = dataset_name.split("/", 1)
    cache_slug = f"{owner}___{repo.lower()}"
    arrow_name = f"{repo.lower()}-{split}.arrow"
    root = Path(cache_root)
    pattern = f"{cache_slug}/default/0.0.0/*/{arrow_name}"
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find cached Arrow split for {dataset_name}::{split} under {cache_root}")
    return Dataset.from_file(str(matches[0]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="smassoudi-eth-z-rich")
    parser.add_argument("--project", default="qwen-coder-sds-rl")
    parser.add_argument("--run", default="onk8fldf")
    parser.add_argument("--dataset", default="SoheylM/OpenR1-SDS-10k-seed101")
    parser.add_argument("--split", default="train")
    parser.add_argument("--token-path", default="~/llm/wandb_token.txt")
    parser.add_argument("--cache-root", default=None)
    args = parser.parse_args()

    open_r1_src = Path("/workspace/open-r1/src")
    if open_r1_src.exists():
        sys.path.insert(0, str(open_r1_src))

    token = os.environ.get("WANDB_TOKEN")
    if not token:
        token = Path(os.path.expanduser(args.token_path)).read_text().strip()
    table_name, table = fetch_completion_table(token, args.entity, args.project, args.run)

    columns = table["columns"]
    row = table["data"][0]
    record = dict(zip(columns, row))
    prompt_obj = record["prompt"]
    completion_obj = record["completion"]
    prompt_text = prompt_obj if isinstance(prompt_obj, str) else json.dumps(prompt_obj, ensure_ascii=False)
    completion_text = completion_obj if isinstance(completion_obj, str) else json.dumps(completion_obj, ensure_ascii=False)

    print(f"TABLE_FILE {table_name}")
    print(f"TABLE_STEP {record['step']}")
    print(f"PROMPT_HEAD {prompt_text[:220].replace(chr(10), ' ')}")

    cache_root = args.cache_root or os.environ.get("HF_DATASETS_CACHE") or "/workspace/hf_datasets_cache"

    try:
        ds = load_dataset(args.dataset, split=args.split)
    except Exception:
        ds = load_cached_split(args.dataset, args.split, cache_root)
    candidate_matches: list[tuple[int, dict]] = []
    for i, ex in enumerate(ds):
        if ex["problem"] in prompt_text:
            candidate_matches.append((i, ex))

    print(f"MATCH_FOUND {bool(candidate_matches)}")
    print(f"CANDIDATE_MATCHES {len(candidate_matches)}")
    if not candidate_matches:
        return 0

    match_idx, match_ex = candidate_matches[0]
    print(f"MATCH_IDX {match_idx}")
    print(f"MATCH_UUID {match_ex['uuid']}")
    print(f"MATCH_PROBLEM_HEAD {match_ex['problem'][:220].replace(chr(10), ' ')}")

    from open_r1.rewards_unified_v2 import extract_block, run_candidate, unified_nominal_reward
    from open_r1.simulators.registry import registry

    logged_nominal = float(record["unified_nominal_reward"])
    recomputed_nominal = float(
        unified_nominal_reward([completion_text], mission=[match_ex["mission"]], domain=["sds"])[0]
    )
    print(f"LOGGED_NOMINAL {logged_nominal}")
    print(f"RECOMPUTED_NOMINAL {recomputed_nominal}")
    print(f"NOMINAL_MATCH {abs(logged_nominal - recomputed_nominal) < 1e-9}")

    exact_nominal_candidates = []
    for cand_idx, cand_ex in candidate_matches:
        cand_reward = float(
            unified_nominal_reward([completion_text], mission=[cand_ex["mission"]], domain=["sds"])[0]
        )
        if abs(logged_nominal - cand_reward) < 1e-9:
            exact_nominal_candidates.append((cand_idx, cand_ex["uuid"]))
    print(f"EXACT_NOMINAL_CANDIDATES {len(exact_nominal_candidates)}")
    if exact_nominal_candidates:
        first_exact_idx, first_exact_uuid = exact_nominal_candidates[0]
        print(f"FIRST_EXACT_NOMINAL_IDX {first_exact_idx}")
        print(f"FIRST_EXACT_NOMINAL_UUID {first_exact_uuid}")

    stdin_obj = reconstruct_requirements(match_ex["mission"])
    code = extract_block(completion_text, "code")
    result = run_candidate(code, stdin_obj, timeout=2.0)
    print(f"RUN_HAS_ERROR {'error' in result}")

    if "selection" in result:
        selection = result["selection"]
        design = selection.get("variables", []) if isinstance(selection, dict) else selection
        sim = registry.simulate("sds", design, stdin_obj["requirements"])
        print(f"SELECTION_LEN {len(design)}")
        print(f"SIM_FEASIBLE {sim.get('feasible')}")
        print(f"SIM_SCORE {sim.get('score')}")
        print(f"SIM_REWARD {registry.get_reward('sds', design, stdin_obj['requirements'])}")
    else:
        print("SELECTION_LEN -1")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
