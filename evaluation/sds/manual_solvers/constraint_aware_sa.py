import json
import math
import random
import sys
import time


def parse_problem():
    payload = json.load(sys.stdin)
    req = payload["requirements"]
    n = req["n_variables"]
    weights = req["weights"]
    interactions = {
        tuple(map(int, key.split(","))): value
        for key, value in req.get("interactions", {}).items()
    }
    groups = {
        int(group_id): members for group_id, members in req.get("groups", {}).items()
    }
    return {
        "n": n,
        "weights": weights,
        "interactions": interactions,
        "cardinality": tuple(req["cardinality_bounds"]),
        "precedence": [tuple(edge) for edge in req.get("precedence", [])],
        "mutex": [tuple(edge) for edge in req.get("mutex", [])],
        "groups": groups,
    }


def build_helpers(problem):
    pred = {i: set() for i in range(problem["n"])}
    succ = {i: set() for i in range(problem["n"])}
    group_of = {}
    mutex = {i: set() for i in range(problem["n"])}
    adj = {i: {} for i in range(problem["n"])}

    for i, j in problem["precedence"]:
        pred[j].add(i)
        succ[i].add(j)
    for a, b in problem["mutex"]:
        mutex[a].add(b)
        mutex[b].add(a)
    for group_id, members in problem["groups"].items():
        for member in members:
            group_of[member] = group_id
    for (i, j), value in problem["interactions"].items():
        adj[i][j] = value
        adj[j][i] = value

    return pred, succ, group_of, mutex, adj


def closure(selection, pred):
    updated = set(selection)
    changed = True
    while changed:
        changed = False
        for item in list(updated):
            missing = pred[item] - updated
            if missing:
                updated.update(missing)
                changed = True
    return updated


def is_feasible(selection, problem, pred, group_of, mutex):
    low, high = problem["cardinality"]
    if not (low <= len(selection) <= high):
        return False
    for item in selection:
        if not pred[item].issubset(selection):
            return False
        if mutex[item] & selection:
            return False
    seen_groups = set()
    for item in selection:
        group_id = group_of.get(item)
        if group_id is None:
            continue
        if group_id in seen_groups:
            return False
        seen_groups.add(group_id)
    return True


def score(selection, problem):
    total = sum(problem["weights"][item] for item in selection)
    for (i, j), value in problem["interactions"].items():
        if i in selection and j in selection:
            total += value
    return total


def marginal_gain(item, selection, problem, adj):
    gain = problem["weights"][item]
    for other in selection:
        gain += adj[item].get(other, 0.0)
    return gain


def can_add(item, selection, high, pred, group_of, mutex):
    expanded = closure(selection | {item}, pred)
    if len(expanded) > high:
        return False
    for member in expanded:
        if mutex[member] & expanded:
            return False
        group_id = group_of.get(member)
        if group_id is None:
            continue
        if sum(1 for node in expanded if group_of.get(node) == group_id) > 1:
            return False
    return True


def greedy_initialize(problem, pred, group_of, mutex, adj):
    low, high = problem["cardinality"]
    selection = set()
    candidates = sorted(
        range(problem["n"]),
        key=lambda item: (
            problem["weights"][item] + sum(max(0.0, v) for v in adj[item].values())
        ),
        reverse=True,
    )

    for item in candidates:
        if can_add(item, selection, high, pred, group_of, mutex):
            trial = closure(selection | {item}, pred)
            if len(trial) <= high:
                selection = trial

    if len(selection) < low:
        for item in candidates:
            if item in selection:
                continue
            if can_add(item, selection, high, pred, group_of, mutex):
                trial = closure(selection | {item}, pred)
                if len(trial) <= high:
                    selection = trial
                if len(selection) >= low:
                    break

    if len(selection) > high:
        trimmed = sorted(
            selection,
            key=lambda item: marginal_gain(item, selection - {item}, problem, adj),
        )
        for item in trimmed:
            if len(selection) <= high:
                break
            trial = selection - {item}
            if is_feasible(trial, problem, pred, group_of, mutex):
                selection = trial

    return selection


def propose_neighbor(selection, problem, pred, succ, group_of, mutex, adj, rng):
    low, high = problem["cardinality"]
    candidates = list(range(problem["n"]))
    move = rng.choice(["swap", "add", "remove"])

    if move == "add":
        rng.shuffle(candidates)
        for item in candidates:
            if item in selection:
                continue
            if can_add(item, selection, high, pred, group_of, mutex):
                trial = closure(selection | {item}, pred)
                if is_feasible(trial, problem, pred, group_of, mutex):
                    return trial

    if move == "remove" and len(selection) > low:
        removable = [item for item in selection if not (succ[item] & selection)]
        rng.shuffle(removable)
        for item in removable:
            trial = selection - {item}
            if is_feasible(trial, problem, pred, group_of, mutex):
                return trial

    if selection:
        removable = [item for item in selection if not (succ[item] & selection)]
        addable = [item for item in candidates if item not in selection]
        rng.shuffle(removable)
        rng.shuffle(addable)
        for remove_item in removable:
            partial = selection - {remove_item}
            for add_item in addable:
                if can_add(add_item, partial, high, pred, group_of, mutex):
                    trial = closure(partial | {add_item}, pred)
                    if is_feasible(trial, problem, pred, group_of, mutex):
                        return trial

    return selection


def solve():
    problem = parse_problem()
    pred, succ, group_of, mutex, adj = build_helpers(problem)
    rng_seed = int(sum(problem["weights"]) * 1000) + problem["n"] * 17
    rng = random.Random(rng_seed)

    current = greedy_initialize(problem, pred, group_of, mutex, adj)
    if not is_feasible(current, problem, pred, group_of, mutex):
        current = set()
    best = set(current)
    current_score = score(current, problem) if current else float("-inf")
    best_score = current_score

    base_temp = max(
        1.0, sum(abs(value) for value in problem["weights"]) / max(1, problem["n"])
    )
    time_budget = 5.0
    end_time = time.time() + time_budget
    step = 0
    max_steps = max(5000, problem["n"] * 2000)

    while time.time() < end_time and step < max_steps:
        candidate = propose_neighbor(
            current, problem, pred, succ, group_of, mutex, adj, rng
        )
        candidate_score = score(candidate, problem) if candidate else float("-inf")
        delta = candidate_score - current_score
        temperature = max(0.05, base_temp * (0.998 ** step))
        accept = delta >= 0 or rng.random() < math.exp(delta / temperature)
        if accept:
            current = candidate
            current_score = candidate_score
            if candidate_score > best_score:
                best = set(candidate)
                best_score = candidate_score
        step += 1

    print(json.dumps({"selection": {"variables": sorted(best)}}))


if __name__ == "__main__":
    solve()
