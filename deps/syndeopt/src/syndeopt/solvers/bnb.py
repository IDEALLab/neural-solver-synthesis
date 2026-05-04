from time import perf_counter
from .base import register, SolveResult
from ..core.instance import SDSInstance, Bitmask
from ..core.scoring import score
from ..core.feasibility import feasible

def optimistic_bound(inst: SDSInstance, fixed: Bitmask, idx: int, order) -> float:
    """Loose admissible UB: current score + all positive remaining contributions."""
    cur = score(inst, fixed)
    bonus = 0.0
    n = inst.n
    chosen = [i for i in range(n) if (fixed >> i) & 1]
    remaining = order[idx:]

    for i in remaining:
        if inst.w[i] > 0:
            bonus += inst.w[i]
        for j in chosen:
            a, b = (min(i, j), max(i, j))
            wij = inst.W.get((a, b), 0.0)
            if wij > 0:
                bonus += wij

    for ai in range(len(remaining)):
        for aj in range(ai + 1, len(remaining)):
            i, j = remaining[ai], remaining[aj]
            wij = inst.W.get((min(i, j), max(i, j)), 0.0)
            if wij > 0:
                bonus += wij

    return cur + bonus

@register
class BranchAndBound:
    name = "bnb"

    def solve(self, inst: SDSInstance, budget_sec: float, seed: int) -> SolveResult:
        n = inst.n
        start = perf_counter()
        deg = [len(inst.adj[i]) for i in range(n)]
        order = [i for i, _ in sorted(enumerate(deg), key=lambda t: -t[1])]

        best_s, best_x = float("-inf"), 0
        stack = [(0, 0)]
        nodes = 0

        while stack:
            if perf_counter() - start > budget_sec:
                break
            idx, x = stack.pop()
            nodes += 1

            ub = optimistic_bound(inst, x, idx, order)
            if ub <= best_s + 1e-12:
                continue

            if idx == n:
                if feasible(inst, x):
                    s = score(inst, x)
                    if s > best_s:
                        best_s, best_x = s, x
                continue

            var = order[idx]
            stack.append((idx + 1, x | (1 << var)))  # include
            stack.append((idx + 1, x))               # exclude

        return SolveResult(
            mask=best_x,
            score=best_s,
            time_sec=perf_counter() - start,
            extras={"nodes": nodes},
        )
