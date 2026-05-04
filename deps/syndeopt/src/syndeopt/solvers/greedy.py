from time import perf_counter
from .base import register, SolveResult
from ..core.instance import SDSInstance, Bitmask, bit_count
from ..core.scoring import score
from ..core.feasibility import feasible, feasible_without_lower

@register
class GreedyMarginal:
    name = "greedy"

    def solve(self, inst: SDSInstance, budget_sec: float, seed: int) -> SolveResult:
        start_time = perf_counter()
        n = inst.n
        L, U = inst.card.L, inst.card.U
        x: Bitmask = 0
        chosen = set()
        remaining = set(range(n))

        def gain_if_add(i: int) -> float:
            g = inst.w[i]
            for j in chosen:
                a, b = (min(i, j), max(i, j))
                if (a, b) in inst.W:
                    g += inst.W[(a, b)]
            return g

        while bit_count(x) < U:
            best_i, best_gain = None, float("-inf")
            for i in list(remaining):
                cand = x | (1 << i)
                if not feasible_without_lower(inst, cand):
                    continue
                g = gain_if_add(i)
                if g > best_gain:
                    best_gain, best_i = g, i
            if best_i is None or best_gain <= 0:
                break
            x |= (1 << best_i)
            chosen.add(best_i)
            remaining.remove(best_i)

        while bit_count(x) < L:
            best_i, best_gain = None, float("-inf")
            for i in list(remaining):
                cand = x | (1 << i)
                if not feasible_without_lower(inst, cand):
                    continue
                g = gain_if_add(i)
                if g > best_gain:
                    best_gain, best_i = g, i
            if best_i is None:
                break
            x |= (1 << best_i)
            chosen.add(best_i)
            remaining.remove(best_i)

        s = score(inst, x) if feasible(inst, x) else float("-inf")
        elapsed = perf_counter() - start_time
        return SolveResult(mask=x, score=s, time_sec=elapsed, extras={"selected": list(chosen)})
