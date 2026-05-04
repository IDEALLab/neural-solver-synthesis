from time import perf_counter
from .base import register, SolveResult
from ..core.instance import SDSInstance, Bitmask, bit_count
from ..core.scoring import score
from ..core.feasibility import feasible, feasible_without_lower
from ..core.rng import make_rng

@register
class LocalSearch1Flip:
    name = "local_search"

    def __init__(self, restarts: int = 20, iters: int = 2000):
        self.restarts = restarts
        self.iters = iters

    def solve(self, inst: SDSInstance, budget_sec: float, seed: int) -> SolveResult:
        start_time = perf_counter()
        rng = make_rng(seed)
        n = inst.n
        L, U = inst.card.L, inst.card.U

        best_s, best_x = float("-inf"), 0

        # Scale number of restarts with budget (simple linear scaling)
        max_restarts = max(1, int(self.restarts * budget_sec / 2.0))

        for restart_idx in range(max_restarts):
            # Check time budget before each restart
            if perf_counter() - start_time > budget_sec:
                break
            x: Bitmask = 0
            idx = list(range(n))
            rng.shuffle(idx)
            for i in idx:
                if bit_count(x) >= U:
                    break
                if feasible_without_lower(inst, x | (1 << i)):
                    x |= (1 << i)

            while bit_count(x) < L:
                cand_i = None
                for i in range(n):
                    if not ((x >> i) & 1) and feasible_without_lower(inst, x | (1 << i)):
                        cand_i = i
                        break
                if cand_i is None:
                    break
                x |= (1 << cand_i)

            if not feasible(inst, x):
                continue

            improved = True
            steps = 0
            while improved and steps < self.iters:
                # Check time budget during local search
                if perf_counter() - start_time > budget_sec:
                    break
                improved = False
                steps += 1
                cur_s = score(inst, x)
                best_gain, best_y = 0.0, None
                for i in range(n):
                    y = x ^ (1 << i)
                    if not feasible(inst, y):
                        continue
                    g = score(inst, y) - cur_s
                    if g > best_gain:
                        best_gain, best_y = g, y
                if best_y is not None and best_gain > 1e-12:
                    x = best_y
                    improved = True

            s = score(inst, x) if feasible(inst, x) else float("-inf")
            if s > best_s:
                best_s, best_x = s, x

        elapsed = perf_counter() - start_time
        return SolveResult(mask=best_x, score=best_s, time_sec=elapsed)
