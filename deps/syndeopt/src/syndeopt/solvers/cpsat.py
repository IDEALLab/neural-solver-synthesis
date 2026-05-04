from .base import register, SolveResult
from ..core.instance import SDSInstance, Bitmask

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

@register
class CPSATSolver:
    name = "cpsat"

    def solve(self, inst: SDSInstance, budget_sec: float, seed: int) -> SolveResult:
        if not HAS_ORTOOLS:
            raise RuntimeError("ortools is not installed; run `pip install ortools`.")

        m = cp_model.CpModel()
        n = inst.n
        x = [m.NewBoolVar(f"x_{i}") for i in range(n)]
        y = {}

        for (i, j), _ in inst.W.items():
            y[(i, j)] = m.NewBoolVar(f"y_{i}_{j}")
            m.Add(y[(i, j)] <= x[i])
            m.Add(y[(i, j)] <= x[j])
            m.Add(y[(i, j)] >= x[i] + x[j] - 1)

        L, U = inst.card.L, inst.card.U
        m.Add(sum(x) >= L)
        m.Add(sum(x) <= U)

        for i, j in inst.precedence:
            m.Add(x[j] <= x[i])

        for a, b in inst.mutex:
            m.Add(x[a] + x[b] <= 1)

        for _, members in inst.groups.items():
            m.Add(sum(x[i] for i in members) <= 1)

        obj = sum(inst.w[i] * x[i] for i in range(n))
        for (i, j), wij in inst.W.items():
            obj += wij * y[(i, j)]
        m.Maximize(obj)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = budget_sec
        solver.parameters.random_seed = seed
        solver.parameters.num_search_workers = 8

        class TraceCB(cp_model.CpSolverSolutionCallback):
            def __init__(self):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self.trace = []

            def OnSolutionCallback(self):
                self.trace.append((self.WallTime(), self.ObjectiveValue()))

        cb = TraceCB()
        status = solver.Solve(m, cb)

        mask: Bitmask = 0
        for i in range(n):
            if solver.BooleanValue(x[i]):
                mask |= (1 << i)

        score_val = solver.ObjectiveValue()
        try:
            bound = solver.BestObjectiveBound()
            gap = bound - score_val
        except Exception:
            gap = None

        return SolveResult(
            mask=mask,
            score=score_val,
            time_sec=solver.WallTime(),
            gap=gap,
            extras={"status": int(status)},
            trace=cb.trace,
        )
