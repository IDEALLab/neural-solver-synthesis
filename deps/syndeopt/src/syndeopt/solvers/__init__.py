# import modules so they register themselves
from . import (
    bnb,  # noqa: F401
    cpsat,  # noqa: F401
    greedy,  # noqa: F401
    local_search,  # noqa: F401
)
from .base import SolveResult, get_solver, list_solvers

__all__ = ["SolveResult", "get_solver", "list_solvers"]
