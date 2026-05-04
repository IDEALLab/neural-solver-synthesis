from dataclasses import dataclass
from typing import Protocol, Dict, Any, Optional, List
from ..core.instance import SDSInstance, Bitmask

@dataclass
class SolveResult:
    """Result from solving an SDS instance."""
    mask: Bitmask
    score: float
    time_sec: float
    gap: Optional[float] = None
    extras: Optional[Dict[str, Any]] = None
    trace: Optional[List[tuple]] = None

class Solver(Protocol):
    """Protocol for solvers that solve SDS instances."""
    name: str
    def solve(self, inst: SDSInstance, budget_sec: float, seed: int) -> SolveResult: ...

_REGISTRY: Dict[str, type] = {}

def register(cls):
    """Register a solver class."""
    _REGISTRY[cls.name] = cls
    return cls

def get_solver(name: str) -> Solver:
    """Get a solver instance by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown solver: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()

def list_solvers() -> Dict[str, type]:
    """List all registered solvers."""
    return dict(_REGISTRY)
