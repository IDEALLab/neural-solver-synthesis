from typing import Optional, Dict, Tuple, List
from ..core.instance import SDSInstance, CardBounds
from ..core.rng import make_rng

def make_tree_instance(
    n: int = 14,
    card=(4, 10),
    weight_scale: float = 10.0,
    pair_scale: float = 6.0,
    seed: Optional[int] = None,
) -> SDSInstance:
    rng = make_rng(seed)
    assert n >= 2
    # Prüfer sequence tree
    prufer = [rng.randrange(n) for _ in range(n - 2)]
    degree = [1] * n
    for v in prufer:
        degree[v] += 1
    leaves = sorted(i for i in range(n) if degree[i] == 1)
    edges: List[Tuple[int, int]] = []
    for v in prufer:
        u = leaves[0]
        leaves.pop(0)
        edges.append((min(u, v), max(u, v)))
        degree[u] -= 1
        degree[v] -= 1
        if degree[v] == 1:
            idx = 0
            while idx < len(leaves) and leaves[idx] < v:
                idx += 1
            leaves.insert(idx, v)
    u, v = leaves
    edges.append((min(u, v), max(u, v)))

    w = [rng.uniform(-0.5, 1.0) * weight_scale for _ in range(n)]
    W: Dict[Tuple[int, int], float] = {e: rng.uniform(0.5, 1.0)*pair_scale for e in edges}

    # small constraint sets
    mutex = rng.sample(edges, k=min(2, len(edges)))
    prec_edges = rng.sample(edges, k=min(2, len(edges)))
    precedence = [(i, j) if rng.random() < 0.5 else (j, i) for (i, j) in prec_edges]

    groups = {}
    return SDSInstance(
        n=n,
        w=w,
        W=W,
        precedence=precedence,
        mutex=mutex,
        groups=groups,
        card=CardBounds(*card),
    )
