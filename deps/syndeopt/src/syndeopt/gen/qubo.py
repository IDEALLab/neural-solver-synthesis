from __future__ import annotations

from typing import Optional, Dict, Tuple, List
import itertools

from ..core.instance import SDSInstance, CardBounds
from ..core.rng import make_rng

Bitmask = int


def _qubo_to_sds(
    Q: List[List[float]],
    card: Tuple[int, int],
) -> SDSInstance:
    """
    Convert a QUBO matrix Q (size n x n) into an SDSInstance.

    QUBO objective:   x^T Q x   with x in {0,1}^n
    SDS objective:    sum_i w_i x_i + sum_{i<j} W_ij x_i x_j

    We map:
      w_i   = Q[i][i]
      W_ij  = Q[i][j] + Q[j][i]  (for i<j), assuming Q symmetric in practice
    """
    n = len(Q)
    w = [Q[i][i] for i in range(n)]
    W: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            wij = Q[i][j] + Q[j][i]
            if abs(wij) > 0.0:
                W[(i, j)] = wij

    precedence: List[Tuple[int, int]] = []
    mutex: List[Tuple[int, int]] = []
    groups: Dict[int, List[int]] = {}

    return SDSInstance(
        n=n,
        w=w,
        W=W,
        precedence=precedence,
        mutex=mutex,
        groups=groups,
        card=CardBounds(*card),
    )


# ---------------------------------------------------------------------------
# 1) Random dense QUBO
# ---------------------------------------------------------------------------

def make_random_qubo_instance(
    n: int = 20,
    card: Tuple[int, int] = (0, 20),
    diag_scale: float = 1.0,
    offdiag_scale: float = 1.0,
    density: float = 0.5,
    seed: Optional[int] = None,
) -> SDSInstance:
    """
    Random dense/sparse QUBO instance:

      x^T Q x with x in {0,1}^n

    - diag entries sampled in [-diag_scale, diag_scale]
    - off-diagonal entries sampled with probability 'density'
      from [-offdiag_scale, offdiag_scale]
    - No additional constraints: precedence, mutex, groups are empty.

    This is a pure QUBO; SDS just wraps it.
    """
    rng = make_rng(seed)
    # build symmetric Q
    Q = [[0.0 for _ in range(n)] for _ in range(n)]
    # diagonal
    for i in range(n):
        Q[i][i] = rng.uniform(-diag_scale, diag_scale)
    # off-diagonal
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                val = rng.uniform(-offdiag_scale, offdiag_scale)
                Q[i][j] = val
                Q[j][i] = val

    return _qubo_to_sds(Q, card=card)


# ---------------------------------------------------------------------------
# 2) Planted-solution QUBO
# ---------------------------------------------------------------------------

def make_planted_qubo_instance(
    n: int = 20,
    card: Tuple[int, int] = (0, 20),
    density: float = 0.5,
    signal_strength: float = 2.0,
    noise_scale: float = 0.5,
    seed: Optional[int] = None,
) -> SDSInstance:
    """
    QUBO with a planted high-value solution:

      - Pick a random planted bitstring x* in {0,1}^n.
      - Construct Q so that x* is a strong local (often global) optimum:
          * Reward agreeing with x* (attraction within the same bit pattern).
          * Add random noise on top.

    Useful to test whether solvers can recover a planted pattern.
    """
    rng = make_rng(seed)
    # random planted solution
    x_star = [rng.randint(0, 1) for _ in range(n)]

    # base Q = 0
    Q = [[0.0 for _ in range(n)] for _ in range(n)]

    # encourage x_i = x*_i using diagonal terms:
    # if x*_i = 1, positive diagonal; if 0, negative diagonal
    for i in range(n):
        if x_star[i] == 1:
            Q[i][i] += signal_strength
        else:
            Q[i][i] -= signal_strength

    # encourage pairs that match planted pattern:
    # if x*_i == x*_j, give positive weight; else negative
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                if x_star[i] == x_star[j]:
                    val = signal_strength
                else:
                    val = -signal_strength
                Q[i][j] += val
                Q[j][i] += val

    # add zero-mean noise to make the landscape less trivial
    for i in range(n):
        for j in range(i, n):
            noise = rng.uniform(-noise_scale, noise_scale)
            Q[i][j] += noise
            if i != j:
                Q[j][i] += noise

    inst = _qubo_to_sds(Q, card=card)
    # we don't store x_star in the instance, but you can
    # return it alongside if needed in your experiments.
    return inst


# ---------------------------------------------------------------------------
# 3) Max-Cut-as-QUBO on Erdos-Rényi graph
# ---------------------------------------------------------------------------

def make_maxcut_qubo_instance(
    n: int = 20,
    edge_prob: float = 0.5,
    weight_scale: float = 1.0,
    card: Tuple[int, int] = (0, 20),
    seed: Optional[int] = None,
) -> SDSInstance:
    """
    Max-Cut QUBO on an Erdos-Rényi graph G(n, p).

    Max-Cut objective (0/1 encoding, up to constant):

      maximize sum_{(i,j) in E} w_ij [ x_i + x_j - 2 x_i x_j ]

    This can be written as:

      constant + sum_i (sum_j w_ij) x_i  - 2 sum_{(i,j)} w_ij x_i x_j

    So we set:
      w_i   = sum_j w_ij
      W_ij  = -2 * w_ij

    No extra constraints are added.
    """
    rng = make_rng(seed)
    # adjacency with weights
    W_edges: Dict[Tuple[int, int], float] = {}
    degree_weight = [0.0 for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                wij = rng.uniform(0.1, 1.0) * weight_scale
                W_edges[(i, j)] = wij
                degree_weight[i] += wij
                degree_weight[j] += wij

    # build Q as implicit, but map directly to SDS:
    # w_i = sum_j w_ij
    w = degree_weight
    W: Dict[Tuple[int, int], float] = {}
    for (i, j), wij in W_edges.items():
        W[(i, j)] = -2.0 * wij

    precedence: List[Tuple[int, int]] = []
    mutex: List[Tuple[int, int]] = []
    groups: Dict[int, List[int]] = {}

    return SDSInstance(
        n=n,
        w=w,
        W=W,
        precedence=precedence,
        mutex=mutex,
        groups=groups,
        card=CardBounds(*card),
    )
