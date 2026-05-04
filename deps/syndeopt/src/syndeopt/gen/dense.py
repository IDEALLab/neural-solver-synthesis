from typing import Optional, Dict, Tuple, List
from ..core.instance import SDSInstance, CardBounds
from ..core.rng import make_rng

def make_dense_instance(
    n: int = 16,
    card=(6, 10),
    pos_pair_frac: float = 0.6,
    neg_pair_frac: float = 0.4,
    weight_scale: float = 8.0,
    pair_scale: float = 6.0,
    seed: Optional[int] = None,
) -> SDSInstance:
    rng = make_rng(seed)
    w = [rng.uniform(-0.2, 1.0) * weight_scale for _ in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(pairs)
    m_pos = int(len(pairs) * pos_pair_frac)
    m_neg = int(len(pairs) * neg_pair_frac)

    W: Dict[Tuple[int, int], float] = {}
    for (i, j) in pairs[:m_pos]:
        W[(i, j)] = rng.uniform(0.3, 1.0) * pair_scale
    for (i, j) in pairs[m_pos:m_pos + m_neg]:
        W[(i, j)] = -rng.uniform(0.3, 1.0) * pair_scale

    mutex: List[Tuple[int, int]] = []
    if W:
        mutex = rng.sample(list(W.keys()), k=min(3, len(W)))

    precedence: List[Tuple[int, int]] = []
    for _ in range(2):
        i, j = rng.sample(range(n), 2)
        precedence.append((i, j))

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
