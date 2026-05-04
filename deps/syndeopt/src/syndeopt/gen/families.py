from __future__ import annotations
from typing import Optional, Dict, Tuple, List
import itertools

from ..core.instance import SDSInstance, CardBounds
from ..core.rng import make_rng

Bitmask = int

# --- small helper for generation-time feasibility probing --------------------

def _has_any_feasible_solution(inst: SDSInstance, max_vars_for_bf: int = 20) -> bool:
    """
    Very small brute-force feasibility probe for generation time.
    Returns True if there exists at least one feasible assignment.
    Only used when n is reasonably small.
    """
    n = inst.n
    if n > max_vars_for_bf:
        # don't brute force on large n; assume it's fine
        return True

    L, U = inst.card.L, inst.card.U
    from ..core.feasibility import feasible

    for k in range(L, U + 1):
        for combo in itertools.combinations(range(n), k):
            x = 0
            for i in combo:
                x |= (1 << i)
            if feasible(inst, x):
                return True
    return False


# ---------------------------------------------------------------------------
# 1) Greedy-friendly: modular objective, positive unaries, no tricky constraints
# ---------------------------------------------------------------------------

def make_greedy_easy_instance(
    n: int = 12,
    card: Tuple[int, int] = (5, 5),
    seed: Optional[int] = 101,
) -> SDSInstance:
    """
    Greedy-friendly instance:
      - No pairwise terms (W = 0) -> modular objective
      - Positive unaries
      - No precedence, no mutex, no groups
    Greedy that picks top weights is optimal.
    """
    rng = make_rng(seed)
    w = sorted([rng.uniform(1.0, 10.0) for _ in range(n)], reverse=True)
    W: Dict[Tuple[int, int], float] = {}
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
# 2) Local-search-friendly: "bait vs clique" local optima
# ---------------------------------------------------------------------------

def make_local_optima_instance(
    n: int = 18,
    card: Tuple[int, int] = (5, 5),
    seed: Optional[int] = 202,
) -> SDSInstance:
    """
    Terrain where local_search tends to beat greedy:

    - Node 0 is a high-unary 'bait', mutex with node 5.
    - Nodes 1..5 form a strong positive clique.
    - Cardinality window around 5 encourages choosing either bait+fillers
      or mostly the clique; the clique is globally better but greedy
      is tempted by the bait.
    """
    rng = make_rng(seed)
    n = max(n, 10)

    # unaries: bait big, clique moderate, fillers small
    w = [6.0] + [2.0]*5 + [0.2]*(n - 6)

    # clique 1..5 with strong positive pairwise interactions
    W: Dict[Tuple[int, int], float] = {}
    clique = [1, 2, 3, 4, 5]
    for i in range(len(clique)):
        for j in range(i + 1, len(clique)):
            W[(clique[i], clique[j])] = 3.0

    # tiny random noise among fillers
    fillers = list(range(6, n))
    all_pairs = [(i, j) for i in fillers for j in range(i + 1, n)]
    rng.shuffle(all_pairs)
    for (i, j) in all_pairs[: min(6, len(all_pairs))]:
        W[(i, j)] = rng.uniform(-0.1, 0.2)

    precedence: List[Tuple[int, int]] = []
    mutex = [(0, 5)]  # bait blocks completing the clique
    groups: Dict[int, List[int]] = {}

    inst = SDSInstance(
        n=n,
        w=w,
        W=W,
        precedence=precedence,
        mutex=mutex,
        groups=groups,
        card=CardBounds(*card),
    )
    return inst


# ---------------------------------------------------------------------------
# 3) Tree-structured showcase (DP- / structure-friendly)
# ---------------------------------------------------------------------------

def make_tree_showcase_instance(
    n: int = 14,
    card: Tuple[int, int] = (4, 10),
    weight_scale: float = 10.0,
    pair_scale: float = 6.0,
    seed: Optional[int] = 404,
) -> SDSInstance:
    """
    Tree-structured dependency graph, similar to make_tree_instance
    but intended as a 'canonical' tree regime for benchmarks.
    """
    from .trees import make_tree_instance
    return make_tree_instance(
        n=n,
        card=card,
        weight_scale=weight_scale,
        pair_scale=pair_scale,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 4) Decomposable clusters: D&C-friendly
# ---------------------------------------------------------------------------

def make_decomposable_instance(
    n: int = 18,
    card: Tuple[int, int] = (6, 6),
    clusters: int = 3,
    seed: Optional[int] = 505,
) -> SDSInstance:
    """
    Deterministic decomposable instance with several disconnected clusters.

    - Cluster A (0..5): weak clique with higher unaries
    - Cluster B (6..11): two strong triangles, slightly lower unaries
    - Remaining vars (12..) isolated

    No cross-cluster edges or constraints: divide-and-conquer
    or component-wise solving is natural.
    """
    rng = make_rng(seed)
    n = max(n, 12)

    # unaries
    w = [2.0]*6 + [1.5]*6 + [1.0]*max(0, n - 12)

    W: Dict[Tuple[int, int], float] = {}

    # cluster A: weak clique 0..5
    A = list(range(0, 6))
    for i in range(6):
        for j in range(i + 1, 6):
            W[(A[i], A[j])] = 0.8

    # cluster B: two strong triangles (6,7,8) and (9,10,11)
    tri1 = [6, 7, 8]
    tri2 = [9, 10, 11]
    for tri in (tri1, tri2):
        for i in range(3):
            for j in range(i + 1, 3):
                W[(tri[i], tri[j])] = 2.5

    # no edges for remaining nodes -> independent variables

    precedence: List[Tuple[int, int]] = []
    mutex: List[Tuple[int, int]] = []
    groups: Dict[int, List[int]] = {}

    inst = SDSInstance(
        n=n,
        w=w,
        W=W,
        precedence=precedence,
        mutex=mutex,
        groups=groups,
        card=CardBounds(*card),
    )
    return inst


# ---------------------------------------------------------------------------
# 5) Dense & deceptive instances: BnB / CP-SAT-friendly
# ---------------------------------------------------------------------------

def make_dense_deceptive_instance(
    n: int = 20,
    card: Tuple[int, int] = (7, 11),
    pos_pair_frac: float = 0.55,
    neg_pair_frac: float = 0.45,
    weight_scale: float = 8.0,
    pair_scale: float = 6.0,
    seed: Optional[int] = 303,
    attempts: int = 100,
) -> SDSInstance:
    """
    Dense, deceptive landscape:

    - Mixed positive / negative pairwise edges.
    - Some precedence + mutex constraints.
    - Regenerated a few times until we find an instance with at least
      one feasible solution (using small brute-force when n is small).

    This regime is meant to be 'hard' for greedy / local search and
    a good playground for BnB / CP-SAT / powerful heuristics.
    """
    base_rng = make_rng(seed)

    for _ in range(attempts):
        # derive a fresh RNG from base
        seed_i = base_rng.randrange(10_000_000)
        rng = make_rng(seed_i)

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

        # constraints to create traps
        mutex: List[Tuple[int, int]] = []
        if W:
            mutex = rng.sample(list(W.keys()), k=min(3, len(W)))

        # [FIX] Harden precedence constraints: scale with N and create chains
        precedence: List[Tuple[int, int]] = []
        # 1. Create a chain dependency (harder to satisfy by luck than random pairs)
        chain_len = min(n, 5)
        chain = rng.sample(range(n), chain_len)
        for k in range(chain_len - 1):
            precedence.append((chain[k], chain[k+1]))  # parent -> child
        
        # 2. Add additional random precedence constraints (scale with N)
        num_additional = max(2, n // 5)
        for _ in range(num_additional):
            i, j = rng.sample(range(n), 2)
            if (i, j) not in precedence and (j, i) not in precedence:
                precedence.append((i, j))
        
        # 3. Add "trap" dependencies: high-value child requires negative-value parent
        high_val_nodes = [i for i in range(n) if w[i] > weight_scale * 0.5]
        neg_val_nodes = [i for i in range(n) if w[i] < 0]
        if high_val_nodes and neg_val_nodes:
            child = rng.choice(high_val_nodes)
            parent = rng.choice(neg_val_nodes)
            if child != parent and (parent, child) not in precedence:
                precedence.append((parent, child))  # Must pick bad parent to get good child

        # groups: small random groups of size 2 or 3, not covering everything
        groups: Dict[int, List[int]] = {}
        G = list(range(n))
        rng.shuffle(G)
        gid = 0
        for start in range(0, min(n, n // 2), 3):
            grp = G[start:start + 3]
            if len(grp) >= 2:
                groups[gid] = grp
                gid += 1

        inst = SDSInstance(
            n=n,
            w=w,
            W=W,
            precedence=precedence,
            mutex=mutex,
            groups=groups,
            card=CardBounds(*card),
        )

        if _has_any_feasible_solution(inst):
            return inst

    # if all attempts fail, just return the last one (very unlikely to be infeasible)
    return inst


# ---------------------------------------------------------------------------
# 6) Structural Trap: Chain of Pain (Precedence-Dominant)
# ---------------------------------------------------------------------------

def make_structural_trap_instance(
    n: int = 20,
    chain_length: int = 5,
    bait_reward: float = 100.0,
    trap_penalty: float = -10.0,
    seed: Optional[int] = None,
) -> SDSInstance:
    """
    Generates a 'Structural Trap' instance designed to fail solvers that
    ignore precedence constraints or lack lookahead.

    Topology:
    1. A 'Chain' of dependencies: P1 -> P2 -> ... -> P_k -> LEAF
    2. Weights:
       - P_i (Parents): Negative weight (trap_penalty)
       - LEAF: Massive positive weight (bait_reward)
    3. Logic:
       - To get LEAF, you MUST pick all P_i.
       - Net Score = bait_reward + (k * trap_penalty) > 0.
       - A solver ignoring precedence will pick LEAF and drop P_i -> INVALID.
       - A greedy solver will see P_1 is negative and stop -> SUBOPTIMAL.

    This ensures that any solver that ignores precedence constraints will
    fail validation on these instances.
    """
    rng = make_rng(seed)
    n = max(n, chain_length + 5)
    
    w = [0.0] * n
    precedence: List[Tuple[int, int]] = []
    
    # 1. Create the Chain (The Trap)
    # Select random indices for the chain
    chain_indices = rng.sample(range(n), chain_length + 1)
    leaf_node = chain_indices[-1]
    parents = chain_indices[:-1]
    
    # Set weights
    w[leaf_node] = bait_reward
    for p in parents:
        w[p] = trap_penalty  # Painful ancestors
        
    # Link dependencies: P1 -> P2 -> ... -> LEAF
    # (i.e., if P(i+1) is selected, P(i) must be selected)
    # Constraint format: (parent, child) -> if child, then parent
    for i in range(len(chain_indices) - 1):
        p, c = chain_indices[i], chain_indices[i+1]
        precedence.append((p, c))
        
    # 2. Add some random noise/distractors for the rest of the nodes
    # so it's not too obvious
    others = [i for i in range(n) if i not in chain_indices]
    for i in others:
        w[i] = rng.uniform(-5.0, 5.0)
        
    # 3. Add random pairwise interactions to create standard noise
    W: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.3:
                W[(i, j)] = rng.uniform(-2.0, 2.0)
    
    # 4. Strict Cardinality to force choices
    # Must be large enough to hold the chain + a few others
    L = chain_length + 1
    U = n
    
    return SDSInstance(
        n=n,
        w=w,
        W=W,
        precedence=precedence,
        mutex=[],
        groups={},
        card=CardBounds(L, U),
    )
