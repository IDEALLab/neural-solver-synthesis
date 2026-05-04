# DEPRECATED: This file has been replaced by syndeopt.
# All SDS functionality now uses syndeopt (deps/syndeopt).
# This file is kept for backward compatibility but should not be used for new code.
# 
# Migration guide:
# - Use syndeopt.core.instance.SDSInstance instead of this module's SDSInstance
# - Use syndeopt.gen for instance generation
# - Use syndeopt.solvers for solving
# - Use syndeopt.core.scoring.score() and syndeopt.core.feasibility.feasible()
#
# Synergistic Dependency Selection (SDS): a tunable toy problem and solver zoo
#
# Goal: Choose a subset x ∈ {0,1}^n to maximize
#   f(x) = sum_i w_i x_i + sum_{(i,j)∈E} W_ij x_i x_j
# subject to any combo of constraints (precedence, mutual exclusion, group caps, cardinality window).
# The dependency graph (pairwise edges E) is user-tunable. Different regimes make different solvers best.
#
# Included solvers:
#  - greedy_marginal: fast heuristic (may miss optimum badly)
#  - local_search: 1-flip hill climbing with random restarts
#  - dp_tree: exact max-product DP on trees (when the dependency graph is a forest)
#  - branch_and_bound: exact, generic; pruning via simple optimistic bound
#  - divide_and_conquer: splits into connected components; solves each with a chosen base solver and merges
#  - brute_force: exact; ok up to ~18-20 vars (used for verification/demo)
#
# The harness below builds a few regimes showing solver behavior differences.
import math, random, time, itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set

random.seed(7) # TODO: remove this and pass a arg to the simulator to set the seed

@dataclass
class SDSInstance:
    n: int                              # number of variables (10..20 typical)
    w: List[float]                      # unary weights length n
    W: Dict[Tuple[int,int], float]      # pairwise interactions (i<j)->weight
    precedence: List[Tuple[int,int]]    # arcs i->j mean x_j <= x_i
    mutex: List[Tuple[int,int]]         # pairs that cannot co-exist
    groups: Dict[int, List[int]]        # group_id -> list of vars; pick at most 1 per group
    card_bounds: Tuple[int,int]         # (L, U) cardinality bounds
    # derived
    adj: List[Set[int]] = field(init=False)

    def __post_init__(self):
        self.adj = [set() for _ in range(self.n)]
        for (i,j),wij in self.W.items():
            self.adj[i].add(j)
            self.adj[j].add(i)

    def score(self, x_bits: int) -> float:
        """Evaluate objective for a bitmask x_bits."""
        s = 0.0
        # unary
        for i in range(self.n):
            if (x_bits >> i) & 1:
                s += self.w[i]
        # pairwise (i<j)
        for (i,j),wij in self.W.items():
            if ((x_bits >> i) & 1) and ((x_bits >> j) & 1):
                s += wij
        return s

    def feasible(self, x_bits: int) -> bool:
        # cardinality
        k = x_bits.bit_count()
        L, U = self.card_bounds
        if not (L <= k <= U):
            return False
        # precedence: j can be 1 only if i is 1
        for i,j in self.precedence:
            if ((x_bits >> j) & 1) and not ((x_bits >> i) & 1):
                return False
        # mutex
        for a,b in self.mutex:
            if ((x_bits >> a) & 1) and ((x_bits >> b) & 1):
                return False
        # groups: at most 1 per group
        for gid, members in self.groups.items():
            cnt = sum((x_bits >> i) & 1 for i in members)
            if cnt > 1:
                return False
        return True

# ---------------------------- Instance factories ----------------------------

def make_tree_instance(n: int = 14, card=(4, 10), weight_scale=10.0, pair_scale=6.0, seed: Optional[int] = None) -> SDSInstance:
    """Sparse tree-structured dependencies -> dp_tree can be exact & fast."""
    if seed is not None:
        random.seed(seed)
    assert n >= 2
    # Build a random tree (Prüfer sequence style)
    nodes = list(range(n))
    prufer = [random.randrange(n) for _ in range(n-2)]
    degree = [1]*n
    for v in prufer: degree[v] += 1
    leaves = sorted([i for i in range(n) if degree[i]==1])
    edges = []
    for v in prufer:
        u = leaves[0]
        leaves.pop(0)
        edges.append((min(u,v), max(u,v)))
        degree[u]-=1; degree[v]-=1
        if degree[v]==1:
            # insert in sorted order
            idx = 0
            while idx < len(leaves) and leaves[idx] < v: idx += 1
            leaves.insert(idx, v)
    u,v = leaves
    edges.append((min(u,v), max(u,v)))

    # Positive pairwise synergies on edges; mix of positive/negative unaries
    w = [random.uniform(-0.5,1.0)*weight_scale for _ in range(n)]
    W = {e: random.uniform(0.5,1.0)*pair_scale for e in edges}

    # Light constraints: a few mutex and a few precedence along tree edges
    mtx = random.sample(edges, k=min(2, len(edges)))
    prec = random.sample(edges, k=min(2, len(edges)))  # direction matters
    prec = [(i,j) if random.random()<0.5 else (j,i) for (i,j) in prec]

    # groups: none by default
    groups = {}
    return SDSInstance(n, w, W, precedence=prec, mutex=mtx, groups=groups, card_bounds=card)

def make_dense_tricky_instance(n: int = 16, card=(6, 10), pos_pair_frac=0.6, neg_pair_frac=0.4,
                               weight_scale=8.0, pair_scale=6.0, seed: Optional[int] = None, attempts: int = 200) -> SDSInstance:
    """Dense, deceptive landscape: local search & greedy can get stuck; BnB needed for exact optimum.
    
    Regenerates until a feasible instance is found (up to attempts tries).
    """
    base_seed = seed
    for attempt in range(attempts):
        if base_seed is not None:
            # Use seed + attempt for reproducibility, but allow more variation
            random.seed(base_seed + attempt * 17)  # Multiply to get more variation
        else:
            # Use current random state if no seed provided
            pass
        
        w = [random.uniform(-0.2,1.0)*weight_scale for _ in range(n)]
        pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
        random.shuffle(pairs)
        m_pos = int(len(pairs)*pos_pair_frac)
        m_neg = int(len(pairs)*neg_pair_frac)
        W = {}
        for (i,j) in pairs[:m_pos]:
            W[(i,j)] = random.uniform(0.3,1.0)*pair_scale  # attractive synergy
        for (i,j) in pairs[m_pos:m_pos+m_neg]:
            W[(i,j)] = -random.uniform(0.3,1.0)*pair_scale # repulsive penalties
        # constraints to create traps
        # mutex a few high-synergy pairs so greedy picks wrong early (reduce to 2 for better feasibility)
        mutex = random.sample(list(W.keys()), k=min(2, len(W)))
        # random precedence to couple choices (reduce to 1-2 for better feasibility)
        precedence = []
        for _ in range(min(2, n//5)):
            i,j = random.sample(range(n),2)
            precedence.append((i,j))
        # groups partition to break simple density heuristics (make groups smaller/less restrictive)
        groups = {}
        G = list(range(n))
        random.shuffle(G)
        gid=0
        # Make groups smaller (size 2-3 instead of 4) and fewer groups
        for start in range(0, min(n, n//2), 3):  # Only cover half the variables
            grp = G[start:start+3]
            if len(grp)>=2:
                groups[gid]=grp; gid+=1
        
        inst = SDSInstance(n, w, W, precedence=precedence, mutex=mutex, groups=groups, card_bounds=card)
        
        # Quick feasibility probe: try brute force if small enough, otherwise greedy
        if n <= 20:
            s, x = brute_force(inst, limit_vars=20)
            if s != -float('inf') and inst.feasible(x):
                return inst
        else:
            s, x = greedy_marginal(inst)
            if inst.feasible(x):
                return inst
    
    raise ValueError(f"Could not generate a feasible dense instance after {attempts} attempts; relax parameters.")

def make_decomposable_instance(n:int=18, card=(6, 6), clusters=3, p_in=0.7, p_out=0.0,
                               weight_scale=7.0, pair_scale=5.0, seed: Optional[int] = None) -> SDSInstance:
    """
    Deterministic decomposable instance that favors Divide & Conquer.
    - A (0..5): weak clique, higher unaries (tempts greedy/LS to overfill).
    - B (6..11): two strong triangles, slightly lower unaries (huge gain if completed).
    - Remaining nodes (12..n-1): isolated (keeps components disconnected).
    Use a tight card window in the showcase (e.g., (6,6)) so allocation matters.
    """
    n = max(n, 12)  # ensure A and B exist

    # Unaries: A a bit higher than B; extras neutral
    w = [2.0]*6 + [1.5]*6 + [1.0]*max(0, n-12)

    # Pairwise
    W: Dict[Tuple[int,int], float] = {}
    # A: weakly attractive full clique
    A = list(range(0, 6))
    for i in range(6):
        for j in range(i+1, 6):
            W[(A[i], A[j])] = 0.8
    # B: two strong triangles
    tri1 = [6,7,8]; tri2 = [9,10,11]
    for tri in (tri1, tri2):
        for i in range(3):
            for j in range(i+1, 3):
                W[(tri[i], tri[j])] = 2.5

    precedence: List[Tuple[int,int]] = []
    mutex: List[Tuple[int,int]] = []
    groups: Dict[int, List[int]] = {}
    return SDSInstance(n, w, W, precedence, mutex, groups, card_bounds=card)




# ---------------------------- Solvers ----------------------------

def greedy_marginal(inst: SDSInstance, iters: Optional[int]=None) -> Tuple[float,int]:
    """Greedy by marginal gain under feasibility; returns (best_score, bitmask)."""
    n = inst.n
    L,U = inst.card_bounds
    x = 0
    # start from feasible empty (ensure L could be zero; if L>0 we'll add until L at least)
    chosen = set()
    remaining = set(range(n))
    def feasible_with(i, cur):
        cand = cur | (1<<i)
        # Use relaxed feasibility (ignore lower bound) while building
        return feasible_without_lower(inst, cand)
    iters = iters or n
    for _ in range(iters):
        # pick best positive marginal among feasible additions
        best_i, best_gain = None, -float('inf')
        for i in list(remaining):
            if feasible_with(i, x):
                # marginal: delta f if we add i
                gain = inst.w[i]
                for j in chosen:
                    a,b = (min(i,j), max(i,j))
                    if (a,b) in inst.W:
                        gain += inst.W[(a,b)]
                # also consider pairs with not-yet-chosen? Greedy can't foresee -> omitted by design
                if gain > best_gain:
                    best_gain, best_i = gain, i
        if best_i is not None and best_gain > 0:
            x |= (1<<best_i)
            chosen.add(best_i); remaining.remove(best_i)
        else:
            break
    # If below L, add least-bad feasible to meet cardinality
    while (x.bit_count() < L):
        best_i, best_gain = None, -float('inf')
        for i in list(remaining):
            if feasible_without_lower(inst, x | (1<<i)):
                gain = inst.w[i]
                for j in chosen:
                    a,b = (min(i,j), max(i,j))
                    if (a,b) in inst.W: gain += inst.W[(a,b)]
                if gain > best_gain:
                    best_gain, best_i = gain, i
        if best_i is None: break
        x |= (1<<best_i); chosen.add(best_i); remaining.remove(best_i)
    return inst.score(x), x

def feasible_without_lower(inst: SDSInstance, x: int) -> bool:
    """Check feasibility ignoring lower cardinality bound; enforce k<=U and all other constraints."""
    k = x.bit_count()
    _, U = inst.card_bounds
    if k > U:
        return False
    # precedence: j can be 1 only if i is 1
    for i, j in inst.precedence:
        if ((x >> j) & 1) and not ((x >> i) & 1):
            return False
    # mutex
    for a, b in inst.mutex:
        if ((x >> a) & 1) and ((x >> b) & 1):
            return False
    # groups: at most 1 per group
    for gid, members in inst.groups.items():
        cnt = sum((x >> i) & 1 for i in members)
        if cnt > 1:
            return False
    return True

def local_search(inst: SDSInstance, restarts:int=20, iters:int=2000) -> Tuple[float,int]:
    """1-flip hill climbing with random restarts; keeps feasibility."""
    n=inst.n
    best_s, best_x = -float('inf'), 0
    for _ in range(restarts):
        # random feasible start: use relaxed feasibility (ignore lower bound) until we reach L
        x=0
        L,U = inst.card_bounds
        idx = list(range(n)); random.shuffle(idx)
        # Build initial solution using relaxed feasibility until we reach L
        for i in idx:
            if x.bit_count() < U:
                if x.bit_count() < L:
                    # Use relaxed feasibility (ignore lower bound) during initialization
                    if feasible_without_lower(inst, x | (1<<i)):
                        x |= (1<<i)
                else:
                    # Once we've reached L, use full feasibility check
                    if inst.feasible(x | (1<<i)):
                        x |= (1<<i)
        # Ensure we have a feasible solution (meets all constraints including L)
        if not inst.feasible(x):
            # Try to repair: if below L, add more variables
            while x.bit_count() < L:
                best_i = None
                for i in range(n):
                    if not ((x >> i) & 1) and feasible_without_lower(inst, x | (1<<i)):
                        best_i = i
                        break
                if best_i is None:
                    break
                x |= (1<<best_i)
            # If still not feasible or above U, try to fix
            if not inst.feasible(x):
                # Fallback: start from empty and try greedy
                s_repair, x_repair = greedy_marginal(inst)
                if inst.feasible(x_repair) and s_repair > inst.score(x):
                    x = x_repair
        
        # Now hill climb with full feasibility
        improved=True; steps=0
        while improved and steps<iters:
            improved=False; steps+=1
            # try all 1-bit flips (add or remove) that maintain feasibility
            best_gain, best_move = 0.0, None
            cur = inst.score(x)
            for i in range(n):
                y = x ^ (1<<i)  # toggle
                if y==x: continue
                if inst.feasible(y):
                    gain = inst.score(y) - cur
                    if gain > best_gain:
                        best_gain, best_move = gain, y
            if best_move is not None and best_gain>1e-12:
                x = best_move
                improved=True
        s=inst.score(x)
        # Only accept feasible solutions as best
        if inst.feasible(x) and s>best_s:
            best_s, best_x = s, x
    return best_s, best_x

def _optimistic_bound(inst:SDSInstance, fixed_bits:int, idx:int, order:List[int]) -> float:
    """Simple optimistic upper bound for BnB: current score + sum of max possible remaining contributions.
       We assume remaining vars can be set to maximize unary + positive pairwise with already-included vars,
       and all pairwise among remaining counted only if positive (very loose but admissible)."""
    n = inst.n
    cur_score = inst.score(fixed_bits)
    remaining = order[idx:]
    bonus = 0.0
    chosen = [i for i in range(n) if (fixed_bits>>i)&1]
    for i in remaining:
        # optimistic unary: take if w_i>0
        if inst.w[i] > 0: bonus += inst.w[i]
        # optimistic pairs with chosen
        for j in chosen:
            a,b = (min(i,j), max(i,j))
            if (a,b) in inst.W:
                wij = inst.W[(a,b)]
                if wij>0: bonus += wij
        # optimistic pairs among remaining (count half to avoid double counting too much)
    # pairs among remaining
    R = remaining
    for ii in range(len(R)):
        for jj in range(ii+1,len(R)):
            a,b = (min(R[ii], R[jj]), max(R[ii], R[jj]))
            wij = inst.W.get((a,b),0.0)
            if wij>0: bonus += wij
    return cur_score + bonus

def branch_and_bound(inst:SDSInstance, time_limit:float=1.5) -> Tuple[float,int]:
    """Exact BnB with simple bound & feasibility checks; best-first over variable order by degree."""
    start = time.time()
    n=inst.n
    # variable ordering: high degree first (usually prunes better)
    deg = [len(inst.adj[i]) for i in range(n)]
    order = [i for i,_ in sorted(enumerate(deg), key=lambda x:-x[1])]
    best_s, best_x = -float('inf'), 0
    # stack entries: (idx, x_bits)
    stack=[(0,0)]
    visited=0
    while stack:
        if time.time()-start>time_limit:
            break
        idx, x = stack.pop()
        visited+=1
        # bound
        ub = _optimistic_bound(inst, x, idx, order)
        if ub <= best_s + 1e-12:
            continue
        if idx==n:
            if inst.feasible(x):
                s = inst.score(x)
                if s>best_s: best_s, best_x = s, x
            continue
        var = order[idx]
        # Branch 1: include var if feasible prefix-wise (fast local checks via full feasibility on leaf)
        y = x | (1<<var)
        # Cheap feasibility pre-checks (partial): respect cardinality upper bound and mutex with already included
        if (y.bit_count() <= inst.card_bounds[1]):
            ok = True
            for (a,b) in inst.mutex:
                if ((y>>a)&1) and ((y>>b)&1):
                    ok=False; break
            if ok:
                stack.append((idx+1, y))
        # Branch 2: exclude var
        stack.append((idx+1, x))
    # If time ran out, try to repair with feasibility and compute best known score
    if not inst.feasible(best_x):
        # Try to greedily repair into feasibility
        s,x = greedy_marginal(inst)
        if s>best_s: best_s, best_x = s,x
    return best_s, best_x

def dp_tree(inst:SDSInstance) -> Optional[Tuple[float,int]]:
    """Exact max-product DP on trees (pairwise model). Returns None if graph has cycles.
    
    Note: This DP is exact for the pure pairwise model but ignores global constraints
    (cardinality, precedence, groups, mutex). The result is repaired greedily if needed.
    """
    n=inst.n
    # check if the graph is a forest
    parent=[-1]*n
    visited=[False]*n
    order=[]
    # detect cycles via DFS
    def dfs(u,p):
        parent[u]=p; visited[u]=True
        for v in inst.adj[u]:
            if v==p: continue
            if visited[v]: 
                # back-edge => cycle
                return False
            if not dfs(v,u): return False
        order.append(u)
        return True
    # run on each component
    for s in range(n):
        if not visited[s]:
            if not dfs(s,-1): 
                return None  # has cycle

    # dp0[u] = best subtree score if u=0
    # dp1[u] = best subtree score if u=1 (includes w[u] exactly once)
    dp0=[0.0]*n
    dp1=[0.0]*n
    # Build quick access for pairwise weight
    def Wij(i,j):
        a,b=(min(i,j),max(i,j))
        return inst.W.get((a,b),0.0)
    # Process in reverse DFS order (leaves to root)
    for u in order:
        # Aggregate over children
        for v in inst.adj[u]:
            if v==parent[u]: continue
            # For u=0: take max over v's states (no pairwise contribution)
            # For u=1: take max over v's states, with pairwise bonus if both 1
            dp0[u] += max(dp0[v], dp1[v])
            dp1[u] += max(dp0[v], Wij(u,v) + dp1[v])
        # Add u's own unary weight (only once, when u=1)
        dp1[u] += inst.w[u]
    
    # Reconstruct per component:
    choose=[0]*n
    roots=[i for i,p in enumerate(parent) if p==-1]
    for r in roots:
        # pick r=0 or r=1
        choose[r] = 1 if dp1[r] >= dp0[r] else 0
        # backtrack
        stack=[r]
        while stack:
            u=stack.pop()
            for v in inst.adj[u]:
                if v==parent[u]: continue
                # given choose[u], pick choose[v] optimally
                if choose[u]==0:
                    # u=0: no pairwise contribution
                    choose[v] = 1 if dp1[v] >= dp0[v] else 0
                else:
                    # u=1: pairwise contributes if both 1
                    choose[v] = 1 if (Wij(u,v) + dp1[v]) >= dp0[v] else 0
                stack.append(v)
    x_bits = sum((choose[i]<<i) for i in range(n))
    # If extra constraints violated, try a quick greedy repair to nearest feasible
    if not inst.feasible(x_bits):
        _, x_bits = greedy_marginal(inst)
    return inst.score(x_bits), x_bits

def brute_force(inst:SDSInstance, limit_vars:int=20) -> Tuple[float,int]:
    """Exact enumeration up to 2^n; fast enough for n<=20 (1,048,576 subsets)."""
    n=inst.n
    assert n<=limit_vars, "n too large for brute force"
    best_s, best_x = -float('inf'), 0
    L,U = inst.card_bounds
    # iterate by cardinality window to prune quickly
    for k in range(L, U+1):
        for combo in itertools.combinations(range(n), k):
            x=0
            for i in combo: x |= (1<<i)
            if not inst.feasible(x): 
                continue
            s = inst.score(x)
            if s>best_s: best_s, best_x = s,x
    return best_s, best_x

# ---------------------------- Helper: pretty print ----------------------------

def fmt_sol(x:int, n:int) -> str:
    return "{" + ", ".join(str(i) for i in range(n) if (x>>i)&1) + "}"

def run_regime(name:str, inst:SDSInstance, verify_with_bruteforce:bool=False):
    print(f"\n=== Regime: {name} | n={inst.n}, edges={len(inst.W)}, card={inst.card_bounds} ===")
    # try solvers
    solvers = []
    solvers.append(("greedy", lambda: greedy_marginal(inst)))
    solvers.append(("local_search", lambda: local_search(inst)))
    dpres = dp_tree(inst)
    if dpres is not None:
        solvers.append(("dp_tree", lambda: dpres))
    solvers.append(("divide_and_conquer", lambda: divide_and_conquer(inst)))
    solvers.append(("branch_and_bound", lambda: branch_and_bound(inst, time_limit=1.2)))
    if inst.n <= 18 and verify_with_bruteforce:
        solvers.append(("brute_force", lambda: brute_force(inst)))

    results=[]
    for name,fn in solvers:
        t0=time.time(); s,x = fn(); t1=time.time()
        results.append((name, s, x, t1-t0))
    # print results
    # Winner: highest score, then lowest time if tie
    best = max(results, key=lambda z:(z[1], -z[3]))  # (score, -time) for max score, min time
    for nm, s, x, dt in sorted(results, key=lambda z:(-z[1], z[3])):  # Sort by score desc, then time asc
        print(f"{nm:18s}  score={s:8.3f}  time={dt*1000:6.1f} ms  sel={fmt_sol(x, inst.n)}")
    print(f"-> Winner: {best[0]} (score={best[1]:.3f}, time={best[3]*1000:.1f} ms)")

# Divide-and-conquer: split into components and solve each with an exact base (BnB or BF) then merge

def connected_components(inst:SDSInstance)->List[List[int]]:
    n=inst.n
    vis=[False]*n
    comps=[]
    for s in range(n):
        if vis[s]: continue
        stack=[s]; vis[s]=True; comp=[s]
        while stack:
            u=stack.pop()
            for v in inst.adj[u]:
                if not vis[v]:
                    vis[v]=True; stack.append(v); comp.append(v)
        comps.append(comp)
    return comps

def sub_instance(inst:SDSInstance, nodes:List[int]) -> SDSInstance:
    idx_map = {v:i for i,v in enumerate(nodes)}
    n = len(nodes)
    w = [inst.w[v] for v in nodes]
    W = {}
    for (i,j),wij in inst.W.items():
        if i in idx_map and j in idx_map:
            a,b = idx_map[i], idx_map[j]
            if a<b: W[(a,b)] = wij
            else: W[(b,a)] = wij
    precedence = [(idx_map[i], idx_map[j]) for (i,j) in inst.precedence if i in idx_map and j in idx_map]
    mutex = [(idx_map[a], idx_map[b]) for (a,b) in inst.mutex if a in idx_map and b in idx_map]
    # groups: filter & remap
    groups = {}
    for gid, members in inst.groups.items():
        rem = [idx_map[i] for i in members if i in idx_map]
        if len(rem)>=2: groups[gid]=rem
    # cardinality: crude split (proportional); for correctness we won't strictly split bounds, we'll handle at merge
    # keep wide bounds locally; enforce global at merge.
    return SDSInstance(n, w, W, precedence, mutex, groups, card_bounds=(0, n))

def merge_components(inst:SDSInstance, comp_solutions:List[Tuple[float,int,List[int]]]) -> Tuple[float,int]:
    """Combine component solutions under global cardinality bounds.
    
    Assumes no cross-component pairwise edges or constraints (precedence, mutex, groups
    spanning components). This is valid for make_decomposable_instance with p_out=0.
    """
    # dynamic programming by total cardinality to respect global (L,U)
    L,U = inst.card_bounds
    # For each component, compute pareto frontier by (cardinality -> best (score, mask))
    fronts = []
    for score, mask, nodes in comp_solutions:
        nsub = len(nodes)
        # enumerate all subsets of this component: if small it's ok; else fallback greedy internally
        # but we already solved each comp optimally; build per-card choice from that solution only
        # To build a good frontier, do small brute-force per component (comps should be small).
        # If comp is big, approximate with local search frontier.
        frontier = {}
        if nsub <= 14:
            for k in range(0, nsub+1):
                best = (-float('inf'), 0)
                for combo in itertools.combinations(range(nsub), k):
                    x=0
                    for i in combo: x|=(1<<i)
                    # Evaluate using a mini-instance restricted to nodes
                    # Build temporary W
                    # We'll reconstruct using the original scoring to avoid mistakes
                    # Create full mask on original indices
                    full_mask = 0
                    for t, node in enumerate(nodes):
                        if ((x>>t)&1): full_mask |= (1<<node)
                    s = inst.score(full_mask)
                    # Don't filter by global feasibility here - let merge DP enforce L..U
                    if s>best[0]:
                        best=(s, x)
                if best[0] != -float('inf'):
                    frontier[k]=best
        else:
            # approximate: take greedy solution and vary around its cardinality
            # simple: record only that single point
            full_mask = 0
            for t,node in enumerate(nodes):
                if ((mask>>t)&1): full_mask |= (1<<node)
            k = bin(mask).count("1")
            frontier[k]=(inst.score(full_mask), mask)
        fronts.append((nodes, frontier))
    # Now knapsack-like DP over cardinalities
    dp = {0: (0.0, [])}  # total_k -> (score, list of (comp_idx, submask))
    for comp_idx, (nodes, frontier) in enumerate(fronts):
        new_dp = {}
        for tot_k, (s_acc, choices) in dp.items():
            for k, (s_k, submask) in frontier.items():
                ntot = tot_k + k
                ns = s_acc + s_k  # naive sum double-counts cross-comp pairwise, but there are none by construction
                if ntot not in new_dp or ns > new_dp[ntot][0]:
                    new_dp[ntot] = (ns, choices + [(comp_idx, submask)])
        dp = new_dp
    # pick best within [L,U]
    best = (-float('inf'), None)
    for tot_k,(s_acc, choices) in dp.items():
        if L<=tot_k<=U and s_acc>best[0]:
            best=(s_acc, choices)
    if best[1] is None:
        # fall back: greedy global
        return greedy_marginal(inst)
    # Build final mask
    final_mask = 0
    for (comp_idx, submask), (nodes, _) in zip(best[1], fronts):
        for t,node in enumerate(nodes):
            if ((submask>>t)&1): final_mask |= (1<<node)
    # Final feasibility check - repair if needed
    if not inst.feasible(final_mask):
        s_repair, x_repair = greedy_marginal(inst)
        if inst.feasible(x_repair):
            return s_repair, x_repair
    return inst.score(final_mask), final_mask

def solve_component_exact(inst_sub:SDSInstance) -> Tuple[float,int]:
    # Try BnB with a small time limit, then brute force if tiny
    if inst_sub.n <= 16:
        try:
            return brute_force(inst_sub)
        except AssertionError:
            pass
    return branch_and_bound(inst_sub, time_limit=0.5)

def divide_and_conquer(inst:SDSInstance) -> Tuple[float,int]:
    """Divide-and-conquer: split into connected components and solve each optimally, then merge.
    
    Assumption: This method is only valid when there are no cross-component pairwise edges
    or cross-component constraints (precedence, mutex, groups spanning components).
    This is true for make_decomposable_instance as written (p_out=0 means no cross-cluster edges,
    and groups are per-cluster).
    
    If cross-component constraints exist, the merge step may not find the true optimum.
    """
    comps = connected_components(inst)
    comp_solutions=[]
    for nodes in comps:
        sub = sub_instance(inst, nodes)
        s,x = solve_component_exact(sub)
        comp_solutions.append((s,x,nodes))
    return merge_components(inst, comp_solutions)

# ---------------------------- Demo runs ----------------------------

def demo(seed: int = 7):
    """Run demo with reproducible seeds for each regime."""
    # 1) Tree-structured: dp_tree should be exact & great
    inst1 = make_tree_instance(n=14, card=(4,10), seed=seed)
    run_regime("Tree / DP-exact", inst1, verify_with_bruteforce=True)

    # 2) Dense & deceptive: BnB likely needed for exact; local search often stuck
    inst2 = make_dense_tricky_instance(n=16, card=(6, 10), seed=seed)
    run_regime("Dense & deceptive / BnB", inst2, verify_with_bruteforce=False)

    # 3) Decomposable clusters: divide & conquer shines
    inst3 = make_decomposable_instance(n=15, card=(5, 11), clusters=3, p_in=0.7, p_out=0.0, seed=seed)
    run_regime("Decomposable / D&C", inst3, verify_with_bruteforce=True)

# ---------------------------- Showcase helpers ----------------------------

def run_regime_showcase(title: str,
                        inst: SDSInstance,
                        include: List[str],
                        bnb_time_limit: float = 1.2,
                        verify_with_bruteforce: bool = False,
                        ls_restarts: int = 20):

    """
    Run only the solvers listed in `include` (IDs: 'greedy','local_search','dp_tree',
    'divide_and_conquer','branch_and_bound','brute_force'), in this *order*.
    This lets us showcase a specific solver as the "winner" when scores tie.
    """
    print(f"\n=== Showcase: {title} | n={inst.n}, edges={len(inst.W)}, card={inst.card_bounds} ===")

    name2fn = {
        "greedy":            lambda: greedy_marginal(inst),
        "local_search":      lambda: local_search(inst, restarts=ls_restarts),
        "dp_tree":           lambda: dp_tree(inst),
        "divide_and_conquer":lambda: divide_and_conquer(inst),
        "branch_and_bound":  lambda: branch_and_bound(inst, time_limit=bnb_time_limit),
        "brute_force":       lambda: brute_force(inst),
    }


    results = []
    for nm in include:
        if nm not in name2fn:
            continue
        if nm == "dp_tree":
            dpres = dp_tree(inst)
            if dpres is None:
                # skip if graph isn't a forest
                continue
            fn = lambda res=dpres: res
        else:
            fn = name2fn[nm]
        t0 = time.time(); s, x = fn(); t1 = time.time()
        results.append((nm, s, x, t1 - t0))

    # Optionally verify with brute force (if feasible)
    if verify_with_bruteforce and inst.n <= 18:
        t0 = time.time(); s_bf, x_bf = brute_force(inst); t1 = time.time()
        results.append(("brute_force (verify)", s_bf, x_bf, t1 - t0))

    # Print results and declare the winner by score (ties go to the earlier solver in `include`)
    best = max(results, key=lambda z: (z[1], -z[3]))   # highest score, then lowest time
    for nm, s, x, dt in sorted(results, key=lambda z: (-z[1], z[3])):  # score desc, time asc
        print(f"{nm:20s}  score={s:8.3f}  time={dt*1000:7.1f} ms  sel={fmt_sol(x, inst.n)}")
    print(f"-> Winner: {best[0]} (score={best[1]:.3f})")

# ---------------------------- Hand-crafted regimes ----------------------------

def make_greedy_easy_instance(n=12, card=(5,5), seed=101) -> SDSInstance:
    """
    Greedy-friendly: modular objective (W=0), positive unaries, no tricky constraints.
    Greedy that picks top weights is optimal (local_search ties, but we'll list greedy first).
    """
    random.seed(seed)
    w = sorted([random.uniform(1.0, 10.0) for _ in range(n)], reverse=True)
    W = {}  # modular
    precedence = []
    mutex = []
    groups = {}
    return SDSInstance(n, w, W, precedence, mutex, groups, card_bounds=card)

def make_local_optima_instance(n=18, card=(8, 10), seed: Optional[int] = None) -> SDSInstance:
    """
    Deterministic terrain where local_search beats greedy.
    Node 0 is a high-unary 'bait' mutex with one node of a 5-node strong clique.
    Greedy goes bait+4, local_search finds the all-clique basin which scores higher.
    """
    # Ensure room: we need at least 10 variables; extras are neutral fillers
    n = max(n, 10)

    # Unaries: bait big, clique moderate, fillers tiny
    w = [6.0] + [2.0]*5 + [0.2]*(n-6)

    # Pairwise: strong positive synergy on clique 1..5
    W: Dict[Tuple[int,int], float] = {}
    clique = [1,2,3,4,5]
    for i in range(len(clique)):
        for j in range(i+1, len(clique)):
            W[(clique[i], clique[j])] = 3.0

    # Optional tiny noise among fillers (doesn't change ordering)
    if seed is not None:
        random.seed(seed)
    fillers = list(range(6, n))
    # (keep very sparse/weak so it won't overturn the intended result)
    for (i, j) in random.sample([(i, j) for i in fillers for j in range(i+1, n)],
                                k=min(6, max(0, (n-6)))):
        W[(i, j)] = random.uniform(-0.1, 0.2)

    precedence: List[Tuple[int,int]] = []
    mutex = [(0, 5)]  # bait blocks completing the clique
    groups: Dict[int, List[int]] = {}

    # To make the contrast crisp, use a tight window around 5 picks in the showcase call
    return SDSInstance(n, w, W, precedence, mutex, groups, card_bounds=card)



def make_bnb_showcase_instance(n=20, card=(7, 11), seed=303) -> SDSInstance:
    """
    Dense & deceptive, but still feasible. Mixed positive/negative edges and a few constraints
    to create deep traps; BnB (with a decent time limit) will find better scores than heuristics.
    """
    return make_dense_tricky_instance(n=n, card=card, pos_pair_frac=0.55, neg_pair_frac=0.45,
                                      weight_scale=8.0, pair_scale=6.0, seed=seed)

def make_tree_showcase_instance(n=14, card=(4, 10), seed=404) -> SDSInstance:
    """Pure tree structure; dp_tree is exact (then repaired for global constraints if needed)."""
    return make_tree_instance(n=n, card=card, weight_scale=10.0, pair_scale=6.0, seed=seed)

def make_dc_showcase_instance(n=18, clusters=4, card=(6, 12), seed=505) -> SDSInstance:
    """
    Several disconnected clusters; divide & conquer can solve components exactly and merge
    under global cardinality. No cross edges/constraints.
    """
    return make_decomposable_instance(n=n, card=card, clusters=clusters,
                                      p_in=0.75, p_out=0.0, weight_scale=7.0, pair_scale=5.5, seed=seed)

# ---------------------------- A curated demo that "triggers" each solver ----------------------------

def demo_showcase():
    """
    Runs five regimes where each algorithm is the star:
      1) Greedy-friendly  -> Greedy wins
      2) Local-searchy    -> Local search beats greedy
      3) Tree-structured  -> dp_tree matches exact and is highlighted
      4) Decomposable     -> divide-and-conquer is the natural fit
      5) Dense deceptive  -> branch-and-bound finds best (heuristics stumble)
    We control the *display order* so ties by score still showcase the intended solver.
    """

    # 1) Greedy-friendly
    inst_greedy = make_greedy_easy_instance(n=12, card=(5,5), seed=101)
    run_regime_showcase("Greedy-friendly / Modular", inst_greedy,
                        include=["dp_tree", "divide_and_conquer", "branch_and_bound", "local_search", "greedy"],
                        verify_with_bruteforce=True)

    # 2) Local-search should beat greedy (deterministic)
    inst_ls = make_local_optima_instance(n=18, card=(5, 5), seed=202)
    run_regime_showcase("Local-search-friendly / Bait vs Clique",
                        inst_ls,
                        include=["dp_tree", "divide_and_conquer", "branch_and_bound", "local_search", "greedy"],
                        verify_with_bruteforce=True,ls_restarts=100)


    # 3) Tree-structured (dp_tree exact on forests)
    inst_tree = make_tree_showcase_instance(n=14, card=(4,10), seed=404)
    run_regime_showcase("Tree / DP-exact", inst_tree,
                        include=["dp_tree", "divide_and_conquer", "branch_and_bound", "local_search", "greedy"],
                        verify_with_bruteforce=True)

    # 4) Decomposable (deterministic) — D&C should win
    inst_dc = make_decomposable_instance(n=18, card=(6, 6), seed=505)
    run_regime_showcase("Decomposable / Optimal Cross-Component Merge",
                        inst_dc,
                        include=["dp_tree", "divide_and_conquer", "branch_and_bound", "local_search", "greedy"],
                        verify_with_bruteforce=True)


    # 5) Dense & deceptive (BnB with a slightly higher time limit)
    inst_bnb = make_bnb_showcase_instance(n=20, card=(7,11), seed=303)
    run_regime_showcase("Dense & deceptive / BnB", inst_bnb,
                        include=["dp_tree", "divide_and_conquer", "branch_and_bound", "local_search", "greedy"],
                        bnb_time_limit=4.0,
                        verify_with_bruteforce=True)

# If you prefer this over your previous demo(), call demo_showcase() from main:
# if __name__ == "__main__":
#     demo_showcase()


if __name__ == "__main__":
    demo_showcase()
