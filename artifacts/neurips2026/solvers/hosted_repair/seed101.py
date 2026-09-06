import sys
import json
import time
import heapq
import random

EPS = 1e-10


class State:
    __slots__ = ("score", "count", "cmask", "vmask", "sel", "seq", "pos", "succ_cnt", "tot")

    def __init__(self, m):
        self.score = 0.0
        self.count = 0
        self.cmask = 0
        self.vmask = 0
        self.sel = [False] * m
        self.seq = []
        self.pos = [-1] * m
        self.succ_cnt = [0] * m
        self.tot = [0.0] * m


def solve_sds():
    data = json.load(sys.stdin)
    req = data.get("requirements", data)
    cat = data.get("catalog", {})

    n = int(req.get("n_variables", 0))
    min_k, max_k = map(int, req.get("cardinality_bounds", [0, n]))
    precedence = req.get("precedence", [])
    mutex = req.get("mutex", [])
    groups = req.get("groups", {})

    if n <= 0:
        sys.stdout.write(json.dumps({"selection": {"variables": []}}, separators=(",", ":")))
        return

    start = time.perf_counter()
    deadline = start + 2.45
    rng = random.Random(0x5D5D5D)
    sys.setrecursionlimit(1000000)

    # ----------------------------
    # Parse weights / interactions
    # ----------------------------
    weights = [0.0] * n
    if "weights" in req:
        wsrc = req["weights"]
        if isinstance(wsrc, dict):
            for k, v in wsrc.items():
                try:
                    i = int(k)
                except Exception:
                    continue
                if 0 <= i < n:
                    weights[i] = float(v)
        else:
            for i, v in enumerate(wsrc):
                if i < n:
                    weights[i] = float(v)
    elif "variables" in cat:
        for v in cat.get("variables", []):
            try:
                vid = int(v.get("id"))
            except Exception:
                continue
            if 0 <= vid < n:
                weights[vid] = float(v.get("weight", 0.0))

    def parse_weight_value(val):
        if isinstance(val, dict):
            if "w" in val:
                return float(val["w"])
            if "weight" in val:
                return float(val["weight"])
        return float(val)

    def iter_interactions(src):
        if not src:
            return
        if isinstance(src, dict):
            for k, v in src.items():
                i = j = None
                if isinstance(k, str):
                    parts = k.split(",")
                    if len(parts) >= 2:
                        try:
                            i = int(parts[0].strip())
                            j = int(parts[1].strip())
                        except Exception:
                            continue
                elif isinstance(k, (list, tuple)) and len(k) >= 2:
                    try:
                        i = int(k[0])
                        j = int(k[1])
                    except Exception:
                        continue
                if i is not None and j is not None:
                    try:
                        yield i, j, parse_weight_value(v)
                    except Exception:
                        continue
        elif isinstance(src, list):
            for item in src:
                if isinstance(item, dict):
                    if "i" in item and "j" in item:
                        try:
                            i = int(item["i"])
                            j = int(item["j"])
                            w = parse_weight_value(item.get("w", item.get("weight", 0.0)))
                            yield i, j, w
                        except Exception:
                            continue
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    try:
                        yield int(item[0]), int(item[1]), parse_weight_value(item[2])
                    except Exception:
                        continue

    interactions_src = cat.get("interactions", req.get("interactions", {}))

    # ----------------------------
    # Build variable-level graphs
    # ----------------------------
    node_bit = [1 << i for i in range(n)]
    succ = [[] for _ in range(n)]
    pred = [[] for _ in range(n)]
    conflict_mask = [0] * n
    var_pred_bits = [0] * n

    for a, b in precedence:
        try:
            a = int(a)
            b = int(b)
        except Exception:
            continue
        if 0 <= a < n and 0 <= b < n and a != b:
            succ[a].append(b)
            pred[b].append(a)
            var_pred_bits[b] |= node_bit[a]

    for a, b in mutex:
        try:
            a = int(a)
            b = int(b)
        except Exception:
            continue
        if 0 <= a < n and 0 <= b < n and a != b:
            conflict_mask[a] |= node_bit[b]
            conflict_mask[b] |= node_bit[a]

    group_iter = groups.values() if isinstance(groups, dict) else groups
    for gvars in group_iter:
        seen = set()
        uniq = []
        for x in gvars:
            try:
                x = int(x)
            except Exception:
                continue
            if 0 <= x < n and x not in seen:
                seen.add(x)
                uniq.append(x)
        if len(uniq) > 1:
            msk = 0
            for x in uniq:
                msk |= node_bit[x]
            for x in uniq:
                conflict_mask[x] |= (msk ^ node_bit[x])

    # ----------------------------
    # SCC condensation on precedence
    # ----------------------------
    visited = [False] * n
    order = []
    for s in range(n):
        if visited[s]:
            continue
        visited[s] = True
        stack = [(s, 0)]
        while stack:
            v, idx = stack[-1]
            if idx < len(succ[v]):
                nx = succ[v][idx]
                stack[-1] = (v, idx + 1)
                if not visited[nx]:
                    visited[nx] = True
                    stack.append((nx, 0))
            else:
                order.append(v)
                stack.pop()

    comp_id = [-1] * n
    comps = []
    for v in reversed(order):
        if comp_id[v] != -1:
            continue
        cid = len(comps)
        comps.append([])
        stack = [v]
        comp_id[v] = cid
        while stack:
            x = stack.pop()
            comps[cid].append(x)
            for p in pred[x]:
                if comp_id[p] == -1:
                    comp_id[p] = cid
                    stack.append(p)

    m = len(comps)
    if m == 0:
        sys.stdout.write(json.dumps({"selection": {"variables": []}}, separators=(",", ":")))
        return

    comp_var_mask = [0] * m
    comp_size = [0] * m
    comp_base = [0.0] * m
    comp_invalid = [False] * m

    for cid, members in enumerate(comps):
        vm = 0
        bad = False
        base = 0.0
        for v in members:
            if conflict_mask[v] & vm:
                bad = True
            vm |= node_bit[v]
            base += weights[v]
        comp_var_mask[cid] = vm
        comp_size[cid] = len(members)
        comp_base[cid] = base
        if comp_size[cid] > max_k:
            bad = True
        comp_invalid[cid] = bad

    comp_adj = [dict() for _ in range(m)]
    comp_pred_sets = [set() for _ in range(m)]
    comp_succ_sets = [set() for _ in range(m)]

    for a, b, w in iter_interactions(interactions_src):
        if not (0 <= a < n and 0 <= b < n):
            continue
        ca = comp_id[a]
        cb = comp_id[b]
        if ca == cb:
            if a != b:
                comp_base[ca] += w
            continue
        comp_adj[ca][cb] = comp_adj[ca].get(cb, 0.0) + w
        comp_adj[cb][ca] = comp_adj[cb].get(ca, 0.0) + w

    for a, b in precedence:
        try:
            a = int(a)
            b = int(b)
        except Exception:
            continue
        if 0 <= a < n and 0 <= b < n:
            ca = comp_id[a]
            cb = comp_id[b]
            if ca != cb:
                comp_succ_sets[ca].add(cb)
                comp_pred_sets[cb].add(ca)

    comp_succ = [list(s) for s in comp_succ_sets]
    comp_pred = [list(s) for s in comp_pred_sets]

    # Topological order of condensed DAG
    indeg = [len(comp_pred[c]) for c in range(m)]
    heap = []
    for c in range(m):
        if indeg[c] == 0:
            heapq.heappush(heap, (-comp_size[c], -comp_base[c], c))
    topo = []
    while heap:
        _, _, c = heapq.heappop(heap)
        topo.append(c)
        for nx in comp_succ[c]:
            indeg[nx] -= 1
            if indeg[nx] == 0:
                heapq.heappush(heap, (-comp_size[nx], -comp_base[nx], nx))
    if len(topo) != m:
        topo = list(range(m))

    comp_pred_bits = [0] * m
    comp_succ_bits = [0] * m
    for c in range(m):
        pb = 0
        sb = 0
        for p in comp_pred[c]:
            pb |= (1 << p)
        for s in comp_succ[c]:
            sb |= (1 << s)
        comp_pred_bits[c] = pb
        comp_succ_bits[c] = sb

    anc_bits = [0] * m
    for c in topo:
        bits = 0
        for p in comp_pred[c]:
            bits |= anc_bits[p] | (1 << p)
        anc_bits[c] = bits

    comp_conf_bits = [0] * m
    for c, members in enumerate(comps):
        bits = 0
        for v in members:
            tmp = conflict_mask[v]
            while tmp:
                b = tmp & -tmp
                u = b.bit_length() - 1
                tmp -= b
                cu = comp_id[u]
                if cu != c:
                    bits |= (1 << cu)
        comp_conf_bits[c] = bits

    # ----------------------------
    # Static potentials / rankings
    # ----------------------------
    direct_pot = [0.0] * m
    pos_deg = [0.0] * m
    for c in range(m):
        if comp_invalid[c]:
            direct_pot[c] = -1e100
            pos_deg[c] = -1e100
            continue
        g = comp_base[c]
        pd = 0.0
        for w in comp_adj[c].values():
            if w > 0.0:
                g += w
                pd += w
        direct_pot[c] = g
        pos_deg[c] = pd

    succ_ranked = [[] for _ in range(m)]
    for c in range(m):
        succ_ranked[c] = sorted(comp_succ[c], key=lambda s: (direct_pot[s], comp_size[s], -s), reverse=True)

    tail_pot = [0.0] * m
    for c in reversed(topo):
        if comp_invalid[c]:
            tail_pot[c] = -1e100
            continue
        bonus = 0.0
        taken = 0
        for s in succ_ranked[c]:
            if tail_pot[s] > 0.0:
                bonus += tail_pot[s]
                taken += 1
                if taken >= 3:
                    break
        tail_pot[c] = direct_pot[c] + 0.30 * bonus

    valid_comp = [not comp_invalid[c] for c in range(m)]

    def unlock_pot_for_add(st, c):
        new_mask = st.cmask | (1 << c)
        val = 0.0
        taken = 0
        for s in succ_ranked[c]:
            if st.sel[s]:
                continue
            if comp_pred_bits[s] & ~new_mask:
                continue
            if comp_conf_bits[s] & new_mask:
                continue
            if st.count + comp_size[s] > max_k:
                continue
            p = tail_pot[s]
            if p > 0.0:
                val += p
                taken += 1
                if taken >= 3:
                    break
        return val

    cand_order = [c for c in range(m) if valid_comp[c]]
    cand_order.sort(
        key=lambda c: (
            tail_pot[c],
            direct_pot[c],
            pos_deg[c],
            -comp_size[c],
            -c,
        ),
        reverse=True,
    )

    # ----------------------------
    # Validation / scoring helpers
    # ----------------------------
    def validate_state(st):
        if st is None:
            return False
        if not (min_k <= st.count <= max_k):
            return False
        mask = st.vmask
        tmp = mask
        while tmp:
            b = tmp & -tmp
            v = b.bit_length() - 1
            tmp -= b
            if conflict_mask[v] & mask:
                return False
            if var_pred_bits[v] & ~mask:
                return False
        return True

    def validate_var_mask(mask):
        cnt = mask.bit_count()
        if cnt < min_k or cnt > max_k:
            return False
        tmp = mask
        while tmp:
            b = tmp & -tmp
            v = b.bit_length() - 1
            tmp -= b
            if conflict_mask[v] & mask:
                return False
            if var_pred_bits[v] & ~mask:
                return False
        return True

    def build_state_from_seq(seq):
        st = State(m)
        for c in seq:
            if c < 0 or c >= m or st.sel[c] or comp_invalid[c]:
                return None
            if comp_pred_bits[c] & ~st.cmask:
                return None
            if comp_conf_bits[c] & st.cmask:
                return None
            if st.count + comp_size[c] > max_k:
                return None
            d = comp_base[c] + st.tot[c]
            st.score += d
            st.count += comp_size[c]
            st.cmask |= (1 << c)
            st.vmask |= comp_var_mask[c]
            st.sel[c] = True
            st.pos[c] = len(st.seq)
            st.seq.append(c)
            for p in comp_pred[c]:
                if st.sel[p]:
                    st.succ_cnt[p] += 1
            for nb, w in comp_adj[c].items():
                st.tot[nb] += w
        return st

    def clone_state(st):
        ns = State(m)
        ns.score = st.score
        ns.count = st.count
        ns.cmask = st.cmask
        ns.vmask = st.vmask
        ns.sel = st.sel[:]
        ns.seq = st.seq[:]
        ns.pos = st.pos[:]
        ns.succ_cnt = st.succ_cnt[:]
        ns.tot = st.tot[:]
        return ns

    def apply_add(st, c):
        d = comp_base[c] + st.tot[c]
        st.score += d
        st.count += comp_size[c]
        st.cmask |= (1 << c)
        st.vmask |= comp_var_mask[c]
        st.sel[c] = True
        st.pos[c] = len(st.seq)
        st.seq.append(c)
        for p in comp_pred[c]:
            if st.sel[p]:
                st.succ_cnt[p] += 1
        for nb, w in comp_adj[c].items():
            st.tot[nb] += w
        return d

    def apply_remove(st, c):
        d = -(comp_base[c] + st.tot[c])
        st.score += d
        st.count -= comp_size[c]
        st.cmask &= ~(1 << c)
        st.vmask ^= comp_var_mask[c]
        st.sel[c] = False
        idx = st.pos[c]
        last = st.seq[-1]
        st.seq[idx] = last
        st.pos[last] = idx
        st.seq.pop()
        st.pos[c] = -1
        for p in comp_pred[c]:
            if st.sel[p]:
                st.succ_cnt[p] -= 1
        for nb, w in comp_adj[c].items():
            st.tot[nb] -= w
        return d

    def feasible_add(st, c):
        if st.sel[c]:
            return False
        if comp_pred_bits[c] & ~st.cmask:
            return False
        if comp_conf_bits[c] & st.cmask:
            return False
        if st.count + comp_size[c] > max_k:
            return False
        return True

    def comp_gain_against_mask(c, sel_mask):
        g = comp_base[c]
        tmp = sel_mask
        while tmp:
            b = tmp & -tmp
            s = b.bit_length() - 1
            tmp -= b
            g += comp_adj[c].get(s, 0.0)
        return g

    # ----------------------------
    # Exact solver for small ancestor-closed subsets
    # ----------------------------
    def exact_solve_subset(subset_list):
        if not subset_list:
            return []
        sub_set = set(subset_list)
        sub_topo = [c for c in topo if c in sub_set]
        L = len(sub_topo)
        if L == 0:
            return []

        idx_of = {c: i for i, c in enumerate(sub_topo)}
        sub_base = [comp_base[c] for c in sub_topo]
        sub_size = [comp_size[c] for c in sub_topo]
        sub_pred_bits = [0] * L
        sub_conf_bits = [0] * L
        sub_ub = [0.0] * L

        for i, c in enumerate(sub_topo):
            pb = 0
            cb = 0
            ub = sub_base[i]
            for p in comp_pred[c]:
                if p in sub_set:
                    pb |= (1 << idx_of[p])
            for s in comp_conf_bits[c].bit_length().__class__:
                pass
            sub_pred_bits[i] = pb
            for j, d in enumerate(sub_topo):
                if comp_conf_bits[c] & (1 << d):
                    cb |= (1 << j)
            sub_conf_bits[i] = cb
            for j, d in enumerate(sub_topo):
                if j == i:
                    continue
                w = comp_adj[c].get(d, 0.0)
                if w > 0.0:
                    ub += w
            sub_ub[i] = ub

        rev = list(reversed(range(L)))
        suffix_size = [0] * (L + 1)
        suffix_ub = [0.0] * (L + 1)
        for i in range(L - 1, -1, -1):
            suffix_size[i] = suffix_size[i + 1] + sub_size[rev[i]]
            suffix_ub[i] = suffix_ub[i + 1] + max(0.0, sub_ub[rev[i]])

        best_score = -1e300
        best_mask = 0

        def dfs(i, sel_mask, need_mask, need_size, count, score):
            nonlocal best_score, best_mask
            if time.perf_counter() > deadline - 0.18:
                raise TimeoutError
            if count + need_size > max_k:
                return
            if count + suffix_size[i] < min_k:
                return
            if score + suffix_ub[i] <= best_score + EPS:
                return
            if i == L:
                if need_mask == 0 and min_k <= count <= max_k and score > best_score + EPS:
                    best_score = score
                    best_mask = sel_mask
                return

            cidx = rev[i]
            c = sub_topo[cidx]
            req = (need_mask >> cidx) & 1

            if req:
                if (sub_conf_bits[cidx] & sel_mask) or (count + sub_size[cidx] > max_k):
                    return
                g = sub_base[cidx]
                tmp = sel_mask
                while tmp:
                    b = tmp & -tmp
                    s = b.bit_length() - 1
                    tmp -= b
                    g += comp_adj[c].get(sub_topo[s], 0.0)
                new_need_mask = (need_mask & ~(1 << cidx))
                added_anc = 0
                for p in comp_pred[c]:
                    if p in sub_set:
                        added_anc |= (1 << idx_of[p])
                new_need_mask |= added_anc
                added_size = 0
                tmp = added_anc & ~need_mask
                while tmp:
                    b = tmp & -tmp
                    s = b.bit_length() - 1
                    tmp -= b
                    added_size += sub_size[s]
                dfs(i + 1, sel_mask | (1 << cidx), new_need_mask, need_size - sub_size[cidx] + added_size, count + sub_size[cidx], score + g)
            else:
                dfs(i + 1, sel_mask, need_mask, need_size, count, score)
                if not (sub_conf_bits[cidx] & sel_mask) and count + sub_size[cidx] <= max_k:
                    g = sub_base[cidx]
                    tmp = sel_mask
                    while tmp:
                        b = tmp & -tmp
                        s = b.bit_length() - 1
                        tmp -= b
                        g += comp_adj[c].get(sub_topo[s], 0.0)
                    new_need_mask = need_mask
                    added_anc = 0
                    for p in comp_pred[c]:
                        if p in sub_set:
                            added_anc |= (1 << idx_of[p])
                    new_need_mask |= added_anc
                    added_size = 0
                    tmp = added_anc & ~need_mask
                    while tmp:
                        b = tmp & -tmp
                        s = b.bit_length() - 1
                        tmp -= b
                        added_size += sub_size[s]
                    dfs(i + 1, sel_mask | (1 << cidx), new_need_mask, need_size + added_size, count + sub_size[cidx], score + g)

        try:
            dfs(0, 0, 0, 0, 0, 0.0)
        except TimeoutError:
            pass

        chosen = [sub_topo[i] for i in range(L) if (best_mask >> i) & 1]
        return chosen

    def choose_exact_core(core_limit, ranking):
        core = []
        seen = set()
        for c in ranking:
            if c in seen:
                continue
            seen.add(c)
            core.append(c)
            if len(core) >= core_limit:
                break
        subset_bits = 0
        for c in core:
            subset_bits |= anc_bits[c] | (1 << c)
        subset = [c for c in topo if (subset_bits >> c) & 1]
        return subset

    # ----------------------------
    # Greedy / beam constructors
    # ----------------------------
    def best_add_candidate(st, order=None, positive_only=False):
        if order is None:
            order = cand_order
        best = None
        cmask = st.cmask
        count = st.count
        for c in order:
            bit = 1 << c
            if cmask & bit:
                continue
            if comp_pred_bits[c] & ~cmask:
                continue
            if comp_conf_bits[c] & cmask:
                continue
            if count + comp_size[c] > max_k:
                continue
            d = comp_base[c] + st.tot[c]
            u = unlock_pot_for_add(st, c)
            key = d + 0.22 * tail_pot[c] + 0.14 * u + 0.01 * comp_size[c]
            if positive_only and d <= EPS and key <= EPS:
                continue
            if best is None or key > best[1] + EPS or (abs(key - best[1]) <= EPS and d > best[0] + EPS):
                best = (d, key, c)
        return best

    def best_rem_candidate(st, positive_only=False):
        best = None
        count = st.count
        for c in st.seq:
            if st.succ_cnt[c] != 0:
                continue
            if count - comp_size[c] < min_k:
                continue
            d = -(comp_base[c] + st.tot[c])
            if positive_only and d <= EPS:
                continue
            key = d - 0.02 * tail_pot[c]
            if best is None or key > best[1] + EPS or (abs(key - best[1]) <= EPS and d > best[0] + EPS):
                best = (d, key, c)
        return best

    def best_swap_candidate(st):
        rems = []
        adds = []
        cmask = st.cmask
        count = st.count

        for c in st.seq:
            if st.succ_cnt[c] == 0:
                contrib = comp_base[c] + st.tot[c]
                rems.append((contrib, c))
        rems.sort(key=lambda x: (x[0], x[1]))
        rems = rems[:12]

        for c in cand_order:
            bit = 1 << c
            if cmask & bit:
                continue
            if comp_pred_bits[c] & ~cmask:
                continue
            if comp_conf_bits[c] & cmask:
                continue
            if count + comp_size[c] > max_k:
                continue
            d = comp_base[c] + st.tot[c]
            key = d + 0.22 * tail_pot[c] + 0.14 * unlock_pot_for_add(st, c) + 0.01 * comp_size[c]
            adds.append((d, key, c))
        adds.sort(key=lambda x: (x[1], x[0], x[2]), reverse=True)
        adds = adds[:20]

        best = None
        # 1-remove + 1-add
        for contrib_r, r in rems:
            delta_r = -contrib_r
            cm2 = cmask & ~(1 << r)
            new_count = count - comp_size[r]
            if new_count < min_k:
                continue
            for d_a, _, a in adds:
                if a == r:
                    continue
                if comp_pred_bits[a] & ~cm2:
                    continue
                if comp_conf_bits[a] & cm2:
                    continue
                if new_count + comp_size[a] > max_k:
                    continue
                da = comp_base[a] + st.tot[a] - comp_adj[a].get(r, 0.0)
                delta = delta_r + da
                if best is None or delta > best[0] + EPS:
                    best = (delta, ("swap1", r, a))

        # 2-remove + 1-add
        for i in range(len(rems)):
            contrib_r1, r1 = rems[i]
            for j in range(i + 1, len(rems)):
                contrib_r2, r2 = rems[j]
                delta_r = -contrib_r1 - contrib_r2
                cm2 = cmask & ~(1 << r1) & ~(1 << r2)
                new_count = count - comp_size[r1] - comp_size[r2]
                if new_count < min_k:
                    continue
                for d_a, _, a in adds:
                    if a == r1 or a == r2:
                        continue
                    if comp_pred_bits[a] & ~cm2:
                        continue
                    if comp_conf_bits[a] & cm2:
                        continue
                    if new_count + comp_size[a] > max_k:
                        continue
                    da = comp_base[a] + st.tot[a] - comp_adj[a].get(r1, 0.0) - comp_adj[a].get(r2, 0.0)
                    delta = delta_r + da
                    if best is None or delta > best[0] + EPS:
                        best = (delta, ("swap2", r1, r2, a))

        # 3-remove + 1-add
        if len(rems) >= 3:
            for i in range(min(4, len(rems))):
                c1, r1 = rems[i]
                for j in range(i + 1, min(5, len(rems))):
                    c2, r2 = rems[j]
                    for k in range(j + 1, min(6, len(rems))):
                        c3, r3 = rems[k]
                        delta_r = -c1 - c2 - c3
                        cm2 = cmask & ~(1 << r1) & ~(1 << r2) & ~(1 << r3)
                        new_count = count - comp_size[r1] - comp_size[r2] - comp_size[r3]
                        if new_count < min_k:
                            continue
                        for d_a, _, a in adds[:10]:
                            if a == r1 or a == r2 or a == r3:
                                continue
                            if comp_pred_bits[a] & ~cm2:
                                continue
                            if comp_conf_bits[a] & cm2:
                                continue
                            if new_count + comp_size[a] > max_k:
                                continue
                            da = comp_base[a] + st.tot[a] - comp_adj[a].get(r1, 0.0) - comp_adj[a].get(r2, 0.0) - comp_adj[a].get(r3, 0.0)
                            delta = delta_r + da
                            if best is None or delta > best[0] + EPS:
                                best = (delta, ("swap3", r1, r2, r3, a))
        return best

    def greedy_construct(seed_order=None, allow_negative_after_min=True):
        st = State(m)
        order = seed_order[:] if seed_order is not None else cand_order[:]
        while st.count < max_k and time.perf_counter() < deadline - 0.12:
            best = None
            for c in order:
                if not feasible_add(st, c):
                    continue
                d = comp_base[c] + st.tot[c]
                key = d + 0.22 * tail_pot[c] + 0.14 * unlock_pot_for_add(st, c) + 0.01 * comp_size[c]
                if best is None or key > best[1] + EPS or (abs(key - best[1]) <= EPS and d > best[0] + EPS):
                    best = (d, key, c)
            if best is None:
                break
            d, key, c = best
            if st.count < min_k:
                apply_add(st, c)
                continue
            if d > EPS or key > 0.0:
                apply_add(st, c)
                continue
            if allow_negative_after_min and key > -0.03 * max(1.0, abs(tail_pot[c])):
                apply_add(st, c)
                continue
            break

        while st.count < min_k and time.perf_counter() < deadline - 0.12:
            best = None
            for c in order:
                if not feasible_add(st, c):
                    continue
                d = comp_base[c] + st.tot[c]
                key = d + 0.22 * tail_pot[c] + 0.14 * unlock_pot_for_add(st, c) + 0.01 * comp_size[c]
                if best is None or key > best[1] + EPS or (abs(key - best[1]) <= EPS and d > best[0] + EPS):
                    best = (d, key, c)
            if best is None:
                break
            _, _, c = best
            apply_add(st, c)

        return st if validate_state(st) else None

    def beam_construct(seed_order=None, width=180, branch_limit=14, allow_skip=True):
        order = seed_order[:] if seed_order is not None else cand_order[:]
        frontier = [State(m)]
        best = None

        for _depth in range(max_k):
            if time.perf_counter() > deadline - 0.10:
                break
            next_best = {}
            any_child = False

            for st in frontier:
                if min_k <= st.count <= max_k and validate_state(st):
                    if best is None or st.score > best.score + EPS:
                        best = clone_state(st)

                if st.count >= max_k:
                    continue

                cand_heap = []
                for c in order:
                    if not feasible_add(st, c):
                        continue
                    d = comp_base[c] + st.tot[c]
                    u = unlock_pot_for_add(st, c)
                    key = d + 0.22 * tail_pot[c] + 0.14 * u + 0.01 * comp_size[c]
                    item = (key, d, c)
                    if len(cand_heap) < branch_limit:
                        heapq.heappush(cand_heap, item)
                    else:
                        if key > cand_heap[0][0] + EPS:
                            heapq.heapreplace(cand_heap, item)

                if not cand_heap and allow_skip:
                    continue

                any_child = True
                for key, d, c in sorted(cand_heap, reverse=True):
                    child = clone_state(st)
                    apply_add(child, c)
                    cm = child.cmask
                    prev = next_best.get(cm)
                    if prev is not None and child.score <= prev.score + EPS:
                        continue
                    next_best[cm] = child

            if not next_best:
                break

            frontier = list(next_best.values())
            frontier.sort(key=lambda s: (s.score + 0.02 * s.count, s.score, s.count), reverse=True)
            if len(frontier) > width:
                frontier = frontier[:width]

            for st in frontier:
                if min_k <= st.count <= max_k and validate_state(st):
                    if best is None or st.score > best.score + EPS:
                        best = clone_state(st)

            if not any_child:
                break

        if best is None:
            best = frontier[0] if frontier else State(m)
        return best

    # ----------------------------
    # Local search / repair
    # ----------------------------
    def top_leaves_by_badness(st, k):
        leaves = []
        for c in st.seq:
            if st.succ_cnt[c] == 0:
                contrib = comp_base[c] + st.tot[c]
                leaves.append((contrib, c))
        leaves.sort(key=lambda x: (x[0], x[1]))
        return [c for _, c in leaves[:k]]

    def local_search(st):
        best = clone_state(st)
        if not validate_state(best):
            return best

        current = clone_state(best)
        steps = 0
        stagnation = 0

        while time.perf_counter() < deadline - 0.06 and steps < 8000:
            steps += 1
            changed = False

            if current.count < min_k:
                cand = best_add_candidate(current, positive_only=False)
                if cand is None:
                    break
                _, _, c = cand
                apply_add(current, c)
                changed = True
            else:
                add = best_add_candidate(current, positive_only=False)
                rem = best_rem_candidate(current, positive_only=False)
                choice = None
                if add is not None and rem is not None:
                    if add[0] + 0.18 * tail_pot[add[2]] >= rem[0] - EPS:
                        choice = ("add", add[2], add[0])
                    else:
                        choice = ("rem", rem[2], rem[0])
                elif add is not None:
                    choice = ("add", add[2], add[0])
                elif rem is not None:
                    choice = ("rem", rem[2], rem[0])

                if choice is not None:
                    if choice[0] == "add":
                        if choice[2] > EPS or rng.random() < 0.16:
                            apply_add(current, choice[1])
                            changed = True
                    else:
                        if choice[2] > EPS or rng.random() < 0.08:
                            apply_remove(current, choice[1])
                            changed = True

                if not changed:
                    swap = best_swap_candidate(current)
                    if swap is not None:
                        delta = swap[0]
                        accept = delta > EPS or (delta > -0.35 and rng.random() < 0.14)
                        if accept:
                            tag = swap[1][0]
                            if tag == "swap1":
                                _, r, a = swap[1]
                                apply_remove(current, r)
                                apply_add(current, a)
                                changed = True
                            elif tag == "swap2":
                                _, r1, r2, a = swap[1]
                                apply_remove(current, r1)
                                apply_remove(current, r2)
                                apply_add(current, a)
                                changed = True
                            else:
                                _, r1, r2, r3, a = swap[1]
                                apply_remove(current, r1)
                                apply_remove(current, r2)
                                apply_remove(current, r3)
                                apply_add(current, a)
                                changed = True

            if changed:
                if validate_state(current) and current.score > best.score + EPS:
                    best = clone_state(current)
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1

            if stagnation >= 3:
                break

        return best

    def beam_repair(base_st, width=150, branch_limit=12):
        frontier = [clone_state(base_st)]
        best = clone_state(base_st) if validate_state(base_st) else None
        for _depth in range(max(1, max_k - base_st.count + 1)):
            if time.perf_counter() > deadline - 0.10:
                break
            next_best = {}
            any_child = False
            for st in frontier:
                if min_k <= st.count <= max_k and validate_state(st):
                    if best is None or st.score > best.score + EPS:
                        best = clone_state(st)
                if st.count >= max_k:
                    continue
                cand_heap = []
                for c in cand_order:
                    if not feasible_add(st, c):
                        continue
                    d = comp_base[c] + st.tot[c]
                    u = unlock_pot_for_add(st, c)
                    key = d + 0.22 * tail_pot[c] + 0.14 * u + 0.01 * comp_size[c]
                    item = (key, d, c)
                    if len(cand_heap) < branch_limit:
                        heapq.heappush(cand_heap, item)
                    else:
                        if key > cand_heap[0][0] + EPS:
                            heapq.heapreplace(cand_heap, item)
                if not cand_heap:
                    continue
                any_child = True
                for key, d, c in sorted(cand_heap, reverse=True):
                    child = clone_state(st)
                    apply_add(child, c)
                    cm = child.cmask
                    prev = next_best.get(cm)
                    if prev is not None and child.score <= prev.score + EPS:
                        continue
                    next_best[cm] = child
            if not next_best:
                break
            frontier = list(next_best.values())
            frontier.sort(key=lambda s: (s.score + 0.02 * s.count, s.score, s.count), reverse=True)
            if len(frontier) > width:
                frontier = frontier[:width]
            if not any_child:
                break

        if best is None:
            best = frontier[0] if frontier else base_st
        return best

    def greedy_extend(st, order=None):
        if order is None:
            order = cand_order
        cur = clone_state(st)
        while cur.count < max_k and time.perf_counter() < deadline - 0.10:
            cand = best_add_candidate(cur, order=order, positive_only=False)
            if cand is None:
                break
            d, key, c = cand
            if cur.count < min_k or d > EPS or key > 0.0 or rng.random() < 0.10:
                apply_add(cur, c)
            else:
                break
        if cur.count < min_k:
            cur = greedy_construct(seed_order=order, allow_negative_after_min=True) or cur
        return cur

    def destroy_and_repair(st, drop_count):
        base = clone_state(st)
        if not base.seq:
            return base
        drop_count = max(1, min(drop_count, len(base.seq)))
        for _ in range(drop_count):
            leaves = []
            for c in base.seq:
                if base.succ_cnt[c] == 0:
                    contrib = comp_base[c] + base.tot[c]
                    leaves.append((contrib, c))
            if not leaves:
                break
            leaves.sort(key=lambda x: (x[0], x[1]))
            band = leaves[:min(10, len(leaves))]
            if rng.random() < 0.70:
                _, c = band[0]
            else:
                _, c = band[rng.randrange(len(band))]
            apply_remove(base, c)
        repaired = beam_repair(base, width=120, branch_limit=10)
        repaired = local_search(repaired)
        return repaired

    def path_relink(a, b):
        st = clone_state(a)
        if not validate_state(st):
            return a
        target_seq = b.seq[:]
        target_set = set(target_seq)
        for c in target_seq:
            if st.sel[c]:
                continue
            if feasible_add(st, c):
                apply_add(st, c)
                continue
            guard = 0
            while guard < 6 and not feasible_add(st, c):
                rems = []
                for x in st.seq:
                    if st.succ_cnt[x] == 0 and x not in target_set:
                        rems.append((comp_base[x] + st.tot[x], x))
                if not rems:
                    break
                rems.sort(key=lambda x: (x[0], x[1]))
                apply_remove(st, rems[0][1])
                guard += 1
            if feasible_add(st, c):
                apply_add(st, c)
        st = beam_repair(st, width=100, branch_limit=10)
        st = local_search(st)
        return st

    # ----------------------------
    # Seed generation
    # ----------------------------
    seeds = []

    def add_seed(st):
        if st is not None and validate_state(st):
            seeds.append(st)

    # Exact seeds on reduced cores
    if m <= 38 and time.perf_counter() < deadline - 0.25:
        try:
            exact_all = exact_solve_subset(topo[:])
            if exact_all:
                add_seed(build_state_from_seq(exact_all))
        except Exception:
            pass

    if time.perf_counter() < deadline - 0.25:
        ranking1 = cand_order[:]
        ranking2 = sorted([c for c in range(m) if valid_comp[c]],
                          key=lambda c: (direct_pot[c], tail_pot[c], pos_deg[c], -comp_size[c], -c),
                          reverse=True)
        ranking3 = sorted([c for c in range(m) if valid_comp[c]],
                          key=lambda c: (pos_deg[c], direct_pot[c], tail_pot[c], -comp_size[c], -c),
                          reverse=True)

        for limit, ranking in ((18, ranking1), (16, ranking2), (16, ranking3)):
            if time.perf_counter() > deadline - 0.25:
                break
            subset = choose_exact_core(limit, ranking)
            if 0 < len(subset) <= 42:
                try:
                    chosen = exact_solve_subset(subset)
                    if chosen:
                        add_seed(build_state_from_seq(chosen))
                except Exception:
                    pass

    add_seed(beam_repair(State(m), width=220 if m <= 120 else 160, branch_limit=18 if m <= 120 else 14))
    add_seed(beam_repair(State(m), width=140, branch_limit=12))

    # Greedy starts from several orders
    add_seed(greedy_construct(seed_order=cand_order, allow_negative_after_min=True))
    add_seed(greedy_construct(seed_order=list(reversed(cand_order)), allow_negative_after_min=True))

    if cand_order:
        for shift in (1, 3, 7, 11):
            if time.perf_counter() > deadline - 0.22:
                break
            order = cand_order[shift:] + cand_order[:shift]
            add_seed(greedy_construct(seed_order=order, allow_negative_after_min=True))

    for _ in range(5):
        if time.perf_counter() > deadline - 0.22:
            break
        order = cand_order[:]
        rng.shuffle(order)
        add_seed(greedy_construct(seed_order=order, allow_negative_after_min=True))

    if not seeds:
        seeds.append(State(m))

    elite = []

    def consider_elite(st):
        if st is None or not validate_state(st):
            return
        elite.append(clone_state(st))
        elite.sort(key=lambda s: (s.score, s.count), reverse=True)
        uniq = []
        seen = set()
        for x in elite:
            if x.cmask in seen:
                continue
            seen.add(x.cmask)
            uniq.append(x)
            if len(uniq) >= 8:
                break
        elite[:] = uniq

    for st in seeds:
        consider_elite(st)

    if not elite:
        elite.append(State(m))

    best_state = clone_state(elite[0])

    # ----------------------------
    # Iterated improvement
    # ----------------------------
    rounds = 0
    while time.perf_counter() < deadline - 0.04 and rounds < 18:
        rounds += 1

        snap = elite[:]
        for st in snap:
            if time.perf_counter() > deadline - 0.04:
                break
            loc = local_search(st)
            consider_elite(loc)

        if elite and elite[0].score > best_state.score + EPS:
            best_state = clone_state(elite[0])

        if time.perf_counter() > deadline - 0.04:
            break

        if elite:
            base = elite[0]
            for drop_count in (2 + rounds % 3, 3 + rounds % 4, 4 + rounds % 5):
                if time.perf_counter() > deadline - 0.04:
                    break
                pert = destroy_and_repair(base, drop_count)
                consider_elite(pert)
                if elite and elite[0].score > best_state.score + EPS:
                    best_state = clone_state(elite[0])

        if len(elite) >= 2 and time.perf_counter() < deadline - 0.04:
            top = elite[: min(4, len(elite))]
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    if time.perf_counter() > deadline - 0.04:
                        break
                    p1 = path_relink(top[i], top[j])
                    p2 = path_relink(top[j], top[i])
                    consider_elite(p1)
                    consider_elite(p2)
                    if elite and elite[0].score > best_state.score + EPS:
                        best_state = clone_state(elite[0])

        if elite and elite[0].seq and time.perf_counter() < deadline - 0.04:
            inc = clone_state(elite[0])
            for _ in range(min(3, len(inc.seq))):
                leaves = []
                for c in inc.seq:
                    if inc.succ_cnt[c] == 0:
                        leaves.append((comp_base[c] + inc.tot[c], c))
                if not leaves:
                    break
                leaves.sort(key=lambda x: (x[0], x[1]))
                _, c = leaves[0]
                apply_remove(inc, c)
            inc = beam_repair(inc, width=100, branch_limit=10)
            inc = local_search(inc)
            consider_elite(inc)
            if elite and elite[0].score > best_state.score + EPS:
                best_state = clone_state(elite[0])

    if not elite:
        elite = [State(m)]
    if best_state is None or not validate_state(best_state):
        best_state = clone_state(elite[0])

    # ----------------------------
    # Final safety / repair
    # ----------------------------
    def repair_to_feasible(st):
        if st is None:
            return State(m)
        if validate_state(st):
            return st
        fb = greedy_construct(seed_order=cand_order, allow_negative_after_min=True)
        if fb is not None and validate_state(fb):
            return fb
        # Minimal fallback: build from best sinks if possible.
        tmp = State(m)
        for c in topo:
            if feasible_add(tmp, c):
                apply_add(tmp, c)
                if tmp.count >= min_k:
                    break
        if validate_state(tmp):
            return tmp
        return State(m)

    best_state = repair_to_feasible(best_state)

    if not validate_state(best_state):
        best_state = State(m)

    var_mask = best_state.vmask
    if not validate_var_mask(var_mask):
        rebuilt = build_state_from_seq(best_state.seq)
        if rebuilt is not None and validate_var_mask(rebuilt.vmask):
            best_state = rebuilt
            var_mask = rebuilt.vmask
        else:
            fb = greedy_construct(seed_order=cand_order, allow_negative_after_min=True)
            if fb is not None and validate_state(fb):
                best_state = fb
                var_mask = fb.vmask

    if not validate_var_mask(var_mask):
        # As a last resort, construct a feasible solution from scratch.
        st = greedy_construct(seed_order=cand_order, allow_negative_after_min=True)
        if st is None or not validate_state(st):
            st = State(m)
        if st.count < min_k:
            # Add feasible components greedily until the lower bound is met.
            while st.count < min_k:
                cand = best_add_candidate(st, positive_only=False)
                if cand is None:
                    break
                apply_add(st, cand[2])
        if validate_state(st):
            best_state = st
            var_mask = st.vmask
        else:
            var_mask = 0

    selected = []
    tmp = var_mask
    while tmp:
        b = tmp & -tmp
        v = b.bit_length() - 1
        selected.append(v)
        tmp -= b
    selected.sort()

    sys.stdout.write(json.dumps({"selection": {"variables": selected}}, separators=(",", ":")))


if __name__ == "__main__":
    solve_sds()
