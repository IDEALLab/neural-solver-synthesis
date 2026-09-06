import sys
import json
import time
import random

EPS = 1e-12
TIME_LIMIT = 1.82
HEURISTIC_SLACK = 0.13
DFS_NODE_LIMIT = 80000


def tarjan_scc(graph):
    n = len(graph)
    sys.setrecursionlimit(1000000)
    idx = [-1] * n
    low = [0] * n
    onstack = [False] * n
    stack = []
    comp_id = [-1] * n
    comps = []
    timer = 0

    def dfs(v):
        nonlocal timer
        idx[v] = low[v] = timer
        timer += 1
        stack.append(v)
        onstack[v] = True
        for w in graph[v]:
            if idx[w] == -1:
                dfs(w)
                if low[w] < low[v]:
                    low[v] = low[w]
            elif onstack[w] and idx[w] < low[v]:
                low[v] = idx[w]
        if low[v] == idx[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack[w] = False
                comp_id[w] = len(comps)
                comp.append(w)
                if w == v:
                    break
            comps.append(comp)

    for v in range(n):
        if idx[v] == -1:
            dfs(v)
    return comp_id, comps


def solve_sds():
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    req = data.get("requirements", {}) if isinstance(data, dict) else {}
    cat = data.get("catalog", {}) if isinstance(data, dict) else {}

    start = time.perf_counter()
    deadline = start + TIME_LIMIT
    heuristic_deadline = deadline - HEURISTIC_SLACK

    def now():
        return time.perf_counter()

    def check_time():
        if now() > deadline:
            raise TimeoutError

    def check_heuristic_time():
        if now() > heuristic_deadline:
            raise TimeoutError

    n = int(req.get("n_variables", 0) or 0)
    bounds = req.get("cardinality_bounds", [0, n])
    if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
        bounds = [0, n]
    L = max(int(bounds[0]), 0)
    U = min(max(int(bounds[1]), 0), n)
    if U < 0:
        U = 0
    if L > n:
        L = n

    weights = [0.0] * n
    if isinstance(req.get("weights"), list):
        wl = req.get("weights", [])
        for i in range(min(n, len(wl))):
            try:
                weights[i] = float(wl[i])
            except Exception:
                pass
    else:
        for item in cat.get("variables", []) or []:
            try:
                vid = int(item.get("id"))
                if 0 <= vid < n:
                    weights[vid] = float(item.get("weight", 0.0))
            except Exception:
                pass

    if n == 0:
        print(json.dumps({"selection": {"variables": []}}))
        return

    prec_graph = [set() for _ in range(n)]
    for a, b in req.get("precedence", []) or []:
        try:
            a = int(a)
            b = int(b)
            if 0 <= a < n and 0 <= b < n and a != b:
                prec_graph[a].add(b)
        except Exception:
            pass

    conflict_pairs = set()

    def add_conflict_pair(a, b):
        try:
            a = int(a)
            b = int(b)
        except Exception:
            return
        if a == b:
            return
        if a > b:
            a, b = b, a
        conflict_pairs.add((a, b))

    for a, b in req.get("mutex", []) or []:
        add_conflict_pair(a, b)

    for members in (req.get("groups", {}) or {}).values():
        try:
            mem = [int(x) for x in members]
        except Exception:
            continue
        for i in range(len(mem)):
            for j in range(i + 1, len(mem)):
                add_conflict_pair(mem[i], mem[j])

    comp_id, comps = tarjan_scc([list(s) for s in prec_graph])
    m = len(comps)
    if m == 0:
        print(json.dumps({"selection": {"variables": []}}))
        return

    comp_size = [len(c) for c in comps]
    comp_member_mask = [0] * m
    for c, vs in enumerate(comps):
        mask = 0
        for v in vs:
            mask |= 1 << v
        comp_member_mask[c] = mask

    succ = [set() for _ in range(m)]
    pred = [set() for _ in range(m)]
    for a in range(n):
        ca = comp_id[a]
        for b in prec_graph[a]:
            cb = comp_id[b]
            if ca != cb:
                succ[ca].add(cb)
                pred[cb].add(ca)

    indeg = [len(pred[i]) for i in range(m)]
    stack = [i for i in range(m) if indeg[i] == 0]
    topo = []
    while stack:
        v = stack.pop()
        topo.append(v)
        for w in succ[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                stack.append(w)
    if len(topo) != m:
        topo = list(range(m))

    anc = [0] * m
    for v in topo:
        mask = 1 << v
        for p in pred[v]:
            mask |= anc[p]
        anc[v] = mask

    desc = [0] * m
    for v in reversed(topo):
        mask = 1 << v
        for s in succ[v]:
            mask |= desc[s]
        desc[v] = mask

    all_mask = (1 << m) - 1

    comp_weight = [0.0] * m
    for c in range(m):
        s = 0.0
        for v in comps[c]:
            s += weights[v]
        comp_weight[c] = s

    comp_pair = [[0.0] * m for _ in range(m)]
    for k, v in (req.get("interactions", {}) or {}).items():
        try:
            if isinstance(k, str):
                a_s, b_s = k.split(",")
                a = int(a_s.strip())
                b = int(b_s.strip())
            elif isinstance(k, (list, tuple)) and len(k) == 2:
                a = int(k[0])
                b = int(k[1])
            else:
                continue
            if not (0 <= a < n and 0 <= b < n) or a == b:
                continue
            val = float(v)
        except Exception:
            continue
        ca = comp_id[a]
        cb = comp_id[b]
        if ca == cb:
            comp_weight[ca] += val
        else:
            comp_pair[ca][cb] += val
            comp_pair[cb][ca] += val

    comp_conf = [0] * m
    impossible = [False] * m
    for a, b in conflict_pairs:
        if not (0 <= a < n and 0 <= b < n):
            continue
        ca = comp_id[a]
        cb = comp_id[b]
        if ca == cb:
            impossible[ca] = True
        else:
            comp_conf[ca] |= 1 << cb
            comp_conf[cb] |= 1 << ca

    pos_neighbors = [[] for _ in range(m)]
    static_score = [0.0] * m
    for i in range(m):
        s = comp_weight[i]
        row = comp_pair[i]
        for j in range(m):
            val = row[j]
            if val > 0.0:
                pos_neighbors[i].append((j, val))
                s += val
        static_score[i] = s

    size_cache = {0: 0}
    bits_cache = {0: []}

    def mask_size(mask):
        cached = size_cache.get(mask)
        if cached is not None:
            return cached
        s = 0
        tmp = mask
        while tmp:
            b = tmp & -tmp
            i = b.bit_length() - 1
            tmp ^= b
            s += comp_size[i]
        size_cache[mask] = s
        return s

    def bits_list(mask):
        cached = bits_cache.get(mask)
        if cached is not None:
            return cached
        out = []
        tmp = mask
        while tmp:
            b = tmp & -tmp
            i = b.bit_length() - 1
            tmp ^= b
            out.append(i)
        bits_cache[mask] = out
        return out

    def score_of_mask(mask):
        bs = bits_list(mask)
        s = 0.0
        for idx, i in enumerate(bs):
            s += comp_weight[i]
            row = comp_pair[i]
            for j in bs[:idx]:
                s += row[j]
        return s

    def add_delta(add_mask, selected, sel_bits, count):
        new = add_mask & ~selected
        if not new:
            return 0.0
        if count + mask_size(new) > U:
            return None
        combined = selected | new
        new_bits = []
        tmp = new
        while tmp:
            b = tmp & -tmp
            i = b.bit_length() - 1
            tmp ^= b
            if impossible[i] or (comp_conf[i] & combined):
                return None
            new_bits.append(i)
        gain = 0.0
        for a, i in enumerate(new_bits):
            gain += comp_weight[i]
            row = comp_pair[i]
            for j in sel_bits:
                gain += row[j]
            for j in new_bits[:a]:
                gain += row[j]
        return gain

    def remove_delta(rem_mask, selected, sel_bits):
        rem = rem_mask & selected
        if not rem:
            return 0.0
        rem_bits = []
        tmp = rem
        while tmp:
            b = tmp & -tmp
            i = b.bit_length() - 1
            tmp ^= b
            rem_bits.append(i)
        rem_set = set(rem_bits)
        delta = 0.0
        for i in rem_bits:
            delta -= comp_weight[i]
            row = comp_pair[i]
            for j in sel_bits:
                if j not in rem_set:
                    delta -= row[j]
        for a, i in enumerate(rem_bits):
            row = comp_pair[i]
            for j in rem_bits[:a]:
                delta -= row[j]
        return delta

    def future_bonus(new_mask, undec_mask):
        fut = 0.0
        tmp = new_mask
        while tmp:
            b = tmp & -tmp
            i = b.bit_length() - 1
            tmp ^= b
            for j, val in pos_neighbors[i]:
                if (undec_mask >> j) & 1:
                    fut += val
        return fut

    def feasible_mask(mask):
        if mask_size(mask) > U:
            return False
        tmp = mask
        while tmp:
            b = tmp & -tmp
            i = b.bit_length() - 1
            tmp ^= b
            if impossible[i]:
                return False
            if anc[i] & ~mask:
                return False
            if comp_conf[i] & mask:
                return False
        return True

    rng = random.Random(123456789)

    def bundle_candidates(selected, sel_bits, count, max_results=6, local_rng=None):
        undec = all_mask & ~selected
        if undec == 0:
            return []
        candidates = []
        tmp = undec
        while tmp:
            b = tmp & -tmp
            v = b.bit_length() - 1
            tmp ^= b
            new = anc[v] & ~selected
            if not new:
                continue
            sz = mask_size(new)
            if count + sz > U:
                continue
            base = add_delta(new, selected, sel_bits, count)
            if base is None:
                continue
            approx = base + 0.16 * future_bonus(new, undec & ~new) + 0.03 * static_score[v]
            if count < L:
                approx += 0.05 * min(sz, L - count)
            if local_rng is not None:
                approx += (local_rng.random() - 0.5) * 1e-8
            candidates.append((approx, base, new, sz))
        if not candidates:
            return []
        candidates.sort(key=lambda x: (x[0], x[1], x[3]), reverse=True)
        K = 8 if m <= 70 else 6
        K = min(K, len(candidates))
        pool = candidates[:K]
        res = {}

        def put(mask, delta):
            if mask == 0:
                return
            sz = mask_size(mask)
            prev = res.get(mask)
            if prev is None or delta > prev[0] + EPS:
                res[mask] = (delta, sz)

        for _, base, new, _ in pool:
            put(new, base)

        for i in range(K):
            mi = pool[i][2]
            for j in range(i + 1, K):
                mj = pool[j][2]
                mu = mi | mj
                if mu == mi or mu == mj:
                    continue
                if count + mask_size(mu & ~selected) > U:
                    continue
                d = add_delta(mu, selected, sel_bits, count)
                if d is not None:
                    put(mu, d)
                if K >= 3:
                    for k in range(j + 1, K):
                        mk = pool[k][2]
                        m3 = mu | mk
                        if m3 == mu or m3 == mk:
                            continue
                        if count + mask_size(m3 & ~selected) > U:
                            continue
                        d3 = add_delta(m3, selected, sel_bits, count)
                        if d3 is not None:
                            put(m3, d3)

        out = [(d, mask, sz) for mask, (d, sz) in res.items()]
        out.sort(key=lambda x: (x[0], x[2]), reverse=True)
        if max_results is not None:
            out = out[:max_results]
        return out

    def removal_candidates(selected, sel_bits, count, max_results=8):
        out = []
        tmp = selected
        while tmp:
            b = tmp & -tmp
            v = b.bit_length() - 1
            tmp ^= b
            rm = selected & desc[v]
            if not rm:
                continue
            sz = mask_size(rm)
            delta = remove_delta(rm, selected, sel_bits)
            out.append((delta, rm, sz))
        out.sort(key=lambda x: (x[0], -x[2]), reverse=True)
        if max_results is not None:
            out = out[:max_results]
        return out

    best_sel = None
    best_score = -1e300
    best_count = 0

    def consider(sel, sc, cnt):
        nonlocal best_sel, best_score, best_count
        if cnt >= L and sc > best_score + EPS:
            best_sel = sel
            best_score = sc
            best_count = cnt

    def greedy_construct(order, randomize=False):
        sel = 0
        sc = 0.0
        cnt = 0
        sel_bits = []
        for v in order:
            new = anc[v] & ~sel
            if not new:
                continue
            d = add_delta(new, sel, sel_bits, cnt)
            if d is None:
                continue
            if cnt < L or d > EPS:
                sel |= new
                sc += d
                cnt += mask_size(new)
                sel_bits = bits_list(sel)
                if cnt >= U:
                    break
        return sel, sc, cnt

    def beam_construct(seed):
        local_rng = random.Random(seed)
        beam = [(0, 0.0, 0)]
        best_local = None
        best_local_score = -1e300
        best_local_count = 0
        max_depth = 14 if m <= 60 else 10
        beam_width = 12 if m <= 60 else 8

        for _ in range(max_depth):
            if now() > heuristic_deadline:
                break
            nxt = {}
            for sel, sc, cnt in beam:
                sel_bits = bits_list(sel)
                bundles = bundle_candidates(sel, sel_bits, cnt, max_results=5, local_rng=local_rng)
                for d, add_mask, sz in bundles:
                    ncnt = cnt + sz
                    if ncnt > U:
                        continue
                    nsel = sel | add_mask
                    nsc = sc + d
                    prev = nxt.get(nsel)
                    if prev is None or nsc > prev[1] + EPS:
                        nxt[nsel] = (nsel, nsc, ncnt)
            if not nxt:
                break
            cand = list(nxt.values())
            cand.sort(key=lambda x: (x[1] + (0.08 * x[2] if x[2] < L else 0.0), x[2]), reverse=True)
            beam = cand[:beam_width]
            for sel, sc, cnt in beam:
                if cnt >= L and sc > best_local_score + EPS:
                    best_local = sel
                    best_local_score = sc
                    best_local_count = cnt
        if best_local is not None:
            return best_local, best_local_score, best_local_count
        return beam[0]

    def local_search(sel, sc, cnt):
        max_steps = 80 if m <= 80 else 55
        kick_budget = 3
        steps = 0
        kicks = 0
        while steps < max_steps and now() < heuristic_deadline:
            check_heuristic_time()
            sel_bits = bits_list(sel)
            bundles = bundle_candidates(sel, sel_bits, cnt, max_results=6, local_rng=rng)
            rems = removal_candidates(sel, sel_bits, cnt, max_results=7)

            best_delta = -1e300
            best_action = None

            if bundles:
                d, am, asz = bundles[0]
                if cnt + asz <= U and (cnt < L or d > EPS):
                    best_delta = d
                    best_action = ("add", am, d, asz)

            if cnt > L and rems:
                for d, rm, rsz in rems:
                    if cnt - rsz >= L and d > best_delta + EPS and d > EPS:
                        best_delta = d
                        best_action = ("rem", rm, d, rsz)
                        break

            if rems:
                rem_pool = min(4, len(rems))
                for i in range(rem_pool):
                    rd1, rm1, rsz1 = rems[i]
                    for j in range(i + 1, rem_pool):
                        rd2, rm2, rsz2 = rems[j]
                        rm12 = rm1 | rm2
                        rdel = remove_delta(rm12, sel, sel_bits)
                        cnt2 = cnt - mask_size(rm12)
                        if cnt2 > U:
                            continue
                        sel2 = sel & ~rm12
                        bits2 = bits_list(sel2)
                        b2 = bundle_candidates(sel2, bits2, cnt2, max_results=4, local_rng=rng)
                        for ad, am, asz in b2:
                            total = rdel + ad
                            ncnt = cnt2 + asz
                            if ncnt < L or ncnt > U:
                                continue
                            if total > best_delta + EPS:
                                best_delta = total
                                best_action = ("swap", rm12, am, rdel, ad, cnt2, asz)

            if best_action is not None:
                kind = best_action[0]
                if kind == "add":
                    am = best_action[1]
                    d = best_action[2]
                    sz = best_action[3]
                    sel = sel | am
                    sc += d
                    cnt += sz
                elif kind == "rem":
                    rm = best_action[1]
                    d = best_action[2]
                    sz = best_action[3]
                    sel = sel & ~rm
                    sc += d
                    cnt -= sz
                else:
                    rm = best_action[1]
                    am = best_action[2]
                    rd = best_action[3]
                    ad = best_action[4]
                    rsz = mask_size(rm)
                    asz = best_action[6]
                    sel = (sel & ~rm) | am
                    sc += rd + ad
                    cnt = cnt - rsz + asz
                steps += 1
                kicks = 0
                if cnt >= L:
                    consider(sel, sc, cnt)
                continue

            if cnt >= L and kicks < kick_budget and rems:
                chosen = None
                lim = min(3, len(rems))
                for idx in range(lim):
                    d, rm, rsz = rems[idx]
                    if cnt - rsz >= L:
                        chosen = (d, rm, rsz)
                        break
                if chosen is None:
                    break
                d, rm, rsz = chosen
                sel = sel & ~rm
                sc += d
                cnt -= rsz
                steps += 1
                kicks += 1
                if cnt >= L:
                    consider(sel, sc, cnt)
                continue

            if cnt < L and bundles:
                d, am, asz = bundles[0]
                if cnt + asz <= U:
                    sel = sel | am
                    sc += d
                    cnt += asz
                    steps += 1
                    if cnt >= L:
                        consider(sel, sc, cnt)
                    continue

            break

        if cnt >= L:
            consider(sel, sc, cnt)
        return sel, sc, cnt

    root_order = sorted(
        range(m),
        key=lambda v: (
            static_score[v] / max(1, comp_size[v]),
            static_score[v],
            -comp_size[v],
        ),
        reverse=True,
    )
    degree_order = sorted(
        range(m),
        key=lambda v: (
            static_score[v] + 0.35 * sum(max(0.0, comp_pair[v][j]) for j in range(m)),
            static_score[v],
            -comp_size[v],
        ),
        reverse=True,
    )

    def eval_seed(sel):
        if now() > heuristic_deadline:
            return
        if not feasible_mask(sel):
            return
        sc = score_of_mask(sel)
        cnt = mask_size(sel)
        try:
            sel2, sc2, cnt2 = local_search(sel, sc, cnt)
            if cnt2 >= L:
                consider(sel2, sc2, cnt2)
        except TimeoutError:
            pass

    try:
        if L == 0:
            consider(0, 0.0, 0)

        eval_seed(0)
        eval_seed(greedy_construct(root_order)[0])
        eval_seed(greedy_construct(degree_order)[0])

        noisy = root_order[:]
        rng.shuffle(noisy)
        eval_seed(greedy_construct(noisy)[0])

        for seed in (101, 202, 303, 404):
            if now() > heuristic_deadline:
                break
            bsel, bsc, bcnt = beam_construct(seed)
            try:
                sel2, sc2, cnt2 = local_search(bsel, bsc, bcnt)
                if cnt2 >= L:
                    consider(sel2, sc2, cnt2)
            except TimeoutError:
                pass

        if best_sel is not None:
            for t in range(6):
                if now() > heuristic_deadline:
                    break
                cur_sel = best_sel
                cur_sc = best_score
                cur_cnt = best_count
                try:
                    rems = removal_candidates(cur_sel, bits_list(cur_sel), cur_cnt, max_results=6)
                    if rems:
                        for step in range(1 + (t & 1)):
                            if not rems:
                                break
                            choice = rems[min(len(rems) - 1, (t + step) % len(rems))]
                            d, rm, rsz = choice
                            if cur_cnt - rsz < L:
                                break
                            cur_sel &= ~rm
                            cur_sc += d
                            cur_cnt -= rsz
                    cur_sel, cur_sc, cur_cnt = local_search(cur_sel, cur_sc, cur_cnt)
                    if cur_cnt >= L:
                        consider(cur_sel, cur_sc, cur_cnt)
                except TimeoutError:
                    pass

        if best_sel is None and L == 0:
            best_sel = 0
            best_score = 0.0
            best_count = 0

    except TimeoutError:
        pass

    def can_add_mask(add_mask, selected, count):
        if count + mask_size(add_mask) > U:
            return False
        selected2 = selected | add_mask
        tmp = add_mask
        while tmp:
            b = tmp & -tmp
            i = b.bit_length() - 1
            tmp ^= b
            if impossible[i]:
                return False
            if anc[i] & ~selected2:
                return False
            if comp_conf[i] & selected2:
                return False
        return True

    def select_mask(add_mask, selected, forbidden, score, count):
        new = add_mask & ~selected
        if not new:
            return selected, forbidden, score, count, True
        if count + mask_size(new) > U:
            return selected, forbidden, score, count, False
        if new & forbidden:
            return selected, forbidden, score, count, False
        sel_bits = bits_list(selected)
        gain = add_delta(new, selected, sel_bits, count)
        if gain is None:
            return selected, forbidden, score, count, False

        selected2 = selected | new
        if selected2 & forbidden:
            return selected, forbidden, score, count, False

        conf = 0
        tmp = new
        while tmp:
            b = tmp & -tmp
            i = b.bit_length() - 1
            tmp ^= b
            if impossible[i]:
                return selected, forbidden, score, count, False
            conf |= comp_conf[i]

        forbidden2 = forbidden
        tmp = conf & ~selected2
        while tmp:
            b = tmp & -tmp
            i = b.bit_length() - 1
            tmp ^= b
            if desc[i] & selected2:
                return selected, forbidden, score, count, False
            forbidden2 |= desc[i]

        if selected2 & forbidden2:
            return selected, forbidden, score, count, False

        return selected2, forbidden2, score + gain, count + mask_size(new), True

    def forbid_component(v, selected, forbidden):
        mask = desc[v]
        if selected & mask:
            return selected, forbidden, False
        forbidden2 = forbidden | mask
        if selected & forbidden2:
            return selected, forbidden, False
        return selected, forbidden2, True

    def propagate(selected, forbidden, score, count):
        while True:
            check_time()
            undec = all_mask & ~(selected | forbidden)
            if not undec:
                return selected, forbidden, score, count, True
            undec_size = mask_size(undec)
            if count + undec_size < L:
                return selected, forbidden, score, count, False
            if count + undec_size == L:
                selected, forbidden, score, count, ok = select_mask(undec, selected, forbidden, score, count)
                if not ok:
                    return selected, forbidden, score, count, False
                continue

            changed = False
            tmp = undec
            while tmp:
                b = tmp & -tmp
                v = b.bit_length() - 1
                tmp ^= b

                need = anc[v] & ~selected
                if (
                    impossible[v]
                    or (comp_conf[v] & selected)
                    or (need & forbidden)
                    or count + mask_size(need) > U
                    or not can_add_mask(need, selected, count)
                ):
                    _, forb2, ok = forbid_component(v, selected, forbidden)
                    if not ok:
                        return selected, forbidden, score, count, False
                    if forb2 != forbidden:
                        forbidden = forb2
                        changed = True
                        break

            if not changed:
                return selected, forbidden, score, count, True

    def package_potential(v, selected, forbidden, undec, count):
        new = anc[v] & ~selected
        if not new or (new & forbidden):
            return -1e100, 0
        sz = mask_size(new)
        if count + sz > U:
            return -1e100, 0
        sel_bits = bits_list(selected)
        gain = add_delta(new, selected, sel_bits, count)
        if gain is None:
            return -1e100, 0
        fut = future_bonus(new, undec & ~new)
        pot = gain + 0.14 * fut + 0.02 * static_score[v]
        return pot, sz

    if m <= 52 and now() < deadline - 0.04:
        try:
            if best_sel is None and L == 0:
                best_sel = 0
                best_score = 0.0
                best_count = 0

            seen_states = set()
            node_counter = 0

            def upper_bound(selected, forbidden, score, count):
                undec = all_mask & ~(selected | forbidden)
                if not undec:
                    return score
                rem_slots = min(undec.bit_count(), U - count)
                if rem_slots <= 0:
                    return score
                pots = []
                sel_bits = bits_list(selected)
                tmp = undec
                while tmp:
                    b = tmp & -tmp
                    v = b.bit_length() - 1
                    tmp ^= b
                    new = anc[v] & ~selected
                    if not new or (new & forbidden):
                        continue
                    sz = mask_size(new)
                    if count + sz > U:
                        continue
                    gain = add_delta(new, selected, sel_bits, count)
                    if gain is None:
                        continue
                    if gain > 0.0:
                        fut = future_bonus(new, undec & ~new)
                        pots.append(gain + 0.12 * fut + 0.02 * static_score[v])
                if not pots:
                    return score
                pots.sort(reverse=True)
                return score + sum(pots[:rem_slots])

            def dfs(selected, forbidden, score, count):
                nonlocal best_sel, best_score, best_count, node_counter
                if node_counter >= DFS_NODE_LIMIT:
                    return
                node_counter += 1
                check_time()

                selected, forbidden, score, count, ok = propagate(selected, forbidden, score, count)
                if not ok:
                    return

                key = (selected, forbidden)
                if key in seen_states:
                    return
                seen_states.add(key)

                if count >= L and score > best_score + EPS:
                    best_sel = selected
                    best_score = score
                    best_count = count

                undec = all_mask & ~(selected | forbidden)
                if not undec or count == U:
                    return

                ub = upper_bound(selected, forbidden, score, count)
                if ub <= best_score + EPS:
                    return

                best_v = None
                best_pot = -1e100
                best_need = 0

                tmp = undec
                while tmp:
                    b = tmp & -tmp
                    v = b.bit_length() - 1
                    tmp ^= b
                    pot, _ = package_potential(v, selected, forbidden, undec, count)
                    if pot > best_pot + EPS:
                        best_pot = pot
                        best_v = v
                        best_need = anc[v] & ~selected

                if best_v is None:
                    return

                ns, nf, nsc, ncnt, ok = select_mask(best_need, selected, forbidden, score, count)
                if ok:
                    dfs(ns, nf, nsc, ncnt)

                ns, nf, ok = forbid_component(best_v, selected, forbidden)
                if ok:
                    dfs(ns, nf, score, count)

            dfs(0, 0, 0.0, 0)
        except TimeoutError:
            pass

    if best_sel is None:
        best_sel = 0 if L == 0 else None

    if best_sel is None or not feasible_mask(best_sel):
        try:
            fb_sel, fb_sc, fb_cnt = beam_construct(404)
            fb_sel, fb_sc, fb_cnt = local_search(fb_sel, fb_sc, fb_cnt)
            if fb_cnt >= L and feasible_mask(fb_sel):
                best_sel = fb_sel
        except Exception:
            pass

    if best_sel is None or not feasible_mask(best_sel):
        best_sel = 0 if L == 0 else 0

    out = []
    tmp = best_sel
    while tmp:
        b = tmp & -tmp
        c = b.bit_length() - 1
        tmp ^= b
        out.extend(comps[c])
    out = sorted(set(int(x) for x in out))

    print(json.dumps({"selection": {"variables": out}}))


if __name__ == "__main__":
    solve_sds()
