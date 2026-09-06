import json
import math
import random
import sys
from collections import deque

EPS = 1e-12
EXACT_LIMIT = 28
CORE_LIMIT = 24
NODE_BUDGET_SMALL = 500000
NODE_BUDGET_MED = 120000
NODE_BUDGET_LARGE = 30000


def solve_sds():
    sys.setrecursionlimit(1_000_000)
    raw = sys.stdin.buffer.read()
    if not raw.strip():
        sys.stdout.write(json.dumps({"selection": {"variables": []}}))
        return

    data = json.loads(raw.decode("utf-8"))
    req = data.get("requirements", {}) or {}
    cat = data.get("catalog", {}) or {}

    var_list = cat.get("variables", []) or []
    n = int(req.get("n_variables", len(var_list) or 0))
    if n < len(var_list):
        n = len(var_list)

    bounds = req.get("cardinality_bounds", [0, n]) or [0, n]
    min_k = int(bounds[0])
    max_k = int(bounds[1])
    if min_k > max_k:
        min_k, max_k = max_k, min_k
    min_k = max(0, min_k)
    max_k = max(0, max_k)

    # Original ids for output
    orig_ids = [i for i in range(n)]
    for i, v in enumerate(var_list):
        if i >= n:
            break
        if isinstance(v, dict) and "id" in v:
            orig_ids[i] = v["id"]
        else:
            orig_ids[i] = i
    for i in range(len(var_list), n):
        orig_ids[i] = i

    # ---------- Parse weights ----------
    weights = [0.0] * n
    req_w = req.get("weights", None)
    if isinstance(req_w, list) and len(req_w) == n:
        for i, x in enumerate(req_w):
            weights[i] = float(x)
    else:
        for i, v in enumerate(var_list):
            if i >= n:
                break
            if isinstance(v, dict):
                weights[i] = float(v.get("weight", 0.0))

    # ---------- Parse pairwise interactions ----------
    pair = [[0.0] * n for _ in range(n)]

    def add_interaction(i, j, w):
        try:
            i = int(i)
            j = int(j)
            w = float(w)
        except Exception:
            return
        if i == j or not (0 <= i < n and 0 <= j < n):
            return
        if i > j:
            i, j = j, i
        pair[i][j] += w
        pair[j][i] += w

    inter_raw = cat.get("interactions", {})
    if isinstance(inter_raw, dict):
        for key, val in inter_raw.items():
            if isinstance(key, str):
                parts = [p.strip() for p in key.split(",")]
                if len(parts) != 2:
                    continue
                i, j = parts
            else:
                try:
                    i, j = key
                except Exception:
                    continue
            add_interaction(i, j, val)
    elif isinstance(inter_raw, list):
        for it in inter_raw:
            if isinstance(it, dict):
                if "i" in it and "j" in it:
                    add_interaction(it["i"], it["j"], it.get("weight", it.get("w", 0.0)))
                elif "u" in it and "v" in it:
                    add_interaction(it["u"], it["v"], it.get("weight", it.get("w", 0.0)))
                elif "from" in it and "to" in it:
                    add_interaction(it["from"], it["to"], it.get("weight", it.get("w", 0.0)))
            elif isinstance(it, (tuple, list)) and len(it) >= 3:
                add_interaction(it[0], it[1], it[2])

    # ---------- Parse precedence / mutex / groups ----------
    g = [[] for _ in range(n)]
    rg = [[] for _ in range(n)]
    conflict_bits = [0] * n

    def parse_edge(edge):
        if isinstance(edge, dict):
            if "i" in edge and "j" in edge:
                return edge["i"], edge["j"]
            if "u" in edge and "v" in edge:
                return edge["u"], edge["v"]
            if "from" in edge and "to" in edge:
                return edge["from"], edge["to"]
        elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
            return edge[0], edge[1]
        return None

    def add_conflict(a, b):
        try:
            a = int(a)
            b = int(b)
        except Exception:
            return
        if a == b or not (0 <= a < n and 0 <= b < n):
            return
        conflict_bits[a] |= 1 << b
        conflict_bits[b] |= 1 << a

    for edge in req.get("precedence", []) or []:
        p = parse_edge(edge)
        if p is None:
            continue
        a, b = p
        try:
            a = int(a)
            b = int(b)
        except Exception:
            continue
        if a == b or not (0 <= a < n and 0 <= b < n):
            continue
        g[a].append(b)
        rg[b].append(a)

    for edge in req.get("mutex", []) or []:
        p = parse_edge(edge)
        if p is None:
            continue
        add_conflict(p[0], p[1])

    groups = req.get("groups") or {}
    if isinstance(groups, dict):
        group_vals = groups.values()
    else:
        group_vals = groups
    for members in group_vals:
        try:
            ids = [int(x) for x in members]
        except Exception:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                add_conflict(ids[i], ids[j])

    # ---------- SCC condensation on precedence ----------
    vis = [False] * n
    order = []

    def dfs1(v):
        vis[v] = True
        for to in g[v]:
            if not vis[to]:
                dfs1(to)
        order.append(v)

    for i in range(n):
        if not vis[i]:
            dfs1(i)

    comp = [-1] * n
    comps = []

    def dfs2(v, cid):
        comp[v] = cid
        comps[cid].append(v)
        for to in rg[v]:
            if comp[to] == -1:
                dfs2(to, cid)

    for v in reversed(order):
        if comp[v] == -1:
            comps.append([])
            dfs2(v, len(comps) - 1)

    m0 = len(comps)

    atom_members = comps
    atom_orig_mask = [0] * m0
    atom_size = [0] * m0
    atom_weight = [0.0] * m0
    atom_invalid = [False] * m0

    for cid, members in enumerate(atom_members):
        mask = 0
        w = 0.0
        invalid = False
        for u in members:
            mask |= 1 << u
            w += weights[u]
        for i in range(len(members)):
            u = members[i]
            for j in range(i + 1, len(members)):
                v = members[j]
                w += pair[u][v]
                if (conflict_bits[u] >> v) & 1:
                    invalid = True
        atom_orig_mask[cid] = mask
        atom_size[cid] = len(members)
        atom_weight[cid] = w
        atom_invalid[cid] = invalid or atom_size[cid] > max_k

    atom_succ = [0] * m0
    atom_pair = [[0.0] * m0 for _ in range(m0)]
    atom_conf = [0] * m0
    for i in range(n):
        ci = comp[i]
        for j in range(i + 1, n):
            cj = comp[j]
            if ci == cj:
                continue
            w = pair[i][j]
            if w != 0.0:
                atom_pair[ci][cj] += w
                atom_pair[cj][ci] += w
            if (conflict_bits[i] >> j) & 1:
                atom_conf[ci] |= 1 << cj
                atom_conf[cj] |= 1 << ci
        for v in g[i]:
            cv = comp[v]
            if ci != cv:
                atom_succ[ci] |= 1 << cv

    # ---------- Connectivity decomposition ----------
    und_adj = [set() for _ in range(m0)]
    for u in range(m0):
        x = atom_succ[u]
        while x:
            lsb = x & -x
            v = lsb.bit_length() - 1
            und_adj[u].add(v)
            und_adj[v].add(u)
            x ^= lsb
        x = atom_conf[u]
        while x:
            lsb = x & -x
            v = lsb.bit_length() - 1
            und_adj[u].add(v)
            und_adj[v].add(u)
            x ^= lsb
        row = atom_pair[u]
        for v in range(m0):
            if row[v] != 0.0:
                und_adj[u].add(v)
                und_adj[v].add(u)

    seen = [False] * m0
    atom_components = []
    for s in range(m0):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        nodes = []
        while stack:
            u = stack.pop()
            nodes.append(u)
            for v in und_adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        atom_components.append(nodes)

    def bits_iter(mask):
        while mask:
            lsb = mask & -mask
            i = lsb.bit_length() - 1
            yield i
            mask ^= lsb

    def mask_to_global_orig_mask(local_mask, local_orig_masks):
        out = 0
        x = local_mask
        while x:
            lsb = x & -x
            i = lsb.bit_length() - 1
            out |= local_orig_masks[i]
            x ^= lsb
        return out

    def score_mask_local(mask, local_weights, local_pair):
        s = 0.0
        chosen = []
        x = mask
        while x:
            lsb = x & -x
            i = lsb.bit_length() - 1
            s += local_weights[i]
            row = local_pair[i]
            for j in chosen:
                s += row[j]
            chosen.append(i)
            x ^= lsb
        return s

    def topo_closures(local_succ):
        m = len(local_succ)
        indeg = [0] * m
        for u in range(m):
            x = local_succ[u]
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                indeg[v] += 1
                x ^= lsb
        dq = deque([i for i in range(m) if indeg[i] == 0])
        topo = []
        while dq:
            u = dq.popleft()
            topo.append(u)
            x = local_succ[u]
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                indeg[v] -= 1
                if indeg[v] == 0:
                    dq.append(v)
                x ^= lsb
        if len(topo) < m:
            # Fallback if malformed input; keep a stable order.
            topo = list(range(m))
        anc = [0] * m
        for u in topo:
            x = local_succ[u]
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                anc[v] |= anc[u] | (1 << u)
                x ^= lsb
        desc = [0] * m
        for u in reversed(topo):
            x = local_succ[u]
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                desc[u] |= desc[v] | (1 << v)
                x ^= lsb
        return topo, anc, desc

    def exact_profile_local(local_sizes, local_weights, local_orig_masks, local_succ, local_pred, local_conflict, local_pair, local_max_k, start_forb=0):
        m = len(local_sizes)
        if m == 0:
            return [(0, 0.0, 0)]
        topo, anc, desc = topo_closures(local_succ)
        desc_self = [desc[i] | (1 << i) for i in range(m)]

        forb0 = start_forb
        for i in range(m):
            if local_sizes[i] > local_max_k:
                forb0 |= desc_self[i]
            if (local_conflict[i] >> i) & 1:
                forb0 |= desc_self[i]

        pos_sum_total = [0.0] * m
        for i in range(m):
            s = 0.0
            row = local_pair[i]
            for j in range(m):
                if i != j and row[j] > 0.0:
                    s += row[j]
            pos_sum_total[i] = s

        best_score = [-math.inf] * (local_max_k + 1)
        best_mask = [0] * (local_max_k + 1)
        best_score[0] = 0.0
        suffix_best = [-math.inf] * (local_max_k + 1)

        def refresh_suffix():
            mx = -math.inf
            for c in range(local_max_k, -1, -1):
                if best_score[c] > mx:
                    mx = best_score[c]
                suffix_best[c] = mx

        refresh_suffix()

        def record(cnt, score, omask):
            if 0 <= cnt <= local_max_k and score > best_score[cnt] + EPS:
                best_score[cnt] = score
                best_mask[cnt] = omask
                refresh_suffix()

        def avail_list(sel, forb):
            rem = ((1 << m) - 1) & ~(sel | forb)
            out = []
            while rem:
                lsb = rem & -rem
                i = lsb.bit_length() - 1
                rem ^= lsb
                if anc[i] & ~sel:
                    continue
                if local_conflict[i] & sel:
                    continue
                out.append(i)
            return out

        def upper_bound_extra(sel, forb, gain, cnt):
            cap = local_max_k - cnt
            if cap <= 0:
                return 0.0
            rem = ((1 << m) - 1) & ~(sel | forb)
            items = []
            while rem:
                lsb = rem & -rem
                i = lsb.bit_length() - 1
                rem ^= lsb
                val = max(0.0, gain[i]) + pos_sum_total[i]
                if val > 0.0:
                    items.append((val, local_sizes[i]))
            if not items:
                return 0.0
            items.sort(key=lambda t: t[0] / t[1], reverse=True)
            extra = 0.0
            for val, sz in items:
                if cap <= 0:
                    break
                if sz <= cap:
                    extra += val
                    cap -= sz
                else:
                    extra += val * (cap / sz)
                    break
            return extra

        node_budget = NODE_BUDGET_SMALL if m <= 26 else NODE_BUDGET_MED if m <= 45 else NODE_BUDGET_LARGE
        if m <= EXACT_LIMIT:
            node_budget = max(node_budget, 500000)

        # Greedy warm starts
        for seed, cap in [(11, local_max_k), (17, local_max_k), (31, min(local_max_k, 8))]:
            rng = random.Random(seed)
            sel = 0
            forb = forb0
            cnt = 0
            score = 0.0
            omask = 0
            gain = local_weights[:]
            record(cnt, score, omask)
            while cnt < cap and cnt < local_max_k:
                cand = []
                rem = ((1 << m) - 1) & ~(sel | forb)
                while rem:
                    lsb = rem & -rem
                    i = lsb.bit_length() - 1
                    rem ^= lsb
                    if anc[i] & ~sel:
                        continue
                    if local_conflict[i] & sel:
                        continue
                    if cnt + local_sizes[i] > local_max_k:
                        continue
                    val = (gain[i] + 0.35 * pos_sum_total[i]) / max(1, local_sizes[i]) + rng.random() * 1e-9
                    cand.append((val, i))
                if not cand:
                    break
                cand.sort(reverse=True)
                top = cand[: min(5, len(cand))]
                i = top[0][1] if len(top) == 1 or rng.random() < 0.7 else top[rng.randrange(len(top))][1]
                if cnt + local_sizes[i] > local_max_k:
                    break
                sel |= 1 << i
                x = local_conflict[i]
                while x:
                    lsb = x & -x
                    j = lsb.bit_length() - 1
                    forb |= desc_self[j]
                    x ^= lsb
                cnt += local_sizes[i]
                score += gain[i]
                omask |= local_orig_masks[i]
                row = local_pair[i]
                for t in range(m):
                    gain[t] += row[t]
                record(cnt, score, omask)

        seen = {}
        node_visits = 0

        def dfs(sel, forb, cnt, score, omask, gain):
            nonlocal node_visits
            if node_visits >= node_budget:
                return
            node_visits += 1

            record(cnt, score, omask)

            key = (sel, forb)
            prev = seen.get(key)
            if prev is not None and score <= prev + EPS:
                return
            seen[key] = score

            ub = score + upper_bound_extra(sel, forb, gain, cnt)
            if ub <= suffix_best[cnt] + EPS:
                return

            avail = avail_list(sel, forb)
            if not avail:
                return

            fit = [i for i in avail if cnt + local_sizes[i] <= local_max_k]
            if fit:
                pivot = max(
                    fit,
                    key=lambda i: (
                        (gain[i] + 0.35 * pos_sum_total[i]) / max(1, local_sizes[i]),
                        gain[i],
                        -local_sizes[i],
                        -i,
                    ),
                )
            else:
                pivot = max(
                    avail,
                    key=lambda i: (
                        (gain[i] + 0.35 * pos_sum_total[i]) / max(1, local_sizes[i]),
                        gain[i],
                        -local_sizes[i],
                        -i,
                    ),
                )

            expected = gain[pivot] + 0.5 * pos_sum_total[pivot]
            if fit and expected >= 0.0:
                i = pivot
                new_sel = sel | (1 << i)
                new_forb = forb
                x = local_conflict[i]
                while x:
                    lsb = x & -x
                    j = lsb.bit_length() - 1
                    new_forb |= desc_self[j]
                    x ^= lsb
                new_cnt = cnt + local_sizes[i]
                new_score = score + gain[i]
                new_omask = omask | local_orig_masks[i]
                new_gain = gain[:]
                row = local_pair[i]
                for t in range(m):
                    new_gain[t] += row[t]
                dfs(new_sel, new_forb, new_cnt, new_score, new_omask, new_gain)
                if node_visits >= node_budget:
                    return
                dfs(sel, forb | desc_self[i], cnt, score, omask, gain)
            else:
                dfs(sel, forb | desc_self[pivot], cnt, score, omask, gain)
                if node_visits >= node_budget:
                    return
                if cnt + local_sizes[pivot] <= local_max_k:
                    i = pivot
                    new_sel = sel | (1 << i)
                    new_forb = forb
                    x = local_conflict[i]
                    while x:
                        lsb = x & -x
                        j = lsb.bit_length() - 1
                        new_forb |= desc_self[j]
                        x ^= lsb
                    new_cnt = cnt + local_sizes[i]
                    new_score = score + gain[i]
                    new_omask = omask | local_orig_masks[i]
                    new_gain = gain[:]
                    row = local_pair[i]
                    for t in range(m):
                        new_gain[t] += row[t]
                    dfs(new_sel, new_forb, new_cnt, new_score, new_omask, new_gain)

        dfs(0, forb0, 0, 0.0, 0, local_weights[:])

        prof = []
        for c in range(local_max_k + 1):
            if best_score[c] > -math.inf / 2:
                prof.append((c, best_score[c], best_mask[c]))
        if not prof:
            prof = [(0, 0.0, 0)]
        prof.sort(key=lambda t: t[0])
        return prof

    def exact_component_profile(atom_ids):
        m = len(atom_ids)
        if m == 0:
            return [(0, 0.0, 0)]

        sizes = [atom_size[a] for a in atom_ids]
        weights_loc = [atom_weight[a] for a in atom_ids]
        orig_masks = [atom_orig_mask[a] for a in atom_ids]
        local_pair = [[0.0] * m for _ in range(m)]
        local_succ = [0] * m
        local_pred = [0] * m
        local_conf = [0] * m

        loc = {a: i for i, a in enumerate(atom_ids)}
        for i, a in enumerate(atom_ids):
            x = atom_succ[a]
            while x:
                lsb = x & -x
                b = lsb.bit_length() - 1
                j = loc.get(b)
                if j is not None:
                    local_succ[i] |= 1 << j
                    local_pred[j] |= 1 << i
                x ^= lsb
            x = atom_conf[a]
            while x:
                lsb = x & -x
                b = lsb.bit_length() - 1
                j = loc.get(b)
                if j is not None:
                    local_conf[i] |= 1 << j
                x ^= lsb
            row = atom_pair[a]
            for j, b in enumerate(atom_ids):
                if i != j:
                    local_pair[i][j] = row[b]

        return exact_profile_local(sizes, weights_loc, orig_masks, local_succ, local_pred, local_conf, local_pair, max_k, 0)

    def pure_component_profile(atom_ids):
        m = len(atom_ids)
        if m == 0:
            return [(0, 0.0, 0)]
        sizes = [1] * m
        weights_loc = [atom_weight[a] for a in atom_ids]
        orig_masks = [atom_orig_mask[a] for a in atom_ids]
        pair_loc = [[0.0] * m for _ in range(m)]
        for i, a in enumerate(atom_ids):
            row = atom_pair[a]
            for j, b in enumerate(atom_ids):
                if i != j:
                    pair_loc[i][j] = row[b]

        def heuristic_state(seed, cap):
            rng = random.Random(seed)
            sel = 0
            score = 0.0
            count = 0
            gain = weights_loc[:]
            local_best = {0: (0.0, 0)}
            # Greedy fill to a cap
            while count < cap and count < max_k:
                cand = []
                for i in range(m):
                    if (sel >> i) & 1:
                        continue
                    if count + 1 > max_k:
                        continue
                    val = (gain[i] + 0.35 * sum(max(0.0, pair_loc[i][j]) for j in range(m) if j != i)) + rng.random() * 1e-9
                    cand.append((val, i))
                if not cand:
                    break
                cand.sort(reverse=True)
                top = cand[: min(6, len(cand))]
                i = top[0][1] if len(top) == 1 or rng.random() < 0.7 else top[rng.randrange(len(top))][1]
                sel |= 1 << i
                count += 1
                score += gain[i]
                row = pair_loc[i]
                for t in range(m):
                    gain[t] += row[t]
                omask = mask_to_global_orig_mask(sel, orig_masks)
                if score > local_best.get(count, (-math.inf, 0))[0] + EPS:
                    local_best[count] = (score, omask)
            return sel, score, count, gain, local_best

        def local_search(sel, score, count, gain, best_by_count, seed):
            rng = random.Random(seed)
            best_mask = sel
            best_score = score
            def record_state(mask, sc, ct):
                nonlocal best_mask, best_score
                omask = mask_to_global_orig_mask(mask, orig_masks)
                prev = best_by_count.get(ct)
                if prev is None or sc > prev[0] + EPS:
                    best_by_count[ct] = (sc, omask)
                if sc > best_score + EPS:
                    best_score = sc
                    best_mask = mask

            record_state(sel, score, count)

            for _ in range(120):
                selected = [i for i in range(m) if (sel >> i) & 1]
                unselected = [i for i in range(m) if not ((sel >> i) & 1)]
                if not selected and not unselected:
                    break

                # Best one-step improvement among add/remove/swap
                best_delta = 0.0
                best_move = None

                # add
                if count < max_k and unselected:
                    add_cands = sorted(unselected, key=lambda i: (gain[i], -i), reverse=True)[:24]
                    for i in add_cands:
                        d = gain[i]
                        if d > best_delta + EPS:
                            best_delta = d
                            best_move = ("add", i)

                # remove
                if selected:
                    rem_cands = sorted(selected, key=lambda i: (gain[i], i))[:24]
                    for j in rem_cands:
                        d = -gain[j]
                        if d > best_delta + EPS:
                            best_delta = d
                            best_move = ("rem", j)

                # swap
                if selected and unselected:
                    add_cands = sorted(unselected, key=lambda i: (gain[i], -i), reverse=True)[:18]
                    rem_cands = sorted(selected, key=lambda i: (gain[i], i))[:18]
                    for i in add_cands:
                        gi = gain[i]
                        rowi = pair_loc[i]
                        for j in rem_cands:
                            d = gi - rowi[j] - gain[j]
                            if d > best_delta + EPS:
                                best_delta = d
                                best_move = ("swap", i, j)

                if best_move is None:
                    break

                typ = best_move[0]
                if typ == "add":
                    i = best_move[1]
                    sel |= 1 << i
                    count += 1
                    score += gain[i]
                    row = pair_loc[i]
                    for t in range(m):
                        gain[t] += row[t]
                elif typ == "rem":
                    j = best_move[1]
                    sel &= ~(1 << j)
                    count -= 1
                    score -= gain[j]
                    row = pair_loc[j]
                    for t in range(m):
                        gain[t] -= row[t]
                else:
                    i, j = best_move[1], best_move[2]
                    sel ^= (1 << i) | (1 << j)
                    score += gain[i] - pair_loc[i][j] - gain[j]
                    rowi = pair_loc[i]
                    rowj = pair_loc[j]
                    for t in range(m):
                        gain[t] += rowi[t] - rowj[t]
                record_state(sel, score, count)

            # Diversification kicks
            for _ in range(4):
                if m == 0:
                    break
                selected = [i for i in range(m) if (sel >> i) & 1]
                if not selected:
                    break
                remove_k = 1 + rng.randrange(1 + min(2, len(selected) - 1))
                remove_k = min(remove_k, len(selected))
                bad = sorted(selected, key=lambda i: (gain[i], i))[:remove_k]
                for j in bad:
                    sel &= ~(1 << j)
                    count -= 1
                    score -= gain[j]
                    row = pair_loc[j]
                    for t in range(m):
                        gain[t] -= row[t]
                record_state(sel, score, count)

                # Repair greedily toward a potentially better count
                for _ in range(max_k):
                    unselected = [i for i in range(m) if not ((sel >> i) & 1)]
                    if not unselected:
                        break
                    add_cands = sorted(unselected, key=lambda i: (gain[i], -i), reverse=True)[:16]
                    if not add_cands:
                        break
                    i = add_cands[0]
                    if gain[i] <= EPS and count >= max_k:
                        break
                    sel |= 1 << i
                    count += 1
                    score += gain[i]
                    row = pair_loc[i]
                    for t in range(m):
                        gain[t] += row[t]
                    record_state(sel, score, count)
                    if gain[i] <= EPS:
                        break

            return best_mask, best_score, best_by_count

        # Seed runs
        best_by_count = {0: (0.0, 0)}
        candidate_states = []
        targets = sorted(set([
            0,
            min(max_k, m),
            min(max_k, max(1, max_k // 4)),
            min(max_k, max(1, max_k // 2)),
            min(max_k, max(1, (3 * max_k) // 4)),
            max_k,
        ]))
        seeds = [11, 17, 23, 31, 47, 59]
        for seed in seeds:
            for cap in targets:
                sel, score, count, gain, local_best = heuristic_state(seed, cap)
                best_mask, best_score, best_by_count = local_search(sel, score, count, gain, best_by_count, seed + 1000 + cap)
                candidate_states.append((best_score, best_mask))
                # merge any counts visited during greedy seed
                for c, (s, om) in local_best.items():
                    prev = best_by_count.get(c)
                    if prev is None or s > prev[0] + EPS:
                        best_by_count[c] = (s, om)

        # Core exact refinement on a few best heuristic states
        candidate_states = sorted(candidate_states, key=lambda t: t[0], reverse=True)
        candidate_masks = []
        seen_masks = set()
        for sc, mask in candidate_states:
            if mask not in seen_masks:
                seen_masks.add(mask)
                candidate_masks.append((sc, mask))
            if len(candidate_masks) >= 3:
                break

        def refine_with_core(mask):
            selected = [i for i in range(m) if (mask >> i) & 1]
            unselected = [i for i in range(m) if not ((mask >> i) & 1)]
            if not selected:
                return
            # Core = half selected + half promising unselected
            sel_keep_n = min(len(selected), CORE_LIMIT // 2)
            sel_keep = sorted(
                selected,
                key=lambda i: (abs(weights_loc[i]) + sum(max(0.0, pair_loc[i][j]) for j in range(m) if j != i), gain[i]),
                reverse=True,
            )[:sel_keep_n]
            core = set(sel_keep)
            need = CORE_LIMIT - len(core)
            if need > 0:
                uns_keep = sorted(
                    unselected,
                    key=lambda i: (max(0.0, gain[i]) + sum(max(0.0, pair_loc[i][j]) for j in range(m) if j != i), gain[i]),
                    reverse=True,
                )[:need]
                core.update(uns_keep)

            core_ids = sorted(core)
            if not core_ids:
                return
            core_mask = 0
            for i in core_ids:
                core_mask |= 1 << i
            fixed_mask = mask & ~core_mask
            fixed_count = fixed_mask.bit_count()
            if fixed_count > max_k:
                return
            fixed_score = score_mask_local(fixed_mask, weights_loc, pair_loc)
            fixed_orig = mask_to_global_orig_mask(fixed_mask, orig_masks)
            rem_cap = max_k - fixed_count

            core_sizes = [1] * len(core_ids)
            core_weights = [0.0] * len(core_ids)
            core_orig_masks = [orig_masks[i] for i in core_ids]
            core_pair = [[0.0] * len(core_ids) for _ in range(len(core_ids))]
            for ii, i in enumerate(core_ids):
                w = weights_loc[i]
                row = pair_loc[i]
                for j in bits_iter(fixed_mask):
                    w += row[j]
                core_weights[ii] = w
            for ii, i in enumerate(core_ids):
                for jj, j in enumerate(core_ids):
                    if ii != jj:
                        core_pair[ii][jj] = pair_loc[i][j]

            core_prof = exact_profile_local(core_sizes, core_weights, core_orig_masks,
                                            [0] * len(core_ids), [0] * len(core_ids), [0] * len(core_ids),
                                            core_pair, rem_cap, 0)
            for cnt, sc, om in core_prof:
                total_cnt = fixed_count + cnt
                total_sc = fixed_score + sc
                total_om = fixed_orig | om
                prev = best_by_count.get(total_cnt)
                if prev is None or total_sc > prev[0] + EPS:
                    best_by_count[total_cnt] = (total_sc, total_om)

        for sc, mask in candidate_masks:
            refine_with_core(mask)

        prof = [(c, s, om) for c, (s, om) in best_by_count.items() if s > -math.inf / 2]
        if not prof:
            prof = [(0, 0.0, 0)]
        prof.sort(key=lambda t: t[0])
        return prof

    def structured_component_profile(atom_ids):
        # Large component with precedence/mutex: use greedy feasible construction and exact if small.
        m = len(atom_ids)
        if m == 0:
            return [(0, 0.0, 0)]
        if m <= EXACT_LIMIT:
            return exact_component_profile(atom_ids)

        sizes = [atom_size[a] for a in atom_ids]
        weights_loc = [atom_weight[a] for a in atom_ids]
        orig_masks = [atom_orig_mask[a] for a in atom_ids]
        local_pair = [[0.0] * m for _ in range(m)]
        local_succ = [0] * m
        local_pred = [0] * m
        local_conf = [0] * m
        loc = {a: i for i, a in enumerate(atom_ids)}

        for i, a in enumerate(atom_ids):
            x = atom_succ[a]
            while x:
                lsb = x & -x
                b = lsb.bit_length() - 1
                j = loc.get(b)
                if j is not None:
                    local_succ[i] |= 1 << j
                    local_pred[j] |= 1 << i
                x ^= lsb
            x = atom_conf[a]
            while x:
                lsb = x & -x
                b = lsb.bit_length() - 1
                j = loc.get(b)
                if j is not None:
                    local_conf[i] |= 1 << j
                x ^= lsb
            row = atom_pair[a]
            for j, b in enumerate(atom_ids):
                if i != j:
                    local_pair[i][j] = row[b]

        topo, anc, desc = topo_closures(local_succ)
        desc_self = [desc[i] | (1 << i) for i in range(m)]
        invalid_forb = 0
        for i in range(m):
            if sizes[i] > max_k or atom_invalid[atom_ids[i]]:
                invalid_forb |= desc_self[i]

        pos_sum_total = [0.0] * m
        for i in range(m):
            s = 0.0
            row = local_pair[i]
            for j in range(m):
                if i != j and row[j] > 0.0:
                    s += row[j]
            pos_sum_total[i] = s

        future = [0.0] * m
        for u in reversed(topo):
            val = max(0.0, weights_loc[u])
            x = local_succ[u]
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                val += max(0.0, local_pair[u][v]) + future[v]
                x ^= lsb
            future[u] = val

        bonus = [0.0] * m
        for i in range(m):
            bonus[i] = 0.3 * future[i] + 0.15 * pos_sum_total[i] + 0.05 * (anc[i].bit_count() + desc[i].bit_count() + local_conf[i].bit_count())

        best_by_count = {0: (0.0, 0)}

        def record(mask, score, cnt):
            om = mask_to_global_orig_mask(mask, orig_masks)
            prev = best_by_count.get(cnt)
            if prev is None or score > prev[0] + EPS:
                best_by_count[cnt] = (score, om)

        def greedy(seed, cap):
            rng = random.Random(seed)
            sel = 0
            forb = invalid_forb
            cnt = 0
            score = 0.0
            gain = weights_loc[:]
            record(sel, score, cnt)
            while cnt < cap and cnt < max_k:
                cand = []
                rem = ((1 << m) - 1) & ~(sel | forb)
                while rem:
                    lsb = rem & -rem
                    i = lsb.bit_length() - 1
                    rem ^= lsb
                    if anc[i] & ~sel:
                        continue
                    if local_conf[i] & sel:
                        continue
                    if cnt + sizes[i] > max_k:
                        continue
                    val = (gain[i] + bonus[i]) / max(1, sizes[i]) + rng.random() * 1e-9
                    cand.append((val, i))
                if not cand:
                    break
                cand.sort(reverse=True)
                top = cand[: min(5, len(cand))]
                i = top[0][1] if len(top) == 1 or rng.random() < 0.7 else top[rng.randrange(len(top))][1]
                if cnt + sizes[i] > max_k:
                    break
                sel |= 1 << i
                cnt += sizes[i]
                score += gain[i]
                x = local_conf[i]
                while x:
                    lsb = x & -x
                    j = lsb.bit_length() - 1
                    forb |= desc_self[j]
                    x ^= lsb
                row = local_pair[i]
                for t in range(m):
                    gain[t] += row[t]
                record(sel, score, cnt)
            return sel, score, cnt, gain

        # Several greedy runs for coverage
        targets = sorted(set([
            0,
            min(max_k, m),
            min(max_k, max(1, max_k // 4)),
            min(max_k, max(1, max_k // 2)),
            min(max_k, max(1, (3 * max_k) // 4)),
            max_k,
        ]))
        for seed in [11, 17, 23, 31, 47, 59]:
            for cap in targets:
                greedy(seed, cap)

        prof = [(c, s, om) for c, (s, om) in best_by_count.items() if s > -math.inf / 2]
        if not prof:
            prof = [(0, 0.0, 0)]
        prof.sort(key=lambda t: t[0])
        return prof

    def solve_component(atom_ids):
        if not atom_ids:
            return [(0, 0.0, 0)]
        structural = any(atom_succ[a] or atom_conf[a] or atom_invalid[a] for a in atom_ids)
        if len(atom_ids) <= EXACT_LIMIT:
            return exact_component_profile(atom_ids)
        if structural:
            return structured_component_profile(atom_ids)
        return pure_component_profile(atom_ids)

    # ---------- Solve all components ----------
    profiles = []
    for atom_ids in atom_components:
        prof = solve_component(atom_ids)
        if not prof:
            prof = [(0, 0.0, 0)]
        profiles.append(prof)

    # ---------- Combine profiles by cardinality ----------
    dp = {0: (0.0, 0)}  # count -> (score, orig_mask)
    for prof in profiles:
        ndp = {}
        prof_sorted = sorted(prof, key=lambda t: (t[1], -t[0]), reverse=True)
        for c0, (s0, m0_mask) in dp.items():
            for c1, s1, m1_mask in prof_sorted:
                nc = c0 + c1
                if nc > max_k:
                    continue
                ns = s0 + s1
                prev = ndp.get(nc)
                if prev is None or ns > prev[0] + EPS:
                    ndp[nc] = (ns, m0_mask | m1_mask)
        dp = ndp if ndp else dp

    best_score = -math.inf
    best_mask = 0

    for c, (s, om) in dp.items():
        if min_k <= c <= max_k and s > best_score + EPS:
            best_score = s
            best_mask = om

    if best_score == -math.inf:
        for c, (s, om) in dp.items():
            if s > best_score + EPS:
                best_score = s
                best_mask = om

    selected = []
    x = best_mask
    while x:
        lsb = x & -x
        i = lsb.bit_length() - 1
        selected.append(orig_ids[i] if 0 <= i < len(orig_ids) else i)
        x ^= lsb

    try:
        selected.sort()
    except Exception:
        selected = [x for x in selected]

    sys.stdout.write(json.dumps({"selection": {"variables": selected}}))


if __name__ == "__main__":
    solve_sds()
