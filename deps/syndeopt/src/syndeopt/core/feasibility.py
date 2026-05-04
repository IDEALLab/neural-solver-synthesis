from .instance import SDSInstance, Bitmask, bit_count

def feasible(inst: SDSInstance, x: Bitmask) -> bool:
    """Full feasibility check: cardinality, precedence, mutex, groups."""
    k = bit_count(x)
    L, U = inst.card.L, inst.card.U
    if not (L <= k <= U):
        return False

    # precedence: j <= i
    for i, j in inst.precedence:
        if ((x >> j) & 1) and not ((x >> i) & 1):
            return False

    # mutex: x_a + x_b <= 1
    for a, b in inst.mutex:
        if ((x >> a) & 1) and ((x >> b) & 1):
            return False

    # groups: at most 1 in each group
    for _, members in inst.groups.items():
        cnt = sum((x >> i) & 1 for i in members)
        if cnt > 1:
            return False

    return True


def feasible_without_lower(inst: SDSInstance, x: Bitmask) -> bool:
    """Feasibility ignoring the lower cardinality bound; used while building solutions."""
    k = bit_count(x)
    U = inst.card.U
    if k > U:
        return False

    for i, j in inst.precedence:
        if ((x >> j) & 1) and not ((x >> i) & 1):
            return False

    for a, b in inst.mutex:
        if ((x >> a) & 1) and ((x >> b) & 1):
            return False

    for _, members in inst.groups.items():
        cnt = sum((x >> i) & 1 for i in members)
        if cnt > 1:
            return False

    return True
