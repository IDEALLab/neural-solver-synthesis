from .instance import SDSInstance, Bitmask

def score(inst: SDSInstance, x: Bitmask) -> float:
    """Compute objective value for a bitmask solution."""
    s = 0.0
    # unary
    for i in range(inst.n):
        if (x >> i) & 1:
            s += inst.w[i]
    # pairwise
    for (i, j), wij in inst.W.items():
        if ((x >> i) & 1) and ((x >> j) & 1):
            s += wij
    return s
