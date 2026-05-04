from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set

Bitmask = int

def bit_count(x: Bitmask) -> int:
    """Count set bits in a bitmask. Compatible with Python < 3.10."""
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count('1')

@dataclass(frozen=True)
class CardBounds:
    L: int
    U: int

@dataclass
class SDSInstance:
    """
    Synergistic Dependency Selection instance:

      max sum_i w[i] x_i + sum_{(i,j) in W} W[i,j] x_i x_j

    with x_i in {0,1} and constraints:
      - precedence: (i,j) means x_j <= x_i
      - mutex: (a,b) means x_a + x_b <= 1
      - groups: group_id -> variables, at most 1 from each group
      - cardinality: L <= sum_i x_i <= U
    """
    n: int
    w: List[float]
    W: Dict[Tuple[int, int], float]      # only stored for i < j
    precedence: List[Tuple[int, int]]
    mutex: List[Tuple[int, int]]
    groups: Dict[int, List[int]]
    card: CardBounds
    adj: List[Set[int]] = field(init=False)

    def __post_init__(self):
        self.adj = [set() for _ in range(self.n)]
        for (i, j), _ in self.W.items():
            self.adj[i].add(j)
            self.adj[j].add(i)
