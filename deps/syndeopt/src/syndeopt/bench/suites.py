from typing import List, Tuple
from ..core.instance import SDSInstance
from ..gen import (
    make_tree_instance,
    make_dense_instance,
    make_greedy_easy_instance,
    make_local_optima_instance,
    make_tree_showcase_instance,
    make_decomposable_instance,
    make_dense_deceptive_instance,
    make_random_qubo_instance,
    make_planted_qubo_instance,
    make_maxcut_qubo_instance,
)

def basic_suite(seed: int = 0) -> List[Tuple[str, SDSInstance]]:
    """
    A small suite that spans the main design space:

      - greedy-friendly modular
      - local-search trap (bait vs clique)
      - tree-structured
      - decomposable clusters
      - dense & deceptive
      - QUBO-native (random, planted, Max-Cut)
      - simple tree/dense baselines
    """
    insts: List[Tuple[str, SDSInstance]] = []

    # 1) Greedy-friendly modular
    insts.append(("greedy_easy", make_greedy_easy_instance(n=12, card=(5, 5), seed=seed + 0)))

    # 2) Local-search-friendly bait vs clique
    insts.append(("local_optima", make_local_optima_instance(n=18, card=(5, 5), seed=seed + 1)))

    # 3) Tree-structured (showcase version)
    insts.append(("tree_showcase", make_tree_showcase_instance(n=14, card=(4, 10), seed=seed + 2)))

    # 4) Decomposable clusters
    insts.append(("decomposable", make_decomposable_instance(n=18, card=(6, 6), seed=seed + 3)))

    # 5) Dense & deceptive (BnB / CP-SAT challenging)
    insts.append(("dense_deceptive", make_dense_deceptive_instance(n=20, card=(7, 11), seed=seed + 4)))

    # 6) QUBO-native: random dense/sparse
    insts.append(("qubo_random", make_random_qubo_instance(n=20, card=(0, 20), seed=seed + 5)))

    # 7) QUBO-native: planted high-value solution
    insts.append(("qubo_planted", make_planted_qubo_instance(n=20, card=(0, 20), seed=seed + 6)))

    # 8) QUBO-native: Max-Cut on random graph
    insts.append(("qubo_maxcut", make_maxcut_qubo_instance(n=20, edge_prob=0.4, card=(0, 20), seed=seed + 7)))

    # 9–10) keep one simple 'dense' and 'tree' for generic baseline
    insts.append(("tree_simple", make_tree_instance(n=14, card=(4, 10), seed=seed + 8)))
    insts.append(("dense_simple", make_dense_instance(n=16, card=(6, 10), seed=seed + 9)))

    return insts
