from .trees import make_tree_instance
from .dense import make_dense_instance

from .families import (
    make_greedy_easy_instance,
    make_local_optima_instance,
    make_tree_showcase_instance,
    make_decomposable_instance,
    make_dense_deceptive_instance,
    make_structural_trap_instance,
)

from .qubo import (
    make_random_qubo_instance,
    make_planted_qubo_instance,
    make_maxcut_qubo_instance,
)
