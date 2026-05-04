# Syndeopt Integration Report

## Executive Summary

**Syndeopt** (Synergistic Discrete Optimization Playground) is a research-grade Python library for binary quadratic optimization problems with combinatorial constraints. It serves as the core implementation for the **SDS (Synergistic Dependency Selection)** problem domain in this LLM fine-tuning project.

This report documents how syndeopt is integrated into the project, the APIs used, and its role in data generation, reward computation, and evaluation.

---

## 1. What is Syndeopt?

### 1.1 Problem Formulation

Syndeopt focuses on problems of the form:

\[
\max_{x\in\{0,1\}^n} \quad f(x) = \sum_i w_i x_i + \sum_{i<j} W_{ij} x_i x_j
\]

subject to combinatorial constraints:
- **Cardinality**: \( L \le \sum_i x_i \le U \)
- **Precedence** (DAG): \( x_j \le x_i \) (if \(i\) is selected, \(j\) must be selected)
- **Mutex** (pairwise exclusion): \( x_a + x_b \le 1 \)
- **Groups**: at most one variable selected per group

This problem family—called **SDS**—captures many NP-hard problems including feature selection, maximum-weight closure, constrained QUBOs, and Max-Cut variants.

### 1.2 Key Features

- **Clean Problem Model**: `SDSInstance` with efficient bitmask representation
- **Parametric Generation**: Generate any problem within the SDS class
- **Solver Zoo**: Unified API for heuristics, exact solvers, and industrial MIP engines
- **Benchmarking Tools**: Performance profiles, anytime curves, suite runners
- **Research-Grade**: Type hints, documentation, testing infrastructure

### 1.3 Repository Location

Syndeopt is included as a git submodule:
- **Path**: `deps/syndeopt/`
- **Repository**: Git submodule (configured in `.gitmodules`)
- **Installation**: Installed as editable package via `setup_dev.sh` or manually

---

## 2. Integration Points in This Project

### 2.1 Core Components Using Syndeopt

1. **Data Generation** (`data/gen_sds_dataset.py`)
   - Uses syndeopt generators to create SDS problem instances
   - Converts instances to JSON format for LLM training

2. **Simulator** (`deps/open-r1/src/open_r1/simulators/sds_simulator.py`)
   - Uses syndeopt for scoring and feasibility checking
   - Core of the reward computation system

3. **Evaluation** (`evaluation/sds/evaluate.py`, `evaluation/sds/utils.py`)
   - Reconstructs `SDSInstance` objects from mission data
   - Computes true scores and constraint violations

4. **Reward Functions** (`deps/open-r1/src/open_r1/rewards_unified_v2.py`)
   - Uses syndeopt generators for generalization testing
   - Tests code on random problem instances

### 2.2 Import Strategy

Syndeopt is imported with path resolution to handle different directory structures:

```python
# Add syndeopt to path
_workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_syndeopt_path = os.path.join(_workspace_root, 'deps', 'syndeopt', 'src')
if _syndeopt_path not in sys.path:
    sys.path.insert(0, _syndeopt_path)

from syndeopt.core.instance import SDSInstance, CardBounds
from syndeopt.core.scoring import score as syndeopt_score
from syndeopt.core.feasibility import feasible as syndeopt_feasible
```

---

## 3. Key APIs Used

### 3.1 Core Data Structures

#### `SDSInstance`

The main problem representation:

```python
from syndeopt.core.instance import SDSInstance, CardBounds

inst = SDSInstance(
    n=10,                                    # Number of variables
    w=[1.0, 2.0, ...],                      # Unary weights (list of n floats)
    W={(0,1): 3.0, (1,2): -2.0, ...},       # Pairwise interactions (dict)
    precedence=[(0,1), (1,2)],              # DAG constraints (list of tuples)
    mutex=[(3,4)],                           # Mutex pairs (list of tuples)
    groups={0: [5,6,7]},                     # Group constraints (dict)
    card=CardBounds(L=3, U=7)                # Cardinality bounds
)
```

**Key Attributes:**
- `inst.n`: Number of variables
- `inst.w`: List of unary weights
- `inst.W`: Dictionary of pairwise interactions `{(i,j): weight}`
- `inst.card.L`, `inst.card.U`: Lower and upper cardinality bounds

#### `CardBounds`

Frozen dataclass for cardinality bounds:

```python
from syndeopt.core.instance import CardBounds

card = CardBounds(L=3, U=7)  # 3 ≤ sum x_i ≤ 7
```

**Note**: The old custom implementation used tuples `(L, U)`, but syndeopt uses `CardBounds` objects accessed via `inst.card.L` and `inst.card.U`.

### 3.2 Scoring and Feasibility

#### Scoring

```python
from syndeopt.core.scoring import score

# Convert selection to bitmask
x_bits = 0
for var in selected_ids:
    x_bits |= (1 << var)

# Calculate score
score_value = score(inst, x_bits)
```

**Note**: The old API was `inst.score(x_bits)`, but syndeopt uses a functional API: `score(inst, x_bits)`.

#### Feasibility Checking

```python
from syndeopt.core.feasibility import feasible

# Check if selection is feasible
is_feasible = feasible(inst, x_bits)
```

**Note**: The old API was `inst.feasible(x_bits)`, but syndeopt uses a functional API: `feasible(inst, x_bits)`.

### 3.3 Problem Generation

Syndeopt provides multiple generator functions for different problem regimes:

```python
from syndeopt.gen import (
    make_tree_instance,
    make_tree_showcase_instance,
    make_dense_deceptive_instance,
    make_decomposable_instance,
    make_greedy_easy_instance,
    make_local_optima_instance,
    make_random_qubo_instance,
    make_planted_qubo_instance,
    make_maxcut_qubo_instance,
    make_dense_instance,
    make_structural_trap_instance,
)

# Example: Generate a tree-structured instance
inst = make_tree_showcase_instance(
    n=14,
    card=CardBounds(L=4, U=10),
    seed=404
)
```

**Generator Types:**
- **Structured**: Tree, decomposable, dense deceptive, greedy-easy, local-optima
- **QUBO-Native**: Random QUBO, planted-solution QUBO, Max-Cut QUBO

### 3.4 Solvers (Optional)

**⚠️ IMPORTANT**: Solvers are **NEVER** run during normal operations (data generation, reward assignment, RL finetuning). They are only used when explicitly requested (e.g., for computing optimal solutions in SFT targets).

```python
from syndeopt.solvers import get_solver

# Get a solver (lazy-loaded, only when needed)
solver = get_solver("cpsat")  # or "greedy", "local_search", "bnb"

# Solve with time budget
result = solver.solve(inst, budget_sec=5.0, seed=0)
best_score = result.score
best_solution = result.mask  # Bitmask representation
```

**Available Solvers:**
- `greedy`: Marginal-gain greedy heuristic
- `local_search`: 1-flip hill-climber with random restarts
- `bnb`: Branch-and-bound (exact)
- `cpsat`: OR-Tools CP-SAT (industrial MIP engine)

---

## 4. Usage in Project Components

### 4.1 Data Generation (`data/gen_sds_dataset.py`)

**Purpose**: Generate SDS problem instances for LLM training datasets.

**Key Operations:**
1. Use syndeopt generators to create `SDSInstance` objects
2. Convert instances to JSON format for prompts
3. Optionally compute optimal solutions (when `compute_optimal=True`)

**Example:**
```python
from syndeopt.gen import make_tree_showcase_instance
from syndeopt.core.instance import CardBounds

# Generate instance
inst = make_tree_showcase_instance(
    n=14,
    card=CardBounds(L=4, U=10),
    seed=404
)

# Convert to problem dict for dataset
problem = {
    "requirements": {
        "n_variables": inst.n,
        "cardinality_bounds": [inst.card.L, inst.card.U],
        "weights": inst.w,
        "interactions": {f"{i},{j}": w for (i,j), w in inst.W.items()},
        # ... constraints
    }
}
```

**Solver Usage**: Only when `compute_optimal=True` (for SFT targets), uses lazy import:
```python
if compute_optimal:
    from syndeopt.solvers import get_solver
    solver = get_solver("cpsat")
    result = solver.solve(inst, budget_sec=5.0, seed=0)
```

### 4.2 Simulator (`deps/open-r1/src/open_r1/simulators/sds_simulator.py`)

**Purpose**: Execute LLM-generated code and compute rewards.

**Key Operations:**
1. Reconstruct `SDSInstance` from requirements dict
2. Convert LLM selection to bitmask
3. Check feasibility using `feasible(inst, x_bits)`
4. Calculate score using `score(inst, x_bits)`

**Example:**
```python
# Create instance from requirements
inst = SDSInstance(
    n=requirements["n_variables"],
    w=requirements["weights"],
    W=requirements["interactions"],
    precedence=requirements["precedence"],
    mutex=requirements["mutex"],
    groups=requirements["groups"],
    card=CardBounds(L=card_bounds[0], U=card_bounds[1])
)

# Convert selection to bitmask
x_bits = 0
for var in selection:
    x_bits |= (1 << var)

# Check feasibility and score
is_feasible = syndeopt_feasible(inst, x_bits)
score_value = syndeopt_score(inst, x_bits) if is_feasible else 0.0
```

**Note**: The simulator does **NOT** use solvers—only scoring and feasibility checking.

### 4.3 Evaluation (`evaluation/sds/utils.py`, `evaluation/sds/evaluate.py`)

**Purpose**: Reconstruct instances from mission data and compute evaluation metrics.

**Key Operations:**
1. Reconstruct `SDSInstance` from mission dictionary
2. Compute true scores for LLM solutions
3. Check constraint violations

**Example:**
```python
def mission_to_instance(m):
    """Reconstruct SDSInstance from mission dictionary."""
    return SDSInstance(
        n=m["n_variables"],
        w=m["weights"],
        W={parse_pair(k): v for k, v in m["interactions"].items()},
        precedence=[tuple(x) for x in m.get("precedence", [])],
        mutex=[tuple(x) for x in m.get("mutex", [])],
        groups={int(k): v for k, v in m.get("groups", {}).items()},
        card=CardBounds(L=m["cardinality_bounds"][0], U=m["cardinality_bounds"][1])
    )

def calculate_true_score(instance, selected_ids):
    """Compute true score for a selection."""
    score = sum(instance.w[i] for i in selected_ids)
    sel_set = set(selected_ids)
    for (i, j), weight in instance.W.items():
        if i in sel_set and j in sel_set:
            score += weight
    return score
```

### 4.4 Reward Functions (`deps/open-r1/src/open_r1/rewards_unified_v2.py`)

**Purpose**: Generalization testing—evaluate code on random problem instances.

**Key Operations:**
1. Use syndeopt generators to create test instances
2. Execute LLM code on test instances
3. Compute rewards based on performance

**Example:**
```python
# Generate test instance for generalization
test_inst = make_dense_deceptive_instance(
    n=random.randint(10, 20),
    card=CardBounds(L=3, U=8),
    seed=test_seed
)

# Execute code and check feasibility/score
is_feasible = syndeopt_feasible(test_inst, x_bits)
score = syndeopt_score(test_inst, x_bits) if is_feasible else 0.0
```

**Note**: Uses generators and scoring/feasibility, but **NOT** solvers.

---

## 5. API Migration Notes

### 5.1 Key Differences from Old Custom Implementation

| Old API | New API (Syndeopt) |
|---------|-------------------|
| `inst.card_bounds` (tuple) | `inst.card.L`, `inst.card.U` (CardBounds) |
| `inst.score(x_bits)` | `score(inst, x_bits)` |
| `inst.feasible(x_bits)` | `feasible(inst, x_bits)` |
| `from open_r1.simulators.sds import ...` | `from syndeopt.core.instance import ...` |
| `make_tree_showcase_instance(n, card=(L,U), seed)` | `make_tree_showcase_instance(n, card=CardBounds(L,U), seed)` |

### 5.2 Deprecated Code

The old custom SDS implementation is marked as deprecated:
- **File**: `deps/open-r1/src/open_r1/simulators/sds.py`
- **Status**: DEPRECATED (kept for backward compatibility)
- **Action**: Should not be used for new code

---

## 6. Solver Usage Policy

### 6.1 Critical Policy

**Solvers are NEVER run during normal operations:**

- ✅ **Data Generation**: `compute_optimal=False` by default (no solvers called)
- ✅ **Reward Assignment**: Uses simulator only (`score` and `feasible` functions)
- ✅ **RL Finetuning**: Outcome-only rewards, no solver involvement

### 6.2 When Solvers Are Used

Solvers are **lazy-loaded** and only execute when:
- `compute_optimal=True` is explicitly passed (for optional SFT targets)
- This requires manual flag setting and is not used in the RL pipeline

### 6.3 Files That Use Solvers

**Only when explicitly requested:**
- `data/gen_sds_dataset.py`: Lazy import inside `if compute_optimal:` block

**Files that do NOT use solvers:**
- `deps/open-r1/src/open_r1/simulators/sds_simulator.py`: Only uses scoring/feasibility
- `deps/open-r1/src/open_r1/rewards_unified_v2.py`: Only uses instance generation for generalization tests
- `evaluation/sds/evaluate.py`: Only uses scoring/feasibility

---

## 7. Installation and Setup

### 7.1 Automatic Setup

Syndeopt is automatically installed via `setup_dev.sh`:

```bash
./setup_dev.sh
```

This installs syndeopt as an editable package from `deps/syndeopt`.

### 7.2 Manual Setup

```bash
cd deps/syndeopt
pip install -e .
```

### 7.3 Requirements

- **Python**: 3.11+ (see `deps/syndeopt/pyproject.toml`)
- **Dependencies**: Automatically handled by syndeopt's `pyproject.toml`

---

## 8. Troubleshooting

### 8.1 Import Errors

**Error**: `ModuleNotFoundError: No module named 'syndeopt'`

**Solutions:**
1. Ensure syndeopt is installed: `pip install -e deps/syndeopt`
2. Check path resolution in import statements
3. Verify `deps/syndeopt/src` exists and contains `syndeopt/` directory

### 8.2 Path Resolution

The codebase uses multiple path resolution strategies to find syndeopt:
- Direct import (if installed as package)
- Relative path from workspace root: `deps/syndeopt/src`
- Relative path from open-r1: `../../../../syndeopt/src`

If imports fail, check the path resolution logic in the importing file.

### 8.3 Linter Warnings

**Expected**: Linter warnings about unresolved imports are normal if syndeopt is not in the linter environment. The code will work correctly when run in the proper Python 3.11+ environment.

---

## 9. Benefits of Using Syndeopt

1. **Better Code Quality**: Cleaner architecture, type hints, better organization
2. **More Solvers**: Access to CP-SAT, Gurobi, HiGHS, PyBnB, and quantum-inspired solvers (when needed)
3. **Benchmarking Tools**: Performance profiles, anytime curves, suite runners
4. **Better Generators**: More problem types and better parameterization
5. **Research Features**: Reproducibility, documentation, testing infrastructure
6. **Maintainability**: Centralized, well-tested implementation

---

## 10. References

- **Syndeopt Repository**: `deps/syndeopt/`
- **Syndeopt README**: `deps/syndeopt/README.md`
- **Project Integration**: See `data/gen_sds_dataset.py`, `deps/open-r1/src/open_r1/simulators/sds_simulator.py`

---

## Appendix: Quick Reference

### Common Imports

```python
from syndeopt.core.instance import SDSInstance, CardBounds
from syndeopt.core.scoring import score
from syndeopt.core.feasibility import feasible
from syndeopt.gen import (
    make_tree_showcase_instance,
    make_dense_deceptive_instance,
    make_decomposable_instance,
    make_greedy_easy_instance,
    make_local_optima_instance,
)
```

### Common Operations

```python
# Create instance
inst = SDSInstance(n=10, w=[...], W={...}, card=CardBounds(L=3, U=7))

# Convert selection to bitmask
x_bits = sum(1 << i for i in selected_ids)

# Check feasibility
is_feasible = feasible(inst, x_bits)

# Calculate score
score_value = score(inst, x_bits) if is_feasible else 0.0
```
