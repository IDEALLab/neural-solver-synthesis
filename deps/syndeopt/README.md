# 🌌 **SYNDEOPT**

### *Synergistic Discrete Optimization Playground*

*A unified benchmark suite + solver zoo for combinatorial problems defined over unary, pairwise, and global constraints.*

---

## 🚀 Overview

**SYNDEOPT** is a research-grade Python library for exploring **binary quadratic optimization problems** with combinatorial constraints. It focuses on problems of the form:

\[
\max_{x\in{0,1}^n} \quad f(x) = \sum_i w_i x_i + \sum_{i<j} W_{ij} x_i x_j
\]

subject to **combinatorial constraints**:

* **Cardinality** ( L \le \sum_i x_i \le U )
* **Precedence** (DAG constraints): ( x_j \le x_i )
* **Mutex** (pairwise exclusion): x_a + x_b ≤ 1
* **Groups**: at most 1 selected in each group

This problem family—called **SDS** (*Synergistic Dependency Selection*)—captures many important NP-hard problems:

* Feature selection with pairwise synergies
* Maximum-weight closure with extra interactions
* Constrained QUBOs
* Max-Cut and clustering problems
* Constraint-augmented selection problems
* Quantum annealing benchmarks

SYNDEOPT is designed as a **specialized benchmark suite and solver comparison platform** for:

* 🔬 *algorithm research* on binary quadratic problems
* ⚙️ *solver comparisons*: exact, heuristic, metaheuristic, quantum-inspired, MILP, BnB
* 🧪 *instance-generation studies* across different problem landscapes
* 📊 *performance profiling* and visualization
* 🎓 *teaching and reproducibility*

---

## 🎯 Key Capabilities

### 🧩 1. A clean, expressive SDS problem model

`SDSInstance(n, w, W, precedence, mutex, groups, card)`

* unary terms `w[i]`
* pairwise terms `W[(i,j)]`
* automatically tracks adjacency
* efficient scoring + feasibility checks
* bitmask representation for speed

---

### 🏭 2. *Fully parametric problem generation*

The SDS model is **fully parametric**—you can generate **any problem** within the SDS class by specifying:

* `n`: number of binary variables
* `w`: unary weights (list of n floats)
* `W`: pairwise interaction terms (dictionary of (i,j) → weight)
* `precedence`: DAG constraints (list of (i,j) tuples)
* `mutex`: pairwise exclusion constraints (list of (i,j) tuples)
* `groups`: group constraints (dictionary of group_id → list of variables)
* `card`: cardinality bounds (L ≤ sum x_i ≤ U)

Simply construct an `SDSInstance` with your desired parameters:

```python
from syndeopt.core.instance import SDSInstance, CardBounds

inst = SDSInstance(
    n=10,
    w=[1.0, 2.0, ...],  # any unary weights
    W={(0,1): 3.0, (1,2): -2.0, ...},  # any pairwise terms
    precedence=[(0,1), (1,2)],  # any DAG constraints
    mutex=[(3,4)],  # any mutex pairs
    groups={0: [5,6,7]},  # any group constraints
    card=CardBounds(L=3, U=7)  # any cardinality bounds
)
```

**8 predefined generator functions** are provided as convenient presets for common landscape types:

#### **Structured SDS Instances**

| Regime                    | Generator                                           | Difficulty            |
| ------------------------- | --------------------------------------------------- | --------------------- |
| Modular / Greedy-friendly | `make_greedy_easy_instance`                         | trivial               |
| Local-search traps        | `make_local_optima_instance`                        | greedy fails          |
| Tree-structured           | `make_tree_instance`, `make_tree_showcase_instance` | DP-exact              |
| Dense deceptive           | `make_dense_deceptive_instance`                     | requires CP-SAT/BnB   |
| Decomposable clusters     | `make_decomposable_instance`                        | divide & conquer wins |

#### **QUBO-Native Families**

| Regime                | Generator                    | Notes                 |
| --------------------- | ---------------------------- | --------------------- |
| Random QUBO           | `make_random_qubo_instance`  | symmetric or sparse Q |
| Planted-solution QUBO | `make_planted_qubo_instance` | recovery benchmark    |
| Max-Cut QUBO          | `make_maxcut_qubo_instance`  | Erdos-Rényi graphs    |

These generators are just convenience functions—the underlying `SDSInstance` model can represent **any** binary quadratic problem with the supported constraint types.

---

### 🧠 3. Solver Zoo (Unified API)

All solvers expose:

```python
solver.solve(inst, budget_sec, seed) -> SolveResult
```

#### **Heuristic Baselines**

* `greedy` – marginal-gain greedy
* `local_search` – 1-flip hill-climber with random restarts

#### **Exact/Exhaustive**

* `bnb` – built-in branch-and-bound

#### **Industrial MIP/QUBO Engines**

* `cpsat` – OR-Tools CP-SAT

---

### 📊 4. Benchmark tooling

Includes:

* `basic_suite()` – a curated cross-regime benchmark
* `run_suite()` – benchmarking harness
* `save_results_csv()` / `load_results_csv()`
* visualization tools:

  * **Performance profiles** (Dolan–Moré)
  * **Anytime curves** (best-so-far score over time)

This makes SYNDEOPT a *complete benchmarking platform* for binary quadratic optimization problems.

---

### 📁 5. Repository Structure

```
syndeopt/

│

├── README.md

├── pyproject.toml

├── environment.yml

├── setup.sh / setup.bat

│

├── src/syndeopt/

│   ├── core/          # SDSInstance / feasibility / scoring

│   ├── gen/           # problem generators (SDS + QUBO)

│   ├── solvers/       # solver zoo (CP-SAT, greedy, local search, BnB)

│   ├── bench/         # benchmark runner + viz + suites

│   └── __init__.py

│

├── tests/             # pytest test suite

│

└── examples/          # usage examples & tutorials

    ├── run_and_plot.py

    └── run_neurips_benchmark.py
```

---

## 🔧 Installation

### **Option A — Automated setup (recommended for Linux/Mac)**

The easiest way to set up SYNDEOPT is using the provided setup script:

```bash
# Linux/Mac
./setup.sh

# Windows
setup.bat
```

This script will:
- Create a conda environment with the required dependencies
- Install the package in development mode
- Set up pre-commit hooks
- Install development tools (ruff, mypy, pytest)

After running the script, activate the environment:
```bash
conda activate syndeopt
```

### **Option B — Manual conda setup**

```bash
conda env create -f environment.yml
conda activate syndeopt
pip install -e .
```

### **Option C — pip only**

```bash
pip install ortools pandas matplotlib
pip install -e .
```

### **Optional dependencies**

```bash
pip install dimod        # for QUBO conversion utilities (optional)
pip install pulp         # for MIP backends (optional)
```

---

## 🎉 Quick Start

### Create an instance

Using a predefined generator:

```python
from syndeopt.gen import make_dense_deceptive_instance

inst = make_dense_deceptive_instance(n=20, card=(7, 11), seed=0)
```

Or create a custom instance:

```python
from syndeopt.core.instance import SDSInstance, CardBounds

# Custom problem: 3 variables with pairwise synergies
inst = SDSInstance(
    n=3,
    w=[1.0, 2.0, 1.5],  # unary weights
    W={(0, 1): 3.0, (1, 2): 2.0},  # pairwise terms
    precedence=[],  # no precedence constraints
    mutex=[],  # no mutex constraints
    groups={},  # no group constraints
    card=CardBounds(L=1, U=3)  # select 1-3 variables
)
```

### Run a solver

```python
from syndeopt.solvers import get_solver

solver = get_solver("cpsat")
res = solver.solve(inst, budget_sec=5.0, seed=0)
print(res.score, bin(res.mask))
```

### Compare solvers on a suite

```python
from syndeopt.bench.suites import basic_suite
from syndeopt.bench.runner import run_suite
from syndeopt.solvers import list_solvers

suite = basic_suite(seed=0)
solvers = list(list_solvers().keys())

rows = run_suite(suite, solvers, budget_sec=5.0, seed=0)
```

### Plot a performance profile

```python
from syndeopt.bench.viz import performance_profile

performance_profile(rows, outfile="perf_profile.png")
```

### Plot an anytime curve (e.g., CP-SAT)

```python
res = solver.solve(inst, budget_sec=5.0, seed=0)
from syndeopt.bench.viz import anytime_curve

anytime_curve({"cpsat": res.trace}, outfile="anytime.png")
```

### Run the NeurIPS-style benchmark

Generate publication-ready figures and tables:

```bash
python examples/run_neurips_benchmark.py
```

This produces:

* **`results_neurips.csv`** – raw benchmark data
* **`perf_profile_neurips.png`** – performance profile (score-based)
* **`bar_norm_score_overall.png`** – overall mean normalized score per solver
* **`bar_norm_score_family_<family>.png`** – per-regime barplots
* **`summary_by_solver.csv`** – summary statistics (mean normalized score, best fraction, mean time)
* **`summary_by_solver.tex`** – LaTeX table (ready for paper inclusion)

These outputs are exactly the kind of artifacts used in NeurIPS/ICLR papers: performance profiles show robustness across regimes, normalized scores reveal "who wins where", and summary tables provide easily citable numbers.

---

## 🧪 Research Applications

SYNDEOPT is suitable for:

* Algorithm benchmarking (exact vs heuristic vs metaheuristic)
* QUBO solver comparisons (classical, quantum-inspired, MILP)
* Datasets for RL-for-CO, LNS controllers, meta-solvers
* Hardness analysis of problem landscapes
* Generating publication-quality results (profiles, anytime curves)

The included generators span diverse problem landscapes:

* tree-structured (DP-friendly)
* decomposable (divide-and-conquer friendly)
* dense deceptive (challenging for heuristics)
* modular (greedy-friendly)
* QUBO-native (random, planted, Max-Cut)
* constraint-rich SDS (with precedence, mutex, groups)

---

## 📋 Scope and Limitations

SYNDEOPT is a **specialized tool** for binary quadratic optimization problems. Understanding its scope helps set appropriate expectations:

### ✅ What SYNDEOPT Supports

* **Problem type**: Binary variables (x ∈ {0,1}ⁿ) with linear + pairwise quadratic objective
* **Constraints**: Cardinality bounds, precedence (DAG), mutex pairs, and group constraints
* **Objective**: Single-objective maximization
* **Problem size**: Designed for small to medium instances (typically n ≤ 100)
* **Extensibility**: Easy to add new solvers via the `@register` decorator

### ❌ What SYNDEOPT Does NOT Support

* **Higher-order interactions**: Only unary and pairwise terms (no 3-way, 4-way, etc.)
* **Continuous variables**: Binary variables only
* **Mixed-integer problems**: No continuous or integer variables
* **Multi-objective optimization**: Single objective only
* **Stochastic/robust optimization**: Deterministic problems only
* **Time-dependent problems**: Static problems only
* **Custom constraint types**: Limited to the 4 predefined constraint types (cardinality, precedence, mutex, groups)
* **Quantum hardware**: Only quantum-inspired classical solvers (no actual quantum devices)
* **Instance import**: No direct import from MPS/LP file formats (but you can construct `SDSInstance` objects programmatically from any data source)

### 🎯 When to Use SYNDEOPT

SYNDEOPT is ideal when you need to:
- Benchmark solvers on binary quadratic problems with cardinality/precedence/mutex/group constraints
- Generate **any** problem within the SDS class by specifying arbitrary parameters (fully parametric)
- Compare algorithm performance across different problem landscapes
- Create reproducible benchmark instances for research
- Study how different solvers handle structured vs. unstructured problems

SYNDEOPT is **not** suitable for:
- Problems requiring higher-order interactions (3-way+ synergies)
- Mixed-integer or continuous optimization
- Multi-objective optimization
- Problems with constraint types beyond the 4 supported (cardinality, precedence, mutex, groups)
- Very large-scale problems (n > 1000)

**Key point**: Within the SDS problem class (binary quadratic with the 4 constraint types), SYNDEOPT can generate **any** problem instance—the 8 predefined generators are just convenient presets. You have full parametric control over all problem parameters.

---

## 🛣️ Roadmap

### Completed
* ✔ Core solver suite (greedy, local search, BnB, CP-SAT)
* ✔ QUBO-native generators (random, planted, Max-Cut)
* ✔ Visualization (performance profiles & anytime curves)

### Future Considerations
* Additional constraint types (if there's demand)
* Support for importing instances from standard formats
* More specialized solvers (e.g., graph-based, SAT-based)
* Performance optimizations for larger instances

---

## 🙏 Contributing

We welcome:

* new solvers
* new instance families
* better visualizations
* documentation improvements
* bug fixes
* benchmarking scripts

Open a PR or create an issue!

---

## 📜 License

MIT License — free for academic and commercial use.

---

If you'd like, I can also generate:

* a **CITATION.cff**
* a **docs/** website (mkdocs or sphinx)
* a logo for SYNDEOPT
* GitHub Actions CI (tests + lint)

Just tell me!
