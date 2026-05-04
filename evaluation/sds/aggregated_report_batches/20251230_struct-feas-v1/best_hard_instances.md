# Best Hard Instances by Seed

This document contains the reasoning traces and generated code for the highest-scoring Hard difficulty instances from each training seed.

---

## Seed 101

**UUID**: `sds_random_sds_009783`

**LLM Score**: 1401.87

**VBS Score**: 1413.86

**Optimality Gap**: 0.85%

**Execution Time**: 3.4815s

**Mission Summary**: n_vars=80, cardinality=[10, 45], precedence=0, mutex=4, groups=0, interactions=1935

### Reasoning Trace

```
To solve the Synergistic Dependency Selection (SDS) optimization problem, we need to select a subset of variables that maximizes the total value while respecting various constraints such as cardinality bounds, precedence, mutual exclusion, and groups. Here's a step-by-step reasoning process to achieve this:

1. **Input Parsing**: Read the input JSON to extract the requirements and catalog. The requirements include constraints like cardinality bounds, precedence, mutex, and groups. The catalog contains the variables and their interactions.

2. **Feasibility Check**: Ensure that any selected subset of variables respects all constraints. This includes checking the cardinality bounds, precedence, mutex, and group constraints.

3. **Objective Function**: Define the objective function to maximize the total value of the selected variables, including their individual weights and pairwise interactions.

4. **Search Algorithm**: Implement a search algorithm to find the best feasible solution. Given the constraints and the need for optimization, a simulated annealing approach is suitable. This approach allows for exploration of the solution space while respecting constraints and gradually converging to a near-optimal solution.

5. **Output**: Print the best feasible selection to stdout in the required JSON format.

**Simulated Annealing Approach**:
- Start with a random feasible solution.
- Iteratively propose new solutions by adding or removing variables.
- Accept new solutions based on the Metropolis criterion, which allows for occasional acceptance of worse solutions to escape local optima.
- Cool the temperature gradually to reduce the acceptance of worse solutions over time.

**Edge Cases**:
- Ensure that the initial solution and all proposed solutions respect the constraints.
- Handle the sparse interactions dictionary efficiently.
```

### Generated Code

```python
import sys
import json
import random
import math

def solve_sds():
    # Read input from stdin
    input_json = sys.stdin.read()
    data = json.loads(input_json)
    
    requirements = data['requirements']
    catalog = data['catalog']
    
    n_variables = requirements['n_variables']
    cardinality_bounds = requirements['cardinality_bounds']
    precedence = requirements['precedence']
    mutex = requirements['mutex']
    groups = requirements['groups']
    interactions = catalog['interactions']
    
    def is_feasible(selection):
        # Check cardinality bounds
        if len(selection) < cardinality_bounds[0] or len(selection) > cardinality_bounds[1]:
            return False
        # Check precedence constraints
        for i, j in precedence:
            if j in selection and i not in selection:
                return False
        # Check mutex constraints
        for a, b in mutex:
            if a in selection and b in selection:
                return False
        # Check group constraints
        for group in groups.values():
            selected_in_group = [var for var in group if var in selection]
            if len(selected_in_group) > 1:
                return False
        return True
    
    def calculate_value(selection):
        value = sum(catalog['variables'][var]['weight'] for var in selection)
        for i in selection:
            for j in selection:
                if f"{i},{j}" in interactions:
                    value += interactions[f"{i},{j}"]
        return value
    
    # Generate an initial feasible solution
    initial_selection = random.sample(range(n_variables), random.randint(cardinality_bounds[0], cardinality_bounds[1]))
    while not is_feasible(initial_selection):
        initial_selection = random.sample(range(n_variables), random.randint(cardinality_bounds[0], cardinality_bounds[1]))
    
    current_selection = initial_selection
    current_value = calculate_value(current_selection)
    best_selection = current_selection
    best_value = current_value
    
    # Simulated annealing
    temperature = 1000
    cooling_rate = 0.995
    
    for _ in range(10000):
        # Propose a new solution
        new_selection = current_selection[:]
        if random.choice([True, False]):
            # Add a variable if possible
            if len(new_selection) < cardinality_bounds[1]:
                candidate = random.choice([i for i in range(n_variables) if i not in new_selection])
                new_selection.append(candidate)
        else:
            # Remove a variable if possible
            if len(new_selection) > cardinality_bounds[0]:
                candidate = random.choice(new_selection)
                new_selection.remove(candidate)
        
        # Ensure the new solution is feasible
        if not is_feasible(new_selection):
            continue
        
        new_value = calculate_value(new_selection)
        
        # Accept the new solution based on the Metropolis criterion
        if new_value > current_value or random.random() < math.exp((new_value - current_value) / temperature):
            current_selection = new_selection
            current_value = new_value
        
        # Update the best solution
        if current_value > best_value:
            best_selection = current_selection
            best_value = current_value
        
        # Cool down
        temperature *= cooling_rate
    
    # Output the best feasible solution
    result = {
        "selection": {
            "variables": best_selection
        }
    }
    sys.stdout.write(json.dumps(result))

if __name__ == "__main__":
    solve_sds()
```

---

## Seed 202

**UUID**: `sds_random_sds_009229`

**LLM Score**: 1702.43

**VBS Score**: 1740.90

**Optimality Gap**: 2.21%

**Execution Time**: 0.3536s

**Mission Summary**: n_vars=91, cardinality=[14, 72], precedence=9, mutex=3, groups=0, interactions=2748

### Reasoning Trace

```
**Deconstruct**:
The Synergistic Dependency Selection (SDS) problem involves selecting a subset of variables to maximize a total value, considering individual weights and pairwise interactions (synergies/penalties). The selection must respect constraints such as precedence, mutual exclusion, and group constraints, along with cardinality bounds.

- **Variables**: Each variable has an ID, weight, and a list of neighboring variables with interactions.
- **Interactions**: A sparse dictionary of pairwise interaction weights between variables.
- **Constraints**:
  - **Precedence**: If a variable is selected, all its predecessors must also be selected.
  - **Mutex**: At most one variable in a mutex pair can be selected.
  - **Groups**: At most one variable in a group can be selected.
  - **Cardinality**: The number of selected variables must be within a specified range.

**Hypothesize**:
An initial approach could be a greedy algorithm that selects variables based on their individual weights and interactions, while respecting constraints. However, this might not yield the optimal solution due to the interdependencies.

A more robust approach could be a simulated annealing algorithm, which allows for exploration of the solution space by accepting worse solutions probabilistically. This can help escape local optima and find a more globally optimal solution.

**Critique & Refine**:
- **Greedy Approach**: May get stuck in local optima due to the complex interplay of constraints and interactions.
- **Simulated Annealing**: Introduces randomness to explore the solution space, allowing for a more thorough search. However, it must be carefully tuned to balance exploration and exploitation.

**Simulate**:
Consider a small hypothetical example with 5 variables and a few constraints:
- Variables: A, B, C, D, E
- Weights: A=3, B=2, C=1, D=4, E=5
- Interactions: A-B=1, B-C=-1, C-D=2, D-E=-2
- Constraints: Precedence (A->B), Mutex (B, C), Group (D, E)

A feasible selection could be {A, D}, which respects all constraints and has a total value of 3 + 4 = 7.

**Finalize**:
The input and output requirements are clear. The solver must read the input JSON, process the constraints, and output the best feasible selection. The use of simulated annealing ensures a robust solution that respects all constraints.
```

### Generated Code

```python
import sys
import random
import math
import json

def solve_sds():
    import sys
    import random
    import math
    import json

    # Read input
    input_data = sys.stdin.read()
    data = json.loads(input_data)
    
    requirements = data["requirements"]
    catalog = data["catalog"]
    
    n_variables = requirements["n_variables"]
    min_card, max_card = requirements["cardinality_bounds"]
    precedence = requirements["precedence"]
    mutex = requirements["mutex"]
    groups = requirements["groups"]
    variables = catalog["variables"]
    interactions = catalog["interactions"]
    
    # Function to calculate the total value of a selection
    def calculate_value(selection):
        total_value = 0
        for idx in selection:
            total_value += variables[idx]["weight"]
        for i in range(len(selection)):
            for j in range(i + 1, len(selection)):
                interaction_key = f"{min(selection[i], selection[j])},{max(selection[i], selection[j])}"
                if interaction_key in interactions:
                    total_value += interactions[interaction_key]
        return total_value
    
    # Function to check feasibility
    def is_feasible(selection):
        # Precedence
        for i, j in precedence:
            if j in selection and i not in selection:
                return False
        # Mutex
        for a, b in mutex:
            if a in selection and b in selection:
                return False
        # Groups
        for group_id, group_vars in groups.items():
            selected_in_group = [var for var in group_vars if var in selection]
            if len(selected_in_group) > 1:
                return False
        # Cardinality
        if not (min_card <= len(selection) <= max_card):
            return False
        return True
    
    # Simulated Annealing
    initial_selection = random.sample(list(range(n_variables)), random.randint(min_card, max_card))
    while not is_feasible(initial_selection):
        initial_selection = random.sample(list(range(n_variables)), random.randint(min_card, max_card))
    
    current_selection = initial_selection
    current_value = calculate_value(current_selection)
    T = 1000
    cooling_rate = 0.995
    
    while T > 1:
        # Propose a new selection
        new_selection = current_selection[:]
        if random.random() < 0.5 and len(new_selection) < max_card:
            candidate = random.choice(list(set(range(n_variables)) - set(new_selection)))
            new_selection.append(candidate)
        elif len(new_selection) > min_card:
            candidate = random.choice(new_selection)
            new_selection.remove(candidate)
        
        if is_feasible(new_selection):
            new_value = calculate_value(new_selection)
            delta = new_value - current_value
            if delta > 0 or random.random() < math.exp(delta / T):
                current_selection = new_selection
                current_value = new_value
        T *= cooling_rate
    
    # Output the best found selection
    print(json.dumps({"selection": {"variables": current_selection}}))

if __name__ == "__main__":
    solve_sds()
```

---

## Seed 303

**UUID**: `sds_random_sds_009064`

**LLM Score**: 1684.53

**VBS Score**: 1686.85

**Optimality Gap**: 0.14%

**Execution Time**: 0.4917s

**Mission Summary**: n_vars=94, cardinality=[28, 76], precedence=0, mutex=1, groups=0, interactions=2190

### Reasoning Trace

```
To solve the Synergistic Dependency Selection (SDS) optimization problem, we need to develop a Python script that reads a JSON input, processes the constraints, and finds a subset of variables that maximizes the total value while respecting the given constraints. The constraints include cardinality bounds, precedence, mutual exclusion, and group constraints. Given the complexity and the need to respect constraints, a simulated annealing approach seems suitable due to its ability to explore the solution space and escape local optima.

### Step-by-Step Reasoning:
1. **Parse Input**: Read the JSON input and extract the requirements and catalog.
2. **Initialize Solution**: Randomly select a feasible initial solution that respects the cardinality bounds.
3. **Simulated Annealing**:
   - Start with an initial temperature.
   - Perform a series of random moves (add or remove a variable while respecting cardinality bounds).
   - Accept the new solution if it improves the objective function.
   - Accept the new solution with a certain probability if it decreases the objective function (to escape local optima).
   - Cool the temperature according to a cooling schedule.
   - Repeat until convergence or a sufficient number of iterations.
4. **Output the Best Solution**: Print the best solution found.

### Constraints Handling:
- **Cardinality Constraints**: Ensure the number of selected variables is within the specified range.
- **Precedence Constraints**: If a variable is selected, all its predecessors must also be selected.
- **Mutex Constraints**: Ensure that at most one of the mutually exclusive variables is selected.
- **Group Constraints**: Ensure that at most one variable from each group is selected.

### Mental Dry Run:
Consider a small example with 5 variables, cardinality bounds [2, 3], one mutex constraint (variables 1 and 2), and one group constraint (variables 3 and 4). The algorithm should respect these constraints while maximizing the total weight and interactions.
```

### Generated Code

```python
import json
import random
import math

def solve_sds():
    import sys
    data = json.load(sys.stdin)
    
    requirements = data['requirements']
    catalog = data['catalog']
    
    n_variables = requirements['n_variables']
    min_cardinality, max_cardinality = requirements['cardinality_bounds']
    precedence_constraints = requirements['precedence']
    mutex_constraints = requirements['mutex']
    group_constraints = requirements['groups']
    
    def is_feasible(solution):
        # Check cardinality
        if not (min_cardinality <= sum(solution) <= max_cardinality):
            return False
        # Check precedence constraints
        for i, j in precedence_constraints:
            if solution[j] and not solution[i]:
                return False
        # Check mutex constraints
        for a, b in mutex_constraints:
            if solution[a] and solution[b]:
                return False
        # Check group constraints
        for group_vars in group_constraints.values():
            selected = [v for v in group_vars if solution[v]]
            if len(selected) > 1:
                return False
        return True
    
    def calculate_score(solution):
        score = sum(catalog['variables'][v]['weight'] for v in range(n_variables) if solution[v])
        for i in range(n_variables):
            if solution[i]:
                for j in catalog['variables'][i]['neighbors']:
                    if solution[j]:
                        score += catalog['interactions'].get(f"{min(i, j)},{max(i, j)}", 0)
        return score
    
    # Initialize a random feasible solution
    current_solution = [random.choice([True, False]) for _ in range(n_variables)]
    while not is_feasible(current_solution):
        current_solution = [random.choice([True, False]) for _ in range(n_variables)]
    
    current_score = calculate_score(current_solution)
    best_solution = current_solution[:]
    best_score = current_score
    
    # Simulated Annealing
    T = 1000  # Initial temperature
    cooling_rate = 0.99
    n_iterations = 1000
    
    for _ in range(n_iterations):
        # Generate a neighbor solution
        neighbor_solution = current_solution[:]
        idx = random.randint(0, n_variables - 1)
        neighbor_solution[idx] = not neighbor_solution[idx]
        while not is_feasible(neighbor_solution):
            idx = random.randint(0, n_variables - 1)
            neighbor_solution[idx] = not neighbor_solution[idx]
        
        neighbor_score = calculate_score(neighbor_solution)
        
        if neighbor_score > current_score:
            current_solution = neighbor_solution[:]
            current_score = neighbor_score
            if current_score > best_score:
                best_solution = current_solution[:]
                best_score = current_score
        else:
            if random.random() < math.exp((neighbor_score - current_score) / T):
                current_solution = neighbor_solution[:]
                current_score = neighbor_score
        
        # Cool
        T *= cooling_rate
    
    # Prepare output
    selected_variables = [v for v in range(n_variables) if best_solution[v]]
    result = {
        "selection": {
            "variables": selected_variables
        }
    }
    json.dump(result, sys.stdout)

solve_sds()
```

---

