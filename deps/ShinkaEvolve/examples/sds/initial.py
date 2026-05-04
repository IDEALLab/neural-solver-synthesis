"""
Initial solution for Synergistic Dependency Selection (SDS) problem.
The code in EVOLVE-BLOCK will be evolved by ShinkaEvolve.
"""

import contextlib
import json
import sys


# EVOLVE-BLOCK-START
def solve_sds():  # noqa: PLR0912, PLR0915
    """
    Solve the Synergistic Dependency Selection problem.
    
    Reads from stdin: JSON with "requirements" and "catalog"
    Outputs to stdout: JSON with "selection" containing "variables" list
    
    This is a greedy baseline - evolution will improve this.
    """
    # Read input
    input_data = json.load(sys.stdin)
    requirements = input_data.get("requirements", {})
    _catalog = input_data.get("catalog", {})
    
    # Extract constraints
    weights = requirements.get("weights", [])
    interactions = requirements.get("interactions", {})
    cardinality_bounds = requirements.get("cardinality_bounds", [0, len(weights)])
    mutex = requirements.get("mutex", [])
    groups = requirements.get("groups", {})
    precedence = requirements.get("precedence", [])
    
    # Simple greedy: select items with highest weights first
    # This is a baseline - evolution will find better strategies
    n = len(weights)
    selected = []
    
    # Create score list (weight + interaction potential)
    scores = []
    for i in range(n):
        score = weights[i] if i < len(weights) else 0.0
        # Add potential interaction value
        for k, v in interactions.items():
            try:
                u, w = map(int, k.split(","))
                if i in (u, w):
                    score += abs(v) * 0.5  # Potential value
            except Exception:
                pass
        scores.append((score, i))
    
    # Sort by score descending
    scores.sort(reverse=True)
    
    # Greedy selection respecting constraints
    selected_set = set()
    for _score, i in scores:
        if len(selected_set) >= cardinality_bounds[1]:
            break
            
        # Check mutex constraints
        can_add = True
        for a, b in mutex:
            if (a == i and b in selected_set) or (b == i and a in selected_set):
                can_add = False
                break
        
        if not can_add:
            continue
            
        # Check group constraints (at most one per group)
        for grp_vars in groups.values():
            if i in grp_vars and len(set(grp_vars).intersection(selected_set)) > 0:
                can_add = False
                break
        
        if not can_add:
            continue
        
        # Check precedence (if j selected, i must be selected)
        precedence_ok = True
        for pred_i, pred_j in precedence:
            if pred_j == i and pred_i not in selected_set:
                precedence_ok = False
                break
        
        if not precedence_ok:
            continue
        
        # Add if we meet minimum cardinality or have room
        if len(selected_set) < cardinality_bounds[1]:
            selected_set.add(i)
            selected.append(i)
    
    # Ensure minimum cardinality
    while len(selected_set) < cardinality_bounds[0] and len(selected_set) < n:
        for i in range(n):
            if i not in selected_set:
                selected_set.add(i)
                selected.append(i)
                break
    
    # Output result
    result = {
        "selection": {
            "variables": sorted(selected)
        }
    }
    print(json.dumps(result))
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_sds(problem_data=None):
    """
    Main function called by the evaluator.
    Can accept problem_data as parameter or read from stdin.
    Returns JSON string (for ShinkaEvolve) or prints to stdout (for direct execution).
    """
    import io  # noqa: PLC0415
    import json  # noqa: PLC0415
    import sys  # noqa: PLC0415
    
    input_data = json.load(sys.stdin) if problem_data is None else problem_data
    
    # Capture stdout from solve_sds
    stdout_capture = io.StringIO()
    old_stdin = sys.stdin
    
    try:
        # Create mock stdin
        sys.stdin = io.StringIO(json.dumps(input_data))
        with contextlib.redirect_stdout(stdout_capture):
            solve_sds()
    finally:
        sys.stdin = old_stdin
    
    result_str = stdout_capture.getvalue()
    return result_str

