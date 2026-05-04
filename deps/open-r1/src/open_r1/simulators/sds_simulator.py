"""
SDS Simulator for the rewards_unified_v2.py framework.

This wraps the SDS problem as a simulator that can be used with the unified reward system.
Uses syndeopt as the core implementation.
"""

import sys
import os
from typing import Dict, List, Any, Optional
from .base import BaseSimulator

# Add syndeopt to path - resolve from workspace root
# Try multiple possible paths to handle different directory structures
# 1. Try importing syndeopt directly (if installed as package)
# 2. Try relative path from llm-finetuning root (deps/syndeopt/src)
# 3. Try relative path from open-r1 root (../syndeopt/src or ../../syndeopt/src)
try:
    from syndeopt.core.instance import SDSInstance, CardBounds
    from syndeopt.core.scoring import score as syndeopt_score
    from syndeopt.core.feasibility import feasible as syndeopt_feasible
except ImportError:
    # syndeopt not found, try adding paths
    _current_file = os.path.abspath(__file__)
    # This file is at: .../open-r1/src/open_r1/simulators/sds_simulator.py
    # Try going up to find workspace root
    _possible_paths = [
        # Path 1: llm-finetuning/deps/open-r1/src/open_r1/simulators/sds_simulator.py
        #         -> llm-finetuning/deps/syndeopt/src
        os.path.join(os.path.dirname(_current_file), '../../../../syndeopt/src'),
        # Path 2: llm-finetuning/deps/open-r1/src/open_r1/simulators/sds_simulator.py
        #         -> llm-finetuning/deps/syndeopt/src (alternative)
        os.path.join(os.path.dirname(_current_file), '../../../../../deps/syndeopt/src'),
        # Path 3: /workspace/open-r1/src/open_r1/simulators/sds_simulator.py
        #         -> /workspace/syndeopt/src (if syndeopt is cloned alongside open-r1)
        os.path.join(os.path.dirname(_current_file), '../../../../syndeopt/src'),
        # Path 4: /workspace/open-r1/src/open_r1/simulators/sds_simulator.py
        #         -> /workspace/llm-finetuning/deps/syndeopt/src (if full repo is mounted)
        os.path.join(os.path.dirname(_current_file), '../../../../../llm-finetuning/deps/syndeopt/src'),
    ]
    
    _syndeopt_found = False
    for _syndeopt_path in _possible_paths:
        _syndeopt_path = os.path.abspath(_syndeopt_path)
        if os.path.exists(_syndeopt_path) and os.path.exists(os.path.join(_syndeopt_path, 'syndeopt')):
            if _syndeopt_path not in sys.path:
                sys.path.insert(0, _syndeopt_path)
            try:
                from syndeopt.core.instance import SDSInstance, CardBounds
                from syndeopt.core.scoring import score as syndeopt_score
                from syndeopt.core.feasibility import feasible as syndeopt_feasible
                _syndeopt_found = True
                break
            except ImportError:
                continue
    
    if not _syndeopt_found:
        raise ImportError(
            "Could not find syndeopt module. Please ensure syndeopt is either:\n"
            "1. Installed as a package: pip install -e /path/to/syndeopt\n"
            "2. Available at one of these paths relative to this file:\n"
            "   - ../../../../syndeopt/src\n"
            "   - ../../../../../deps/syndeopt/src\n"
            "   - ../../../../../llm-finetuning/deps/syndeopt/src\n"
            f"Current file location: {_current_file}"
        )


def normalize_sds_score(
    score: float,
    requirements: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Normalize a raw SDS score into [0, 1] using the simulator's heuristic.

    Shared between the default hard-gated reward and the soft nominal ablation
    so both paths use identical score scaling.
    """
    config = config or {}
    normalization_variant = config.get(
        "normalization_variant",
        "avg_positive_times_max_pairs",
    )

    weights_list = requirements.get("weights", [])
    n = requirements.get("n_variables", len(weights_list))
    card_bounds = requirements.get("cardinality_bounds", [0, n])
    U = card_bounds[1] if len(card_bounds) > 1 else n

    weights_abs_sum = sum(
        abs(w) for w in weights_list
        if w is not None and isinstance(w, (int, float))
    )

    interactions_dict = requirements.get("interactions", {})
    interactions_abs_sum = sum(
        abs(v) for v in interactions_dict.values()
        if v is not None and isinstance(v, (int, float))
    )

    if weights_abs_sum > 1e-6 or interactions_abs_sum > 1e-6:
        positive_weights = sorted(
            [
                w for w in weights_list
                if w is not None and isinstance(w, (int, float)) and w > 0
            ],
            reverse=True,
        )
        max_weight_contribution = sum(positive_weights[:U]) if positive_weights else 0.0

        positive_interactions = [
            v for v in interactions_dict.values()
            if v is not None and isinstance(v, (int, float)) and v > 0
        ]
        if positive_interactions and U > 1:
            max_pairs = min(len(positive_interactions), U * (U - 1) // 2)
            if normalization_variant == "topk_positive_interactions":
                max_interaction_contribution = sum(
                    sorted(positive_interactions, reverse=True)[:max_pairs]
                )
            else:
                avg_positive = sum(positive_interactions) / len(positive_interactions)
                max_interaction_contribution = avg_positive * max_pairs
        else:
            max_interaction_contribution = 0.0

        normalization_base = max_weight_contribution + max_interaction_contribution
        if normalization_base > 1e-6:
            normalized_score = score / normalization_base
        else:
            normalized_score = score / max(weights_abs_sum + interactions_abs_sum, 1.0)
    else:
        if abs(score) > 1e-6:
            normalized_score = 0.5 * (1.0 + score / (1.0 + abs(score)))
        else:
            normalized_score = 0.5

    return max(0.0, min(1.0, normalized_score))

class SDSSimulator(BaseSimulator):
    """Synergistic Dependency Selection (SDS) Simulator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "sds"
    
    def _setup(self):
        """Setup SDS simulator."""
        # No special setup needed
        pass
    
    def simulate(self, selection: List[int], requirements: Dict[str, Any]) -> Dict[str, float]:
        """
        Simulate SDS problem with given selection.
        
        Args:
            selection: List of selected variable indices
            requirements: Problem requirements including n_variables, cardinality_bounds, etc.
            
        Returns:
            Dict with score, feasibility, and other metrics
        """
        try:
            # Validate selection input
            if not isinstance(selection, list):
                selection = []
            if selection is None:
                selection = []
            
            # Create SDS instance from requirements
            n = requirements.get("n_variables", len(selection) if selection else 0)
            if n <= 0:
                raise ValueError(f"Invalid n_variables: {n}")
            
            w_raw = requirements.get("weights", [0.0] * n)
            
            # Filter out None values and convert to float
            # Ensure all weights are numeric (not None)
            w = []
            for weight in w_raw:
                if weight is None:
                    w.append(0.0)  # Replace None with 0.0
                else:
                    try:
                        w.append(float(weight))
                    except (ValueError, TypeError):
                        w.append(0.0)  # Replace invalid values with 0.0
            
            # Ensure weights list has exactly n elements
            if len(w) < n:
                # Pad with zeros if too short
                w = list(w) + [0.0] * (n - len(w))
            elif len(w) > n:
                # Truncate if too long
                w = w[:n]
            
            W = requirements.get("interactions", {})
            if W is None or not isinstance(W, dict):
                W = {}
            
            precedence = requirements.get("precedence", [])
            mutex = requirements.get("mutex", [])
            groups = requirements.get("groups", {})
            card_bounds = tuple(requirements.get("cardinality_bounds", [0, n]))
            
            # Validate and filter constraint indices to be within [0, n)
            # Handle None or non-list precedence
            if precedence is None or not isinstance(precedence, list):
                precedence = []
            precedence_filtered = []
            for item in precedence:
                if item is None or not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                try:
                    i, j = item
                    if isinstance(i, (int, float)) and isinstance(j, (int, float)) and 0 <= int(i) < n and 0 <= int(j) < n:
                        precedence_filtered.append((int(i), int(j)))
                except (ValueError, TypeError):
                    continue
            precedence = precedence_filtered
            
            # Handle None or non-list mutex
            if mutex is None or not isinstance(mutex, list):
                mutex = []
            mutex_filtered = []
            for item in mutex:
                if item is None or not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                try:
                    a, b = item
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and 0 <= int(a) < n and 0 <= int(b) < n:
                        mutex_filtered.append((int(a), int(b)))
                except (ValueError, TypeError):
                    continue
            mutex = mutex_filtered
            
            # Filter groups to only include valid indices
            # Handle None or non-dict groups
            if groups is None or not isinstance(groups, dict):
                groups = {}
            groups_filtered = {}
            for gid, members in groups.items():
                # Skip if members is None or not a list
                if members is None or not isinstance(members, list):
                    continue
                valid_members = [int(i) for i in members if isinstance(i, (int, float)) and 0 <= int(i) < n]
                if len(valid_members) >= 2:  # Only keep groups with at least 2 members
                    groups_filtered[gid] = valid_members
            groups = groups_filtered
            
            # Convert interactions dict to proper format and filter invalid indices
            # Also filter out None values and ensure values are numeric
            W_dict = {}
            for key, value in W.items():
                # Skip None values or non-numeric values
                if value is None:
                    continue
                try:
                    # Ensure value is numeric
                    value = float(value)
                except (ValueError, TypeError):
                    continue
                
                if isinstance(key, str) and ',' in key:
                    try:
                        i, j = map(int, key.split(','))
                        if 0 <= i < n and 0 <= j < n:
                            W_dict[(i, j)] = value
                    except (ValueError, IndexError):
                        continue
                elif isinstance(key, tuple):
                    try:
                        i, j = key
                        if 0 <= i < n and 0 <= j < n:
                            W_dict[(i, j)] = value
                    except (ValueError, TypeError):
                        continue
            
            # Create SDS instance using syndeopt
            inst = SDSInstance(
                n=n,
                w=w,
                W=W_dict,
                precedence=precedence,
                mutex=mutex,
                groups=groups,
                card=CardBounds(L=card_bounds[0], U=card_bounds[1])
            )
            
            # Convert selection to bitmask
            x_bits = 0
            for var in selection:
                if 0 <= var < n:
                    x_bits |= (1 << var)
            
            # Check feasibility using syndeopt
            is_feasible = syndeopt_feasible(inst, x_bits)
            
            # Keep the raw objective score even for infeasible selections.
            # The default reward path still hard-gates infeasible outputs later,
            # but the soft-gate ablation needs the ungated score signal.
            score_value = syndeopt_score(inst, x_bits)
            
            # Calculate cardinality (use actual bit count, not selection length, since we filtered invalid indices)
            cardinality = x_bits.bit_count()
            
            # Calculate constraint violations
            violations = 0.0
            if not is_feasible:
                # Count constraint violations
                k = x_bits.bit_count()
                L, U = inst.card.L, inst.card.U
                if k < L or k > U:
                    violations += abs(k - max(L, min(U, k)))
                
                # Check precedence violations
                for i, j in precedence:
                    if ((x_bits >> j) & 1) and not ((x_bits >> i) & 1):
                        violations += 1.0
                
                # Check mutex violations
                for a, b in mutex:
                    if ((x_bits >> a) & 1) and ((x_bits >> b) & 1):
                        violations += 1.0
                
                # Check group violations
                for gid, members in groups.items():
                    cnt = sum((x_bits >> i) & 1 for i in members)
                    if cnt > 1:
                        violations += (cnt - 1)
            
            return {
                "score": float(score_value),
                "feasible": is_feasible,
                "cardinality": cardinality,
                "constraint_violations": violations,
                "selection": selection
            }
            
        except Exception as e:
            import traceback
            print(f"SDS Simulator error: {e}")
            print(f"  Selection: {selection}")
            print(f"  Requirements keys: {list(requirements.keys()) if isinstance(requirements, dict) else 'N/A'}")
            if isinstance(requirements, dict):
                n = requirements.get("n_variables", "unknown")
                w_len = len(requirements.get("weights", [])) if isinstance(requirements.get("weights"), list) else "N/A"
                print(f"  n_variables: {n}, weights length: {w_len}")
            print(f"  Traceback: {traceback.format_exc()}")
            return {
                "score": 0.0,
                "feasible": False,
                "cardinality": 0,
                "constraint_violations": 1.0,
                "selection": selection
            }
    
    def validate_design(self, selection: List[int], requirements: Dict[str, Any]) -> bool:
        """Validate SDS selection format."""
        if not isinstance(selection, list):
            return False
        
        n = requirements.get("n_variables", 0)
        for var in selection:
            if not isinstance(var, int) or var < 0 or var >= n:
                return False
        
        return True

    def _calculate_reward(self, results: Dict[str, float], requirements: Dict[str, Any]) -> float:
        """Calculate SDS reward: score maximization + constraint penalties."""
        score = results["score"]
        violations = results["constraint_violations"]
        feasible = results["feasible"]
        
        if not feasible:
            return 0.0
        
        normalized_score = normalize_sds_score(score, requirements, self.config)
        
        # Apply constraint penalty
        penalty = min(violations, 1.0)  # Cap penalty at 1.0
        final_score = max(0.0, normalized_score - penalty)
        
        # Final safety check: ensure reward is in [0, 1]
        return max(0.0, min(1.0, final_score))
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Return SDS simulator capabilities."""
        return {
            "input_format": "List of selected variable indices",
            "constraints": ["cardinality", "precedence", "mutex", "groups"],
            "objective": "maximize_score",
            "feasibility_check": True
        }
