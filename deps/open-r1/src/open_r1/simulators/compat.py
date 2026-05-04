"""
Backward compatibility layer for existing reward functions.
This allows existing code to work without changes while providing
access to the new simulator interface.
"""

from typing import Any, Dict, List, Optional
from .registry import registry


def eps_simulator_reward(completions, **kwargs) -> List[float]:
    """
    Backward-compatible EPS simulator reward.
    Uses new simulator interface internally.
    """
    # Extract mission data
    missions = []
    if "mission" in kwargs:
        if isinstance(kwargs["mission"], list):
            missions = kwargs["mission"]
        else:
            missions = [kwargs["mission"]] * len(completions)
    else:
        missions = [{}] * len(completions)
    
    rewards = []
    for idx, (comp, mission) in enumerate(zip(completions, missions)):
        try:
            # Extract design from completion
            text = comp[0].get("content", "")
            design = _extract_eps_design(text)
            
            if not design:
                rewards.append(0.0)
                continue
            
            # Use new simulator interface
            simulator = registry.get_simulator("eps")
            reward = simulator.get_reward(design, mission)
            rewards.append(reward)
            
        except Exception as e:
            print(f"EPS simulation error for completion {idx}: {e}")
            rewards.append(0.0)
    
    return rewards


def beams2d_simulator_reward(completions, **kwargs) -> List[float]:
    """
    Backward-compatible Beams2D simulator reward.
    Uses new simulator interface internally.
    """
    # Extract mission data
    missions = []
    if "mission" in kwargs:
        if isinstance(kwargs["mission"], list):
            missions = kwargs["mission"]
        else:
            missions = [kwargs["mission"]] * len(completions)
    else:
        missions = [{}] * len(completions)
    
    rewards = []
    for idx, (comp, mission) in enumerate(zip(completions, missions)):
        try:
            # Extract design from completion
            text = comp[0].get("content", "")
            design = _extract_beams2d_design(text)
            
            if design is None:
                rewards.append(0.0)
                continue
            
            # Use new simulator interface
            simulator = registry.get_simulator("beams2d")
            reward = simulator.get_reward(design, mission)
            rewards.append(reward)
            
        except Exception as e:
            print(f"Beams2D simulation error for completion {idx}: {e}")
            rewards.append(0.0)
    
    return rewards


def knapsack_simulator_reward(completions, **kwargs) -> List[float]:
    """
    Backward-compatible Knapsack simulator reward.
    Uses new simulator interface internally.
    """
    # Extract mission data
    missions = []
    if "mission" in kwargs:
        if isinstance(kwargs["mission"], list):
            missions = kwargs["mission"]
        else:
            missions = [kwargs["mission"]] * len(completions)
    else:
        missions = [{}] * len(completions)
    
    rewards = []
    for idx, (comp, mission) in enumerate(zip(completions, missions)):
        try:
            # Extract selection from completion
            text = comp[0].get("content", "")
            selection = _extract_knapsack_selection(text)
            
            if not selection:
                rewards.append(0.0)
                continue
            
            # Use new simulator interface
            simulator = registry.get_simulator("knapsack")
            reward = simulator.get_reward(selection, mission)
            rewards.append(reward)
            
        except Exception as e:
            print(f"Knapsack simulation error for completion {idx}: {e}")
            rewards.append(0.0)
    
    return rewards


def _extract_eps_design(text: str) -> Optional[str]:
    """Extract EPS design code from text."""
    import re
    from .utils import extract_block, normalize
    
    # Extract answer block
    answer = extract_block(text, "answer")
    if not answer:
        return None
    
    # Extract design components (simplified version)
    # This would need to be more robust in practice
    try:
        # Look for design code pattern
        design_match = re.search(r'design[:\s]*([0-9]{4})', answer, re.IGNORECASE)
        if design_match:
            return design_match.group(1)
        
        # Fallback: try to extract components and build design code
        # This is a simplified version - you'd want the full logic from rewards_eps.py
        return None
        
    except Exception:
        return None


def _extract_beams2d_design(text: str) -> Optional[Any]:
    """Extract Beams2D design matrix from text."""
    import re
    import numpy as np
    from .utils import extract_block
    
    # Extract answer block
    answer = extract_block(text, "answer")
    if not answer:
        return None
    
    try:
        # Parse design matrix (simplified version)
        matrix_lines = []
        in_matrix_section = False
        
        for line in answer.split('\n'):
            line = line.strip()
            if "Design matrix" in line:
                in_matrix_section = True
                continue
            
            if in_matrix_section and line and all(c in '01' for c in line):
                matrix_lines.append(line)
            elif in_matrix_section and line and not all(c in '01' for c in line):
                break
        
        if not matrix_lines:
            return None
        
        # Convert to numpy array
        design = np.array([[float(c) for c in row] for row in matrix_lines])
        return design
        
    except Exception:
        return None


def _extract_knapsack_selection(text: str) -> Optional[List[str]]:
    """Extract knapsack selection from text."""
    import re
    from .utils import extract_block
    
    # Extract answer block
    answer = extract_block(text, "answer")
    if not answer:
        return None
    
    try:
        # Look for selected items
        for line in answer.split('\n'):
            if 'Selected:' in line:
                items_str = line.split('Selected:')[1].strip()
                return [item.strip() for item in items_str.split(',')]
        
        return None
        
    except Exception:
        return None
