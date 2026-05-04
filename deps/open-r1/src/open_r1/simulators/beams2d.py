"""
Beams2D Simulator for topology optimization.
"""

import numpy as np
from typing import Any, Dict, List, Optional
from .base import BaseSimulator

try:
    from engibench.problems.beams2d.v0 import Beams2D
    ENGI_BENCH_AVAILABLE = True
except ImportError:
    ENGI_BENCH_AVAILABLE = False
    Beams2D = None


class Beams2DSimulator(BaseSimulator):
    """
    Beams2D Simulator for structural topology optimization.
    """
    
    domain = "beams2d"
    
    def _setup(self):
        """Setup Beams2D simulator."""
        if not ENGI_BENCH_AVAILABLE:
            print("Warning: EngiBench not available. Beams2D simulator will use fallback mode.")
            self.problem = None
        else:
            self.problem = Beams2D()
        
        # Load default requirements if not provided in config
        if "default_requirements" not in self.config:
            from .catalogs import get_beams2d_catalog
            self.config["default_requirements"] = get_beams2d_catalog()["default_requirements"]
    
    def simulate(self, design: np.ndarray, requirements: Dict[str, Any]) -> Dict[str, float]:
        """
        Simulate Beams2D design.
        
        Args:
            design: 2D numpy array of 0s and 1s
            requirements: Problem requirements (volfrac, rmin, etc.)
            
        Returns:
            Dict with compliance and constraint violations
        """
        if not ENGI_BENCH_AVAILABLE:
            return {"compliance": 1e6, "constraint_violations": 1.0}
        
        try:
            # Configure problem
            self.problem.reset()
            self.problem.nelx = design.shape[1]
            self.problem.nely = design.shape[0]
            self.problem.conditions = requirements
            
            # Run simulation
            compliance = self.problem.simulate(design)
            compliance_value = compliance[0] if isinstance(compliance, np.ndarray) else compliance
            
            # Calculate constraint violations
            violations = self._calculate_constraint_violations(design, requirements)
            
            return {
                "compliance": float(compliance_value),
                "constraint_violations": violations,
                "design": design
            }
        except Exception as e:
            print(f"Beams2D Simulator error: {e}")
            return {"compliance": 1e6, "constraint_violations": 1.0}
    
    def validate_design(self, design: np.ndarray, requirements: Dict[str, Any]) -> bool:
        """Validate Beams2D design format."""
        if not isinstance(design, np.ndarray):
            return False
        if design.ndim != 2:
            return False
        if not np.all(np.isin(design, [0, 1])):
            return False
        return True
    
    def _calculate_constraint_violations(self, design: np.ndarray, requirements: Dict[str, Any]) -> float:
        """Calculate constraint violation penalty."""
        violations = 0.0
        
        # Volume fraction constraint
        target_volfrac = requirements.get("volfrac", 0.4)
        actual_volfrac = np.mean(design)
        volfrac_error = abs(actual_volfrac - target_volfrac)
        if volfrac_error > 0.05:  # 5% tolerance
            violations += volfrac_error * 2
        
        # Overhang constraint
        if requirements.get("overhang_constraint", False):
            overhang_violations = self._check_overhang_constraint(design)
            violations += overhang_violations * 0.1
        
        return min(violations, 1.0)
    
    def _check_overhang_constraint(self, design: np.ndarray) -> int:
        """Check overhang constraint violations."""
        height, width = design.shape
        violations = 0
        
        for i in range(1, height):
            for j in range(width):
                if design[i, j] == 1:  # Solid cell
                    # Check if supported from below (45-degree rule)
                    supported = False
                    for k in range(max(0, j-1), min(width, j+2)):
                        if design[i-1, k] == 1:
                            supported = True
                            break
                    if not supported:
                        violations += 1
        
        return violations
    
    def _calculate_constraint_penalties_dict(self, design: np.ndarray, requirements: Dict[str, Any]) -> Dict[str, float]:
        """Calculate constraint penalties in the same format as original function."""
        penalties = {}
        
        # Volume fraction penalty
        volfrac_respected, actual_volfrac, volfrac_error = self._check_volfrac_constraint(
            design, requirements.get("volfrac", 0.4)
        )
        penalties["volfrac"] = 0.0 if volfrac_respected else min(1.0, volfrac_error * 2)
        
        # Overhang penalty
        if requirements.get("overhang_constraint", False):
            overhang_violations = self._check_overhang_constraint(design)
            penalties["overhang"] = min(1.0, overhang_violations * 0.1)
        else:
            penalties["overhang"] = 0.0
        
        return penalties
    
    def _check_volfrac_constraint(self, design: np.ndarray, target_volfrac: float) -> tuple:
        """Check volume fraction constraint."""
        actual_volfrac = np.mean(design)
        volfrac_error = abs(actual_volfrac - target_volfrac)
        volfrac_respected = volfrac_error <= 0.05  # 5% tolerance
        return volfrac_respected, actual_volfrac, volfrac_error
    
    def _calculate_reward(self, results: Dict[str, float], requirements: Dict[str, Any]) -> float:
        """Calculate Beams2D reward: compliance minimization + constraint penalties."""
        compliance = results["compliance"]
        
        # Use default compliance range if not provided
        c_min = requirements.get("compliance_min", 8.97)
        c_max = requirements.get("compliance_max", 2129.30)
        
        # Normalize compliance (lower is better)
        if compliance <= 0:
            compliance_score = 0.0
        else:
            compliance_norm = 1.0 - (compliance - c_min) / (c_max - c_min)
            compliance_score = max(0.0, min(1.0, compliance_norm))
        
        # Calculate constraint penalties using the same method as original
        penalties = self._calculate_constraint_penalties_dict(results["design"], requirements)
        weights = {"volfrac": 0.5, "overhang": 0.5}
        total_penalty = sum(penalties[key] * weights[key] for key in penalties)
        
        # Apply constraint penalty using the same formula as original
        final_score = max(0.0, compliance_score - total_penalty)
        return final_score
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Return Beams2D simulator capabilities."""
        return {
            "input_format": "2D numpy array (0s and 1s)",
            "outputs": ["compliance", "constraint_violations"],
            "constraints": ["volume_fraction", "overhang"],
            "optimization": "compliance minimization"
        }
