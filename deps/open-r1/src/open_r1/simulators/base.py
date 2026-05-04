"""
Base simulator interface for all domain simulators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import subprocess
import tempfile
import signal
import sys
import os
from typing import Union
import numpy as np


class BaseSimulator(ABC):
    """
    Base class for all simulators.
    Provides a clean interface for agentic use and external access.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize simulator with optional configuration.
        
        Args:
            config: Simulator-specific configuration parameters
        """
        self.config = config or {}
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Setup simulator-specific resources."""
        pass
    
    @abstractmethod
    def simulate(self, design: Any, requirements: Dict[str, Any]) -> Dict[str, float]:
        """
        Run simulation and return results.
        
        Args:
            design: Design to simulate (domain-specific format)
            requirements: Problem requirements/constraints
            
        Returns:
            Dict with simulation results (performance, cost, etc.)
        """
        pass
    
    @abstractmethod
    def validate_design(self, design: Any, requirements: Dict[str, Any]) -> bool:
        """
        Validate if design meets basic requirements.
        
        Args:
            design: Design to validate
            requirements: Problem requirements
            
        Returns:
            True if design is valid, False otherwise
        """
        pass
    
    def get_reward(self, design: Any, requirements: Dict[str, Any]) -> float:
        """
        Calculate reward score for a design.
        Default implementation uses simulation results.
        
        Args:
            design: Design to evaluate
            requirements: Problem requirements
            
        Returns:
            Reward score (0.0 to 1.0)
        """
        try:
            if not self.validate_design(design, requirements):
                return 0.0
            
            results = self.simulate(design, requirements)
            reward = self._calculate_reward(results, requirements)
            
            # Validate reward is a valid float in [0, 1]
            if reward is None:
                import warnings
                warnings.warn(f"{self.__class__.__name__}.get_reward returned None, returning 0.0")
                return 0.0
            if not isinstance(reward, (int, float)):
                import warnings
                warnings.warn(f"{self.__class__.__name__}.get_reward returned {reward} (type: {type(reward)}), returning 0.0")
                return 0.0
            if not (0.0 <= reward <= 1.0):
                import warnings
                warnings.warn(f"{self.__class__.__name__}.get_reward returned {reward} outside [0, 1], clipping")
                return max(0.0, min(1.0, float(reward)))
            
            return float(reward)
        except Exception as e:
            import warnings
            warnings.warn(f"{self.__class__.__name__}.get_reward raised exception: {e}, returning 0.0")
            return 0.0
    
    @abstractmethod
    def _calculate_reward(self, results: Dict[str, float], requirements: Dict[str, Any]) -> float:
        """Calculate reward from simulation results."""
        pass
    
    def batch_simulate(self, designs: List[Any], requirements: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Run multiple simulations efficiently.
        
        Args:
            designs: List of designs to simulate
            requirements: List of requirements for each design
            
        Returns:
            List of simulation results
        """
        if len(designs) != len(requirements):
            raise ValueError("Number of designs must match number of requirements")
        
        return [self.simulate(design, req) for design, req in zip(designs, requirements)]
    
    def get_info(self) -> Dict[str, Any]:
        """Get simulator information and capabilities."""
        return {
            "name": self.__class__.__name__,
            "domain": getattr(self, 'domain', 'unknown'),
            "config": self.config,
            "capabilities": self._get_capabilities()
        }
    
    @abstractmethod
    def _get_capabilities(self) -> Dict[str, Any]:
        """Return simulator capabilities."""
        pass
    
    def execute_code(self, code: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute code and return results.
        This is a common interface for code execution across all simulators.
        """
        from .utils import run_candidate, validate_code_structure
        
        # Validate code structure
        if not validate_code_structure(code):
            return {"error": "Invalid code structure"}
        
        # Prepare input for code execution
        stdin_obj = {
            "requirements": requirements,
            "catalog": self._get_catalog_info()
        }
        
        # Execute code
        result = run_candidate(code, stdin_obj)
        return result
    
    def _get_catalog_info(self) -> Dict[str, Any]:
        """Get catalog information for code execution."""
        return {"domain": self.domain}
