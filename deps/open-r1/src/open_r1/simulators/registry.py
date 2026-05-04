"""
Simulator registry for managing and accessing all simulators.
"""

from typing import Any, Dict, List, Optional, Type
from .base import BaseSimulator
from .eps import EPSSimulator
from .beams2d import Beams2DSimulator
from .knapsack import KnapsackSimulator
from .sds_simulator import SDSSimulator
from .decadal_simulator import DecadalSimulator


class SimulatorRegistry:
    """
    Registry for managing all available simulators.
    Provides clean access for agentic use and external applications.
    """
    
    def __init__(self):
        """Initialize registry with default simulators."""
        self._simulators = {}
        self._simulator_classes = {
            "eps": EPSSimulator,
            "beams2d": Beams2DSimulator,
            "knapsack": KnapsackSimulator,
            "sds": SDSSimulator,
            "decadal": DecadalSimulator,
        }
        self._instances = {}
    
    def register_simulator(self, name: str, simulator_class: Type[BaseSimulator]):
        """Register a new simulator class."""
        self._simulator_classes[name] = simulator_class
    
    def get_simulator(self, name: str, config: Optional[Dict[str, Any]] = None) -> BaseSimulator:
        """
        Get simulator instance.
        
        Args:
            name: Simulator name (eps, beams2d, knapsack)
            config: Optional configuration
            
        Returns:
            Simulator instance
        """
        if name not in self._simulator_classes:
            raise ValueError(f"Unknown simulator: {name}. Available: {list(self._simulator_classes.keys())}")
        
        # Create instance if not exists or config changed
        instance_key = f"{name}_{hash(str(config))}"
        if instance_key not in self._instances:
            simulator_class = self._simulator_classes[name]
            self._instances[instance_key] = simulator_class(config)
        
        return self._instances[instance_key]
    
    def list_simulators(self) -> List[str]:
        """List all available simulators."""
        return list(self._simulator_classes.keys())
    
    def get_simulator_info(self, name: str) -> Dict[str, Any]:
        """Get information about a simulator."""
        if name not in self._simulator_classes:
            raise ValueError(f"Unknown simulator: {name}")
        
        # Create temporary instance to get info
        simulator_class = self._simulator_classes[name]
        temp_instance = simulator_class()
        return temp_instance.get_info()
    
    def simulate(self, domain: str, design: Any, requirements: Dict[str, Any], 
                 config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Convenience method to run simulation.
        
        Args:
            domain: Simulator domain
            design: Design to simulate
            requirements: Problem requirements
            config: Optional simulator configuration
            
        Returns:
            Simulation results
        """
        simulator = self.get_simulator(domain, config)
        return simulator.simulate(design, requirements)
    
    def get_reward(self, domain: str, design: Any, requirements: Dict[str, Any],
                   config: Optional[Dict[str, Any]] = None) -> float:
        """
        Convenience method to get reward score.
        
        Args:
            domain: Simulator domain
            design: Design to evaluate
            requirements: Problem requirements
            config: Optional simulator configuration
            
        Returns:
            Reward score (0.0 to 1.0)
        """
        simulator = self.get_simulator(domain, config)
        return simulator.get_reward(design, requirements)
    
    def batch_simulate(self, domain: str, designs: List[Any], requirements: List[Dict[str, Any]],
                       config: Optional[Dict[str, Any]] = None) -> List[Dict[str, float]]:
        """
        Convenience method for batch simulation.
        
        Args:
            domain: Simulator domain
            designs: List of designs
            requirements: List of requirements
            config: Optional simulator configuration
            
        Returns:
            List of simulation results
        """
        simulator = self.get_simulator(domain, config)
        return simulator.batch_simulate(designs, requirements)


# Global registry instance
registry = SimulatorRegistry()
