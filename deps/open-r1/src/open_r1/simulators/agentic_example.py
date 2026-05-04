"""
Example of how to use the new simulator interface for agentic applications.
"""

from typing import Any, Dict, List, Optional
from .registry import registry


class SimulatorAgent:
    """
    Example agent that uses simulators for engineering design.
    """
    
    def __init__(self):
        self.registry = registry
    
    def evaluate_design(self, domain: str, design: Any, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a design using the appropriate simulator.
        
        Args:
            domain: Problem domain (eps, beams2d, knapsack)
            design: Design to evaluate
            requirements: Problem requirements
            
        Returns:
            Evaluation results
        """
        try:
            # Get simulator
            simulator = self.registry.get_simulator(domain)
            
            # Validate design
            if not simulator.validate_design(design, requirements):
                return {"valid": False, "error": "Invalid design format"}
            
            # Run simulation
            results = simulator.simulate(design, requirements)
            
            # Calculate reward
            reward = simulator.get_reward(design, requirements)
            
            return {
                "valid": True,
                "results": results,
                "reward": reward,
                "domain": domain
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def compare_designs(self, domain: str, designs: List[Any], requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Compare multiple designs.
        
        Args:
            domain: Problem domain
            designs: List of designs to compare
            requirements: Problem requirements
            
        Returns:
            List of evaluation results
        """
        results = []
        for i, design in enumerate(designs):
            result = self.evaluate_design(domain, design, requirements)
            result["design_id"] = i
            results.append(result)
        
        # Sort by reward (best first)
        results.sort(key=lambda x: x.get("reward", 0.0), reverse=True)
        return results
    
    def get_simulator_info(self, domain: str) -> Dict[str, Any]:
        """Get information about a simulator."""
        return self.registry.get_simulator_info(domain)
    
    def list_available_simulators(self) -> List[str]:
        """List all available simulators."""
        return self.registry.list_simulators()


# Example usage
def example_usage():
    """Example of how to use the simulator interface."""
    
    # Create agent
    agent = SimulatorAgent()
    
    # Example 1: EPS Design
    eps_design = "0123"  # 4-digit design code
    eps_requirements = {
        "lifetime_years": 5.0,
        "delta_v_ms": 120.0,
        "payload_power_avg_w": 150.0,
        # ... other mission parameters
    }
    
    eps_result = agent.evaluate_design("eps", eps_design, eps_requirements)
    print(f"EPS Design Result: {eps_result}")
    
    # Example 2: Beams2D Design
    import numpy as np
    beams2d_design = np.array([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ])
    beams2d_requirements = {
        "volfrac": 0.4,
        "rmin": 2.0,
        "forcedist": 0.5,
        "overhang_constraint": False
    }
    
    beams2d_result = agent.evaluate_design("beams2d", beams2d_design, beams2d_requirements)
    print(f"Beams2D Design Result: {beams2d_result}")
    
    # Example 3: Knapsack Selection
    knapsack_selection = ["it-000", "it-002", "it-005"]
    knapsack_requirements = {
        "weight_capacity": 100,
        "volume_capacity": 50,
        "items": [
            {"id": "it-000", "weight": 12, "volume": 8, "value": 25},
            {"id": "it-002", "weight": 8, "volume": 6, "value": 20},
            {"id": "it-005", "weight": 22, "volume": 14, "value": 40},
            # ... more items
        ]
    }
    
    knapsack_result = agent.evaluate_design("knapsack", knapsack_selection, knapsack_requirements)
    print(f"Knapsack Selection Result: {knapsack_result}")
    
    # Example 4: Compare multiple designs
    eps_designs = ["0123", "1111", "2222"]
    comparison_results = agent.compare_designs("eps", eps_designs, eps_requirements)
    print(f"EPS Design Comparison: {comparison_results}")


if __name__ == "__main__":
    example_usage()
