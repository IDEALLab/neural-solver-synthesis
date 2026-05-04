"""
Knapsack Simulator for combinatorial optimization.
"""

import random
from typing import Any, Dict, List, Optional
from .base import BaseSimulator


class KnapsackSimulator(BaseSimulator):
    """
    Knapsack Simulator for multi-constraint knapsack problems.
    """
    
    domain = "knapsack"
    
    def _setup(self):
        """Setup Knapsack simulator."""
        # Load default catalog if not provided in config
        if "catalog" not in self.config:
            from .catalogs import get_knapsack_catalog
            self.config["catalog"] = get_knapsack_catalog()
    
    def simulate(self, selection: List[str], requirements: Dict[str, Any]) -> Dict[str, float]:
        """
        Simulate knapsack selection.
        
        Args:
            selection: List of selected item IDs
            requirements: Problem requirements (capacities, items)
            
        Returns:
            Dict with value, weight, volume, and feasibility
        """
        # Get items from requirements, with fallback to config catalog
        items = requirements.get("items")
        if not items:
            # Fallback to config catalog if no items in requirements
            items = self.config.get("catalog")
            if not items:
                raise ValueError("Knapsack requirements must include 'items' list or configure default catalog")
        
        weight_capacity = requirements.get("weight_capacity")
        volume_capacity = requirements.get("volume_capacity")
        
        if weight_capacity is None or volume_capacity is None:
            raise ValueError("Knapsack requirements must include 'weight_capacity' and 'volume_capacity'")
        
        # Create item lookup
        item_lookup = {item["id"]: item for item in items}
        
        # Calculate totals
        total_weight = 0
        total_volume = 0
        total_value = 0
        
        for item_id in selection:
            if item_id in item_lookup:
                item = item_lookup[item_id]
                total_weight += item["weight"]
                total_volume += item["volume"]
                total_value += item["value"]
        
        # Check feasibility
        feasible = (total_weight <= weight_capacity and total_volume <= volume_capacity)
        
        return {
            "value": total_value,
            "weight": total_weight,
            "volume": total_volume,
            "feasible": feasible,
            "weight_utilization": total_weight / weight_capacity if weight_capacity > 0 else 0,
            "volume_utilization": total_volume / volume_capacity if volume_capacity > 0 else 0
        }
    
    def validate_design(self, selection: List[str], requirements: Dict[str, Any]) -> bool:
        """Validate knapsack selection format."""
        if not isinstance(selection, list):
            return False
        if not all(isinstance(item_id, str) for item_id in selection):
            return False
        return True
    
    def _calculate_reward(self, results: Dict[str, float], requirements: Dict[str, Any]) -> float:
        """Calculate knapsack reward: value maximization + capacity utilization."""
        if not results["feasible"]:
            return 0.0
        
        value = results["value"]
        weight_util = results["weight_utilization"]
        volume_util = results["volume_utilization"]
        
        # Calculate maximum possible value (greedy approximation)
        items = requirements.get("items")
        if not items:
            items = self.config.get("catalog")
        
        weight_capacity = requirements.get("weight_capacity")
        volume_capacity = requirements.get("volume_capacity")
        
        # Sort by value density and take until capacity
        sorted_items = sorted(
            items, 
            key=lambda x: x["value"] / (x["weight"] + x["volume"]), 
            reverse=True
        )
        
        max_value = 0
        current_weight = 0
        current_volume = 0
        
        for item in sorted_items:
            if (current_weight + item["weight"] <= weight_capacity and 
                current_volume + item["volume"] <= volume_capacity):
                max_value += item["value"]
                current_weight += item["weight"]
                current_volume += item["volume"]
        
        # Normalize by maximum possible value
        if max_value > 0:
            value_score = value / max_value
        else:
            value_score = 0.0
        
        # Apply utilization penalty (encourage good capacity usage)
        avg_utilization = (weight_util + volume_util) / 2
        utilization_penalty = min(1.0, avg_utilization / 0.5)  # Penalty if < 50%
        
        return min(1.0, value_score * utilization_penalty)
    
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Return knapsack simulator capabilities."""
        return {
            "input_format": "List of item IDs",
            "outputs": ["value", "weight", "volume", "feasible", "utilization"],
            "constraints": ["weight_capacity", "volume_capacity"],
            "optimization": "value maximization"
        }
