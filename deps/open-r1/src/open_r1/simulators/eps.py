"""
EPS (Electrical Power System) Simulator for satellite design.
"""

import os
import requests
from typing import Any, Dict, List, Optional
from .base import BaseSimulator
from .utils import extract_design_from_text, convert_mission_to_requirements


class EPSSimulator(BaseSimulator):
    """
    EPS Simulator for satellite electrical power system design.
    """
    
    domain = "eps"
    
    def _setup(self):
        """Setup EPS simulator connection."""
        self.base_url = os.environ.get("SIMULATOR_URL", "http://localhost:8001")
        self.url = self.base_url.rstrip("/") + "/eval"
        self.timeout = self.config.get("timeout", 10.0)
    
    def simulate(self, design: str, requirements: Dict[str, Any]) -> Dict[str, float]:
        """
        Simulate EPS design.
        
        Args:
            design: 4-digit design code (e.g., "0123")
            requirements: Mission parameters
            
        Returns:
            Dict with performance and cost
        """
        # Convert requirements to mission_params format
        mission_params = []
        for name, value in requirements.items():
            if value is not None:
                mission_params.append({"name": name, "value": value})
        
        try:
            resp = requests.post(
                self.url,
                json={"design": design, "mission_params": mission_params},
                timeout=self.timeout
            )
            data = resp.json()
            return {
                "performance": float(data.get("performance", 0.0)),
                "cost": float(data.get("cost", 1e8))
            }
        except Exception as e:
            print(f"EPS Simulator error: {e}")
            return {"performance": 0.0, "cost": 1e8}
    
    def validate_design(self, design: str, requirements: Dict[str, Any]) -> bool:
        """Validate EPS design format."""
        if not isinstance(design, str) or len(design) != 4:
            return False
        if not all(c.isdigit() for c in design):
            return False
        return True
    
    def _calculate_reward(self, results: Dict[str, float], requirements: Dict[str, Any]) -> float:
        """Calculate EPS reward: performance constraint + cost minimization."""
        perf = results["performance"]
        cost = results["cost"]
        
        # Performance constraint: must exceed threshold
        if perf <= 1.0:
            return 0.0  # Failed performance constraint
        
        # Cost minimization: normalize cost and invert
        # Use reasonable cost normalization
        cost_norm = min(cost / 1e8, 1.0)
        score = 1.0 - cost_norm  # Lower cost = higher reward
        
        return max(0.0, min(1.0, score))  # Ensure score is in [0, 1]
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Return EPS simulator capabilities."""
        return {
            "input_format": "4-digit design code",
            "outputs": ["performance", "cost"],
            "constraints": ["performance > 1.0"],
            "optimization": "cost minimization"
        }
