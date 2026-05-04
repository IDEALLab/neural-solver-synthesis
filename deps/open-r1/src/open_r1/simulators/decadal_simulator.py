"""
Decadal Survey Simulator for satellite constellation design.
"""

import os
import requests
from typing import Any, Dict, List, Optional
from .base import BaseSimulator


class DecadalSimulator(BaseSimulator):
    """
    Decadal Survey Simulator for Earth observing satellite constellation design.
    
    Evaluates designs by assigning instruments to orbits and computing:
    - Science: Weighted stakeholder satisfaction across 6 panels
    - Cost: Lifecycle cost via EOSS spacecraft sizing
    """
    
    domain = "decadal"
    
    def _setup(self):
        """Setup Decadal simulator connection to VASSAR server."""
        # DECADAL_SIMULATOR_URL should be set to the actual server IP (e.g., http://172.28.31.228:8080)
        # The server binds to 0.0.0.0 to accept connections from any interface
        self.base_url = os.environ.get("DECADAL_SIMULATOR_URL", "http://0.0.0.0:8080")
        self.evaluate_url = self.base_url.rstrip("/") + "/evaluate"
        self.initialize_url = self.base_url.rstrip("/") + "/initialize"
        self.timeout = self.config.get("timeout", 120.0)  # Longer timeout for complex evaluations (2 minutes)
    
    def simulate(self, design: Dict[str, List[str]], requirements: Dict[str, Any]) -> Dict[str, float]:
        """
        Simulate decadal constellation design.
        
        Args:
            design: Dictionary mapping orbits to instrument lists
                   e.g., {"GEO-36000-equat-NA": ["ACE_CPR", "ACE_POL"], ...}
            requirements: Problem requirements including:
                - panelWeights: Optional dict of panel weights (WEA, CLI, ECO, WAT, HEA, SOL)
                - instruments: Optional list of valid instruments (for validation)
                - orbits: Optional list of valid orbits (for validation)
            
        Returns:
            Dict with science and cost scores
        """
        # Prepare evaluation request
        eval_payload = {"design": design}
        
        # Add panel weights if provided in requirements
        if "panelWeights" in requirements and requirements["panelWeights"]:
            eval_payload["panelWeights"] = requirements["panelWeights"]
        
        # Retry logic for connection errors (server might still be starting)
        max_retries = 3
        retry_delay = 2.0  # seconds
        
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    self.evaluate_url,
                    json=eval_payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )
                
                if resp.status_code != 200:
                    print(f"Decadal Simulator error: HTTP {resp.status_code} - {resp.text}")
                    return {"science": 0.0, "cost": 1e10}
                
                data = resp.json()
                
                # Extract science and cost from response
                science = float(data.get("science", 0.0))
                cost = float(data.get("cost", 1e10))
                
                return {
                    "science": science,
                    "cost": cost
                }
            except requests.exceptions.Timeout:
                print(f"Decadal Simulator error: Request timeout after {self.timeout}s")
                return {"science": 0.0, "cost": 1e10}
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    # Retry with exponential backoff
                    import time
                    delay = retry_delay * (2 ** attempt)
                    print(f"Decadal Simulator: Cannot connect to {self.evaluate_url}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Decadal Simulator error: Cannot connect to {self.evaluate_url} after {max_retries} attempts")
                    return {"science": 0.0, "cost": 1e10}
            except Exception as e:
                print(f"Decadal Simulator error: {e}")
                return {"science": 0.0, "cost": 1e10}
        
        # Should never reach here, but just in case
        return {"science": 0.0, "cost": 1e10}
    
    def validate_design(self, design: Dict[str, List[str]], requirements: Dict[str, Any]) -> bool:
        """
        Validate decadal design format.
        
        Args:
            design: Design dictionary {orbit: [instruments]}
            requirements: Requirements dict (may contain valid instruments/orbits lists)
        
        Returns:
            True if design is valid, False otherwise
        """
        if not isinstance(design, dict):
            return False
        
        # Check that all values are lists
        if not all(isinstance(v, list) for v in design.values()):
            return False
        
        # Check that all list items are strings (instrument names)
        for orbit, instruments in design.items():
            if not isinstance(orbit, str):
                return False
            if not all(isinstance(inst, str) for inst in instruments):
                return False
        
        # Optional: validate against provided instrument/orbit lists
        if "instruments" in requirements:
            valid_instruments = set(requirements["instruments"])
            for instruments in design.values():
                if not all(inst in valid_instruments for inst in instruments):
                    return False
        
        if "orbits" in requirements:
            valid_orbits = set(requirements["orbits"])
            if not all(orbit in valid_orbits for orbit in design.keys()):
                return False
        
        return True
    
    def _calculate_reward(self, results: Dict[str, float], requirements: Dict[str, Any]) -> float:
        """
        Calculate decadal reward: science maximization - normalized cost.
        
        Args:
            results: Dict with "science" and "cost" keys
            requirements: Problem requirements
        
        Returns:
            Reward score in [0, 1]
        """
        science = results["science"]
        cost = results["cost"]
        
        # Science is already normalized (0-1 range typically)
        # But ensure it's non-negative
        if science < 0:
            science = 0.0
        
        # Normalize cost (typical range: 0 to ~10B, normalize to [0, 1])
        # Use reasonable cost normalization based on typical decadal costs
        max_cost = requirements.get("max_cost", 1e10)  # Default 10B
        cost_norm = min(cost / max_cost, 1.0)
        
        # Reward = science - normalized_cost (weighted)
        # Balance: 80% science, 20% cost minimization
        cost_weight = requirements.get("cost_weight", 0.2)
        science_weight = 1.0 - cost_weight
        
        score = science_weight * science - cost_weight * cost_norm
        
        # Ensure score is in [0, 1]
        return max(0.0, min(1.0, score))
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Return Decadal simulator capabilities."""
        return {
            "input_format": "Dictionary mapping orbits to instrument lists",
            "outputs": ["science", "cost"],
            "stakeholder_panels": ["WEA", "CLI", "ECO", "WAT", "HEA", "SOL"],
            "optimization": "Science maximization with cost minimization",
            "server_url": self.base_url
        }

