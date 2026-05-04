"""
Common utilities for simulators.
Contains code execution, safety checks, and other shared functionality.
"""

import subprocess
import tempfile
import signal
import sys
import os
import re
import json
import resource
import ast
from typing import Any, Dict, List, Optional, Union, Tuple


def normalize(s: str) -> str:
    """Normalize string for comparison."""
    return re.sub(r'[^a-zA-Z0-9]', '', s.lower())


def extract_block(text: str, tag: str) -> str:
    """Extract content from XML-like tags."""
    pattern = rf'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------
# 1. STATIC ANALYSIS (The Shield)
# ---------------------------------------------------------

def validate_safety(code: str) -> Tuple[bool, str]:
    """
    Parses code into an AST to detect dangerous imports or calls.
    Returns (is_safe, error_message).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    
    # Dangerous modules we explicitly ban
    BANNED_MODULES = {'os', 'subprocess', 'shutil', 'pathlib', 'pickle', 'socket', 
                      'urllib', 'requests', 'ftplib', 'smtplib', 'http'}
    
    # Dangerous built-in functions (execution)
    BANNED_FUNCTIONS = {'exec', 'eval', 'open', 'input', '__import__', 'compile'}
    
    for node in ast.walk(tree):
        # 1. Check 'import x' (e.g., import os)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in BANNED_MODULES:
                    return False, f"Security violation: Import of '{alias.name}' is forbidden."
        
        # 2. Check 'from x import y' (e.g., from os import system)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in BANNED_MODULES:
                return False, f"Security violation: Import from '{node.module}' is forbidden."
        
        # 3. Check for dangerous function calls (e.g., open(), exec())
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BANNED_FUNCTIONS:
                    return False, f"Security violation: Function '{node.func.id}' is forbidden."
            # Also check for attribute calls like os.system()
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in BANNED_MODULES:
                        return False, f"Security violation: Call to '{node.func.value.id}.{node.func.attr}' is forbidden."
    
    return True, ""


# ---------------------------------------------------------
# 2. EXECUTION ENGINE
# ---------------------------------------------------------

def run_candidate(code: str, stdin_obj: dict, timeout: float = 5.0) -> dict:
    """
    Safe execution pipeline:
    1. AST Scan (rejection)
    2. Subprocess execution with Resource Limits
    3. Environment stripping
    
    This is the main interface for code execution in simulators.
    
    Supports two code formats:
    - Direct execution: Code that reads stdin and prints stdout directly
    - Function-based (ShinkaEvolve): Code with `def solve_sds():` that needs to be called
    
    Auto-fixes 'missing execution block' errors to prevent false negatives.
    """
    
    # --- STEP 1: STATIC ANALYSIS ---
    is_safe, error_msg = validate_safety(code)
    if not is_safe:
        return {"error": error_msg}
    
    # --- STEP 1.5: SAFETY NET - Auto-inject execution block if missing ---
    # This prevents false negatives when model forgets the main block but has valid logic
    code_lower = code.lower()
    if "def solve_sds" in code_lower:
        # Check if the function is actually called
        if "if __name__" not in code_lower and "solve_sds()" not in code_lower:
            # Auto-inject execution block
            code += "\n\nif __name__ == '__main__':\n    solve_sds()\n"
    
    # --- STEP 2: PREPARE EXECUTION ---
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            script_path = f.name
        
        stdin_data = json.dumps(stdin_obj)
        
        # Strip environment to prevent token leakage
        safe_env = {
            "PYTHONPATH": "",
            "PATH": os.environ.get("PATH", ""),
            "PYTHONUNBUFFERED": "1"  # Ensure unbuffered output
        }
        
        def set_limits():
            """Set resource limits for the subprocess."""
            try:
                # CPU Time Limit (Soft, Hard) - Add small buffer
                resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 2))
                # Memory Limit (4GB)
                mem_limit = 4 * 1024 * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            except (ValueError, OSError):
                # Resource limits may not be available on all systems
                # Continue without them (subprocess timeout will still apply)
                pass
        
        # --- STEP 3: EXECUTE ---
        result = subprocess.run(
            [sys.executable, script_path],
            input=stdin_data,
            text=True,
            capture_output=True,
            timeout=timeout,  # Use exact timeout (no buffer needed with resource limits)
            cwd=os.path.dirname(script_path),  # Run in /tmp usually
            env=safe_env,
            preexec_fn=set_limits
        )
        
        # Cleanup
        if script_path and os.path.exists(script_path):
            os.unlink(script_path)
        
        if result.returncode == 0:
            try:
                return json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                return {"error": "Invalid JSON Output", "stdout": result.stdout[:500]}
        elif result.returncode == -signal.SIGXCPU:
            return {"error": "Time limit exceeded (SIGXCPU)"}
        else:
            return {
                "error": f"Runtime Error (Exit {result.returncode})", 
                "stderr": result.stderr[:500] if result.stderr else ""
            }
            
    except subprocess.TimeoutExpired:
        if script_path and os.path.exists(script_path):
            os.unlink(script_path)
        return {"error": "Timeout"}
    except Exception as e:
        if script_path and os.path.exists(script_path):
            os.unlink(script_path)
        return {"error": f"System Error: {str(e)}"}


def validate_code_structure(code: str) -> bool:
    """Validate that code has the required structure."""
    # Check for function definitions (Required)
    if not re.search(r'def\s+\w+.*\(', code, re.IGNORECASE):
        return False
    
    # Note: We don't require __main__ block here because run_candidate() will
    # auto-inject the main block if missing (handles ShinkaEvolve format where
    # only def solve_sds(): ... is provided without __main__ guard)
    
    return True


def extract_design_from_text(text: str, domain: str) -> Optional[str]:
    """Extract design from text based on domain."""
    if domain == "eps":
        # Look for answer block first
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
        if not answer_match:
            return None
        
        answer_text = answer_match.group(1)
        
        # Look for JSON selection format first
        json_match = re.search(r'\{.*"selection".*\}', answer_text, re.IGNORECASE | re.DOTALL)
        if json_match:
            try:
                import json
                json_str = json_match.group(0)
                json_str = re.sub(r'[^\w\s\-":,{}]', '', json_str)
                selection_data = json.loads(json_str)
                if "selection" in selection_data:
                    # Convert to design code format
                    orbit = selection_data["selection"].get("orbit", "")
                    solar_array = selection_data["selection"].get("solar_array", "")
                    battery = selection_data["selection"].get("battery", "")
                    array_dof = selection_data["selection"].get("array_dof", "")
                    
                    # Map to design code indices (simplified mapping)
                    orbit_map = {"LEO-400-DD": "0", "LEO-500-DD": "1", "MEO-1000-DD": "2"}
                    array_map = {"XTE-SF": "0", "XTE-LILT": "1", "XTE-HF": "2", "XTJ-CIC": "3", "UTJ-CIC": "4", "XTJ-Prime": "5", "Azur 3G30C": "6"}
                    battery_map = {"Saft 8s4p": "0", "Saft 11s16p": "1", "Saft 4s1p VES16": "2", "EP-SAR-10197": "3", "EP-SAR-10199": "4", "EP-SAR-10207": "5", "EP-SAR-10215": "6"}
                    dof_map = {"0": "0", "1": "1", "2": "2"}
                    
                    orbit_idx = orbit_map.get(orbit, "0")
                    array_idx = array_map.get(solar_array, "0")
                    battery_idx = battery_map.get(battery, "0")
                    dof_idx = dof_map.get(array_dof, "0")
                    
                    return orbit_idx + array_idx + battery_idx + dof_idx
            except:
                pass
        
        # Fallback: look for component names in answer text
        orbit_match = re.search(r'[Oo]rbit[:\s]*([A-Za-z0-9\-]+)', answer_text)
        array_match = re.search(r'[Ss]olar\s+[Aa]rray[:\s]*([A-Za-z0-9\-]+)', answer_text)
        battery_match = re.search(r'[Bb]attery[:\s]*([A-Za-z0-9\-]+)', answer_text)
        dof_match = re.search(r'[Dd]egrees\s+of\s+[Ff]reedom[:\s]*([0-9]+)', answer_text)
        
        if orbit_match and array_match and battery_match and dof_match:
            orbit = orbit_match.group(1)
            solar_array = array_match.group(1)
            battery = battery_match.group(1)
            dof = dof_match.group(1)
            
            # Map to design code indices (simplified mapping)
            orbit_map = {"LEO-400-DD": "0", "LEO-500-DD": "1", "MEO-1000-DD": "2"}
            array_map = {"XTE-SF": "0", "XTE-LILT": "1", "XTE-HF": "2", "XTJ-CIC": "3", "UTJ-CIC": "4", "XTJ-Prime": "5", "Azur 3G30C": "6"}
            battery_map = {"Saft 8s4p": "0", "Saft 11s16p": "1", "Saft 4s1p VES16": "2", "EP-SAR-10197": "3", "EP-SAR-10199": "4", "EP-SAR-10207": "5", "EP-SAR-10215": "6"}
            dof_map = {"0": "0", "1": "1", "2": "2"}
            
            orbit_idx = orbit_map.get(orbit, "0")
            array_idx = array_map.get(solar_array, "0")
            battery_idx = battery_map.get(battery, "0")
            dof_idx = dof_map.get(dof, "0")
            
            return orbit_idx + array_idx + battery_idx + dof_idx
        
        # Final fallback: look for design code pattern
        design_match = re.search(r'design[:\s]*([0-9]{4})', text, re.IGNORECASE)
        return design_match.group(1) if design_match else None
    elif domain == "beams2d":
        # Look for design matrix
        answer = extract_block(text, "answer")
        if not answer:
            return None
        
        # Parse design matrix from answer block
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
        
        if matrix_lines:
            # Convert list of strings to numpy array
            import numpy as np
            matrix_array = np.array([[int(c) for c in line] for line in matrix_lines])
            return matrix_array
        return None
    elif domain == "knapsack":
        # Look for selected items
        answer = extract_block(text, "answer")
        if not answer:
            return None
        
        for line in answer.split('\n'):
            if 'Selected:' in line:
                items_str = line.split('Selected:')[1].strip()
                return [item.strip() for item in items_str.split(',')]
        
        return None
    elif domain == "sds":
        # For SDS, design is extracted from code execution result, not from answer block
        # The code outputs JSON: {"selection": {"variables": [0, 2, 5, ...]}}
        # This function is called by simulator reward, but SDS doesn't use simulator reward
        # So we return None here (design extraction happens in code execution reward)
        return None
    elif domain == "decadal":
        # Look for design dictionary in answer block or JSON format
        answer = extract_block(text, "answer")
        if not answer:
            # Try to find JSON in code execution result
            json_match = re.search(r'\{.*"design".*\}', text, re.IGNORECASE | re.DOTALL)
            if json_match:
                try:
                    import json
                    json_str = json_match.group(0)
                    # Clean up JSON string
                    json_str = re.sub(r'[^\w\s\-":,{}[\]\']', '', json_str)
                    data = json.loads(json_str)
                    if "design" in data:
                        return data["design"]
                except:
                    pass
            return None
        
        # Try to parse JSON format first
        json_match = re.search(r'\{.*"design".*\}', answer, re.IGNORECASE | re.DOTALL)
        if json_match:
            try:
                import json
                json_str = json_match.group(0)
                json_str = re.sub(r'[^\w\s\-":,{}[\]\']', '', json_str)
                data = json.loads(json_str)
                if "design" in data:
                    return data["design"]
            except:
                pass
        
        # Try to parse dictionary format from text
        # Look for patterns like: "GEO-36000-equat-NA": ["ACE_CPR", "ACE_POL"]
        design = {}
        orbit_pattern = r'["\']?([A-Z0-9\-]+)["\']?\s*:\s*\[(.*?)\]'
        matches = re.finditer(orbit_pattern, answer, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            orbit = match.group(1)
            instruments_str = match.group(2)
            # Parse instrument list
            instruments = [inst.strip().strip('"\'') for inst in instruments_str.split(',') if inst.strip()]
            if instruments:
                design[orbit] = instruments
        
        if design:
            return design
        
        return None
    
    return None


def convert_mission_to_requirements(mission: dict) -> dict:
    """Convert mission parameters to simulator requirements format."""
    requirements = {}
    
    # Check if this is already in requirements format
    if "weight_capacity" in mission and "volume_capacity" in mission:
        # This is already Knapsack requirements format
        return mission
    elif "volfrac" in mission and "rmin" in mission:
        # This is already Beams2D requirements format
        return mission
    elif "n_variables" in mission and "cardinality_bounds" in mission:
        # This is already SDS requirements format
        # Handle interactions_list format from HuggingFace (convert to interactions dict)
        requirements = mission.copy()
        if "interactions_list" in requirements and "interactions" not in requirements:
            # Convert list format back to dict format
            requirements["interactions"] = {
                item["pair"]: item["weight"]
                for item in requirements["interactions_list"]
            }
            # Optionally remove interactions_list to avoid confusion
            # (keep it for backward compatibility if needed)
        return requirements
    elif "instruments" in mission or "orbits" in mission or "panelWeights" in mission:
        # This is already Decadal requirements format
        return mission.copy()
    elif "lifetime_years" in mission or "Lifetime" in mission:
        # This is EPS mission format - convert it
        if "Lifetime" in mission:
            requirements["lifetime_years"] = mission["Lifetime"]
        if "Delta-V" in mission:
            requirements["delta_v_ms"] = mission["Delta-V"]
        if "Payload Power" in mission:
            requirements["payload_power_avg_w"] = mission["Payload Power"]
        if "Payload Peak Power" in mission:
            requirements["payload_power_peak_w"] = mission["Payload Peak Power"]
        if "Bus Power" in mission:
            requirements["bus_power_w"] = mission["Bus Power"]
        if "Payload Mass" in mission:
            requirements["payload_mass_kg"] = mission["Payload Mass"]
        if "Payload Dimension" in mission:
            requirements["payload_dimension_m"] = mission["Payload Dimension"]
        if "Spacecraft Dimensions" in mission:
            requirements["spacecraft_dimensions_m"] = mission["Spacecraft Dimensions"]
        if "Inclination" in mission:
            requirements["inclination_deg"] = mission["Inclination"]
        if "RAAN" in mission:
            requirements["raan_deg"] = mission["RAAN"]
        if "Link Data Volume" in mission:
            requirements["link_data_gb_per_day"] = mission["Link Data Volume"]
        if "Link Datarate" in mission:
            requirements["link_datarate_mbps"] = mission["Link Datarate"]
        if "Pointing Requirement" in mission:
            requirements["pointing_requirement_deg"] = mission["Pointing Requirement"]
        if "Pointing Off Nadir" in mission:
            requirements["pointing_off_nadir_deg"] = mission["Pointing Off Nadir"]
        
        # Add default values for EPS
        requirements.setdefault("performance_threshold", 1.0)
        requirements.setdefault("max_cost", 1e8)
    else:
        # Default fallback - return as is
        requirements = mission.copy()
    
    return requirements
