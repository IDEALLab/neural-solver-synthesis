import ast
import json
import os
import platform
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# --- 1. PATH SETUP FOR SYNDEOPT ---
# Ensure we can find syndeopt in deps/
current_dir = Path(__file__).resolve().parent
# Assuming repo_root/evaluation/sds -> we go up two levels to repo_root
repo_root = (current_dir / "../../").resolve()
syndeopt_path = repo_root / "deps" / "syndeopt" / "src"

if str(syndeopt_path) not in sys.path:
    sys.path.insert(0, str(syndeopt_path))

from syndeopt.core.instance import CardBounds, SDSInstance  # noqa: E402

# --- 2. DATA CONVERTERS ---


def deserialize_mission(mission_data):
    """
    Handles the case where 'mission' is a JSON string (due to HF dataset format)
    or a dictionary. Returns a dictionary.
    """
    if isinstance(mission_data, str):
        try:
            return json.loads(mission_data)
        except json.JSONDecodeError:
            return {}
    return mission_data if isinstance(mission_data, dict) else {}


def mission_to_instance(m):
    """
    Reconstruct a Syndeopt SDSInstance object from the mission dictionary.
    Necessary for running the Baseline Solvers (Greedy/LocalSearch).
    """
    m = deserialize_mission(m)

    # Reconstruct W (interactions)
    W = {}  # noqa: N806

    # Handle list-of-dicts format (from your push script)
    if "interactions_list" in m:
        for item in m["interactions_list"]:
            try:
                # Keys might be "0,1" or strings in the list
                pair_str = item.get("pair")
                if pair_str:
                    u, v = map(int, pair_str.split(","))
                    W[(u, v)] = item["weight"]
            except Exception:
                pass
    # Handle standard dict format
    elif "interactions" in m:
        for k, weight_val in m["interactions"].items():
            try:
                u, v = map(int, k.split(","))
                W[(u, v)] = weight_val  # Use the weight value, not the variable index
            except Exception:
                pass

    # Reconstruct Constraints
    prec = [tuple(x) for x in m.get("precedence", [])]
    mutex = [tuple(x) for x in m.get("mutex", [])]

    # Reconstruct Groups
    groups = {}
    if "groups_list" in m:
        for idx, item in enumerate(m["groups_list"]):
            groups[idx] = item["members"]
    else:
        groups = m.get("groups", {})
        # Ensure keys are ints for syndeopt
        groups = {int(k): v for k, v in groups.items()}

    # Reconstruct Weights
    weights = m.get("weights", [])
    n_vars = m.get("n_variables", len(weights))
    if not weights:
        weights = [0.0] * n_vars

    return SDSInstance(
        n=n_vars,
        w=weights,
        W=W,
        precedence=prec,
        mutex=mutex,
        groups=groups,
        card=CardBounds(L=m["cardinality_bounds"][0], U=m["cardinality_bounds"][1]),
    )


def check_constraint_violations(inst: SDSInstance, selected_ids: list[int]) -> dict:
    """
    Check which specific constraints are violated by a solution.
    Returns dict with violation details.
    """
    violations = {
        "cardinality": False,
        "precedence": [],
        "mutex": [],
        "groups": [],
        "all_valid": True,
    }

    if not selected_ids:
        violations["cardinality"] = True
        violations["all_valid"] = False
        return violations

    # Convert to bitmask
    mask = 0
    for idx in selected_ids:
        mask |= 1 << idx

    k = mask.bit_count()
    L, U = inst.card.L, inst.card.U  # noqa: N806

    # Check cardinality
    if not (L <= k <= U):
        violations["cardinality"] = True
        violations["all_valid"] = False

    # Check precedence: j <= i means if j is selected, i must be selected
    sel_set = set(selected_ids)
    for i, j in inst.precedence:
        if j in sel_set and i not in sel_set:
            violations["precedence"].append((i, j))
            violations["all_valid"] = False

    # Check mutex: at most one of a or b can be selected
    for a, b in inst.mutex:
        if a in sel_set and b in sel_set:
            violations["mutex"].append((a, b))
            violations["all_valid"] = False

    # Check groups: at most 1 from each group
    for group_id, members in inst.groups.items():
        selected_in_group = [m for m in members if m in sel_set]
        if len(selected_in_group) > 1:
            violations["groups"].append(
                {
                    "group_id": group_id,
                    "members": members,
                    "selected": selected_in_group,
                }
            )
            violations["all_valid"] = False

    return violations


# --- 3. SECURITY SANDBOX ---


def validate_safety(code: str) -> tuple[bool, str]:  # noqa: PLR0911, PLR0912
    """Enhanced static analysis to block dangerous operations."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"

    # Allow 'sys' and 'json' as they are needed for I/O, but ban OS interaction
    BANNED_MODULES = {  # noqa: N806
        "os",
        "subprocess",
        "shutil",
        "pathlib",
        "pickle",
        "ctypes",
        "multiprocessing",
    }
    BANNED_FUNCTIONS = {  # noqa: N806
        "exec",
        "eval",
        "open",
        "input",
        "__import__",
        "compile",
        "reload",
        "__builtins__",
    }
    BANNED_ATTRIBUTES = {"__import__", "__getattribute__", "__setattr__", "__delattr__"}  # noqa: N806

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name in BANNED_MODULES:
                    return False, f"Security: Import of '{alias.name}' is forbidden."
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BANNED_MODULES:
                return False, f"Security: Import from '{node.module}' is forbidden."

        # Check function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BANNED_FUNCTIONS:
                    return False, f"Security: Function '{node.func.id}' is forbidden."
            elif isinstance(node.func, ast.Attribute):
                # Check for dangerous attribute access like __import__
                if node.func.attr in BANNED_ATTRIBUTES:
                    return (
                        False,
                        f"Security: Attribute access '{node.func.attr}' is forbidden.",
                    )
            elif (
                isinstance(node.func, ast.Call)
                and isinstance(node.func.func, ast.Name)
                and node.func.func.id == "getattr"
            ):
                # Check for getattr(__builtins__, '__import__') patterns
                return (
                    False,
                    "Security: Dynamic attribute access via getattr is forbidden.",
                )

        # Check for __import__ in expressions
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRIBUTES:
            return False, f"Security: Attribute '{node.attr}' access is forbidden."

    return True, ""


def run_candidate(code: str, stdin_obj: dict, timeout: float = 5.0) -> dict:  # noqa: PLR0911, PLR0912, PLR0915
    """
    Safe execution of generated code with detailed error reporting.
    Returns dict with 'error' key for failures, or execution result on success.
    Error types: 'syntax', 'security', 'runtime', 'timeout', 'json', 'unknown'

    Supports two code formats:
    - Direct execution: Code that reads stdin and prints stdout directly
    - Function-based (newer trained models): Code with `def solve_sds():` that needs to be called

    Auto-fixes 'missing execution block' errors to prevent false negatives.
    """
    start_time = time.time()

    is_safe, error_msg = validate_safety(code)
    if not is_safe:
        return {
            "error": error_msg,
            "error_type": "security" if "Security:" in error_msg else "syntax",
            "execution_time": 0.0,
        }

    # SAFETY NET: Auto-inject execution block if missing
    # This handles newer trained models that generate `def solve_sds():` but forget the main block
    code_lower = code.lower()
    if "def solve_sds" in code_lower:
        # Check if the function is actually called (not just mentioned in comments/strings)
        # Use AST to check for actual function calls, not just string matching
        has_execution_block = False
        if "if __name__" in code_lower:
            has_execution_block = True
        else:
            # Parse AST to check for actual solve_sds() calls (not in strings/comments)
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "solve_sds"
                    ):
                        has_execution_block = True
                        break
            except Exception:
                # If AST parsing fails, fall back to simple string check
                # But be more careful - check for solve_sds() followed by newline or end
                if re.search(r"solve_sds\s*\(\s*\)\s*$", code, re.MULTILINE):
                    has_execution_block = True

        if not has_execution_block:
            # Auto-inject execution block
            code += "\n\nif __name__ == '__main__':\n    solve_sds()\n"

    script_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            script_path = f.name

        stdin_data = json.dumps(stdin_obj)

        # Minimal environment
        safe_env = {
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "PATH": os.environ.get("PATH", ""),
        }

        def set_limits():
            try:
                resource.setrlimit(
                    resource.RLIMIT_CPU, (int(timeout), int(timeout) + 1)
                )
                mem_limit = 4 * 1024 * 1024 * 1024  # 4GB
                resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            except (ValueError, OSError, AttributeError):
                # On some systems (e.g., macOS), RLIMIT_AS might not be available
                # or resource limits might be restricted. Continue without limits.
                pass

        # On macOS, preexec_fn might not work, so make it optional
        run_kwargs = {
            "args": [sys.executable, script_path],
            "input": stdin_data,
            "text": True,
            "capture_output": True,
            "timeout": timeout + 0.5,
            "env": safe_env,
        }

        # Only use preexec_fn on Linux (not macOS/Windows)
        if platform.system() == "Linux":
            run_kwargs["preexec_fn"] = set_limits

        result = subprocess.run(**run_kwargs, check=False)

        execution_time = time.time() - start_time

        script_path_obj = Path(script_path) if script_path else None
        if script_path_obj and script_path_obj.exists():
            script_path_obj.unlink()

        if result.returncode == 0:
            try:
                output = json.loads(result.stdout.strip())
                output["execution_time"] = execution_time
            except json.JSONDecodeError as e:
                return {
                    "error": f"Invalid JSON output: {e!s}",
                    "error_type": "json",
                    "stdout": result.stdout[:500],
                    "execution_time": execution_time,
                }
            else:
                return output
        elif result.returncode == -signal.SIGXCPU:
            return {
                "error": "Time limit exceeded",
                "error_type": "timeout",
                "execution_time": execution_time,
            }
        else:
            # Extract error details from stderr
            stderr_preview = result.stderr[:500] if result.stderr else "No error output"
            error_type = "runtime"
            if "Traceback" in stderr_preview:
                if "SyntaxError" in stderr_preview:
                    error_type = "syntax"
                elif "NameError" in stderr_preview or "TypeError" in stderr_preview:
                    error_type = "runtime"

            return {
                "error": f"Runtime Error (Exit {result.returncode}): {stderr_preview}",
                "error_type": error_type,
                "stderr": stderr_preview,
                "execution_time": execution_time,
            }

    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        script_path_obj = Path(script_path) if script_path else None
        if script_path_obj and script_path_obj.exists():
            script_path_obj.unlink()
        return {
            "error": "Subprocess timeout exceeded",
            "error_type": "timeout",
            "execution_time": execution_time,
        }
    except Exception as e:
        execution_time = time.time() - start_time
        script_path_obj = Path(script_path) if script_path else None
        if script_path_obj and script_path_obj.exists():
            script_path_obj.unlink()
        return {
            "error": f"Execution exception: {e!s}",
            "error_type": "unknown",
            "execution_time": execution_time,
        }
