#!/usr/bin/env python3
"""
Clean unified reward functions for multi-domain engineering design problems.
This is the clean version that replaces the polluted rewards_eps.py approach.
"""

import ast
import hashlib
import json
import os
import random
import re
import statistics
import sys
import traceback
from pathlib import Path

# Import clean simulators
from .simulators.registry import registry
from .simulators.sds_simulator import normalize_sds_score
from .simulators.utils import extract_block, run_candidate, validate_code_structure

# ---------------------------------------------------------
# 1. GLOBAL SETUP - Run once when module loads, not per loop
# ---------------------------------------------------------

# Attempt to locate and setup Syndeopt once
_SYNDEOPT_AVAILABLE = False

# 1. Try standard import first
try:
    from syndeopt.core.instance import CardBounds, SDSInstance
    from syndeopt.gen import (
        make_decomposable_instance,
        make_dense_deceptive_instance,
        make_greedy_easy_instance,
        make_local_optima_instance,
        make_maxcut_qubo_instance,
        make_planted_qubo_instance,
        make_random_qubo_instance,
        make_structural_trap_instance,
        make_tree_showcase_instance,
    )
    from syndeopt.solvers.greedy import GreedyMarginal
    _SYNDEOPT_AVAILABLE = True
except ImportError:
    pass

# 2. If failed, try injecting paths based on container structure
if not _SYNDEOPT_AVAILABLE:
    _current_file = Path(__file__).resolve()
    
    # Define potential roots for syndeopt
    _possible_paths = [
        # 1. Container Mount (Priority matches your .toml)
        # Check both src layout and flat layout
        Path("/workspace/syndeopt/src"),
        Path("/workspace/syndeopt"),
        
        # 2. Relative from this file (Fixed logic)
        # File: /workspace/open-r1/src/open_r1/rewards_unified_v2.py
        # Target: /workspace/syndeopt
        (_current_file.parent / "../../../syndeopt/src").resolve(),
        (_current_file.parent / "../../../syndeopt").resolve(),
    ]
    
    for path in _possible_paths:
        if path.exists() and (path / "syndeopt" / "__init__.py").exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            try:
                from syndeopt.core.instance import CardBounds, SDSInstance
                from syndeopt.gen import (
                    make_decomposable_instance,
                    make_dense_deceptive_instance,
                    make_greedy_easy_instance,
                    make_local_optima_instance,
                    make_maxcut_qubo_instance,
                    make_planted_qubo_instance,
                    make_random_qubo_instance,
                    make_structural_trap_instance,
                    make_tree_showcase_instance,
                )
                from syndeopt.solvers.greedy import GreedyMarginal
                _SYNDEOPT_AVAILABLE = True
                break
            except ImportError:
                continue
    
    if not _SYNDEOPT_AVAILABLE:
        print("WARNING: Syndeopt not found. SDS generalization tests will fail.")
        GreedyMarginal = None


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS - Defined once
# ---------------------------------------------------------

# Constants for random SDS instance generation
_PRECEDENCE_PROB = 0.7
_MUTEX_PROB = 0.6
_MIN_N_FOR_GROUPS = 2
_GROUPS_PROB = 0.5
_MIN_CARDINALITY_BOUNDS_LEN = 2
_SDS_SOFT_GATE_LAMBDA = 0.15
_CURRICULUM_THRESHOLD = 0.4
_EPSILON_TINY = 1e-9
_ORACLE_MATCH_THRESHOLD = 0.001
_MIN_TEST_INSTANCES = 3
_TOPK_POSITIVE_INTERACTION_CONFIG = {
    "normalization_variant": "topk_positive_interactions"
}
_FEASIBILITY_LOG_CALL_COUNTER = 0


class SyndeoptNotAvailableError(RuntimeError):
    """Raised when syndeopt is required but not available."""

    def __init__(self, context: str):
        super().__init__(
            f"Syndeopt is not available. {context} "
            "Please check that syndeopt is properly mounted/installed and that the path resolution is working."
        )


class GreedyMarginalNotAvailableError(RuntimeError):
    """Raised when GreedyMarginal solver is required but not available."""

    def __init__(self, context: str):
        super().__init__(
            f"GreedyMarginal solver is not available. {context} "
            "Please check that syndeopt.solvers.greedy is properly imported."
        )


def _make_random_sds_instance_local(n: int, seed: int | None = None):
    """Helper defined globally to avoid re-definition overhead."""
    rng = random.Random(seed) if seed is not None else random
    
    w = [rng.uniform(-5.0, 10.0) for _ in range(n)]
    interaction_density = rng.uniform(0.2, 0.8)
    W = {}  # noqa: N806
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < interaction_density:
                W[(i, j)] = rng.uniform(-8.0, 8.0)
    
    precedence = []
    if n > 1 and rng.random() < _PRECEDENCE_PROB:
        ordering = list(range(n))
        rng.shuffle(ordering)
        num_precedence = rng.randint(0, min(n * (n - 1) // 4, 10))
        for _ in range(num_precedence):
            i_idx = rng.randint(0, n - 2)
            j_idx = rng.randint(i_idx + 1, n - 1)
            i, j = ordering[i_idx], ordering[j_idx]
            if (i, j) not in precedence:
                precedence.append((i, j))
    
    mutex = []
    if n > 1 and rng.random() < _MUTEX_PROB:
        num_mutex = rng.randint(0, min(n // 2, 8))
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_pairs)
        mutex = all_pairs[:num_mutex]
    
    groups = {}
    if n > _MIN_N_FOR_GROUPS and rng.random() < _GROUPS_PROB:
        num_groups = rng.randint(1, min(n // 3, 5))
        remaining_vars = list(range(n))
        rng.shuffle(remaining_vars)
        group_size = len(remaining_vars) // num_groups
        for gid in range(num_groups):
            start_idx = gid * group_size
            end_idx = start_idx + group_size if gid < num_groups - 1 else len(remaining_vars)
            if start_idx < len(remaining_vars):
                groups[gid] = remaining_vars[start_idx:end_idx]
    
    min_card = max(0, rng.randint(0, n // 3))
    max_card = rng.randint(min_card + 1, n)
    card = CardBounds(L=min_card, U=max_card)
    
    return SDSInstance(n=n, w=w, W=W, precedence=precedence, mutex=mutex, groups=groups, card=card)


# ---------------------------------------------------------
# 3. SEEDING HELPERS - Deterministic seed generation
# ---------------------------------------------------------

def get_deterministic_seed(*args):
    """
    Creates a unique 32-bit integer seed from any number of arguments.
    Ensures no overlap between runs (e.g., Seed 101 vs 202).
    
    Args:
        *args: Any number of arguments to combine into a seed
        
    Returns:
        int: 32-bit integer seed
    """
    # Combine all args into a single unique string
    seed_str = "_".join(str(arg) for arg in args)
    # Hash it (SHA256 is robust enough to prevent collision)
    hash_bytes = hashlib.sha256(seed_str.encode('utf-8')).digest()
    # Convert to integer and clip to 32-bit (standard for numpy/random)
    return int.from_bytes(hash_bytes[:4], byteorder='big')


def _deserialize_mission(mission):
    """
    Deserialize mission if it's a JSON string.
    
    HuggingFace datasets may store complex mission objects as JSON strings
    to avoid Arrow schema issues with dynamic keys. This function handles
    both string and dict formats.
    
    Args:
        mission: Mission object (dict or JSON string)
        
    Returns:
        dict: Deserialized mission dictionary
    """
    if isinstance(mission, str):
        try:
            return json.loads(mission)
        except json.JSONDecodeError:
            # If deserialization fails, return empty dict
            return {}
    return mission


def _extract_user_prompt_text(prompt_obj):
    """
    Extract the user-visible problem text from the prompt object.
    """
    if isinstance(prompt_obj, list):
        for msg in prompt_obj:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return str(prompt_obj)

    if isinstance(prompt_obj, dict):
        if "content" in prompt_obj:
            return prompt_obj.get("content", "")
        if "prompt" in prompt_obj:
            return _extract_user_prompt_text(prompt_obj["prompt"])
        return str(prompt_obj)

    if prompt_obj is None:
        return ""

    return str(prompt_obj)


def _extract_completion_text(completion_obj):
    """
    Extract the assistant completion text from TRL/GRPO completion objects.
    """
    if isinstance(completion_obj, list) and completion_obj:
        head = completion_obj[0]
        if isinstance(head, dict):
            return head.get("content", "")
        return str(head)

    if isinstance(completion_obj, dict):
        return completion_obj.get("content", "")

    if completion_obj is None:
        return ""

    return str(completion_obj)


def _get_metadata_value(values, idx, default=""):
    """
    Retrieve the idx-th element from metadata passed via kwargs.
    """
    if values is None:
        return default
    if isinstance(values, list):
        if idx < len(values):
            return values[idx]
        return default
    return values


def _next_feasibility_log_call_index():
    """
    Return a per-process monotonic counter for reward-hook invocations.

    We use this as a stable fallback training-order identifier because
    `trainer_state.global_step` is not always available in this reward path.
    """
    global _FEASIBILITY_LOG_CALL_COUNTER
    call_index = _FEASIBILITY_LOG_CALL_COUNTER
    _FEASIBILITY_LOG_CALL_COUNTER += 1
    return call_index


def _extract_distributed_context():
    """
    Best-effort distributed context for per-rank logging.
    """
    rank = os.environ.get("RANK")
    if rank is None:
        rank = os.environ.get("SLURM_PROCID", "0")

    world_size = os.environ.get("WORLD_SIZE")
    if world_size is None:
        world_size = os.environ.get("SLURM_NTASKS", "1")

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        local_rank = os.environ.get("SLURM_LOCALID", "0")

    try:
        rank = int(rank)
    except (TypeError, ValueError):
        rank = 0

    try:
        world_size = int(world_size)
    except (TypeError, ValueError):
        world_size = 1

    try:
        local_rank = int(local_rank)
    except (TypeError, ValueError):
        local_rank = 0

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
    }


def _extract_training_position(trainer_state):
    """
    Best-effort training position for instrumentation records.

    Returns a tuple of:
    - trainer_global_step: optimizer step if available, else None
    - reward_call_index: per-process monotonic reward-hook invocation index
    """
    trainer_global_step = getattr(trainer_state, "global_step", None)
    if trainer_global_step is not None:
        try:
            trainer_global_step = int(trainer_global_step)
        except (TypeError, ValueError):
            trainer_global_step = None

    return trainer_global_step, _next_feasibility_log_call_index()


def _build_logging_context(trainer_state):
    """
    Build a single shared logging context for one reward-hook invocation.
    """
    trainer_global_step, reward_call_index = _extract_training_position(trainer_state)
    return {
        "dist": _extract_distributed_context(),
        "trainer_global_step": trainer_global_step,
        "reward_call_index": reward_call_index,
    }


def _log_instrumentation_error(label, exc):
    """
    Best-effort error logging for instrumentation failures.

    This prevents silent failures in cluster runs where stdout/stderr may not
    preserve the specific exception context we need for debugging.
    """
    try:
        run_id = os.environ.get("WANDB_RUN_ID") or os.environ.get("SLURM_JOB_ID") or "unknown"
        dist = _extract_distributed_context()
        error_dir = Path(
            os.environ.get("FEASIBILITY_INSTRUMENTATION_ERROR_DIR", "/workspace/logs/feasibility_instrumentation_errors")
        )
        error_dir.mkdir(parents=True, exist_ok=True)
        error_path = error_dir / f"{run_id}_rank{dist['rank']:05d}.log"
        with error_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{label}] {type(exc).__name__}: {exc}\n")
            handle.write(traceback.format_exc())
            handle.write("\n")
    except Exception:
        pass


def _iter_local_prompt_groups(prompts, missions, feasible_flags):
    """
    Yield contiguous local prompt groups from a reward invocation.

    The reward hook can contain multiple prompts worth of generations. Grouping
    by contiguous prompt/mission runs avoids collapsing duplicate prompts that
    might legitimately appear more than once in the same batch.
    """
    current = None

    for idx, is_feasible in enumerate(feasible_flags):
        mission = _deserialize_mission(missions[idx])
        mission_blob = json.dumps(mission, sort_keys=True, separators=(",", ":"))
        mission_hash = hashlib.sha256(mission_blob.encode("utf-8")).hexdigest()[:16]
        prompt_text = _extract_user_prompt_text(prompts[idx])
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16] if prompt_text else ""
        group_key = (mission_hash, prompt_hash)

        if current is None or current["group_key"] != group_key:
            if current is not None:
                yield current
            current = {
                "group_key": group_key,
                "mission_hash": mission_hash,
                "prompt_hash": prompt_hash,
                "group_size": 0,
                "feasible_count_in_group": 0,
            }

        current["group_size"] += 1
        current["feasible_count_in_group"] += int(bool(is_feasible))

    if current is not None:
        yield current


def _log_generation_traces(
    *,
    completions,
    prompts,
    missions,
    rewards,
    feasible_flags,
    trainer_state,
    problem_uuids=None,
    problem_prompt_hashes=None,
    problem_mission_hashes=None,
    logging_context=None,
):
    """
    Persist raw generation traces keyed by stable dataset identity.

    This is the lossless artifact for offline analysis: each completion is
    written as its own record with the raw text plus SDS instance identity.
    """
    if not completions or not prompts or not missions:
        return

    if len(completions) != len(prompts) or len(completions) != len(missions):
        return

    if logging_context is None:
        logging_context = _build_logging_context(trainer_state)

    dist = logging_context["dist"]
    trainer_global_step = logging_context["trainer_global_step"]
    reward_call_index = logging_context["reward_call_index"]

    records = []
    current_group_key = None
    local_group_ordinal = -1
    sample_ordinal_in_group = 0

    for idx, completion_obj in enumerate(completions):
        mission = _deserialize_mission(missions[idx])
        mission_blob = json.dumps(mission, sort_keys=True, separators=(",", ":"))
        computed_mission_hash = hashlib.sha256(mission_blob.encode("utf-8")).hexdigest()[:16]

        prompt_text = _extract_user_prompt_text(prompts[idx])
        computed_prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16] if prompt_text else ""

        problem_uuid = _get_metadata_value(problem_uuids, idx, "")
        mission_hash = _get_metadata_value(problem_mission_hashes, idx, computed_mission_hash) or computed_mission_hash
        prompt_hash = _get_metadata_value(problem_prompt_hashes, idx, computed_prompt_hash) or computed_prompt_hash
        completion_text = _extract_completion_text(completion_obj)

        group_key = (problem_uuid, mission_hash, prompt_hash)
        if group_key != current_group_key:
            current_group_key = group_key
            local_group_ordinal += 1
            sample_ordinal_in_group = 0

        records.append(
            {
                "trainer_global_step": trainer_global_step,
                "reward_call_index": reward_call_index,
                "rank": dist["rank"],
                "local_rank": dist["local_rank"],
                "world_size": dist["world_size"],
                "local_group_ordinal": local_group_ordinal,
                "sample_ordinal_in_group": sample_ordinal_in_group,
                "problem_uuid": problem_uuid,
                "mission_hash": mission_hash,
                "prompt_hash": prompt_hash,
                "completion_sha256": hashlib.sha256(completion_text.encode("utf-8")).hexdigest()[:16]
                if completion_text
                else "",
                "completion_text": completion_text,
                "reward": rewards[idx] if idx < len(rewards) else None,
                "exact_feasible": int(bool(feasible_flags[idx])) if idx < len(feasible_flags) else None,
            }
        )
        sample_ordinal_in_group += 1

    if not records:
        return

    try:
        run_id = os.environ.get("WANDB_RUN_ID") or os.environ.get("SLURM_JOB_ID") or "unknown"
        log_dir = Path(
            os.environ.get("FEASIBILITY_GENERATION_TRACE_DIR", "/workspace/logs/feasibility_generation_traces")
        )
        run_dir = log_dir / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / f"rank{dist['rank']:05d}.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception as exc:
        _log_instrumentation_error("generation_traces", exc)


def _log_group_feasibility_stats(*, prompts, missions, feasible_flags, trainer_state, logging_context=None):
    """
    Log exact feasibility sparsity statistics grouped by mission identity.

    This is intentionally side-effect-only instrumentation: it does not affect
    training rewards and is safe to skip if any logging path fails.
    """
    if not feasible_flags or not prompts or not missions:
        return

    if len(feasible_flags) != len(prompts) or len(feasible_flags) != len(missions):
        return

    if logging_context is None:
        logging_context = _build_logging_context(trainer_state)

    dist = logging_context["dist"]
    trainer_global_step = logging_context["trainer_global_step"]
    reward_call_index = logging_context["reward_call_index"]

    records = []
    for local_group_ordinal, record in enumerate(_iter_local_prompt_groups(prompts, missions, feasible_flags)):
        feasible_count = record["feasible_count_in_group"]
        group_size = record["group_size"]
        records.append(
            {
                "trainer_global_step": trainer_global_step,
                "reward_call_index": reward_call_index,
                "rank": dist["rank"],
                "local_rank": dist["local_rank"],
                "world_size": dist["world_size"],
                "local_group_ordinal": local_group_ordinal,
                "mission_hash": record["mission_hash"],
                "prompt_hash": record["prompt_hash"],
                "local_group_size": group_size,
                "feasible_count_in_group": feasible_count,
                "has_any_feasible_in_group": int(feasible_count > 0),
                "feasible_fraction_in_group": feasible_count / group_size if group_size else 0.0,
            }
        )

    if not records:
        return

    feasible_counts = [record["feasible_count_in_group"] for record in records]
    any_feasible = [record["has_any_feasible_in_group"] for record in records]
    total_group_size = sum(record["local_group_size"] for record in records)
    total_feasible = sum(feasible_counts)
    current_step = trainer_global_step

    try:
        import wandb

        if wandb.run is not None and dist["rank"] == 0:
            metrics = {
                # These are rank-local shard metrics, not full 64-generation group metrics.
                "analysis/feasibility_local_shard/group_count": len(records),
                "analysis/feasibility_local_shard/mean_feasible_count_in_group": statistics.mean(feasible_counts),
                "analysis/feasibility_local_shard/max_feasible_count_in_group": max(feasible_counts),
                "analysis/feasibility_local_shard/min_feasible_count_in_group": min(feasible_counts),
                "analysis/feasibility_local_shard/frac_groups_with_any_feasible": sum(any_feasible) / len(any_feasible),
                "analysis/feasibility_local_shard/feasible_completion_rate": total_feasible / total_group_size if total_group_size else 0.0,
            }
            if current_step is None:
                wandb.log(metrics)
            else:
                wandb.log(metrics, step=current_step)
    except Exception as exc:
        _log_instrumentation_error("feasibility_wandb", exc)

    try:
        run_id = os.environ.get("WANDB_RUN_ID") or os.environ.get("SLURM_JOB_ID") or "unknown"
        log_dir = Path(os.environ.get("FEASIBILITY_SPARSITY_LOG_DIR", "/workspace/logs/feasibility_sparsity"))
        run_dir = log_dir / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / f"rank{dist['rank']:05d}.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception as exc:
        _log_instrumentation_error("feasibility_sparsity", exc)


# --- ORACLE-ANCHORED REWARD HELPERS ---

def _calculate_score_trusted(mission, selection):
    """
    Calculates the SDS objective score using ground truth data.
    Objective = Sum(Weights) + Sum(Interactions)
    
    Args:
        mission: Mission dictionary with weights and interactions
        selection: List of selected variable indices
        
    Returns:
        float: Total objective score
    """
    if not selection:
        return 0.0
    
    score = 0.0
    sel_set = set(selection)
    
    # 1. Weights
    weights = mission.get('weights', [])
    for idx in selection:
        if 0 <= idx < len(weights):
            score += weights[idx]
    
    # 2. Interactions (Keys are "u,v")
    for k, v in mission.get('interactions', {}).items():
        try:
            u, v_idx = map(int, k.split(','))
            # Check if both nodes in interaction pair are selected
            if u in sel_set and v_idx in sel_set:
                score += v
        except (ValueError, IndexError):
            pass
    
    return score


def unified_format_reward(completions, **kwargs) -> list[float]:
    """
    Strict Format Reward (SDS-only).
    
    Enforces the following structure for SDS:
    1. Exactly one <think>...</think> block.
    2. Exactly one <code>...</code> block.
    3. The <think> block must appear BEFORE the <code> block.
    4. The <think> block must NOT contain <code> tags (prevents hallucinated nesting).
    """
    # Regex to capture the specific blocks and their positions
    # We use capture groups (.*?) to inspect what is inside
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    code_pattern = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
    
    rewards = []
    
    for idx, comp in enumerate(completions):
        # --- 1. Extract Content ---
        if isinstance(comp, list) and len(comp) > 0:
            content = comp[0].get("content", "")
        elif isinstance(comp, dict):
            content = comp.get("content", "")
        else:
            content = str(comp)
        
        # --- 2. Determine Domain ---
        domain = "sds"  # Default (SDS-only)
        if "domain" in kwargs:
            domain_data = kwargs["domain"]
            if isinstance(domain_data, list) and idx < len(domain_data):
                domain = domain_data[idx]
            else:
                domain = domain_data
        
        # If no explicit domain field, default to sds
        if not domain:
            domain = "sds"
        
        # --- 3. Strict Validation Logic ---
        
        # Find all matches (to check counts)
        think_matches = list(think_pattern.finditer(content))
        code_matches = list(code_pattern.finditer(content))
        
        # RULE A: Must have exactly ONE <think> block
        if len(think_matches) != 1:
            rewards.append(0.0)
            continue
        
        # RULE B: Check for "Code Hallucination" inside Think
        # The user's specific error was <think> ... <code> ... </think>
        # We check if the content INSIDE the think block contains the string "<code>"
        think_content = think_matches[0].group(1)
        if "<code>" in think_content or "</code>" in think_content:
            rewards.append(0.0)
            continue
        
        # SDS Strict Rules: Think + Code, No Answer
        
        # RULE C: Must have exactly ONE <code> block
        if len(code_matches) != 1:
            rewards.append(0.0)
            continue
        
        # RULE D: Order Check (Think must end before Code starts)
        think_end_idx = think_matches[0].end()
        code_start_idx = code_matches[0].start()
        
        if think_end_idx > code_start_idx:
            # This catches cases where code is before think, or overlapping
            rewards.append(0.0)
            continue
        
        # Pass!
        rewards.append(1.0)
    
    return rewards


def _quick_check_sds_constraints(mission, selection):  # noqa: PLR0912
    """
    Lightweight constraint checker to provide partial credit/penalty 
    inside the reward loop without overhead.
    
    Args:
        mission: Mission dictionary with constraints
        selection: List of selected variable indices
        
    Returns:
        int: Number of constraint violations (0 = feasible)
    """
    if not selection:
        return 10  # High penalty for empty (but not too harsh for gradient flow)
    
    sel_set = set(selection)
    violations = 0
    
    # 1. Precedence: if u is selected, v must be selected
    for u, v in mission.get('precedence', []):
        if u in sel_set and v not in sel_set:
            violations += 1
            
    # 2. Mutex: cannot select both
    for u, v in mission.get('mutex', []):
        if u in sel_set and v in sel_set:
            violations += 1
            
    # 3. Groups: max 1 per group
    # (Assuming groups is dict {id: [members]})
    groups = mission.get('groups', {})
    if isinstance(groups, dict):
        for members in groups.values():
            if sum(1 for m in members if m in sel_set) > 1:
                violations += 1
    elif isinstance(groups, list):  # Handle list format if applicable
        for group in groups:
            members = group.get('members', [])
            if sum(1 for m in members if m in sel_set) > 1:
                violations += 1
    
    # 4. Cardinality bounds
    cardinality_bounds = mission.get('cardinality_bounds', [0, float('inf')])
    if len(cardinality_bounds) >= _MIN_CARDINALITY_BOUNDS_LEN:
        min_card, max_card = cardinality_bounds[0], cardinality_bounds[1]
        if len(selection) < min_card or len(selection) > max_card:
            violations += 1
    
    return violations


def unified_code_execution_reward(completions, **kwargs) -> list[float]:  # noqa: PLR0912, PLR0915
    """
    Structurally-Guided Code Execution Reward (SDS-only).
    
    Implements a Curriculum of Dense Rewards to break Mode Collapse in Combinatorial Optimization:
    1. Syntax (0.1): Reward for running without error.
    2. Structure (0.2): Reward for using Graph Theory concepts (vs Lazy Sorting).
    3. Feasibility (0.5): Reward for respecting constraints (Partial credit applied).
    4. Optimality (Bonus): Reward for valid solutions.
    
    Curriculum:
    - Early Training (<40%): Heavy reward for Structure/Syntax to stop hallucinations.
    - Late Training (>40%): High penalty for invalid solutions to force correctness.
    """
    rewards = []
    
    # 1. EXTRACT TRAINING STATE (Curriculum)
    # GRPOTrainer injects 'trainer_state' into kwargs automatically
    trainer_state = kwargs.get("trainer_state")
    
    if trainer_state is not None and hasattr(trainer_state, 'global_step'):
        current_step = trainer_state.global_step
        total_steps = trainer_state.max_steps
    else:
        # Fallback for inference/debugging
        current_step = 0
        total_steps = 1000
    
    # Avoid division by zero
    progress = current_step / total_steps if total_steps > 0 else 0.0
    
    for idx, comp in enumerate(completions):
        # Handle both list format [{"content": "..."}] and direct format {"content": "..."}
        if isinstance(comp, list) and len(comp) > 0:
            text = comp[0].get("content", "")
        elif isinstance(comp, dict):
            text = comp.get("content", "")
        else:
            text = str(comp)
        
        # Get domain from dataset
        domain = None
        if "domain" in kwargs:
            domain_data = kwargs["domain"]
            if isinstance(domain_data, list) and idx < len(domain_data):
                domain = domain_data[idx]
            else:
                domain = domain_data
        
        # If no explicit domain field, default to sds
        if not domain:
            domain = "sds"
        
        # Extract code block
        code = extract_block(text, "code")
        if not code:
            rewards.append(0.0)
            continue
        
        # Validate code structure
        if not validate_code_structure(code):
            rewards.append(0.0)
            continue
        
        # Get mission parameters
        mission = {}
        if "mission" in kwargs:
            if isinstance(kwargs["mission"], list):
                mission = kwargs["mission"][idx] if idx < len(kwargs["mission"]) else {}
            else:
                mission = kwargs["mission"]
        
        # Deserialize mission if it's a JSON string (from HuggingFace dataset)
        mission = _deserialize_mission(mission)
        
        # Test code execution (SDS-only)
        try:
            if domain == "sds":
                # SDS code execution test
                # For SDS, mission contains the requirements directly
                # Build catalog from mission (if available) or use minimal example
                # Mission is now deserialized, so interactions should be a dict
                interactions = mission.get("interactions", {})
                weights = mission.get("weights", [1.0] * mission.get("n_variables", 10))
                
                # FIX: Reconstruct Adjacency List for Neighbors
                # Many algorithms crash if 'neighbors' is empty but 'interactions' exists.
                n_vars = mission.get("n_variables", 10)
                adj = {i: [] for i in range(n_vars)}
                for k in interactions:
                    try:
                        # keys are "i,j" format
                        u, v = map(int, k.split(','))
                        if u in adj and u < n_vars:
                            adj[u].append(v)
                        if v in adj and v < n_vars:
                            adj[v].append(u)
                    except (ValueError, IndexError):
                        # Skip malformed interaction keys
                        pass
                
                # FIX: Include interactions and weights in requirements to match training data format
                # This allows code to read from either requirements["interactions"] or catalog["interactions"]
                # Use **mission to spread all fields, then ensure weights/interactions are explicitly included
                test_requirements = {
                    **mission,  # Spread all mission fields (includes n_variables, bounds, precedence, etc.)
                    "weights": weights,  # Explicitly include weights in requirements (CRITICAL FOR SFT)
                    "interactions": interactions  # Explicitly include interactions in requirements (CRITICAL FOR SFT)
                }
                
                test_catalog = {
                    "variables": [
                        {
                            "id": j, 
                            "weight": weights[j] if j < len(weights) else 1.0, 
                            "neighbors": adj.get(j, [])  # Pass real neighbors, not empty list
                        }
                        for j in range(n_vars)
                    ],
                    "interactions": interactions  # Also in catalog for consistency
                }
                
                stdin_obj = {
                    "requirements": test_requirements,
                    "catalog": test_catalog
                }
            else:
                # Only SDS domain is supported
                rewards.append(0.0)
                continue
            
            # Run candidate program
            result = run_candidate(code, stdin_obj, timeout=5.0)
            
            # --- CALCULATE REWARD ---
            
            # Dense Structured Reward (SDS Domain - Anti-Mode-Collapse)
            
            # 1. Syntax (0.1) - Did it run?
            current_reward = 0.0
            if "error" not in result:
                current_reward += 0.1
            else:
                # If syntax error, stop here (0.0)
                rewards.append(0.0)
                continue
                
            if "selection" in result:
                current_reward += 0.1  # Valid Schema Bonus
            else:
                # Running but no output = minimal reward
                rewards.append(current_reward)
                continue
            
            # 2. Structure (0.2) - "Anti-Lazy" Layer
            # Force model to think about graphs, not just lists
            code_lower = code.lower()
            graph_keywords = [
                "networkx", "adjacency", "neighbor", "interactions", 
                "precedence", "mutex", "recursion", "memoization", "backtrack",
                "graph", "edge", "vertex", "topological", "dag"
            ]
            
            # Curriculum: Fade out structure reward later to focus on pure results
            structure_scale = 1.0 if progress < _CURRICULUM_THRESHOLD else 0.2
            
            if any(k in code_lower for k in graph_keywords):
                current_reward += (0.2 * structure_scale)
            
            # Explicit Penalty for "Lazy Sort Mode Collapse"
            # If it sorts by weight but ignores interactions completely
            if "sorted" in code_lower and "weight" in code_lower and "interactions" not in code_lower:
                current_reward -= 0.2
            
            # 3. Feasibility (0.3) - The "Constraint" Layer
            selection = result.get("selection", {}).get("variables", [])
            violations = _quick_check_sds_constraints(mission, selection)
            
            if violations == 0:
                current_reward += 0.3
            else:
                # Partial credit: Decay penalty based on count
                # -0.03 per violation, max penalty -0.2
                penalty = min(0.2, violations * 0.03)
                current_reward -= penalty
            
            # 4. Oracle-Anchored Optimality (0.4) - Compare to Greedy Baseline
            # This prevents mode collapse by ensuring solutions worse than greedy get negative signal
            # CRITICAL: Syndeopt must be available for SDS optimality rewards
            if violations == 0:
                if not _SYNDEOPT_AVAILABLE:
                    raise SyndeoptNotAvailableError("SDS optimality rewards require syndeopt to be installed and accessible.")  # noqa: TRY003, TRY301
                if GreedyMarginal is None:
                    raise GreedyMarginalNotAvailableError("SDS optimality rewards require GreedyMarginal from syndeopt.")  # noqa: TRY003, TRY301
                
                try:
                    # Convert mission to SDSInstance for greedy solver
                    from syndeopt.core.instance import (  # noqa: PLC0415
                        CardBounds,
                        SDSInstance,
                    )
                    
                    # Reconstruct interactions dict
                    # SDSInstance expects W keys where i < j (first element < second)
                    W = {}  # noqa: N806
                    for k, weight_val in mission.get("interactions", {}).items():
                        try:
                            u, v = map(int, k.split(','))
                            # Normalize to ensure u < v (required by SDSInstance)
                            if u > v:
                                u, v = v, u
                            W[(u, v)] = weight_val
                        except (ValueError, IndexError):
                            pass
                    
                    # Reconstruct constraints
                    prec = [tuple(x) for x in mission.get('precedence', [])]
                    mutex = [tuple(x) for x in mission.get('mutex', [])]
                    groups = mission.get('groups', {})
                    if isinstance(groups, dict):
                        groups = {int(k): v for k, v in groups.items()}
                    
                    weights = mission.get('weights', [])
                    n_vars = mission.get('n_variables', len(weights))
                    # Ensure weights list matches n_vars (pad with zeros if needed)
                    if len(weights) < n_vars:
                        weights = list(weights) + [0.0] * (n_vars - len(weights))
                    elif len(weights) > n_vars:
                        weights = weights[:n_vars]
                    
                    cardinality_bounds = mission.get('cardinality_bounds', [0, n_vars])
                    card = CardBounds(L=cardinality_bounds[0], U=cardinality_bounds[1])
                    
                    # Create instance
                    instance = SDSInstance(
                        n=n_vars,
                        w=weights,
                        W=W,
                        precedence=prec,
                        mutex=mutex,
                        groups=groups,
                        card=card
                    )
                    
                    # Run greedy oracle
                    # Use same timeout as LLM code execution (default 5.0) for consistency
                    # Can be overridden via kwargs if needed
                    oracle_timeout = kwargs.get("oracle_timeout", 5.0)
                    greedy_solver = GreedyMarginal()
                    oracle_result = greedy_solver.solve(instance, budget_sec=oracle_timeout, seed=0)
                    oracle_score = oracle_result.score
                    
                    # Calculate LLM score
                    llm_score = _calculate_score_trusted(mission, selection)
                    
                    # Oracle-anchored reward: normalized difference (handles negative scores correctly)
                    # This provides continuous feedback: negative if worse than greedy, zero if equal, positive if better
                    # FIX: Use normalized difference instead of ratio to handle negative oracle scores correctly
                    if abs(oracle_score) > _EPSILON_TINY:
                        # Fix: Use Normalized Difference instead of Ratio
                        # This works correctly even if scores are negative.
                        # (LLM - Oracle) is positive if LLM is better.
                        raw_diff = llm_score - oracle_score
                        
                        # Normalize by the magnitude of the oracle score 
                        # (max 1.0 prevents explosion on tiny scores)
                        norm_diff = raw_diff / max(abs(oracle_score), 1.0)
                        
                        # --- NEW AGGRESSIVE SCALING (AlphaZero Logic) ---
                        if norm_diff > _ORACLE_MATCH_THRESHOLD: 
                            # BEATING GREEDY: Massive bonus
                            # +1.0 base, plus huge scaler for margin
                            oracle_reward = 1.0 + (norm_diff * 10.0)
                        elif norm_diff > -_ORACLE_MATCH_THRESHOLD:
                            # MATCHING GREEDY: Zero reward.
                            # We force the model to abandon the local optimum.
                            oracle_reward = 0.0
                        else:
                            # LOSING TO GREEDY: Punishment.
                            oracle_reward = -0.5
                            
                        current_reward += oracle_reward
                    elif llm_score > 0:
                        # Edge case: Oracle is exactly 0.
                        current_reward += 0.2
                    elif llm_score < 0:
                        current_reward -= 0.1
                            
                except Exception:
                    # If oracle fails, no optimality reward (syndeopt must be available)
                    # Reward stops at feasibility layer (max 0.5: 0.1 syntax + 0.1 schema + 0.3 feasibility)
                    pass
            # If syndeopt not available, no optimality reward is given
            # Reward stops at feasibility layer (max 0.5: 0.1 syntax + 0.1 schema + 0.3 feasibility)
            
            # Clip to [-1.0, 1.0] to preserve negative feedback signals
            # This allows penalties (e.g., from oracle anchor or lazy sorting) to be preserved
            rewards.append(max(-1.0, min(1.0, current_reward)))
                
        except Exception:
            rewards.append(0.0)
    
    return rewards


def unified_generalization_reward(completions, **kwargs) -> list[float]:  # noqa: PLR0912, PLR0915
    """
    Generalization reward - tests code on random requirements using clean simulators.
    Uses PROMPT-BASED HASHING for publication-grade reproducibility.
    
    Reproducibility Strategy:
    - Uses prompt content (not dataset index) as seed anchor
    - Same prompt = same generalization tests, regardless of dataset shuffling
    - Resume-safe: dataloader restores to same index, so prompts[idx] is identical
    """
    # A. Get Global Experiment Seed (Set in Bash via export SEED=...)
    # Priority: 1. Environment Variable (passed from Bash), 2. Kwargs, 3. Default (42)
    env_seed = os.environ.get("SEED")
    if env_seed is not None:
        try:
            experiment_seed = int(env_seed)
        except ValueError:
            experiment_seed = kwargs.get("seed", 42)
    else:
        experiment_seed = kwargs.get("seed", 42)
    
    print(f"DEBUG: Generalization Reward utilizing Seed: {experiment_seed}")  # Verify this in logs!
    
    # B. Get Prompts for Hashing
    # GRPOTrainer passes 'prompts' in kwargs. They align 1:1 with 'completions'.
    prompts = kwargs.get("prompts", [])
    
    rewards = []
    
    for idx, comp in enumerate(completions):
        # Handle both list format [{"content": "..."}] and direct format {"content": "..."}
        if isinstance(comp, list) and len(comp) > 0:
            text = comp[0].get("content", "")
        elif isinstance(comp, dict):
            text = comp.get("content", "")
        else:
            text = str(comp)
        
        # Get domain from dataset
        domain = None
        if "domain" in kwargs:
            domain_data = kwargs["domain"]
            if isinstance(domain_data, list) and idx < len(domain_data):
                domain = domain_data[idx]
            else:
                domain = domain_data
        
        # If no explicit domain field, default to sds
        if not domain:
            domain = "sds"
        
        # Extract code block
        code = extract_block(text, "code")
        if not code:
            rewards.append(0.0)
            continue
        
        # Validate code structure
        if not validate_code_structure(code):
            rewards.append(0.0)
            continue
        
        # Generate random test requirements based on domain
        test_scores = []
        
        # Check if syndeopt is available for SDS domain
        # CRITICAL: Syndeopt must be available for SDS generalization rewards
        if domain == "sds":
            if not _SYNDEOPT_AVAILABLE:
                raise SyndeoptNotAvailableError("SDS generalization rewards require syndeopt to be installed and accessible.")  # noqa: TRY003
            if GreedyMarginal is None:
                raise GreedyMarginalNotAvailableError("SDS generalization rewards require GreedyMarginal from syndeopt.")  # noqa: TRY003
        
        # C. Get the Prompt Content for this specific completion
        # This is the anchor. If training resumes, this prompt content 
        # will be identical for this dataset index.
        #
        # Why prompt-based hashing instead of index-based?
        # 1. Zero Config: Don't need to inject global_step or other training state
        # 2. Dataset Invariance: Same prompt = same tests, even if dataset is shuffled differently
        #    (e.g., Seed 202 vs Seed 101). This is scientifically cleaner - we test if the model
        #    can solve THIS specific problem + these specific generalization checks.
        # 3. Resume Safety: Accelerate restores dataloader to exact index, so prompts[idx] is identical
        # 4. Problem-Centric: Generalization tests are tied to the problem itself, not position
        if idx < len(prompts):
            # Prompts can be strings or list of dicts (chat format)
            # Converting to str() handles both deterministically.
            prompt_obj = prompts[idx]
            if isinstance(prompt_obj, list):
                # Chat format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
                # Extract user content (the actual problem)
                prompt_content = ""
                for msg in prompt_obj:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        prompt_content = msg.get("content", "")
                        break
                if not prompt_content:
                    # Fallback: stringify entire prompt list
                    prompt_content = str(prompt_obj)
            elif isinstance(prompt_obj, dict):
                # Single dict format: {"role": "user", "content": "..."} or {"prompt": [...]}
                if "content" in prompt_obj:
                    prompt_content = prompt_obj["content"]
                elif "prompt" in prompt_obj:
                    # Nested prompt structure
                    prompt_content = str(prompt_obj["prompt"])
                else:
                    prompt_content = str(prompt_obj)
            else:
                # String format or other
                prompt_content = str(prompt_obj)
        else:
            # Fallback (should not happen in TRL, but safe guard)
            prompt_content = f"unknown_prompt_{idx}"
        
        # D. Run 5 Generalization Tests
        for test_i in range(5):  # Test on 5 random requirements (test_i: 0 to 4)
            # This ensures:
            # - Same prompt = same tests (dataset shuffle invariant)
            # - Resume-safe (dataloader restores to same index, prompts[idx] is identical)
            # - Different test iterations get different seeds
            local_seed = get_deterministic_seed(experiment_seed, prompt_content, test_i)
            
            # Create a localized random generator
            # Do NOT use 'random.uniform' directly anymore. Use 'rng.uniform'
            rng = random.Random(local_seed)
            
            try:
                if domain == "sds":
                    # Generate random SDS requirements using syndeopt
                    # Use showcase instances to match dataset generation
                    # (Imports and path setup already done above, outside the loop)
                    
                    # UPDATED: Weighted Problem Sampling (Curriculum of Hardness)
                    # STRATEGY:
                    # 1. "Teachers" (60%): Hard instances that punish greedy/lazy logic.
                    # 2. "Scale" (15%): Large N instances to verify algorithmic scaling.
                    # 3. "Diversity" (25%): Pattern-matching tasks to prevent overfitting.
                    problem_types = [
                        # --- The Teachers (Compliance & Optimization) ---
                        "structural_trap",  # 25%: Forces Precedence compliance (Kills constraint-ignoring code)
                        "dense",            # 20%: Forces non-greedy search (Kills naive greedy)
                        "bnb_showcase",     # 15%: Forces deep lookahead (Kills shallow heuristics)
                        # --- The Scale Check ---
                        "random_sds",       # 15%: N=50-100 check (Kills brute force / slow code)
                        # --- The Diversity Mix (Prevent Overfitting) ---
                        "maxcut_qubo",      # 5%
                        "planted_qubo",      # 5%
                        "tree",             # 5%
                        "decomposable",     # 5%
                        "greedy_easy",      # 5%
                    ]
                    # Weights corresponding to the list above
                    weights = [0.25, 0.20, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05]
                    
                    # Select one problem type based on weights
                    problem_type = rng.choices(problem_types, weights=weights, k=1)[0]
                    
                    # Use parameter ranges matching dataset generation (gen_sds_dataset.py)
                    # SDS Generators accept a seed argument directly - we pass the local_seed
                    if problem_type == "tree":
                        n = rng.randint(8, 18)
                        card = (max(2, n//4), min(n, n//2 + 3))
                        inst = make_tree_showcase_instance(n=n, card=card, seed=local_seed)
                    elif problem_type == "dense":
                        n = rng.randint(10, 20)
                        card = (max(3, n//3), min(n, n//2 + 4))
                        inst = make_dense_deceptive_instance(n=n, card=card, seed=local_seed)
                    elif problem_type == "decomposable":
                        n = rng.randint(12, 20)
                        card = (max(3, n//4), min(n, n//2 + 5))
                        clusters = rng.randint(2, 4)
                        inst = make_decomposable_instance(n=n, card=card, clusters=clusters, seed=local_seed)
                    elif problem_type == "greedy_easy":
                        n = rng.randint(8, 18)
                        card = (max(2, n//3), min(n, n//2 + 2))
                        inst = make_greedy_easy_instance(n=n, card=card, seed=local_seed)
                    elif problem_type == "local_optima":
                        n = rng.randint(10, 20)
                        card = (max(3, n//3), min(n, n//2 + 3))
                        inst = make_local_optima_instance(n=n, card=card, seed=local_seed)
                    elif problem_type == "bnb_showcase":
                        n = rng.randint(14, 20)
                        card = (max(4, n//3), min(n, n//2 + 4))
                        inst = make_dense_deceptive_instance(
                            n=n, 
                            card=card, 
                            pos_pair_frac=0.55,
                            neg_pair_frac=0.45,
                            seed=local_seed
                        )
                    elif problem_type == "random_qubo":
                        n = rng.randint(10, 20)
                        card = (max(0, n//4), min(n, n//2 + 5))
                        diag_scale = rng.uniform(0.5, 2.0)
                        offdiag_scale = rng.uniform(0.5, 2.0)
                        density = rng.uniform(0.3, 0.7)
                        inst = make_random_qubo_instance(
                            n=n, card=card, diag_scale=diag_scale,
                            offdiag_scale=offdiag_scale, density=density, seed=local_seed
                        )
                    elif problem_type == "planted_qubo":
                        n = rng.randint(10, 20)
                        card = (max(0, n//4), min(n, n//2 + 5))
                        density = rng.uniform(0.3, 0.7)
                        signal_strength = rng.uniform(1.5, 3.0)
                        noise_scale = rng.uniform(0.3, 0.8)
                        inst = make_planted_qubo_instance(
                            n=n, card=card, density=density,
                            signal_strength=signal_strength, noise_scale=noise_scale, seed=local_seed
                        )
                    elif problem_type == "maxcut_qubo":
                        n = rng.randint(10, 20)
                        card = (max(0, n//4), min(n, n//2 + 5))
                        edge_prob = rng.uniform(0.3, 0.7)
                        weight_scale = rng.uniform(0.5, 2.0)
                        inst = make_maxcut_qubo_instance(
                            n=n, edge_prob=edge_prob, weight_scale=weight_scale, card=card, seed=local_seed
                        )
                    elif problem_type == "random_sds":
                        # Scale check: N=50-100 to verify algorithmic scaling (kills brute force / slow code)
                        n = rng.randint(50, 100)  # Large N range for runtime complexity filter
                        # Note: Local helper function must accept seed parameter
                        inst = _make_random_sds_instance_local(n=n, seed=local_seed)
                    elif problem_type == "structural_trap":
                        # [NEW] Structural trap: Chain of pain (precedence-dominant)
                        # Matches dataset generation parameters for consistency
                        n = rng.randint(18, 28)
                        chain_length = rng.randint(4, 7)  # Chain length varies
                        inst = make_structural_trap_instance(
                            n=n,
                            chain_length=chain_length,
                            bait_reward=100.0,  # High reward at leaf
                            trap_penalty=-10.0,  # Penalty for each parent
                            seed=local_seed
                        )
                    else:
                        # Fallback
                        n = rng.randint(8, 18)
                        card = (max(2, n//4), min(n, n//2 + 3))
                        inst = make_tree_showcase_instance(n=n, card=card, seed=local_seed)
                    
                    # Convert to requirements format (syndeopt uses CardBounds)
                    test_requirements = {
                        "n_variables": inst.n,
                        "cardinality_bounds": [inst.card.L, inst.card.U],  # syndeopt uses CardBounds
                        "precedence": inst.precedence,
                        "mutex": inst.mutex,
                        "groups": inst.groups,
                        "weights": inst.w,
                        "interactions": {f"{i},{j}": inst.W[(i,j)] for (i,j) in inst.W}
                    }
                    
                    # Build catalog for stdin
                    test_catalog = {
                        "variables": [
                            {"id": j, "weight": inst.w[j], "neighbors": list(inst.adj[j])}
                            for j in range(inst.n)
                        ],
                        "interactions": {
                            f"{i},{j}": inst.W[(i,j)]
                            for (i,j) in inst.W
                        }
                    }
                    
                    # Execute LLM code to get selection
                    stdin_obj = {
                        "requirements": test_requirements,
                        "catalog": test_catalog
                    }
                    result = run_candidate(code, stdin_obj)
                    
                    if "error" in result:
                        # Assign 0.0 for crashes so they contribute to variance (inconsistency penalty)
                        test_scores.append(0.0)
                        continue
                    
                    # Extract selection from code execution result
                    if "selection" in result:
                        selection_obj = result["selection"]
                        if "variables" in selection_obj:
                            test_design = selection_obj["variables"]
                        else:
                            test_design = selection_obj
                        
                        # Use clean simulator
                        try:
                            reward = registry.get_reward("sds", test_design, test_requirements)
                            # Validate reward is a valid float in [0, 1] range
                            if reward is None:
                                print(f"WARNING: SDS reward returned None for design {test_design}")
                                test_scores.append(0.0)
                            elif not isinstance(reward, (int, float)):
                                print(f"WARNING: SDS reward is not a number: {reward} (type: {type(reward)})")
                                test_scores.append(0.0)
                            elif not (0.0 <= reward <= 1.0):
                                print(f"WARNING: SDS reward {reward} outside [0, 1] range for design {test_design}")
                                print(f"  Requirements keys: {list(test_requirements.keys())}")
                                print(f"  n_variables: {test_requirements.get('n_variables')}")
                                print(f"  weights length: {len(test_requirements.get('weights', []))}")
                                print(f"  interactions count: {len(test_requirements.get('interactions', {}))}")
                                # Clip to valid range instead of appending invalid value
                                test_scores.append(max(0.0, min(1.0, float(reward))))
                            else:
                                test_scores.append(float(reward))
                        except Exception as err:
                            print(f"ERROR: Exception in SDS get_reward: {err}")
                            traceback.print_exc()
                            test_scores.append(0.0)
                    else:
                        # Assign 0.0 for missing selection so it contributes to variance
                        test_scores.append(0.0)
                else:
                    # Only SDS domain is supported
                    test_scores.append(0.0)
                    
            except Exception as err:
                print(f"Generalization test error: {err}")
                # Assign 0.0 for exceptions so they contribute to variance (inconsistency penalty)
                test_scores.append(0.0)
        
        # Calculate magnitude-weighted stability score
        if len(test_scores) == 0:
            rewards.append(0.0)
            continue
        
        # Use all scores (crashes are now 0.0, not filtered out)
        # This ensures crashes contribute to variance, properly penalizing inconsistent code
        all_scores = test_scores
        
        if len(all_scores) < _MIN_TEST_INSTANCES:  # Need at least 3 test instances for meaningful statistics
            rewards.append(0.0)
            continue
        
        # Validate all scores are valid numbers before calculating statistics
        # This catches None, inf, nan, or invalid types that might have slipped through
        # Note: We allow scores outside [0, 1] to pass through so we can detect normalization bugs
        # but we'll log warnings to help debug
        valid_scores = []
        for score_idx, score in enumerate(all_scores):
            if score is None:
                print(f"WARNING: Found None at index {score_idx} in test_scores, replacing with 0.0")
                valid_scores.append(0.0)
            elif not isinstance(score, (int, float)):
                print(f"WARNING: Found non-numeric value {score} (type: {type(score)}) at index {score_idx}, replacing with 0.0")
                valid_scores.append(0.0)
            elif not (0.0 <= score <= 1.0):
                # Score outside [0, 1] indicates a normalization bug - log detailed warning
                print(f"ERROR: Found score {score} outside [0, 1] at index {score_idx}!")
                print("  This indicates a bug in reward normalization. Score should be in [0, 1].")
                print(f"  All scores: {all_scores}")
                # Replace with 0.0 to prevent numerical explosion, but this is a bug that needs fixing
                valid_scores.append(0.0)
            else:
                valid_scores.append(float(score))
        
        # Use validated scores for statistics
        all_scores = valid_scores
        
        # Calculate mean and variance (including crashes as 0.0)
        mean_score = statistics.mean(all_scores)
        variance = statistics.variance(all_scores) if len(all_scores) > 1 else 0.0
        
        # Final validation: mean_score should be in [0, 1] if individual scores are valid
        # If not, there's a bug in the normalization logic above
        if not (0.0 <= mean_score <= 1.0):
            print(f"ERROR: mean_score {mean_score} outside [0, 1]!")
            print(f"  Individual scores: {all_scores}")
            print("  This indicates a normalization bug in the simulator.")
            # Don't clip - return 0.0 to signal error, but log the issue
            rewards.append(0.0)
            continue
        
        # Calculate magnitude-weighted stability score: R = mean³ / (mean² + variance)
        numerator = mean_score**3
        denominator = mean_score**2 + variance
        
        reward = 0.0 if denominator < _EPSILON_TINY else numerator / denominator
        
        # Validate final reward is in [0, 1] - if not, there's a bug
        # The magnitude-weighted formula should produce values in [0, 1] when mean_score is in [0, 1]
        if not (0.0 <= reward <= 1.0):
            print(f"ERROR: Final reward {reward} outside [0, 1]!")
            print(f"  mean_score: {mean_score}, variance: {variance}")
            print(f"  numerator: {numerator}, denominator: {denominator}")
            print("  This indicates a bug in the aggregation formula.")
            # Return 0.0 to signal error, but don't clip (clipping hides bugs)
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards



def unified_nominal_reward(completions, **kwargs) -> list[float]:  # noqa: PLR0912, PLR0915
    """
    ABLATION REWARD: Nominal Signal Only.
    
    Evaluates the generated code ONLY on the specific 'mission' provided in the prompt.
    This replaces the 5-sample generalization test for ablation studies.
    
    Returns rewards in [0, 1] range to match unified_generalization_reward scale.
    """
    return _unified_nominal_reward_with_simulator_config(completions, None, **kwargs)


def unified_nominal_reward_topk_interaction_bound(
    completions, **kwargs
) -> list[float]:  # noqa: PLR0912, PLR0915
    """
    ABLATION REWARD: Nominal reward with top-k interaction normalization.

    This keeps the nominal SDS reward path unchanged except for the simulator's
    optimistic interaction normalization term.
    """
    return _unified_nominal_reward_with_simulator_config(
        completions,
        _TOPK_POSITIVE_INTERACTION_CONFIG,
        **kwargs,
    )


def _unified_nominal_reward_with_simulator_config(
    completions,
    simulator_config,
    **kwargs,
) -> list[float]:  # noqa: PLR0912, PLR0915
    """Shared nominal SDS reward path with optional simulator configuration."""
    rewards = []
    
    # Handle GRPOTrainer inputs
    missions_data = kwargs.get("mission", [])
    domain_data = kwargs.get("domain", "sds")
    
    for idx, comp in enumerate(completions):
        # Extract Text
        if isinstance(comp, list) and len(comp) > 0:
            text = comp[0].get("content", "")
        elif isinstance(comp, dict):
            text = comp.get("content", "")
        else:
            text = str(comp)
        
        # Extract code block
        code = extract_block(text, "code")
        
        # 1. Basic Validation (minimal - just to save compute on broken code)
        if not code or not validate_code_structure(code):
            rewards.append(0.0)
            continue
        
        # 2. Extract Mission
        raw_mission = missions_data[idx] if isinstance(missions_data, list) and idx < len(missions_data) else missions_data
        mission = _deserialize_mission(raw_mission)
        
        # 3. Extract Domain
        current_domain = domain_data[idx] if isinstance(domain_data, list) and idx < len(domain_data) else domain_data
        if not current_domain:
            current_domain = "sds"
        
        # 4. Fail gracefully for non-SDS domains
        if current_domain != "sds":
            rewards.append(0.0)
            continue
        
        try:
            # 5. Setup Simulator Input (SDS Logic)
            interactions = mission.get("interactions", {})
            weights = mission.get("weights", [])
            n_vars = mission.get("n_variables", 10)
            
            # Reconstruct neighbors
            adj = {i: [] for i in range(n_vars)}
            for k in interactions:
                try:
                    u, v = map(int, k.split(','))
                    if u < n_vars:
                        adj[u].append(v)
                    if v < n_vars:
                        adj[v].append(u)
                except (ValueError, IndexError):
                    pass
            
            test_requirements = {**mission, "weights": weights, "interactions": interactions}
            test_catalog = {
                "variables": [
                    {"id": j, "weight": weights[j] if j < len(weights) else 1.0, "neighbors": adj.get(j, [])}
                    for j in range(n_vars)
                ],
                "interactions": interactions
            }
            stdin_obj = {"requirements": test_requirements, "catalog": test_catalog}
            
            # 6. Execute
            result = run_candidate(code, stdin_obj, timeout=2.0)
            if "error" in result:
                rewards.append(0.0)
                continue
            
            # 7. Score (Pure Simulator Value)
            if "selection" in result:
                selection_obj = result["selection"]
                test_design = selection_obj.get("variables", []) if isinstance(selection_obj, dict) else selection_obj
                
                reward = registry.get_reward(
                    current_domain,
                    test_design,
                    test_requirements,
                    config=simulator_config,
                )
                rewards.append(float(max(0.0, min(1.0, reward if reward else 0.0))))
            else:
                rewards.append(0.0)
        
        except Exception:
            rewards.append(0.0)
    
    return rewards


def unified_soft_nominal_reward(completions, **kwargs) -> list[float]:  # noqa: PLR0912, PLR0915
    """
    Soft-gated SDS nominal reward.

    Keeps Hero's nominal score normalization but replaces the hard infeasible
    gate with a fixed penalty on simulator-reported constraint violations.
    """
    rewards = []

    missions_data = kwargs.get("mission", [])
    domain_data = kwargs.get("domain", "sds")

    for idx, comp in enumerate(completions):
        if isinstance(comp, list) and len(comp) > 0:
            text = comp[0].get("content", "")
        elif isinstance(comp, dict):
            text = comp.get("content", "")
        else:
            text = str(comp)

        code = extract_block(text, "code")
        if not code or not validate_code_structure(code):
            rewards.append(0.0)
            continue

        raw_mission = missions_data[idx] if isinstance(missions_data, list) and idx < len(missions_data) else missions_data
        mission = _deserialize_mission(raw_mission)

        current_domain = domain_data[idx] if isinstance(domain_data, list) and idx < len(domain_data) else domain_data
        if not current_domain:
            current_domain = "sds"

        if current_domain != "sds":
            rewards.append(0.0)
            continue

        try:
            interactions = mission.get("interactions", {})
            weights = mission.get("weights", [])
            n_vars = mission.get("n_variables", 10)

            adj = {i: [] for i in range(n_vars)}
            for k in interactions:
                try:
                    u, v = map(int, k.split(','))
                    if u < n_vars:
                        adj[u].append(v)
                    if v < n_vars:
                        adj[v].append(u)
                except (ValueError, IndexError):
                    pass

            test_requirements = {**mission, "weights": weights, "interactions": interactions}
            test_catalog = {
                "variables": [
                    {"id": j, "weight": weights[j] if j < len(weights) else 1.0, "neighbors": adj.get(j, [])}
                    for j in range(n_vars)
                ],
                "interactions": interactions
            }
            stdin_obj = {"requirements": test_requirements, "catalog": test_catalog}

            result = run_candidate(code, stdin_obj, timeout=2.0)
            if "error" in result or "selection" not in result:
                rewards.append(0.0)
                continue

            selection_obj = result["selection"]
            test_design = selection_obj.get("variables", []) if isinstance(selection_obj, dict) else selection_obj
            sim_result = registry.simulate(current_domain, test_design, test_requirements)

            raw_score = float(sim_result.get("score", 0.0) or 0.0)
            violation_count = float(sim_result.get("constraint_violations", 0.0) or 0.0)
            normalized_score = normalize_sds_score(raw_score, test_requirements)

            soft_reward = normalized_score - (_SDS_SOFT_GATE_LAMBDA * violation_count)
            rewards.append(float(max(0.0, min(1.0, soft_reward))))
        except Exception:
            rewards.append(0.0)

    return rewards


def unified_code_execution_reward_no_oracle(completions, **kwargs) -> list[float]:  # noqa: PLR0912, PLR0915
    """
    ABLATION REWARD: Code Execution WITHOUT Oracle Anchoring.
    
    Same as unified_code_execution_reward, but WITHOUT the Oracle-Anchored Optimality block.
    This allows us to test if the oracle signal is necessary for learning.
    
    Max reward: 0.6 (Syntax 0.1 + Schema 0.1 + Structure 0.2 + Feasibility 0.3)
    We do NOT renormalize to preserve the same component weights for fair ablation.
    """
    rewards = []
    feasible_flags = []
    
    # 1. EXTRACT TRAINING STATE (Curriculum)
    # GRPOTrainer injects 'trainer_state' into kwargs automatically
    trainer_state = kwargs.get("trainer_state")
    
    if trainer_state is not None and hasattr(trainer_state, 'global_step'):
        current_step = trainer_state.global_step
        total_steps = trainer_state.max_steps
    else:
        # Fallback for inference/debugging
        current_step = 0
        total_steps = 1000
    
    # Avoid division by zero
    progress = current_step / total_steps if total_steps > 0 else 0.0
    
    for idx, comp in enumerate(completions):
        # Handle both list format [{"content": "..."}] and direct format {"content": "..."}
        if isinstance(comp, list) and len(comp) > 0:
            text = comp[0].get("content", "")
        elif isinstance(comp, dict):
            text = comp.get("content", "")
        else:
            text = str(comp)
        
        # Get domain from dataset
        domain = None
        if "domain" in kwargs:
            domain_data = kwargs["domain"]
            if isinstance(domain_data, list) and idx < len(domain_data):
                domain = domain_data[idx]
            else:
                domain = domain_data
        
        # If no explicit domain field, default to sds
        if not domain:
            domain = "sds"
        
        # Extract code block
        code = extract_block(text, "code")
        if not code:
            rewards.append(0.0)
            feasible_flags.append(False)
            continue
        
        # Validate code structure
        if not validate_code_structure(code):
            rewards.append(0.0)
            feasible_flags.append(False)
            continue
        
        # Get mission parameters
        mission = {}
        if "mission" in kwargs:
            if isinstance(kwargs["mission"], list):
                mission = kwargs["mission"][idx] if idx < len(kwargs["mission"]) else {}
            else:
                mission = kwargs["mission"]
        
        # Deserialize mission if it's a JSON string (from HuggingFace dataset)
        mission = _deserialize_mission(mission)
        
        # Test code execution (SDS-only)
        try:
            if domain == "sds":
                # SDS code execution test
                # For SDS, mission contains the requirements directly
                # Build catalog from mission (if available) or use minimal example
                # Mission is now deserialized, so interactions should be a dict
                interactions = mission.get("interactions", {})
                weights = mission.get("weights", [1.0] * mission.get("n_variables", 10))
                
                # FIX: Reconstruct Adjacency List for Neighbors
                # Many algorithms crash if 'neighbors' is empty but 'interactions' exists.
                n_vars = mission.get("n_variables", 10)
                adj = {i: [] for i in range(n_vars)}
                for k in interactions:
                    try:
                        # keys are "i,j" format
                        u, v = map(int, k.split(','))
                        if u in adj and u < n_vars:
                            adj[u].append(v)
                        if v in adj and v < n_vars:
                            adj[v].append(u)
                    except (ValueError, IndexError):
                        # Skip malformed interaction keys
                        pass
                
                # FIX: Include interactions and weights in requirements to match training data format
                # This allows code to read from either requirements["interactions"] or catalog["interactions"]
                # Use **mission to spread all fields, then ensure weights/interactions are explicitly included
                test_requirements = {
                    **mission,  # Spread all mission fields (includes n_variables, bounds, precedence, etc.)
                    "weights": weights,  # Explicitly include weights in requirements (CRITICAL FOR SFT)
                    "interactions": interactions  # Explicitly include interactions in requirements (CRITICAL FOR SFT)
                }
                
                test_catalog = {
                    "variables": [
                        {
                            "id": j, 
                            "weight": weights[j] if j < len(weights) else 1.0, 
                            "neighbors": adj.get(j, [])  # Pass real neighbors, not empty list
                        }
                        for j in range(n_vars)
                    ],
                    "interactions": interactions  # Also in catalog for consistency
                }
                
                stdin_obj = {
                    "requirements": test_requirements,
                    "catalog": test_catalog
                }
            else:
                # Only SDS domain is supported
                rewards.append(0.0)
                feasible_flags.append(False)
                continue
            
            # Run candidate program
            result = run_candidate(code, stdin_obj, timeout=5.0)
            
            # --- CALCULATE REWARD ---
            
            # Dense Structured Reward (SDS Domain - WITHOUT Oracle)
            
            # 1. Syntax (0.1) - Did it run?
            current_reward = 0.0
            if "error" not in result:
                current_reward += 0.1
            else:
                # If syntax error, stop here (0.0)
                rewards.append(0.0)
                feasible_flags.append(False)
                continue
                
            if "selection" in result:
                current_reward += 0.1  # Valid Schema Bonus
            else:
                # Running but no output = minimal reward
                rewards.append(current_reward)
                feasible_flags.append(False)
                continue
            
            # 2. Structure (0.2) - "Anti-Lazy" Layer
            # Force model to think about graphs, not just lists
            code_lower = code.lower()
            graph_keywords = [
                "networkx", "adjacency", "neighbor", "interactions", 
                "precedence", "mutex", "recursion", "memoization", "backtrack",
                "graph", "edge", "vertex", "topological", "dag"
            ]
            
            # Curriculum: Fade out structure reward later to focus on pure results
            structure_scale = 1.0 if progress < _CURRICULUM_THRESHOLD else 0.2
            
            if any(k in code_lower for k in graph_keywords):
                current_reward += (0.2 * structure_scale)
            
            # Explicit Penalty for "Lazy Sort Mode Collapse"
            # If it sorts by weight but ignores interactions completely
            if "sorted" in code_lower and "weight" in code_lower and "interactions" not in code_lower:
                current_reward -= 0.2
            
            # 3. Feasibility (0.3) - The "Constraint" Layer
            selection = result.get("selection", {}).get("variables", [])
            exact_feasible = False
            if isinstance(selection, list):
                try:
                    sim_results = registry.simulate("sds", selection, test_requirements)
                    exact_feasible = bool(sim_results.get("feasible", False))
                except Exception:
                    exact_feasible = False
            violations = _quick_check_sds_constraints(mission, selection)
            
            if violations == 0:
                current_reward += 0.3
            else:
                # Partial credit: Decay penalty based on count
                # -0.03 per violation, max penalty -0.2
                penalty = min(0.2, violations * 0.03)
                current_reward -= penalty
            
            # 4. Oracle-Anchored Optimality (0.4) - REMOVED FOR ABLATION
            # This block is intentionally omitted to test if oracle signal is necessary
            
            # Clip to [-1.0, 1.0] to preserve negative feedback signals
            # This allows penalties (e.g., from lazy sorting) to be preserved
            rewards.append(max(-1.0, min(1.0, current_reward)))
            feasible_flags.append(exact_feasible)
                
        except Exception:
            rewards.append(0.0)
            feasible_flags.append(False)

    logging_context = _build_logging_context(trainer_state)

    _log_group_feasibility_stats(
        prompts=kwargs.get("prompts", []),
        missions=kwargs.get("mission", []),
        feasible_flags=feasible_flags,
        trainer_state=trainer_state,
        logging_context=logging_context,
    )
    _log_generation_traces(
        completions=completions,
        prompts=kwargs.get("prompts", []),
        missions=kwargs.get("mission", []),
        rewards=rewards,
        feasible_flags=feasible_flags,
        trainer_state=trainer_state,
        problem_uuids=kwargs.get("problem_uuid", kwargs.get("uuid")),
        problem_prompt_hashes=kwargs.get("problem_prompt_hash"),
        problem_mission_hashes=kwargs.get("problem_mission_hash"),
        logging_context=logging_context,
    )

    return rewards

def minimal_feasibility_reward(completions, **kwargs) -> list[float]:  # noqa: PLR0912, PLR0915
    """
    TRUE MINIMALIST REWARD: Syntax + Schema + Feasibility.
    
    Removes all "engineering hacks":
    - No keyword/structure detection (no 'networkx' bonus).
    - No curriculum fading.
    - No partial credit for violations.
    - No 'lazy sort' penalties.
    
    Returns:
        1.0 if Valid Solution (Runs + Returns JSON + No Constraint Violations)
        0.5 if Execution Success (Runs + Returns JSON + But Invalid Solution)
        0.1 if Syntax Success (Runs + Wrong Schema)
        0.0 otherwise.
    """
    rewards = []
    
    for idx, comp in enumerate(completions):
        # 1. Extract Code
        # (Standard text extraction logic)
        if isinstance(comp, list) and len(comp) > 0:
            text = comp[0].get("content", "")
        elif isinstance(comp, dict):
            text = comp.get("content", "")
        else:
            text = str(comp)
        
        code = extract_block(text, "code")
        if not code:
            rewards.append(0.0)
            continue

        # Optional but consistent with other SDS reward funcs: reject obviously malformed code
        if not validate_code_structure(code):
            rewards.append(0.0)
            continue
            
        # 2. Get Mission/Inputs
        # (Standard boilerplate to get inputs for the test)
        mission = kwargs.get("mission", [])
        if isinstance(mission, list): 
            mission = mission[idx] if idx < len(mission) else {}
        mission = _deserialize_mission(mission)

        # Determine domain (default to SDS)
        domain = "sds"
        if "domain" in kwargs:
            domain_data = kwargs["domain"]
            if isinstance(domain_data, list) and idx < len(domain_data):
                domain = domain_data[idx] or "sds"
            else:
                domain = domain_data or "sds"
        if domain != "sds":
            rewards.append(0.0)
            continue
        
        # Prepare STDIN payload (MUST match the same schema used elsewhere in this module)
        # Be strict: we should NEVER silently fabricate SDS fields.
        # If these are missing/malformed, treat as failure (0.0) so issues surface immediately.
        if "n_variables" not in mission or "weights" not in mission or "interactions" not in mission:
            rewards.append(0.0)
            continue
        n_vars = mission["n_variables"]
        weights = mission["weights"]
        interactions = mission["interactions"]

        if not isinstance(n_vars, int) or n_vars <= 0:
            rewards.append(0.0)
            continue
        if not isinstance(weights, list) or len(weights) != n_vars:
            rewards.append(0.0)
            continue
        if not isinstance(interactions, dict):
            rewards.append(0.0)
            continue

        # Reconstruct adjacency list for neighbors (prevents downstream solvers crashing)
        adj = {i: [] for i in range(n_vars)}
        for k in interactions:
            try:
                u, v = map(int, k.split(','))
                if u in adj and u < n_vars:
                    adj[u].append(v)
                if v in adj and v < n_vars:
                    adj[v].append(u)
            except (ValueError, IndexError):
                pass

        test_requirements = {
            **mission,
            "weights": weights,
            "interactions": interactions,
        }
        test_catalog = {
            "variables": [
                {
                    "id": j,
                    "weight": weights[j] if j < len(weights) else 1.0,
                    "neighbors": adj.get(j, []),
                }
                for j in range(n_vars)
            ],
            "interactions": interactions,
        }
        stdin_obj = {"requirements": test_requirements, "catalog": test_catalog}

        # 3. Execution & Evaluation
        try:
            # Run code (Timeout 5s)
            result = run_candidate(code, stdin_obj, timeout=5.0)
            
            # --- THE MINIMALIST LOGIC ---
            
            # Level 1: Syntax Error / Crash
            if "error" in result:
                rewards.append(0.0)
                continue
                
            # Level 2: Wrong Schema (Didn't print {"selection": ...})
            if "selection" not in result:
                rewards.append(0.1)  # Minimal signal: valid Python syntax but incorrect output format
                continue
                
            # Level 3: Feasibility Check
            selection_raw = result.get("selection", {}).get("variables", [])
            # Normalize selection to List[int] for the checker (models sometimes emit strings or dicts)
            selection: list[int] = []
            if isinstance(selection_raw, list):
                for item in selection_raw:
                    if isinstance(item, int):
                        selection.append(item)
                    elif isinstance(item, str):
                        try:
                            selection.append(int(item))
                        except ValueError:
                            continue
                    elif isinstance(item, dict) and "id" in item:
                        try:
                            selection.append(int(item["id"]))
                        except (TypeError, ValueError):
                            continue
            violations = _quick_check_sds_constraints(mission, selection)
            
            if violations == 0:
                rewards.append(1.0)  # It works.
            else:
                rewards.append(0.5)  # It ran, but broke rules.
                
        except Exception:
            rewards.append(0.0)
            
    return rewards


def reward_code_diversity_fast(completions, **kwargs) -> list[float]:  # noqa: ARG001
    """
    Penalizes Low Entropy. Forces the model to try different logical approaches
    within the same batch (Group Size). Uses 4-gram Jaccard similarity.
    """
    rewards = []
    # Extract code blocks
    codes = [extract_block(c.get("content", "") if isinstance(c, dict) else str(c), "code") or "" for c in completions]
    
    # Helper: Tokenize into 4-gram shingles
    def get_shingles(text, n=4):
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < n:
            return set(words)
        return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}

    shingle_sets = [get_shingles(code) for code in codes]
    
    for i, my_set in enumerate(shingle_sets):
        if not my_set:
            rewards.append(0.0)
            continue
            
        similarities = []
        for j, peer_set in enumerate(shingle_sets):
            if i == j or not peer_set:
                continue
            
            # Jaccard Similarity
            intersection = len(my_set & peer_set)
            union = len(my_set | peer_set)
            similarities.append(intersection / union if union > 0 else 0.0)
        
        # Average similarity to peers
        avg_sim = sum(similarities) / max(1, len(similarities))
        
        # Reward: +0.5 if unique, -0.5 if identical to batch
        rewards.append(0.5 - avg_sim)

    return rewards


def reward_iterative_structure(completions, **kwargs) -> list[float]:  # noqa: ARG001, PLR0912
    """
    Rewards structural complexity that implies 'Search' rather than 'Greedy'.
    Detects: While loops, Nested For loops, Randomness using AST.
    """
    rewards = []
    for comp in completions:
        # Handle both list format [{"content": "..."}] and direct format {"content": "..."}
        if isinstance(comp, list) and len(comp) > 0:
            text = comp[0].get("content", "")
        elif isinstance(comp, dict):
            text = comp.get("content", "")
        else:
            text = str(comp)
        code = extract_block(text, "code")
        if not code:
            rewards.append(0.0)
            continue
            
        try:
            tree = ast.parse(code)
            score = 0.0
            has_while = False
            has_nested_for = False
            has_random = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.While):
                    has_while = True
                if isinstance(node, ast.For):
                    for child in ast.walk(node):
                        if isinstance(child, ast.For) and child is not node:
                            has_nested_for = True
                # Check for random module usage: random.choice(), random.shuffle(), etc.
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == 'random'
                    and node.attr in ['choice', 'shuffle', 'random', 'randint', 'sample', 'uniform', 'seed', 'randrange']
                ):
                    has_random = True
            
            # Grading the "Algorithmic Complexity"
            if has_while:
                score += 0.4
            if has_nested_for:
                score += 0.3
            if has_random:
                score += 0.3
            
            rewards.append(min(1.0, score))
        except Exception:
            rewards.append(0.0)
            
    return rewards


# Export the main reward functions
__all__ = [
    'reward_code_diversity_fast',  # Discovery: Forces batch diversity
    'reward_iterative_structure',  # Discovery: Detects search algorithms
    'unified_code_execution_reward',
    'unified_code_execution_reward_no_oracle',  # ABLATION: No oracle anchoring
    'unified_format_reward',
    'unified_generalization_reward',
    'unified_nominal_reward',  # ABLATION: No generalization testing
    'unified_soft_nominal_reward',
    'unified_nominal_reward_topk_interaction_bound',
]
