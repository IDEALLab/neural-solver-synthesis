"""Run the frozen SDS adaptive-repair protocol through OpenAI Responses."""

# ruff: noqa: PLC0415, PLR0912, PLR0913, PLR0915, PLR2004, TRY003

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evaluation.sds.adaptive_repair import (
    Candidate,
    ProtocolError,
    build_budgeted_repair_user_prompt,
    completion_schedule,
    evaluate_candidates,
    git_provenance,
    load_development_references,
    load_frozen_rows,
    load_protocol,
    load_system_prompt,
    make_candidate,
    mission_sha256,
    select_initial_prompt_rows,
    sha256_text,
    write_jsonl,
)


@dataclass(frozen=True)
class ApiCompletion:
    text: str
    response_id: str
    request_id: str
    model: str
    status: str
    incomplete_reason: str | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    latency_seconds: float
    attempts: int


def _read_attr(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def response_to_completion(response: Any, latency: float, attempts: int) -> ApiCompletion:
    usage = _read_attr(response, "usage", {})
    input_details = _read_attr(usage, "input_tokens_details", {})
    output_details = _read_attr(usage, "output_tokens_details", {})
    incomplete = _read_attr(response, "incomplete_details", {})
    return ApiCompletion(
        text=str(_read_attr(response, "output_text", "") or ""),
        response_id=str(_read_attr(response, "id", "")),
        request_id=str(_read_attr(response, "_request_id", "") or ""),
        model=str(_read_attr(response, "model", "")),
        status=str(_read_attr(response, "status", "unknown")),
        incomplete_reason=(
            str(_read_attr(incomplete, "reason"))
            if _read_attr(incomplete, "reason") is not None
            else None
        ),
        input_tokens=int(_read_attr(usage, "input_tokens", 0) or 0),
        cached_input_tokens=int(_read_attr(input_details, "cached_tokens", 0) or 0),
        output_tokens=int(_read_attr(usage, "output_tokens", 0) or 0),
        reasoning_tokens=int(_read_attr(output_details, "reasoning_tokens", 0) or 0),
        latency_seconds=float(latency),
        attempts=int(attempts),
    )


def completion_cost_usd(completion: ApiCompletion, pricing: dict[str, float]) -> float:
    cached = min(completion.input_tokens, completion.cached_input_tokens)
    uncached = completion.input_tokens - cached
    return (
        uncached * float(pricing["input_usd_per_million"])
        + cached * float(pricing["cached_input_usd_per_million"])
        + completion.output_tokens * float(pricing["output_usd_per_million"])
    ) / 1_000_000.0


def theoretical_max_cost_usd(protocol: dict[str, Any], completions: int) -> float:
    limits = protocol["request_limits"]
    pricing = protocol["pricing"]
    return completions * (
        int(limits["max_input_tokens"]) * float(pricing["input_usd_per_million"])
        + int(limits["max_output_tokens"])
        * float(pricing["output_usd_per_million"])
    ) / 1_000_000.0


def openai_rank_key(summary: dict[str, Any]) -> tuple[float, float, str]:
    return (
        -float(summary["pass_rate"]),
        float(summary["mean_penalized_gap"]),
        str(summary.get("code_sha256") or "f" * 64),
    )


def load_openai_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    protocol = json.loads(path.read_text())
    profiles = {
        "N26-E10-v5-openai-adaptive-repair": {
            "reasoning": {"effort": "high"},
            "max_output_tokens": 32_000,
            "prior_cost_keys": (
                "prior_cancelled_v3_observed_journal_cost",
                "prior_cancelled_v4_observed_journal_cost",
            ),
            "combined_cost_key": "combined_cancelled_recorded_plus_v5_theoretical",
        },
        "N26-E10-v6-openai-adaptive-repair-medium-reasoning": {
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 120_000,
            "prior_cost_keys": (
                "prior_cancelled_v3_observed_journal_cost",
                "prior_cancelled_v4_observed_journal_cost",
                "prior_v5_observed_journal_cost",
            ),
            "combined_cost_key": (
                "combined_recorded_v3_v4_v5_plus_v6_theoretical"
            ),
        },
    }
    profile = profiles.get(protocol.get("protocol_id"))
    if profile is None:
        raise ProtocolError("unexpected OpenAI adaptive-repair protocol ID")
    if protocol.get("provider") != "openai_responses":
        raise ProtocolError("E10 requires the OpenAI Responses provider")
    schedule = completion_schedule(protocol)
    if schedule != [8] * 8:
        raise ProtocolError("E10 requires exactly 8 initial plus 7x8 repair completions")
    if protocol.get("seeds") != [101, 202, 303]:
        raise ProtocolError("E10 requires all three predeclared seeds")
    if protocol.get("reasoning") != profile["reasoning"]:
        raise ProtocolError("E10 reasoning effort drifted from its frozen profile")
    if protocol.get("store_responses") is not False:
        raise ProtocolError("E10 responses must not be retained by the API")
    limits = protocol["request_limits"]
    if int(limits["max_input_tokens"]) != 16_000:
        raise ProtocolError("E10 input budget must remain 16,000 tokens")
    if int(limits["max_output_tokens"]) != profile["max_output_tokens"]:
        raise ProtocolError("E10 output/reasoning budget drifted from its frozen profile")
    if int(limits["max_concurrency"]) != 8:
        raise ProtocolError("E10 uses one eight-response batch per round")
    edf_path = Path(protocol["edf"]["repo_path"])
    if sha256_text(edf_path.read_text()) != protocol["edf"]["sha256"]:
        raise ProtocolError("E10 dedicated EDF hash mismatch")

    base_path = Path(protocol["source_e4_protocol"])
    base = load_protocol(base_path)
    if sha256_text(base_path.read_text()) != protocol["source_e4_protocol_sha256"]:
        raise ProtocolError("source E4 protocol hash mismatch")
    manifest_path = Path(protocol["source_e4_manifest"])
    if sha256_text(manifest_path.read_text()) != protocol["source_e4_manifest_sha256"]:
        raise ProtocolError("source E4 manifest hash mismatch")
    expected = {
        "initial_completions": 8,
        "repair_rounds": 7,
        "completions_per_repair_round": 8,
        "total_completion_budget": 64,
        "expected_development_cases": 40,
    }
    for key, value in expected.items():
        if int(base[key]) != value:
            raise ProtocolError(f"source E4 protocol changed {key}")

    per_seed_max = theoretical_max_cost_usd(protocol, sum(schedule))
    recorded_per_seed = float(
        protocol["cost_caps_usd"]["theoretical_per_seed_at_declared_token_limits"]
    )
    if abs(per_seed_max - recorded_per_seed) > 1e-9:
        raise ProtocolError("recorded E10 per-seed theoretical cost drift")
    if per_seed_max > float(protocol["cost_caps_usd"]["per_seed"]):
        raise ProtocolError("theoretical per-seed API cost exceeds the hard cap")
    campaign_max = per_seed_max * len(protocol["seeds"])
    recorded_campaign = float(
        protocol["cost_caps_usd"]["theoretical_campaign_at_declared_token_limits"]
    )
    if abs(campaign_max - recorded_campaign) > 1e-9:
        raise ProtocolError("recorded E10 campaign theoretical cost drift")
    if campaign_max > float(protocol["cost_caps_usd"]["campaign"]):
        raise ProtocolError("theoretical campaign API cost exceeds the hard cap")
    prior_cost = sum(
        float(protocol["cost_caps_usd"][key])
        for key in profile["prior_cost_keys"]
    )
    combined = float(protocol["cost_caps_usd"][profile["combined_cost_key"]])
    overall_limit = float(protocol["cost_caps_usd"]["overall_campaign_limit"])
    if (
        abs(prior_cost + campaign_max - combined) > 1e-9
        or combined > overall_limit
    ):
        raise ProtocolError("combined E10 superseded-run cost accounting drift")
    return protocol, base, manifest_path


def make_message_counter(model: str) -> Callable[[str, str], int]:
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    def count(system_prompt: str, user_prompt: str) -> int:
        # The fixed allowance conservatively covers native Responses message framing.
        return len(encoding.encode(system_prompt)) + len(encoding.encode(user_prompt)) + 16

    return count


def request_one(
    client: Any,
    protocol: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> ApiCompletion:
    retry = protocol["retry"]
    attempts = int(retry["max_attempts"])
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            response = client.responses.create(
                model=protocol["model"],
                instructions=system_prompt,
                input=user_prompt,
                reasoning=protocol["reasoning"],
                max_output_tokens=int(protocol["request_limits"]["max_output_tokens"]),
                store=False,
            )
            return response_to_completion(
                response, time.monotonic() - started, attempt
            )
        except Exception as exc:  # The SDK exposes several retryable subclasses.
            last_error = exc
            if attempt == attempts:
                break
            delay = min(
                float(retry["max_backoff_seconds"]),
                float(retry["initial_backoff_seconds"]) * (2 ** (attempt - 1)),
            )
            time.sleep(delay)
    raise ProtocolError(f"OpenAI response failed after {attempts} attempts") from last_error


def request_batch(
    client: Any,
    protocol: dict[str, Any],
    system_prompt: str,
    user_prompts: list[str],
    on_completion: Callable[[int, ApiCompletion], None] | None = None,
) -> list[ApiCompletion]:
    if not user_prompts or len(user_prompts) > int(
        protocol["request_limits"]["max_concurrency"]
    ):
        raise ProtocolError("invalid E10 request batch size")
    completed: dict[int, ApiCompletion] = {}
    with ThreadPoolExecutor(max_workers=len(user_prompts)) as executor:
        futures = {
            executor.submit(
                request_one, client, protocol, system_prompt, prompt
            ): index
            for index, prompt in enumerate(user_prompts)
        }
        for future in as_completed(futures):
            index = futures[future]
            completed[index] = future.result()
            if on_completion is not None:
                on_completion(index, completed[index])
    if set(completed) != set(range(len(user_prompts))):
        raise ProtocolError("OpenAI response batch is incomplete")
    return [completed[index] for index in range(len(user_prompts))]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    temporary.replace(path)


def generation_record(
    *,
    completion_index: int,
    round_index: int,
    parent_code_sha256: str | None,
    prompt: str,
    prompt_tokens: int,
    exact_requirement_uuids: list[str],
    completion: ApiCompletion,
    pricing: dict[str, float],
) -> dict[str, Any]:
    candidate = make_candidate(
        completion_index,
        round_index,
        completion.text,
        parent_code_sha256,
    )
    return {
        "completion_index": completion_index,
        "round_index": round_index,
        "parent_code_sha256": parent_code_sha256,
        "prompt_sha256": sha256_text(prompt),
        "prompt_token_count_local": prompt_tokens,
        "exact_requirement_uuids": exact_requirement_uuids,
        "exact_requirement_count": len(exact_requirement_uuids),
        "prompt": prompt,
        "generated_text": completion.text,
        "code_sha256": candidate.code_sha256,
        "api": asdict(completion),
        "cost_usd": completion_cost_usd(completion, pricing),
    }


def request_or_resume_batch(
    *,
    client: Any,
    protocol: dict[str, Any],
    system_prompt: str,
    completion_indexes: list[int],
    round_index: int,
    parent_code_sha256: str | None,
    user_prompts: list[str],
    prompt_token_counts: list[int],
    exact_requirement_uuids: list[str],
    journal_dir: Path,
) -> tuple[list[ApiCompletion], list[dict[str, Any]]]:
    if not (
        len(completion_indexes) == len(user_prompts) == len(prompt_token_counts)
    ):
        raise ProtocolError("journal batch inputs have inconsistent lengths")
    journal_dir.mkdir(parents=True, exist_ok=True)
    records: dict[int, dict[str, Any]] = {}
    missing_positions: list[int] = []
    for position, completion_index in enumerate(completion_indexes):
        path = journal_dir / f"slot-{completion_index:03d}.json"
        if not path.exists():
            missing_positions.append(position)
            continue
        record = json.loads(path.read_text())
        expected = {
            "completion_index": completion_index,
            "round_index": round_index,
            "parent_code_sha256": parent_code_sha256,
            "prompt_sha256": sha256_text(user_prompts[position]),
            "prompt_token_count_local": prompt_token_counts[position],
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ProtocolError(f"paid-response journal mismatch for {key}")
        completion = ApiCompletion(**record["api"])
        candidate = make_candidate(
            completion_index, round_index, completion.text, parent_code_sha256
        )
        if candidate.code_sha256 != record.get("code_sha256"):
            raise ProtocolError("paid-response journal code hash mismatch")
        records[position] = record

    if missing_positions:
        prompts = [user_prompts[position] for position in missing_positions]

        def persist(local_index: int, completion: ApiCompletion) -> None:
            position = missing_positions[local_index]
            completion_index = completion_indexes[position]
            record = generation_record(
                completion_index=completion_index,
                round_index=round_index,
                parent_code_sha256=parent_code_sha256,
                prompt=user_prompts[position],
                prompt_tokens=prompt_token_counts[position],
                exact_requirement_uuids=exact_requirement_uuids,
                completion=completion,
                pricing=protocol["pricing"],
            )
            atomic_write_json(
                journal_dir / f"slot-{completion_index:03d}.json", record
            )
            records[position] = record

        request_batch(
            client,
            protocol,
            system_prompt,
            prompts,
            on_completion=persist,
        )
    if set(records) != set(range(len(completion_indexes))):
        raise ProtocolError("paid-response journal batch is incomplete")
    ordered_records = [records[position] for position in range(len(records))]
    return (
        [ApiCompletion(**record["api"]) for record in ordered_records],
        ordered_records,
    )


def archived_repair_prompt(
    *,
    journal_dir: Path,
    completion_indexes: list[int],
    round_index: int,
    computed_parent_code_sha256: str | None,
    computed_prompt: str,
    computed_prompt_tokens: int,
    computed_exact_requirement_uuids: list[str],
) -> tuple[str, int, list[str], str | None]:
    """Recover the exact paid prompt and parent for a partially saved round."""
    paths = [
        journal_dir / f"slot-{completion_index:03d}.json"
        for completion_index in completion_indexes
    ]
    records = [json.loads(path.read_text()) for path in paths if path.exists()]
    if not records:
        return (
            computed_prompt,
            computed_prompt_tokens,
            computed_exact_requirement_uuids,
            computed_parent_code_sha256,
        )
    for record in records:
        if record.get("round_index") != round_index:
            raise ProtocolError("archived repair round mismatch for round_index")
        prompt = str(record.get("prompt", ""))
        if not prompt or sha256_text(prompt) != record.get("prompt_sha256"):
            raise ProtocolError("archived repair prompt hash mismatch")
    prompts = {str(record["prompt"]) for record in records}
    token_counts = {int(record["prompt_token_count_local"]) for record in records}
    exact_uuid_sets = {
        tuple(str(uuid) for uuid in record.get("exact_requirement_uuids", []))
        for record in records
    }
    parent_hashes = {record.get("parent_code_sha256") for record in records}
    if (
        len(prompts) != 1
        or len(token_counts) != 1
        or len(exact_uuid_sets) != 1
        or len(parent_hashes) != 1
    ):
        raise ProtocolError("archived repair prompts disagree within one round")
    return (
        prompts.pop(),
        token_counts.pop(),
        list(exact_uuid_sets.pop()),
        parent_hashes.pop(),
    )


def prepare_paid_run_state(
    *,
    output_dir: Path,
    protocol_id: str,
    protocol_sha256: str,
    seed: int,
    current_source_commit: str,
    resume_source_commit: str | None,
) -> tuple[Path, str]:
    """Create a run state or validate an explicit cross-commit continuation."""
    state_path = output_dir / "run_state.json"
    state_common = {
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "seed": seed,
        "status": "incomplete",
    }
    if output_dir.exists():
        if (output_dir / "run_manifest.json").exists():
            raise FileExistsError(f"completed immutable output exists: {output_dir}")
        if not state_path.exists():
            raise ProtocolError("incomplete paid-response run state is missing")
        existing_state = json.loads(state_path.read_text())
        initial_source_commit = str(existing_state.get("source_commit", ""))
        expected_state = {**state_common, "source_commit": initial_source_commit}
        if existing_state != expected_state:
            raise ProtocolError("incomplete paid-response run state does not match")
        if current_source_commit != initial_source_commit:
            if resume_source_commit != initial_source_commit:
                raise ProtocolError(
                    "cross-commit resume requires the exact initial source commit"
                )
        elif resume_source_commit not in (None, initial_source_commit):
            raise ProtocolError("resume source override does not match current source")
        return state_path, initial_source_commit

    if resume_source_commit is not None:
        raise ProtocolError("resume source override is invalid for a fresh run")
    output_dir.mkdir(parents=True)
    atomic_write_json(
        state_path, {**state_common, "source_commit": current_source_commit}
    )
    return state_path, current_source_commit


def run_openai_repair(
    protocol_path: Path,
    development_references_path: Path,
    seed: int,
    output_dir: Path,
) -> None:
    from openai import OpenAI

    started = time.time()
    protocol, base, manifest_path = load_openai_protocol(protocol_path)
    if seed not in protocol["seeds"]:
        raise ProtocolError(f"undeclared seed: {seed}")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ProtocolError("OPENAI_API_KEY is not set")
    for package, expected in (
        ("openai", protocol["openai_sdk_version"]),
        ("tiktoken", protocol["tiktoken_version"]),
    ):
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ProtocolError(f"{package} version {actual} != frozen {expected}")
    provenance = git_provenance()
    if provenance["dirty"]:
        raise ProtocolError("source checkout is dirty")

    protocol_sha256 = sha256_text(protocol_path.read_text())
    state_path, initial_source_commit = prepare_paid_run_state(
        output_dir=output_dir,
        protocol_id=protocol["protocol_id"],
        protocol_sha256=protocol_sha256,
        seed=seed,
        current_source_commit=provenance["top_level_commit"],
        resume_source_commit=os.environ.get("N26_E10_RESUME_FROM_SOURCE_COMMIT"),
    )

    manifest = json.loads(manifest_path.read_text())
    reference_spec = protocol["development_references"][str(seed)]
    if str(development_references_path) != reference_spec["path"]:
        raise ProtocolError("development-reference path is not the frozen E4 path")
    if sha256_text(development_references_path.read_text()) != reference_spec["sha256"]:
        raise ProtocolError("development-reference hash mismatch")
    rows = load_frozen_rows(base, manifest, seed)
    references_raw, references = load_development_references(
        development_references_path, rows, base
    )
    statuses = {str(row["uuid"]): str(row["status"]) for row in references_raw}
    system_prompt = load_system_prompt(Path(base["system_prompt_config"]))
    count_messages = make_message_counter(protocol["model"])
    schedule = completion_schedule(protocol)
    client = OpenAI(max_retries=0, timeout=float(protocol["request_timeout_seconds"]))

    journal_dir = output_dir / "response_journal"
    candidates: list[Candidate] = []
    summaries: list[dict[str, Any]] = []
    evaluations: dict[int, list[dict[str, Any]]] = {}
    generation_records: list[dict[str, Any]] = []
    api_completions: list[ApiCompletion] = []
    completion_index = 0

    initial_rows = select_initial_prompt_rows(rows, schedule[0])
    initial_prompts = [str(row["problem"]) for row in initial_rows]
    initial_counts = [count_messages(system_prompt, prompt) for prompt in initial_prompts]
    limit = int(protocol["request_limits"]["max_input_tokens"])
    if any(count > limit for count in initial_counts):
        raise ProtocolError("an initial E10 prompt exceeds the input-token limit")
    initial_outputs, initial_records = request_or_resume_batch(
        client=client,
        protocol=protocol,
        system_prompt=system_prompt,
        completion_indexes=list(range(schedule[0])),
        round_index=0,
        parent_code_sha256=None,
        user_prompts=initial_prompts,
        prompt_token_counts=initial_counts,
        exact_requirement_uuids=[],
        journal_dir=journal_dir,
    )
    round_candidates = []
    for output, record in zip(initial_outputs, initial_records, strict=True):
        candidate = make_candidate(completion_index, 0, output.text, None)
        round_candidates.append(candidate)
        api_completions.append(output)
        generation_records.append(record)
        completion_index += 1
    round_summaries, round_evaluations = evaluate_candidates(
        round_candidates, rows, references, base, statuses
    )
    candidates.extend(round_candidates)
    summaries.extend(round_summaries)
    evaluations.update(round_evaluations)

    for round_index, round_count in enumerate(schedule[1:], start=1):
        incumbent_summary = min(summaries, key=openai_rank_key)
        incumbent = next(
            candidate
            for candidate in candidates
            if candidate.completion_index == incumbent_summary["completion_index"]
        )
        user_prompt, prompt_tokens, exact_uuids = build_budgeted_repair_user_prompt(
            system_prompt,
            incumbent.code or "",
            incumbent_summary,
            evaluations[incumbent.completion_index],
            rows,
            base,
            round_index,
            limit,
            count_messages,
        )
        indexes = list(range(completion_index, completion_index + round_count))
        user_prompt, prompt_tokens, exact_uuids, parent_code_sha256 = (
            archived_repair_prompt(
                journal_dir=journal_dir,
                completion_indexes=indexes,
                round_index=round_index,
                computed_parent_code_sha256=incumbent.code_sha256,
                computed_prompt=user_prompt,
                computed_prompt_tokens=prompt_tokens,
                computed_exact_requirement_uuids=exact_uuids,
            )
        )
        known_code_hashes = {candidate.code_sha256 for candidate in candidates}
        if parent_code_sha256 not in known_code_hashes:
            raise ProtocolError("archived repair parent is not a prior candidate")
        outputs, records = request_or_resume_batch(
            client=client,
            protocol=protocol,
            system_prompt=system_prompt,
            completion_indexes=indexes,
            round_index=round_index,
            parent_code_sha256=parent_code_sha256,
            user_prompts=[user_prompt] * round_count,
            prompt_token_counts=[prompt_tokens] * round_count,
            exact_requirement_uuids=exact_uuids,
            journal_dir=journal_dir,
        )
        round_candidates = []
        for output, record in zip(outputs, records, strict=True):
            candidate = make_candidate(
                completion_index,
                round_index,
                output.text,
                parent_code_sha256,
            )
            round_candidates.append(candidate)
            api_completions.append(output)
            generation_records.append(record)
            completion_index += 1
        round_summaries, round_evaluations = evaluate_candidates(
            round_candidates, rows, references, base, statuses
        )
        candidates.extend(round_candidates)
        summaries.extend(round_summaries)
        evaluations.update(round_evaluations)

    if completion_index != int(protocol["total_completion_budget"]):
        raise ProtocolError(f"produced {completion_index} completions")
    selected_summary = min(summaries, key=openai_rank_key)
    selected = next(
        candidate
        for candidate in candidates
        if candidate.completion_index == selected_summary["completion_index"]
    )
    if selected.code is None:
        raise ProtocolError("selected E10 candidate has no code")

    observed_cost = sum(
        completion_cost_usd(completion, protocol["pricing"])
        for completion in api_completions
    )
    if observed_cost > float(protocol["cost_caps_usd"]["per_seed"]):
        raise ProtocolError("observed per-seed API cost exceeded the hard cap")
    models = sorted({completion.model for completion in api_completions})
    if models != [protocol["model"]]:
        raise ProtocolError(f"API returned unexpected model identifiers: {models}")
    response_ids = [completion.response_id for completion in api_completions]
    if not all(response_ids) or len(set(response_ids)) != completion_index:
        raise ProtocolError("paid Responses IDs are missing or repeated")

    (output_dir / "selected_solver.py").write_text(selected.code)
    write_jsonl(output_dir / "generations.jsonl", generation_records)
    write_jsonl(
        output_dir / "candidate_summaries.jsonl",
        sorted(summaries, key=lambda row: row["completion_index"]),
    )
    write_jsonl(
        output_dir / "development_evaluations.jsonl",
        [row for index in sorted(evaluations) for row in evaluations[index]],
    )
    manifest_out = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "source_e4_protocol_sha256": protocol["source_e4_protocol_sha256"],
        "source_e4_manifest_sha256": protocol["source_e4_manifest_sha256"],
        "development_references_sha256": sha256_text(
            development_references_path.read_text()
        ),
        "seed": seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": protocol["model"],
        "openai_sdk_version": importlib.metadata.version("openai"),
        "tiktoken_version": importlib.metadata.version("tiktoken"),
        "system_prompt_sha256": sha256_text(system_prompt),
        "completion_count": completion_index,
        "response_ids": response_ids,
        "status_counts": {
            status: sum(completion.status == status for completion in api_completions)
            for status in sorted({completion.status for completion in api_completions})
        },
        "input_tokens": sum(completion.input_tokens for completion in api_completions),
        "cached_input_tokens": sum(
            completion.cached_input_tokens for completion in api_completions
        ),
        "output_tokens": sum(completion.output_tokens for completion in api_completions),
        "reasoning_tokens": sum(
            completion.reasoning_tokens for completion in api_completions
        ),
        "observed_api_cost_usd": observed_cost,
        "theoretical_max_api_cost_usd": theoretical_max_cost_usd(
            protocol, completion_index
        ),
        "development_execution_count": completion_index * len(rows),
        "development_uuids": [str(row["uuid"]) for row in rows],
        "development_mission_sha256": {
            str(row["uuid"]): mission_sha256(row["mission"]) for row in rows
        },
        "selected_completion_index": selected.completion_index,
        "selected_code_sha256": selected.code_sha256,
        "selected_summary": selected_summary,
        "test_outcomes_opened": False,
        "synthesis_wall_clock_seconds": time.time() - started,
        "paid_response_initial_source_commit": initial_source_commit,
        "resume_execution_source_commit": provenance["top_level_commit"],
        "provenance": provenance,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest_out, indent=2) + "\n"
    )
    state_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--development-references", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[101, 202, 303], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_openai_repair(
        args.protocol, args.development_references, args.seed, args.output_dir
    )


if __name__ == "__main__":
    main()
