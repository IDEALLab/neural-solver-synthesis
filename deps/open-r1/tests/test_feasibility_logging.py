import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from open_r1 import rewards_unified_v2 as rewards


class TestFeasibilityLogging(unittest.TestCase):
    def setUp(self):
        rewards._FEASIBILITY_LOG_CALL_COUNTER = 0

    def test_logs_rank_shard_with_call_index_and_local_group_size(self):
        prompts = [{"role": "user", "content": "prompt-a"}] * 8
        missions = [{"id": "mission-a"}] * 8
        feasible_flags = [1, 0, 1, 0, 1, 0, 1, 0]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "SLURM_JOB_ID": "unit-run",
                    "SLURM_PROCID": "5",
                    "SLURM_LOCALID": "1",
                    "SLURM_NTASKS": "8",
                    "FEASIBILITY_SPARSITY_LOG_DIR": tmpdir,
                },
                clear=False,
            ):
                rewards._log_group_feasibility_stats(
                    prompts=prompts,
                    missions=missions,
                    feasible_flags=feasible_flags,
                    trainer_state=SimpleNamespace(global_step=12),
                )

            path = Path(tmpdir) / "unit-run" / "rank00005.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rank"], 5)
        self.assertEqual(rows[0]["world_size"], 8)
        self.assertEqual(rows[0]["local_group_size"], 8)
        self.assertEqual(rows[0]["feasible_count_in_group"], 4)
        self.assertEqual(rows[0]["reward_call_index"], 0)
        self.assertEqual(rows[0]["trainer_global_step"], 12)

    def test_call_index_increments_between_invocations(self):
        prompts = [{"role": "user", "content": "prompt-a"}] * 2
        missions = [{"id": "mission-a"}] * 2
        feasible_flags = [1, 0]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "SLURM_JOB_ID": "unit-run",
                    "SLURM_PROCID": "0",
                    "SLURM_LOCALID": "0",
                    "SLURM_NTASKS": "8",
                    "FEASIBILITY_SPARSITY_LOG_DIR": tmpdir,
                },
                clear=False,
            ):
                rewards._log_group_feasibility_stats(
                    prompts=prompts,
                    missions=missions,
                    feasible_flags=feasible_flags,
                    trainer_state=None,
                )
                rewards._log_group_feasibility_stats(
                    prompts=prompts,
                    missions=missions,
                    feasible_flags=feasible_flags,
                    trainer_state=None,
                )

            path = Path(tmpdir) / "unit-run" / "rank00000.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["reward_call_index"], 0)
        self.assertEqual(rows[1]["reward_call_index"], 1)
        self.assertIsNone(rows[0]["trainer_global_step"])
        self.assertIsNone(rows[1]["trainer_global_step"])

    def test_generation_traces_include_problem_uuid_and_raw_completion(self):
        prompts = [{"role": "user", "content": "prompt-a"}] * 2
        missions = [{"id": "mission-a"}] * 2
        completions = [
            [{"content": "<think>a</think><code>print(1)</code>"}],
            [{"content": "<think>b</think><code>print(2)</code>"}],
        ]
        feasible_flags = [1, 0]
        reward_values = [0.5, -0.1]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "SLURM_JOB_ID": "unit-run",
                    "SLURM_PROCID": "2",
                    "SLURM_LOCALID": "0",
                    "SLURM_NTASKS": "8",
                    "FEASIBILITY_GENERATION_TRACE_DIR": tmpdir,
                },
                clear=False,
            ):
                rewards._log_generation_traces(
                    completions=completions,
                    prompts=prompts,
                    missions=missions,
                    rewards=reward_values,
                    feasible_flags=feasible_flags,
                    trainer_state=SimpleNamespace(global_step=7),
                    problem_uuids=["sds_tree_000001", "sds_tree_000001"],
                    problem_prompt_hashes=["prompt-hash", "prompt-hash"],
                    problem_mission_hashes=["mission-hash", "mission-hash"],
                )

            path = Path(tmpdir) / "unit-run" / "rank00002.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["problem_uuid"], "sds_tree_000001")
        self.assertEqual(rows[0]["prompt_hash"], "prompt-hash")
        self.assertEqual(rows[0]["mission_hash"], "mission-hash")
        self.assertIn("print(1)", rows[0]["completion_text"])
        self.assertEqual(rows[0]["reward"], 0.5)
        self.assertEqual(rows[0]["exact_feasible"], 1)
        self.assertEqual(rows[0]["sample_ordinal_in_group"], 0)
        self.assertEqual(rows[1]["sample_ordinal_in_group"], 1)

    def test_shared_logging_context_keeps_call_index_aligned(self):
        prompts = [{"role": "user", "content": "prompt-a"}] * 2
        missions = [{"id": "mission-a"}] * 2
        feasible_flags = [1, 0]
        completions = [
            [{"content": "alpha"}],
            [{"content": "beta"}],
        ]
        logging_context = rewards._build_logging_context(SimpleNamespace(global_step=9))

        with tempfile.TemporaryDirectory() as tmpdir:
            feasibility_dir = str(Path(tmpdir) / "feasibility")
            generation_dir = str(Path(tmpdir) / "generation")
            with patch.dict(
                os.environ,
                {
                    "SLURM_JOB_ID": "unit-run",
                    "SLURM_PROCID": "0",
                    "SLURM_LOCALID": "0",
                    "SLURM_NTASKS": "8",
                    "FEASIBILITY_SPARSITY_LOG_DIR": feasibility_dir,
                    "FEASIBILITY_GENERATION_TRACE_DIR": generation_dir,
                },
                clear=False,
            ):
                rewards._log_group_feasibility_stats(
                    prompts=prompts,
                    missions=missions,
                    feasible_flags=feasible_flags,
                    trainer_state=SimpleNamespace(global_step=9),
                    logging_context=logging_context,
                )
                rewards._log_generation_traces(
                    completions=completions,
                    prompts=prompts,
                    missions=missions,
                    rewards=[0.3, 0.0],
                    feasible_flags=feasible_flags,
                    trainer_state=SimpleNamespace(global_step=9),
                    problem_uuids=["u1", "u1"],
                    problem_prompt_hashes=["ph", "ph"],
                    problem_mission_hashes=["mh", "mh"],
                    logging_context=logging_context,
                )

            feasibility_path = Path(feasibility_dir) / "unit-run" / "rank00000.jsonl"
            generation_path = Path(generation_dir) / "unit-run" / "rank00000.jsonl"
            feasibility_rows = [json.loads(line) for line in feasibility_path.read_text().splitlines() if line.strip()]
            generation_rows = [json.loads(line) for line in generation_path.read_text().splitlines() if line.strip()]

        self.assertEqual(feasibility_rows[0]["reward_call_index"], 0)
        self.assertEqual(generation_rows[0]["reward_call_index"], 0)


if __name__ == "__main__":
    unittest.main()
