"""Integration tests for the full aggregation pipeline."""

import json

import pandas as pd
import pytest

HAS_AGGREGATION = False
IMPORT_ERROR = None
try:
    from evaluation.bigcode.aggregate_results import (
        load_all_data as load_bigcode_data,
    )
    from evaluation.sds.aggregate_plots import (
        load_all_data,
    )

    HAS_AGGREGATION = True
except ImportError as e:
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(
    not HAS_AGGREGATION, reason=f"Could not import aggregation modules: {IMPORT_ERROR}"
)
@pytest.mark.integration
class TestAggregationPipeline:
    """Test the full aggregation pipeline with mock data."""

    def test_sds_aggregation_with_mock_data(self, tmp_path):
        """Test SDS aggregation with mock CSV files."""
        # Create mock metrics files
        base = (
            tmp_path / "results" / "qwen2.5-coder-14b" / "grpo-config_hero" / "seed101"
        )
        base.mkdir(parents=True)

        job_dir = base / "job-100"
        job_dir.mkdir()

        # Create mock CSV with required columns
        csv_data = pd.DataFrame(
            {
                "uuid": ["uuid1", "uuid2"],
                "feasible": [True, True],
                "llm_score": [100.0, 95.0],
                "score_greedy": [90.0, 85.0],
                "vbs_score": [100.0, 95.0],
                "difficulty_class": ["Hard", "Moderate"],
                "gap": [0.0, 0.0],
                "cost": [0.1, 0.1],
                "pass": [1, 1],
            }
        )
        csv_path = job_dir / "metrics_final.csv"
        csv_data.to_csv(csv_path, index=False)

        # Create metadata
        metadata = {
            "method_name": "Ours (Hero)",
            "seed": 101,
            "model": "qwen2.5-coder-14b",
            "job_id": 100,
        }
        metadata_path = job_dir / "experiment_metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        # Test loading
        files = [str(csv_path)]
        df = load_all_data(files, include_baselines=False)

        assert len(df) == 2
        assert "Method" in df.columns
        assert df["Method"].iloc[0] == "Ours (Hero)"

    def test_bigcode_aggregation_with_mock_data(self, tmp_path):
        """Test BigCode aggregation with mock JSON files."""
        base = tmp_path / "results" / "qwen2.5-coder-14b" / "grpo" / "seed101"
        base.mkdir(parents=True)

        job_dir = base / "job-100"
        job_dir.mkdir()

        # Create mock metrics JSON (BigCode format)
        # BigCode metrics have task name as top-level key
        metrics = {
            "humaneval": {"pass@1": 0.5},
            "config": {
                "model": "/path/to/qwen2.5-coder-14b/grpo-config_hero/checkpoint-90"
            },
        }
        json_path = job_dir / "metrics_humaneval.json"
        json_path.write_text(json.dumps(metrics))

        # Create metadata
        metadata = {
            "method_name": "Ours (Hero)",
            "seed": 101,
            "model": "qwen2.5-coder-14b",
            "job_id": 100,
        }
        metadata_path = job_dir / "experiment_metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        # Test loading
        files = [str(json_path)]
        df = load_bigcode_data(files)

        assert len(df) > 0
        assert "Method" in df.columns
        assert "Pass@1" in df.columns
