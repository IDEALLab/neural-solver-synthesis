"""Tests for BigCode aggregation functionality."""

import json

import pytest

HAS_BIGCODE = False
IMPORT_ERROR = None
try:
    from evaluation.bigcode.aggregate_results import (
        infer_method_from_model_path,
        parse_path_metadata,
    )

    HAS_BIGCODE = True
except ImportError as e:
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(
    not HAS_BIGCODE, reason=f"Could not import aggregate_results: {IMPORT_ERROR}"
)
class TestMethodInference:
    """Test method inference from model paths."""

    def test_infer_hero_from_path(self):
        """Test inferring Hero from checkpoint path."""
        path = "/path/to/checkpoints/qwen2.5-coder-14b/grpo-config_hero/checkpoint-90"
        method = infer_method_from_model_path(path)
        assert method == "Ours (Hero)"

    def test_infer_ablation_from_path(self):
        """Test inferring ablation from checkpoint path."""
        path = "/path/to/checkpoints/qwen2.5-coder-14b/grpo-config_ablation_oracle/checkpoint-90"
        method = infer_method_from_model_path(path)
        assert method == "Ours (+Oracle)"

    def test_infer_minimalist_from_path(self):
        """Test inferring Minimalist from checkpoint path."""
        path = "/path/to/checkpoints/qwen2.5-coder-14b/grpo-config_minimalist/checkpoint-90"
        method = infer_method_from_model_path(path)
        assert method == "Ours (w/o Structure)"

    def test_infer_base_from_path(self):
        """Test inferring Base from path."""
        # Base model is identified by HF identifier format
        path = "Qwen/Qwen2.5-Coder-14B-Instruct"
        method = infer_method_from_model_path(path)
        assert method == "Base"

    def test_infer_none_for_unknown(self):
        """Test that unknown paths return None."""
        path = "/path/to/unknown/config"
        method = infer_method_from_model_path(path)
        assert method is None


@pytest.mark.skipif(
    not HAS_BIGCODE, reason=f"Could not import aggregate_results: {IMPORT_ERROR}"
)
class TestBigCodePathParsing:
    """Test BigCode path parsing."""

    def test_parse_bigcode_path(self):
        """Test parsing BigCode metrics path."""
        path = "evaluation/bigcode/results_batches/20251230_struct-feas-v1/qwen2.5-coder-14b/grpo/seed101/job-1315163/metrics_humaneval.json"
        _method, seed, model, job_id = parse_path_metadata(path)
        assert seed == 101
        assert model == "qwen2.5-coder-14b"
        assert job_id == 1315163

    def test_parse_with_metadata(self, tmp_path):
        """Test parsing with metadata file."""
        json_dir = tmp_path / "job-12345"
        json_dir.mkdir()
        json_path = json_dir / "metrics_humaneval.json"
        json_path.write_text(json.dumps({"pass@1": 0.5}))

        metadata = {
            "method_name": "Ours (Hero)",
            "seed": 999,
            "model": "test-model",
            "job_id": 12345,
        }
        metadata_path = json_dir / "experiment_metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        method, seed, _model, _job_id = parse_path_metadata(str(json_path))
        assert method == "Ours (Hero)"
        assert seed == 999
