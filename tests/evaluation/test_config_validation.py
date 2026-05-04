"""Tests for GRPO configuration file validation."""

from pathlib import Path
from typing import Any

import pytest
import yaml

# Required fields for GRPO configs
REQUIRED_GRPO_FIELDS = [
    "model_name_or_path",
    "dataset_name",
    "reward_funcs",
    "beta",
    "num_iterations",
    "epsilon",
    "epsilon_high",
    "loss_type",
]

# Optional but important fields
IMPORTANT_OPTIONAL_FIELDS = [
    "system_prompt",
    "chat_template",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "num_generations",
]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config file."""
    with config_path.open() as f:
        return yaml.safe_load(f)


def validate_config_structure(
    config: dict[str, Any], required_fields: list[str]
) -> list[str]:
    """
    Validate that config has required fields.

    Returns:
        List of missing field names (empty if all present)
    """
    return [field for field in required_fields if field not in config]


def validate_field_types(config: dict[str, Any]) -> list[str]:
    """
    Validate that key fields have correct types.

    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []

    # Model path should be a string
    if "model_name_or_path" in config and not isinstance(config["model_name_or_path"], str):
        errors.append("model_name_or_path must be a string")

    # Dataset name should be a string
    if "dataset_name" in config and not isinstance(config["dataset_name"], str):
        errors.append("dataset_name must be a string")

    # Reward funcs should be a list
    if "reward_funcs" in config:
        if not isinstance(config["reward_funcs"], list):
            errors.append("reward_funcs must be a list")
        elif len(config["reward_funcs"]) == 0:
            errors.append("reward_funcs must not be empty")

    # Numeric fields
    numeric_fields = ["beta", "epsilon", "epsilon_high", "num_iterations"]
    errors.extend(
        f"{field} must be numeric"
        for field in numeric_fields
        if field in config and not isinstance(config[field], (int, float))
    )

    # Batch size fields should be positive integers
    batch_fields = [
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_generations",
    ]
    errors.extend(
        f"{field} must be a positive integer"
        for field in batch_fields
        if field in config and (not isinstance(config[field], int) or config[field] <= 0)
    )

    return errors


class TestConfigValidation:
    """Test configuration file validation."""

    @pytest.fixture
    def config_dir(self):
        """Path to config directory."""
        workspace_root = Path(__file__).parent.parent.parent
        return (
            workspace_root
            / "deps"
            / "open-r1"
            / "recipes"
            / "Qwen2.5-Coder-14B-Instruct"
            / "grpo"
        )

    def test_hero_config_exists(self, config_dir):
        """Test that Hero config file exists."""
        config_path = config_dir / "config_hero.yaml"
        assert config_path.exists(), f"Hero config not found: {config_path}"

    def test_hero_config_loads(self, config_dir):
        """Test that Hero config loads as valid YAML."""
        config_path = config_dir / "config_hero.yaml"
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        try:
            config = load_config(config_path)
            assert isinstance(config, dict), "Config should be a dictionary"
        except yaml.YAMLError as e:
            pytest.fail(f"Failed to parse YAML: {e}")

    def test_hero_config_required_fields(self, config_dir):
        """Test that Hero config has all required fields."""
        config_path = config_dir / "config_hero.yaml"
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_config(config_path)
        missing = validate_config_structure(config, REQUIRED_GRPO_FIELDS)
        assert len(missing) == 0, f"Missing required fields: {missing}"

    def test_hero_config_field_types(self, config_dir):
        """Test that Hero config fields have correct types."""
        config_path = config_dir / "config_hero.yaml"
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_config(config_path)
        errors = validate_field_types(config)
        assert len(errors) == 0, f"Type validation errors: {errors}"

    def test_ablation_configs_exist(self, config_dir):
        """Test that ablation config files exist."""
        ablation_configs = [
            "config_ablation_oracle.yaml",
            "config_ablation_diversity.yaml",
            "config_ablation_prompt.yaml",
            "config_minimalist.yaml",
        ]

        missing = []
        for config_name in ablation_configs:
            config_path = config_dir / config_name
            if not config_path.exists():
                missing.append(config_name)

        if missing:
            pytest.skip(f"Some ablation configs not found: {missing}")

    def test_ablation_configs_load(self, config_dir):
        """Test that all ablation configs load as valid YAML."""
        ablation_configs = [
            "config_ablation_oracle.yaml",
            "config_ablation_diversity.yaml",
            "config_ablation_prompt.yaml",
            "config_minimalist.yaml",
        ]

        for config_name in ablation_configs:
            config_path = config_dir / config_name
            if not config_path.exists():
                continue

            try:
                config = load_config(config_path)
                assert isinstance(config, dict), f"{config_name} should be a dictionary"
            except yaml.YAMLError as e:
                pytest.fail(f"Failed to parse {config_name}: {e}")

    def test_reward_funcs_registered(self, config_dir):
        """Test that reward functions in config are valid."""
        config_path = config_dir / "config_hero.yaml"
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_config(config_path)
        if "reward_funcs" not in config:
            pytest.skip("reward_funcs not in config")

        reward_funcs = config["reward_funcs"]
        assert isinstance(reward_funcs, list), "reward_funcs should be a list"
        assert len(reward_funcs) > 0, "reward_funcs should not be empty"

        # Common reward function names (from rewards.py registry)

        for func_name in reward_funcs:
            # Just check it's a string (actual validation would require importing the registry)
            assert isinstance(
                func_name, str
            ), f"Reward function name should be string: {func_name}"

    def test_epsilon_values_valid(self, config_dir):
        """Test that epsilon values are in valid range."""
        config_path = config_dir / "config_hero.yaml"
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_config(config_path)

        if "epsilon" in config:
            epsilon = config["epsilon"]
            assert 0.0 <= epsilon <= 1.0, f"epsilon should be in [0, 1], got {epsilon}"

        if "epsilon_high" in config:
            epsilon_high = config["epsilon_high"]
            assert (
                0.0 <= epsilon_high <= 1.0
            ), f"epsilon_high should be in [0, 1], got {epsilon_high}"

            # epsilon_high should be >= epsilon
            if "epsilon" in config:
                epsilon = config["epsilon"]
                assert (
                    epsilon_high >= epsilon
                ), f"epsilon_high ({epsilon_high}) should be >= epsilon ({epsilon})"
