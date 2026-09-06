from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_neurips2026_public_evidence.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_public_evidence", VALIDATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_evidence_package_passes_privacy_and_integrity_gates() -> None:
    validator = load_validator()
    errors = validator.validate(
        REPO_ROOT / "artifacts" / "neurips2026",
        REPO_ROOT / "docs" / "final_evidence_index.json",
    )
    assert errors == []


def test_private_publication_documents_are_excluded_from_release_manifest() -> None:
    import json

    manifest_path = REPO_ROOT / "docs" / "paper_release_export_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        assert "docs/internal" in manifest["exclude_globs"]
        assert "docs/internal/**" in manifest["exclude_globs"]
        assert "docs/openreview_responses" in manifest["exclude_globs"]
        assert "docs/openreview_responses/**" in manifest["exclude_globs"]
        return

    assert not (REPO_ROOT / "docs" / "internal").exists()
    assert not (REPO_ROOT / "docs" / "openreview_responses").exists()
