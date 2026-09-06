"""Common helpers for problem-suite generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    """Write records to a JSONL file."""
    out_path = Path(path)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
