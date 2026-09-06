"""Fetch the frozen TSPLIB catalog over HTTPS when legacy HTTP is blocked."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from evaluation.text_suite.tsp_compile_once import (
    TSPLIB_NAMES,
    load_protocol,
    parse_tsplib,
    sha256_bytes,
    sha256_file,
    sha256_text,
    validate_global_freeze,
)

OFFICIAL_HOST = "comopt.ifi.uni-heidelberg.de"
OFFICIAL_PATH_PREFIX = "/software/TSPLIB95/tsp/"
OFFICIAL_ARCHIVE_URL = (
    "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz"
)
INVALID_URL_ERROR = "HTTPS fallback accepts only frozen Heidelberg TSPLIB URLs"
TERMS_ERROR = "explicit --accept-external-terms is required"
MISSING_INSTANCE_ERROR = "official TSPLIB archive lacks a frozen instance"
NON_FILE_MEMBER_ERROR = "frozen TSPLIB archive member is not a regular file"
UNREADABLE_MEMBER_ERROR = "frozen TSPLIB archive member cannot be read"


def validate_declared_url(name: str, declared_url: str) -> None:
    """Require the exact frozen Heidelberg per-instance URL shape."""
    expected = f"http://{OFFICIAL_HOST}{OFFICIAL_PATH_PREFIX}{name}.tsp.gz"
    if declared_url != expected:
        raise ValueError(INVALID_URL_ERROR)


def fetch_https(
    protocol_path: Path,
    global_freeze_path: Path,
    output_root: Path,
    source_commit: str,
    accept_external_terms: bool,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path, require_launch_ready=True)
    freeze = json.loads(global_freeze_path.read_text())
    validate_global_freeze(protocol, freeze)
    if not accept_external_terms:
        raise ValueError(TERMS_ERROR)
    output_root.mkdir(parents=True, exist_ok=False)
    archive_bytes = urllib.request.urlopen(
        OFFICIAL_ARCHIVE_URL, timeout=120
    ).read()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        required = {f"{name}.tsp.gz" for name in TSPLIB_NAMES}
        if not required.issubset(members):
            raise ValueError(MISSING_INSTANCE_ERROR)
        compressed_by_name = {}
        for name in TSPLIB_NAMES:
            member = members[f"{name}.tsp.gz"]
            if not member.isfile():
                raise ValueError(NON_FILE_MEMBER_ERROR)
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(UNREADABLE_MEMBER_ERROR)
            compressed_by_name[name] = handle.read()
    rows = []
    for name in TSPLIB_NAMES:
        declared_url = protocol["benchmark"]["urls"][name]
        validate_declared_url(name, declared_url)
        compressed = compressed_by_name[name]
        raw = (
            gzip.decompress(compressed)
            if declared_url.endswith(".gz")
            else compressed
        )
        path = output_root / f"{name}.tsp"
        path.write_bytes(raw)
        mission = parse_tsplib(path, name)
        rows.append(
            {
                "name": name,
                "url": declared_url,
                "retrieval_archive_url": OFFICIAL_ARCHIVE_URL,
                "retrieval_member": f"{name}.tsp.gz",
                "transport_fallback": (
                    "legacy HTTP timed out; same official path retrieved over HTTPS"
                ),
                "compressed_sha256": sha256_bytes(compressed),
                "raw_sha256": sha256_bytes(raw),
                "canonical_instance_sha256": sha256_text(
                    json.dumps(mission, sort_keys=True, separators=(",", ":"))
                ),
                "dimension": mission["n_cities"],
                "path": path.name,
            }
        )
    created = time.time()
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "created_unix": created,
        "created_after_global_freeze": created >= float(freeze["created_unix"]),
        "retrieval_source_commit": source_commit,
        "retrieval_archive_url": OFFICIAL_ARCHIVE_URL,
        "retrieval_archive_sha256": sha256_bytes(archive_bytes),
        "upstream": protocol["benchmark"]["upstream"],
        "license_notice": protocol["benchmark"]["license_notice"],
        "redistribution": (
            "raw TSPLIB files remain external on capstor and are never committed "
            "or uploaded"
        ),
        "global_freeze_sha256": sha256_file(global_freeze_path),
        "instances": rows,
    }
    (output_root / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--global-freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--accept-external-terms", action="store_true")
    args = parser.parse_args()
    result = fetch_https(
        args.protocol,
        args.global_freeze,
        args.output_root,
        args.source_commit,
        args.accept_external_terms,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
