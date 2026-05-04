#!/usr/bin/env python3
"""Export a clean public release tree and an anonymized supplementary bundle."""

from __future__ import annotations

import argparse
import filecmp
import fnmatch
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".slurm",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class ReplacementRule:
    find: str
    replace: str


@dataclass(frozen=True)
class RegexReplacementRule:
    pattern: str
    replacement: str


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def git_repo_root() -> Path:
    result = run(["git", "rev-parse", "--show-toplevel"])
    return Path(result.stdout.strip())


def verify_source_tag(repo_root: Path, manifest: dict) -> str:
    source_tag = manifest["source_tag"]
    result = run(["git", "rev-parse", f"{source_tag}^{{commit}}"], cwd=repo_root)
    return result.stdout.strip()


def extract_tar_bytes(data: bytes, dest_root: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
        tf.extractall(dest_root, filter="data")


def git_ls_tree(repo_root: Path, source_tag: str, rel_path: str) -> tuple[str, str] | None:
    result = run(["git", "ls-tree", source_tag, rel_path], cwd=repo_root)
    line = result.stdout.strip()
    if not line:
        return None
    meta, path = line.split("\t", 1)
    mode, obj_type, sha = meta.split()
    return mode, sha


def archive_repo_path(repo_root: Path, source_tag: str, rel_path: str, dest_root: Path) -> None:
    result = run(
        ["git", "archive", "--format=tar", source_tag, rel_path],
        cwd=repo_root,
        capture_output=True,
    )
    extract_tar_bytes(result.stdout.encode("utf-8", "surrogateescape"), dest_root)


def archive_repo_path_bytes(repo_root: Path, source_tag: str, rel_path: str, dest_root: Path) -> None:
    result = subprocess.run(
        ["git", "archive", "--format=tar", source_tag, rel_path],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
    )
    extract_tar_bytes(result.stdout, dest_root)


def archive_submodule_path(
    repo_root: Path, submodule_path: str, commit_sha: str, dest_root: Path
) -> None:
    submodule_root = repo_root / submodule_path
    result = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            f"--prefix={submodule_path}/",
            commit_sha,
        ],
        cwd=str(submodule_root),
        check=True,
        capture_output=True,
    )
    extract_tar_bytes(result.stdout, dest_root)


def export_included_paths(repo_root: Path, manifest: dict, export_root: Path) -> None:
    source_tag = manifest["source_tag"]
    export_root.mkdir(parents=True, exist_ok=True)
    for rel_path in manifest["include_paths"]:
        tree_info = git_ls_tree(repo_root, source_tag, rel_path)
        if tree_info is None:
            raise FileNotFoundError(f"Path {rel_path} is not present in {source_tag}.")
        mode, sha = tree_info
        if mode == "160000":
            archive_submodule_path(repo_root, rel_path, sha, export_root)
        else:
            archive_repo_path_bytes(repo_root, source_tag, rel_path, export_root)


def iter_rel_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        yield path.relative_to(root)


def remove_matches(root: Path, pattern: str) -> None:
    matches = list(root.glob(pattern))
    for match in sorted(matches, key=lambda p: len(p.parts), reverse=True):
        if not match.exists():
            continue
        if match.is_dir():
            shutil.rmtree(match)
        else:
            match.unlink()


def apply_excludes(root: Path, patterns: list[str]) -> None:
    for pattern in patterns:
        remove_matches(root, pattern)


def matches_any(rel_path: Path, patterns: Iterable[str]) -> bool:
    rel_str = rel_path.as_posix()
    return any(fnmatch.fnmatchcase(rel_str, pattern) for pattern in patterns)


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def rewrite_text_files(
    root: Path,
    globs: list[str],
    string_rules: list[ReplacementRule],
    regex_rules: list[RegexReplacementRule],
) -> None:
    for rel_path in iter_rel_paths(root):
        if not matches_any(rel_path, globs):
            continue
        full_path = root / rel_path
        if not full_path.is_file() or not is_text_file(full_path):
            continue
        try:
            original = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = original
        for rule in string_rules:
            updated = updated.replace(rule.find, rule.replace)
        for rule in regex_rules:
            updated = re.sub(rule.pattern, rule.replacement, updated)
        if updated != original:
            full_path.write_text(updated, encoding="utf-8")


def ensure_required_paths(root: Path, required_paths: list[str]) -> None:
    missing = [path for path in required_paths if not (root / path).exists()]
    if missing:
        missing_str = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(f"Missing required exported paths:\n{missing_str}")


def ensure_forbidden_filenames_absent(root: Path, forbidden_names: list[str]) -> None:
    found = []
    forbidden_set = set(forbidden_names)
    for path in root.rglob("*"):
        if path.name in forbidden_set:
            found.append(path.relative_to(root).as_posix())
    if found:
        found_str = "\n".join(f"  - {path}" for path in found)
        raise RuntimeError(f"Forbidden filenames found:\n{found_str}")


def ensure_no_vcs_metadata(root: Path) -> None:
    offenders = []
    for path in root.rglob(".git"):
        offenders.append(path.relative_to(root).as_posix())
    for path in root.rglob(".gitmodules"):
        offenders.append(path.relative_to(root).as_posix())
    if offenders:
        offenders_str = "\n".join(f"  - {path}" for path in offenders)
        raise RuntimeError(f"Unexpected VCS metadata found:\n{offenders_str}")


def scan_content_for_patterns(root: Path, patterns: list[str], label: str) -> None:
    regexes = [re.compile(pattern) for pattern in patterns]
    matches = []
    for path in root.rglob("*"):
        if not path.is_file() or not is_text_file(path):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for regex in regexes:
            if regex.search(content):
                matches.append(f"{path.relative_to(root).as_posix()}: {regex.pattern}")
    if matches:
        matches_str = "\n".join(f"  - {match}" for match in matches[:50])
        extra = "" if len(matches) <= 50 else f"\n  ... and {len(matches) - 50} more"
        raise RuntimeError(f"{label} validation failed:\n{matches_str}{extra}")


def compare_exports(
    public_root: Path,
    anon_root: Path,
    allowed_diff_globs: list[str],
    allowed_public_only_globs: list[str] | None = None,
) -> None:
    allowed_public_only_globs = allowed_public_only_globs or []
    public_files = {
        path.relative_to(public_root)
        for path in public_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    anon_files = {path.relative_to(anon_root) for path in anon_root.rglob("*") if path.is_file()}
    public_only_paths = public_files - anon_files
    unexpected_public_only = sorted(
        str(path)
        for path in public_only_paths
        if not matches_any(path, allowed_public_only_globs)
    )
    anon_only_paths = anon_files - public_files
    if unexpected_public_only or anon_only_paths:
        anon_only = sorted(str(path) for path in anon_files - public_files)
        raise RuntimeError(
            "Public and anonymized exports diverged in file membership.\n"
            f"Public-only: {unexpected_public_only[:20]}\nAnon-only: {anon_only[:20]}"
        )

    comparable_files = sorted(
        rel_path
        for rel_path in public_files
        if rel_path in anon_files
    )
    for rel_path in comparable_files:
        public_path = public_root / rel_path
        anon_path = anon_root / rel_path
        if filecmp.cmp(public_path, anon_path, shallow=False):
            continue
        if not matches_any(rel_path, allowed_diff_globs):
            raise RuntimeError(
                f"Unexpected content difference outside allowed rewrite surface: {rel_path}"
            )


def init_public_git_repo(public_root: Path, source_commit: str) -> None:
    run(["git", "init", "--initial-branch=main"], cwd=public_root)
    run(["git", "config", "user.name", "Release Export Bot"], cwd=public_root)
    run(
        ["git", "config", "user.email", "release-export@example.invalid"],
        cwd=public_root,
    )
    run(["git", "add", "."], cwd=public_root)
    run(
        ["git", "commit", "-m", f"Paper release export from private baseline {source_commit[:8]}"],
        cwd=public_root,
    )


def ensure_public_repo_ready(public_root: Path) -> None:
    run(["git", "rev-parse", "--is-inside-work-tree"], cwd=public_root)
    status = run(["git", "status", "--short"], cwd=public_root)
    if status.stdout.strip():
        raise RuntimeError("Fresh public release repo is not clean after initialization.")
    log = run(["git", "rev-list", "--count", "HEAD"], cwd=public_root)
    if log.stdout.strip() != "1":
        raise RuntimeError("Fresh public release repo must contain exactly one commit.")


def zip_tree(source_root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(source_root))


def build_output_trees(repo_root: Path, manifest: dict, output_root: Path) -> tuple[Path, Path, Path]:
    public_root = output_root / manifest["public_repo_dirname"]
    anon_root = output_root / manifest["anonymized_dirname"]
    dist_root = output_root / manifest["dist_dirname"]

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    export_included_paths(repo_root, manifest, public_root)
    shutil.copytree(public_root, anon_root)

    apply_excludes(public_root, manifest["exclude_globs"])
    apply_excludes(anon_root, manifest["exclude_globs"])
    apply_excludes(anon_root, manifest.get("anonymized_exclude_globs", []))

    return public_root, anon_root, dist_root


def inject_release_snapshot_note(
    repo_root: Path, public_root: Path, anon_root: Path, source_commit: str
) -> None:
    source_note = repo_root / "docs/NEURIPS_2026_CODE_RELEASE_SNAPSHOT.md"
    rendered = source_note.read_text(encoding="utf-8").replace(
        "__SOURCE_COMMIT__", source_commit
    )
    for root in (public_root, anon_root):
        target = root / "docs/NEURIPS_2026_CODE_RELEASE_SNAPSHOT.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")


def validate_export(root: Path, manifest: dict, *, anonymized: bool) -> None:
    ensure_required_paths(root, manifest["required_paths"])
    ensure_forbidden_filenames_absent(root, manifest["forbidden_filenames"])
    ensure_no_vcs_metadata(root)

    forbidden = list(manifest["forbidden_content_patterns"]["both"])
    if anonymized:
        forbidden.extend(manifest["forbidden_content_patterns"]["anonymized"])
    scan_content_for_patterns(root, forbidden, "Content")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="docs/paper_release_export_manifest.json",
        help="Path to the export manifest JSON.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override the output root declared in the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = git_repo_root()
    manifest_path = (repo_root / args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    output_root = Path(args.output_root or manifest["output_root"]).resolve()

    source_commit = verify_source_tag(repo_root, manifest)

    public_root, anon_root, dist_root = build_output_trees(repo_root, manifest, output_root)
    inject_release_snapshot_note(repo_root, public_root, anon_root, source_commit)

    public_string_rules = [
        ReplacementRule(**rule) for rule in manifest["public_string_replacements"]
    ]
    anon_string_rules = [
        ReplacementRule(**rule) for rule in manifest["anonymized_string_replacements"]
    ]
    anon_regex_rules = [
        RegexReplacementRule(**rule) for rule in manifest["anonymized_regex_replacements"]
    ]

    rewrite_text_files(
        public_root,
        manifest["public_rewrite_globs"],
        public_string_rules,
        [],
    )
    rewrite_text_files(
        anon_root,
        manifest["anonymized_rewrite_globs"],
        public_string_rules + anon_string_rules,
        anon_regex_rules,
    )

    validate_export(public_root, manifest, anonymized=False)
    validate_export(anon_root, manifest, anonymized=True)
    compare_exports(
        public_root,
        anon_root,
        manifest["allowed_content_diff_globs"],
        manifest.get("anonymized_exclude_globs", []),
    )

    init_public_git_repo(public_root, source_commit)
    ensure_public_repo_ready(public_root)

    zip_tree(anon_root, dist_root / manifest["anonymized_zip_name"])

    print("✅ Export complete")
    print(f"   Public release: {public_root}")
    print(f"   Anonymized tree: {anon_root}")
    print(f"   Zip bundle: {dist_root / manifest['anonymized_zip_name']}")


if __name__ == "__main__":
    main()
