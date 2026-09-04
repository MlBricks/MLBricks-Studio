from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _git_lines(*args: str) -> list[str]:
    if not (ROOT / ".git").exists():
        pytest.skip("release hygiene checks require a Git checkout")
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def test_no_python_cache_files_are_tracked():
    tracked = _git_lines("ls-files")
    bad = [
        path
        for path in tracked
        if "__pycache__/" in path.replace("\\", "/") or path.endswith(".pyc")
    ]
    assert bad == [], f"tracked Python cache files: {bad}"


def test_gitignore_covers_python_and_build_artifacts():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    required = [
        "__pycache__/",
        "*.py[cod]",
        ".pytest_cache/",
        "build/",
        "dist/",
        "*.egg-info/",
    ]
    missing = [pattern for pattern in required if pattern not in ignore]
    assert missing == [], f".gitignore missing release-hygiene patterns: {missing}"
