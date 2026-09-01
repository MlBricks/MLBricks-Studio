from __future__ import annotations

import importlib.metadata

import pytest


mlbricks = pytest.importorskip("mlbricks", reason="MLBricks integration dependency is not installed in the source-test environment")

from mlb_studio import Builder  # noqa: E402


@pytest.mark.integration
def test_pinned_mlbricks_release_and_core_api_resolution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert importlib.metadata.version("mlbricks") == "1.0.0"
    builder = Builder()
    # Resolve representative public components through the same lazy import pool
    # used by model compilation. This catches packaging/API drift immediately.
    for component in ("embedding", "esa", "ffn", "rmsnorm", "residual", "lm_head"):
        status = builder.ensure_component_import(component)
        assert status.get("ok"), status
