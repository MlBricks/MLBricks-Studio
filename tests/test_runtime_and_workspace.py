from __future__ import annotations

import importlib.metadata
from pathlib import Path

from mlb_studio import Builder, __version__
from mlb_studio.graph import new_project
from mlb_studio.runtime import get_mlbricks_info


def test_versions_are_consistent():
    assert __version__ == "1.0.0"
    assert new_project()["format_version"] == __version__


def test_workspace_does_not_shadow_mlbricks_namespace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    b = Builder()
    root = Path(b.local_environment["paths"]["root"])
    assert root.name == "mlbricks_workspace"
    assert root.exists()
    assert not (tmp_path / "mlbricks").exists()


def test_mlbricks_diagnostics_does_not_treat_plain_namespace_dir_as_install(tmp_path, monkeypatch):
    # This test only exercises the false-positive case when mlbricks distribution
    # metadata is not present in the test environment.
    try:
        importlib.metadata.distribution("mlbricks")
    except importlib.metadata.PackageNotFoundError:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "mlbricks").mkdir()
        info = get_mlbricks_info()
        assert info == {"installed": False, "version": None, "module_path": None}


def test_builder_html_no_longer_duplicates_popout_asset_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    html = Builder()._repr_html_()
    # Original V1.0 output was ~1.5 MB because JS/CSS were embedded twice.
    assert len(html.encode("utf-8")) < 1_100_000
    assert "window.__MLB_STUDIO_JS_SOURCE__" in html


def test_studio_import_does_not_eagerly_import_torch(tmp_path):
    import os
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")
    code = (
        "import sys; import mlb_studio; "
        "assert 'torch' not in sys.modules; "
        "b=mlb_studio.Builder(); "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
