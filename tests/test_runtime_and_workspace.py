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
    def missing_distribution(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing_distribution)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mlbricks").mkdir()
    info = get_mlbricks_info()
    assert info == {"installed": False, "version": None, "module_path": None}


def test_mlbricks_diagnostics_prefers_mlbricks_kit_distribution(monkeypatch):
    calls = []

    class Distribution:
        version = "1.0.0b1"

    def distribution(name):
        calls.append(name)
        if name == "mlbricks-kit":
            return Distribution()
        raise importlib.metadata.PackageNotFoundError(name)

    class Module:
        __file__ = "/tmp/mlbricks/__init__.py"

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    monkeypatch.setattr("mlb_studio.runtime.importlib.import_module", lambda name: Module())

    info = get_mlbricks_info()
    assert calls == ["mlbricks-kit"]
    assert info == {
        "installed": True,
        "version": "1.0.0b1",
        "module_path": "/tmp/mlbricks/__init__.py",
    }


def test_builder_html_no_longer_duplicates_popout_asset_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    html = Builder()._repr_html_()
    # Frontend assets are gzip+base64 encoded once, then expanded in-browser.
    # This keeps notebook output compact and avoids reparsing raw source text.
    assert len(html.encode("utf-8")) < 450_000
    assert "DecompressionStream" in html
    assert "window.__MLB_STUDIO_ASSETS_READY__" in html
    assert "runtimeScript.textContent = jsText" in html


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


def test_builder_startup_does_not_call_nvidia_smi():
    source = (Path(__file__).resolve().parents[1] / "src/mlb_studio/builder.py").read_text(encoding="utf-8")
    start = source.index("def _detect_runtime_capabilities")
    end = source.index("    def to_dict", start)
    block = source[start:end]
    assert "nvidia-smi" in block
    assert "subprocess.run(" not in block
    assert 'Path("/dev/nvidia0").exists()' in block
