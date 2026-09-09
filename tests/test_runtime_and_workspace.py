from __future__ import annotations

import importlib.metadata
from pathlib import Path

from mlb_studio import Builder, __version__
from mlb_studio.graph import new_project
from mlb_studio.runtime import get_mlbricks_info


def test_versions_are_consistent():
    assert __version__ == "1.0.0b1"
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
        version = "1.0.0b2"

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
        "version": "1.0.0b2",
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


def test_windows_runtime_capabilities_show_torch_visible_cuda_devices(monkeypatch):
    import mlb_studio.builder as builder_module

    monkeypatch.setattr(builder_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        builder_module,
        "_probe_windows_torch_cuda",
        lambda: {
            "probed": True,
            "available": True,
            "cuda_version": "12.8",
            "devices": [{
                "index": 0,
                "name": "NVIDIA GeForce RTX 2050",
                "compute_capability": "8.6",
                "total_memory": 4 * 1024**3,
            }],
        },
    )

    capabilities = builder_module.Builder._detect_runtime_capabilities(
        object.__new__(builder_module.Builder)
    )
    gpu = next(item for item in capabilities["devices"] if item["kind"] == "cuda")
    assert gpu["id"] == "cuda:0"
    assert gpu["label"] == "GPU 0 — NVIDIA GeForce RTX 2050"
    assert gpu["compute_capability"] == "8.6"
    assert capabilities["cuda_version"] == "12.8"


def test_windows_cpu_only_torch_does_not_advertise_unusable_cuda(monkeypatch):
    import mlb_studio.builder as builder_module

    monkeypatch.setattr(builder_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        builder_module,
        "_probe_windows_torch_cuda",
        lambda: {
            "probed": True,
            "available": False,
            "cuda_version": None,
            "devices": [],
        },
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    capabilities = builder_module.Builder._detect_runtime_capabilities(
        object.__new__(builder_module.Builder)
    )
    assert all(item["kind"] != "cuda" for item in capabilities["devices"])


def test_macos_runtime_capabilities_show_available_mps_device(monkeypatch):
    import mlb_studio.builder as builder_module

    monkeypatch.setattr(builder_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(builder_module, "_probe_windows_torch_cuda", lambda: None)
    monkeypatch.setattr(
        builder_module,
        "_probe_macos_torch_mps",
        lambda: {"probed": True, "built": True, "available": True},
    )

    capabilities = builder_module.Builder._detect_runtime_capabilities(
        object.__new__(builder_module.Builder)
    )
    gpu = next(item for item in capabilities["devices"] if item["kind"] == "mps")
    assert gpu["id"] == "mps"
    assert gpu["label"] == "GPU — Apple Metal (MPS)"
    assert capabilities["mps_available"] is True


def test_macos_without_torch_mps_does_not_advertise_gpu(monkeypatch):
    import mlb_studio.builder as builder_module

    monkeypatch.setattr(builder_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(builder_module, "_probe_windows_torch_cuda", lambda: None)
    monkeypatch.setattr(
        builder_module,
        "_probe_macos_torch_mps",
        lambda: {"probed": True, "built": False, "available": False},
    )

    capabilities = builder_module.Builder._detect_runtime_capabilities(
        object.__new__(builder_module.Builder)
    )
    assert all(item["kind"] != "mps" for item in capabilities["devices"])
    assert capabilities["mps_available"] is False


def test_runtime_defaults_use_device_aware_precision():
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "mlb_studio" / "static" / "builder.js"
    ).read_text(encoding="utf-8")
    training_start = source.index("function defaultTrainingConfig")
    generation_start = source.index("function defaultGenerationConfig", training_start)
    merge_start = source.index("function mergeRuntimeDefaults", generation_start)
    assert 'precision:"auto"' in source[training_start:generation_start]
    assert 'precision:"auto"' in source[generation_start:merge_start]


def test_training_reports_runtime_import_before_lazy_model_runtime_import():
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "mlb_studio" / "builder.py"
    ).read_text(encoding="utf-8")
    start = source.index("    def train_model(")
    end = source.index("    def generate_model(", start)
    block = source[start:end]
    assert block.index('"phase":"runtime_import"') < block.index(
        "from .model_runtime import train_builder_model"
    )
    assert "if cached_runtimes:" in block


def test_amp_training_keeps_fp32_master_parameters():
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "mlb_studio" / "model_runtime.py"
    ).read_text(encoding="utf-8")
    start = source.index("def compile_builder_model(")
    end = source.index("\n\nclass _PackedLMBatcher", start)
    block = source[start:end]
    assert 'if for_training and precision in {"fp16", "bf16"}' in block
    assert "raw.to(device=device,dtype=parameter_dtype)" in block


def test_generation_cache_prefers_resident_python_model():
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "mlb_studio" / "builder.py"
    ).read_text(encoding="utf-8")
    start = source.index("    def _resident_generation_runtime(")
    end = source.index("    def start_model_server(", start)
    block = source[start:end]
    assert "desired_device = resident.device" in block
    assert "requested_precision" in block
    assert '"phase":"resident_reuse"' in block
    assert "no checkpoint reload" in block
    assert "load_trained_for_generation" in block
    assert block.index("_resident_generation_runtime(model_id, config") < block.index("load_trained_for_generation(")


def test_generation_start_clears_stale_output():
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "mlb_studio" / "static" / "builder.js"
    ).read_text(encoding="utf-8")
    start = source.index("function startGenerationFromRuntime")
    end = source.index("function generationActionButton", start)
    assert 'entry.last_generation="";' in source[start:end]
