from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from mlb_studio.graph import primitive_catalog
from mlb_studio.model_runtime import CompiledModel, run_universal_inference
from mlb_studio.universal_io import load_single_input, normalize_input_config


def test_universal_input_catalog_exposes_video_and_signal_sources():
    by_type = {item["type"]: item for item in primitive_catalog()}
    assert "video_input" in by_type
    assert "signal_input" in by_type

    video_fields = {field["key"]: field for field in by_type["video_input"]["api"]}
    assert video_fields["input_mode"]["options"] == ["file", "live", "cctv"]
    assert "camera" in video_fields["source_type"]["options"]
    assert "cctv" in video_fields["source_type"]["options"]

    signal_fields = {field["key"]: field for field in by_type["signal_input"]["api"]}
    assert signal_fields["input_mode"]["options"] == ["static", "continuous"]
    assert {"serial", "tcp", "sensor", "antenna", "file_tail"} <= set(signal_fields["source_type"]["options"])


def test_normalize_continuous_antenna_signal_input():
    env = normalize_input_config(
        {
            "input_kind": "signal",
            "input_mode": "continuous",
            "task_type": "anomaly",
            "input_source_type": "antenna",
            "input_source": "COM7",
            "input_sample_rate": 48000,
            "input_buffer_size": 512,
        }
    )
    assert env.kind == "signal"
    assert env.mode == "continuous"
    assert env.task == "anomaly"
    assert env.source_type == "antenna"
    assert env.source == "COM7"
    assert env.sample_rate == 48000
    assert env.buffer_size == 512
    assert env.continuous is True


def test_static_signal_becomes_model_ready_tensor():
    env = normalize_input_config(
        {
            "input_kind": "signal",
            "input_mode": "static",
            "input_data": "0.1, 0.2, -0.3, 1.0",
            "input_sample_rate": 1000,
        }
    )
    value, meta = load_single_input(env)
    assert isinstance(value, torch.Tensor)
    assert value.shape == (1, 4, 1)
    assert torch.allclose(value[0, :, 0], torch.tensor([0.1, 0.2, -0.3, 1.0]))
    assert meta["samples"] == 4
    assert meta["sample_rate"] == 1000


def test_image_input_loads_and_resizes_for_model(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (12, 8), (10, 20, 30)).save(path)
    env = normalize_input_config(
        {
            "input_kind": "image",
            "input_mode": "single",
            "input_source_type": "path_or_url",
            "input_source": str(path),
        }
    )
    value, meta = load_single_input(env, image_size=16)
    assert value.shape == (1, 3, 16, 16)
    assert {k: meta[k] for k in ("width", "height", "channels")} == {"width": 16, "height": 16, "channels": 3}
    assert meta["display_image"].startswith("data:image/jpeg;base64,")
    assert (meta["source_width"], meta["source_height"]) == (12, 8)


class _PromptAwareVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen = None

    def process_input(self, sample, *, prompt="", task="", metadata=None):
        self.seen = {
            "shape": list(sample.shape),
            "prompt": prompt,
            "task": task,
            "metadata": dict(metadata or {}),
        }
        return {"caption": prompt, "task": task, "shape": list(sample.shape)}


def test_universal_model_adapter_receives_image_prompt_and_task():
    model = _PromptAwareVisionModel()
    compiled = CompiledModel(
        model=model,
        raw_model=model,
        training_model=None,
        device=torch.device("cpu"),
        precision="fp32",
        vocab_size=0,
        parameter_count=1,
        compile_used=False,
        compile_error=None,
    )
    image = torch.zeros(1, 3, 16, 16)
    output = run_universal_inference(
        compiled,
        image,
        input_kind="image",
        output_type="logits_output",
        task="edit",
        prompt="remove the background",
        metadata={"frame": 1},
    )
    assert model.seen == {
        "shape": [1, 3, 16, 16],
        "prompt": "remove the background",
        "task": "edit",
        "metadata": {"frame": 1},
    }
    assert output["kind"] == "json"
    assert output["data"]["caption"] == "remove the background"


def test_signal_tensor_output_uses_signal_renderer_contract():
    class Echo(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward(self, x):
            return x * 2

    model = Echo()
    compiled = CompiledModel(
        model=model,
        raw_model=model,
        training_model=None,
        device=torch.device("cpu"),
        precision="fp32",
        vocab_size=0,
        parameter_count=1,
        compile_used=False,
        compile_error=None,
    )
    output = run_universal_inference(
        compiled,
        torch.tensor([[[1.0], [2.0], [3.0]]]),
        input_kind="signal",
        output_type="logits_output",
        task="analyze",
    )
    assert output["kind"] == "signal"
    assert output["data"] == [2.0, 4.0, 6.0]
    assert output["metadata"]["samples"] == 3


def test_frontend_exposes_adaptive_universal_input_controls():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "mlb_studio"
        / "static"
        / "builder.js"
    ).read_text(encoding="utf-8")
    assert 'runtimeSection("Universal Input")' in source
    assert '"Input Type","select",config.input_kind' in source
    assert '"Start Monitoring"' in source
    assert '"Start Stream"' in source
    assert '"Start Listening"' in source
    assert '"Edit Image"' in source
    assert '"Process Video"' in source
    assert 'runtimeSection("Generated Output")' in source


def test_non_text_artifact_load_skips_tokenizer(monkeypatch, tmp_path):
    import mlb_studio.model_runtime as runtime

    artifact = tmp_path / "vision-model"
    artifact.mkdir()
    (artifact / "model.pt").write_bytes(b"placeholder")
    loaded = nn.Identity()

    def _no_tokenizer(*args, **kwargs):
        raise AssertionError("non-text inference must not load a tokenizer")

    monkeypatch.setattr(runtime, "_tokenizer_for", _no_tokenizer)
    monkeypatch.setattr(runtime.IMPORT_POOL, "resolve_api", lambda name: (lambda *args, **kwargs: loaded))

    compiled, tokenizer = runtime.load_trained_for_generation(
        state={},
        model_entry={
            "requirements": {"modality": "image", "output_type": "logits_output"},
            "architecture": {"nodes": [{"id": "in", "type": "image_input", "params": {}}], "edges": []},
            "path": str(artifact),
        },
        dataset_meta={},
        config={"device": "cpu", "precision": "fp32", "backend": "auto", "execution_mode": "eager"},
    )
    assert compiled.raw_model is loaded
    assert tokenizer is None


def test_builder_universal_signal_runtime_is_end_to_end():
    import threading
    from mlb_studio import Builder

    class Echo(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward(self, x):
            return x + 1

    model = Echo()
    compiled = CompiledModel(
        model=model,
        raw_model=model,
        training_model=None,
        device=torch.device("cpu"),
        precision="fp32",
        vocab_size=0,
        parameter_count=1,
        compile_used=False,
        compile_error=None,
    )
    env = normalize_input_config(
        {
            "input_kind": "signal",
            "input_mode": "static",
            "task_type": "analyze",
            "input_data": [1.0, 2.0, 3.0],
        }
    )
    builder = object.__new__(Builder)
    builder._stop_event = threading.Event()
    events = []
    entry = {
        "requirements": {"modality": "signal", "output_type": "logits_output"},
        "architecture": {"nodes": [{"type": "signal_input", "params": {}}]},
    }

    output = builder._run_universal_input_runtime(compiled, entry, env, events.append)

    assert output["kind"] == "signal"
    assert output["data"] == [2.0, 3.0, 4.0]
    assert entry["last_generated_output_kind"] == "signal"
    assert events[0]["phase"] == "input_ready"
    assert events[-1]["status"] == "done"
    assert events[-1]["processed_items"] == 1
