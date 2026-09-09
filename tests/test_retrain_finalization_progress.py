from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from mlb_studio.builder import Builder
from mlb_studio.model_runtime import _evaluate


class _EvalBatcher:
    def __init__(self):
        self.calls = 0

    def batch(self, batch_size, device):
        self.calls += 1
        x = torch.ones((batch_size, 4), dtype=torch.long, device=device) * self.calls
        y = torch.ones((batch_size, 4), dtype=torch.long, device=device)
        return x, y, int(batch_size * 4)


class _EvalLoss(torch.nn.Module):
    def forward(self, x, y):
        return x.float().mean() + y.float().mean()


class _Raw(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x


def test_validation_reports_every_bounded_batch_progress():
    batcher = _EvalBatcher()
    updates = []
    loss = _evaluate(
        _EvalLoss(), _Raw(), batcher,
        steps=3, batch_size=2, device=torch.device("cpu"), precision="fp32",
        progress_callback=lambda done, total, running: updates.append((done, total, running)),
    )

    assert batcher.calls == 3
    assert [item[:2] for item in updates] == [(1, 3), (2, 3), (3, 3)]
    assert loss == updates[-1][2]


def test_retrain_emits_commit_progress_before_terminal_done(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path / ".studio"))
    builder = Builder()
    output_root = tmp_path / "models"
    existing = output_root / "Model-1"
    (existing / "last").mkdir(parents=True)
    (existing / "last" / "model.pt").write_bytes(b"old")

    builder.state["prepared_datasets"] = [{"id": "d1", "name": "Data"}]
    builder.prepared_datasets["d1"] = {"train": [{"input_ids": [1, 2, 3]}]}
    builder.state["model_outputs"] = [{
        "id": "m1", "name": "Model 1", "selected_dataset_id": "d1",
        "training_config": {"output_dir": str(output_root), "execution_mode": "eager"},
    }]
    info = {
        "present": True, "resumable": True, "path": str(existing / "last"),
        "kind": "trained_model", "resume_mode": "weights", "metadata": {}, "reason": None,
    }
    monkeypatch.setattr(builder, "_existing_model_training_artifact", lambda *a, **k: dict(info))

    import mlb_studio.model_runtime as runtime
    compiled = SimpleNamespace(compile_used=False)

    def fake_train_builder_model(**kwargs):
        staged = Path(kwargs["config"]["output_dir"]) / "Model-1"
        (staged / "last").mkdir(parents=True)
        (staged / "last" / "model.pt").write_bytes(b"new")
        return {
            "model_update": {
                "training_status": "trained", "weights_ready": True,
                "path": str(staged / "last"), "checkpoint_path": str(staged / "last"),
                "execution_mode_used": "eager",
            },
            "compiled": compiled, "tokenizer": object(), "last_sample": None,
        }

    monkeypatch.setattr(runtime, "train_builder_model", fake_train_builder_model)
    events = []
    builder.train_model("m1", resume_existing=True, progress_callback=events.append)

    phases = [event.get("phase") for event in events]
    assert "retrain_commit" in phases
    assert "retrain_committed" in phases
    assert phases[-1] == "done"
    assert events[-1]["status"] == "done"
    assert events[-1]["overall"] == 100


def test_frontend_rejects_stale_progress_after_newer_python_event():
    source = (Path(__file__).parents[1] / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    assert "acceptExecutionProgressSequence" in source
    assert "seq<lastBridgeEventSeq" in source
    assert "lower-sequence event overwrite a terminal 100% event" in source


def test_training_runtime_reserves_final_progress_for_validation_and_save():
    source = (Path(__file__).parents[1] / "src" / "mlb_studio" / "model_runtime.py").read_text(encoding="utf-8")
    assert '"phase":"final_save","overall":99' in source
    assert '"phase":"validation_generation"' in source
    assert "return min(95,max(2,round(ratio*95)))" in source
