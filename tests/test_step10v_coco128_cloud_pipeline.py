from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from mlb_studio import data as data_api
from mlb_studio.builder import Builder
from mlb_studio.graph import primitive_catalog
from mlb_studio.model_runtime import _supervised_graph_info


ROOT = Path(__file__).resolve().parents[1]


def _tiny_coco128_zip() -> bytes:
    buf = io.BytesIO()
    image_buf = io.BytesIO()
    Image.new("RGB", (20, 10), (120, 80, 30)).save(image_buf, format="JPEG")
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("coco128/images/train2017/000000000001.jpg", image_buf.getvalue())
        # class 16 = dog, centered box with width=0.5 and height=0.4
        zf.writestr("coco128/labels/train2017/000000000001.txt", "16 0.5 0.5 0.5 0.4\n")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._io = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size=-1):
        return self._io.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_coco128_cloud_loader_uses_temporary_download_and_returns_studio_detection_contract(monkeypatch):
    pytest.importorskip("datasets")
    payload = _tiny_coco128_zip()
    monkeypatch.setattr(data_api, "urlopen", lambda *args, **kwargs: _FakeResponse(payload))
    progress = []

    dataset = data_api.load_coco128_cloud_dataset(progress_callback=progress.append)

    assert len(dataset) == 1
    assert dataset.column_names == ["image", "boxes", "class_ids", "image_id"]
    row = dataset[0]
    assert row["image"].size == (20, 10)
    assert row["class_ids"] == [16]
    x, y, w, h = row["boxes"][0]
    assert round(x, 4) == 5.0
    assert round(y, 4) == 3.0
    assert round(w, 4) == 10.0
    assert round(h, 4) == 4.0
    assert dataset.features["class_ids"].feature.names[16] == "dog"
    assert progress[0]["storage"] == "temporary"
    assert progress[-1]["storage"] == "memory"
    assert progress[-1]["percent"] == 100.0


def test_prepared_dataset_summary_surfaces_coco_class_names():
    class ClassFeature:
        names = list(data_api.COCO80_CLASS_NAMES)

    class SequenceFeature:
        feature = ClassFeature()

    class FakeDataset:
        column_names = ["image", "boxes", "class_ids", "image_id"]
        features = {"class_ids": SequenceFeature()}
        def __len__(self):
            return 128

    summary = Builder._summarize_prepared_result(Builder.__new__(Builder), FakeDataset())
    assert summary["num_classes"] == 80
    assert summary["class_names"][16] == "dog"
    assert summary["splits"]["train"]["num_classes"] == 80


def test_coco128_is_a_public_data_source_component_and_gallery_pipeline():
    catalog = {item["type"]: item for item in primitive_catalog()}
    source = catalog["coco128_cloud"]
    assert source["category"] == "Data Source"
    assert source["builder_utility"] is True
    assert "temporary" in source["description"].lower()

    for rel in ("frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"):
        js = (ROOT / rel).read_text(encoding="utf-8")
        assert 'id:"vision_coco128",name:"COCO128 Cloud"' in js
        assert 'source=makeNode(cat(catalog,"coco128_cloud"))' in js
        assert 'det.params.width=preset.width||16' in js
        assert 'COCO128 cloud · temporary session storage' in js
        assert 'datasetClasses===modelClasses' in js
        assert 'Set the Detection Head classes to ' in js


def test_supervised_graph_info_exposes_detection_head_class_count():
    model_entry = {
        "name": "Detector",
        "task": "Object detection",
        "architecture": {
            "nodes": [
                {"id": "x", "type": "image_input", "params": {"channels": 3, "image_size": 128}},
                {"id": "head", "type": "detection_pyramid_head", "params": {"classes": 80}},
            ],
            "edges": [{"source": "x", "target": "head"}],
        },
    }
    info = _supervised_graph_info(model_entry, {"project": {"task": "Object detection"}})
    assert info["task"] == "object_detection"
    assert info["output_classes"] == 80


def test_training_backend_contains_hard_detection_class_mismatch_guard():
    source = (ROOT / "src/mlb_studio/model_runtime.py").read_text(encoding="utf-8")
    assert "Detection-class mismatch: model head has" in source
    assert '"class_names":dataset_class_names or None' in source
