from __future__ import annotations

import math
import threading
from pathlib import Path

import torch

from mlb_studio.data import process_detection_dataset
from mlb_studio.graph import primitive_catalog
import mlb_studio.model_runtime as runtime


class MiniDataset:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.column_names = list(self.rows[0]) if self.rows else []

    def map(self, fn):
        return MiniDataset([fn(dict(row)) for row in self.rows])

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def __len__(self):
        return len(self.rows)


class Split:
    def __init__(self, data):
        self.data = data
        self.column_names = list(data)

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(next(iter(self.data.values()))) if self.data else 0


def edge(a, b, *, source_port="main_out", target_port="main_in", kind="main"):
    return {
        "id": f"{a}-{b}-{target_port}",
        "source": a,
        "target": b,
        "kind": kind,
        "source_port": source_port,
        "target_port": target_port,
    }


def named(a, b, target):
    return edge(a, b, source_port="named_out:main", target_port=f"named_in:{target}", kind="named")


def yolo_nodes_edges():
    nodes = [
        {"id":"x","type":"image_input","name":"Image Input","params":{"channels":1,"image_size":16}},
        {"id":"c1","type":"conv2d","name":"Stem","params":{"in_channels":1,"out_channels":16,"kernel_size":3,"stride":1,"padding":1,"bias":False}},
        {"id":"a1","type":"silu","name":"SiLU","params":{}},
        {"id":"c2","type":"conv2d","name":"P3 Conv","params":{"in_channels":16,"out_channels":32,"kernel_size":3,"stride":2,"padding":1,"bias":False}},
        {"id":"p3","type":"silu","name":"P3","params":{}},
        {"id":"c3","type":"conv2d","name":"P4 Conv","params":{"in_channels":32,"out_channels":64,"kernel_size":3,"stride":2,"padding":1,"bias":False}},
        {"id":"p4","type":"silu","name":"P4","params":{}},
        {"id":"fpn","type":"fpn_fusion","name":"FPN Fusion","params":{"high_channels":64,"lateral_channels":32,"out_channels":32,"fusion":"add"}},
        {"id":"pan","type":"pan_fusion","name":"PAN Fusion","params":{"fine_channels":32,"coarse_channels":64,"out_channels":64,"fusion":"concat"}},
        {"id":"det","type":"detection_head","name":"Detection Head","params":{"in_channels":64,"classes":3,"anchors":1}},
    ]
    edges = [
        edge("x","c1"), edge("c1","a1"), edge("a1","c2"), edge("c2","p3"),
        edge("p3","c3"), edge("c3","p4"),
        named("p4","fpn","high"), named("p3","fpn","lateral"),
        named("fpn","pan","fine"), named("p4","pan","coarse"), edge("pan","det"),
    ]
    return nodes, edges


def test_vision_catalog_exposes_rebuildable_yolo_components():
    catalog = {item["type"]: item for item in primitive_catalog()}
    assert {"detection_process", "fpn_fusion", "pan_fusion", "detection_head"} <= set(catalog)
    assert [p["id"] for p in catalog["fpn_fusion"]["runtime_ports"]["inputs"]] == ["high", "lateral"]
    assert [p["id"] for p in catalog["pan_fusion"]["runtime_ports"]["inputs"]] == ["fine", "coarse"]
    assert catalog["detection_head"]["category"] == "Vision"


def test_detection_processing_resizes_image_and_boxes_together():
    raw = MiniDataset([{
        "image": [[0.0 for _ in range(8)] for _ in range(8)],
        "boxes": [[1.0, 2.0, 2.0, 3.0]],
        "class_ids": [2],
    }])
    out = process_detection_dataset(raw, width=16, height=16, mode="L")
    image = out["image"][0]
    box = out["boxes"][0][0]
    assert len(image) == 16 and len(image[0]) == 16
    assert box == [2.0, 4.0, 4.0, 6.0]
    assert out["class_ids"][0] == [2]


def test_yolo_style_public_graph_executes_and_backpropagates():
    nodes, edges = yolo_nodes_edges()
    graph = runtime.TensorGraph(
        nodes=nodes, edges=edges, custom_components={},
        runtime={"device":"cpu","backend":"pytorch","precision":"fp32","model_dim":64},
    )
    images = torch.randn(3, 1, 16, 16, requires_grad=True)
    pred = graph(images)
    assert pred.shape == (3, 8, 4, 4)
    pred.square().mean().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()


def test_detection_targets_and_loss_are_supported():
    images = [[[0.0] * 16 for _ in range(16)] for _ in range(2)]
    split = Split({
        "image": images,
        "boxes": [[[2.0, 2.0, 7.0, 7.0]], [[5.0, 4.0, 6.0, 5.0]]],
        "class_ids": [[0], [2]],
    })
    x, y, fields = runtime._supervised_xy(split, {"input_type":"image_input", "task":"object_detection"})
    assert x.shape == (2, 1, 16, 16)
    assert y.shape == (2, 1, 6)
    assert fields == ["image"]
    pred = torch.randn(2, 8, 4, 4, requires_grad=True)
    loss, metrics = runtime._supervised_loss_and_metrics(pred, y, "object_detection")
    assert math.isfinite(float(loss.detach()))
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["mae"] >= 0.0
    loss.backward()
    assert pred.grad is not None


def test_yolo_style_detector_trains_one_step(monkeypatch, tmp_path):
    nodes, edges = yolo_nodes_edges()
    model_entry = {
        "name":"YOLO-style Detector",
        "architecture":{"nodes":nodes,"edges":edges},
        "requirements":{"modality":"image","training_mode":"supervised","training_task":"object_detection","requires_tokenizer":False},
    }
    images=[]; boxes=[]; classes=[]
    for i in range(12):
        label=i%3
        image=[[0.0]*16 for _ in range(16)]
        image[4+label][4+label]=1.0
        images.append(image); boxes.append([[2.0+label,2.0,7.0,7.0]]); classes.append([label])
    ds={"train":Split({"image":images,"boxes":boxes,"class_ids":classes}),"validation":Split({"image":images[:6],"boxes":boxes[:6],"class_ids":classes[:6]})}

    original = runtime.IMPORT_POOL.resolve_api
    def resolve(key):
        if key == "lifecycle.save":
            def save(model, path, metadata=None):
                path=Path(path); path.mkdir(parents=True, exist_ok=True); (path/"model.pt").write_bytes(b"test")
            return save
        return original(key)
    monkeypatch.setattr(runtime.IMPORT_POOL, "resolve_api", resolve)

    result=runtime.train_builder_model(
        state={"project":{"task":"Object detection"},"custom_components":{}}, model_entry=model_entry,
        dataset=ds, dataset_meta={"name":"Object Detection Demo"},
        config={"seed":7,"batch_size":4,"gradient_accumulation":1,"budget_type":"steps","max_steps":1,"epochs":1,
                "optimizer":"sgd","learning_rate":0.01,"weight_decay":0.0,"beta1":0.9,"beta2":0.95,"warmup_steps":0,
                "validation_split":"validation","validate_every":1,"validation_steps":1,"checkpoint_every":0,
                "device":"cpu","backend":"pytorch","execution_mode":"eager","precision":"fp32","output_dir":str(tmp_path)},
        progress=lambda event: None, stop_event=threading.Event(),
    )
    update=result["model_update"]
    assert update["training_task"] == "object_detection"
    assert update["training_mode"] == "supervised"
    assert math.isfinite(update["last_loss"])
    assert 0.0 <= update["supervised_metrics"]["accuracy"] <= 1.0


def test_step4_gallery_and_data_pipeline_are_exposed_in_frontend():
    root = Path(__file__).resolve().parents[1]
    for rel in ["frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"]:
        text=(root/rel).read_text(encoding="utf-8")
        assert 'name:"Image Classifier",category:"Vision"' in text
        assert 'name:"YOLO-style Detector",category:"Vision"' in text
        assert 'name:"VESA-YOLO experimental",category:"Vision"' in text
        assert 'makeNode(cat(catalog,"detection_process"))' in text
        assert 'named(p5,fpn4,"high")' in text
        assert 'named(fpn3,det,"p3")' in text
        assert 'trainingTask="object_detection"' in text
