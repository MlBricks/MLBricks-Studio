from __future__ import annotations

import math
from pathlib import Path

import torch

from mlb_studio.graph import primitive_catalog
import mlb_studio.model_runtime as runtime
from mlb_studio.data import generate_demo_dataset


def edge(a,b,*,source_port="main_out",target_port="main_in",kind="main"):
    return {"id":f"{a}-{b}-{target_port}","source":a,"target":b,"kind":kind,"source_port":source_port,"target_port":target_port}


def named(a,b,target):
    return edge(a,b,source_port="named_out:main",target_port=f"named_in:{target}",kind="named")


def pyramid_graph():
    nodes=[
        {"id":"x","type":"image_input","name":"Image","params":{"channels":1,"image_size":16}},
        {"id":"c3","type":"conv2d","name":"P3C","params":{"in_channels":1,"out_channels":32,"kernel_size":3,"stride":2,"padding":1}},
        {"id":"p3","type":"silu","name":"P3","params":{}},
        {"id":"c4","type":"conv2d","name":"P4C","params":{"in_channels":32,"out_channels":64,"kernel_size":3,"stride":2,"padding":1}},
        {"id":"p4","type":"silu","name":"P4","params":{}},
        {"id":"c5","type":"conv2d","name":"P5C","params":{"in_channels":64,"out_channels":96,"kernel_size":3,"stride":2,"padding":1}},
        {"id":"p5","type":"silu","name":"P5","params":{}},
        {"id":"head","type":"detection_pyramid_head","name":"Head","params":{"p3_channels":32,"p4_channels":64,"p5_channels":96,"classes":3,"slots":3}},
    ]
    edges=[edge("x","c3"),edge("c3","p3"),edge("p3","c4"),edge("c4","p4"),edge("p4","c5"),edge("c5","p5"),named("p3","head","p3"),named("p4","head","p4"),named("p5","head","p5")]
    return nodes,edges


def test_catalog_has_production_detection_components():
    cat={x["type"]:x for x in primitive_catalog()}
    assert "detection_pyramid_head" in cat
    assert "detection_nms" in cat
    assert [p["id"] for p in cat["detection_pyramid_head"]["runtime_ports"]["inputs"]]==["p3","p4","p5"]
    assert cat["detection_pyramid_head"]["api"][-1]["key"]=="slots"


def test_pyramid_head_runs_three_scales_and_backpropagates():
    nodes,edges=pyramid_graph()
    graph=runtime.TensorGraph(nodes=nodes,edges=edges,custom_components={},runtime={"device":"cpu","backend":"pytorch","precision":"fp32","model_dim":64})
    x=torch.randn(2,1,16,16,requires_grad=True)
    pred=graph(x)
    assert isinstance(pred,tuple) and len(pred)==3
    assert pred[0].shape==(2,3,8,8,8)
    assert pred[1].shape==(2,3,8,4,4)
    assert pred[2].shape==(2,3,8,2,2)
    sum(p.square().mean() for p in pred).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_multi_object_targets_ciou_loss_nms_and_map_are_finite():
    target=torch.tensor([
        [[0.20,0.20,0.18,0.20,0,1],[0.70,0.30,0.28,0.18,1,1],[0.55,0.75,0.35,0.30,2,1]],
        [[0.30,0.65,0.25,0.25,1,1],[0,0,0,0,-1,0],[0,0,0,0,-1,0]],
    ],dtype=torch.float32)
    pred=(
        torch.randn(2,3,8,8,8,requires_grad=True),
        torch.randn(2,3,8,4,4,requires_grad=True),
        torch.randn(2,3,8,2,2,requires_grad=True),
    )
    loss,metrics=runtime._supervised_loss_and_metrics(pred,target,"object_detection")
    assert math.isfinite(float(loss.detach()))
    assert metrics["objects"]==4.0
    for key in ["box_iou","map50","precision","recall"]:
        assert 0.0 <= metrics[key] <= 1.0
    loss.backward()
    assert all(p.grad is not None for p in pred)
    det=runtime.decode_detection_predictions(pred,score_threshold=0.05,iou_threshold=0.5,max_detections=25)
    assert len(det)==2
    assert all(d.ndim==2 and d.size(-1)==6 and d.size(0)<=25 for d in det)


def test_map50_can_reach_one_for_exact_high_confidence_prediction():
    wh_logit=math.log(0.4/0.6)
    pred=torch.full((1,1,6,1,1),-12.0)
    pred[0,0,0,0,0]=0.0;pred[0,0,1,0,0]=0.0
    pred[0,0,2,0,0]=wh_logit;pred[0,0,3,0,0]=wh_logit
    pred[0,0,4,0,0]=12.0;pred[0,0,5,0,0]=12.0
    target=torch.tensor([[[0.5,0.5,0.4,0.4,0,1]]],dtype=torch.float32)
    score,precision,recall=runtime.detection_map50(pred,target,score_threshold=0.05)
    assert score > 0.99 and precision > 0.99 and recall > 0.99


def test_generated_detection_demo_contains_multiple_objects(monkeypatch):
    class FakeDataset(dict):
        @classmethod
        def from_dict(cls, payload):
            return cls(payload)
    class FakeDatasets:
        Dataset=FakeDataset
    monkeypatch.setattr(runtime, "__name__", runtime.__name__)  # keep fixture use explicit
    import mlb_studio.data as data_mod
    monkeypatch.setattr(data_mod, "_datasets", lambda: FakeDatasets)
    ds=generate_demo_dataset("object_detection",samples=12,seed=3,classes=3)
    counts=[len(v) for v in ds["boxes"]]
    assert max(counts)>=3
    assert min(counts)>=1
    assert all(len(b)==len(c) for b,c in zip(ds["boxes"],ds["class_ids"]))


def test_step10_gallery_uses_three_scale_head_and_reports_map():
    root=Path(__file__).resolve().parents[1]
    for rel in ["frontend/src/legacy-builder.js","src/mlb_studio/static/builder.js"]:
        text=(root/rel).read_text(encoding="utf-8")
        assert 'add("detection_pyramid_head","P3/P4/P5 Detection Head · COCO80"' in text
        assert 'named(fpn3,det,"p3")' in text
        assert 'CIoU training' in text
        assert 'mAP@.50 evaluation' in text


def test_pyramid_detector_trains_one_step(monkeypatch, tmp_path):
    import threading
    nodes,edges=pyramid_graph()
    model_entry={
        "name":"YOLO-style Detector",
        "architecture":{"nodes":nodes,"edges":edges},
        "requirements":{"modality":"image","training_mode":"supervised","training_task":"object_detection","requires_tokenizer":False},
    }
    class Split:
        def __init__(self,data): self.data=data; self.column_names=list(data)
        def __getitem__(self,key): return self.data[key]
        def __len__(self): return len(next(iter(self.data.values())))
    images=[];boxes=[];classes=[]
    for i in range(8):
        img=[[0.0]*16 for _ in range(16)]
        objs=[[1.0,1.0,4.0,4.0],[9.0,9.0,3.0,3.0]] if i%2==0 else [[5.0,4.0,5.0,4.0]]
        cls=[i%3,(i+1)%3] if i%2==0 else [i%3]
        images.append(img);boxes.append(objs);classes.append(cls)
    ds={"train":Split({"image":images,"boxes":boxes,"class_ids":classes}),"validation":Split({"image":images[:4],"boxes":boxes[:4],"class_ids":classes[:4]})}
    original=runtime.IMPORT_POOL.resolve_api
    def resolve(key):
        if key=="lifecycle.save":
            def save(model,path,metadata=None):
                path=Path(path);path.mkdir(parents=True,exist_ok=True);(path/"model.pt").write_bytes(b"test")
            return save
        return original(key)
    monkeypatch.setattr(runtime.IMPORT_POOL,"resolve_api",resolve)
    result=runtime.train_builder_model(
        state={"project":{"task":"Object detection"},"custom_components":{}},model_entry=model_entry,dataset=ds,dataset_meta={"name":"Multi-object Demo"},
        config={"seed":3,"batch_size":2,"gradient_accumulation":1,"budget_type":"steps","max_steps":1,"epochs":1,"optimizer":"sgd","learning_rate":0.01,"weight_decay":0.0,"beta1":0.9,"beta2":0.95,"warmup_steps":0,"validation_split":"validation","validate_every":1,"validation_steps":1,"checkpoint_every":0,"device":"cpu","backend":"pytorch","execution_mode":"eager","precision":"fp32","output_dir":str(tmp_path)},
        progress=lambda event:None,stop_event=threading.Event(),
    )
    metrics=result["model_update"]["supervised_metrics"]
    assert result["model_update"]["training_task"]=="object_detection"
    assert math.isfinite(result["model_update"]["last_loss"])
    assert "map50" in metrics and "validation_map50" in metrics
