from __future__ import annotations

from pathlib import Path
import threading
import pytest

import mlb_studio.model_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]


class Split:
    def __init__(self, data):
        self.data = data
        self.column_names = list(data)
    def __getitem__(self, key):
        return self.data[key]
    def __len__(self):
        return len(next(iter(self.data.values()))) if self.data else 0


def _edge(a, b, *, kind="main", source_port="main_out", target_port="main_in"):
    return {"id": f"{a}-{b}-{kind}-{target_port}", "source": a, "target": b, "kind": kind, "source_port": source_port, "target_port": target_port}


def _named(a, b, key):
    return _edge(a, b, kind="named", source_port="named_out:main", target_port=f"named_in:{key}")


def test_ml_gallery_demo_widths_match_first_principles_models():
    for rel in ["frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"]:
        js = (ROOT / rel).read_text(encoding="utf-8")
        regression = next(line for line in js.splitlines() if 'id:"ml_regression"' in line)
        binary = next(line for line in js.splitlines() if 'id:"ml_binary"' in line)
        assert 'feature_count:4' in regression
        assert 'feature_count:4' in binary
        assert '"Feature width"' in js


def test_supervised_training_rejects_feature_width_mismatch_before_matmul(tmp_path):
    nodes = [
        {"id":"x","type":"feature_input","name":"Feature Input","params":{"feature_dim":4}},
        {"id":"w","type":"learnable_parameter","name":"Weight W","params":{"shape":"4,1","init":"normal"}},
        {"id":"mm","type":"matmul","name":"X × W","params":{}},
        {"id":"b","type":"learnable_parameter","name":"Bias b","params":{"shape":"1","init":"zeros"}},
        {"id":"add","type":"tensor_add","name":"+ Bias","params":{}},
        {"id":"o","type":"tensor_output","name":"Output","params":{}},
    ]
    edges = [_named("x","mm","a"), _named("w","mm","b"), _named("mm","add","a"), _named("b","add","b"), _edge("add","o")]
    entry = {"name":"Linear Regression","architecture":{"nodes":nodes,"edges":edges},"requirements":{"modality":"signal","training_mode":"supervised","training_task":"regression","requires_tokenizer":False}}
    cols = {f"feature_{i}":[float(i)]*16 for i in range(1,9)}
    cols["target"] = [1.0]*16
    ds = {"train": Split(cols), "validation": Split(cols)}
    cfg = {"seed":1,"batch_size":4,"gradient_accumulation":1,"budget_type":"steps","max_steps":1,"optimizer":"sgd","learning_rate":0.01,"weight_decay":0.0,"beta1":0.9,"beta2":0.95,"warmup_steps":0,"validation_split":"validation","validate_every":0,"validation_steps":1,"checkpoint_every":0,"device":"cpu","backend":"pytorch","execution_mode":"eager","precision":"fp32","output_dir":str(tmp_path)}
    with pytest.raises(ValueError, match="Feature-width mismatch: model input expects 4 features, but the selected dataset provides 8"):
        runtime.train_builder_model(state={"project":{"task":"Regression"},"custom_components":{}}, model_entry=entry, dataset=ds, dataset_meta={"name":"bad"}, config=cfg, progress=lambda e: None, stop_event=threading.Event())
