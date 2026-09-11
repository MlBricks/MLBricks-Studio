from __future__ import annotations

from pathlib import Path

from mlb_studio.diagnostics import (
    analyze_graph_contract,
    compare_experiments,
    load_project_bundle,
    make_project_bundle,
    profile_graph,
)


def model_component():
    return {
        "id": "model",
        "nodes": [
            {"id": "x", "type": "feature_input", "name": "Input", "params": {"feature_dim": 4}},
            {"id": "w", "type": "learnable_parameter", "name": "W", "params": {"shape": "4,3"}},
            {"id": "mm", "type": "matmul", "name": "MatMul", "params": {}},
            {"id": "out", "type": "tensor_output", "name": "Output", "params": {}},
        ],
        "edges": [
            {"id": "e1", "source": "x", "target": "mm"},
            {"id": "e2", "source": "w", "target": "mm"},
            {"id": "e3", "source": "mm", "target": "out"},
        ],
    }


def test_step9_contract_and_profile_api():
    comp = model_component()
    contract = analyze_graph_contract(comp, "model")
    assert contract["ok"] is True
    assert any(c["label"] == "Model input" and c["ok"] for c in contract["checks"])
    profile = profile_graph(comp)
    assert profile["nodes"] == 4
    assert profile["edges"] == 3
    assert profile["estimated_parameters"] == 12


def test_step9_contract_detects_broken_edge_and_missing_data_output():
    comp = {"nodes": [{"id": "s", "type": "demo_dataset", "params": {}}], "edges": [{"source": "s", "target": "missing"}]}
    report = analyze_graph_contract(comp, "data")
    assert report["ok"] is False
    assert any("broken" in c["detail"] for c in report["checks"])
    assert any(c["label"] == "Prepared dataset output" and not c["ok"] for c in report["checks"])


def test_step9_project_bundle_roundtrip_and_secret_cleanup():
    state = {
        "project": {"name": "Research Model"},
        "components": {"m": model_component()},
        "root_component_id": "m",
        "active_workspace": "model",
        "experiments": [{"name": "A"}],
        "_runtime_command": {"action": "train"},
        "_session_secrets": {"token": "secret"},
    }
    bundle = make_project_bundle(state)
    assert bundle["format"] == "mlbricks-project-bundle"
    assert "model_graph" in bundle["manifest"]["contains"]
    restored = load_project_bundle(bundle)
    assert restored["project"]["name"] == "Research Model"
    assert "_session_secrets" not in restored
    assert "_runtime_command" not in restored


def test_step9_experiment_comparison():
    a = {"name": "A", "profile": {"nodes": 3, "edges": 2, "estimated_parameters": 100}, "metrics": {"loss": 1.0}}
    b = {"name": "B", "profile": {"nodes": 4, "edges": 3, "estimated_parameters": 120}, "metrics": {"loss": 0.8}}
    result = compare_experiments(a, b)
    assert result["nodes_delta"] == 1
    assert result["parameters_delta"] == 20
    assert result["metrics"][0]["delta"] == -0.2


def test_step9_frontend_keeps_diagnostics_experiments_and_bundle_without_modes():
    root = Path(__file__).resolve().parents[1]
    js = (root / "frontend/src/legacy-builder.js").read_text(encoding="utf-8")
    css = (root / "frontend/src/builder.css").read_text(encoding="utf-8")
    for token in [
        "graphContractReport", "profileComponentGraph", "Explain Graph",
        "Save Experiment", "Compare Last 2", "mlbricks-project-bundle",
        "showDataPresetInspector", 'btn("Inspect","mlb-gallery-action")',
    ]:
        assert token in js
    assert 'modeSwitch.className="mlb-studio-mode-switch"' not in js
    assert "mlb-step9-diagnostics" in css


def test_step9_static_distribution_contains_same_features():
    root = Path(__file__).resolve().parents[1]
    js = (root / "src/mlb_studio/static/builder.js").read_text(encoding="utf-8")
    assert "mlbricks-project-bundle" in js
    assert "EXPERIMENT HISTORY" in js
    assert "Data Inspector" in js
