from __future__ import annotations

"""Step 9 diagnostics and portable-project helpers for MLBricks Studio.

The browser UI mirrors these rules for immediate feedback.  Keeping a small
Python implementation makes graph contracts, profiling, experiment comparison
and project bundles available to notebooks/tests without requiring the UI.
"""

from copy import deepcopy
from math import prod
from typing import Any, Mapping

MODEL_INPUT_TYPES = {
    "text_input", "image_input", "video_input", "audio_input", "signal_input",
    "feature_input", "abstract_input",
}
MODEL_OUTPUT_TYPES = {
    "text_output", "audio_output", "tensor_output", "abstract_output",
    "lm_head", "classifier", "detection_head", "detection_pyramid_head", "detection_nms", "regression_head",
}
DATA_SOURCE_TYPES = {
    "demo_dataset", "coco128_cloud", "manual_dataset", "hf_dataset", "kaggle_dataset",
    "url_dataset", "local_dataset",
}


def _nodes(component: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    return [dict(x) for x in (component or {}).get("nodes", []) if isinstance(x, Mapping)]


def _edges(component: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    return [dict(x) for x in (component or {}).get("edges", []) if isinstance(x, Mapping)]


def analyze_graph_contract(component: Mapping[str, Any] | None, workspace: str = "model") -> dict[str, Any]:
    """Return structural contract checks without compiling or executing the graph."""
    nodes, edges = _nodes(component), _edges(component)
    node_ids = {str(n.get("id") or "") for n in nodes}
    checks: list[dict[str, Any]] = []

    def add(label: str, ok: bool, detail: str, severity: str = "error") -> None:
        checks.append({"label": label, "ok": bool(ok), "detail": detail, "severity": severity})

    add("Graph has nodes", bool(nodes), f"{len(nodes)} node(s)")
    bad_edges = [e for e in edges if str(e.get("source") or "") not in node_ids or str(e.get("target") or "") not in node_ids]
    add("Connections reference existing nodes", not bad_edges, "All connections valid" if not bad_edges else f"{len(bad_edges)} broken connection(s)")

    incoming = {nid: 0 for nid in node_ids}
    outgoing = {nid: 0 for nid in node_ids}
    for e in edges:
        s, t = str(e.get("source") or ""), str(e.get("target") or "")
        if s in outgoing: outgoing[s] += 1
        if t in incoming: incoming[t] += 1

    if str(workspace).lower() == "data":
        sources = [n for n in nodes if str(n.get("type") or "") in DATA_SOURCE_TYPES]
        outputs = [n for n in nodes if str(n.get("type") or "") == "prepared_dataset"]
        add("Data source", bool(sources), f"{len(sources)} source(s)" if sources else "Add a dataset source")
        add("Prepared dataset output", bool(outputs), f"{len(outputs)} prepared output(s)" if outputs else "Add Prepared Dataset")
    else:
        inputs = [n for n in nodes if str(n.get("type") or "") in MODEL_INPUT_TYPES]
        outputs = [n for n in nodes if str(n.get("type") or "") in MODEL_OUTPUT_TYPES]
        add("Model input", bool(inputs), f"{len(inputs)} input node(s)" if inputs else "Add an input component")
        add("Model output/head", bool(outputs), f"{len(outputs)} output/head node(s)" if outputs else "Add an output or task head")
        keys: list[str] = []
        for n in inputs:
            key = str((n.get("params") or {}).get("input_key") or "").strip()
            if key: keys.append(key)
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        add("Named runtime inputs", not duplicates, "Unique runtime input keys" if not duplicates else "Duplicate keys: " + ", ".join(duplicates))

    isolated = [n for n in nodes if incoming.get(str(n.get("id") or ""), 0) == 0 and outgoing.get(str(n.get("id") or ""), 0) == 0]
    if isolated:
        add("Isolated nodes", False, f"{len(isolated)} disconnected node(s)", severity="warning")

    errors = [c for c in checks if not c["ok"] and c["severity"] == "error"]
    warnings = [c for c in checks if not c["ok"] and c["severity"] == "warning"]
    return {"ok": not errors, "checks": checks, "errors": errors, "warnings": warnings, "nodes": len(nodes), "edges": len(edges)}


def _int(params: Mapping[str, Any], *names: str, default: int = 0) -> int:
    for name in names:
        try:
            value = params.get(name)
            if value is not None and str(value) != "": return int(value)
        except Exception:
            pass
    return int(default)


def _shape_product(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        vals = [int(x) for x in value]
    else:
        text = str(value or "").replace("x", ",")
        vals = [int(x.strip()) for x in text.split(",") if x.strip().lstrip("-").isdigit()]
    return int(prod(vals)) if vals else 0


def estimate_node_parameters(node: Mapping[str, Any]) -> int:
    """Best-effort transparent estimate from visible node configuration."""
    typ = str(node.get("type") or "")
    p = node.get("params") or {}
    bias = str(p.get("bias", "true")).lower() not in {"false", "0", "no"}
    if typ == "learnable_parameter": return _shape_product(p.get("shape"))
    if typ in {"linear", "dense"}:
        i = _int(p, "in_features", "dim", "input_dim")
        o = _int(p, "out_features", "hidden_size", "output_dim")
        return i * o + (o if bias else 0)
    if typ == "embedding":
        return _int(p, "vocab_size", default=0) * _int(p, "embedding_dim", "hidden_size", "dim", default=0)
    if typ in {"conv1d", "conv2d", "conv3d"}:
        i, o = _int(p, "in_channels"), _int(p, "out_channels")
        k = _int(p, "kernel_size", default=1)
        power = {"conv1d": 1, "conv2d": 2, "conv3d": 3}[typ]
        return o * i * (k ** power) + (o if bias else 0)
    if typ in {"batchnorm1d", "batchnorm2d", "batchnorm3d"}:
        return 2 * _int(p, "num_features")
    if typ in {"layernorm", "rmsnorm"}:
        return (2 if typ == "layernorm" else 1) * _int(p, "normalized_shape", "hidden_size", "dim")
    if typ == "classifier":
        d, h, c = _int(p, "dim", "input_dim"), _int(p, "hidden_size"), _int(p, "classes", "num_classes")
        if h and c: return d*h+h + h*c+c
        return d*c+c
    if typ in {"rnn", "gru", "lstm"}:
        i, h = _int(p, "input_size", "dim"), _int(p, "hidden_size", "dim")
        gates = {"rnn": 1, "gru": 3, "lstm": 4}[typ]
        return gates * (i*h + h*h + 2*h)
    if typ == "jepa_predictor":
        d, h = _int(p, "latent_dim", "dim"), _int(p, "hidden_dim", "hidden_size")
        return d*h+h + h*d+d if d and h else 0
    if typ == "speaker_embedding":
        return _int(p, "speakers", "num_embeddings") * _int(p, "dim", "embedding_dim")
    return 0


def profile_graph(component: Mapping[str, Any] | None) -> dict[str, Any]:
    nodes, edges = _nodes(component), _edges(component)
    per_node = []
    total = 0
    for n in nodes:
        count = estimate_node_parameters(n)
        total += count
        per_node.append({"id": n.get("id"), "name": n.get("name") or n.get("type"), "type": n.get("type"), "parameters": count})
    trainable_nodes = sum(1 for x in per_node if x["parameters"] > 0)
    return {
        "nodes": len(nodes), "edges": len(edges), "estimated_parameters": total,
        "trainable_nodes": trainable_nodes, "parameterized_nodes": per_node,
    }


def compare_experiments(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two Studio experiment snapshots with numeric deltas when possible."""
    pa, pb = a.get("profile") or {}, b.get("profile") or {}
    ma, mb = a.get("metrics") or {}, b.get("metrics") or {}
    keys = sorted(set(ma) | set(mb))
    metric_rows = []
    for key in keys:
        av, bv = ma.get(key), mb.get(key)
        delta = None
        try: delta = round(float(bv) - float(av), 12)
        except Exception: pass
        metric_rows.append({"metric": key, "a": av, "b": bv, "delta": delta})
    return {
        "a": a.get("name") or "Experiment A", "b": b.get("name") or "Experiment B",
        "nodes_delta": int(pb.get("nodes") or 0) - int(pa.get("nodes") or 0),
        "edges_delta": int(pb.get("edges") or 0) - int(pa.get("edges") or 0),
        "parameters_delta": int(pb.get("estimated_parameters") or 0) - int(pa.get("estimated_parameters") or 0),
        "metrics": metric_rows,
    }


def make_project_bundle(state: Mapping[str, Any]) -> dict[str, Any]:
    """Create a portable, explicit Step-9 bundle. Raw dataset/model bytes stay external."""
    clean = deepcopy(dict(state))
    clean.pop("_runtime_command", None)
    clean.pop("_session_secrets", None)
    return {
        "format": "mlbricks-project-bundle",
        "format_version": "1.0",
        "manifest": {
            "project_name": (clean.get("project") or {}).get("name") or "Untitled Model",
            "active_workspace": clean.get("active_workspace") or "model",
            "contains": ["model_graph", "data_graph", "training_recipes", "custom_components", "experiments", "gallery_metadata"],
        },
        "state": clean,
    }


def load_project_bundle(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("format") != "mlbricks-project-bundle":
        raise ValueError("Not an MLBricks project bundle")
    state = payload.get("state")
    if not isinstance(state, Mapping) or not isinstance(state.get("components"), Mapping):
        raise ValueError("Project bundle is missing Studio state/components")
    return deepcopy(dict(state))
