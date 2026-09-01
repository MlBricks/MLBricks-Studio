from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import threading
import time

from . import data as data_api


SOURCE_TYPES = {
    "manual_dataset",
    "hf_dataset",
    "kaggle_dataset",
    "url_dataset",
    "local_dataset",
}

EXECUTABLE_TYPES = SOURCE_TYPES | {
    "text_process",
    "train_test_split",
    "tokenize_text",
    "image_process",
    "audio_process",
    "batch_data",
    "prepared_dataset",
}


class PipelineValidationError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(e["message"] for e in errors))


class PipelineStopped(RuntimeError):
    pass


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_positive(value: Any):
    if value in (None, "", 0, "0"):
        return None
    value = int(value)
    return value if value > 0 else None


def get_data_component(state: dict) -> dict:
    workspaces = state.get("workspaces") or {}
    ws = workspaces.get("data")
    if ws:
        component_id = ws.get("root_component_id")
    else:
        component_id = state.get("view_component_id")
    component = (state.get("components") or {}).get(component_id)
    if not component:
        raise PipelineValidationError([{
            "node_id": None,
            "message": "Data Processing workspace was not found."
        }])
    return component


def validate_data_pipeline(state: dict):
    component = get_data_component(state)
    nodes = list(component.get("nodes") or [])
    edges = [
        e for e in (component.get("edges") or [])
        if e.get("kind", "main") == "main"
    ]
    by_id = {n["id"]: n for n in nodes}
    errors = []

    if not nodes:
        errors.append({"node_id": None, "message": "The Data Processing canvas is empty."})
        return [], errors

    for node in nodes:
        if node.get("type") not in EXECUTABLE_TYPES:
            errors.append({
                "node_id": node.get("id"),
                "message": f'{node.get("name", "Node")} is not executable in the Data Processing runner.'
            })

    sources = [n for n in nodes if n.get("type") in SOURCE_TYPES]
    if len(sources) != 1:
        ids = [n.get("id") for n in sources]
        errors.append({
            "node_id": ids[0] if ids else None,
            "node_ids": ids,
            "message": (
                "Use exactly one Data Source in the beginner pipeline. "
                f"Found {len(sources)}. Remove extra Hugging Face/Kaggle/URL/Local/Manual sources."
            ),
        })

    incoming = {n["id"]: [] for n in nodes}
    outgoing = {n["id"]: [] for n in nodes}
    for e in edges:
        a, b = e.get("source"), e.get("target")
        if a in by_id and b in by_id:
            outgoing[a].append(b)
            incoming[b].append(a)

    # Beginner runner is deliberately linear: one main input and one main output.
    for n in nodes:
        nid = n["id"]
        if len(incoming[nid]) > 1:
            errors.append({
                "node_id": nid,
                "message": f'{n["name"]} has more than one Main input. Add a merge processor before running.'
            })
        if len(outgoing[nid]) > 1:
            errors.append({
                "node_id": nid,
                "message": f'{n["name"]} branches to multiple Main outputs. Use one linear beginner pipeline for now.'
            })

    if sources:
        source = sources[0]
        if incoming[source["id"]]:
            errors.append({
                "node_id": source["id"],
                "message": "The Data Source must be the first step."
            })

    outputs = [n for n in nodes if n.get("type") == "prepared_dataset"]
    if len(outputs) != 1:
        errors.append({
            "node_id": outputs[0]["id"] if outputs else None,
            "node_ids": [n["id"] for n in outputs],
            "message": f"Use exactly one Prepared Dataset output. Found {len(outputs)}."
        })
    elif outgoing[outputs[0]["id"]]:
        errors.append({
            "node_id": outputs[0]["id"],
            "message": "Prepared Dataset must be the final step, not a middle step."
        })

    # Split percentages.
    for n in nodes:
        if n.get("type") == "train_test_split":
            p = n.get("params") or {}
            train = float(p.get("train_size", 90))
            validation = float(p.get("validation_size", 5))
            test = float(p.get("test_size", 5))
            total = train + validation + test
            if train <= 0 or min(train, validation, test) < 0 or abs(total - 100.0) > 1e-6:
                errors.append({
                    "node_id": n["id"],
                    "message": (
                        "Train + Validation + Test must equal 100%. "
                        f"Current total is {total:g}%."
                    )
                })

    # Topological order.
    indegree = {nid: len(v) for nid, v in incoming.items()}
    queue = [nid for nid, degree in indegree.items() if degree == 0]
    order_ids = []
    while queue:
        nid = queue.pop(0)
        order_ids.append(nid)
        for nxt in outgoing[nid]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(order_ids) != len(nodes):
        errors.append({"node_id": None, "message": "The Main data flow contains a cycle."})
        return [], errors

    order = [by_id[nid] for nid in order_ids]

    if sources and order and order[0]["id"] != sources[0]["id"]:
        errors.append({
            "node_id": sources[0]["id"],
            "message": "The Data Source is disconnected from the start of the pipeline."
        })

    if outputs and order and order[-1]["id"] != outputs[0]["id"]:
        errors.append({
            "node_id": outputs[0]["id"],
            "message": "Prepared Dataset must be the last connected step."
        })

    # Require one connected chain, not disconnected islands.
    if len(order) > 1:
        for left, right in zip(order[:-1], order[1:]):
            if right["id"] not in outgoing[left["id"]]:
                errors.append({
                    "node_id": right["id"],
                    "message": f'{right["name"]} is disconnected. Connect every step through the Main lane.'
                })
                break

    return order, errors


def _emit(callback, payload):
    if callback:
        callback(payload)


def execute_data_pipeline(
    state: dict,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
):
    """Execute the Data Processing graph in Main-lane order."""
    stop_event = stop_event or threading.Event()
    order, errors = validate_data_pipeline(state)

    statuses = {
        n["id"]: {"status": "queued", "message": "Waiting"}
        for n in get_data_component(state).get("nodes", [])
    }

    if errors:
        for err in errors:
            ids = err.get("node_ids") or ([err.get("node_id")] if err.get("node_id") else [])
            for nid in ids:
                if nid in statuses:
                    statuses[nid] = {"status": "error", "message": err["message"]}
        _emit(progress_callback, {
            "status": "error",
            "overall": 0,
            "message": errors[0]["message"],
            "errors": errors,
            "nodes": statuses,
        })
        raise PipelineValidationError(errors)

    result = None
    total = len(order)

    for index, node in enumerate(order):
        if stop_event.is_set():
            statuses[node["id"]] = {"status": "stopped", "message": "Stopped before this step"}
            payload = {
                "status": "stopped",
                "overall": round(index / max(total, 1) * 100),
                "message": "Run stopped.",
                "nodes": statuses,
            }
            _emit(progress_callback, payload)
            raise PipelineStopped("Run stopped.")

        nid = node["id"]
        p = node.get("params") or {}
        statuses[nid] = {"status": "running", "message": f'Running {node["name"]}…'}
        _emit(progress_callback, {
            "status": "running",
            "overall": round(index / max(total, 1) * 100),
            "step": index + 1,
            "total_steps": total,
            "current_node_id": nid,
            "message": f'Running {node["name"]}…',
            "nodes": statuses,
        })

        try:
            typ = node["type"]

            if typ == "manual_dataset":
                result = data_api.load_manual_text_dataset(
                    p.get("text", "Once upon a time"),
                    text_column=p.get("text_column", "text"),
                    one_line_per_sample=_bool(p.get("one_line_per_sample", True)),
                )

            elif typ == "hf_dataset":
                result = data_api.load_huggingface_dataset(
                    p.get("dataset_id", "roneneldan/TinyStories"),
                    config=(p.get("config") or None),
                    split=p.get("split", "train"),
                    text_column=p.get("text_column", "text"),
                    streaming=_bool(p.get("streaming", False)),
                    max_rows=_optional_positive(p.get("max_rows")),
                )

            elif typ == "kaggle_dataset":
                result = data_api.load_kaggle_dataset(
                    p.get("dataset_handle", ""),
                    file_pattern=p.get("file_pattern", "*"),
                    format=p.get("format", "auto"),
                    text_column=p.get("text_column", "text"),
                    max_rows=_optional_positive(p.get("max_rows")),
                )

            elif typ == "url_dataset":
                result = data_api.load_url_dataset(
                    p.get("url", ""),
                    format=p.get("format", "auto"),
                    text_column=p.get("text_column", "text"),
                    max_rows=_optional_positive(p.get("max_rows")),
                )

            elif typ == "local_dataset":
                result = data_api.load_local_dataset(
                    p.get("path", ""),
                    format=p.get("format", "auto"),
                    text_column=p.get("text_column", "text"),
                    max_rows=_optional_positive(p.get("max_rows")),
                )

            elif typ == "text_process":
                result = data_api.process_text_dataset(
                    result,
                    text_column=p.get("text_column", "text"),
                    lowercase=_bool(p.get("lowercase", False)),
                    strip=_bool(p.get("strip", True)),
                    normalize_whitespace=_bool(p.get("normalize_whitespace", True)),
                    unicode_nfkc=_bool(p.get("unicode_nfkc", True)),
                    remove_empty=_bool(p.get("remove_empty", True)),
                    min_chars=int(p.get("min_chars", 1)),
                    max_chars=_optional_positive(p.get("max_chars")),
                )

            elif typ == "train_test_split":
                result = data_api.train_validation_test_split(
                    result,
                    train_size=float(p.get("train_size", 90)) / 100.0,
                    validation_size=float(p.get("validation_size", 5)) / 100.0,
                    test_size=float(p.get("test_size", 5)) / 100.0,
                    seed=int(p.get("seed", 42)),
                    shuffle=_bool(p.get("shuffle", True)),
                )

            elif typ == "tokenize_text":
                result = data_api.tokenize_text_dataset(
                    result,
                    tokenizer_name=p.get("tokenizer_name", "gpt2"),
                    text_column=p.get("text_column", "text"),
                    context_length=int(p.get("context_length", 512)),
                    truncation=_bool(p.get("truncation", True)),
                    padding=p.get("padding", False),
                    add_special_tokens=_bool(p.get("add_special_tokens", True)),
                )

            elif typ == "image_process":
                result = data_api.process_image_dataset(
                    result,
                    image_column=p.get("image_column", "image"),
                    width=int(p.get("width", 224)),
                    height=int(p.get("height", 224)),
                    mode=p.get("mode", "RGB"),
                    center_crop=_bool(p.get("center_crop", False)),
                )

            elif typ == "audio_process":
                result = data_api.process_audio_dataset(
                    result,
                    audio_column=p.get("audio_column", "audio"),
                    sample_rate=int(p.get("sample_rate", 16000)),
                    normalize=_bool(p.get("normalize", True)),
                    trim_silence=_bool(p.get("trim_silence", False)),
                    silence_threshold=float(p.get("silence_threshold", 0.01)),
                )

            elif typ == "batch_data":
                result = data_api.make_torch_dataloader(
                    result,
                    batch_size=int(p.get("batch_size", 16)),
                    shuffle=_bool(p.get("shuffle", True)),
                    num_workers=int(p.get("num_workers", 2)),
                    drop_last=_bool(p.get("drop_last", False)),
                )

            elif typ == "prepared_dataset":
                result = data_api.prepared_dataset_output(
                    result,
                    save_to_disk=_bool(p.get("save_to_disk", False)),
                    path=p.get("path", "mlbricks/data/prepared_dataset"),
                )

            else:
                raise ValueError(f"Unsupported data node type: {typ}")

        except Exception as exc:
            statuses[nid] = {"status": "error", "message": str(exc)}
            _emit(progress_callback, {
                "status": "error",
                "overall": round(index / max(total, 1) * 100),
                "step": index + 1,
                "total_steps": total,
                "current_node_id": nid,
                "message": f'{node["name"]} failed: {exc}',
                "nodes": statuses,
            })
            raise

        statuses[nid] = {"status": "done", "message": "Done"}
        _emit(progress_callback, {
            "status": "running" if index + 1 < total else "done",
            "overall": round((index + 1) / max(total, 1) * 100),
            "step": index + 1,
            "total_steps": total,
            "current_node_id": None if index + 1 == total else nid,
            "message": "Pipeline complete." if index + 1 == total else f'{node["name"]} complete.',
            "nodes": statuses,
        })

    return result
