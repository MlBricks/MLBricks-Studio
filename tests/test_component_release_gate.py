from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlb_studio import Builder
from mlb_studio.api_graph_runtime import API_COMPONENTS
from mlb_studio.graph import (
    new_project,
    primitive_catalog,
    soup_30m_1l_project,
    soup_200m_project,
    esa_200m_project,
    slm_200m_project,
    stateaware_esa_200m_project,
    tinystories_30m_project,
)
from mlb_studio.import_pool import COMPONENT_IMPORTS
from mlb_studio.runner import EXECUTABLE_TYPES, SOURCE_TYPES, validate_data_pipeline


EXPECTED_COMPONENT_TYPES = {
    "text_input", "image_input", "audio_input",
    "hf_dataset", "kaggle_dataset", "url_dataset", "local_dataset",
    "text_process", "train_test_split", "tokenize_text", "manual_dataset",
    "image_process", "audio_process", "batch_data", "prepared_dataset",
    "embedding", "esa", "layer_block", "soup", "stateaware_esa_stack", "vesa", "rmsnorm",
    "ffn", "saffn", "residual", "dropout", "bolt", "visualbolt",
    "value_buffer", "linear", "layernorm", "rescontroller", "micro_ffn",
    "virtual_saffn", "elasticbit_runtime", "rope", "learned_position",
    "sinusoidal_position", "lm_head", "classifier", "text_output", "logits_output",
}

VALID_FIELD_TYPES = {
    "text", "textarea", "number", "select", "bool", "percent",
    "dataset_select", "dataset_split_select",
}

VALID_RUNTIME_SOCKETS = {
    "top", "back", "front", "bottom", "top_aux", "bottom_aux",
    "left", "right",
}


def _catalog_by_type():
    return {item["type"]: item for item in primitive_catalog()}


def _topological_ok(component):
    nodes = component.get("nodes") or []
    edges = component.get("edges") or []
    ids = {n["id"] for n in nodes}
    incoming = {nid: 0 for nid in ids}
    outgoing = {nid: [] for nid in ids}
    for edge in edges:
        a, b = edge.get("source"), edge.get("target")
        assert a in ids, f"edge source {a!r} is missing"
        assert b in ids, f"edge target {b!r} is missing"
        outgoing[a].append(b)
        incoming[b] += 1
    queue = [nid for nid, degree in incoming.items() if degree == 0]
    seen = []
    while queue:
        nid = queue.pop(0)
        seen.append(nid)
        for nxt in outgoing[nid]:
            incoming[nxt] -= 1
            if incoming[nxt] == 0:
                queue.append(nxt)
    return len(seen) == len(ids)


def test_release_gate_catalog_has_exactly_the_42_supported_studio_components():
    catalog = primitive_catalog()
    types = [item.get("type") for item in catalog]
    assert len(catalog) == 42
    assert len(types) == len(set(types))
    assert set(types) == EXPECTED_COMPONENT_TYPES


@pytest.mark.parametrize("component_type", sorted(EXPECTED_COMPONENT_TYPES))
def test_every_component_card_has_valid_metadata_and_api_schema(component_type):
    item = _catalog_by_type()[component_type]
    for key in ("name", "icon", "category", "description", "accent"):
        assert str(item.get(key) or "").strip(), f"{component_type} missing {key}"

    fields = item.get("api") or []
    keys = [field.get("key") for field in fields]
    assert len(keys) == len(set(keys)), f"{component_type} has duplicate API keys"
    for field in fields:
        assert str(field.get("key") or "").strip()
        assert str(field.get("label") or "").strip()
        assert field.get("type") in VALID_FIELD_TYPES
        if field.get("type") == "select":
            options = field.get("options") or []
            assert options, f"{component_type}.{field['key']} select has no options"
            if field.get("value") is not None:
                assert field.get("value") in options, (
                    f"{component_type}.{field['key']} default {field.get('value')!r} "
                    f"is not in {options!r}"
                )


def test_builder_catalog_uses_canonical_mlbricks_schema_for_schema_backed_components():
    builder = Builder()
    schema_path = Path(__file__).parents[1] / "src" / "mlb_studio" / "mlbricks_api_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))["components"]
    by_type = {item["type"]: item for item in builder.catalog}

    for component_type, api in schema.items():
        assert component_type in by_type
        builder_keys = [field.get("key") for field in by_type[component_type].get("api") or []]
        schema_keys = [field.get("key") for field in api.get("parameters") or []]
        assert builder_keys == schema_keys, component_type


def test_every_non_utility_component_has_an_import_route_or_declared_compound_runtime():
    builder = Builder()
    report = builder.validate_component_imports(eager=False)
    assert report["ok"], report["failures"]
    reported = {row["component_type"] for row in report["components"]}
    assert reported == EXPECTED_COMPONENT_TYPES

    by_type = {item["type"]: item for item in builder.catalog}
    for component_type, item in by_type.items():
        if item.get("builder_utility"):
            continue
        if component_type == "stateaware_esa_stack":
            continue
        assert component_type in COMPONENT_IMPORTS


def test_runtime_port_descriptions_are_unique_and_use_known_visual_sockets():
    for item in primitive_catalog():
        ports = item.get("runtime_ports") or {}
        for direction in ("inputs", "outputs"):
            seen = set()
            for port in ports.get(direction) or []:
                port_id = str(port.get("id") or "")
                assert port_id
                assert port_id not in seen
                seen.add(port_id)
                assert str(port.get("name") or "").strip()
                sockets = []
                if port.get("socket"):
                    sockets.append(port["socket"])
                sockets.extend(port.get("sockets") or [])
                assert sockets, f"{item['type']}.{direction}.{port_id} has no visual socket"
                assert set(sockets) <= VALID_RUNTIME_SOCKETS


def test_declarative_runtime_contracts_match_catalog_named_ports():
    by_type = _catalog_by_type()
    for component_type in ("saffn", "virtual_saffn"):
        contract = API_COMPONENTS.get(component_type)
        assert contract is not None
        item = by_type[component_type]
        declared_inputs = {p["id"] for p in item["runtime_ports"]["inputs"]}
        declared_outputs = {p["id"] for p in item["runtime_ports"]["outputs"]}
        assert set(contract.input_ports) == declared_inputs
        assert set(contract.output_ports) == declared_outputs


def test_all_component_cards_serialize_into_notebook_html():
    builder = Builder()
    html = builder._html(include_assets=False)
    assert len(html) > 10_000
    for item in builder.catalog:
        assert item["type"] in html
        assert item["name"] in html


@pytest.mark.parametrize(
    "factory",
    [
        new_project,
        tinystories_30m_project,
        esa_200m_project,
        stateaware_esa_200m_project,
        soup_200m_project,
        soup_30m_1l_project,
    ],
)
def test_every_builtin_project_has_valid_node_types_edges_and_acyclic_components(factory):
    state = factory()
    known = EXPECTED_COMPONENT_TYPES | {"custom", "api_step"}
    components = state.get("components") or {}
    assert components
    for component_id, component in components.items():
        for node in component.get("nodes") or []:
            assert node.get("type") in known, (component_id, node)
        assert _topological_ok(component), component_id


def test_default_data_workspace_is_a_valid_executable_pipeline():
    state = new_project()
    order, errors = validate_data_pipeline(state)
    assert not errors
    assert order
    assert order[0]["type"] in SOURCE_TYPES
    assert order[-1]["type"] == "prepared_dataset"
    assert {node["type"] for node in order} <= EXECUTABLE_TYPES


def test_catalog_json_roundtrip_preserves_all_component_defaults():
    catalog = primitive_catalog()
    restored = json.loads(json.dumps(catalog))
    assert restored == catalog


def test_release_slm_presets_have_requested_depths_and_names():
    slm50 = tinystories_30m_project()
    assert slm50["project"]["name"] == "50M SLM"
    assert slm50["project"]["estimated_parameters"] == "~50M"
    model50 = slm50["components"][slm50["root_component_id"]]
    layers50 = [n for n in model50["nodes"] if n.get("type") == "layer_block"]
    assert len(layers50) == 10
    assert all(n["params"]["dim"] == 480 for n in layers50)
    assert all(n["params"]["ffn_dim"] == 1920 for n in layers50)

    soup50 = soup_30m_1l_project()
    assert soup50["project"]["name"] == "50M SLM · SOUP"
    soup50_node = next(n for n in soup50["components"][soup50["root_component_id"]]["nodes"] if n.get("type") == "soup")
    assert soup50_node["params"]["depth"] == 2

    slm200 = esa_200m_project()
    assert slm200["project"]["name"] == "200M SLM"
    assert slm200["project"]["model_settings"]["embedding_size"] == 1024
    assert slm200["project"]["model_settings"]["heads"] == 16
    model200 = slm200["components"][slm200["root_component_id"]]
    layers200 = [n for n in model200["nodes"] if n.get("type") == "layer_block"]
    assert len(layers200) == 12
    assert any(n.get("type") == "learned_position" and n["params"]["dim"] == 1024 for n in model200["nodes"])
    head200 = next(n for n in model200["nodes"] if n.get("type") == "lm_head")
    assert head200["params"]["hidden_size"] == 1024
    assert head200["params"]["vocab_size"] == 50257
    assert head200["params"]["tie_embeddings"] is True
    assert all(n["params"]["dim"] == 1024 for n in layers200)
    assert all(n["params"]["heads"] == 16 for n in layers200)
    assert all(n["params"]["ffn_dim"] == 4096 for n in layers200)
    # Every adjacent Layer Block pair carries both explicit graph lanes.
    edges200 = model200["edges"]
    for left, right in zip(layers200[:-1], layers200[1:]):
        pair = [e for e in edges200 if e.get("source") == left["id"] and e.get("target") == right["id"]]
        assert {e.get("source_port") for e in pair} == {"named_out:signal", "named_out:residual"}
        assert {e.get("target_port") for e in pair} == {"named_in:signal", "named_in:residual"}
    assert slm_200m_project()["project"]["name"] == "200M SLM"

    stateaware200 = stateaware_esa_200m_project()
    assert stateaware200["project"]["name"] == "200M SLM · StateAware"
    stateaware_node = next(n for n in stateaware200["components"][stateaware200["root_component_id"]]["nodes"] if n.get("type") == "stateaware_esa_stack")
    assert stateaware_node["params"]["layers"] == 12

    soup200 = soup_200m_project()
    assert soup200["project"]["name"] == "200M SLM · SOUP"
    soup200_node = next(n for n in soup200["components"][soup200["root_component_id"]]["nodes"] if n.get("type") == "soup")
    assert soup200_node["params"]["depth"] == 3
