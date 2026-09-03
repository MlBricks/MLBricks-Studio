from __future__ import annotations

import torch
import pytest

from mlb_studio.model_runtime import (
    ModelCompileError,
    _APIOperation,
    _lane_output,
    _named_output,
)


def _binding():
    return {
        "call_type": "user_function",
        "user_function_name": "custom_function",
        "user_code": (
            "def custom_function(x, gain):\n"
            "    return x * gain, x + gain\n"
        ),
        "port_mode": "extended",
        "auto_main_input": True,
        "input_ports": [
            {
                "id": "gain_in",
                "name": "gain",
                "parameter": "gain",
                "required": True,
                "positional": False,
                "side": "top",
                "order": 0,
            }
        ],
        "output_ports": [
            {
                "id": "sum_out",
                "name": "sum",
                "selector": "1",
                "side": "bottom",
                "order": 0,
            }
        ],
        "output_selector": "0",
        "parameters": [],
    }


def test_extended_custom_terminals_keep_universal_main_and_named_output():
    op = _APIOperation(
        binding=_binding(),
        params={},
        runtime={"allow_user_code": True},
        label="Custom Function",
    )
    x = torch.tensor([[1.0, 2.0]])
    result = op(x, named_inputs={"gain_in": torch.tensor(3.0)})

    # Universal Main output remains available.
    assert torch.equal(_lane_output(result, "main"), torch.tensor([[3.0, 6.0]]))
    # Extra output terminal independently maps the second return value.
    assert torch.equal(_named_output(result, "sum_out"), torch.tensor([[4.0, 5.0]]))


def test_required_custom_input_terminal_must_be_connected():
    op = _APIOperation(
        binding=_binding(),
        params={},
        runtime={"allow_user_code": True},
        label="Custom Function",
    )
    with pytest.raises(ModelCompileError, match="input terminal 'gain' is not connected"):
        op(torch.tensor([[1.0, 2.0]]), named_inputs={})


def test_tensor_graph_can_feed_universal_main_and_extra_terminal_together():
    from mlb_studio.model_runtime import TensorGraph

    source = {
        "id": "source",
        "type": "api_step",
        "name": "Source",
        "params": {},
        "api_binding": {
            "call_type": "user_function",
            "user_function_name": "source",
            "user_code": "def source(x):\n    return x + 1\n",
            "port_mode": "standard",
            "auto_main_input": True,
            "parameters": [],
        },
    }
    target_binding = _binding()
    target_binding["user_code"] = "def custom_function(x, gain):\n    return x + gain\n"
    target_binding["output_ports"] = []
    target_binding["output_selector"] = "auto"
    target = {
        "id": "target",
        "type": "api_step",
        "name": "Target",
        "params": {},
        "api_binding": target_binding,
    }
    graph = TensorGraph(
        nodes=[source, target],
        edges=[
            {"id": "e1", "source": "source", "target": "target", "kind": "main", "source_port": "main_out", "target_port": "main_in"},
            {"id": "e2", "source": "source", "target": "target", "kind": "named", "source_port": "main_out", "target_port": "named_in:gain_in"},
        ],
        custom_components={},
        runtime={"device": "cpu", "precision": "fp32", "backend": "pytorch", "allow_user_code": True},
    )
    x = torch.tensor([[1.0, 2.0]])
    assert torch.equal(graph(x), torch.tensor([[4.0, 6.0]]))


def test_tensor_graph_can_route_custom_output_terminal_into_fixed_main_input():
    from mlb_studio.model_runtime import TensorGraph

    binding = _binding()
    binding["user_code"] = "def custom_function(x, gain=2):\n    return x + 1, x * gain\n"
    binding["input_ports"][0]["required"] = False
    source = {"id": "source", "type": "api_step", "name": "Source", "params": {}, "api_binding": binding}
    sink = {
        "id": "sink",
        "type": "api_step",
        "name": "Sink",
        "params": {},
        "api_binding": {
            "call_type": "user_function",
            "user_function_name": "sink",
            "user_code": "def sink(x):\n    return x + 5\n",
            "port_mode": "standard",
            "auto_main_input": True,
            "parameters": [],
        },
    }
    graph = TensorGraph(
        nodes=[source, sink],
        edges=[
            {"id": "e1", "source": "source", "target": "sink", "kind": "main", "source_port": "named_out:sum_out", "target_port": "main_in"},
        ],
        custom_components={},
        runtime={"device": "cpu", "precision": "fp32", "backend": "pytorch", "allow_user_code": True},
    )
    x = torch.tensor([[1.0, 2.0]])
    assert torch.equal(graph(x), torch.tensor([[7.0, 9.0]]))


def test_fixed_visual_lanes_can_map_directly_to_function_arguments():
    binding = {
        "call_type": "user_function",
        "user_function_name": "mix",
        "user_code": "def mix(main_tensor, residual, mask):\n    return main_tensor + residual * mask\n",
        "port_mode": "extended",
        "auto_main_input": False,
        "parameters": [
            {"name": "main_tensor", "stage": "call", "source": "main", "positional": True, "required": True},
            {"name": "residual", "stage": "call", "source": "skip", "positional": False, "required": True},
            {"name": "mask", "stage": "call", "source": "extra", "positional": False, "required": True},
        ],
        "input_ports": [],
        "output_ports": [],
        "output_selector": "auto",
    }
    op = _APIOperation(
        binding=binding,
        params={},
        runtime={"allow_user_code": True},
        label="Visual Mapping",
    )
    main = torch.tensor([[1.0, 2.0]])
    top = torch.tensor([[10.0, 20.0]])
    bottom = torch.tensor([[0.5, 0.25]])
    assert torch.equal(op(main, skip=top, extra=bottom), torch.tensor([[6.0, 7.0]]))


def test_user_function_validation_returns_signature_for_visual_mapping():
    from mlb_studio.builder import Builder

    builder = Builder()
    result = builder.validate_user_function(
        "def combine(x, skip, scale=1.0, *, state=None):\n    return x\n",
        "combine",
    )
    assert result["ok"] is True
    assert result["signature"]["name"] == "combine"
    assert [item["name"] for item in result["signature"]["parameters"]] == ["x", "skip", "scale", "state"]
    assert [item["required"] for item in result["signature"]["parameters"]] == [True, True, False, False]


def test_fixed_visual_outputs_map_main_top_and_bottom_return_values():
    binding = {
        "call_type": "user_function",
        "user_function_name": "split",
        "user_code": "def split(x):\n    return x, x + 1, x + 2\n",
        "port_mode": "extended",
        "auto_main_input": True,
        "parameters": [],
        "input_ports": [],
        "output_ports": [],
        "multi_output": True,
        "output_map": {"main": "0", "skip": "1", "extra": "2"},
    }
    op = _APIOperation(
        binding=binding,
        params={},
        runtime={"allow_user_code": True},
        label="Visual Outputs",
    )
    x = torch.tensor([[2.0, 4.0]])
    result = op(x)
    assert torch.equal(_lane_output(result, "main"), x)
    assert torch.equal(_lane_output(result, "skip"), x + 1)
    assert torch.equal(_lane_output(result, "extra"), x + 2)
