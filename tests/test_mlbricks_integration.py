from __future__ import annotations

import importlib.metadata
import inspect

import pytest


mlbricks = pytest.importorskip("mlbricks", reason="MLBricks integration dependency is not installed in the source-test environment")

from mlb_studio import Builder  # noqa: E402
from mlb_studio.api_registry import refresh_component_api  # noqa: E402
from mlb_studio.import_pool import API_IMPORTS, COMPONENT_IMPORTS, IMPORT_POOL  # noqa: E402


EXPECTED_DISTRIBUTION = "mlbricks-kit"
EXPECTED_VERSION = "1.0.0b1"


@pytest.mark.integration
def test_pinned_mlbricks_kit_release_and_all_api_resolution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert importlib.metadata.version(EXPECTED_DISTRIBUTION) == EXPECTED_VERSION
    assert mlbricks.__version__ == EXPECTED_VERSION

    builder = Builder()
    for component in COMPONENT_IMPORTS:
        status = builder.ensure_component_import(component)
        assert status.get("ok"), status

    for key in API_IMPORTS:
        assert IMPORT_POOL.resolve_api(key) is not None


@pytest.mark.integration
def test_bolt_position_choices_match_current_api():
    meta = refresh_component_api("bolt")
    position = next(field for field in meta["parameters"] if field["key"] == "position")
    assert position["options"] == ["none", "rope"]

    mlbricks.Bolt(32, 4, position="rope")
    with pytest.raises(ValueError):
        mlbricks.Bolt(32, 4, position="auto")


@pytest.mark.integration
@pytest.mark.parametrize(
    "component_type, config_cls",
    [
        ("vesa", mlbricks.VesaConfig),
        ("visualbolt", mlbricks.VisionBoltConfig),
    ],
)
def test_vision_config_schema_matches_current_api(component_type, config_cls):
    meta = refresh_component_api(component_type)
    fields = {field["key"]: field for field in meta["parameters"]}

    assert fields["position"]["options"] == ["auto", "none", "2d_sincos", "learned"]
    assert fields["scan"]["options"] == ["cross", "horizontal", "vertical", "raster"]

    signature_keys = [
        name
        for name, parameter in inspect.signature(config_cls).parameters.items()
        if parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    assert list(fields) == signature_keys

    config_cls(position="2d_sincos", scan="horizontal")
    config_cls(position="learned", scan="vertical")
    with pytest.raises(ValueError):
        config_cls(position="rope")
    with pytest.raises(ValueError):
        config_cls(scan="serpentine")


@pytest.mark.integration
def test_source_defined_adapters_still_match_current_mlbricks_surface():
    lm = refresh_component_api("lm_head")
    assert [field["key"] for field in lm["parameters"]] == [
        "hidden_size", "vocab_size", "bias", "tie_embeddings", "device", "dtype"
    ]
    assert "tie_to" in str(inspect.signature(mlbricks.LMHead))
    assert hasattr(mlbricks.LMHead, "tie_weights")

    elastic = refresh_component_api("elasticbit_runtime")
    assert [field["key"] for field in elastic["parameters"]] == [
        "threshold", "min_bits", "max_bits", "runtime_mode"
    ]
    assert hasattr(mlbricks.ElasticBit, "bitsAnaliser")
    assert hasattr(mlbricks.ElasticBit, "RuntimeMatrix")
    assert hasattr(mlbricks.ElasticBit, "native_runtime_available")
