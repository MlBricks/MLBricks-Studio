from __future__ import annotations

import json
from pathlib import Path

from mlb_studio.api_registry import COMPONENT_CHOICES


def _schema():
    path = Path(__file__).resolve().parents[1] / "src" / "mlb_studio" / "mlbricks_api_schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _field(component, key):
    fields = _schema()["components"][component]["parameters"]
    return next(item for item in fields if item["key"] == key)


def test_schema_targets_current_mlbricks_kit_beta():
    payload = _schema()
    assert payload["mlbricks_version"] == "1.0.0b1"
    assert payload["generated_from"] == "MLBricks Kit 1.0.0b1 public API"


def test_component_specific_position_and_scan_choices():
    assert COMPONENT_CHOICES["bolt"]["position"] == ["none", "rope"]
    assert COMPONENT_CHOICES["vesa"]["position"] == ["auto", "none", "2d_sincos", "learned"]
    assert COMPONENT_CHOICES["visualbolt"]["position"] == ["auto", "none", "2d_sincos", "learned"]
    assert COMPONENT_CHOICES["vesa"]["scan"] == ["cross", "horizontal", "vertical", "raster"]
    assert COMPONENT_CHOICES["visualbolt"]["scan"] == ["cross", "horizontal", "vertical", "raster"]

    assert _field("bolt", "position")["options"] == ["none", "rope"]
    for component in ("vesa", "visualbolt"):
        assert _field(component, "position")["options"] == ["auto", "none", "2d_sincos", "learned"]
        assert _field(component, "scan")["options"] == ["cross", "horizontal", "vertical", "raster"]
