from __future__ import annotations

from mlb_studio.design_io import load_design_file, save_design_file
from mlb_studio.graph import new_project


def test_json_and_binary_design_roundtrip(tmp_path):
    state = new_project("Roundtrip")
    json_path = save_design_file(state, tmp_path / "project.mlbricks.json")
    bin_path = save_design_file(state, tmp_path / "project.mlbricks.bin")
    assert load_design_file(json_path) == state
    assert load_design_file(bin_path) == state
