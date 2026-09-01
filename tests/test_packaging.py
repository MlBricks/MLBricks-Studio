from __future__ import annotations

from pathlib import Path


def test_pyproject_pins_mlbricks_and_has_dev_tests():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"mlbricks==1.0.0"' in text
    assert '"pytest>=8,<10"' in text
    assert "git+https://github.com/MlBricks/MLBricks.git" not in text
