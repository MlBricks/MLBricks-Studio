from __future__ import annotations

from pathlib import Path


def test_pyproject_has_studio_runtime_and_dev_dependencies():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"torch>=2.2"' in text
    assert '"pytest>=8,<10"' in text
    assert '"build>=1.2,<2"' in text


def test_unpublished_mlbricks_is_not_required():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"mlbricks==1.0.0"' not in text
    assert "git+https://github.com/MlBricks/MLBricks.git" not in text
