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


def test_data_processing_dependencies_are_installed_by_default():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    project_deps = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    for dep in (
        '"datasets>=2.18,<5"',
        '"kagglehub>=0.3,<1"',
        '"transformers>=4.40,<6"',
        '"pandas>=2,<4"',
        '"pyarrow>=15,<23"',
        '"pillow>=10,<13"',
        '"numpy>=1.26,<3"',
    ):
        assert dep in project_deps
    assert '"ipython>=7.34,<10"' in project_deps
