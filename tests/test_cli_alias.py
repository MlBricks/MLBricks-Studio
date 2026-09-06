from __future__ import annotations

from pathlib import Path


def test_console_script_is_declared():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert '[project.scripts]' in text
    assert 'name = "mlbricks-studio"' in text
    assert 'mlbricks-studio = "mlbstudio.cli:main"' in text
    assert 'mlb-studio = "mlbstudio.cli:main"' in text


def test_cli_launches_local_app(monkeypatch):
    import mlbstudio.cli as cli

    called = {}

    class FakeBuilder:
        def app(self):
            called["app"] = True

    monkeypatch.setattr(cli, "Builder", FakeBuilder)
    cli.main()
    assert called == {"app": True}
