from pathlib import Path


def test_manifest_excludes_frontend_sources_and_includes_built_static():
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "prune frontend" in manifest
    assert "recursive-include frontend" not in manifest
    assert 'mlb_studio = ["static/*.js", "static/*.css", "static/*.svg", "static/*.png", "static/*.ico", "*.json"]' in pyproject
    assert (root / "src/mlb_studio/static/builder.js").is_file()
    assert (root / "src/mlb_studio/static/builder.css").is_file()
