import base64
import gzip
from pathlib import Path

import mlb_studio.builder as builder_mod


def test_compressed_frontend_bundle_refreshes_when_assets_change(tmp_path, monkeypatch):
    static = tmp_path / "static"
    static.mkdir()
    css = static / "builder.css"
    js = static / "builder.js"
    css.write_text("body{color:red}", encoding="utf-8")
    js.write_text("window.TEST_ASSET='one';", encoding="utf-8")

    monkeypatch.setattr(builder_mod, "_STATIC", static)
    builder_mod.refresh_frontend_assets()

    first_css, first_js = builder_mod._compressed_frontend_bundle()
    assert gzip.decompress(base64.b64decode(first_css)).decode() == "body{color:red}"
    assert gzip.decompress(base64.b64decode(first_js)).decode() == "window.TEST_ASSET='one';"

    js.write_text("window.TEST_ASSET='two-two';", encoding="utf-8")
    second_css, second_js = builder_mod._compressed_frontend_bundle()

    assert second_css == first_css
    assert second_js != first_js
    assert gzip.decompress(base64.b64decode(second_js)).decode() == "window.TEST_ASSET='two-two';"


def test_refresh_frontend_assets_resets_process_emission_state(monkeypatch):
    monkeypatch.setattr(builder_mod, "_FRONTEND_ASSETS_EMITTED", True)
    monkeypatch.setattr(builder_mod, "_FRONTEND_ASSETS_EMITTED_SIGNATURE", (("builder.js", 1, 1),))
    monkeypatch.setattr(builder_mod, "_FRONTEND_BUNDLE_CACHE", ("css", "js"))
    monkeypatch.setattr(builder_mod, "_FRONTEND_BUNDLE_CACHE_SIGNATURE", (("builder.js", 1, 1),))

    builder_mod.refresh_frontend_assets()

    assert builder_mod._FRONTEND_ASSETS_EMITTED is False
    assert builder_mod._FRONTEND_ASSETS_EMITTED_SIGNATURE is None
    assert builder_mod._FRONTEND_BUNDLE_CACHE is None
    assert builder_mod._FRONTEND_BUNDLE_CACHE_SIGNATURE is None
