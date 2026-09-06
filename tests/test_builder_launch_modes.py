import json
import urllib.request


def test_public_mlbstudio_alias_imports_builder():
    from mlbstudio import Builder
    from mlb_studio import Builder as LegacyBuilder

    assert Builder is LegacyBuilder


def test_web_is_explicit_notebook_entrypoint(monkeypatch):
    from mlbstudio import Builder

    builder = Builder()
    called = []
    monkeypatch.setattr(builder, "_ipython_display_", lambda: called.append(True))

    assert builder.web() is None
    assert called == [True]


def test_local_app_serves_full_page_without_full_window_control():
    from mlbstudio import Builder

    builder = Builder()
    url = builder.app(open_browser=False, block=False)
    try:
        page = urllib.request.urlopen(url, timeout=5).read().decode("utf-8")
        assert "MLB Studio" in page
        assert '"allow_full_window":false' in page or "allow_full_window&quot;:false" in page
        assert "/api/run" in page
        assert "/api/progress" in page
        assert urllib.request.urlopen(url + "favicon.svg", timeout=5).status == 200
    finally:
        builder.stop_app()


def test_local_app_python_bridge_runs_commands():
    from mlbstudio import Builder

    builder = Builder()
    url = builder.app(open_browser=False, block=False)
    try:
        body = json.dumps({
            "state_raw": "{}",
            "command_raw": json.dumps({"action": "persistence_list"}),
        }).encode("utf-8")
        request = urllib.request.Request(
            url + "api/run",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        assert urllib.request.urlopen(request, timeout=5).status == 202

        import time
        result = None
        for _ in range(50):
            raw = urllib.request.urlopen(url + "api/progress", timeout=5).read().decode("utf-8")
            result = json.loads(raw)
            if result.get("status") in {"done", "error", "stopped"}:
                break
            time.sleep(0.05)

        assert result is not None
        assert result.get("status") == "done"
        assert result.get("runtime_kind") == "persistence"
    finally:
        builder.stop_app()
