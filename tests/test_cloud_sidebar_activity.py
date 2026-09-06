from pathlib import Path


def test_cloud_workspace_uses_right_sidebar_for_connection_and_transfer_activity():
    root = Path(__file__).resolve().parents[1]
    js = (root / "src/mlb_studio/static/builder.js").read_text(encoding="utf-8")
    css = (root / "src/mlb_studio/static/builder.css").read_text(encoding="utf-8")
    assert 'function renderCloudInspector(body)' in js
    assert 'TRANSFER ACTIVITY' in js
    assert 'Check Connection' in js
    assert 'Cancel' in js
    assert 'Retry' in js
    assert 'data-cloud-role="bar"' in js
    assert '.mlb-cloud-transfer-track' in css
    assert 'cloudWorkspace.open?"Cloud"' in js
    assert 'inspectorTab=cloudTerminal?"info":"settings";' in js
    assert 'if(inspectorTab==="info")renderCloudInfoInspector(body);' in js
    assert 'function renderCloudSettingsInspector(body)' in js
    assert 'renderCloudInspector(body);' in js
    assert 'mlb-cloud-result-message' in js
    assert 'height:116px' in css
    assert 'overflow-y:auto' in css


def test_cloud_backend_emits_staged_progress_and_target_metadata():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/mlb_studio/builder.py").read_text(encoding="utf-8")
    assert '"cloud_target": target' in source
    assert '"cloud_provider": provider' in source
    assert 'bytes_done=done' in source
    assert 'Packaging local content for upload' in source
    assert 'Downloading remote bundle' in source
    assert 'Cloud transfer cancelled.' in source


def test_huggingface_cloud_push_emits_connection_identity_and_progress(monkeypatch):
    from mlb_studio.builder import Builder
    from mlb_studio import hub

    builder = Builder()
    monkeypatch.setattr(hub, "resolve_repo_id", lambda repo_id, token=None: "DemoOrg/repo" if repo_id == "demo/repo" else repo_id)
    events = []
    monkeypatch.setattr(
        builder,
        "_cloud_provider_status",
        lambda provider, cloud: {
            "ok": True,
            "authenticated": True,
            "username": "demo-user",
            "message": "Connected as demo-user.",
        },
    )
    monkeypatch.setattr(
        builder,
        "push_project_to_hub",
        lambda repo_id, private=True, token=None: {
            "repo_id": repo_id,
            "url": f"https://example.invalid/{repo_id}",
        },
    )

    builder._execute_cloud_command(
        {
            "action": "cloud_push",
            "cloud": {
                "provider": "huggingface",
                "content_type": "project",
                "repo": "demo/repo",
                "revision": "main",
                "credentials": {"token": "secret"},
            },
        },
        events.append,
    )

    assert [event["overall"] for event in events] == [4, 10, 16, 22, 94, 100]
    assert events[1]["cloud_status"]["username"] == "demo-user"
    assert events[2]["phase"] == "namespace"
    assert events[2]["cloud_target"]["repository"] == "DemoOrg/repo"
    assert events[3]["cloud_target"]["repository"] == "DemoOrg/repo"
    assert events[-1]["status"] == "done"
