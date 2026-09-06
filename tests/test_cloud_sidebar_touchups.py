from pathlib import Path


def test_secret_input_enables_save_credential_without_full_redraw():
    root = Path(__file__).resolve().parents[1]
    js = (root / "src/mlb_studio/static/builder.js").read_text(encoding="utf-8")
    assert "mlb-cloud-credential-save" in js
    assert 'if(secret){' in js
    assert 'save.disabled=!currentCredentialHasInput(cloudForm.provider)' in js
    assert 'dispatchDelay=(action==="persistence_save_credentials"||action==="persistence_delete_credentials")?10:120' in js


def test_live_cloud_progress_forces_visual_cloud_tab_after_routing_change():
    root = Path(__file__).resolve().parents[1]
    js = (root / "src/mlb_studio/static/builder.js").read_text(encoding="utf-8")
    assert "const previousCloudInspectorTab=inspectorTab;" in js
    assert 'inspectorTab=cloudTerminal?"info":"settings";' in js
    assert 'if(previousCloudInspectorTab!==inspectorTab)setTimeout(draw,0);' in js
    assert 'mlb-cloud-tab-info' in js
    assert 'mlb-cloud-tab-live' in js


def test_headless_keyring_probe_is_skipped_and_cached():
    root = Path(__file__).resolve().parents[1]
    py = (root / "src/mlb_studio/persistence.py").read_text(encoding="utf-8")
    assert "_KEYRING_BACKEND_CACHE" in py
    assert 'not os.environ.get("DBUS_SESSION_BUS_ADDRESS")' in py
    assert 'not os.environ.get("DISPLAY")' in py
