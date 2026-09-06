from pathlib import Path


def test_cloud_live_activity_stays_on_cloud_and_terminal_result_moves_to_info():
    root = Path(__file__).resolve().parents[1]
    js = (root / "src/mlb_studio/static/builder.js").read_text(encoding="utf-8")

    assert 'inspectorTab="settings";' in js
    assert 'const cloudTerminal=next.status==="done"||next.status==="error"||next.status==="stopped";' in js
    assert 'inspectorTab=cloudTerminal?"info":"settings";' in js
    assert 'function renderCloudSettingsInspector(body)' in js
    assert 'renderCloudInspector(body);' in js
    assert 'function renderCloudInfoInspector(body)' in js
    assert 'Completion, failure, cancellation and warning details will appear here automatically.' in js


def test_cloud_info_has_conditional_fixed_scrollable_message_box():
    root = Path(__file__).resolve().parents[1]
    js = (root / "src/mlb_studio/static/builder.js").read_text(encoding="utf-8")
    css = (root / "src/mlb_studio/static/builder.css").read_text(encoding="utf-8")

    assert 'if(message){' in js
    assert 'mlb-cloud-result-message-wrap' in js
    assert 'finalState.cls==="error"?"ERROR":(finalState.cls==="warn"?"WARNING":"MESSAGE")' in js
    assert '.mlb-cloud-result-message{' in css
    assert 'height:116px' in css
    assert 'max-height:116px' in css
    assert 'overflow-y:auto' in css
