from pathlib import Path


def _builder_js() -> str:
    return (Path(__file__).parents[1] / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")


def test_cloud_footer_uses_short_status_instead_of_provider_message():
    js = _builder_js()
    assert 'function cloudBottomStatus(activity=cloudActivity)' in js
    assert 'execution.runtime_kind==="cloud"?cloudBottomStatus(cloudActivity):(execution.message||status)' in js
    assert 'setStatus(cloudBottomStatus(cloudActivity));' in js


def test_cloud_footer_has_terminal_summary_states():
    js = _builder_js()
    assert 'return action+": Running";' in js
    assert 'return action+": Completed";' in js
    assert 'return action+": Failed";' in js
    assert 'return action+": Cancelled";' in js
