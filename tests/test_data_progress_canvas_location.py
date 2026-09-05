from pathlib import Path


def _static(name: str) -> str:
    return (Path(__file__).parents[1] / "src" / "mlb_studio" / "static" / name).read_text(encoding="utf-8")


def test_data_progress_lives_in_canvas_hud_not_toolbar():
    js = _static("builder.js")
    css = _static("builder.css")

    assert 'className="mlb-data-canvas-progress idle"' in js
    assert 'className="mlb-data-canvas-hud"' in js
    assert 'dataHud.append(dataProgress,mini)' in js
    assert 'canvas.appendChild(dataHud)' in js
    assert 'updateDataCanvasProgress(next)' in js
    assert '.mlb-data-canvas-progress{' in css
    assert '.mlb-data-canvas-progress.idle{display:none}' in css

    # Kernel health remains in the toolbar, but the old live progress pill is no
    # longer created/appended there, avoiding horizontal toolbar overflow.
    start = js.index('// Data runtime health/progress belongs only to the Data Processing workspace.')
    end = js.index('const tsp=document.createElement("div");tsp.className="mlb-toolspacer"', start)
    toolbar_section = js[start:end]
    assert 'toolbar.appendChild(kernel)' in toolbar_section
    assert 'toolbar.appendChild(live)' not in toolbar_section


def test_data_progress_is_parallel_with_blueprint():
    js = _static("builder.js")
    css = _static("builder.css")

    # The processing rail and minimap are siblings in the same HUD, in that order.
    assert 'dataHud.append(dataProgress,mini)' in js
    assert '.mlb-data-canvas-hud{' in css
    assert 'display:flex;align-items:center;justify-content:flex-end;gap:14px' in css
    assert '.mlb-data-canvas-hud>.mlb-minimap{' in css
    assert 'flex:0 0 132px;margin:0' in css
    assert 'width:min(64%,680px);min-width:420px' in css
