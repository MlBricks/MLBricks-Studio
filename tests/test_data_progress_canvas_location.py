from pathlib import Path


def _static(name: str) -> str:
    return (Path(__file__).parents[1] / "src" / "mlb_studio" / "static" / name).read_text(encoding="utf-8")


def test_data_progress_lives_above_canvas_cards_not_toolbar():
    js = _static("builder.js")
    css = _static("builder.css")

    assert 'className="mlb-data-canvas-progress idle"' in js
    assert 'canvas.appendChild(dataProgress)' in js
    assert 'updateDataCanvasProgress(next)' in js
    assert '.mlb-data-canvas-progress{' in css
    assert '.mlb-data-canvas-progress.idle{display:none}' in css

    # Kernel health remains in the toolbar, but the old live progress pill is no
    # longer created/appended there, avoiding horizontal toolbar overflow.
    toolbar_section = js[js.index('// Data runtime health/progress belongs only to the Data Processing workspace.'):js.index('const tsp=document.createElement("div");tsp.className="mlb-toolspacer"', js.index('// Data runtime health/progress belongs only to the Data Processing workspace.'))]
    assert 'toolbar.appendChild(kernel)' in toolbar_section
    assert 'toolbar.appendChild(live)' not in toolbar_section
