from pathlib import Path


def test_generation_controls_are_locked_while_training_is_live():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "mlb_studio"
        / "static"
        / "builder.js"
    ).read_text(encoding="utf-8")

    helper_start = source.index("function trainingLocksGeneration")
    helper_end = source.index("function startGenerationFromRuntime", helper_start)
    helper = source[helper_start:helper_end]
    assert "trainingIsRunning()" in helper
    assert 'liveStatus==="starting"' in helper
    assert 'liveStatus==="running"' in helper
    assert 'liveStatus==="stopping"' in helper

    runtime_start = source.index("function requestRuntimeCommand")
    runtime_end = source.index("function requestLocalCommand", runtime_start)
    runtime_block = source[runtime_start:runtime_end]
    assert 'action==="generate"&&trainingLocksGeneration(entry)' in runtime_block

    panel_start = source.index("function openRuntimePanel")
    panel_end = source.index("function requestBuiltModelTraining", panel_start)
    panel_block = source[panel_start:panel_end]
    assert 'mode==="generate"&&trainingLocksGeneration(entry)' in panel_block

    assert 'gen.disabled=trainingLocked;' in source
    assert 'run.disabled=trainingLocked;' in source
    assert 'Model runtime is disabled while training is running' in source
