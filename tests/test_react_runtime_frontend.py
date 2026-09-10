from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC_JS = ROOT / "src" / "mlb_studio" / "static" / "builder.js"
FRONTEND = ROOT / "frontend"


def test_react_frontend_sources_and_build_are_shipped():
    assert (FRONTEND / "package.json").exists()
    assert (FRONTEND / "build.mjs").exists()
    assert (FRONTEND / "src" / "react-runtime.js").exists()
    assert (FRONTEND / "src" / "legacy-builder.js").exists()
    assert (FRONTEND / "vendor" / "react.production.min.js").exists()
    assert (FRONTEND / "vendor" / "react-dom.production.min.js").exists()

    js = STATIC_JS.read_text(encoding="utf-8")
    assert "MLBricks Studio compiled frontend" in js
    assert "__MLBReactRuntime" in js
    assert "React runtime islands" in js
    assert "__MLB_REACT_BOOTSTRAP_SOURCE__" in js
    assert "__MLB_STUDIO_GET_JS_SOURCE__" in js


def test_training_and_generation_status_use_react_islands_for_hot_updates():
    legacy = (FRONTEND / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    assert 'mountReactRuntimeStatus(main,side,entry,mode)' in legacy
    assert 'refreshReactRuntimeStatus(entry,runtimePanel.mode)' in legacy
    assert 'if(refreshReactRuntimeStatus(entry,runtimePanel.mode))return;' in legacy
    assert 'React status islands own hot training/generation telemetry' in legacy
    assert 'if(!mountReactRuntimeStatus(main,side,entry,mode))' in legacy


def test_react_runtime_updates_slices_instead_of_rebuilding_studio_root():
    react_runtime = (FRONTEND / "src" / "react-runtime.js").read_text(encoding="utf-8")
    assert "class ConnectedSlice extends React.PureComponent" in react_runtime
    assert "store.subscribe" in react_runtime
    assert "updateStatus" in react_runtime
    assert "root.innerHTML" not in react_runtime


def test_runtime_status_has_render_error_boundary_and_payload_normalization():
    react_runtime = (FRONTEND / "src" / "react-runtime.js").read_text(encoding="utf-8")
    assert "class RuntimeErrorBoundary extends React.Component" in react_runtime
    assert "function asArray(value)" in react_runtime
    assert "function displayList(value,fallback)" in react_runtime
    assert "asArray(s.history)" in react_runtime
    assert "displayList(l.generation_algorithms,'Compatibility path')" in react_runtime
    assert "asArray(x.compat.checks).map" in react_runtime
    assert "h(RuntimeErrorBoundary,{store:store}" in react_runtime


def test_full_studio_draw_never_fails_to_blank_screen_silently():
    legacy = (FRONTEND / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    assert "function renderDrawRecovery(error)" in legacy
    assert "function drawUnsafe(force=false)" in legacy
    assert 'console.error("MLBricks Studio full view render failed",error)' in legacy
    assert 'retry.addEventListener("click",()=>{drawRecoveryAttempts=0;draw(true);});' in legacy
    assert "One automatic retry handles transient host/React timing faults" in legacy
    assert 'console.error("MLBricks Studio runtime status mount failed",error)' in legacy
    assert 'console.error("MLBricks Studio runtime status update failed",error)' in legacy
