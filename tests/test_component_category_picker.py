from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_component_library_has_separate_category_picker():
    js = (ROOT / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    css = (ROOT / "frontend" / "src" / "builder.css").read_text(encoding="utf-8")

    assert 'libraryCategoryByWorkspace={model:"All Components",data:"All Components"}' in js
    assert '"ML Core","Deep Learning Core","Math & Tensor Ops"' in js
    assert '"COMPONENT CATEGORY"' in js
    assert '"My Modules / API"' in js
    assert ".mlb-library-category-box" in css


def test_built_frontend_contains_category_picker_and_foundation_categories():
    js = (ROOT / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")
    assert "COMPONENT CATEGORY" in js
    assert "ML Core" in js
    assert "Deep Learning Core" in js
    assert "Math & Tensor Ops" in js
