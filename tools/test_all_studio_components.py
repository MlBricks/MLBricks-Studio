from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlb_studio import Builder  # noqa: E402
from mlb_studio.api_graph_runtime import API_COMPONENTS  # noqa: E402
from mlb_studio.graph import primitive_catalog  # noqa: E402
from mlb_studio.import_pool import COMPONENT_IMPORTS  # noqa: E402
from mlb_studio.runner import EXECUTABLE_TYPES  # noqa: E402

# Component capability groups in the current Studio runtime.
DATA_RUNNER = set(EXECUTABLE_TYPES)

# Runtime operations execute end-to-end in Studio, but are intentionally not
# differentiable TensorGraph layers. ElasticBit is the first such operation.
RUNTIME_OPERATION = {"elasticbit_runtime"}

MODEL_RUNTIME = {
    "text_input", "image_input", "audio_input", "text_output", "logits_output", "value_buffer", "dropout",
    "classifier",
    "embedding", "lm_head", "learned_position", "sinusoidal_position",
    "esa", "stateaware_esa_stack", "soup", "rmsnorm", "layernorm", "linear",
    "ffn", "residual",
    *tuple(API_COMPONENTS._contracts.keys()),
}

# These cards are valid Studio components but do not currently have a complete
# execution path in the causal-LM compiler/data runner. They are release-gate
# items if the goal is "every visible component executes end-to-end".
LIMITED_EXECUTION = {
}


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def run_pytest() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(output.strip().splitlines()[-8:])
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "tail": tail}


def js_syntax_check() -> dict:
    node = shutil.which("node")
    if not node:
        return {"ok": True, "skipped": True, "message": "Node.js not installed; JS syntax check skipped."}
    target = ROOT / "src" / "mlb_studio" / "static" / "builder.js"
    proc = subprocess.run([node, "--check", str(target)], text=True, capture_output=True)
    return {
        "ok": proc.returncode == 0,
        "skipped": False,
        "message": (proc.stdout + proc.stderr).strip(),
    }


def audit(*, eager_mlbricks: bool = False) -> dict:
    builder = Builder()
    catalog = builder.catalog
    by_type = {item["type"]: item for item in catalog}

    checks = []
    checks.append({
        "name": "Catalog has 41 unique components",
        "ok": len(catalog) == 41 and len(by_type) == 41,
        "detail": f"catalog={len(catalog)}, unique={len(by_type)}",
    })

    missing_metadata = []
    duplicate_api_keys = []
    for item in catalog:
        typ = item["type"]
        for key in ("name", "icon", "category", "description", "accent"):
            if not str(item.get(key) or "").strip():
                missing_metadata.append(f"{typ}.{key}")
        keys = [field.get("key") for field in item.get("api") or []]
        if len(keys) != len(set(keys)):
            duplicate_api_keys.append(typ)

    checks.append({
        "name": "Every component has complete card metadata",
        "ok": not missing_metadata,
        "detail": ", ".join(missing_metadata) or "all complete",
    })
    checks.append({
        "name": "Inspector/API fields have unique keys",
        "ok": not duplicate_api_keys,
        "detail": ", ".join(duplicate_api_keys) or "all unique",
    })

    import_report = builder.validate_component_imports(eager=eager_mlbricks)
    checks.append({
        "name": "Every component has a Studio/import-pool route",
        "ok": bool(import_report.get("ok")),
        "detail": json.dumps(import_report.get("failures") or [], default=str),
    })

    html = builder._html(include_assets=False)
    missing_html = [typ for typ, item in by_type.items() if typ not in html or item["name"] not in html]
    checks.append({
        "name": "Every component serializes into notebook frontend state",
        "ok": not missing_html,
        "detail": ", ".join(missing_html) or "all 41 present",
    })

    js = js_syntax_check()
    checks.append({
        "name": "builder.js syntax",
        "ok": bool(js["ok"]),
        "detail": js.get("message") or ("skipped" if js.get("skipped") else "valid"),
        "skipped": bool(js.get("skipped")),
    })

    component_rows = []
    for item in catalog:
        typ = item["type"]
        if typ in DATA_RUNNER:
            execution = "data-runner"
            execution_ok = True
            note = "Executable through Data Processing runner."
        elif typ in RUNTIME_OPERATION:
            execution = "runtime-op"
            execution_ok = True
            note = "Executable as a post-training/inference Studio runtime operation."
        elif typ in MODEL_RUNTIME:
            execution = "model-runtime"
            execution_ok = True
            note = "Has a TensorGraph/runtime execution path."
        elif typ in LIMITED_EXECUTION:
            execution = "limited"
            execution_ok = False
            note = LIMITED_EXECUTION[typ]
        else:
            execution = "unknown"
            execution_ok = False
            note = "No execution classification registered."

        if item.get("builder_utility"):
            import_mode = "builder"
            import_ok = True
        elif typ == "stateaware_esa_stack":
            import_mode = "compound"
            import_ok = True
        else:
            import_mode = "mlbricks"
            import_ok = typ in COMPONENT_IMPORTS

        component_rows.append({
            "type": typ,
            "name": item["name"],
            "category": item["category"],
            "catalog": True,
            "inspector_fields": len(item.get("api") or []),
            "import_mode": import_mode,
            "import_route": import_ok,
            "execution": execution,
            "execution_ok": execution_ok,
            "note": note,
        })

    limited = [row for row in component_rows if not row["execution_ok"]]
    return {
        "ok": all(check["ok"] for check in checks),
        "catalog_count": len(catalog),
        "checks": checks,
        "components": component_rows,
        "limited_execution": limited,
        "mlbricks_installed": importlib.util.find_spec("mlbricks") is not None,
        "eager_mlbricks": eager_mlbricks,
    }


def print_report(report: dict, pytest_result: dict | None, *, strict_runtime: bool):
    print("=" * 92)
    print("MLB STUDIO COMPONENT RELEASE GATE")
    print("=" * 92)
    for check in report["checks"]:
        marker = "SKIP" if check.get("skipped") else status(check["ok"])
        print(f"{marker:5}  {check['name']}")
        if not check["ok"] and check.get("detail"):
            print(f"       {check['detail']}")

    print("\nCOMPONENT MATRIX")
    print("-" * 92)
    print(f"{'STATUS':7} {'TYPE':24} {'EXECUTION':15} {'IMPORT':10} NAME")
    print("-" * 92)
    for row in report["components"]:
        marker = "PASS" if row["execution_ok"] else "LIMIT"
        print(
            f"{marker:7} {row['type'][:24]:24} {row['execution'][:15]:15} "
            f"{row['import_mode'][:10]:10} {row['name']}"
        )

    print("\nSUMMARY")
    print("-" * 92)
    passed = sum(1 for row in report["components"] if row["execution_ok"])
    limited = len(report["limited_execution"])
    print(f"Catalog components: {report['catalog_count']}")
    print(f"Components with an execution path: {passed}")
    print(f"Components with limited/no end-to-end execution path: {limited}")
    print(f"MLBricks installed: {report['mlbricks_installed']}")

    if report["limited_execution"]:
        print("\nCURRENT EXECUTION LIMITATIONS")
        for row in report["limited_execution"]:
            print(f"- {row['type']}: {row['note']}")

    if pytest_result is not None:
        print("\nPYTEST")
        print("-" * 92)
        print(status(pytest_result["ok"]))
        print(pytest_result["tail"])

    static_ok = report["ok"] and (pytest_result is None or pytest_result["ok"])
    strict_ok = static_ok and (not strict_runtime or not report["limited_execution"])
    print("\nRELEASE GATE:", "PASS" if strict_ok else "FAIL")
    if static_ok and strict_runtime and report["limited_execution"]:
        print("Reason: --strict-runtime requires every visible component to have an end-to-end execution path.")
    return strict_ok


def main():
    parser = argparse.ArgumentParser(description="Audit every MLB Studio component before release.")
    parser.add_argument("--eager-mlbricks", action="store_true", help="Actually import every MLBricks-backed component.")
    parser.add_argument("--strict-runtime", action="store_true", help="Fail if any visible component lacks end-to-end execution support.")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip the full pytest suite.")
    parser.add_argument("--json", dest="json_path", help="Write the machine-readable audit report to this path.")
    args = parser.parse_args()

    report = audit(eager_mlbricks=args.eager_mlbricks)
    pytest_result = None if args.skip_pytest else run_pytest()

    if args.json_path:
        payload = {**report, "pytest": pytest_result}
        Path(args.json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ok = print_report(report, pytest_result, strict_runtime=args.strict_runtime)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
