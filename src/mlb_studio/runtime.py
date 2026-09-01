from __future__ import annotations
import importlib
import importlib.metadata

def get_mlbricks_info():
    try:
        module = importlib.import_module("mlbricks")
    except Exception:
        return {"installed":False,"version":None,"module_path":None}
    try:
        version = importlib.metadata.version("mlbricks")
    except Exception:
        version = getattr(module, "__version__", None)
    return {
        "installed":True,
        "version":version,
        "module_path":getattr(module, "__file__", None)
    }
