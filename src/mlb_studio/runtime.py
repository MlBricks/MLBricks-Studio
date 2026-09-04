from __future__ import annotations

import importlib
import importlib.metadata


def get_mlbricks_info():
    """Return installed MLBricks package information without namespace false positives.

    A plain directory named ``mlbricks`` can form an importable namespace package.
    Production diagnostics therefore verify distribution metadata first instead of
    treating any successful ``import mlbricks`` as an installed MLBricks release.
    """
    distribution = None
    for distribution_name in ("mlbricks-kit", "mlbricks"):
        try:
            distribution = importlib.metadata.distribution(distribution_name)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    if distribution is None:
        return {"installed": False, "version": None, "module_path": None}

    version = distribution.version
    try:
        module = importlib.import_module("mlbricks")
        module_path = getattr(module, "__file__", None)
    except Exception:
        module_path = None

    return {
        "installed": True,
        "version": version,
        "module_path": module_path,
    }
