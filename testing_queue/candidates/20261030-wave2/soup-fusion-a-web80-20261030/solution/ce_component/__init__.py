"""Materialized CE v3: reversible narrow loss hook and timeout-safe route evidence."""
import atexit
from importlib.metadata import version
import json

_handle = None
_stats = None
_report_registered = False


def diagnostics():
    return {} if _stats is None else _stats.snapshot()


def _report():
    print("SOUP_CE_V3_SUMMARY", json.dumps(diagnostics(), sort_keys=True), flush=True)


def apply():
    global _handle, _stats, _report_registered
    if _handle is not None and _handle.active:
        if _handle.model_class.__init__ is not _handle.hook:
            raise RuntimeError("CE constructor ownership changed after installation")
        return True
    expected = {"torch": "2.11.0", "transformers": "5.16.1", "triton": "3.6.0"}
    actual = {name: version(name).split("+", 1)[0] for name in expected}
    if actual != expected:
        print("SOUP_CE_V3_SKIPPED", json.dumps({"versions": actual}), flush=True)
        return False
    from .adapter import RouteStats, install_pinned_qwen2
    _stats = RouteStats()
    _handle = install_pinned_qwen2(_stats)
    if not _report_registered:
        atexit.register(_report)
        _report_registered = True
    print("SOUP_CE_V3_INSTALLED loss_boundary_only single_denominator_read", flush=True)
    return True


def restore():
    return {"constructor": "not_installed", "model_losses": {}} if _handle is None else _handle.restore()
