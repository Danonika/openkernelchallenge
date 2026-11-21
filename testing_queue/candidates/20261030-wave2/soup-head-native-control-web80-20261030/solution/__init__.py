"""Private-half native-CE head-memory research control. No original Triton share."""
from importlib.metadata import PackageNotFoundError, version
import json

_handle = None
_telemetry = None


def apply():
    global _handle, _telemetry
    if _handle is not None and _handle.active:
        if _handle.cls.__init__ is not _handle.hook:
            raise RuntimeError("Head constructor ownership changed after install")
        return True
    expected = {"torch": "2.11.0+cu128", "transformers": "5.16.1"}
    try:
        actual = {name: version(name) for name in expected}
    except PackageNotFoundError as exc:
        print("SOUP_HEAD_NATIVE_SKIPPED", str(exc), flush=True)
        return False
    if actual != expected:
        print("SOUP_HEAD_NATIVE_SKIPPED", json.dumps({"versions": actual}), flush=True)
        return False
    from .native_head import Telemetry
    from .install import install_pinned
    _telemetry = Telemetry()
    _handle = install_pinned(_telemetry)
    _telemetry.emit("installed", original_triton_kernels=[],
                    status="T4_UNTESTED_NATIVE_REPLAY_CONTROL")
    return True


def diagnostics():
    return {} if _telemetry is None else _telemetry.snapshot()


def restore():
    return {"constructor": "not_installed"} if _handle is None else _handle.restore()
