"""Native shared-cast control: zero original Triton ownership, no CE replay."""
from .telemetry import Telemetry
_stats=Telemetry()
_handle=None

def apply():
    global _handle
    if _handle is not None and _handle.active:return True
    try:
        from .install import install_pinned
        _handle=install_pinned(_stats)
    except (ImportError,RuntimeError,AttributeError,ValueError,OSError) as exc:
        _stats.emit("installation_unavailable",reason=type(exc).__name__+": "+str(exc))
        return False
    _stats.emit("installed",status="UNVALIDATED_NATIVE_SHARED_CAST_CONTROL",original_triton_kernels=[])
    return True

def diagnostics():return _stats.snapshot()

def restore():
    global _handle
    return False if _handle is None else _handle.restore()
