"""Original CE producer plus source-owned reusable metric consumer."""
import atexit,json
from .telemetry import Telemetry
from .producer.telemetry import RouteStats
_handle=None
_stats=Telemetry()
_ce_stats=RouteStats()
_report_registered=False

def apply():
    global _handle,_report_registered
    if _handle is not None and _handle.active:return True
    from .install import install_pinned
    _handle=install_pinned(_stats,_ce_stats)
    if not _report_registered:atexit.register(_report);_report_registered=True
    _stats.emit('installed',status='UNVALIDATED_COMBINED_CE_METRIC_CONSUMER')
    return True

def diagnostics():
    if _handle is not None:_handle.flush()
    return {'consumer':_stats.snapshot(),'producer':_ce_stats.snapshot()}

def _report():print('SOUP_CE_METRIC_SUMMARY '+json.dumps(diagnostics(),sort_keys=True),flush=True)

def restore():return None if _handle is None else _handle.restore()
