"""Full-M original CE/NF4/RMS plus compact MLP and attested lossless frozen storage.

Starter math receives ZERO original-kernel credit. This retention mechanism is
also present in the user's activation_storage.py; novelty is not claimed.
"""
import atexit,json
from .telemetry import Telemetry
_handle=None
_support=None
_mlp=None
_storage=None
_stats=Telemetry()
_registered=False


def _restore_owned():
    results={};errors={}
    for name,handle in (('frozen_storage',_storage),('compact_mlp',_mlp),('support',_support),('head',_handle)):
        if handle is None:results[name]=None;continue
        try:results[name]=handle.restore()
        except BaseException as error:results[name]=False;errors[name]=repr(error)
    return results,errors


def apply():
    global _handle,_support,_mlp,_storage,_registered
    if _handle is not None and _handle.active:
        if (_support is None or not _support.intact() or _mlp is None or not _mlp.intact()
                or _storage is None or not _storage.intact()):
            raise RuntimeError('Composition auxiliary owners changed after installation')
        return True
    from .streaming_boundary import install_pinned
    from .token_chunk_head import token_chunk_boundary
    from .support_install import install_support
    from .compact_mlp_install import install_compact_mlp
    from .frozen_storage import install_storage
    _handle=None;_support=None;_mlp=None;_storage=None
    try:
        _handle=install_pinned(token_chunk_boundary,chunk=2048,observer=_stats.emit)
        _support=install_support()
        _mlp=install_compact_mlp()
        _storage=install_storage()
    except BaseException as original_error:
        restored,errors=_restore_owned()
        _handle=None;_support=None;_mlp=None;_storage=None
        if errors or any(value is False for value in restored.values()):
            raise RuntimeError('Composition rollback incomplete: '+repr({'restored':restored,'errors':errors})) from original_error
        raise
    if not _registered:atexit.register(_report);_registered=True
    _stats.emit('installed',status='UNVALIDATED_BATCHED_CE_STORAGE_SHARED_CAST_EDGES',nf4=True,rmsnorm=True,
                compact_mlp=True,frozen_storage=True,starter_original_credit=0,storage_original_credit=0)
    return True


def diagnostics():
    if _handle is not None:_handle.flush()
    result=_stats.snapshot();result['support']={} if _support is None else _support.status()
    result['compact_mlp']={} if _mlp is None else _mlp.status()
    result['frozen_storage']={} if _storage is None else _storage.status()
    from .token_chunk_head import batch_ce_status
    result['batch_ce']=batch_ce_status()
    return result


def _report():
    diagnostic=diagnostics()
    print('SOUP_STREAMED_CE_SUMMARY '+json.dumps(diagnostic,sort_keys=True),flush=True)
    counts=diagnostic['compact_mlp'].get('counts',{})
    print('SOUP_COMPACT_MLP '+json.dumps({'event':'owned_mlp_forward','count':counts.get('owned',0),
        'native_fallback':counts.get('native',0),'starter_original_credit':0},sort_keys=True),flush=True)
    print('SOUP_SHARED_CAST '+json.dumps({'event':'independent_input_cast',**diagnostic['compact_mlp'].get('independent_input_cast',{})},sort_keys=True),flush=True)
    print('SOUP_BATCH_CE '+json.dumps({'event':'batched_ce_head',**diagnostic['batch_ce']},sort_keys=True),flush=True)
    storage=diagnostic['frozen_storage'];counts=storage.get('counts',{})
    print('SOUP_FROZEN_STORAGE '+json.dumps({'event':'owned_embedding_forward','count':counts.get('embedding_calls',0),
        'installed':counts.get('installed',0),'head_fp16_calls':counts.get('head_fp16_calls',0),
        'head_native_operand_calls':counts.get('head_native_operand_calls',0),'owners_intact':storage.get('owners_intact',False),
        'storage_original_credit':0},sort_keys=True),flush=True)


def restore():
    result,errors=_restore_owned()
    if errors:raise RuntimeError('Composition restore raised: '+repr(errors))
    return result
