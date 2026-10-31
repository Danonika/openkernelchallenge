"""Bounded observational wrappers; native calls, outputs, exceptions and RNG survive.

Host monotonic intervals are diagnostic, not synchronized GPU durations. A late
atexit marker never establishes native command completion. No caches, trainer
settings, callbacks, data, optimizer, stdout, or native save behavior are changed.
"""
import atexit
import functools
import json
import os
import time

_PREFIX = b'SOUP_STAGE_TIMING '
_MAX_RECORDS = 48
_sequence = 0
_origin = None
_owned = []
_installed = False


def emit(stage, event):
    global _sequence, _origin
    if _sequence >= _MAX_RECORDS:
        return
    try:
        now = time.monotonic_ns()
        if _origin is None:
            _origin = now
        _sequence += 1
        value = {'schema': 1, 'stage': stage, 'event': event, 'sequence': _sequence,
                 'count': 1, 'since_apply_ns': now - _origin}
        raw = _PREFIX + json.dumps(value, separators=(',', ':'), allow_nan=False).encode() + b'\n'
        if len(raw) <= 1024:
            os.write(1, raw)
    except Exception:
        # Diagnostic I/O failure is never allowed to change native arithmetic.
        pass


def wrap(owner, name, label):
    """Wrap one existing plain method/function, preserving its normal contract."""
    original = getattr(owner, name, None)
    if not callable(original) or getattr(original, '__soup_stage_probe__', False):
        return False
    @functools.wraps(original)
    def observed(*args, **kwargs):
        emit(label, 'enter')
        try:
            result = original(*args, **kwargs)
        except BaseException:
            emit(label, 'raise')
            raise
        emit(label, 'return')
        return result
    observed.__soup_stage_probe__ = True
    setattr(owner, name, observed)
    _owned.append((owner, name, original, observed))
    return True


def restore():
    global _installed
    for owner, name, original, observed in reversed(_owned):
        if getattr(owner, name, None) is observed:
            setattr(owner, name, original)
    _owned.clear()
    _installed = False


def install():
    global _installed
    if _installed:
        return
    emit('apply', 'enter')
    # These are the unchanged native libraries; their import cost remains inside
    # the measured command. Nothing is loaded in an external warm cache.
    import soup_cli.trainer.sft as sft
    from transformers import Trainer
    emit('native_dependencies', 'loaded')
    for owner, name, label in (
        (sft.SFTTrainerWrapper, 'setup', 'soup_setup'),
        (sft.SFTTrainerWrapper, '_setup_transformers', 'model_setup'),
        (sft.SFTTrainerWrapper, 'train', 'soup_train_and_final_save'),
        (sft, '_assert_finite_training_state', 'final_finite_validation'),
        (Trainer, '__init__', 'trainer_init'),
        (Trainer, 'train', 'trainer_train'),
        (Trainer, '_save_checkpoint', 'checkpoint_save'),
        (Trainer, 'save_model', 'adapter_save'),
    ):
        wrap(owner, name, label)
    atexit.register(emit, 'python_atexit_callback_not_process_exit', 'observed')
    _installed = True
    emit('apply', 'return')
