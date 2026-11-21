"""Fixed read-only observation sessions for the diagnostic installer derivative.

No mask discovery, callback, producer math, backward dispatch or kernel credit.
The original batch plan enters explicitly through begin_session(plan). Stored
records omit output tensors: only the exact native contexts/bindings survive.
"""
from collections import Counter
import inspect
from . import forward_observer as observer
from .mlp_half import native_owner as native, starter_half as half
from .row_adjoint import OriginalBatchMask


class ObservationSession:
    def __init__(self, plan):
        if type(plan) is not OriginalBatchMask or not _PLAN_UNCHANGED(plan):
            raise ValueError('Exact unchanged original batch plan required')
        self.plan = plan
        self._plan_owner = plan
        self.active = True
        self._observations = []
        self._observations_owner = self._observations
        self.counts = Counter(owned=0, native=0, bound_contexts=0)
        self.reasons = Counter()

    @property
    def observations(self):
        return tuple(self._observations)

    def _append(self, bindings, reason):
        if not self.active:
            raise RuntimeError('Observation session already ended')
        self._observations.append(observer.DenseObservation(None, tuple(bindings), reason))
        self.counts['owned'] += 1
        self.counts['bound_contexts'] += len(bindings)
        if reason is not None:
            self.reasons[reason] += 1

    def _native(self, reason):
        if not self.active:
            raise RuntimeError('Observation session already ended')
        self._observations.append(observer.DenseObservation(None, (), 'native_fallback:' + reason))
        self.counts['native'] += 1
        self.reasons['native_fallback:' + reason] += 1

    def _close(self):
        if not self.active:
            return False
        self.active = False
        return True

    def status(self):
        return {'active': self.active, 'counts': dict(self.counts), 'reasons': dict(self.reasons),
                'original_mask_identity': id(self.plan.mask),
                'original_mask_unchanged': _PLAN_UNCHANGED(self.plan),
                'observation_count': len(self._observations), 'retained_output_tensors': 0,
                'starter_original_credit': 0, 'scope': 'dense_starter_forward_read_only_bindings'}


def observe_verified(module, x, guard, session):
    """Called only after the derivative's ordinary InstalledGuard eligibility.

    The installed forward substitution is already explicitly owned by guard.
    We reuse its exact native reason() after removing our temporary hooks;
    we do not construct a native owner against the substituted class method.
    """
    from . import compact_mlp_install as installation
    if (not sources_intact() or not session_intact(session) or not session.active
            or type(guard) is not installation._InstalledGuard or guard.module is not module
            or not installation._module_intact()):
        raise RuntimeError('Installed observation adapter ownership changed')
    plan = session.plan
    reason = observer._plan_reason(plan, x)
    if reason is not None:
        result = half.owned_starter_mlp(module, x)
        _SESSION_APPEND(session, (), reason)
        return result
    bindings, failures, hooks = [], [], []
    # Capture exact dispatch before the owned hook phase, not ambient callbacks.
    bind = observer._bind_projection
    dense = half.owned_starter_mlp
    try:
        import torch
        for label in ('gate_proj', 'up_proj', 'down_proj'):
            projection = getattr(module, label)
            def hook(actual, args, output, *, label=label, projection=projection):
                if actual is not projection:
                    failures.append('projection_identity_changed')
                else:
                    try:
                        bindings.append(bind(label, actual, args, output, plan))
                    except ValueError as error:
                        failures.append(str(error))
                return None
            handle = torch.nn.Module.register_forward_hook(projection, hook)
            hooks.append((projection, handle, hook))
        result = dense(module, x)
    finally:
        for projection, handle, hook in hooks:
            if projection._forward_hooks.get(handle.id) is hook:
                handle.remove()
            else:
                failures.append('observer_hook_ownership_changed')
    after = _NATIVE_REASON(guard, x)
    if after is not None:
        failures.append('post_forward_' + after)
    after = observer._plan_reason(plan, x)
    if after is not None:
        failures.append('post_forward_' + after)
    if not sources_intact() or not session_intact(session):
        raise RuntimeError('Observation dispatch changed during dense forward')
    if tuple(b.label for b in bindings) != ('gate_proj', 'up_proj', 'down_proj'):
        failures.append('three_exact_projections_absent')
    if len({id(b.node) for b in bindings}) != 3:
        failures.append('three_distinct_contexts_absent')
    _SESSION_APPEND(session, () if failures else tuple(bindings), ';'.join(failures) or None)
    return result


def session_intact(session):
    return (type(session) is _SESSION_CLASS
            and session.plan is session._plan_owner
            and session._observations is session._observations_owner
            and all(name not in session.__dict__ for name, _, _ in _SESSION_METHODS)
            and 'observations' not in session.__dict__)


def sources_intact():
    return (ObservationSession is _SESSION_CLASS
            and _SESSION_APPEND is _SESSION_METHODS[1][1]
            and all(getattr(ObservationSession, name) is fn and fn.__code__ is code
                    for name, fn, code in _SESSION_METHODS)
            and isinstance(ObservationSession.observations, property)
            and ObservationSession.observations.fget is _OBSERVATIONS_GETTER
            and _OBSERVATIONS_GETTER.__code__ is _OBSERVATIONS_CODE
            and all(getattr(observer, name) is obj for name, obj in _OBSERVER_OBJECTS)
            and all(fn.__code__ is code for fn, code in _OBSERVER_CODES)
            and all(cls.__init__ is fn and fn.__code__ is code for cls, fn, code in _OBSERVER_CLASS_INITS)
            and observer.OriginalBatchMask is OriginalBatchMask
            and OriginalBatchMask.unchanged is _PLAN_UNCHANGED
            and _PLAN_UNCHANGED.__code__ is _PLAN_CODE
            and native.NativeMLPHalfOwner.reason is _NATIVE_REASON
            and _NATIVE_REASON.__code__ is _NATIVE_REASON_CODE
            and all(globals()[name] is fn and fn.__code__ is code for name, fn, code in _ADAPTER_FUNCTIONS)
            and observer._primitive_intact())


_SESSION_CLASS = ObservationSession
_SESSION_METHODS = tuple((name, getattr(ObservationSession, name), getattr(ObservationSession, name).__code__)
                         for name in ('__init__', '_append', '_native', '_close', 'status'))
_SESSION_APPEND = ObservationSession._append
_OBSERVATIONS_GETTER = ObservationSession.observations.fget
_OBSERVATIONS_CODE = _OBSERVATIONS_GETTER.__code__
_OBSERVER_OBJECTS = tuple((name, getattr(observer, name)) for name in
                         ('_stamp', '_quant_metadata', '_bind_projection', '_primitive_intact', '_plan_reason',
                          'TensorStamp', 'ProjectionBinding', 'DenseObservation'))
_OBSERVER_CODES = tuple((obj, obj.__code__) for _, obj in _OBSERVER_OBJECTS if inspect.isfunction(obj))
_OBSERVER_CLASS_INITS = tuple((cls, cls.__init__, cls.__init__.__code__) for cls in
                              (observer.TensorStamp, observer.ProjectionBinding, observer.DenseObservation))
_PLAN_UNCHANGED = OriginalBatchMask.unchanged
_PLAN_CODE = _PLAN_UNCHANGED.__code__
_NATIVE_REASON = native.NativeMLPHalfOwner.reason
_NATIVE_REASON_CODE = _NATIVE_REASON.__code__
_ADAPTER_FUNCTIONS = tuple((name, globals()[name], globals()[name].__code__)
                         for name in ('observe_verified', 'session_intact', 'sources_intact'))
