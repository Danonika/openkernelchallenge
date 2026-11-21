"""Read-only observer of an unchanged, source-owned dense starter MLP.

COMPONENT PHASE ONLY. Ordinary native owner eligibility is checked before
temporarily installing three owned output observers; every observer returns
None. No producer, activation math, backward, weight, mask or output is changed.
The later conditional-backward owner must independently validate these records.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import weakref
import torch
from bitsandbytes.autograd._functions import MatMul4Bit
from bitsandbytes.nn import Linear4bit, Params4bit
from .mlp_half import native_owner as no, starter_half as sh
from .row_adjoint import OriginalBatchMask

_HASHES = {
    'native_owner.py': '7b69f2a354def25cb53d4ca78f3c19082497838c5f6ee133d94694e6113d540c',
    'owned_half.py': '8a714205c4f1de8ff7997f9430e47fb417b54e66879b64fa80144896c52a807b',
    'starter_half.py': '3759f3761b3b3d2a8e018b80a8571c9cc1456b8adc1ed40e47e1c919132e7dda',
    'starter_swiglu.py': 'e665910faeca6091c958ed7a191604484a44216f51d6fa069f5f48220e7472e6',
}
_DENSE = sh.owned_starter_mlp
_OWNER_CLASS = no.NativeMLPHalfOwner
_REASON = no.NativeMLPHalfOwner.reason
_INIT = no.NativeMLPHalfOwner.__init__
_FUNCTIONS = (_DENSE, _REASON, _INIT, sh.OwnedStarterHalf.forward, sh.OwnedStarterHalf.backward)
_CODES = tuple(fn.__code__ for fn in _FUNCTIONS)
_PLAN_UNCHANGED = OriginalBatchMask.unchanged
_PLAN_CODE = _PLAN_UNCHANGED.__code__
with torch.inference_mode(False), torch.enable_grad():
    # Discover the exact native edge class without names, GPU or RNG use.
    _COPY_TYPE = type(torch.empty(0, dtype=torch.float16, requires_grad=True).float().grad_fn)


@dataclass(frozen=True)
class TensorStamp:
    tensor: object
    version: int
    pointer: int
    shape: tuple
    stride: tuple
    dtype: object
    device: object


def _stamp(tensor):
    if not isinstance(tensor, torch.Tensor):
        return None
    return TensorStamp(tensor, tensor._version, tensor.untyped_storage().data_ptr(),
                       tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


@dataclass(frozen=True)
class ProjectionBinding:
    node: object
    label: str
    plan: object
    output_shape: tuple
    input_width: int
    state: object
    packed: torch.Tensor
    packed_version: int
    packed_pointer: int
    weight_ref: object
    weight_version: int
    quant_metadata: tuple


@dataclass(frozen=True)
class DenseObservation:
    output: torch.Tensor
    bindings: tuple
    reason: object


def _quant_metadata(state):
    child = getattr(state, 'state2', None)
    return (tuple(state.shape), state.blocksize, state.quant_type, state.dtype,
            state.nested, child,
            tuple((name, _stamp(getattr(state, name, None)))
                  for name in ('absmax', 'code', 'offset')),
            None if child is None else
            (child.blocksize, child.quant_type, child.dtype,
             tuple((name, _stamp(getattr(child, name, None)))
                   for name in ('absmax', 'code', 'offset'))))


def _bind_projection(label, projection, args, output, plan):
    """Exact direct/one-native-cast edge inspection; never broad graph search."""
    if (label not in ('gate_proj', 'up_proj', 'down_proj')
            or type(projection) is not Linear4bit or type(projection.weight) is not Params4bit
            or projection.bias is not None or projection.weight.requires_grad
            or projection.compute_dtype != torch.float16 or not projection.compute_type_is_set
            or not isinstance(args, tuple) or len(args) != 1 or type(args[0]) is not torch.Tensor
            or type(output) is not torch.Tensor or not output.requires_grad):
        raise ValueError('projection_contract')
    incoming = args[0]
    expected = torch.float32 if label == 'down_proj' else torch.float16
    if incoming.dtype != expected or output.dtype != expected:
        raise ValueError('projection_return_dtype')
    state = projection.weight.quant_state
    if (state is None or tuple(state.shape) != (projection.out_features, projection.in_features)
            or tuple(output.shape) != tuple(incoming.shape[:-1]) + (projection.out_features,)
            or output.device != incoming.device):
        raise ValueError('projection_return_geometry')
    node = output.grad_fn
    if label == 'down_proj':
        if type(node) is not _COPY_TYPE or len(node.next_functions) != 1 or node.next_functions[0][1] != 0:
            raise ValueError('expected_one_native_output_cast')
        node = node.next_functions[0][0]
    if type(node) is not MatMul4Bit._backward_cls or MatMul4Bit._backward_cls._forward_cls is not MatMul4Bit:
        raise ValueError('exact_native_context_required')
    if (getattr(node, 'is_empty', True) or tuple(node.needs_input_grad) != (True, False, False, False, False)
            or node.state is not state or node.dtype_A != torch.float16
            or node.dtype_B != torch.uint8 or node.dtype_bias is not None):
        raise ValueError('native_context_contract')
    tensors = node.tensors
    if type(tensors) is not tuple or len(tensors) != 2 or tensors[0] is not None:
        raise ValueError('native_context_payload')
    packed = tensors[1]
    weight = projection.weight
    if (type(packed) is not torch.Tensor or packed.requires_grad or packed.dtype != torch.uint8
            or packed.device != weight.device or not packed.is_contiguous()
            or tuple(packed.shape) != (weight.numel(), 1)
            or packed.data_ptr() != weight.data_ptr()
            or packed.untyped_storage().data_ptr() != weight.untyped_storage().data_ptr()):
        raise ValueError('packed_native_weight_identity')
    return ProjectionBinding(node, label, plan, tuple(output.shape), projection.in_features,
                             state, packed, packed._version, packed.untyped_storage().data_ptr(),
                             weakref.ref(weight), weight._version, _quant_metadata(state))


def _primitive_intact():
    if (_DENSE is not _FUNCTIONS[0] or _REASON is not _FUNCTIONS[1]
            or _INIT is not _FUNCTIONS[2] or no.NativeMLPHalfOwner is not _OWNER_CLASS):
        return False
    current = (sh.owned_starter_mlp, no.NativeMLPHalfOwner.reason, no.NativeMLPHalfOwner.__init__,
               sh.OwnedStarterHalf.forward, sh.OwnedStarterHalf.backward)
    if any(a is not b or a.__code__ is not code for a, b, code in zip(current, _FUNCTIONS, _CODES)):
        return False
    root = Path(sh.__file__).parent
    return all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
               for name, digest in _HASHES.items())


def _plan_reason(plan, x):
    if (type(plan) is not OriginalBatchMask or OriginalBatchMask.unchanged is not _PLAN_UNCHANGED
            or _PLAN_UNCHANGED.__code__ is not _PLAN_CODE or not _PLAN_UNCHANGED(plan)):
        return 'original_batch_plan_changed_or_foreign'
    if x.ndim != 3 or tuple(x.shape[:2]) != plan.shape or plan.mask.device != x.device:
        return 'original_batch_plan_geometry_device'
    return None


def observe_starter_mlp(module, x, *, plan):
    """Return unchanged output plus exactly three bindings, or no bindings.

    ``plan`` is the caller's exact explicit OriginalBatchMask, carried by
    identity and validated at entry/end against the input shape/device. This
    observer never derives padding policy; the consumer revalidates its ticket.
    This entrypoint is for uninstalled native Qwen2MLP component forwards only.
    Existing compact/full-model installations require a separate reviewed seam.
    """
    if not _primitive_intact():
        raise RuntimeError('Dense observer primitive ownership changed')
    dense, check_reason, bind_projection = _DENSE, _REASON, _bind_projection
    owner = no.NativeMLPHalfOwner(module, first_order_only=True)
    reason = check_reason(owner, x)
    if reason is not None:
        return DenseObservation(module(x), (), reason)
    reason = _plan_reason(plan, x)
    if reason is not None:
        return DenseObservation(dense(module, x), (), reason)
    bindings, failures, hooks = [], [], []
    try:
        for label in ('gate_proj', 'up_proj', 'down_proj'):
            projection = getattr(module, label)
            def hook(actual, args, output, *, label=label, projection=projection):
                if actual is not projection:
                    failures.append('projection_identity_changed')
                else:
                    try:
                        bindings.append(bind_projection(label, actual, args, output, plan))
                    except ValueError as error:
                        failures.append(str(error))
                return None
            handle = torch.nn.Module.register_forward_hook(projection, hook)
            hooks.append((projection, handle, hook))
        output = dense(module, x)
    finally:
        # Remove only our own hooks; never overwrite a foreign replacement.
        for projection, handle, hook in hooks:
            if projection._forward_hooks.get(handle.id) is hook:
                handle.remove()
            else:
                failures.append('observer_hook_ownership_changed')
    after = check_reason(owner, x)
    if after is not None:
        failures.append('post_forward_' + after)
    if not _primitive_intact():
        failures.append('post_forward_primitive_owner_changed')
    plan_reason = _plan_reason(plan, x)
    if plan_reason is not None:
        failures.append('post_forward_' + plan_reason)
    if tuple(b.label for b in bindings) != ('gate_proj', 'up_proj', 'down_proj'):
        failures.append('three_exact_projections_absent')
    if len({id(b.node) for b in bindings}) != 3:
        failures.append('three_distinct_contexts_absent')
    return DenseObservation(output, () if failures else tuple(bindings), ';'.join(failures) or None)
