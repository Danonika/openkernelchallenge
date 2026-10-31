"""Same-dtype saved-activation offload scoped to the native Qwen backbone.

This changes saved-state placement only. It does not enable a Soup/Transformers
offloading switch, change a task field, replay a layer, change an RNG draw, or
claim original-kernel credit. Actual measurements and numerical validation are
required before using this experimental component as a competition solution.
"""
from collections import Counter
from dataclasses import dataclass
from functools import wraps
import hashlib
import inspect
import math
import weakref
import torch

MIN_BYTES = 1024 * 1024
MODEL_SOURCE_SHA256 = '523115d50c78f82f5256cc0aa9cba43732f8f6b92bef4831382a07c570c2b076'
_ACTIVE = None


def _safe_strides(shape, strides):
    """Accept nonoverlapping positive strides with bounded allocation overhead."""
    if (len(shape) != len(strides) or not shape or any(n <= 0 for n in shape)
            or any(s < 0 for s in strides)):
        return False
    dimensions = sorted((s, n) for s, n in zip(strides, shape) if n > 1)
    extent = 1
    for stride, count in dimensions:
        if stride < extent:
            return False
        extent = stride * count
    span = 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    return span <= 2 * math.prod(shape)


@dataclass
class _Packed:
    host: object
    ready: object
    device: object
    shape: tuple
    strides: tuple
    original_ref: object
    original_version: int


class ActivationOffloadInstallation:
    def __init__(self, owner, native):
        self.owner = owner
        self.native = native
        self.active = True
        self.counts = Counter()
        self.forward = self._make_forward()

    def _make_forward(self):
        native = self.native
        installation = self

        @wraps(native)
        def forward(model, *args, **kwargs):
            if not installation.intact():
                raise RuntimeError('Saved-activation offload owner changed')
            # Evaluation and unsupported public calls retain their native path.
            config = getattr(model, 'config', None)
            supported = (type(model) is installation.owner and model.training
                         and torch.is_grad_enabled()
                         and getattr(config, 'model_type', None) == 'qwen2'
                         and getattr(config, 'hidden_size', None) == 1536
                         and not getattr(model, 'gradient_checkpointing', False)
                         and kwargs.get('past_key_values') is None
                         and kwargs.get('use_cache') is False
                         and not torch.compiler.is_compiling())
            if not supported:
                installation.counts['native_forward'] += 1
                return native(model, *args, **kwargs)
            installation.counts['offloaded_forward'] += 1
            with torch.autograd.graph.saved_tensors_hooks(installation.pack,
                                                          installation.unpack):
                return native(model, *args, **kwargs)

        return forward

    def intact(self):
        return self.active and self.owner.forward is self.forward

    def pack(self, tensor):
        # Parameter subclasses, persistent frozen operands and small scalars
        # remain on device. In particular, no NF4 weight or LoRA parameter moves.
        if (type(tensor) is not torch.Tensor or tensor.device.type != 'cuda'
                or tensor.ndim < 2 or tensor.numel() * tensor.element_size() < MIN_BYTES
                or tensor.is_conj() or tensor.is_neg()
                or not (tensor.requires_grad or tensor.dtype in (torch.bool, torch.uint8))
                or not _safe_strides(tuple(tensor.shape), tuple(tensor.stride()))):
            return tensor
        shape, strides = tuple(tensor.shape), tuple(tensor.stride())
        with torch.cuda.device(tensor.device):
            stream = torch.cuda.current_stream(tensor.device)
            host = torch.empty_strided(shape, strides, dtype=tensor.dtype,
                                       device='cpu', pin_memory=True)
            host.copy_(tensor, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
            # Preserve source allocation until the enqueued D2H read completes.
            # Producer readiness follows the native current-stream contract.
            tensor.record_stream(stream)
        self.counts['packed_tensors'] += 1
        self.counts['packed_logical_bytes'] += tensor.numel() * tensor.element_size()
        return _Packed(host, ready, tensor.device, shape, strides,
                       weakref.ref(tensor), tensor._version)

    def unpack(self, packed):
        if type(packed) is not _Packed:
            return packed
        original = packed.original_ref()
        if original is not None and original._version != packed.original_version:
            raise RuntimeError('Saved activation was modified in place after packing')
        with torch.cuda.device(packed.device):
            stream = torch.cuda.current_stream(packed.device)
            # The H2D copy must not read pinned host data before D2H completes.
            stream.wait_event(packed.ready)
            result = torch.empty_strided(packed.shape, packed.strides,
                                         dtype=packed.host.dtype, device=packed.device)
            # Blocking host return deliberately establishes host-buffer lifetime
            # and completed readback; no host slot can be freed during H2D.
            result.copy_(packed.host, non_blocking=False)
        self.counts['unpacked_tensors'] += 1
        self.counts['unpacked_logical_bytes'] += result.numel() * result.element_size()
        return result

    def status(self):
        return {'active': self.active, 'owners_intact': self.intact(),
                'counts': dict(self.counts), 'min_bytes': MIN_BYTES,
                'same_dtype': True, 'original_kernel_credit': 0,
                'scope': 'native_backbone_saved_activations_only',
                'gpu_validated': False}

    def restore(self):
        if not self.active:
            return False
        restored = self.owner.forward is self.forward
        if restored:
            self.owner.forward = self.native
        self.active = False
        return restored


def install_activation_offload():
    global _ACTIVE
    if _ACTIVE is not None and _ACTIVE.active:
        if not _ACTIVE.intact():
            raise RuntimeError('Saved-activation offload installation changed')
        return _ACTIVE
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    native = Qwen2Model.forward
    source = inspect.getsource(native)
    if hashlib.sha256(source.encode()).hexdigest() != MODEL_SOURCE_SHA256:
        raise RuntimeError('Native Qwen2Model.forward differs from reviewed 5.16.1 source: observed_sha256=' + hashlib.sha256(source.encode()).hexdigest())
    handle = ActivationOffloadInstallation(Qwen2Model, native)
    Qwen2Model.forward = handle.forward
    _ACTIVE = handle
    return handle
