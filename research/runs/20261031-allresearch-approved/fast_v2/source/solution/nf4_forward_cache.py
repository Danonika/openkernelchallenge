"""Inactive, bounded cache of frozen NF4 decoded FP16 weights.

Nothing is installed on import. An eventual forward fallback may call
``cache.decode(components, nf4._execute, training=known_training_intent)``.
The optional installer may also admit a native backward's exact frozen input;
ordinary public dequantization remains outside that scope. The decoder and
native linear/matmul operations are unchanged.

Admission is first-fit under a fixed byte budget. Each packed-weight root has
at most one entry. Source roots are weakly held; output matrices are strongly
held. Every hit checks ordinary Torch version counters, storage identities,
offsets, shapes, strides, dtypes, devices, and the output's own metadata.
Ephemeral transpose/view arguments share the persistent source-root identity.
All supplied out buffers bypass the cache. This module never writes a source
tensor or reuses a decoded tensor as a writable output buffer.

This is a frozen-weight contract: mutations must use normal version-tracked
Torch operations. Untracked writes through raw pointers, storage, or .data
aliases are outside that contract. Cross-stream and CUDA-graph reuse bypass
the cache instead of adding synchronization or changing graph behavior.
No weights, packages, or cache files are written outside this process.
"""

from dataclasses import dataclass
import math
import threading
import weakref

import torch


_BLOCKS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
_TENSOR_NAMES = ("packed", "scales", "nested_scales", "nested_code", "offset")
_KEYS = frozenset((*_TENSOR_NAMES, "shape", "dtype", "blocksize", "nested_blocksize", "out"))


def _stamp(tensor):
    """Only metadata, including TensorImpl's version and StorageImpl identity."""
    if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
            or tensor.device.type not in ("cpu", "cuda") or tensor.is_meta
            or tensor.requires_grad or tensor.grad_fn is not None
            or tensor.is_conj() or tensor.is_neg() or tensor.is_inference()):
        raise ValueError("cache identity requires a frozen, ordinary strided tensor")
    storage = tensor.untyped_storage()
    version = tensor._version
    if not isinstance(version, int) or version < 0:
        raise ValueError("tensor has no usable version counter")
    return (version, storage._cdata, storage.data_ptr(), storage.nbytes(),
            tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device, tensor.layout)


def _root_and_stamp(tensor):
    root = tensor
    seen = set()
    while getattr(root, "_base", None) is not None:
        if id(root) in seen:
            raise ValueError("cyclic tensor base chain")
        seen.add(id(root))
        root = root._base
    view_stamp = _stamp(tensor)
    root_stamp = view_stamp if root is tensor else _stamp(root)
    # A normal Torch view shares both the StorageImpl and version counter.
    if view_stamp[:2] != root_stamp[:2]:
        raise ValueError("view does not expose a reliable source-root identity")
    weakref.ref(root)  # Refuse sources whose lifetimes cannot be tracked.
    return root, (id(root), view_stamp, root_stamp)


def _snapshot(components, execute):
    if type(components) is not dict or set(components) - _KEYS:
        return None
    if components.get("out") is not None or components.get("dtype") != torch.float16:
        return None
    shape, packed = components.get("shape"), components.get("packed")
    if (not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 2
            or any(type(x) is not int or x <= 0 for x in shape)
            or not isinstance(packed, torch.Tensor)):
        return None
    block = components.get("blocksize")
    nested_block = components.get("nested_blocksize", 256)
    if type(block) is not int or block not in _BLOCKS:
        return None
    if (packed.dtype != torch.uint8 or packed.ndim != 2 or min(packed.shape) != 1
            or not packed.is_contiguous() or packed.numel() != (math.prod(shape) + 1) // 2):
        return None
    device = packed.device
    nested = components.get("nested_scales") is not None
    scale_count = (math.prod(shape) + block - 1) // block

    def valid(name, dtype, count):
        value = components.get(name)
        return (isinstance(value, torch.Tensor) and value.dtype == dtype
                and value.device == device and value.numel() == count
                and value.is_contiguous())

    if not valid("scales", torch.uint8 if nested else torch.float32, scale_count):
        return None
    if nested:
        if (type(nested_block) is not int or nested_block not in _BLOCKS
                or not valid("nested_scales", torch.float32,
                             (scale_count + nested_block - 1) // nested_block)
                or not valid("nested_code", torch.float32, 256)
                or not valid("offset", torch.float32, 1)):
            return None
    elif components.get("nested_code") is not None or components.get("offset") is not None:
        return None

    roots, identities = [], []
    for name in _TENSOR_NAMES:
        tensor = components.get(name)
        if tensor is None:
            identities.append(None)
        else:
            root, identity = _root_and_stamp(tensor)
            roots.append(root)
            identities.append(identity)
    weakref.ref(execute)
    roots.append(execute)
    stream = None
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                return None
            stream = torch.cuda.current_stream(device).cuda_stream
    signature = (tuple(shape), torch.float16, block, nested_block,
                 tuple(identities), id(execute))
    return roots[0], tuple(roots), signature, math.prod(shape) * 2, stream


@dataclass
class _Entry:
    token: object
    references: tuple
    signature: tuple
    value: torch.Tensor
    value_stamp: tuple
    byte_count: int
    stream: object


class FrozenNF4ForwardCache:
    """A fixed-capacity, process-local first-fit cache; disabled by default.

    ``training=True`` must come from the caller's independent training
    admission. Custom autograd forward runs with grad mode disabled even in
    training, so checking torch.is_grad_enabled() here would be incorrect.
    Explicit inference_mode always bypasses, even with training=True.
    """

    def __init__(self, byte_budget):
        if type(byte_budget) is not int or byte_budget < 0:
            raise ValueError("byte_budget must be a nonnegative integer")
        self._byte_budget = byte_budget
        self._bytes = 0
        self._entries = {}
        self._lock = threading.RLock()
        self._hits = self._misses = self._bypasses = 0

    @property
    def byte_budget(self):
        return self._byte_budget

    def info(self):
        with self._lock:
            return dict(byte_budget=self._byte_budget, cached_bytes=self._bytes,
                        entries=len(self._entries), hits=self._hits,
                        misses=self._misses, bypasses=self._bypasses)

    def _drop(self, owner_id, token=None):
        entry = self._entries.get(owner_id)
        if entry is not None and (token is None or entry.token is token):
            self._bytes -= entry.byte_count
            del self._entries[owner_id]

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def _delegate(self, components, execute):
        with self._lock:
            self._bypasses += 1
        return execute(**components)

    def decode(self, components, execute, *, training=False):
        """Return the unchanged decoder result, possibly reusing a frozen one.

        The decoder is called once on misses/bypasses, and never replayed
        after exceptions. Returned cache entries are read-only by contract;
        ordinary accidental output mutations invalidate their next lookup.
        """
        if (training is not True or torch.is_inference_mode_enabled()
                or torch.compiler.is_compiling() or self._byte_budget == 0):
            return self._delegate(components, execute)
        try:
            plan = _snapshot(components, execute)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            plan = None
        if plan is None:
            return self._delegate(components, execute)
        owner, roots, signature, byte_count, stream = plan
        owner_id = id(owner)
        with self._lock:
            entry = self._entries.get(owner_id)
            if entry is not None:
                try:
                    valid = (entry.signature == signature
                             and len(entry.references) == len(roots)
                             and all(ref() is root for ref, root in zip(entry.references, roots))
                             and _stamp(entry.value) == entry.value_stamp)
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    valid = False
                if valid:
                    if entry.stream != stream:
                        return self._delegate(components, execute)
                    self._hits += 1
                    return entry.value
                self._drop(owner_id)
            self._misses += 1
            result = execute(**components)
            if byte_count > self._byte_budget - self._bytes:
                return result
            try:
                after = _snapshot(components, execute)
                value_stamp = _stamp(result)
                source_storages = {identity[1][1] for identity in signature[4] if identity is not None}
                admissible = (after is not None and after[2] == signature and after[4] == stream
                              and type(result) is torch.Tensor and result._base is None
                              and result.dtype == torch.float16 and result.device == components["packed"].device
                              and tuple(result.shape) == signature[0] and result.is_contiguous()
                              and result.storage_offset() == 0 and value_stamp[3] == byte_count
                              and value_stamp[1] not in source_storages)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                admissible = False
            if not admissible:
                return result

            token, cache_ref = object(), weakref.ref(self)

            def release(_reference, owner_id=owner_id, token=token, cache_ref=cache_ref):
                cache = cache_ref()
                if cache is not None:
                    with cache._lock:
                        cache._drop(owner_id, token)

            references = tuple(weakref.ref(root, release) for root in roots)
            self._entries[owner_id] = _Entry(token, references, signature, result,
                                            value_stamp, byte_count, stream)
            self._bytes += byte_count
            return result
