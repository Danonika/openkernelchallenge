"""Original zero-storage version witness for a native saved FP32 operand.

A detached Tensor shares the input's version counter. Replacing ONLY that
witness's .data with an empty tensor discards its storage without modifying the
source's value, storage or version. This is not .data mutation of model values.
The private/version-sensitive mechanism is checked on the installed runtime;
unsupported behavior declines the codec. .data mutations are outside native
PyTorch version tracking too, and are not claimed detectable here.
"""
import gc
import weakref

import torch


class UnsupportedVersionWitness(RuntimeError):
    pass


def make_witness(source):
    if type(source) is not torch.Tensor or source.layout != torch.strided:
        raise UnsupportedVersionWitness("ordinary strided Tensor required")
    before = (source.data_ptr(), source._version, tuple(source.shape),
              tuple(source.stride()), source.dtype, source.device)
    witness = source.detach()
    # set_ would increment the SOURCE's shared version: do not use it here.
    witness.data = torch.empty(0, dtype=source.dtype, device=source.device)
    after = (source.data_ptr(), source._version, tuple(source.shape),
             tuple(source.stride()), source.dtype, source.device)
    if (before != after or witness._version != source._version
            or witness.numel() != 0 or witness.untyped_storage().nbytes() != 0
            or witness.grad_fn is not None or witness.requires_grad):
        raise UnsupportedVersionWitness("runtime does not preserve a zero-storage shared version")
    return witness


def runtime_contract(device="cpu"):
    """Bounded synthetic check; consumes no RNG and changes no model tensor."""
    wide = torch.arange(4, dtype=torch.float32, device=device)
    alias = wide.detach()
    ref = weakref.ref(wide)
    witness = make_witness(wide)
    expected = wide._version
    del wide
    gc.collect()
    if ref() is not None:
        raise UnsupportedVersionWitness("witness retained original Python Tensor")
    alias.add_(1)
    if witness._version != expected + 1:
        raise UnsupportedVersionWitness("detached alias mutations are not witnessed")
    del alias
    if witness.untyped_storage().nbytes() != 0:
        raise UnsupportedVersionWitness("witness retained nonzero storage")
    return {"device": str(device), "wide_object_released": True,
            "witness_storage_bytes": 0, "detached_alias_mutation_detected": True}
