"""Inactive direct native linear route for frozen, cached NF4 MLP matrices.

This optional installer shares the already-installed fixed-capacity cache;
it creates no second weight cache and performs no eager/pre-timed decoding.
Only contiguous two-dimensional training inputs enter the direct path, and
only when the pinned bitsandbytes dispatcher selects dequantize + F.linear.
Native fused GEMM, 3D inputs, evaluation and unsupported modules delegate.

The casts are the pinned Linear4bit sequence: input -> compute FP16, native
F.linear, output -> original input dtype. Weight and optional bias are frozen,
and the bias must already be FP16; a required native first-call bias cast is
delegated. With row-major 2D input, native linear autograd uses grad.mm(W),
the same operation as MatMul4Bit.backward's 2D torch.matmul(grad, W).

The source weights must remain frozen for the lifetime of each autograd
graph. F.linear saves the decoded matrix for its native backward. Ordinary
source mutations between forwards still invalidate the shared cache; changing
a packed weight between a forward and its backward is outside this contract.
If a matrix is not retained under the cache budget, native linear autograd
can hold its fresh decoded result until backward. Thus the cache byte budget
is not a bound on total live activation/storage memory; measure the real peak.

Install after patch_nf4_forward_cache and before other patches capture
Linear4bit.forward (notably compact MLP and padding compaction). Nothing is
installed on import. This module adds no countable original GPU kernel.
"""

from functools import wraps
from importlib.metadata import version

import torch

from .nf4_forward_cache import FrozenNF4ForwardCache
from .nf4_forward_cache_patch import _canonical_components


def _is_cuda_tensor(tensor):
    return tensor.is_cuda


def _plain_module(module):
    return ("forward" not in module.__dict__ and not hasattr(module, "_hf_hook")
            and not module._forward_hooks and not module._forward_pre_hooks
            and not module._backward_hooks and not module._backward_pre_hooks)


def _selects_native_linear(ops, x, shape, blocksize):
    """Use the pinned native metadata heuristic; never replace a fused GEMM.

    A misaligned K at or below the native hard cap delegates so that native
    warning/dispatch behavior is preserved. The native cap check precedes its
    alignment check, so an above-cap shape is unconditionally the fallback.
    """
    rows, width = x.shape
    if rows > ops._gemm_4bit_custom_max_m:
        return True
    if width % blocksize:
        return False
    return not ops._gemm_4bit_use_custom_fn(
        x.device.index, torch.float16, rows, shape[0], width)


def _eligible_components(module, x, linear_class, ops, nf4):
    if (type(module) is not linear_class or type(x) is not torch.Tensor
            or not _is_cuda_tensor(x) or x.layout != torch.strided
            or x.dtype not in (torch.float16, torch.float32) or x.ndim != 2
            or min(x.shape) <= 1 or not x.is_contiguous()
            or x.is_conj() or x.is_neg() or x.is_inference()
            or not module.training or not torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
            or torch.compiler.is_compiling() or torch.jit.is_scripting()
            or torch.jit.is_tracing() or not _plain_module(module)
            or module.compute_dtype != torch.float16 or not module.compute_type_is_set):
        return None
    if (torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") != torch.float16):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    hooks = torch.nn.modules.module
    if (hooks._global_forward_hooks or hooks._global_forward_pre_hooks
            or hooks._global_backward_hooks or hooks._global_backward_pre_hooks):
        return None
    packed = module.weight
    if (not isinstance(packed, torch.Tensor) or packed.requires_grad
            or packed.grad_fn is not None or packed.device != x.device
            or packed.dtype != torch.uint8):
        return None
    state = getattr(packed, "quant_state", None)
    shape = getattr(state, "shape", None)
    if (state is None or getattr(state, "dtype", None) != torch.float16
            or not isinstance(shape, (tuple, list, torch.Size))
            or tuple(shape) != (module.out_features, module.in_features)
            or x.shape[1] != module.in_features
            or getattr(state, "packing_format_for_cpu", False)):
        return None
    bias = module.bias
    if bias is not None:
        if (type(bias) not in (torch.Tensor, torch.nn.Parameter) or bias.requires_grad
                or bias.grad_fn is not None or bias.dtype != torch.float16
                or bias.device != x.device or tuple(bias.shape) != (module.out_features,)
                or not bias.is_contiguous() or bias.is_conj() or bias.is_neg()
                or bias.is_inference()):
            return None
    components = nf4._state_components(packed, state, None, cuda_only=True)
    if (components is None
            or (components["nested_scales"] is not None
                and components["nested_blocksize"] != 256)
            or not _selects_native_linear(ops, x, state.shape, state.blocksize)):
        return None
    return components


def _make_forward(original, cache, linear_class, ops, nf4):
    @wraps(original)
    def forward(self, x):
        components = _eligible_components(self, x, linear_class, ops, nf4)
        if components is None:
            return original(self, x)
        compute_input = x.to(torch.float16)
        decoded = cache.decode(_canonical_components(components), nf4._execute,
                               training=True)
        return torch.nn.functional.linear(compute_input, decoded, self.bias).to(x.dtype)

    forward._soup_nf4_cached_linear = cache
    return forward


def patch_nf4_cached_linear(cache):
    """Opt in using the exact cache already installed at the native boundary.

    No installation is attempted for another version, another cache, or a
    missing original NF4 patch. A repeated call must use the same instance.
    """
    if (not isinstance(cache, FrozenNF4ForwardCache) or cache.byte_budget <= 0
            or version("bitsandbytes") != "0.50.2"
            or version("torch").split("+")[0] != "2.11.0"
            or torch.version.cuda is None or torch.version.hip is not None):
        return False
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    from bitsandbytes.backends.cuda import ops
    from . import nf4

    original = bnb.nn.Linear4bit.forward
    existing = getattr(original, "_soup_nf4_cached_linear", None)
    if existing is not None:
        if existing is not cache:
            raise ValueError("direct NF4 linear already uses a different fixed cache")
        return existing
    if (getattr(ops._dequant_linear_fallback, "_soup_nf4_forward_cache", None) is not cache
            or not getattr(original, "_soup_nf4_cache_training_context", False)
            or not getattr(functional.dequantize_4bit, "_soup_candidate_nf4", False)
            or ops._gemm_4bit_custom_max_m != 1536):
        return False
    bnb.nn.Linear4bit.forward = _make_forward(original, cache, bnb.nn.Linear4bit, ops, nf4)
    return cache
