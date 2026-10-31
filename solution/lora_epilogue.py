"""Optional original scale/add kernels for the pinned vanilla NF4 LoRA path.

PEFT 0.20.0's bitsandbytes wrapper clones the FP32 base result, multiplies
the FP16 adapter result by its scale, then adds the two. The frozen base
projection already computes in FP16: passing a separate FP16 input avoids
its output upcast. A fresh output below combines that upcast, the defensive
copy, the separately rounded FP16 scale, and the FP32 addition.

The backward must round the incoming gradient to FP16 BEFORE scaling the
adapter branch. Even a power-of-two scale cannot move across this cast:
subnormal gradients give counterexamples. No activation is needed by this
epilogue's backward. Native dropout and both ordinary adapter Linear calls
remain in their original order, on the original dropout input.

Sources for the version-specific dataflow (these kernels are original):
https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py
https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py
https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml
"""

from functools import wraps
from importlib import metadata

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


BLOCK = 1024


@triton.jit
def soup_lora_epilogue_forward(BASE, ADAPTER, OUT, N, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    base = tl.load(BASE + index, valid, 0).to(tl.float32)
    adapter = tl.load(ADAPTER + index, valid, 0).to(tl.float32)
    # The reference multiply produces FP16 before the mixed-dtype addition.
    scaled = (adapter * 2.0).to(tl.float16).to(tl.float32)
    tl.store(OUT + index, base + scaled, valid)


@triton.jit
def soup_lora_epilogue_backward(DOUT, DBASE, DADAPTER, N, BLOCK: tl.constexpr,
                                WRITE_BASE: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    grad = tl.load(DOUT + index, valid, 0).to(tl.float32)
    # AddBackward's adapter edge and the base output cast each round here.
    half_grad = grad.to(tl.float16)
    adapter_grad = (half_grad.to(tl.float32) * 2.0).to(tl.float16)
    if WRITE_BASE:
        tl.store(DBASE + index, half_grad, valid)
    tl.store(DADAPTER + index, adapter_grad, valid)


class _LoRAEpilogue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base_half, adapter_half):
        # This allocation replaces the clone's alias protection. Inputs are
        # read-only, and no base view or adapter tensor is returned or retained.
        output = torch.empty_like(base_half, dtype=torch.float32)
        n = base_half.numel()
        soup_lora_epilogue_forward[(triton.cdiv(n, BLOCK),)](
            base_half, adapter_half, output, n, BLOCK,
            num_warps=4, enable_fp_fusion=False,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        grad = grad.resolve_neg().resolve_conj().to(torch.float32).contiguous()
        # Frozen embeddings may leave the first block's input without a
        # gradient edge. In that case only the adapter derivative is needed.
        write_base = ctx.needs_input_grad[0]
        base_grad = torch.empty_like(grad, dtype=torch.float16) if write_base else None
        adapter_grad = torch.empty_like(grad, dtype=torch.float16)
        n = grad.numel()
        soup_lora_epilogue_backward[(triton.cdiv(n, BLOCK),)](
            grad, base_grad if write_base else adapter_grad, adapter_grad, n, BLOCK, write_base,
            num_warps=4, enable_fp_fusion=False,
        )
        return base_grad, adapter_grad


def lora_epilogue(base_half, adapter_half):
    """Return FP32 ``base + half(adapter * 2)`` with exact first-order casts.

    Both inputs must be matching, contiguous, nonempty FP16 tensors. Direct
    calls permit Triton's CPU interpreter for development; the installed
    model patch below selects only the guarded CUDA training path.
    """
    if (base_half.dtype != torch.float16 or adapter_half.dtype != torch.float16
            or base_half.shape != adapter_half.shape or base_half.device != adapter_half.device
            or not base_half.is_contiguous() or not adapter_half.is_contiguous()
            or base_half.is_neg() or adapter_half.is_neg()
            or base_half.is_conj() or adapter_half.is_conj()
            or not base_half.numel()):
        raise ValueError("LoRA epilogue requires matching nonempty contiguous FP16 tensors")
    return _LoRAEpilogue.apply(base_half, adapter_half)


def _is_cuda_tensor(tensor):
    return tensor.is_cuda


def _cuda_fp16_autocast_active():
    return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16


def _plain_module(module):
    return ("forward" not in module.__dict__ and not hasattr(module, "_hf_hook")
            and not module._forward_hooks and not module._forward_pre_hooks
            and not module._backward_hooks and not module._backward_pre_hooks)


def _eligible_lora(module, x, lora_class, linear_class, linear_forward,
                   adapter_forward, dropout_forward):
    """Metadata-only proof, evaluated before any projection or random call."""
    if (type(module) is not lora_class or type(x) is not torch.Tensor
            or not _is_cuda_tensor(x) or x.dtype != torch.float32
            or x.ndim < 2 or not x.is_contiguous() or not x.numel()
            or not module.training or not torch.is_grad_enabled()
            or not _cuda_fp16_autocast_active() or not _plain_module(module)
            or module.disable_adapters or module.merged):
        return False
    hooks = torch.nn.modules.module
    if (hooks._global_forward_hooks or hooks._global_forward_pre_hooks
            or hooks._global_backward_hooks or hooks._global_backward_pre_hooks):
        return False
    active = module.active_adapters
    if len(active) != 1:
        return False
    name = active[0]
    if (name not in module.lora_A or name not in module.lora_B
            or name not in module.lora_dropout or name in module.lora_variant
            or module.r.get(name) != 16 or module.lora_alpha.get(name) != 32
            or module.scaling.get(name) != 2.0):
        return False
    base = module.base_layer
    if (type(base) is not linear_class or type(base).forward is not linear_forward
            or not _plain_module(base) or not base.training
            or base.compute_dtype != torch.float16 or not base.compute_type_is_set
            or base.weight.requires_grad or base.weight.dtype != torch.uint8
            or not _is_cuda_tensor(base.weight) or base.weight.device != x.device):
        return False
    state = getattr(base.weight, "quant_state", None)
    if (state is None or state.quant_type != "nf4" or state.dtype != torch.float16
            or tuple(state.shape) != (base.out_features, base.in_features)):
        return False
    bias = base.bias
    if (bias is not None and (bias.requires_grad or bias.device != x.device
            or bias.dtype not in (torch.float16, torch.float32))):
        return False
    adapter_a, adapter_b = module.lora_A[name], module.lora_B[name]
    for layer in (adapter_a, adapter_b):
        if (type(layer) is not torch.nn.Linear or type(layer).forward is not adapter_forward
                or not _plain_module(layer) or not layer.training or layer.bias is not None
                or layer.weight.dtype != torch.float32 or not layer.weight.requires_grad
                or layer.weight.device != x.device or not layer.weight.is_contiguous()):
            return False
    dropout = module.lora_dropout[name]
    if (type(dropout) is not torch.nn.Dropout or type(dropout).forward is not dropout_forward
            or not _plain_module(dropout) or not dropout.training
            or dropout.p != 0.05 or dropout.inplace):
        return False
    return (base.in_features == adapter_a.in_features == x.shape[-1]
            and adapter_a.out_features == adapter_b.in_features == 16
            and adapter_b.out_features == base.out_features)


def patch_lora_epilogue():
    """Install the optional pinned-stack patch; unsupported calls use PEFT."""
    try:
        supported = (metadata.version("peft") == "0.20.0"
                     and metadata.version("bitsandbytes") == "0.50.2"
                     and metadata.version("torch").split("+")[0] == "2.11.0")
    except metadata.PackageNotFoundError:
        return False
    if not supported:
        return False
    import bitsandbytes as bnb
    from peft.tuners.lora.bnb import Linear4bit

    original = Linear4bit.forward
    if getattr(original, "_soup_candidate_lora_epilogue", False):
        return True
    linear_class, linear_forward = bnb.nn.Linear4bit, bnb.nn.Linear4bit.forward
    adapter_forward, dropout_forward = torch.nn.Linear.forward, torch.nn.Dropout.forward

    @wraps(original)
    def forward(self, x, *args, **kwargs):
        # Do not partially run a stochastic branch and retry the reference.
        # Mixed adapters and variant kwargs delegate before touching the input.
        if (args or kwargs or not _eligible_lora(
                self, x, Linear4bit, linear_class, linear_forward,
                adapter_forward, dropout_forward)):
            return original(self, x, *args, **kwargs)
        self._check_forward_args(x)
        name = self.active_adapters[0]
        # Keep the dropout branch on the original FP32 input. Sharing this
        # half cast with dropout changes both rounding and gradient addition.
        base = self.base_layer(x.to(torch.float16))
        # The optional dropout-gradient patch normally enters this context in
        # PEFT's native forward. This fast path owns that call, so it must enter
        # explicitly to preserve composition instead of bypassing the patch.
        from .lora_dropout import lora_dropout_context
        with lora_dropout_context(self, x):
            adapter = self.lora_B[name](self.lora_A[name](self.lora_dropout[name](x)))
        return lora_epilogue(base, adapter)

    forward._soup_candidate_lora_epilogue = True
    Linear4bit.forward = forward
    return True
