"""Optional within-batch removal of unused rows from frozen Qwen2 MLPs.

This implements P1 from the user's ``Qwen_Kernel_Optimization_Review.md``:
attention, positions, q/v LoRA and its original rectangular dropout calls
stay unchanged. Only the deterministic gate/up/SwiGLU/down computation sees
the valid rows. A compact row is always mapped back to its original (b, t).

The caller must own private training outputs, pass the original labels/mask,
and suppress padded logits/hidden states from public consumers. Consequently
this is opt-in even after installation. It is not an inference/model-wide
unpadding patch. Public calls and unsupported masks retain the complete MLP.

The two row-copy kernels below are original implementations, not adaptations
of a library kernel. They replace gather/scatter work used by training. The
MLP arithmetic, including the separately cast gate/up gradient branches,
remains in the previously installed native/credited-activation implementation.
Changing GEMM row geometry requires physical-device gradient/quality checks;
neither bitwise GEMM parity nor a speed improvement is claimed.

Masked padding has no mathematical contribution for the finite right-padded
training path. As with other padding-removal schemes, arbitrary NaN/Inf
propagation through unused padded queries is not an equivalence guarantee.
This experiment must remain disabled if its target training produces them.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from importlib import metadata
import os

import torch
import triton
import triton.language as tl


_ACTIVE_MLP_PADDING = ContextVar("soup_mlp_padding", default=None)
_QWEN_CAUSAL_LM = None
_QWEN_MLP = None
_BNB_LINEAR = None
_BNB_FORWARD = None
_SILU_CLASS = None
_SILU_FORWARD = None
_BLOCK = 1024


@triton.jit(do_not_specialize=["N"])
def soup_mlp_gather_rows(X, ROWS, OUT, N, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < N
    row = offset // WIDTH
    column = offset % WIDTH
    source_row = tl.load(ROWS + row, active, other=0)
    value = tl.load(X + source_row * WIDTH + column, active, other=0)
    tl.store(OUT + offset, value, active)


@triton.jit(do_not_specialize=["N"])
def soup_mlp_scatter_rows(X, INVERSE, OUT, N, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < N
    row = offset // WIDTH
    column = offset % WIDTH
    source_row = tl.load(INVERSE + row, active, other=-1)
    # Every output location has one writer; padding is initialized here, not
    # by a second memset, and no atomic gradient accumulation is needed.
    value = tl.load(X + tl.maximum(source_row, 0) * WIDTH + column,
                    active & (source_row >= 0), other=0)
    tl.store(OUT + offset, value, active)


def _gather(x, rows):
    width = x.shape[-1]
    out = torch.empty((rows.numel(), width), dtype=x.dtype, device=x.device)
    soup_mlp_gather_rows[(triton.cdiv(out.numel(), _BLOCK),)](
        x, rows, out, out.numel(), width, _BLOCK, num_warps=4)
    return out


def _scatter(x, inverse):
    width = x.shape[-1]
    out = torch.empty((inverse.numel(), width), dtype=x.dtype, device=x.device)
    soup_mlp_scatter_rows[(triton.cdiv(out.numel(), _BLOCK),)](
        x, inverse, out, out.numel(), width, _BLOCK, num_warps=4)
    return out


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rows, inverse):
        ctx.save_for_backward(rows, inverse)
        ctx.input_shape = x.shape
        ctx.set_materialize_grads(False)
        return _gather(x, rows)

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None, None
        rows, inverse = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Preserve higher-order differentiation for the row permutation;
            # the installed MLP's own higher-order contract is unchanged.
            out = grad.new_zeros((inverse.numel(), grad.shape[-1]))
            out = out.index_copy(0, rows, grad)
        else:
            out = _scatter(grad.resolve_conj().resolve_neg().contiguous(), inverse)
        return out.reshape(ctx.input_shape), None, None


class _ScatterRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rows, inverse, output_shape):
        ctx.save_for_backward(rows)
        ctx.set_materialize_grads(False)
        return _scatter(x, inverse).reshape(output_shape)

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None, None, None
        (rows,) = ctx.saved_tensors
        if torch.is_grad_enabled():
            out = grad.reshape(-1, grad.shape[-1]).index_select(0, rows)
        else:
            out = _gather(grad.resolve_conj().resolve_neg().contiguous(), rows)
        return out, None, None, None


@dataclass
class _PaddingCall:
    shape: tuple
    mask: torch.Tensor
    mask_version: int
    rows: torch.Tensor
    inverse: torch.Tensor
    mlp_ids: frozenset
    lengths: tuple = ()
    attention_ids: frozenset = frozenset()
    labels: object = None
    labels_version: object = None
    row_versions: tuple = ()


class _RowPlan(tuple):
    """Keep the established two-map interface and its CPU length metadata."""
    def __new__(cls, rows, inverse, lengths):
        result = super().__new__(cls, (rows, inverse))
        result.lengths = lengths
        return result


def _supported_device(tensor):
    return (os.environ.get("TRITON_INTERPRET") != "1" and tensor.device.type == "cuda"
            and torch.cuda.get_device_capability(tensor.device) == (7, 5)
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16)


def _plain(module):
    return ("forward" not in module.__dict__ and not hasattr(module, "_hf_hook")
            and not module._forward_hooks and not module._forward_pre_hooks
            and not module._backward_hooks and not module._backward_pre_hooks)


def _frozen_mlp(module, device):
    """Exclude adapters, stochastic overrides, and trainable MLP projections."""
    if (type(module) is not _QWEN_MLP or not _plain(module)
            or not module.training
            or getattr(getattr(module, "config", None), "hidden_act", None) != "silu"):
        return False
    activation = getattr(module, "act_fn", None)
    if (type(activation) is not _SILU_CLASS or not _plain(activation)
            or type(activation).forward is not _SILU_FORWARD):
        return False
    projections = [getattr(module, name, None)
                   for name in ("gate_proj", "up_proj", "down_proj")]
    for layer in projections:
        if (type(layer) is not _BNB_LINEAR or type(layer).forward is not _BNB_FORWARD
                or not _plain(layer) or not layer.training or layer.bias is not None
                or layer.weight.requires_grad or layer.weight.dtype != torch.uint8
                or layer.weight.device != device or layer.compute_dtype != torch.float16
                or not layer.compute_type_is_set):
            return False
        state = getattr(layer.weight, "quant_state", None)
        if (state is None or state.quant_type != "nf4" or state.dtype != torch.float16
                or tuple(state.shape) != (layer.out_features, layer.in_features)):
            return False
    gate, up, down = projections
    return (gate.in_features == up.in_features == down.out_features
            and gate.out_features == up.out_features == down.in_features)


def _host_row_plan(mask, labels):
    """One bounded GPU-to-host transfer per forward, never per MLP layer.

    The supported recipe has at most 4*512 positions, but there is no data-
    dependent benchmark branch or hardcoded batch shape. CPU planning derives
    the exact shape from these call inputs. An eventual CPU-collator ticket
    could avoid this transfer; this implementation does not alter a collator.
    """
    snapshot = torch.stack((mask, labels), dim=0).detach().to(device="cpu")
    host_mask, host_labels = snapshot.unbind(0)
    if not bool(((host_mask == 0) | (host_mask == 1)).all()):
        return None
    valid = host_mask == 1
    if (not bool(valid[:, 0].all())
            or bool((valid[:, 1:] & ~valid[:, :-1]).any())
            or bool((host_labels[~valid] != -100).any())
            or bool(valid.all())):
        return None
    rows = valid.flatten().nonzero(as_tuple=False).flatten()
    inverse = torch.full((valid.numel(),), -1, dtype=torch.long)
    inverse.index_copy_(0, rows, torch.arange(rows.numel(), dtype=torch.long))
    lengths = tuple(int(value) for value in valid.sum(dim=1).tolist())
    return _RowPlan(rows.to(mask.device), inverse.to(mask.device), lengths)


def _make_call(causal_lm, mask, labels, *, private_outputs, use_cache,
               past_key_values, output_hidden_states, output_attentions):
    if (private_outputs is not True or _QWEN_CAUSAL_LM is None
            or type(causal_lm) is not _QWEN_CAUSAL_LM or not causal_lm.training
            or not torch.is_grad_enabled() or torch.is_inference_mode_enabled()
            or use_cache is not False or past_key_values is not None
            or torch.jit.is_tracing() or torch.jit.is_scripting()
            or torch.compiler.is_compiling()):
        return None
    config = causal_lm.config
    if ((output_hidden_states if output_hidden_states is not None else
         getattr(config, "output_hidden_states", False))
            or (output_attentions if output_attentions is not None else
                getattr(config, "output_attentions", False))):
        return None
    if (type(mask) is not torch.Tensor or type(labels) is not torch.Tensor
            or mask.dtype not in (torch.bool, torch.int32, torch.int64)
            or labels.dtype != torch.long or mask.ndim != 2 or labels.shape != mask.shape
            or min(mask.shape) < 1 or labels.device != mask.device
            or mask.requires_grad or mask.is_inference() or not _supported_device(mask)):
        return None
    decoder = causal_lm.model
    if (not decoder.training or getattr(decoder, "gradient_checkpointing", False)
            or getattr(decoder, "has_sliding_layers", False)
            or getattr(config, "_attn_implementation", None) != "sdpa"):
        return None
    hooks = torch.nn.modules.module
    if (hooks._global_forward_hooks or hooks._global_forward_pre_hooks
            or hooks._global_backward_hooks or hooks._global_backward_pre_hooks):
        return None
    mlps = [layer.mlp for layer in decoder.layers]
    if not mlps or not all(_frozen_mlp(module, mask.device) for module in mlps):
        return None
    plan = _host_row_plan(mask, labels)
    if plan is None:
        return None
    return _PaddingCall(tuple(mask.shape), mask, mask._version, *plan,
                        frozenset(id(module) for module in mlps),
                        getattr(plan, "lengths", ()),
                        frozenset(id(layer.self_attn) for layer in decoder.layers),
                        labels, labels._version, tuple(value._version for value in plan))


@contextmanager
def mlp_padding_context(causal_lm, *, attention_mask, labels,
                        private_outputs=False, use_cache=None, past_key_values=None,
                        output_hidden_states=None, output_attentions=None):
    """Enter around one existing private training-model/decoder call.

    Install ``patch_mlp_padding`` after any compact-activation patch first.
    Pass the exact original 2-D mask and unshifted labels, and the actual cache
    and output flags. A None cache flag inherits model.config.use_cache for
    admission only, without changing the arguments sent to the native model.
    Only the compact-head owner may set private_outputs=True.
    Do not enter this context for calls that return public full logits. Nothing
    changes labels, row order, masks, denominators, metrics or Rewind payloads.
    Nested unsupported calls clear, rather than inherit, an outer row plan.
    """
    effective_cache = (getattr(getattr(causal_lm, "config", None), "use_cache", None)
                       if use_cache is None else use_cache)
    call = _make_call(causal_lm, attention_mask, labels,
        private_outputs=private_outputs, use_cache=effective_cache,
        past_key_values=past_key_values, output_hidden_states=output_hidden_states,
        output_attentions=output_attentions)
    token = _ACTIVE_MLP_PADDING.set(call)
    try:
        yield call
    finally:
        if hasattr(call, "_soup_varlen_attention_state"):
            from .varlen_attention_patch import clear_varlen_attention_call
            clear_varlen_attention_call(call)
        _ACTIVE_MLP_PADDING.reset(token)


def patch_mlp_padding():
    """Install an idempotent wrapper; no work is selected without a context."""
    global _QWEN_CAUSAL_LM, _QWEN_MLP, _BNB_LINEAR, _BNB_FORWARD
    global _SILU_CLASS, _SILU_FORWARD
    try:
        if (metadata.version("transformers") != "5.16.1"
                or metadata.version("bitsandbytes") != "0.50.2"
                or metadata.version("torch").split("+")[0] != "2.11.0"):
            return False
        import bitsandbytes as bnb
        from transformers.activations import SiLUActivation
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2ForCausalLM
    except (ImportError, AttributeError, metadata.PackageNotFoundError):
        return False
    if not all(isinstance(cls, type) and issubclass(cls, torch.nn.Module)
               for cls in (Qwen2MLP, Qwen2ForCausalLM, bnb.nn.Linear4bit, SiLUActivation)):
        return False
    original = Qwen2MLP.forward
    if getattr(original, "_soup_candidate_mlp_padding", False):
        return True
    # The known implementations contain only deterministic MLP operations.
    # Do not compact an arbitrary user's replacement forward by assumption.
    if (not getattr(original, "_soup_candidate_compact_activations", False)
            and getattr(original, "__module__", None) not in (
                Qwen2MLP.__module__, __package__ + ".swiglu")):
        return False
    _QWEN_CAUSAL_LM = Qwen2ForCausalLM
    _QWEN_MLP = Qwen2MLP
    _BNB_LINEAR = bnb.nn.Linear4bit
    _BNB_FORWARD = bnb.nn.Linear4bit.forward
    _SILU_CLASS = SiLUActivation
    _SILU_FORWARD = SiLUActivation.forward

    @wraps(original)
    def forward(self, x):
        call = _ACTIVE_MLP_PADDING.get()
        if (call is None or id(self) not in call.mlp_ids or type(x) is not torch.Tensor
                or x.ndim != 3 or tuple(x.shape[:2]) != call.shape
                or x.device != call.rows.device or x.dtype != torch.float32
                or not x.requires_grad or not x.is_contiguous() or x.is_neg() or x.is_conj()
                or call.mask._version != call.mask_version or not _supported_device(x)
                or not _frozen_mlp(self, x.device)):
            return original(self, x)
        selected = _GatherRows.apply(x, call.rows, call.inverse)
        compact = original(self, selected)
        # All eligible native/credited paths obey this contract. Never replay
        # a completed MLP silently if a later replacement changes its result.
        if (compact.shape != selected.shape or compact.dtype != x.dtype
                or compact.device != x.device or not compact.is_contiguous()):
            raise RuntimeError("the admitted MLP changed its output shape, dtype or layout")
        return _ScatterRows.apply(compact, call.rows, call.inverse, x.shape)

    forward._soup_candidate_mlp_padding = True
    Qwen2MLP.forward = forward
    return True
