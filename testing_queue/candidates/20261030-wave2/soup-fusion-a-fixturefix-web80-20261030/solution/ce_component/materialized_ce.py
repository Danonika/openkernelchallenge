"""Original experimental Triton materialized causal CE, v3 loss-boundary integration.

Design follows this project's independently authored CPU prototype, not any
third-party optimized kernel. Source semantics: transformers 93c8b7b4
loss_utils.py:32-71; pytorch 70d99e99 SoftMax.cu:39-63, Loss.cu:237-475.
Preparation is CPU/static only. Running the supplied notebook later launches CUDA.
V3 keeps the numerical kernel bodies unchanged and improves route evidence.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _IMPORT_ERROR = None
except (ImportError, RuntimeError, OSError) as error:
    triton = None
    tl = None
    _IMPORT_ERROR = f"{type(error).__name__}: {error}"


VOCABULARY = 151936
TILE = 2048
# Conservative numerical-route guard, not an acceptance tolerance. Tiny native
# target losses are cancellation-sensitive; route the entire call to native CE.
NEAR_CERTAIN_NLL = 1e-3


@dataclass
class Diagnostics:
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: str, **values: Any) -> None:
        self.events.append({"event": event, **values})


def _record(diagnostics, event, **values):
    if diagnostics is not None:
        diagnostics.record(event, **values)


def backend_info():
    """Import metadata only: does not query CUDA or initialize a driver."""
    return {"triton_imported": triton is not None,
            "triton_version": getattr(triton, "__version__", None),
            "torch_version": torch.__version__, "import_error": _IMPORT_ERROR}


if triton is not None:
    @triton.jit
    def soup_ce_exp_tile_max(Z, Target, Partial, Bad, V: tl.constexpr,
                             NT: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        column = tile * BLOCK + tl.arange(0, BLOCK)
        real = column < V
        value = tl.load(Z + row * V + column, mask=real, other=-float("inf")).to(tl.float32)
        finite = (value == value) & (tl.abs(value) != float("inf"))
        invalid_values = tl.sum((real & ~finite).to(tl.int32), axis=0)
        target = tl.load(Target + row)
        invalid_target = (target != -100) & ((target < 0) | (target >= V))
        bad = invalid_values + tl.where(tile == 0, invalid_target.to(tl.int32), 0)
        tl.store(Partial + row * NT + tile, tl.max(value, axis=0))
        tl.store(Bad + row * NT + tile, bad)

    @triton.jit
    def soup_ce_exp_row_max(Partial, Maximum, NT: tl.constexpr, MERGE: tl.constexpr):
        row = tl.program_id(0)
        index = tl.arange(0, MERGE)
        values = tl.load(Partial + row * NT + index, mask=index < NT, other=-float("inf"))
        tl.store(Maximum + row, tl.max(values, axis=0))

    @triton.jit
    def soup_ce_exp_tile_sum(Z, Maximum, Partial, V: tl.constexpr,
                             NT: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        column = tile * BLOCK + tl.arange(0, BLOCK)
        real = column < V
        value = tl.load(Z + row * V + column, mask=real, other=-float("inf")).to(tl.float32)
        maximum = tl.load(Maximum + row)
        exponential = tl.exp(value - maximum)
        tl.store(Partial + row * NT + tile, tl.sum(tl.where(real, exponential, 0.0), axis=0))

    @triton.jit
    def soup_ce_exp_selected(Z, Target, Maximum, Partial, LogSum, Selected, Guard,
                             V: tl.constexpr, NT: tl.constexpr, MERGE: tl.constexpr,
                             NEAR: tl.constexpr):
        row = tl.program_id(0)
        index = tl.arange(0, MERGE)
        values = tl.load(Partial + row * NT + index, mask=index < NT, other=0.0)
        logsum = tl.log(tl.sum(values, axis=0))
        maximum = tl.load(Maximum + row)
        target = tl.load(Target + row)
        supervised = target != -100
        safe_target = tl.where(supervised, target, 0)
        selected_value = tl.load(Z + row * V + safe_target).to(tl.float32)
        logp = (selected_value - maximum) - logsum
        tl.store(LogSum + row, logsum)
        tl.store(Selected + row, tl.where(supervised, logp, 0.0))
        finite = (logp == logp) & (tl.abs(logp) != float("inf"))
        unsafe = supervised & ((logp >= -NEAR) | ~finite)
        tl.store(Guard + row, unsafe.to(tl.int32))

    @triton.jit
    def soup_ce_exp_backward(Z, Target, Maximum, LogSum, Q, Grad,
                             V: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        column = tile * BLOCK + tl.arange(0, BLOCK)
        real = column < V
        value = tl.load(Z + row * V + column, mask=real, other=0.0).to(tl.float32)
        maximum = tl.load(Maximum + row)
        logsum = tl.load(LogSum + row)
        q = tl.load(Q).to(tl.float32)
        target = tl.load(Target + row)
        adjoint = tl.where(target != -100, -q, 0.0)
        logp = (value - maximum) - logsum
        product = tl.exp(logp) * adjoint
        sparse = tl.where(column == target, adjoint, 0.0)
        gradient = sparse - product
        # Grad is FP16: narrowing happens only here, after normalized scaling.
        tl.store(Grad + row * V + column, gradient, mask=real)


def _targets(labels, ignore_index=-100):
    return F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()


def native_causal_loss(logits, labels, num_items_in_batch=None, *, ignore_index=-100, shift_labels=None):
    """Unpatched native expression used by unsupported/exceptional calls."""
    target = _targets(labels, ignore_index) if shift_labels is None else shift_labels
    source = logits.float().view(-1, logits.shape[-1])
    target = target.view(-1).to(source.device)
    total = F.cross_entropy(source, target, ignore_index=ignore_index,
                           reduction="mean" if num_items_in_batch is None else "sum")
    if num_items_in_batch is not None:
        denominator = num_items_in_batch
        if isinstance(denominator, torch.Tensor):
            denominator = denominator.to(total.device)
        total = total / denominator
    return total


def _denominator_metadata(value, device):
    """One host read at most; preserve the original object for external SUM / D."""
    if type(value) is int:
        return (value, "python_int") if 0 < value <= torch.iinfo(torch.int64).max else None
    if (isinstance(value, torch.Tensor) and value.dtype == torch.int64
            and value.ndim == 0 and not value.requires_grad
            and (value.device.type == "cpu" or value.device == device)):
        count = int(value.item())
        return (count, "int64_" + value.device.type) if count > 0 else None
    return None


def _launch_config():
    # No autotuning, hidden warmup, precompiled cache or CUDA graph.
    return {"num_warps": 4, "enable_fp_fusion": False}


class _TritonSumCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, maximum, logsum, selected, diagnostics):
        n = target.numel()
        nt = triton.cdiv(VOCABULARY, TILE)
        flat_target = target.view(-1)
        internal_target = torch.where(flat_target != -100, torch.zeros_like(flat_target), flat_target)
        # The table already contains selected full-vocabulary log probabilities.
        # This is NLL, never a second cross-entropy/softmax over one class.
        total = F.nll_loss(selected[:, None], internal_target, reduction="sum", ignore_index=-100)
        ctx.save_for_backward(logits, target, maximum, logsum)
        ctx.diagnostics = diagnostics
        ctx.set_materialize_grads(False)
        _record(diagnostics, "triton_forward", rows=n, vocabulary=VOCABULARY, tiles=nt)
        return total

    @staticmethod
    def backward(ctx, normalized_upstream):
        if normalized_upstream is None:
            _record(ctx.diagnostics, "unused_backward")
            return None, None, None, None, None, None
        logits, target, maximum, logsum = ctx.saved_tensors
        higher_order = torch.is_grad_enabled()
        finite = bool(torch.isfinite(normalized_upstream).all().item())
        if (higher_order or not finite or normalized_upstream.dtype != torch.float32
                or normalized_upstream.is_neg() or normalized_upstream.is_conj()):
            with torch.enable_grad():
                work = logits if higher_order else logits.detach().requires_grad_(True)
                native_sum = F.cross_entropy(work.float().view(-1, work.shape[-1]), target.view(-1),
                                            reduction="sum", ignore_index=-100)
                (gradient,) = torch.autograd.grad(native_sum, work, grad_outputs=normalized_upstream,
                                                  create_graph=higher_order)
            _record(ctx.diagnostics, "native_backward_fallback",
                    reason="higher_order" if higher_order else "nonfinite_or_unsupported_upstream")
            return gradient, None, None, None, None, None
        gradient = torch.empty_like(logits)
        q = normalized_upstream.contiguous()
        n, nt = target.numel(), triton.cdiv(VOCABULARY, TILE)
        soup_ce_exp_backward[(n, nt)](logits, target, maximum, logsum, q, gradient,
                                      V=VOCABULARY, BLOCK=TILE, **_launch_config())
        _record(ctx.diagnostics, "triton_backward", rows=n, phase="enqueued_not_completion")
        return gradient, None, None, None, None, None


def materialized_causal_loss(logits, labels, num_items_in_batch=None, *,
                            ignore_index=-100, shift_labels=None, diagnostics=None, native_fallback=None):
    """No model/forward/trainer patch. Adapter fallback delegates to captured native loss."""
    def fallback(reason):
        _record(diagnostics, "native_forward_fallback", reason=reason)
        if native_fallback is not None:
            return native_fallback(reason)
        return native_causal_loss(logits, labels, num_items_in_batch,
                                  ignore_index=ignore_index, shift_labels=shift_labels)

    if not logits.is_cuda:
        return fallback("not_cuda")
    if triton is None:
        return fallback("backend_unavailable")
    if not torch.is_grad_enabled() or not logits.requires_grad:
        return fallback("no_gradient_path")
    if logits.dtype != torch.float16 or logits.is_neg() or logits.is_conj() or not logits.is_contiguous():
        return fallback("dtype_or_stride")
    if (logits.ndim != 3 or logits.shape[0] != 4 or not 1 <= logits.shape[1] <= 512
            or logits.shape[2] != VOCABULARY or labels.shape != logits.shape[:-1]):
        return fallback("shape")
    if (labels.dtype != torch.int64 or labels.device != logits.device
            or ignore_index != -100 or shift_labels is not None):
        return fallback("label_contract")
    if torch.cuda.get_device_capability(logits.device) != (7, 5):
        return fallback("not_sm75")
    denominator_metadata = _denominator_metadata(num_items_in_batch, logits.device)
    if denominator_metadata is None:
        return fallback("denominator")
    count, kind = denominator_metadata
    _record(diagnostics, "denominator_validated", count=count, kind=kind)

    _record(diagnostics, "triton_prepass_started", phase="before_first_enqueue")
    target = _targets(labels)
    n, nt = target.numel(), triton.cdiv(VOCABULARY, TILE)
    partial = torch.empty((n, nt), device=logits.device, dtype=torch.float32)
    bad = torch.empty((n, nt), device=logits.device, dtype=torch.int32)
    maximum = torch.empty(n, device=logits.device, dtype=torch.float32)
    with torch.no_grad():
        soup_ce_exp_tile_max[(n, nt)](logits, target, partial, bad,
                                     V=VOCABULARY, NT=nt, BLOCK=TILE, **_launch_config())
        # Small flags only. This host read synchronizes and is a real cost.
        invalid = bool(torch.any(bad).item())
        if not invalid:
            soup_ce_exp_row_max[(n,)](partial, maximum, NT=nt,
                                      MERGE=triton.next_power_of_2(nt), **_launch_config())
    # Outside the prepass no-grad context: fallback must build the native graph.
    if invalid:
        return fallback("nonfinite_logits_or_invalid_target")
    logsum = torch.empty(n, device=logits.device, dtype=torch.float32)
    selected = torch.empty_like(logsum)
    guard = torch.empty(n, device=logits.device, dtype=torch.int32)
    with torch.no_grad():
        soup_ce_exp_tile_sum[(n, nt)](logits, maximum, partial,
                                     V=VOCABULARY, NT=nt, BLOCK=TILE, **_launch_config())
        soup_ce_exp_selected[(n,)](logits, target, maximum, partial, logsum, selected, guard,
                                   V=VOCABULARY, NT=nt, MERGE=triton.next_power_of_2(nt),
                                   NEAR=NEAR_CERTAIN_NLL, **_launch_config())
        near_certain = bool(torch.any(guard).item())
    # No partial-row mixing, clamping or fake zero loss. Original full geometry
    # is recomputed with native autograd, outside no_grad, including ignored rows.
    if near_certain:
        return fallback("near_certain_or_invalid_selected_logp")
    total = _TritonSumCE.apply(logits, target, maximum, logsum, selected, diagnostics)
    denominator = num_items_in_batch
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.to(total.device)
    return total / denominator
