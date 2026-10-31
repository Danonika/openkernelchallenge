"""PROPOSAL: exact native CE replay on guarded rows, disjoint custom remainder.

Not target validated. This is a distinct algorithm, not a telemetry-only rerun.
Full-shape native NLL summation order is retained; never add two subset totals.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def soup_ce_mixed_backward(Z, Target, Maximum, LogSum, Q, NativeRow, Grad,
                           V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    # Uniform per-program branch. Native rows are not computed or stored here.
    if tl.load(NativeRow + row) == 0:
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
        tl.store(Grad + row * V + column, gradient, mask=real)


@dataclass(frozen=True)
class RowPartition:
    """Host counts from one classification read; no tensor values copied here."""
    native_rows: torch.Tensor | None
    native_supervised_rows: int
    safe_supervised_rows: int
    ignored_rows: int

    def counts(self):
        return {"native_supervised_rows": self.native_supervised_rows,
                "safe_supervised_rows": self.safe_supervised_rows,
                "ignored_rows": self.ignored_rows}


def partition_rows(selected, target, native_guard):
    """Keep every guard; consolidate classification into one 3-int host read.

    Masked logical operations avoid boolean-indexing selected (which can require
    a separate dynamic-size synchronization). Native nonzero is intentionally
    retained and still has a synchronization/allocation cost. No timing claim.
    """
    supervised = target.view(-1) != -100
    invalid_selected = torch.any(supervised & ~torch.isfinite(selected))
    native_count = native_guard.sum(dtype=torch.int64)
    ignored_count = (~supervised).sum(dtype=torch.int64)
    invalid, native_count, ignored_count = torch.stack((
        invalid_selected.to(torch.int64), native_count, ignored_count)).tolist()
    safe_count = target.numel() - native_count - ignored_count
    if invalid:
        return None, "invalid_selected_logp"
    if ignored_count == target.numel():
        return None, "no_supervised_rows"
    if safe_count == 0:
        return None, "all_supervised_rows_near_certain"
    # No allocation/nonzero for the ordinary all-safe route.
    rows = torch.nonzero(native_guard, as_tuple=False).flatten() if native_count else None
    return RowPartition(rows, native_count, safe_count, ignored_count), None


def mixed_selected_values(logits, target, selected, native_rows):
    """Native per-row CE only for the selected supervised rows; preserve row order."""
    flat = logits.view(-1, logits.shape[-1])
    native_logits = flat.index_select(0, native_rows)
    native_target = target.view(-1).index_select(0, native_rows)
    native_loss = F.cross_entropy(native_logits.float(), native_target,
                                 reduction="none", ignore_index=-100)
    values = selected.clone()
    values.index_copy_(0, native_rows, -native_loss)
    return values


def ordered_nll_sum(selected, target):
    flat_target = target.view(-1)
    internal_target = torch.where(flat_target != -100, torch.zeros_like(flat_target), flat_target)
    return F.nll_loss(selected[:, None], internal_target, reduction="sum", ignore_index=-100)


def _record(diagnostics, event, **values):
    if diagnostics is not None:
        diagnostics.record(event, **values)


class MixedRowSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, maximum, logsum, selected, native_guard, partition, diagnostics,
                summary_sink=None, token_entropy=None, first_argmax=None):
        ctx.input_count = len(ctx.needs_input_grad)
        native_rows = partition.native_rows
        values = mixed_selected_values(logits, target, selected, native_rows)
        total = ordered_nll_sum(values, target)
        if summary_sink is not None:
            from .metric_summary import publish
            publish(summary_sink, values, target, token_entropy, first_argmax, route="mixed_rows")
        ctx.save_for_backward(logits, target, maximum, logsum, native_guard, native_rows)
        ctx.diagnostics = diagnostics
        ctx.partition_counts = partition.counts()
        ctx.set_materialize_grads(False)
        _record(diagnostics, "triton_forward", route="mixed_rows", rows=target.numel(),
                **ctx.partition_counts,
                forward_sum="full_row_order_native_nll_not_subset_totals")
        return total

    @staticmethod
    def backward(ctx, normalized_upstream):
        if normalized_upstream is None:
            _record(ctx.diagnostics, "unused_backward")
            return (None,) * ctx.input_count
        logits, target, maximum, logsum, native_guard, native_rows = ctx.saved_tensors
        higher_order = torch.is_grad_enabled()
        finite = bool(torch.isfinite(normalized_upstream).all().item())
        if (higher_order or not finite or normalized_upstream.dtype != torch.float32
                or normalized_upstream.is_neg() or normalized_upstream.is_conj()):
            with torch.enable_grad():
                work = logits if higher_order else logits.detach().requires_grad_(True)
                native_sum = F.cross_entropy(work.float().view(-1, work.shape[-1]), target.view(-1),
                                            reduction="sum", ignore_index=-100)
                gradient, = torch.autograd.grad(native_sum, work, grad_outputs=normalized_upstream,
                                                create_graph=higher_order)
            _record(ctx.diagnostics, "native_backward_fallback",
                    reason="higher_order" if higher_order else "nonfinite_or_unsupported_upstream")
            return (gradient,) + (None,) * (ctx.input_count - 1)
        gradient = torch.empty_like(logits)
        q = normalized_upstream.contiguous()
        n, v = target.numel(), logits.shape[-1]
        soup_ce_mixed_backward[(n, triton.cdiv(v, 2048))](
            logits, target, maximum, logsum, q, native_guard, gradient,
            V=v, BLOCK=2048, num_warps=4, enable_fp_fusion=False)
        # Recompute native CE only for near-certain rows. Its FP32 adjoint narrows
        # through the ordinary .float() cast to FP16 BEFORE row insertion.
        with torch.enable_grad():
            native_logits = logits.detach().view(n, v).index_select(0, native_rows).requires_grad_(True)
            native_target = target.view(-1).index_select(0, native_rows)
            native_sum = F.cross_entropy(native_logits.float(), native_target,
                                        reduction="sum", ignore_index=-100)
            native_gradient, = torch.autograd.grad(native_sum, native_logits, grad_outputs=q)
        # Disjoint partition: no half-rounded gradient addition, no second divide.
        gradient.view(n, v).index_copy_(0, native_rows, native_gradient)
        _record(ctx.diagnostics, "triton_backward", route="mixed_rows",
                **ctx.partition_counts, phase="enqueued_not_completion")
        return (gradient,) + (None,) * (ctx.input_count - 1)
