"""Original private-FP16-state/native-CE replay; NOT an original Triton kernel.

Only loss saved-state ownership changes. The native head, returned logits,
Trainer metrics, Rewind, AMP/GradScaler and supplied denominator stay native.
Native summed CE is replayed at full B*T geometry, including ignored final rows.
This intentionally does not port the recovered H0 summary/output-suppression hook.
"""
from collections import Counter
import json
import weakref

import torch
from torch.nn import functional as F


class Telemetry:
    """Bounded metadata with immediate flushing, including before expensive work."""
    def __init__(self, stream=True):
        self.counts = Counter()
        self.reasons = Counter()
        self.stream = stream

    def emit(self, event, **fields):
        self.counts[event] += 1
        if "reason" in fields:
            self.reasons[str(fields["reason"])] += 1
        if self.stream:
            print("SOUP_HEAD_NATIVE", json.dumps(
                {"event": event, "count": self.counts[event], **fields},
                sort_keys=True, allow_nan=False), flush=True)

    def snapshot(self):
        return {"counts": dict(self.counts), "reasons": dict(self.reasons),
                "original_triton_kernels": [], "own_triton_share": 0.0}


def summed_ce(logits, targets):
    return F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                           targets.view(-1), ignore_index=-100, reduction="sum")


class _PrivateHalfSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, telemetry):
        telemetry.emit("native_sum_forward_enter", rows=targets.numel(),
                       vocabulary=logits.shape[-1], dtype=str(logits.dtype))
        value = summed_ce(logits, targets)
        ctx.save_for_backward(logits, targets)
        ctx.telemetry = telemetry
        ctx.set_materialize_grads(False)
        telemetry.emit("native_sum_forward_return")
        return value

    @staticmethod
    def backward(ctx, normalized_upstream):
        if normalized_upstream is None:
            ctx.telemetry.emit("unused_output")
            return None, None, None
        logits, targets = ctx.saved_tensors
        higher_order = torch.is_grad_enabled()
        ctx.telemetry.emit("native_sum_backward_enter", higher_order=higher_order)
        with torch.enable_grad():
            source = logits if higher_order else logits.detach().requires_grad_(True)
            replay = summed_ce(source, targets)
            gradient, = torch.autograd.grad(replay, source,
                                           grad_outputs=normalized_upstream,
                                           create_graph=higher_order)
        ctx.telemetry.emit("native_sum_backward_return")
        return gradient, None, None


class NativeHeadLoss:
    """A narrow per-model callable; unsupported calls use the SAME original logits.

    devices and expected_vocabulary are dependency-injection knobs for unit
    fixtures. apply() always selects cuda / 151936, never a test-detection flag.
    No value reads, finiteness scans, gather/compaction, head replay or RNG calls.
    """
    def __init__(self, native_loss, telemetry, owner=None,
                 expected_vocabulary=151936, devices=("cuda",)):
        self.native_loss = native_loss
        self.telemetry = telemetry
        self.owner = None if owner is None else weakref.ref(owner)
        self.expected_vocabulary = expected_vocabulary
        self.devices = tuple(devices)

    def __call__(self, logits, labels, vocab_size, num_items_in_batch=None,
                 ignore_index=-100, shift_labels=None, **kwargs):
        def native(reason):
            self.telemetry.emit("native_fallback", reason=reason)
            return self.native_loss(logits=logits, labels=labels, vocab_size=vocab_size,
                                    num_items_in_batch=num_items_in_batch,
                                    ignore_index=ignore_index, shift_labels=shift_labels,
                                    **kwargs)

        if self.owner is not None:
            owner = self.owner()
            if owner is None or not owner.training:
                return native("owner_or_eval")
            head = getattr(owner, "lm_head", None)
            if head is None or head.weight.requires_grad or head.bias is not None:
                return native("unfrozen_or_biased_head")
        if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
            return native("not_tensor")
        if not torch.is_grad_enabled() or not logits.requires_grad:
            return native("no_gradient_path")
        # These two pinned Qwen controls already affected its forward; pinned
        # ForCausalLMLoss accepts and ignores them. Other extras stay native.
        allowed = {"output_attentions", "output_hidden_states"}
        if (set(kwargs) - allowed or any(v is not None and type(v) is not bool
                                        for v in kwargs.values())
                or shift_labels is not None or ignore_index != -100):
            return native("alternate_loss_arguments")
        if (type(vocab_size) is not int or vocab_size != self.expected_vocabulary
                or logits.ndim != 3 or logits.shape[-1] != vocab_size
                or logits.shape[:-1] != labels.shape or logits.numel() == 0):
            return native("shape_or_vocabulary")
        if (logits.dtype != torch.float16 or labels.dtype != torch.int64
                or not logits.is_contiguous() or not labels.is_contiguous()
                or logits.device != labels.device or logits.device.type not in self.devices):
            return native("dtype_layout_or_device")
        if logits.is_cuda and torch.cuda.get_device_capability(logits.device) != (7, 5):
            return native("not_sm75")
        denominator = num_items_in_batch
        supported_count = (type(denominator) is int or
                           (isinstance(denominator, torch.Tensor)
                            and denominator.ndim == 0 and denominator.dtype == torch.int64
                            and not denominator.requires_grad
                            and (denominator.device.type == "cpu" or
                                 denominator.device == logits.device)))
        if not supported_count:
            return native("denominator_contract")
        # Native per-example final ignored row remains present. Native CE also
        # handles ignored NaN/Inf rows; never force their gradients to zero.
        targets = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
        total = _PrivateHalfSum.apply(logits, targets, self.telemetry)
        if isinstance(denominator, torch.Tensor):
            denominator = denominator.to(total.device)
        # Division remains OUTSIDE the custom node, exactly once. DivBackward
        # supplies q before native CE's FP32 adjoint narrows to the head's FP16.
        # Zero/negative D and nonfinite incoming q deliberately stay native.
        return total / denominator
