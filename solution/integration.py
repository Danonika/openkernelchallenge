"""Preserve Soup/TRL metrics while optimizing the frozen Qwen2 training head.

Contracts: transformers 5.16.1 Trainer and Qwen2ForCausalLM; TRL 0.29.1
SFTTrainer; Soup 0.77.0 rewind_hf. The call remains Trainer -> PEFT ->
Accelerate -> Qwen2. No task settings or Soup feature switches are changed.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from types import SimpleNamespace

import torch

from .settings import (
    CHUNK_TOKENS, HEAD_PROJECTION, SPARSE_BACKWARD,
    HEAD_MODE, CACHED_CE_CHUNK_TOKENS,
)


@dataclass
class _HeadCall:
    causal_lm: object
    result: object = None


_HEAD_CALL = ContextVar("soup_candidate_head_call", default=None)
_REWIND_STEP = "_soup_chunked_step"  # Existing Rewind per-token-loss interface.


def _cached_weight_fp16(causal_lm):
    """Cache a frozen cast, without adding parameters or persistent buffers."""
    weight = causal_lm.lm_head.weight
    if weight.requires_grad:
        raise ValueError("The streamed output head must remain frozen")
    key = (id(weight), weight.data_ptr(), weight._version,
           tuple(weight.shape), weight.device, weight.dtype)
    cached = causal_lm.__dict__.get("_candidate_head_cast")
    if cached is None or cached[0] != key:
        cached = (key, weight.detach().to(dtype=torch.float16).contiguous())
        causal_lm.__dict__["_candidate_head_cast"] = cached
    return cached[1]


def _forward_streamed(causal_lm, *, input_ids=None, attention_mask=None,
                      position_ids=None, past_key_values=None, inputs_embeds=None,
                      labels=None, use_cache=None, logits_to_keep=0, **kwargs):
    """One decoder call followed by the original streamed loss and metrics."""
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from .streamed_head import streamed_linear_ce

    if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
        raise ValueError("Streamed training requires the full token sequence")
    decoded = causal_lm.model(
        input_ids=input_ids, attention_mask=attention_mask,
        position_ids=position_ids, past_key_values=past_key_values,
        inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
    )
    result = streamed_linear_ce(
        decoded.last_hidden_state, _cached_weight_fp16(causal_lm), labels,
        num_items_in_batch=kwargs.get("num_items_in_batch"),
        chunk_size=CHUNK_TOKENS, projection=HEAD_PROJECTION,
        sparse_backward=SPARSE_BACKWARD,
    )
    outputs = CausalLMOutputWithPast(
        loss=result.loss, logits=None, past_key_values=decoded.past_key_values,
        hidden_states=decoded.hidden_states, attentions=decoded.attentions,
    )
    return outputs, result


def _forward_cached(causal_lm, original_forward, **kwargs):
    """Run Qwen's original forward once, changing only its per-call loss slot.

    Keep the native full-shape lm_head/autograd graph.  Suppress the private
    logits before Accelerate's outward FP32 conversion because every supported
    training consumer is supplied with the true compact statistics instead.
    Outside this scoped call (including ordinary inference), outputs are native.
    """
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from .cached_head import cached_logits_ce

    absent = object()
    saved_override = causal_lm.__dict__.get("_loss_function", absent)
    result = None

    def compact_loss(logits, labels, vocab_size, num_items_in_batch=None, **loss_kwargs):
        nonlocal result
        if loss_kwargs.get("shift_labels") is not None:
            raise RuntimeError("cached causal head does not support pre-shifted labels")
        if logits.shape[-1] != vocab_size:
            raise RuntimeError("native output head and vocabulary size differ")
        if result is not None:
            raise RuntimeError("expected exactly one native causal head loss")
        result = cached_logits_ce(logits, labels, num_items_in_batch,
                                   chunk_size=CACHED_CE_CHUNK_TOKENS)
        return result.loss

    causal_lm.loss_function = compact_loss
    try:
        native = original_forward(causal_lm, **kwargs)
    finally:
        # Preserve identity and the existence/absence of a prior property value,
        # including on a decoder/loss failure.  Never retry after dropout.
        if saved_override is absent:
            causal_lm.__dict__.pop("_loss_function", None)
        else:
            causal_lm.__dict__["_loss_function"] = saved_override
    if result is None:
        raise RuntimeError("native Qwen forward did not execute the cached loss")
    if not isinstance(native, CausalLMOutputWithPast):
        raise RuntimeError("expected the normal Qwen causal model output")
    # A fresh ModelOutput also clears its mapping; assigning .logits=None alone
    # would leave the old tensor in the dict and invite an FP32 output copy.
    outputs = CausalLMOutputWithPast(
        loss=native.loss, logits=None, past_key_values=native.past_key_values,
        hidden_states=native.hidden_states, attentions=native.attentions,
    )
    return outputs, result


def _supported_causal_lm(trainer, model, inputs, qwen_class):
    """Capability guards, independent of seeds, data, tests, or measurement."""
    if not isinstance(model, torch.nn.Module) or not model.training or not isinstance(inputs, dict):
        return None
    labels, ids, mask = (inputs.get(k) for k in ("labels", "input_ids", "attention_mask"))
    if not all(isinstance(t, torch.Tensor) for t in (labels, ids, mask)):
        return None
    if not (labels.is_cuda and ids.is_cuda and mask.is_cuda):
        return None
    if labels.ndim != 2 or labels.shape != ids.shape or labels.shape != mask.shape:
        return None
    if labels.shape[1] < 2 or labels.numel() == 0:
        return None
    allowed = {"input_ids", "labels", "attention_mask", "position_ids", "use_cache",
               "_prediction_loss_only", "num_items_in_batch"}
    if set(inputs) - allowed:
        return None
    args = getattr(trainer, "args", None)
    if not getattr(args, "fp16", False) or getattr(args, "bf16", False):
        return None
    if getattr(args, "loss_type", None) != "nll":
        return None
    if getattr(args, "use_liger_kernel", False) or getattr(args, "packing", False):
        return None
    if getattr(trainer, "padding_free", False) or getattr(trainer, "num_virtual_tokens", 0):
        return None
    if getattr(trainer, "aux_loss_enabled", False):
        return None
    if getattr(trainer, "label_smoother", None) is not None or getattr(trainer, "compute_loss_func", None) is not None:
        return None
    if getattr(getattr(trainer, "accelerator", None), "num_processes", 1) != 1:
        return None
    if not hasattr(trainer, "_metrics") or not hasattr(trainer, "_total_train_tokens"):
        return None
    get_base = getattr(model, "get_base_model", None)
    causal_lm = get_base() if callable(get_base) else model
    if not isinstance(causal_lm, qwen_class):
        return None
    if not getattr(causal_lm.config, "return_dict", True):
        return None
    if HEAD_MODE not in ("cached", "streamed"):
        return None
    # A custom model loss must remain its owner's behavior.  The Qwen default
    # property is resolved lazily and can have an explicit identical override.
    from transformers.loss.loss_utils import ForCausalLMLoss
    if causal_lm.loss_function is not ForCausalLMLoss:
        return None
    head = getattr(causal_lm, "lm_head", None)
    if type(head) is not torch.nn.Linear or head.bias is not None:
        return None
    weight = head.weight
    if (weight.requires_grad or not weight.is_cuda or not weight.is_contiguous()
            or weight.device != labels.device or weight.dtype not in (torch.float16, torch.float32)):
        return None
    if HEAD_MODE == "streamed" and HEAD_PROJECTION == "t4_fused" and (
            weight.shape[1] % 32
            or (weight.dtype == torch.float16 and weight.data_ptr() % 4)):
        # Reject unsupported fused shapes before the decoder consumes dropout
        # RNG. The fixed task's width1536 and allocated head satisfy this.
        return None
    return causal_lm


def _log_statistics(trainer, inputs, result):
    """TRL's unchanged entropy mask, supervised accuracy, and token counting."""
    with torch.no_grad():
        gather = trainer.accelerator.gather_for_metrics
        attention_mask = inputs["attention_mask"]
        entropy = (result.entropy * attention_mask).sum() / attention_mask.sum()
        trainer._metrics["train"]["entropy"].append(gather(entropy).mean().item())
        tokens = gather(attention_mask.sum()).sum().item()
        trainer._total_train_tokens += tokens
        trainer._metrics["train"]["num_tokens"] = [trainer._total_train_tokens]
        targets = inputs["labels"][..., 1:]
        supervised = targets != -100
        correct = ((result.predictions[..., :-1] == targets) & supervised).sum()
        correct = gather(correct)
        count = gather(supervised.sum()).sum()
        accuracy = (correct.sum() / count).item() if count > 0 else 0.0
        trainer._metrics["train"]["mean_token_accuracy"].append(accuracy)


def patch_training_head():
    """Idempotent patch; unsupported model/trainer calls retain their behavior."""
    from transformers import Trainer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from trl import SFTTrainer

    if not isinstance(Qwen2ForCausalLM, type) or not issubclass(Qwen2ForCausalLM, torch.nn.Module):
        return False
    if not isinstance(SFTTrainer, type) or not issubclass(SFTTrainer, Trainer):
        return False
    if getattr(SFTTrainer.compute_loss, "_soup_candidate_head", False):
        return True
    original_forward = Qwen2ForCausalLM.forward
    original_compute_loss = SFTTrainer.compute_loss

    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        call = _HEAD_CALL.get()
        if call is None or call.causal_lm is not self:
            return original_forward(self, *args, **kwargs)
        if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16:
            raise RuntimeError("Supported optimized training requires CUDA FP16 autocast")
        if args or kwargs.get("labels") is None:
            raise RuntimeError("Expected the standard labelled Qwen2 keyword call")
        if HEAD_MODE == "cached":
            outputs, call.result = _forward_cached(self, original_forward, **kwargs)
        else:
            outputs, call.result = _forward_streamed(self, **kwargs)
        return outputs

    @wraps(original_compute_loss)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        causal_lm = _supported_causal_lm(self, model, inputs, Qwen2ForCausalLM)
        if causal_lm is None:
            setattr(self, _REWIND_STEP, None)
            return original_compute_loss(self, model, inputs, return_outputs=return_outputs,
                                         num_items_in_batch=num_items_in_batch)
        current_inputs = dict(inputs)
        current_inputs.pop("_prediction_loss_only", None)
        current_inputs["use_cache"] = False
        call = _HeadCall(causal_lm)
        token = _HEAD_CALL.set(call)
        try:
            # Keep HF's real model call, AMP entry, and denominator forwarding.
            loss, outputs = Trainer.compute_loss(
                self, model, current_inputs, return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        finally:
            _HEAD_CALL.reset(token)
        if call.result is None:
            # Never retry a forward after dropout has consumed RNG state.
            raise RuntimeError("The supported model did not execute the optimized head")
        _log_statistics(self, current_inputs, call.result)
        setattr(self, _REWIND_STEP, SimpleNamespace(token_loss=call.result.token_loss))
        return (loss, outputs) if return_outputs else loss

    compute_loss._soup_candidate_head = True
    forward._soup_candidate_head = True
    Qwen2ForCausalLM.forward = forward
    SFTTrainer.compute_loss = compute_loss
    return True


def patch_streamed_training():
    """Compatibility alias; the fixed HEAD_MODE setting selects the head."""
    return patch_training_head()
