"""Hero1 original experimental FP16 residual bridge for the fixed Soup T4 workload.

Based on the numerical/architectural idea of eliminating unnecessary FP32
decoder activations. The code is independently authored: it contains no copied
third-party kernels. Tested ONLY after full T4 parity/quality gates.

The frozen tied embedding is already stored in exact FP16 by our audited
frozen_storage.patch_frozen_storage; this module changes just its output
activation dtype during training. Qwen2 RMSNorm keeps its frozen FP32 scale
parameter and computes the full original FP32 result with our own existing
Triton kernel, then rounds the *output* to the original input dtype. No
trainable parameter/optimizer/LoRA/dropout/task switches are touched.

The half residual stream is a **deliberate numerical approximation**, not an
exact replacement for baseline FP32 residuals. Hidden-data and metric checks
must decide whether it is acceptable; this module never claims lossless model
behavior.
"""
from __future__ import annotations

from functools import wraps
from types import MethodType

import torch
import torch.nn.functional as F


def _half_embedding_forward(module, input_ids):
    # Same frozen embedding table and torch autograd hooks; only output dtype
    # differs from the prior FP32 materialization.
    if module.weight.requires_grad or module.weight.dtype != torch.float16:
        raise RuntimeError("Hero1 half residual requires verified frozen FP16 tied embedding")
    return F.embedding(
        input_ids, module.weight, module.padding_idx, module.max_norm,
        module.norm_type, module.scale_grad_by_freq, module.sparse,
    )


def patch_half_residual_bridge():
    import peft
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2RMSNorm

    previous_prepare = peft.prepare_model_for_kbit_training
    if getattr(previous_prepare, "_hero1_half_residual", False):
        return True

    # Our own existing fused RMSNorm remains the numerical producer; its
    # native FP32 output is rounded once at a separately tagged training node.
    previous_norm_forward = Qwen2RMSNorm.forward

    @wraps(previous_norm_forward)
    def bridge_norm(self, hidden_states):
        output = previous_norm_forward(self, hidden_states)
        if (getattr(self, "_hero1_half_norm", False)
                and self.training and torch.is_grad_enabled()
                and isinstance(hidden_states, torch.Tensor)
                and hidden_states.is_cuda
                and hidden_states.dtype == torch.float16
                and isinstance(output, torch.Tensor)
                and output.dtype == torch.float32):
            return output.to(torch.float16)
        return output

    if not getattr(previous_norm_forward, "_hero1_half_norm", False):
        bridge_norm._hero1_half_norm = True
        Qwen2RMSNorm.forward = bridge_norm

    @wraps(previous_prepare)
    def prepared_with_half_residual(model, *args, **kwargs):
        prepared = previous_prepare(model, *args, **kwargs)
        # Prevent changes on generic PEFT regressions, AMP/bf16 variants,
        # alternate model classes or unsupported training configs.
        if (type(prepared) is not Qwen2ForCausalLM
                or not getattr(prepared, "is_loaded_in_4bit", False)
                or not getattr(prepared, "_candidate_frozen_storage", False)
                or getattr(prepared, "_hero1_half_residual_ready", False)):
            return prepared
        embedding = prepared.get_input_embeddings()
        head = prepared.get_output_embeddings()
        if (type(embedding) is not torch.nn.Embedding
                or type(head) is not torch.nn.Linear
                or head.weight is not embedding.weight
                or not embedding.weight.is_cuda
                or embedding.weight.requires_grad
                or embedding.weight.dtype != torch.float16
                or embedding.max_norm is not None
                or embedding.scale_grad_by_freq
                or embedding.sparse
                or getattr(prepared.config, "tie_word_embeddings", False) is not True):
            return prepared

        norms = [m for m in prepared.modules() if type(m) is Qwen2RMSNorm]
        if len(norms) != 57:
            return prepared
        if any((m.weight.dtype != torch.float32 or m.weight.requires_grad
                or m.weight.device != embedding.weight.device)
               for m in norms):
            return prepared
        if "forward" not in embedding.__dict__:
            # The earlier verified frozen-storage patch must own this method.
            return prepared

        # Commit only after checking the entire model/parameter contract.
        embedding.forward = MethodType(_half_embedding_forward, embedding)
        for m in norms:
            m._hero1_half_norm = True
        prepared._hero1_half_residual_ready = True
        return prepared

    prepared_with_half_residual._hero1_half_residual = True
    peft.prepare_model_for_kbit_training = prepared_with_half_residual
    return True
