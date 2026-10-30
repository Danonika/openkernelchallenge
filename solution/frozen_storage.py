"""Optional lossless storage for a previously FP16 tied embedding/output head.

Soup loads the frozen base in FP16 on T4; PEFT k-bit preparation then converts
non-quantized parameters to FP32. This experiment records the actual FP16
storage *before* that conversion, runs unmodified preparation, and checks that
the resulting frozen tied parameter contains exactly the same values before
restoring the smaller storage. It never compresses arbitrary FP32 values.

Embedding lookup still returns FP32, preserving the decoder's input dtype.
Ordinary output-head calls use FP32 operands outside autocast; FP16 autocast
and the streamed training head can consume the retained half storage directly.
Full base state_dict entries intentionally use the compact storage dtype;
their names, shapes, values and parameter tying are preserved. Adapter tensors
are not touched. This option is disabled by default pending physical T4 tests.
"""

from dataclasses import dataclass
from functools import wraps
from importlib import metadata
import inspect
from types import MethodType

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n_elements"])
def soup_frozen_embedding_forward(
    TOKEN_IDS, WEIGHT, OUTPUT, n_elements,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):
    """Gather exact FP16-representable frozen weights into FP32 activations."""
    element = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = element < n_elements
    token = tl.load(TOKEN_IDS + element // WIDTH, active, other=0)
    value = tl.load(WEIGHT + token * WIDTH + element % WIDTH, active, other=0)
    tl.store(OUTPUT + element, value.to(tl.float32), active)


def frozen_embedding(weight, token_ids):
    """Direct kernel entry point; CPU is supported by Triton's interpreter."""
    if (weight.ndim != 2 or weight.dtype != torch.float16
            or not weight.is_contiguous() or weight.requires_grad):
        raise ValueError("expected a contiguous frozen FP16 embedding matrix")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("embedding indices must be int32 or int64")
    if token_ids.device != weight.device:
        raise ValueError("embedding matrix and indices must share a device")
    token_ids = token_ids.contiguous()
    output = torch.empty((*token_ids.shape, weight.shape[1]), dtype=torch.float32,
                         device=weight.device)
    if output.numel():
        soup_frozen_embedding_forward[(triton.cdiv(output.numel(), 512),)](
            token_ids, weight, output, output.numel(), WIDTH=weight.shape[1],
            BLOCK=512, num_warps=4,
        )
    return output


@dataclass(frozen=True)
class _Candidate:
    model: object
    embedding: object
    head: object
    parameter: object
    half_storage: object


def _supported_device(weight):
    return weight.is_cuda


def _capture_candidate(model, use_gradient_checkpointing):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    if (type(model) is not Qwen2ForCausalLM
            or not getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_loaded_in_8bit", False)
            or use_gradient_checkpointing
            or getattr(model, "_candidate_frozen_storage", False)):
        return None
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    if (type(embedding) is not torch.nn.Embedding or type(head) is not torch.nn.Linear
            or head.bias is not None or embedding.max_norm is not None
            or embedding.sparse or embedding.scale_grad_by_freq
            or "forward" in embedding.__dict__ or "forward" in head.__dict__
            or hasattr(embedding, "_hf_hook") or hasattr(head, "_hf_hook")):
        return None
    weight = embedding.weight
    if (weight is not head.weight or weight.dtype != torch.float16
            or not _supported_device(weight) or not weight.is_contiguous()
            or weight.ndim != 2 or not weight.numel()
            or tuple(weight.shape) != (embedding.num_embeddings, embedding.embedding_dim)
            or tuple(weight.shape) != (head.out_features, head.in_features)
            or not getattr(model.config, "tie_word_embeddings", False)):
        return None
    # A detached view keeps the original half allocation alive while PEFT
    # replaces parameter.data with an upcast allocation. It consumes no RNG.
    return _Candidate(model, embedding, head, weight, weight.detach())


def _unchanged_after_preparation(candidate, model):
    c = candidate
    weight = c.parameter
    if (model is not c.model or model.get_input_embeddings() is not c.embedding
            or model.get_output_embeddings() is not c.head
            or c.embedding.weight is not weight or c.head.weight is not weight
            or weight.requires_grad or weight.dtype != torch.float32
            or weight.device != c.half_storage.device
            or weight.shape != c.half_storage.shape or not weight.is_contiguous()):
        return False
    # Bounded verification, once at setup: do not allocate another full FP32
    # vocabulary matrix. Equal accepts exact FP16 round trips only; NaNs fail.
    current, original = weight.detach().view(-1), c.half_storage.view(-1)
    for start in range(0, current.numel(), 4 * 1024 * 1024):
        end = min(start + 4 * 1024 * 1024, current.numel())
        if not torch.equal(current[start:end], original[start:end].float()):
            return False
    return True


@wraps(torch.nn.Embedding.forward)
def _embedding_forward(module, token_ids):
    parameter = module.weight
    if parameter.dtype != torch.float16:
        return torch.nn.Embedding.forward(module, token_ids)
    if parameter.requires_grad:
        raise RuntimeError("losslessly packed embedding must remain frozen")
    if parameter.is_cuda:
        return frozen_embedding(parameter, token_ids)
    # Device moves after setup preserve ordinary lookup semantics. The
    # production installer itself only captures CUDA models.
    return torch.nn.Embedding.forward(module, token_ids).float()


@wraps(torch.nn.Linear.forward)
def _head_forward(module, hidden_states):
    parameter = module.weight
    if parameter.dtype != torch.float16:
        return torch.nn.Linear.forward(module, hidden_states)
    if parameter.requires_grad:
        raise RuntimeError("losslessly packed output head must remain frozen")
    device_type = hidden_states.device.type
    if (torch.is_autocast_enabled(device_type)
            and torch.get_autocast_dtype(device_type) == torch.float16):
        return torch.nn.Linear.forward(module, hidden_states)
    # Outside the target AMP path, reproduce the original FP32 weight
    # operand, including the usual mixed-dtype error for FP16 inputs.
    return F.linear(hidden_states, parameter.float(), module.bias)


def _install_storage(candidate):
    c = candidate
    # Keep the same Parameter object and its shared ownership. No parameter or
    # buffer is added, no task/model config is changed, and normal Module hooks
    # (including enable_input_require_grads) still surround each forward.
    c.parameter.data = c.half_storage
    # Global methods avoid retaining the full candidate or original modules
    # in closures, and allow deepcopy to bind each method to the copied module.
    c.embedding.forward = MethodType(_embedding_forward, c.embedding)
    c.head.forward = MethodType(_head_forward, c.head)
    c.model.__dict__["_candidate_frozen_storage"] = True
    c.model.__dict__.pop("_candidate_head_cast", None)


def patch_frozen_storage():
    """Install the optional preparation hook for the audited pinned stack."""
    import peft

    if metadata.version("peft") != "0.20.0" or metadata.version("transformers") != "5.16.1":
        return False
    original = peft.prepare_model_for_kbit_training
    if getattr(original, "_soup_candidate_frozen_storage", False):
        return True
    signature = inspect.signature(original)

    @wraps(original)
    def prepare(model, *args, **kwargs):
        bound = signature.bind(model, *args, **kwargs)
        bound.apply_defaults()
        candidate = _capture_candidate(model, bound.arguments["use_gradient_checkpointing"])
        prepared = original(model, *args, **kwargs)
        if candidate is not None and _unchanged_after_preparation(candidate, prepared):
            _install_storage(candidate)
        return prepared

    prepare._soup_candidate_frozen_storage = True
    peft.prepare_model_for_kbit_training = prepare
    return True
