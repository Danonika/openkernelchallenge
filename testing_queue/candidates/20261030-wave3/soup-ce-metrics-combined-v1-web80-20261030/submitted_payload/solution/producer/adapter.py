"""Original reversible Qwen loss-boundary adapter; no forward/trainer patch.

The five numerical kernel bodies and numerical gates retain the v2 source AST.
V3 consolidates denominator checking and flushes bounded route evidence.
Native constructor/head/outputs and supplied normalization count are unchanged.
"""
from collections import Counter
from dataclasses import dataclass
from functools import wraps
import inspect
import weakref

import torch

from . import materialized_ce as ce


from .telemetry import RouteStats


class MaterializedCELoss:
    def __init__(self, owner, native_loss, stats, summary_sink_provider=None):
        self.owner = weakref.ref(owner)
        self.native_loss = native_loss
        self.stats = stats
        # Captured consumer-owned callable only; no Trainer imports/context here.
        self.summary_sink_provider = summary_sink_provider

    def __call__(self, logits, labels, vocab_size, num_items_in_batch=None,
                 ignore_index=-100, shift_labels=None, **kwargs):
        self.stats.record("loss_boundary", kwargs={str(k)[:80]: (v if type(v) is bool or v is None else type(v).__name__) for k, v in sorted(kwargs.items())[:16]}, keyword_count=len(kwargs),
                          training=bool(getattr(self.owner(), "training", False)),
                          logits_dtype=str(getattr(logits, "dtype", None)),
                          logits_shape=list(logits.shape) if isinstance(logits, torch.Tensor) and logits.ndim <= 8 else None,
                          denominator_type=type(num_items_in_batch).__name__)

        def native(reason):
            self.stats.record("adapter_native_fallback", reason=reason)
            return self.native_loss(
                logits=logits, labels=labels, vocab_size=vocab_size,
                num_items_in_batch=num_items_in_batch, ignore_index=ignore_index,
                shift_labels=shift_labels, **kwargs)

        owner = self.owner()
        if owner is None or not owner.training:
            return native("owner_or_evaluation")
        # Pinned ForCausalLMLoss only forwards **kwargs to fixed_cross_entropy,
        # whose body never reads them. These TWO standard output controls have
        # already taken effect in the unmodified Qwen forward. Whitelist only
        # their bool/None forms; never mutate kwargs or any returned field.
        allowed_controls = {"output_attentions", "output_hidden_states"}
        unknown = set(kwargs) - allowed_controls
        unsupported_values = any(value is not None and type(value) is not bool
                                 for key, value in kwargs.items() if key in allowed_controls)
        if unknown or unsupported_values or shift_labels is not None or ignore_index != -100:
            return native("alternate_arguments:" + ",".join(sorted(kwargs)))
        if kwargs:
            key = ",".join(f"{name}={kwargs[name]!r}" for name in sorted(kwargs))
            self.stats.observe("output_controls", key)
        if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
            return native("not_tensor")
        if not torch.is_grad_enabled() or not logits.requires_grad:
            return native("no_gradient_path")
        if (type(vocab_size) is not int or vocab_size != ce.VOCABULARY
                or logits.ndim != 3 or logits.shape[0] != 4
                or not 1 <= logits.shape[1] <= 512
                or logits.shape[2] != vocab_size or labels.shape != logits.shape[:-1]):
            return native("shape_or_vocabulary")
        if (logits.dtype != torch.float16 or logits.is_neg() or logits.is_conj()
                or not logits.is_contiguous()
                or labels.dtype != torch.int64 or not labels.is_contiguous()
                or not logits.is_cuda or labels.device != logits.device):
            return native("dtype_layout_or_device")
        if torch.cuda.get_device_capability(logits.device) != (7, 5):
            return native("not_sm75")

        # V3 validates the denominator once in the core. In v2 both adapter
        # and core read a CUDA scalar with .item(), adding duplicate host sync.
        # The original supplied object is still used for native SUM / D.
        self.stats.observe("boundaries", f"fp16:{tuple(logits.shape)}")
        self.stats.record("candidate_loss_call")
        # Loss only. Do not replace logits, return outputs, re-run the head,
        # recount supervision, divide again, unscale or touch trainer metrics.
        summary_sink = None
        if self.summary_sink_provider is not None:
            summary_sink = self.summary_sink_provider(owner, logits, labels)
        return ce.materialized_causal_loss(
            logits, labels, num_items_in_batch=num_items_in_batch,
            diagnostics=self.stats, native_fallback=native, summary_sink=summary_sink)


@dataclass
class ModelInstallation:
    model_ref: object
    wrapper: MaterializedCELoss
    had_override: bool
    previous_override: object

    def restore(self):
        model = self.model_ref()
        if model is None:
            return "model_gone"
        if model.loss_function is not self.wrapper:
            return "foreign_loss_preserved"
        if self.had_override:
            model.loss_function = self.previous_override
        else:
            delattr(model, "_loss_function")
        return "restored"


class ConstructorInstallation:
    def __init__(self, model_class, native_loss, native_property, stats, summary_sink_provider=None):
        self.model_class = model_class
        self.native_loss = native_loss
        self.native_property = native_property
        self.summary_sink_provider = summary_sink_provider
        self.stats = stats
        self.native_init = model_class.__init__
        # Values contain only weak model references and metadata/callables.
        self.models = weakref.WeakKeyDictionary()
        self.active = True

        @wraps(self.native_init)
        def initialize(model, *args, **kwargs):
            self.native_init(model, *args, **kwargs)
            self.install_model(model)

        self.hook = initialize
        model_class.__init__ = initialize

    def install_model(self, model):
        if type(model) is not self.model_class:
            self.stats.record("model_not_installed", reason="subclass")
            return None
        config = model.config
        if (getattr(config, "model_type", None) != "qwen2"
                or getattr(config, "vocab_size", None) != ce.VOCABULARY):
            self.stats.record("model_not_installed", reason="configuration")
            return None
        if inspect.getattr_static(type(model), "loss_function") is not self.native_property:
            self.stats.record("model_not_installed", reason="foreign_property")
            return None
        previous = self.models.get(model)
        if previous is not None and model.loss_function is previous.wrapper:
            return previous
        if model.loss_function is not self.native_loss:
            self.stats.record("model_not_installed", reason="foreign_loss")
            return None
        had = "_loss_function" in vars(model)
        prior = vars(model).get("_loss_function")
        wrapper = MaterializedCELoss(model, self.native_loss, self.stats, self.summary_sink_provider)
        handle = ModelInstallation(weakref.ref(model), wrapper, had, prior)
        model.loss_function = wrapper
        self.models[model] = handle
        self.stats.record("model_loss_installed")
        return handle

    def restore(self):
        if not self.active:
            return {"constructor": "already_restored", "model_losses": {}}
        if self.model_class.__init__ is self.hook:
            self.model_class.__init__ = self.native_init
            constructor = "restored"
        else:
            constructor = "foreign_constructor_preserved"
        outcomes = Counter(handle.restore() for handle in list(self.models.values()))
        self.active = False
        return {"constructor": constructor, "model_losses": dict(outcomes)}


def install_pinned_qwen2(stats, summary_sink_provider=None):
    """Call only after entrypoint version guards; never traverses loaded models."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from transformers.modeling_utils import PreTrainedModel
    from transformers.loss.loss_utils import ForCausalLMLoss, LOSS_MAPPING

    cls, loss = Qwen2ForCausalLM, ForCausalLMLoss
    init = cls.__init__
    descriptor = inspect.getattr_static(cls, "loss_function")
    if (cls.__module__ != "transformers.models.qwen2.modeling_qwen2"
            or init.__module__ != cls.__module__
            or init.__qualname__ != "Qwen2ForCausalLM.__init__"
            or hasattr(init, "__wrapped__")
            or loss.__module__ != "transformers.loss.loss_utils"
            or loss.__qualname__ != "ForCausalLMLoss"
            or hasattr(loss, "__wrapped__")
            or LOSS_MAPPING.get("ForCausalLM") is not loss
            or descriptor is not inspect.getattr_static(PreTrainedModel, "loss_function")
            or not isinstance(descriptor, property) or descriptor.fset is None):
        raise RuntimeError("Foreign or unsupported Qwen/native-loss owner; no patch installed")
    if summary_sink_provider is not None and not callable(summary_sink_provider):
        raise TypeError("summary_sink_provider must be an owned callable or None")
    return ConstructorInstallation(cls, loss, descriptor, stats, summary_sink_provider)
