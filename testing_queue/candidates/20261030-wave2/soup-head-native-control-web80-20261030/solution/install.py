"""Original constructor/property-owned installation. No Trainer/forward patch."""
from functools import wraps
import inspect
import weakref

from .native_head import NativeHeadLoss


class Installation:
    def __init__(self, cls, native_loss, property_descriptor, telemetry):
        self.cls, self.native_loss = cls, native_loss
        self.property_descriptor, self.telemetry = property_descriptor, telemetry
        self.previous_init = cls.__init__
        self.models = weakref.WeakKeyDictionary()
        self.active = True

        @wraps(self.previous_init)
        def construct(model, *args, **kwargs):
            self.previous_init(model, *args, **kwargs)
            self.attach(model)

        self.hook = construct
        cls.__init__ = construct

    def attach(self, model):
        if type(model) is not self.cls or getattr(model.config, "model_type", None) != "qwen2":
            self.telemetry.emit("model_skipped", reason="class")
            return
        if getattr(model.config, "vocab_size", None) != 151936:
            self.telemetry.emit("model_skipped", reason="vocabulary")
            return
        if inspect.getattr_static(type(model), "loss_function") is not self.property_descriptor:
            self.telemetry.emit("model_skipped", reason="foreign_property")
            return
        if model.loss_function is not self.native_loss:
            self.telemetry.emit("model_skipped", reason="foreign_loss")
            return
        had, prior = "_loss_function" in vars(model), vars(model).get("_loss_function")
        wrapper = NativeHeadLoss(self.native_loss, self.telemetry, owner=model)
        model.loss_function = wrapper
        self.models[model] = (wrapper, had, prior)
        self.telemetry.emit("model_loss_attached")

    def restore(self):
        if not self.active:
            return {"constructor": "already_restored", "restored_models": 0}
        constructor = "foreign_owner_preserved"
        if self.cls.__init__ is self.hook:
            self.cls.__init__ = self.previous_init
            constructor = "restored"
        count = 0
        for model, (wrapper, had, prior) in list(self.models.items()):
            if model.loss_function is wrapper:
                if had:
                    model.loss_function = prior
                else:
                    delattr(model, "_loss_function")
                count += 1
        self.active = False
        return {"constructor": constructor, "restored_models": count}


def install_pinned(telemetry):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from transformers.modeling_utils import PreTrainedModel
    from transformers.loss.loss_utils import ForCausalLMLoss, LOSS_MAPPING
    cls, loss = Qwen2ForCausalLM, ForCausalLMLoss
    descriptor = inspect.getattr_static(cls, "loss_function")
    if (cls.__module__ != "transformers.models.qwen2.modeling_qwen2"
            or cls.__init__.__module__ != cls.__module__
            or cls.__init__.__qualname__ != "Qwen2ForCausalLM.__init__"
            or hasattr(cls.__init__, "__wrapped__")
            or loss.__module__ != "transformers.loss.loss_utils"
            or loss.__qualname__ != "ForCausalLMLoss" or hasattr(loss, "__wrapped__")
            or LOSS_MAPPING.get("ForCausalLM") is not loss
            or descriptor is not inspect.getattr_static(PreTrainedModel, "loss_function")
            or not isinstance(descriptor, property) or descriptor.fset is None):
        raise RuntimeError("Unsupported/foreign Qwen loss or constructor owner; not installed")
    return Installation(cls, loss, descriptor, telemetry)
