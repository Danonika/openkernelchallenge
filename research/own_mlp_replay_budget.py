"""Independent, source-guarded replay budget for a 28-layer frozen Qwen2 MLP.

This is a deterministic per-layer lifetime optimization, not Soup
gradient_checkpointing and not a benchmark/data/seed switch. Every layer
remains trained, every LoRA parameter and dropout RNG unchanged. Frozen MLP
layers not selected for input-only rematerialization run their original
forward + backward. Selected layers use our independent grouped backward
optimization when the qgroup span is available.
"""
from functools import wraps


def install(replay_layers=24, split_forward=None, split_backward=None):
    from transformers.models.qwen2 import modeling_qwen2

    if type(replay_layers) is not int or not 1 <= replay_layers <= 28:
        raise ValueError("Expected 1..28 MLP replay layers")
    if (split_forward is None) != (split_backward is None):
        raise ValueError("Forward/backward split changes must be paired")
    if split_forward is not None and not (
        type(split_forward) is int and 1 <= split_forward <= 4 and
        type(split_backward) is int and 1 <= split_backward <= 5
    ):
        raise ValueError("Unsupported grouped NF4 boundary lifetime")

    layer_cls = modeling_qwen2.Qwen2DecoderLayer
    mlp_cls = modeling_qwen2.Qwen2MLP
    previous_init = layer_cls.__init__
    previous_forward = mlp_cls.forward
    if getattr(previous_forward, "_own_replay_budget_v1", False):
        return True

    # Must sit directly on top of the measured input-only replay path,
    # optionally with our original grouped backward-only GEMM wrapper.
    if getattr(previous_forward, "_soup_packed_mlp_replay", False):
        input_only = getattr(previous_forward, "_previous", None)
        if not getattr(input_only, "_soup_input_only_replay", False):
            raise RuntimeError("Unexpected packed-replay parent")
        native = getattr(input_only, "_previous", None)
    elif getattr(previous_forward, "_soup_input_only_replay", False):
        native = getattr(previous_forward, "_previous", None)
    else:
        raise RuntimeError("Expected verified input-only MLP parent")
    if not callable(native):
        raise RuntimeError("Missing original MLP fallback")

    @wraps(previous_init)
    def tagged_init(self, *args, **kwargs):
        result = previous_init(self, *args, **kwargs)
        index = getattr(getattr(self, "self_attn", None), "layer_idx", None)
        if type(index) is int and 0 <= index < 28:
            self.mlp._own_replay_layer_idx = index
        return result

    @wraps(previous_forward)
    def budgeted_forward(self, x):
        index = getattr(self, "_own_replay_layer_idx", None)
        if type(index) is int and 0 <= index < 28 and index >= replay_layers:
            return native(self, x)
        return previous_forward(self, x)

    tagged_init._own_replay_budget_init_v1 = True
    budgeted_forward._own_replay_budget_v1 = True
    budgeted_forward._original = previous_forward

    # All validation and references happen before altering module classes.
    qgroup = None
    if split_forward is not None:
        from . import qgroup as qgroup
        current = getattr(qgroup, "SPLIT_LAYERS", None)
        if not isinstance(current, dict) or not all(
            k in current and type(current[k]) is int for k in ("fwd", "bwd")
        ):
            raise RuntimeError("Unexpected grouped NF4 lifetime configuration")

    layer_cls.__init__ = tagged_init
    mlp_cls.forward = budgeted_forward
    if qgroup is not None:
        qgroup.SPLIT_LAYERS["fwd"] = split_forward
        qgroup.SPLIT_LAYERS["bwd"] = split_backward
    return True
