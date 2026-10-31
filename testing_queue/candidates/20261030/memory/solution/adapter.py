"""Memory VS1: original revision of pinned native producer codec.

New: zero-storage shared-version witnesses replace weak-only wide checks;
flushed cumulative activation checkpoints survive process termination.

Only two named native SavedTensor objects may receive hooks. There is no broad
saved_tensors_hooks context, value scan, CPU synchronization, custom derivative,
shared input cast, alternate GEMM, or model/task-dependent measurement routing.
Production and explicit CPU-equivalent fixture modes are separate contracts.
"""
import contextvars
import atexit
import functools
import hashlib
import inspect
import json
import weakref
from dataclasses import dataclass
from importlib.metadata import version

import torch

from .restore import restore_half
from .version_sentinel import make_witness, runtime_contract, UnsupportedVersionWitness


EXPECTED = {
    "bnb_matmul": "a219d40d62264de60a424e8f9f1bab7656136eb7264a575d39d57a7e2f25ed4a",
    "linear_forward": "d52cf63457717cc718ca0a449dad0967171e679be90d5d2adcafe2e8a141093d",
    "mlp_forward": "03df2d7685cf8446594448cbd33ef676a2b088779d99d9803abe3bed2ec7ad9b",
    "silu_forward": "f9eaaa437c9f969e5f5c6f347b85a09fefcbae8824433dff69e0e83faa098863",
}
CONFLICT = "Calling register_hooks on a saved tensor whose hooks have already been set. Hint: only one pair of hooks is allowed at a time."
_ACTIVE = contextvars.ContextVar("soup_mem001_invocation", default=None)
_STATE = None
_STATS = {}
_SUMMARY_REGISTERED = False


def _count(name, value=1):
    _STATS[name] = _STATS.get(name, 0) + value
    count = _STATS[name]
    # Geometric checkpoints keep stdout small and survive SIGKILL. Counts are
    # lower bounds until the final record; they are NOT CUDA timing or steps.
    if (_STATE is not None and _STATE["mode"] == "production"
            and name in ("encoded_operands", "restores", "observed_mlp_calls")
            and count > 0 and count & (count - 1) == 0):
        print("SOUP_MEMORY_VS1_PROGRESS " + json.dumps(
            {"event": name, "cumulative_lower_bound": snapshot(),
             "completed_training": False}, sort_keys=True), flush=True)


def snapshot():
    return dict(_STATS)


def reset_stats():
    _STATS.clear()


def _hash(function):
    # inspect.getsource unwraps decorated callables; a foreign/hub wrapper
    # must not impersonate the pinned native implementation through metadata.
    if hasattr(function, "__wrapped__"):
        raise TypeError("Preexisting wrapped callable is outside the native contract")
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _exit_summary():
    print("SOUP_MEMORY_VS1_RUNTIME " + json.dumps(snapshot(), sort_keys=True), flush=True)


def _foreign_hooks(module, binding=None):
    own = set() if binding is None else binding.handle_ids
    for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"):
        if any(key not in own for key in getattr(module, name, {})):
            return True
    return False


def _global_hooks():
    module = torch.nn.modules.module
    return any(bool(getattr(module, name, {})) for name in (
        "_global_forward_hooks", "_global_forward_pre_hooks",
        "_global_backward_hooks", "_global_backward_pre_hooks"))


def _module_reason(mlp, x, state, binding=None):
    if type(mlp) is not state["Qwen2MLP"] or type(x) is not torch.Tensor:
        return "class"
    if not state["first_order_only"]:
        return "first_order_contract_absent"
    if not mlp.training or not torch.is_grad_enabled() or torch.is_inference_mode_enabled() or not x.requires_grad:
        return "not_first_order_training"
    if x.dtype != torch.float32 or x.ndim not in (2, 3) or not x.is_contiguous() or x.numel() == 0:
        return "input_metadata"
    if torch.autograd.forward_ad.unpack_dual(x).tangent is not None:
        return "forward_ad"
    if type(mlp.act_fn) is not state["SiLUActivation"]:
        return "activation"
    if ("forward" in mlp.__dict__ or "forward" in mlp.act_fn.__dict__
            or type(mlp.act_fn).forward is not state["silu_forward"]
            or state["Linear4bit"].forward is not state["originals"]["linear_forward"]
            or state["bnb"].matmul_4bit is not state["matmul_observer"]):
        return "runtime_forward_identity"
    if _global_hooks():
        return "global_module_hooks"
    modules = [mlp, mlp.act_fn, mlp.gate_proj, mlp.up_proj, mlp.down_proj]
    if any(_foreign_hooks(module, binding) for module in modules):
        return "foreign_module_hooks"
    if state["mode"] == "production":
        if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) != (7, 5):
            return "not_t4"
        if (mlp.hidden_size, mlp.intermediate_size) != (1536, 8960):
            return "model_shape"
        if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != torch.float16:
            return "autocast_dtype"
    elif x.device.type != "cpu" or torch.is_autocast_enabled("cpu"):
        return "cpu_equivalent_device_or_autocast"
    if x.shape[-1] != mlp.hidden_size:
        return "hidden_shape"
    for role, expected_shape in (("gate_proj", (mlp.intermediate_size, mlp.hidden_size)),
                                 ("up_proj", (mlp.intermediate_size, mlp.hidden_size)),
                                 ("down_proj", (mlp.hidden_size, mlp.intermediate_size))):
        module = getattr(mlp, role)
        if "forward" in module.__dict__:
            return "instance_projection_forward"
        if type(module) is not state["Linear4bit"] or module.bias is not None or not module.training:
            return "projection_class_bias_or_mode"
        if module.compute_dtype != torch.float16 or not module.compute_type_is_set:
            return "projection_compute_contract"
        if module.weight.requires_grad or module.weight.dtype != torch.uint8:
            return "projection_weight"
        qs = getattr(module.weight, "quant_state", None)
        if (qs is None or tuple(qs.shape) != expected_shape or qs.quant_type != "nf4"
                or qs.dtype != torch.float16 or not qs.nested or qs.blocksize != 64
                or qs.state2.blocksize != 256 or getattr(qs, "packing_format_for_cpu", False)):
            return "quantization_contract"
    return None


class _Binding:
    def __init__(self, mlp):
        self.mlp = weakref.ref(mlp)
        self.modules = {name: weakref.ref(getattr(mlp, name)) for name in ("gate_proj", "up_proj", "down_proj", "act_fn")}
        self.handles = []
        self.handle_ids = set()
        self._add(mlp.gate_proj.register_forward_hook(self._wide))
        self._add(mlp.up_proj.register_forward_hook(self._wide))
        self._add(mlp.act_fn.register_forward_hook(self._activation))
        self._add(mlp.down_proj.register_forward_pre_hook(self._product))

    def _add(self, handle):
        self.handles.append(handle)
        self.handle_ids.add(handle.id)

    def remove(self):
        for handle in self.handles:
            handle.remove()

    def _invocation(self):
        current = _ACTIVE.get()
        return current if current is not None and current["binding"] is self else None

    def _wide(self, module, inputs, output):
        current = self._invocation()
        if current is None:
            return
        role = "gate" if module is self.modules["gate_proj"]() else "up"
        current[role + "_wide"] = output

    def _activation(self, module, inputs, output):
        current = self._invocation()
        if current is not None and type(output) is torch.Tensor:
            current["silu_node"] = output.grad_fn

    def _product(self, module, inputs):
        current = self._invocation()
        if current is not None and len(inputs) == 1 and type(inputs[0]) is torch.Tensor:
            current["mul_node"] = inputs[0].grad_fn


@dataclass
class _Packet:
    payload: torch.Tensor
    version: int
    wide_witness: torch.Tensor
    wide_version: int
    encoded: bool


def _unpack(packet):
    if torch.is_grad_enabled():
        raise RuntimeError("MEM001 first-order contract: opt out before forward for higher-order AD or saved-state inspection with grad enabled")
    if packet.payload._version != packet.version:
        raise RuntimeError("MEM001 saved producer payload was modified in-place")
    if (packet.wide_witness._version != packet.wide_version
            or packet.wide_witness.untyped_storage().nbytes() != 0):
        raise RuntimeError("MEM001 widened operand version witness changed")
    if packet.encoded:
        restored = restore_half(packet.payload)
        # Count only after the native CPU operation / CUDA launch returns.
        # A CUDA return is dispatch evidence, not a synchronization proof.
        _count("restores")
        return restored
    return packet.payload


def _register_one(raw_saved, wide, half):
    # Witness holds a version counter but ZERO bytes of the FP32 storage.
    # Never capture the invocation, modules, output or graph nodes.
    try:
        witness = make_witness(wide)
    except UnsupportedVersionWitness:
        _count("version_witness_declined")
        return False
    payload = half.detach()
    packet = _Packet(payload, payload._version, witness, wide._version, True)
    meta = (wide.data_ptr(), tuple(wide.shape), tuple(wide.stride()), wide.storage_offset(), wide.dtype, wide.device)
    entered = False

    def pack(tensor):
        nonlocal entered
        entered = True
        actual = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(), tensor.dtype, tensor.device)
        if actual != meta or tensor._version != packet.wide_version:
            _count("saved_operand_identity_fallback")
            original = tensor.detach()
            return _Packet(original, original._version, make_witness(tensor), tensor._version, False)
        _count("encoded_operands")
        _count("logical_half_payload_bytes", payload.numel() * payload.element_size())
        return packet

    try:
        raw_saved.register_hooks(pack, _unpack)
    except RuntimeError as exc:
        if not entered and str(exc) == CONFLICT:
            _count("existing_saved_hooks_declined")
            return False
        raise
    return True


def _finish(current):
    pairs = []
    for role in ("gate", "up"):
        half, wide = current.get(role + "_half"), current.get(role + "_wide")
        if (current.get(role + "_calls") != 1 or type(half) is not torch.Tensor or type(wide) is not torch.Tensor
                or half.dtype != torch.float16 or wide.dtype != torch.float32 or half.shape != wide.shape
                or not half.is_contiguous() or not wide.is_contiguous() or half.device != wide.device
                or half.grad_fn is None or wide.grad_fn is None or wide.grad_fn.name() != "ToCopyBackward0"
                or wide.grad_fn.next_functions[0][0] is not half.grad_fn):
            _count("producer_provenance_fallback")
            return
        pairs.append((half, wide))
    silu, mul = current.get("silu_node"), current.get("mul_node")
    if (silu is None or mul is None or silu.name() != "SiluBackward0" or mul.name() != "MulBackward0"
            or not hasattr(silu, "_raw_saved_self") or not hasattr(mul, "_raw_saved_other")
            or silu.next_functions[0][0] is not pairs[0][1].grad_fn
            or mul.next_functions[0][0] is not silu or mul.next_functions[1][0] is not pairs[1][1].grad_fn):
        _count("native_graph_fallback")
        return
    # Existing hooks are respected independently. No pair-atomicity claim.
    _register_one(silu._raw_saved_self, pairs[0][1], pairs[0][0])
    _register_one(mul._raw_saved_other, pairs[1][1], pairs[1][0])


def install(*, mode="production", first_order_only=True):
    global _STATE, _SUMMARY_REGISTERED
    if mode not in ("production", "cpu_equivalent"):
        raise ValueError("Explicit production or cpu_equivalent mode required")
    if _STATE is not None:
        if (_STATE["bnb"].matmul_4bit is not _STATE["matmul_observer"]
                or _STATE["Qwen2MLP"].forward is not _STATE["mlp_observer"]):
            return {"installed": False, "reason": "foreign_postpatch"}
        if _STATE["mode"] == mode and _STATE["first_order_only"] == first_order_only:
            return {"installed": True, "reason": "already_installed", "mode": mode}
        return {"installed": False, "reason": "different_installation_contract"}
    import bitsandbytes as bnb
    from transformers.activations import SiLUActivation
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    if version("bitsandbytes") != "0.50.2" or version("transformers") != "5.16.1":
        return {"installed": False, "reason": "package_version"}
    if mode == "production" and (torch.__version__.split("+")[0] != "2.11.0" or version("triton") != "3.6.0"):
        return {"installed": False, "reason": "target_torch_triton_version"}
    originals = {"bnb_matmul": bnb.matmul_4bit, "linear_forward": bnb.nn.Linear4bit.forward,
                 "mlp_forward": Qwen2MLP.forward, "silu_forward": SiLUActivation.forward}
    try:
        if any(_hash(function) != EXPECTED[name] for name, function in originals.items()):
            return {"installed": False, "reason": "source_identity"}
    except (OSError, TypeError):
        return {"installed": False, "reason": "source_unavailable"}
    try:
        version_probe = runtime_contract("cpu")
    except UnsupportedVersionWitness as exc:
        return {"installed": False, "reason": "version_witness_contract", "detail": str(exc)}
    state = {"mode": mode, "first_order_only": first_order_only, "bnb": bnb,
             "Qwen2MLP": Qwen2MLP, "Linear4bit": bnb.nn.Linear4bit,
             "originals": originals, "bindings": weakref.WeakKeyDictionary(),
             "silu_forward": SiLUActivation.forward, "SiLUActivation": SiLUActivation}

    @functools.wraps(originals["bnb_matmul"])
    def matmul_observer(*args, **kwargs):
        output = originals["bnb_matmul"](*args, **kwargs)
        current = _ACTIVE.get()
        if current is not None and len(args) >= 2:
            a, weight = args[:2]
            for role in ("gate", "up"):
                module = current["binding"].modules[role + "_proj"]()
                if module is not None and weight is module.weight:
                    current[role + "_calls"] = current.get(role + "_calls", 0) + 1
                    qs = kwargs.get("quant_state", args[2] if len(args) > 2 else None)
                    bias = kwargs.get("bias", args[4] if len(args) > 4 else None)
                    out = kwargs.get("out", args[3] if len(args) > 3 else None)
                    if a.dtype == torch.float16 and qs is module.weight.quant_state and bias is None and out is None:
                        current[role + "_half"] = output
        return output

    @functools.wraps(originals["mlp_forward"])
    def mlp_observer(mlp, x):
        if _ACTIVE.get() is not None:
            return originals["mlp_forward"](mlp, x)
        binding = state["bindings"].get(mlp)
        reason = _module_reason(mlp, x, state, binding)
        if reason is not None:
            _count("fallback_" + reason)
            return originals["mlp_forward"](mlp, x)
        if binding is None:
            binding = _Binding(mlp)
            state["bindings"][mlp] = binding
        current = {"binding": binding}
        token = _ACTIVE.set(current)
        try:
            output = originals["mlp_forward"](mlp, x)
            _finish(current)
            _count("observed_mlp_calls")
            return output
        finally:
            _ACTIVE.reset(token)
            # All wide/capture/node references expire at this boundary.
            current.clear()

    state["matmul_observer"] = matmul_observer
    state["mlp_observer"] = mlp_observer
    bnb.matmul_4bit = matmul_observer
    Qwen2MLP.forward = mlp_observer
    _STATE = state
    if mode == "production" and not _SUMMARY_REGISTERED:
        atexit.register(_exit_summary)
        _SUMMARY_REGISTERED = True
    return {"installed": True, "mode": mode, "source_hashes": dict(EXPECTED), "first_order_only": first_order_only, "version_witness_probe": version_probe}


def uninstall():
    global _STATE
    if _STATE is None:
        return
    if _ACTIVE.get() is not None:
        raise RuntimeError("Cannot uninstall an active MEM001 invocation")
    state = _STATE
    if state["bnb"].matmul_4bit is not state["matmul_observer"] or state["Qwen2MLP"].forward is not state["mlp_observer"]:
        raise RuntimeError("MEM001 cannot overwrite a later foreign patch")
    for binding in list(state["bindings"].values()):
        binding.remove()
    state["bnb"].matmul_4bit = state["originals"]["bnb_matmul"]
    state["Qwen2MLP"].forward = state["originals"]["mlp_forward"]
    _STATE = None
