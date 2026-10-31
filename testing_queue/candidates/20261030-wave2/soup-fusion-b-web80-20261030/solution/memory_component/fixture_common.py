"""Original tiny-module/future-T4 fixture helpers; never used by apply()."""
import torch
import bitsandbytes as bnb
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP


def make_mlp(hidden=64, intermediate=128, device="cpu"):
    config = Qwen2Config(hidden_size=hidden, intermediate_size=intermediate,
                        num_attention_heads=4, num_key_value_heads=2,
                        num_hidden_layers=1, hidden_act="silu")
    mlp = Qwen2MLP(config)
    generator = torch.Generator(device="cpu").manual_seed(305001)
    for name, n, k in (("gate_proj", intermediate, hidden),
                       ("up_proj", intermediate, hidden),
                       ("down_proj", hidden, intermediate)):
        layer = bnb.nn.Linear4bit(k, n, bias=False, compute_dtype=torch.float16,
                                 compress_statistics=True, quant_type="nf4")
        dense = (torch.randn((n, k), generator=generator) * 0.025).half()
        layer.weight = bnb.nn.Params4bit(dense, requires_grad=False,
                                         compress_statistics=True, quant_type="nf4")
        layer = layer.to(device)
        setattr(mlp, name, layer)
    return mlp.train()


def compare(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    masks = {"nan": torch.isnan, "posinf": torch.isposinf, "neginf": torch.isneginf}
    for name, function in masks.items():
        assert torch.equal(function(actual), function(expected)), name + " classification"
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    dtype = {torch.float16: torch.int16, torch.float32: torch.int32}[actual.dtype]
    assert torch.equal(actual[finite].contiguous().view(dtype), expected[finite].contiguous().view(dtype)), "finite bits/zero signs"
    return {"finite_elements": int(finite.sum()), "nan_elements": int(torch.isnan(actual).sum()),
            "inf_elements": int(torch.isinf(actual).sum()), "finite_bit_mismatches": 0}


def graph_signature(tensor):
    nodes, seen = [], {}
    def visit(node):
        if node is None:
            return None
        if node in seen:
            return seen[node]
        index = len(nodes)
        seen[node] = index
        nodes.append(None)
        nodes[index] = [node.name(), [visit(child) for child, _ in node.next_functions]]
        return index
    visit(tensor.grad_fn)
    return nodes


def state_snapshot(mlp):
    records = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(mlp, name)
        qs = layer.weight.quant_state
        tensors = [layer.weight, qs.absmax, qs.offset, qs.state2.absmax, qs.state2.code]
        records.append((name, [(t.data_ptr(), t._version, t.detach().clone()) for t in tensors]))
    return records


def check_state(mlp, before):
    after = state_snapshot(mlp)
    for (name1, a), (name2, b) in zip(before, after):
        assert name1 == name2
        for (ptr1, ver1, value1), (ptr2, ver2, value2) in zip(a, b):
            assert ptr1 == ptr2 and ver1 == ver2 and torch.equal(value1, value2)


def run_one(mlp, value, cotangent, *, repeat=False, amp=False):
    x = value.detach().clone().requires_grad_()
    cpu_rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(x.device).clone() if x.device.type == "cuda" else None
    context = torch.autocast(x.device.type, dtype=torch.float16, enabled=amp)
    with context:
        output = mlp(x)
    signature = graph_signature(output)
    output.backward(cotangent, retain_graph=repeat)
    if repeat:
        output.backward(cotangent)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if gpu_rng is not None:
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state(x.device))
    return output.detach(), x.grad.detach().clone(), signature
