"""Standalone row-copy proofs and physical native-SM75 attention screens.

CPU interpreter tests execute only our gather/scatter kernels and their
autograd adjoints. They never substitute an attention implementation into a
production call. Physical tests call the actual Torch efficient-attention
forward/backward, native AMP/GEMMs/dropout/RoPE/repeat_kv and GradScaler.
Nothing installs this experiment into the solution or changes a task flag.
"""

from dataclasses import replace
import inspect
import json
import math
import os

import pytest
import torch
import torch.nn.functional as F

from . import varlen_attention as varlen


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
CPU_KERNEL = pytest.mark.skipif(not INTERPRETING, reason="requires TRITON_INTERPRET=1")
PHYSICAL = pytest.mark.skipif(not torch.cuda.is_available() or INTERPRETING,
                             reason="requires physical CUDA, not the interpreter")


def _mask(lengths, sequence=None):
    sequence = max(lengths) if sequence is None else sequence
    return torch.arange(sequence)[None, :] < torch.tensor(lengths)[:, None]


def _reference_pack(value, plan):
    return value.permute(0, 2, 1, 3).reshape(
        plan.batch * plan.sequence, value.shape[1], value.shape[3]
    ).half().index_select(0, plan.rows.long()).unsqueeze(0)


def _reference_scatter(value, plan, dtype):
    out = torch.zeros((plan.batch * plan.sequence, *value.shape[-2:]),
                      dtype=dtype, device=value.device)
    out = out.index_copy(0, plan.rows.long(), value.squeeze(0).to(dtype))
    return out.reshape(plan.batch, plan.sequence, *value.shape[-2:]).permute(0, 2, 1, 3)


def _assert_native_backward_maximum(function, args, kwargs, expected):
    # Bind by the production wrapper's names: logsumexp follows maxQ/maxK
    # and is a multi-element tensor, not another sequence-length argument.
    bound = inspect.signature(function).bind(*args, **kwargs).arguments
    assert type(bound["maximum_q"]) is type(bound["maximum_k"]) is int
    assert bound["maximum_q"] == bound["maximum_k"] == expected


def test_varlen_cpu_backward_spy_binds_maximum_metadata_without_tensor_truth():
    """Only argument binding is tested; no native CUDA work is substituted."""
    placeholders = [object() for _ in range(6)]
    logsumexp = torch.zeros(2, 12, 128)
    arguments = (*placeholders, 127, 127, logsumexp, object(), object(), 128**-0.5)
    _assert_native_backward_maximum(varlen._native_backward, arguments, {}, 127)
    keyword_arguments = inspect.signature(varlen._native_backward).bind(*arguments).arguments
    assert keyword_arguments["logsumexp"] is logsumexp
    _assert_native_backward_maximum(varlen._native_backward, (), keyword_arguments, 127)
    with pytest.raises(AssertionError):
        _assert_native_backward_maximum(varlen._native_backward,
            (*arguments[:7], 128, *arguments[8:]), {}, 127)


@pytest.mark.parametrize("lengths", [(9, 5, 1), (8, 8, 8), (1,)])
def test_varlen_cpu_plan_preserves_sequences_and_order(lengths):
    mask = _mask(lengths)
    before = mask.clone()
    plan = varlen.make_varlen_plan(mask, device="cpu")
    assert plan.lengths == lengths and plan.total == sum(lengths)
    assert plan.maximum == max(lengths) and plan.batch == len(lengths)
    assert plan.cumulative.dtype == torch.int32
    assert plan.cumulative.tolist() == [0, *torch.tensor(lengths).cumsum(0).tolist()]
    assert plan.rows.tolist() == mask.flatten().nonzero().flatten().tolist()
    assert torch.equal(plan.inverse[plan.rows.long()], torch.arange(plan.total, dtype=torch.int32))
    assert torch.all(plan.inverse[~mask.flatten()] == -1)
    assert torch.equal(mask, before)
    varlen._check_plan(plan, torch.device("cpu"))


@pytest.mark.parametrize("mask", [
    torch.tensor([[0, 1, 1]]), torch.tensor([[1, 0, 1]]), torch.tensor([[0, 0, 0]]),
    torch.tensor([[1, 2, 0]]), torch.tensor([[1.0, 0.0]]), torch.tensor([1, 1]),
    torch.empty((0, 3), dtype=torch.long), torch.empty((2, 0), dtype=torch.long),
])
def test_varlen_cpu_plan_rejects_unsupported_masks(mask):
    with pytest.raises(ValueError):
        varlen.make_varlen_plan(mask, device="cpu")


def test_varlen_plan_rejects_mutation_and_wrong_metadata():
    plan = varlen.make_varlen_plan(_mask((5, 3)), device="cpu")
    with pytest.raises(ValueError, match="invalid"):
        varlen._check_plan(replace(plan, total=plan.total + 1), torch.device("cpu"))
    plan.inverse[0] = -1
    with pytest.raises(ValueError, match="mutated"):
        varlen._check_plan(plan, torch.device("cpu"))


def test_varlen_plan_can_reuse_original_int64_mlp_maps():
    original = varlen.make_varlen_plan(_mask((9, 5, 1)), device="cpu")
    rows, inverse, cumulative = original.rows.long(), original.inverse.long(), original.cumulative
    shared = varlen.plan_from_row_tensors(rows, inverse, cumulative,
                                         lengths=original.lengths, sequence=original.sequence)
    assert shared.rows is rows and shared.inverse is inverse and shared.cumulative is cumulative
    assert shared.maximum == 9 and shared.total == 15
    with pytest.raises(ValueError, match="invalid"):
        varlen.plan_from_row_tensors(rows, inverse, cumulative.long(),
                                     lengths=original.lengths, sequence=original.sequence)


def test_varlen_entry_requires_private_finite_physical_training():
    plan = varlen.make_varlen_plan(_mask((5, 3)), device="cpu")
    q = torch.zeros((2, 2, 5, 8), requires_grad=True)
    for flags in ({}, {"private_training": True}, {"finite_training": True}):
        with pytest.raises(ValueError, match="private, finite"):
            varlen.varlen_attention(q, q, q, plan, **flags)
    with pytest.raises(RuntimeError, match="physical CUDA"):
        varlen.varlen_attention(q, q, q, plan, private_training=True, finite_training=True)


@CPU_KERNEL
@pytest.mark.parametrize("lengths", [(9, 5, 1), (8, 8, 8)])
@pytest.mark.parametrize("layout", ["bthd", "bhtd"])
def test_varlen_interpreter_pack_scatter_exact_casts_and_padding(lengths, layout):
    generator = torch.Generator().manual_seed(11231)
    plan = varlen.make_varlen_plan(_mask(lengths), device="cpu")
    batch, sequence, heads, width = plan.batch, plan.sequence, 4, 16
    values = tuple(torch.randn((batch, sequence, heads, width), generator=generator).permute(0, 2, 1, 3)
                   for _ in range(3))
    if layout == "bhtd":
        values = tuple(value.contiguous() for value in values)
    values[0][0, 0, 0, 0] = -0.0
    rng = torch.random.get_rng_state().clone()
    actual = varlen._pack_qkv(*values, plan)
    for got, source in zip(actual, values):
        expected = _reference_pack(source, plan)
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
        assert got.dtype == torch.float16 and got.is_contiguous()
        assert torch.equal(torch.signbit(got), torch.signbit(expected))
    scattered = varlen._scatter_qkv(actual, plan)
    for got, packed in zip(scattered, actual):
        torch.testing.assert_close(got, _reference_scatter(packed, plan, torch.float32), atol=0, rtol=0)
        assert got.dtype == torch.float32
        assert not got.permute(0, 2, 1, 3)[~_mask(lengths)].count_nonzero()
    assert torch.equal(rng, torch.random.get_rng_state())


@CPU_KERNEL
def test_varlen_interpreter_pack_adjoint_keeps_grouped_kv_sum_fp32():
    from transformers.integrations.sdpa_attention import repeat_kv

    generator = torch.Generator().manual_seed(941)
    plan = varlen.make_varlen_plan(_mask((7, 3)), device="cpu")
    q = torch.randn((2, 12, 7, 8), generator=generator, requires_grad=True)
    k = torch.randn((2, 2, 7, 8), generator=generator, requires_grad=True)
    v = torch.randn((2, 2, 7, 8), generator=generator, requires_grad=True)
    packed = varlen._PackQKV.apply(q, repeat_kv(k, 6), repeat_kv(v, 6), plan)
    head_weights = torch.tensor([1, 2**-11, 0, 0, 0, 0] * 2, dtype=torch.float16)
    upstream = head_weights[None, None, :, None].expand_as(packed[0]).contiguous()
    actual = torch.autograd.grad(packed, (q, k, v), grad_outputs=(upstream,) * 3)
    native = tuple(_reference_pack(value, plan) for value in (q, repeat_kv(k, 6), repeat_kv(v, 6)))
    expected = torch.autograd.grad(native, (q, k, v), grad_outputs=(upstream,) * 3)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
        assert got.dtype == torch.float32
    # A half sum before upcasting would round this halfway value to one.
    assert actual[1][0, 0, 0, 0] == 1 + 2**-11
    assert actual[1][0, 0, 0, 0] != (head_weights[:6].sum().float())
    # An unused Q/V output must not manufacture an input derivative.
    q2, k2, v2 = (value.detach().clone().requires_grad_(True) for value in (q, k, v))
    only_key = varlen._PackQKV.apply(q2, repeat_kv(k2, 6), repeat_kv(v2, 6), plan)[1]
    dq, dk, dv = torch.autograd.grad(only_key, (q2, k2, v2), grad_outputs=upstream, allow_unused=True)
    assert dq is None and dv is None
    torch.testing.assert_close(dk, expected[1], atol=0, rtol=0)


@CPU_KERNEL
def test_varlen_interpreter_output_scatter_adjoint_handles_strided_gradient():
    generator = torch.Generator().manual_seed(891)
    plan = varlen.make_varlen_plan(_mask((9, 4, 2)), device="cpu")
    packed = torch.randn((1, plan.total, 4, 16), dtype=torch.float16,
                          generator=generator, requires_grad=True)
    actual = varlen._ScatterOutput.apply(packed, plan)
    expected = _reference_scatter(packed, plan, torch.float16)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.permute(0, 2, 1, 3).is_contiguous()
    gradient = torch.randn((3, 4, 9, 32), dtype=torch.float16, generator=generator)[..., ::2]
    assert gradient.stride(-1) == 2
    got, = torch.autograd.grad(actual, packed, grad_outputs=gradient)
    want, = torch.autograd.grad(expected, packed, grad_outputs=gradient)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@CPU_KERNEL
def test_varlen_interpreter_backward_rejects_changed_plan_and_higher_order():
    plan = varlen.make_varlen_plan(_mask((5, 3)), device="cpu")
    q = torch.ones((2, 2, 5, 8), requires_grad=True)
    packed = varlen._PackQKV.apply(q, q, q, plan)
    plan.rows[0] = 1
    with pytest.raises(ValueError, match="mutated"):
        torch.autograd.grad(packed[0].sum(), q)
    plan = varlen.make_varlen_plan(_mask((5, 3)), device="cpu")
    packed = varlen._PackQKV.apply(q, q, q, plan)
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(packed[0].sum(), q, create_graph=True)


def _physical_t4():
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("native variable-length admission is specific to SM75 T4")
    assert torch.cuda.device_count() == 1
    assert torch.__version__.split("+")[0] == "2.11.0"


def _fp16_screen(name, actual, expected, scale=1.0):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a, b = (value.detach().to(device="cpu", dtype=torch.float64) / scale
            for value in (actual, expected))
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
    delta = a - b
    norm, error = float(torch.linalg.vector_norm(b)), float(torch.linalg.vector_norm(delta))
    relative = error / norm if norm else (0.0 if error == 0 else None)
    peak, absolute = float(b.abs().max()), float(delta.abs().max())
    epsilon = torch.finfo(torch.float16).eps
    details = {"name": name, "relative_l2": relative, "max_abs": absolute,
               "reference_max_abs": peak, "relative_limit": 2 * epsilon,
               "max_abs_limit": 4 * epsilon * peak + 1e-7,
               "reference_zero_norm": norm == 0,
               "legacy_element_mismatches": int((~torch.isclose(a, b, atol=3e-5, rtol=3e-3)).sum())}
    details["passed"] = relative is not None and relative <= details["relative_limit"] and absolute <= details["max_abs_limit"]
    return details


def _pipeline_inputs(lengths):
    generator = torch.Generator().manual_seed(73129)
    batch, length, hidden, kv_width = len(lengths), max(lengths), 1536, 256
    def draw(shape, amplitude, trainable=False):
        return (torch.randn(shape, generator=generator) * amplitude).cuda().requires_grad_(trainable)
    return {"x": draw((batch, length, hidden), 0.25, True),
            "wq": draw((hidden, hidden), hidden**-0.5).half(),
            "wk": draw((kv_width, hidden), hidden**-0.5).half(),
            "wv": draw((kv_width, hidden), hidden**-0.5).half(),
            "aq": draw((16, hidden), 0.02, True), "bq": draw((hidden, 16), 0.02, True),
            "av": draw((16, hidden), 0.02, True), "bv": draw((kv_width, 16), 0.02, True),
            "upstream": draw((batch, 12, length, 128), (batch * 12 * length * 128)**-0.5)}


def _run_physical_pipeline(values, lengths, scale, *, packed):
    from transformers.integrations.sdpa_attention import repeat_kv
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    plan = varlen.make_varlen_plan(_mask(lengths), device="cuda")
    mask = _mask(lengths).cuda()
    batch, length = mask.shape
    x = values["x"]
    torch.manual_seed(18237)
    torch.cuda.manual_seed_all(18237)
    rng_before = torch.cuda.get_rng_state().clone()
    dropout_shapes, dropout_values = [], []
    def native_dropout(value):
        dropout_shapes.append((tuple(value.shape), str(value.dtype)))
        result = F.dropout(value, p=0.05, training=True)
        dropout_values.append(result.detach().clone())
        return result
    with torch.autocast("cuda", dtype=torch.float16):
        # Real frozen native GEMMs and FP32-storage rank16 adapter GEMMs.
        # Their only role is producing normal rectangular post-LoRA Q/K/V;
        # this primitive makes no claim about replacing NF4 or PEFT itself.
        q = F.linear(x, values["wq"]).float()
        q = q + F.linear(F.linear(native_dropout(x), values["aq"]), values["bq"]) * 2.0
        k = F.linear(x, values["wk"]).float()
        v = F.linear(x, values["wv"]).float()
        v = v + F.linear(F.linear(native_dropout(x), values["av"]), values["bv"]) * 2.0
        q = q.view(batch, length, 12, 128).transpose(1, 2)
        k = k.view(batch, length, 2, 128).transpose(1, 2)
        v = v.view(batch, length, 2, 128).transpose(1, 2)
        angles = (torch.arange(length, device="cuda", dtype=torch.float32)[:, None]
                  * torch.arange(1, 65, device="cuda", dtype=torch.float32)[None, :] * 0.001)
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, angles.cos(), angles.sin())
        expanded_k, expanded_v = repeat_kv(k, 6), repeat_kv(v, 6)
        before_attention = torch.cuda.get_rng_state().clone()
        if packed:
            output = varlen.varlen_attention(q, expanded_k, expanded_v, plan,
                scale=128**-0.5, private_training=True, finite_training=True)
        else:
            positions = torch.arange(length, device="cuda")
            causal = positions[:, None] >= positions[None, :]
            allowed = causal[None, None, :, :] & mask[:, None, None, :]
            # This queries normal native eligibility; it never forces flags.
            assert torch._fused_sdp_choice(q.half(), expanded_k.half(), expanded_v.half(),
                                            allowed, 0.0, False, scale=128**-0.5) == 2
            output = F.scaled_dot_product_attention(q, expanded_k, expanded_v,
                attn_mask=allowed, dropout_p=0.0, is_causal=False, scale=128**-0.5)
    assert output.dtype == torch.float16 and tuple(output.shape) == (batch, 12, length, 128)
    loss = (output.float() * values["upstream"] * mask[:, None, :, None]).sum()
    scaler = torch.amp.GradScaler("cuda", init_scale=scale, growth_interval=1000)
    named = {"q": q, "expanded_k": expanded_k, "expanded_v": expanded_v,
             "unexpanded_k": k, "unexpanded_v": v,
             **{name: values[name] for name in ("x", "aq", "bq", "av", "bv")}}
    gradients = torch.autograd.grad(scaler.scale(loss), tuple(named.values()))
    rng_after = torch.cuda.get_rng_state().clone()
    assert torch.equal(before_attention, rng_after), "dropout-zero attention/backward consumed RNG"
    assert not torch.equal(rng_before, rng_after), "rectangular native dropout did not execute"
    assert dropout_shapes == [((batch, length, 1536), "torch.float32")] * 2
    return {"output": output.detach(), "gradients": dict(zip(named, gradients)),
            "dropout_values": dropout_values, "rng": rng_after, "scaler": scaler.state_dict(),
            "mask": mask, "plan": plan, "dropout_shapes": dropout_shapes}


@PHYSICAL
@pytest.mark.parametrize("lengths", [(127, 83, 45, 7), (128, 128, 128, 128), (512, 303, 163, 84)])
@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_physical_t4_native_varlen_attention_gradients_rng(monkeypatch, record_property, lengths, scale):
    _physical_t4()
    values = _pipeline_inputs(lengths)
    baseline = _run_physical_pipeline(values, lengths, scale, packed=False)
    counts = {"forward": 0, "backward": 0, "packed_shapes": []}
    original_forward, original_backward = varlen._native_forward, varlen._native_backward
    def forward(*args, **kwargs):
        counts["forward"] += 1
        counts["packed_shapes"].append(list(args[0].shape))
        assert args[0].dtype == args[1].dtype == args[2].dtype == torch.float16
        assert args[3].dtype == torch.int32 and args[3].is_cuda
        assert args[3].tolist() == [0, *torch.tensor(lengths).cumsum(0).tolist()]
        assert args[4] == max(lengths)
        return original_forward(*args, **kwargs)
    def backward(*args, **kwargs):
        counts["backward"] += 1
        _assert_native_backward_maximum(original_backward, args, kwargs, max(lengths))
        return original_backward(*args, **kwargs)
    monkeypatch.setattr(varlen, "_native_forward", forward)
    monkeypatch.setattr(varlen, "_native_backward", backward)
    candidate = _run_physical_pipeline(values, lengths, scale, packed=True)
    evidence = []
    for batch, length in enumerate(lengths):
        evidence.append(_fp16_screen(f"output.sequence{batch}",
            candidate["output"][batch, :, :length], baseline["output"][batch, :, :length]))
    assert candidate["gradients"].keys() == baseline["gradients"].keys()
    for name, actual in candidate["gradients"].items():
        evidence.append(_fp16_screen(name, actual, baseline["gradients"][name], scale))
    record_property("soup_varlen_fp16_screen", json.dumps(evidence, allow_nan=False))
    record_property("soup_varlen_native_calls", json.dumps(counts, allow_nan=False))
    assert all(row["passed"] for row in evidence), json.dumps([row for row in evidence if not row["passed"]], indent=2)
    assert counts["forward"] == counts["backward"] == 1
    assert counts["packed_shapes"] == [[1, sum(lengths), 12, 128]]
    assert candidate["scaler"] == baseline["scaler"]
    assert torch.equal(candidate["rng"], baseline["rng"])
    for actual, expected in zip(candidate["dropout_values"], baseline["dropout_values"]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    padded = ~candidate["mask"]
    assert not candidate["output"].permute(0, 2, 1, 3)[padded].count_nonzero()
    for name in ("q", "expanded_k", "expanded_v", "unexpanded_k", "unexpanded_v"):
        assert not candidate["gradients"][name].permute(0, 2, 1, 3)[padded].count_nonzero(), name


@PHYSICAL
def test_physical_t4_native_varlen_causal_sequence_boundaries():
    _physical_t4()
    lengths = (17, 9, 3, 1)
    plan = varlen.make_varlen_plan(_mask(lengths), device="cuda")
    q = torch.zeros((4, 12, 17, 128), device="cuda", requires_grad=True)
    k = torch.zeros_like(q, requires_grad=True)
    positions = torch.arange(17, device="cuda", dtype=torch.float32)
    v = (torch.arange(4, device="cuda", dtype=torch.float32)[:, None, None, None] * 128
         + positions[None, None, :, None]).expand_as(q).contiguous().requires_grad_(True)
    def run(value):
        with torch.autocast("cuda", dtype=torch.float16):
            return varlen.varlen_attention(q, k, value, plan,
                private_training=True, finite_training=True)
    rng = torch.cuda.get_rng_state().clone()
    output = run(v)
    for batch, length in enumerate(lengths):
        # With zero Q/K, each prefix is uniform over its own original example.
        expected = (128 * batch + positions[:length] * 0.5)[None, :, None].expand(12, length, 128)
        torch.testing.assert_close(output[batch, :, :length].float(), expected, atol=0.5, rtol=0.001)
    changed = v.detach().clone().requires_grad_(True)
    with torch.no_grad():
        changed[0, :, lengths[0] - 1] += 1024
        changed[1] += 2048
    modified = run(changed)
    torch.testing.assert_close(modified[0, :, :lengths[0]-1], output[0, :, :lengths[0]-1], atol=0, rtol=0)
    torch.testing.assert_close(modified[2:], output[2:], atol=0, rtol=0)
    mask = _mask(lengths).cuda()
    gradients = torch.autograd.grad((output.float() * mask[:, None, :, None]).sum(), (q, k, v))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert not gradients[0].count_nonzero() and not gradients[1].count_nonzero()
    assert not gradients[2].permute(0, 2, 1, 3)[~mask].count_nonzero()
    assert torch.equal(rng, torch.cuda.get_rng_state())
