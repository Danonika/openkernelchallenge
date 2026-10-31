"""Optional MLP-only padding checks, with explicit CPU fixture boundaries.

CPU interpreter tests do not claim CUDA arithmetic equivalence. The physical
test below uses actual nested-NF4 CUDA modules and does not replace a GEMM,
device guard, dropout call, or random generator.
"""

import os
import json

import pytest
import torch
import torch.nn.functional as F

from . import mlp_padding as padding


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
CPU_KERNEL = pytest.mark.skipif(not INTERPRETING, reason="requires TRITON_INTERPRET=1")
PHYSICAL = pytest.mark.skipif(not torch.cuda.is_available() or INTERPRETING,
                             reason="requires physical CUDA, not the interpreter")


def _restore(monkeypatch):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    for name in ("_QWEN_CAUSAL_LM", "_QWEN_MLP", "_BNB_LINEAR", "_BNB_FORWARD",
                 "_SILU_CLASS", "_SILU_FORWARD"):
        monkeypatch.setattr(padding, name, getattr(padding, name))
    monkeypatch.setattr(Qwen2MLP, "forward", Qwen2MLP.forward)


def _cpu_device(tensor):
    return (tensor.device.type == "cpu" and torch.is_autocast_enabled("cpu")
            and torch.get_autocast_dtype("cpu") == torch.float16)


def _model(device="cpu", width=32, intermediate=64, layers=2):
    import bitsandbytes as bnb
    from bitsandbytes.functional import QuantState, get_4bit_type
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(17313)
    heads = 12 if width == 1536 else 4
    cfg = Qwen2Config(vocab_size=257, hidden_size=width, intermediate_size=intermediate,
        num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=2,
        head_dim=width // heads, max_position_embeddings=256, tie_word_embeddings=True,
        attention_dropout=0.0, pad_token_id=0, eos_token_id=1)
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    model = Qwen2ForCausalLM(cfg)
    dense = {}
    for block in model.model.layers:
        if device == "cuda":
            # The physical screen keeps q/k/v/o native NF4 as well, matching
            # the real PEFT Linear4bit adapter wrapper rather than dense LoRA.
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                native = getattr(block.self_attn, name)
                module = bnb.nn.Linear4bit(native.in_features, native.out_features,
                    bias=native.bias is not None, compute_dtype=torch.float16,
                    compress_statistics=True, quant_type="nf4").half()
                with torch.no_grad():
                    module.weight.copy_(native.weight.half())
                    if native.bias is not None:
                        module.bias.copy_(native.bias.half())
                module = module.to("cuda")
                assert module.weight.quant_state.nested
                setattr(block.self_attn, name, module)
        for name in ("gate_proj", "up_proj", "down_proj"):
            native = getattr(block.mlp, name)
            weight = native.weight.detach().half().clone()
            module = bnb.nn.Linear4bit(native.in_features, native.out_features, bias=False,
                compute_dtype=torch.float16, compress_statistics=True, quant_type="nf4")
            if device == "cuda":
                module = module.half()
                with torch.no_grad():
                    module.weight.copy_(weight)
                module = module.to("cuda")  # Actual BNB nested-NF4 quantization.
                assert module.weight.quant_state.nested
            else:
                # CPU fixture ONLY: real Linear4bit.forward still supplies its
                # casts; the unavailable native GPU matmul is replaced below.
                state = QuantState(absmax=torch.ones((weight.numel() + 63) // 64),
                    code=get_4bit_type("nf4", device="cpu"),
                    shape=weight.shape, dtype=torch.float16, blocksize=64, quant_type="nf4")
                module.weight = bnb.nn.Params4bit(
                    torch.zeros(((weight.numel() + 1) // 2, 1), dtype=torch.uint8),
                    requires_grad=False, quant_state=state, bnb_quantized=True, quant_type="nf4")
                dense[id(module.weight)] = weight
            setattr(block.mlp, name, module)
    model.to(device)
    model.is_loaded_in_4bit = True
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.02)
    assert all("q_proj" in name or "v_proj" in name
               for name, param in model.named_parameters() if param.requires_grad)
    return model.train(), dense


def _batch(device="cpu", length=13):
    generator = torch.Generator().manual_seed(613)
    ids = torch.randint(3, 257, (4, length), generator=generator).to(device)
    lengths = (length, max(1, length * 3 // 4), max(1, length // 2), max(1, length // 4))
    mask = torch.arange(length, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    labels = ids.clone()
    labels[:, :2] = -100
    labels[~mask] = -100
    labels[-1] = -100  # A valid input row may have no supervised targets.
    return dict(input_ids=ids, attention_mask=mask.long(), labels=labels, use_cache=False)


def _context(model, batch, **kwargs):
    return padding.mlp_padding_context(model.get_base_model(),
        attention_mask=batch["attention_mask"], labels=batch["labels"],
        use_cache=batch.get("use_cache"), **kwargs)


def _private_training(model, batch, *, compact, device="cpu", scale=1):
    input_snapshots = {name: batch[name].detach().clone()
                       for name in ("labels", "attention_mask")}
    torch.manual_seed(9431)
    if device == "cuda":
        torch.cuda.manual_seed_all(9431)
    with torch.autocast(device, dtype=torch.float16):
        with _context(model, batch, private_outputs=compact) as call:
            output = model(**batch)
    named_params = tuple((name, p) for name, p in model.named_parameters() if p.requires_grad)
    params = tuple(p for _, p in named_params)
    gradients = torch.autograd.grad(output.loss * scale, params)
    gradient_metadata = _gradient_metadata(named_params, gradients)
    # Consume private logits immediately as the real compact-head owner does;
    # no full logits tensor is published to a user/model-output consumer.
    logits = output.logits.detach()
    with torch.no_grad():
        target = F.pad(batch["labels"], (0, 1), value=-100)[..., 1:]
        valid = batch["attention_mask"].bool()
        logp = F.log_softmax(logits.float(), -1)
        token_loss = F.nll_loss(logp.view(-1, logits.shape[-1]), target.reshape(-1),
                                reduction="none").view_as(target)
        counts = (target != -100).sum(-1)
        means = torch.where(counts > 0, token_loss.sum(-1) / counts, 0.0)
        entropy = (-(logp.exp() * logp).sum(-1) * valid).sum() / valid.sum()
        correct = ((logits.argmax(-1) == target) & (target != -100)).sum()
    rng = torch.cuda.get_rng_state() if device == "cuda" else torch.random.get_rng_state()
    for name, snapshot in input_snapshots.items():
        current = batch[name]
        assert current.shape == snapshot.shape and current.dtype == snapshot.dtype, name
        assert current.device == snapshot.device and torch.equal(current, snapshot), name
    return dict(loss=output.loss.detach(), valid_logits=logits[valid], gradients=gradients,
                gradient_metadata=gradient_metadata,
                row_loss=means, row_tokens=counts, entropy=entropy, correct=correct,
                tokens=valid.sum(), rng=rng, call=call, **input_snapshots)


def _gradient_metadata(named_params, gradients):
    assert len(named_params) == len(gradients) > 0, "missing adapter gradients"
    names = tuple(name for name, _ in named_params)
    assert len(names) == len(set(names)), "duplicate adapter parameter names"
    metadata = []
    for index, (name, parameter) in enumerate(named_params):
        gradient = gradients[index]
        assert gradient.shape == parameter.shape and gradient.dtype == parameter.dtype, name
        assert gradient.device == parameter.device, name
        metadata.append(dict(name=name, shape=list(gradient.shape),
                             dtype=str(gradient.dtype), device=str(gradient.device)))
    return metadata


def _gradient_pairs(actual, expected):
    """Require complete named correspondence before any numerical comparison."""
    assert len(actual["gradients"]) == len(expected["gradients"]) > 0
    assert len(actual["gradient_metadata"]) == len(actual["gradients"])
    assert len(expected["gradient_metadata"]) == len(expected["gradients"])
    assert actual["gradient_metadata"] == expected["gradient_metadata"]
    names = [row["name"] for row in actual["gradient_metadata"]]
    assert len(names) == len(set(names))
    pairs = []
    for index, metadata in enumerate(actual["gradient_metadata"]):
        a, b = actual["gradients"][index], expected["gradients"][index]
        assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device, metadata
        assert (list(a.shape), str(a.dtype), str(a.device)) == (
            metadata["shape"], metadata["dtype"], metadata["device"]), metadata
        pairs.append((metadata, a, b))
    return pairs


def _assert_training(actual, expected, *, scale=1, atol=0, rtol=0):
    for key in ("loss", "valid_logits", "row_loss", "entropy"):
        torch.testing.assert_close(actual[key], expected[key], atol=atol, rtol=rtol,
                                   msg=lambda msg, k=key: f"{k}: {msg}")
    for key in ("row_tokens", "correct", "tokens", "rng", "labels", "attention_mask"):
        assert torch.equal(actual[key], expected[key]), key
    for metadata, actual_grad, expected_grad in _gradient_pairs(actual, expected):
        assert torch.isfinite(actual_grad).all() and torch.isfinite(expected_grad).all()
        torch.testing.assert_close(actual_grad / scale, expected_grad / scale,
                                   atol=atol, rtol=rtol,
                                   msg=lambda msg, n=metadata["name"]: f"{n}: {msg}")


def _fp16_error_screen(actual, expected, *, rows=False):
    """Explicit FP16 graph budget, independent of near-zero element ratios.

    The smaller GEMMs have a different reduction geometry. These are the same
    engineering bounds used in the complete trainer fixture, not bitwise
    equivalence or a replacement for full-task loss/metric verification.
    Logit bounds apply independently to every valid token row; gradient
    bounds apply independently to every trainable adapter tensor.
    """
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.device == expected.device and actual.numel() > 0
    if rows:
        assert actual.ndim == 2
    # Reproduce the old elementwise check in its original tensor dtype. The
    # higher-precision norm calculation below must not redefine its counts.
    old = ~torch.isclose(actual.detach(), expected.detach(), atol=3e-5, rtol=3e-3)
    a, b = (x.detach().to(device="cpu", dtype=torch.float64) for x in (actual, expected))
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    if not rows:
        a, b = a.reshape(1, -1), b.reshape(1, -1)
    old = old.to(device="cpu").reshape(a.shape)
    delta = a - b
    norms = torch.linalg.vector_norm(b, dim=1)
    errors = torch.linalg.vector_norm(delta, dim=1)
    relative = torch.where(norms > 0, errors / norms,
                           torch.where(errors == 0, 0., torch.inf))
    peak = b.abs().amax(dim=1)
    absolute = delta.abs().amax(dim=1)
    epsilon = torch.finfo(torch.float16).eps
    absolute_limit = 4 * epsilon * peak + 1e-7
    passed = (relative <= 2 * epsilon) & (absolute <= absolute_limit)
    failures = (~passed).nonzero(as_tuple=False).flatten().tolist()
    finite_relative = relative[torch.isfinite(relative)]
    # Keep bounded, attributable diagnostics even when an assertion stops
    # the fixture: the first failures plus the worst norm/absolute residuals.
    detail_indices = sorted(set(failures[:16] + [int(relative.argmax()),
        int(absolute.argmax()), int((absolute / absolute_limit).argmax())]))
    details = []
    for index in detail_indices:
        column = int(delta[index].abs().argmax())
        details.append({
            "unit_index": index, "screen_passed": bool(passed[index]),
            "reference_zero_norm": bool(norms[index] == 0),
            "reference_l2": float(norms[index]), "error_l2": float(errors[index]),
            "relative_l2": float(relative[index]) if torch.isfinite(relative[index]) else None,
            "reference_max_abs": float(peak[index]),
            "max_abs_error": float(absolute[index]),
            "max_abs_limit": float(absolute_limit[index]),
            "legacy_element_mismatches": int(old[index].sum()),
            "worst_flat_coordinate_in_unit": column,
            "worst_actual": float(a[index, column]),
            "worst_reference": float(b[index, column]),
        })
    return {
        "screen_passed": bool(passed.all()), "units_checked": len(passed),
        "unit": "valid_token_row" if rows else "adapter_tensor",
        "legacy_element_mismatches": int(old.sum()),
        "maximum_relative_l2": float(finite_relative.max()) if finite_relative.numel() else None,
        "nonfinite_relative_l2_count": int((~torch.isfinite(relative)).sum()),
        "maximum_absolute_error": float(absolute.max()),
        "maximum_absolute_budget_fraction": float((absolute / absolute_limit).max()),
        "relative_l2_limit": 2 * epsilon,
        "absolute_limit_formula": "4 * fp16_epsilon * reference_max_abs + 1e-7",
        "failed_units": failures[:16], "failed_unit_count": len(failures),
        "unit_details": details,
    }


def _spies(monkeypatch):
    counts = {"gather": 0, "scatter": 0}
    class Spy:
        def __init__(self, kernel, name):
            self.kernel, self.name = kernel, name
        def __getitem__(self, grid):
            native = self.kernel[grid]
            def run(*args, **kwargs):
                result = native(*args, **kwargs)
                counts[self.name] += 1
                return result
            return run
    for attr, name in (("soup_mlp_gather_rows", "gather"), ("soup_mlp_scatter_rows", "scatter")):
        monkeypatch.setattr(padding, attr, Spy(getattr(padding, attr), name))
    return counts


def _check_row_copy_exact(width, device):
    torch.manual_seed(59)
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0]], device=device)
    labels = torch.where(mask.bool(), 5, -100)
    rows, inverse = padding._host_row_plan(mask, labels)
    x = torch.randn(3, 4, width, device=device, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    selected = padding._GatherRows.apply(x, rows, inverse)
    actual = padding._ScatterRows.apply(selected.square(), rows, inverse, x.shape)
    expected_selected = expected_x.view(-1, width).index_select(0, rows)
    torch.testing.assert_close(selected, expected_selected, atol=0, rtol=0)
    expected = torch.zeros_like(expected_x).view(-1, width).index_copy(
        0, rows, expected_selected.square()).view_as(expected_x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert rows.tolist() == [0, 1, 2, 3, 4, 8, 9]
    assert torch.count_nonzero(actual[~mask.bool()]) == 0
    # Expanded upstream gradient exercises the copy wrapper's stride fallback.
    upstream = torch.randn(width, device=device).expand_as(actual)
    actual_dx, = torch.autograd.grad(actual, x, upstream)
    expected_dx, = torch.autograd.grad(expected, expected_x, upstream)
    torch.testing.assert_close(actual_dx, expected_dx, atol=0, rtol=0)
    assert torch.count_nonzero(actual_dx[~mask.bool()]) == 0


@CPU_KERNEL
@pytest.mark.parametrize("width", [17, 1536])
def test_row_kernels_preserve_original_order_gradients_and_empty_padding(width):
    _check_row_copy_exact(width, "cpu")


@PHYSICAL
@pytest.mark.parametrize("width", [17, 1536])
def test_physical_t4_mlp_padding_row_copy_exact(width):
    """Isolate physical row-copy indexing and its adjoint from GEMM rounding."""
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("the production candidate supports SM75 T4")
    _check_row_copy_exact(width, "cuda")
    torch.cuda.synchronize()


@CPU_KERNEL
def test_row_copy_higher_order_gradients_use_native_differentiable_fallback():
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    rows, inverse = padding._host_row_plan(mask, torch.where(mask.bool(), 5, -100))
    x = torch.randn(2, 3, 7, requires_grad=True)
    gathered = padding._GatherRows.apply(x, rows, inverse)
    result = padding._ScatterRows.apply(gathered.square(), rows, inverse, x.shape)
    dx, = torch.autograd.grad(result.sum(), x, create_graph=True)
    d2x, = torch.autograd.grad(dx.sum(), x)
    torch.testing.assert_close(d2x, mask[..., None].expand_as(x).float() * 2, atol=0, rtol=0)


def test_host_plan_rejects_unsupported_masks_and_supervised_padding():
    good = torch.tensor([[1, 1, 1], [1, 0, 0]])
    labels = torch.where(good.bool(), 5, -100)
    assert padding._host_row_plan(good, labels) is not None
    for mask in (torch.ones_like(good), torch.tensor([[1, 1, 1], [0, 1, 1]]),
                 torch.tensor([[1, 0, 1], [1, 0, 0]]), torch.tensor([[1, 1, 1], [0, 0, 0]]),
                 torch.tensor([[1, 1, 1], [1, 2, 0]])):
        assert padding._host_row_plan(mask, torch.where(mask.bool(), 5, -100)) is None
    labels[1, 2] = 7
    assert padding._host_row_plan(good, labels) is None


@CPU_KERNEL
@pytest.mark.parametrize("compact_activations,scale", [(False, 1), (True, 65536)])
def test_real_qwen_peft_cpu_fixture_retains_valid_metrics_gradients_and_dropout_rng(
        monkeypatch, compact_activations, scale):
    import bitsandbytes as bnb
    from . import activation_storage

    _restore(monkeypatch)
    model, dense = _model()
    shapes = []
    def cpu_gemm(x, weight, bias=None, quant_state=None, **kwargs):
        shapes.append(tuple(x.shape))
        return F.linear(x, dense[id(weight)], bias)
    monkeypatch.setattr(bnb, "matmul_4bit", cpu_gemm)
    if compact_activations:
        monkeypatch.setattr(activation_storage, "_is_cuda_tensor", lambda x: x.device.type == "cpu")
        assert activation_storage.patch_compact_mlp_activations()
    batch = _batch()
    dropout_calls = []
    original_dropout = torch.nn.Dropout.forward
    def observe(self, x):
        before = torch.random.get_rng_state().clone()
        result = original_dropout(self, x)
        dropout_calls.append((tuple(x.shape), before, torch.random.get_rng_state().clone()))
        return result
    monkeypatch.setattr(torch.nn.Dropout, "forward", observe)
    expected = _private_training(model, batch, compact=False, scale=scale)
    native_dropout = list(dropout_calls)
    assert expected["call"] is None and len(shapes) == 6
    assert all(shape[:2] == (4, 13) for shape in shapes)
    assert padding.patch_mlp_padding()
    installed = type(model.get_base_model().model.layers[0].mlp).forward
    assert padding.patch_mlp_padding()
    assert type(model.get_base_model().model.layers[0].mlp).forward is installed
    monkeypatch.setattr(padding, "_supported_device", _cpu_device)
    counts = _spies(monkeypatch)
    shapes.clear()
    dropout_calls.clear()
    actual = _private_training(model, batch, compact=True, scale=scale)
    _assert_training(actual, expected, scale=scale, atol=3e-6, rtol=3e-4)
    assert actual["call"] is not None
    assert all(shape[0] == int(batch["attention_mask"].sum()) and len(shape) == 2 for shape in shapes)
    assert len(shapes) == 6
    assert counts == {"gather": 4, "scatter": 4}  # Two forward and two backward layers.
    assert len(native_dropout) == len(dropout_calls) == 4
    for (shape, before, after), (native_shape, native_before, native_after) in zip(dropout_calls, native_dropout):
        assert shape == native_shape == (4, 13, 32)
        assert torch.equal(before, native_before) and torch.equal(after, native_after)
    assert padding._ACTIVE_MLP_PADDING.get() is None
    # Even with the installer present, ordinary public model calls still take
    # the complete shape, including their original padding outputs.
    shapes.clear()
    public = _private_training(model, batch, compact=False, scale=scale)
    _assert_training(public, expected, scale=scale, atol=0, rtol=0)
    assert public["call"] is None and all(shape[:2] == (4, 13) for shape in shapes)


def test_context_falls_back_on_public_calls_cpu_and_incompatible_model_contract(monkeypatch):
    _restore(monkeypatch)
    model, _ = _model()
    batch = _batch()
    assert padding.patch_mlp_padding()
    with torch.autocast("cpu", dtype=torch.float16):
        with _context(model, batch, private_outputs=True) as call:
            assert call is None  # Production never admits CPU/interpreter input.
    monkeypatch.setattr(padding, "_supported_device", _cpu_device)
    with torch.autocast("cpu", dtype=torch.float16):
        with _context(model, batch) as call:
            assert call is None
        with _context(model, batch, private_outputs=True, output_hidden_states=True) as call:
            assert call is None
        with _context(model, batch, private_outputs=True, output_attentions=True) as call:
            assert call is None
        with _context(model, dict(batch, use_cache=True), private_outputs=True) as call:
            assert call is None
        with _context(model, batch, private_outputs=True) as outer:
            assert outer is not None
            with _context(model, batch, private_outputs=False) as inner:
                assert inner is None and padding._ACTIVE_MLP_PADDING.get() is None
            assert padding._ACTIVE_MLP_PADDING.get() is outer
        with pytest.raises(RuntimeError, match="fixture"):
            with _context(model, batch, private_outputs=True):
                raise RuntimeError("fixture")
        assert padding._ACTIVE_MLP_PADDING.get() is None
        decoder = model.get_base_model().model
        decoder.gradient_checkpointing = True
        with _context(model, batch, private_outputs=True) as call:
            assert call is None
        decoder.gradient_checkpointing = False
        original_activation = decoder.layers[0].mlp.act_fn
        decoder.layers[0].mlp.act_fn = torch.nn.Dropout(0.1)
        with _context(model, batch, private_outputs=True) as call:
            assert call is None  # Do not compact a stochastic MLP override.
        decoder.layers[0].mlp.act_fn = original_activation
        # An MLP LoRA/other adapter fails exact ordinary-Linear4bit admission.
        decoder.layers[0].mlp.gate_proj = torch.nn.Linear(32, 64, bias=False)
        with _context(model, batch, private_outputs=True) as call:
            assert call is None


@CPU_KERNEL
@pytest.mark.parametrize("head_mode", ["cached", "streamed"])
@pytest.mark.parametrize("cache_argument,config_cache,admitted", [
    (None, False, True), (False, True, True), (None, True, False),
])
def test_private_head_wiring_builds_one_plan_and_preserves_effective_cache_behavior(
        monkeypatch, head_mode, cache_argument, config_cache, admitted):
    """Use the actual head helper, original decoder and all adapter gradients."""
    import bitsandbytes as bnb
    from . import integration

    _restore(monkeypatch)
    model, dense = _model()
    monkeypatch.setattr(bnb, "matmul_4bit", lambda x, weight, bias=None, **kwargs:
                        F.linear(x, dense[id(weight)], bias))
    causal = model.get_base_model()
    causal.config.use_cache = config_cache
    batch = dict(_batch(), use_cache=cache_argument)
    original_labels = batch["labels"].clone()
    original_mask = batch["attention_mask"].clone()
    expected = _private_training(model, batch, compact=False)
    assert padding.patch_mlp_padding()
    monkeypatch.setattr(padding, "_supported_device", _cpu_device)
    monkeypatch.setattr(integration, "MLP_PADDING_COMPACTION", True)
    monkeypatch.setattr(integration, "HEAD_NUMERICS", "native")
    monkeypatch.setattr(integration, "HEAD_PROJECTION", "torch")
    plans, decoder_calls = [], []
    native_plan = padding._host_row_plan
    def plan(mask, labels):
        result = native_plan(mask, labels)
        plans.append(result)
        return result
    monkeypatch.setattr(padding, "_host_row_plan", plan)
    counts = _spies(monkeypatch)
    def observe(module, args, kwargs):
        decoder_calls.append((kwargs.get("use_cache"), padding._ACTIVE_MLP_PADDING.get()))
    handle = causal.model.register_forward_pre_hook(observe, with_kwargs=True)
    torch.manual_seed(9431)
    try:
        with torch.autocast("cpu", dtype=torch.float16):
            if head_mode == "cached":
                output, result = integration._forward_cached(causal, type(causal).forward, **batch)
            else:
                output, result = integration._forward_streamed(causal, **batch)
        named_params = tuple((name, p) for name, p in model.named_parameters() if p.requires_grad)
        gradients = torch.autograd.grad(output.loss, tuple(p for _, p in named_params))
    finally:
        handle.remove()
    assert len(decoder_calls) == 1 and decoder_calls[0][0] is cache_argument
    assert (decoder_calls[0][1] is not None) is admitted
    assert len(plans) == int(admitted)
    assert counts == {"gather": 4 * int(admitted), "scatter": 4 * int(admitted)}
    assert causal.config.use_cache is config_cache
    assert batch["use_cache"] is cache_argument
    assert (output.past_key_values is None) is (cache_argument is False or not config_cache)
    assert output.logits is None
    assert torch.equal(batch["labels"], original_labels)
    assert torch.equal(batch["attention_mask"], original_mask)
    assert padding._ACTIVE_MLP_PADDING.get() is None
    assert torch.equal(torch.random.get_rng_state(), expected["rng"])
    torch.testing.assert_close(output.loss, expected["loss"], atol=3e-6, rtol=3e-4)
    valid = batch["attention_mask"].bool()
    entropy = (result.entropy * valid).sum() / valid.sum()
    torch.testing.assert_close(entropy, expected["entropy"], atol=3e-6, rtol=3e-4)
    targets = F.pad(batch["labels"], (0, 1), value=-100)[..., 1:]
    correct = ((result.predictions == targets) & (targets != -100)).sum()
    assert torch.equal(correct, expected["correct"])
    row_tokens = (targets != -100).sum(-1)
    row_loss = torch.where(row_tokens > 0, result.token_loss.sum(-1) / row_tokens, 0.0)
    torch.testing.assert_close(row_loss, expected["row_loss"], atol=3e-6, rtol=3e-4)
    gradient_record = dict(gradients=gradients,
                          gradient_metadata=_gradient_metadata(named_params, gradients))
    for metadata, actual, native in _gradient_pairs(gradient_record, expected):
        torch.testing.assert_close(actual, native, atol=3e-6, rtol=3e-4,
                                   msg=lambda msg, n=metadata["name"]: f"{n}: {msg}")


@PHYSICAL
@pytest.mark.parametrize("width,intermediate,layers,length", [(64, 128, 2, 13), (1536, 8960, 1, 128)])
def test_physical_t4_mlp_padding_native_nf4_gradients_rng_and_metrics(
        monkeypatch, width, intermediate, layers, length, record_property):
    """Real frozen NF4 MLPs, real q/v PEFT, no guard or GEMM substitutions."""
    from . import apply
    from .test_fusion_cuda import _register_patch_restoration

    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("the production candidate supports SM75 T4")
    _register_patch_restoration(monkeypatch)
    _restore(monkeypatch)
    apply()  # Exercise the current body composition before adding this option.
    model, _ = _model("cuda", width=width, intermediate=intermediate, layers=layers)
    batch = _batch("cuda", length=length)
    expected = _private_training(model, batch, compact=False, device="cuda", scale=65536)
    assert padding.patch_mlp_padding()
    counts = _spies(monkeypatch)
    actual = _private_training(model, batch, compact=True, device="cuda", scale=65536)
    assert actual["call"] is not None
    assert actual["call"].rows.numel() == int(batch["attention_mask"].sum())
    assert counts == {"gather": layers * 2, "scatter": layers * 2}
    gradient_pairs = _gradient_pairs(actual, expected)
    evidence = {"valid_logits": _fp16_error_screen(actual["valid_logits"],
                                                   expected["valid_logits"], rows=True),
                "gradients": [dict(metadata, **_fp16_error_screen(a / 65536, b / 65536))
                              for metadata, a, b in gradient_pairs]}
    positions = actual["attention_mask"].bool().nonzero(as_tuple=False).cpu().tolist()
    for row in evidence["valid_logits"]["unit_details"]:
        row["batch_position"] = positions[row["unit_index"]]
    record_property("soup_mlp_padding_fp16_evidence", json.dumps(evidence, allow_nan=False))
    assert evidence["valid_logits"]["screen_passed"], evidence["valid_logits"]
    assert all(row["screen_passed"] for row in evidence["gradients"]), evidence["gradients"]
    # Retain the original consumer, label/token, exact RNG and accuracy gates.
    for key in ("loss", "row_loss", "entropy"):
        torch.testing.assert_close(actual[key], expected[key], atol=3e-5, rtol=3e-3)
    for key in ("row_tokens", "correct", "tokens", "rng", "labels", "attention_mask"):
        assert torch.equal(actual[key], expected[key]), key
    torch.cuda.synchronize()
