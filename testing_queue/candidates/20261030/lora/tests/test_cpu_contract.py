"""CPU interpreter/synthetic tests, explicitly NOT CUDA or pinned PEFT integration."""
import ast
import copy
import os
import sys
from pathlib import Path
os.environ['TRITON_INTERPRET'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from solution.epilogue import _ScaleAdd, scale_add
from solution.patch import transform_forward


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    assert torch.equal(torch.isposinf(a), torch.isposinf(b))
    assert torch.equal(torch.isneginf(a), torch.isneginf(b))
    finite = torch.isfinite(a) & torch.isfinite(b)
    assert torch.equal(a[finite], b[finite])
    zeros = finite & (a == 0) & (b == 0)
    assert torch.equal(torch.signbit(a[zeros]), torch.signbit(b[zeros]))


def test_interpreter_values_and_vjp():
    torch.manual_seed(710)
    for shape in [(1, 1), (17, 256), (2, 9, 1536), (1, 1025)]:
        for dtype in [torch.float16, torch.float32]:
            base = torch.randn(shape, dtype=dtype, requires_grad=True)
            delta = torch.randn(shape, dtype=torch.float16, requires_grad=True)
            before_base, before_delta = base.detach().clone(), delta.detach().clone()
            actual = _ScaleAdd.apply(base, delta)
            expected = base + delta * 2.0
            exact(actual, expected)
            assert actual.data_ptr() != base.data_ptr() and actual.data_ptr() != delta.data_ptr()
            exact(base, before_base); exact(delta, before_delta)
            for incoming_scale in [1.0, 1000.0, 65536.0]:
                g = torch.randn_like(actual) * incoming_scale
                got = torch.autograd.grad(actual, (base, delta), g, retain_graph=True)
                want = torch.autograd.grad(expected, (base, delta), g, retain_graph=True)
                for a, b in zip(got, want): exact(a, b)


def test_nonfinite_signed_zero_and_half_overflow():
    d = torch.tensor([-0., 0., 40000., -40000., float('inf'), -float('inf'), float('nan'), 2**-24], dtype=torch.float16)
    for dtype in [torch.float16, torch.float32]:
        b = torch.tensor([-0., 0., -80000., 80000., -float('inf'), float('inf'), 3., -2**-23], dtype=dtype)
        exact(_ScaleAdd.apply(b, d), b + d * 2.0)
    # Mandatory half overflow cannot be canceled by the float base.
    got = _ScaleAdd.apply(torch.tensor([-80000.]), torch.tensor([40000.], dtype=torch.float16))
    assert torch.isposinf(got).all()


def test_native_fallback_cpu_and_scaling():
    for scale in [0.0, 1.0, 2.0, 3.5]:
        b = torch.randn(3, 8, requires_grad=True)
        d = torch.randn(3, 8, dtype=torch.float16, requires_grad=True)
        exact(scale_add(b, d, scale), b + d * scale)


def test_second_derivative_native_backward():
    base = torch.randn(3, 9, requires_grad=True)
    delta = torch.randn(3, 9, dtype=torch.float16, requires_grad=True)
    def grad2(y):
        first = torch.autograd.grad(y.square().sum(), (base, delta), create_graph=True)
        second = torch.autograd.grad(sum(g.float().sum() for g in first), (base, delta))
        return first, second
    for a, b in zip(grad2(_ScaleAdd.apply(base, delta)), grad2(base + delta * 2.0)):
        for x, y in zip(a, b): exact(x, y)


# Synthetic miniature of the operation ordering, not a copy of PEFT source.
FORWARD = '''
def forward(self, x):
    result = self.base(x).clone()
    lora_A, lora_B, dropout, scaling = self.a, self.b, self.drop, 2.0
    output = lora_B(lora_A(dropout(x))) * scaling
    result = result + output
    return result
'''


def test_ast_rejects_unrecognized_source():
    import pytest
    for bad in [FORWARD.replace('* scaling', '* 3'), FORWARD.replace('result + output', 'output + result')]:
        with pytest.raises(RuntimeError): transform_forward(bad)
    transformed = ast.unparse(transform_forward(FORWARD))
    assert 'self.base(x).clone()' in transformed
    assert 'lora_B(lora_A(dropout(x)))' in transformed
    assert transformed.count('_soup_original_scale_add') == 1


def test_separate_qv_rng_native_dropout_and_training():
    torch.manual_seed(23)
    class Fixture(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.base = torch.nn.Linear(32, width, bias=False).to(torch.float16)
            self.a = torch.nn.Linear(32, 16, bias=False).to(torch.float16)
            self.b = torch.nn.Linear(16, width, bias=False).to(torch.float16)
            self.drop = torch.nn.Dropout(.05)
            self.base.weight.requires_grad_(False)
    native_scope = {}; exec(compile(ast.parse(FORWARD), '<synthetic-native>', 'exec'), native_scope)
    custom_scope = {'_soup_original_scale_add': lambda b, d, s: _ScaleAdd.apply(b, d)}
    exec(compile(transform_forward(FORWARD), '<synthetic-transformed>', 'exec'), custom_scope)
    native = [Fixture(32), Fixture(8)]; candidate = copy.deepcopy(native)
    x = torch.randn(2, 9, 32, dtype=torch.float16)
    rng = torch.get_rng_state()
    def run(layers, fn):
        observed_masks = []
        handles = [m.drop.register_forward_hook(lambda module, args, out: observed_masks.append((out != 0).clone())) for m in layers]
        inp = x.detach().clone().requires_grad_(True)
        outputs = [fn(m, inp) for m in layers]
        sum(y.float().square().sum() for y in outputs).backward()
        for h in handles: h.remove()
        return outputs, inp.grad, [p.grad for m in layers for p in m.parameters() if p.requires_grad], torch.get_rng_state(), observed_masks
    a = run(native, native_scope['forward']); torch.set_rng_state(rng)
    b = run(candidate, custom_scope['forward'])
    for xa, xb in zip(a[0], b[0]): exact(xa, xb)
    exact(a[1], b[1])
    for xa, xb in zip(a[2], b[2]): exact(xa, xb)
    assert torch.equal(a[3], b[3])
    for ma, mb in zip(a[4], b[4]): assert torch.equal(ma, mb)
    assert not torch.equal(a[4][0], a[4][1]), 'q/v must have separate native dropout draws'
