"""Actual CUDA same-bit, stream-order, backward and RNG component probe."""
import contextlib
import torch
from . import activation_offload as module


def probe():
    class Owner:
        def forward(self, value): return value
    handle = module.ActivationOffloadInstallation(Owner, Owner.forward)
    Owner.forward = handle.forward
    records = []
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(123)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.randn((4, 256, 1536), device='cuda', dtype=torch.float32).requires_grad_()
            packed = handle.pack(x)
        assert type(packed) is module._Packed
        restored = handle.unpack(packed)
        torch.cuda.synchronize()
        assert torch.equal(restored.view(torch.int32), x.view(torch.int32))
        records.append({'name': 'cross_stream_same_bits', 'passed': True})
        transposed = x.transpose(0, 1)
        restored_transpose = handle.unpack(handle.pack(transposed))
        assert restored_transpose.stride() == transposed.stride()
        assert torch.equal(restored_transpose.contiguous().view(torch.int32),
                           transposed.contiguous().view(torch.int32))
        records.append({'name': 'transpose_layout_same_bits', 'passed': True})
        base = torch.randn((256, 1536), device='cuda')
        weight = torch.randn((1536, 1024), device='cuda') / 64
        incoming = torch.randn((256, 1024), device='cuda') / 256
        def run(offload):
            leaf = base.clone().requires_grad_()
            torch.cuda.manual_seed(9)
            context = (torch.autograd.graph.saved_tensors_hooks(handle.pack, handle.unpack)
                       if offload else contextlib.nullcontext())
            with context:
                dropped = torch.nn.functional.dropout(leaf, p=.05, training=True)
                output = (dropped @ weight).sin()
            output.backward(incoming)
            return output.detach(), leaf.grad.detach(), torch.cuda.get_rng_state()
        reference, observed = run(False), run(True)
        for name, actual, expected in zip(('output', 'input_gradient', 'rng_state'), observed, reference):
            assert torch.equal(actual, expected), name
            records.append({'name': name, 'passed': True})
        assert handle.counts['packed_tensors'] >= 3
        assert handle.counts['unpacked_tensors'] >= 3
        torch.cuda.synchronize()
    return {'status': 'PASS', 'scope': 'CUDA_COMPONENT_PROBE_NOT_FULL_TASK',
            'checks': records, 'counts': dict(handle.counts)}
