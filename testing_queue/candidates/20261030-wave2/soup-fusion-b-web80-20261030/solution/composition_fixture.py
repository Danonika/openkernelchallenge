"""Original combined boundary fixture, explicitly not full PEFT/TRL training."""
import os
import torch
import torch.nn.functional as F
from transformers.loss.loss_utils import ForCausalLMLoss
from .memory_component import adapter as memory
from .memory_component.fixture_common import make_mlp, compare, state_snapshot, check_state
from .head_component.native_head import NativeHeadLoss, Telemetry
from .rotary_component import rotary
from .rotary_component.cpu_contract import inputs, reference


def run_combined(device='cpu', scale=1.0, outer_saved_hooks=False, retain=False):
    if device == 'cpu' and os.environ.get('TRITON_INTERPRET') != '1':
        raise RuntimeError('Combined CPU kernel fixture explicitly requires Triton interpreter')
    hidden, intermediate = (64, 128) if device == 'cpu' else (1536, 8960)
    mlp = make_mlp(hidden, intermediate, device)
    before = state_snapshot(mlp)
    q, k, c, s = inputs(b=1, t=3, device=device)
    generator = torch.Generator(device='cpu').manual_seed(68126)
    head_weight = (torch.randn((37, hidden), generator=generator) * .05).to(device)
    labels = torch.tensor([[1, 2, 3]], device=device, dtype=torch.int64)
    telemetry = Telemetry(stream=False)
    head = NativeHeadLoss(ForCausalLMLoss, telemetry, expected_vocabulary=37, devices=(device,))
    initial_cpu = torch.random.get_rng_state().clone()
    initial_gpu = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    records = []
    for enabled in (False, True):
        memory.uninstall()
        if enabled:
            result = memory.install(mode='cpu_equivalent' if device == 'cpu' else 'production')
            assert result['installed'], result
        memory.reset_stats(); rotary._COUNTS.clear()
        torch.random.set_rng_state(initial_cpu)
        if initial_gpu is not None: torch.cuda.set_rng_state(initial_gpu)
        local_q = q.detach().clone().requires_grad_()
        local_k = k.detach().clone().requires_grad_()
        pack_counts = {'pack': 0, 'unpack': 0}
        def pack(value):
            pack_counts['pack'] += 1
            return value.detach()
        def unpack(value):
            pack_counts['unpack'] += 1
            return value
        import contextlib
        context = torch.autograd.graph.saved_tensors_hooks(pack, unpack) if outer_saved_hooks else contextlib.nullcontext()
        with context:
            fn = rotary.fused_apply_rotary_pos_emb if enabled else reference
            qr, kr = fn(local_q, local_k, c, s)
            qh = qr.transpose(1, 2).reshape(1, 3, -1)
            kh = kr.repeat_interleave(6, dim=1).transpose(1, 2).reshape(1, 3, -1)
            # An explicit toy residual/dropout edge exercises shared ancestry
            # and RNG; it is not a replacement of the official task's dropout.
            x = F.dropout((qh[..., :hidden] + kh[..., :hidden]).contiguous() * .01,
                          p=.05, training=True)
            y = mlp(x)
            with torch.autocast(device, dtype=torch.float16):
                logits = F.linear(y, head_weight)
                loss = head(logits, labels, 37, num_items_in_batch=2) if enabled else ForCausalLMLoss(
                    logits=logits, labels=labels, vocab_size=37, num_items_in_batch=2)
        gradients = torch.autograd.grad(loss, (local_q, local_k),
                                        grad_outputs=torch.full_like(loss, scale), retain_graph=retain)
        if retain:
            repeated = torch.autograd.grad(loss, (local_q, local_k), grad_outputs=torch.full_like(loss, scale))
            for a, b in zip(gradients, repeated): compare(a, b)
        records.append({'logits': logits.detach(), 'loss': loss.detach(),
                        'gradients': gradients, 'cpu_rng': torch.random.get_rng_state().clone(),
                        'cuda_rng': torch.cuda.get_rng_state().clone() if device == 'cuda' else None,
                        'memory': memory.snapshot(), 'rotary': dict(rotary._COUNTS), 'outer_hooks': pack_counts})
    baseline, candidate = records
    checks = {'logits': compare(candidate['logits'], baseline['logits']),
              'loss': compare(candidate['loss'], baseline['loss']),
              'gradients': [compare(a,b) for a,b in zip(candidate['gradients'],baseline['gradients'])]}
    assert torch.equal(candidate['cpu_rng'], baseline['cpu_rng'])
    if device == 'cuda': assert torch.equal(candidate['cuda_rng'], baseline['cuda_rng'])
    assert candidate['rotary'].get('forward_launch_returned') == 2
    assert candidate['rotary'].get('backward_launch_returned') == (4 if retain else 2)
    assert telemetry.counts['native_sum_forward_return'] == 1
    assert telemetry.counts['native_sum_backward_return'] == (2 if retain else 1)
    if outer_saved_hooks:
        assert candidate['memory'].get('encoded_operands',0) == 0
        assert candidate['memory'].get('existing_saved_hooks_declined') == 2
    else:
        assert candidate['memory'].get('encoded_operands') == 2
        assert candidate['memory'].get('restores') == (4 if retain else 2)
    check_state(mlp, before)
    memory.uninstall()
    return {'device': device, 'scale': scale, 'outer_saved_hooks': outer_saved_hooks,
            'retain_graph': retain, 'checks': checks, 'memory': candidate['memory'],
            'rotary': candidate['rotary'], 'head': telemetry.snapshot(),
            'rng_equal': True, 'frozen_weights_unchanged': True,
            'scope': 'Combined toy boundary graph, not full Qwen/PEFT/TRL/LoRA trajectory'}
