"""Small CPU-only fixtures. Passing these does not validate a CUDA numerical path."""
import contextlib
import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from . import materialized_ce as ce
from .adapter import ConstructorInstallation, MaterializedCELoss
from .telemetry import RouteStats


def oracle(logits, labels, denominator, ignore_index=-100, shift_labels=None):
    targets = (F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
               if shift_labels is None else shift_labels)
    loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=ignore_index,
                           reduction="mean" if denominator is None else "sum")
    return loss if denominator is None else loss / denominator


def same_values(a, b):
    finite = torch.isfinite(a) & torch.isfinite(b)
    return (torch.equal(a[finite], b[finite]) and torch.equal(a.isnan(), b.isnan())
            and torch.equal(a.isposinf(), b.isposinf()) and torch.equal(a.isneginf(), b.isneginf())
            and torch.equal(a.signbit()[a == 0], b.signbit()[b == 0]))


class CPUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_fallback_values_gradients_rng_and_no_cuda_queries(self):
        gen = torch.Generator().manual_seed(9293)
        base = torch.randn(4, 3, 7, generator=gen).half()
        labels = torch.tensor([[0, 1, 2], [0, -100, 3], [0, 4, -100], [0, 6, 1]])
        cases = [(base, labels, 7, 65536., {}),
                 (base, labels, torch.tensor(7), float("inf"), {}),
                 (base, torch.full_like(labels, -100), None, 1., {}),
                 (base, labels, 3, 0., {}),
                 (base, labels, 3, 1., {"ignore_index": -1,
                    "shift_labels": torch.tensor([[1,-1,2],[-1,3,-1],[4,-1,-1],[6,1,-1]])})]
        bad = base.clone(); bad[0, -1, 0] = float("nan")
        cases.append((bad, labels, 7, 0., {}))
        for data, lab, denominator, upstream, kwargs in cases:
            with self.subTest(upstream=upstream, kwargs=tuple(kwargs)):
                a, b = data.clone().requires_grad_(), data.clone().requires_grad_()
                rng = torch.random.get_rng_state().clone(); saved = lab.clone()
                stats = ce.Diagnostics()
                with patch.object(torch.cuda, "get_device_capability", side_effect=AssertionError("CUDA query")):
                    loss = ce.materialized_causal_loss(a, lab, denominator, diagnostics=stats, **kwargs)
                expected = oracle(b, lab, denominator, **kwargs)
                ga, = torch.autograd.grad(loss, a, grad_outputs=torch.tensor(upstream))
                gb, = torch.autograd.grad(expected, b, grad_outputs=torch.tensor(upstream))
                self.assertTrue(same_values(loss, expected)); self.assertTrue(same_values(ga, gb))
                self.assertEqual(stats.events, [{"event":"native_forward_fallback","reason":"not_cuda"}])
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                self.assertTrue(torch.equal(saved, lab))

    def test_native_exceptions_and_denominator_version_are_preserved(self):
        z = torch.zeros((4, 2, 7), dtype=torch.float16)
        labels = torch.tensor([[0,7],[0,1],[0,2],[0,3]])
        messages=[]
        for fn in (oracle, ce.materialized_causal_loss):
            with self.assertRaises(IndexError) as error:
                fn(z, labels, 7)
            messages.append(str(error.exception))
        self.assertEqual(*messages)
        labels[0,1] = 6
        for fn in (oracle, ce.materialized_causal_loss):
            x = z.clone().requires_grad_(); denominator = torch.tensor(7)
            loss = fn(x, labels, denominator); denominator.add_(1)
            with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                loss.backward()

    def test_denominator_single_read_and_identity(self):
        class CountingTensor(torch.Tensor):
            reads = 0
            def item(self):
                type(self).reads += 1
                return super().item()
        value = torch.tensor(7, dtype=torch.int64).as_subclass(CountingTensor)
        version = value._version
        self.assertEqual(ce._denominator_metadata(value, torch.device("cpu")), (7,"int64_cpu"))
        self.assertEqual(value.reads, 1)
        self.assertEqual(value._version, version)
        for bad in (True, False, 0, -1, 1.5, None, 2**63,
                    torch.tensor(0), torch.tensor(1.), torch.tensor([1])):
            self.assertIsNone(ce._denominator_metadata(bad, torch.device("cpu")))
        self.assertEqual(ce._denominator_metadata(9, torch.device("cpu")), (9,"python_int"))

    def test_core_delegates_same_objects_to_supplied_native(self):
        z = torch.zeros(4,2,7, requires_grad=True); labels = torch.ones(4,2,dtype=torch.int64)
        marker = object(); seen=[]
        result = ce.materialized_causal_loss(z, labels, 7,
                    native_fallback=lambda reason: seen.append(reason) or marker)
        self.assertIs(result, marker); self.assertEqual(seen, ["not_cuda"])

    def test_adapter_output_controls_and_original_arguments(self):
        owner = SimpleNamespace(training=True)
        # SimpleNamespace is not weak-referenceable; model-shaped fixture is.
        class Owner: training = True
        owner = Owner(); seen=[]; marker=object(); stats=RouteStats()
        wrapper = MaterializedCELoss(owner, lambda **kw: seen.append(kw) or marker, stats)
        logits=torch.zeros((4,2,7),requires_grad=True); labels=torch.ones((4,2),dtype=torch.int64)
        denominator=torch.tensor(3)
        with contextlib.redirect_stdout(io.StringIO()):
            for a, b in ((False, False), (True, None), (None, True)):
                self.assertIs(wrapper(logits, labels, 7, num_items_in_batch=denominator,
                        output_attentions=a, output_hidden_states=b), marker)
            token=object()
            self.assertIs(wrapper(logits, labels, 7, num_items_in_batch=denominator,
                                 unsupported_probe=token), marker)
            self.assertIs(seen[-1]["unsupported_probe"], token)
            self.assertIs(wrapper(logits, labels, 7, num_items_in_batch=denominator,
                                 output_attentions=1), marker)
        for kw in seen:
            self.assertIs(kw["logits"], logits); self.assertIs(kw["labels"], labels)
            self.assertIs(kw["num_items_in_batch"], denominator)
        self.assertEqual(sum(stats.output_controls.values()), 3)
        self.assertEqual(stats.reasons["shape_or_vocabulary"], 3)
        self.assertEqual(stats.events["triton_forward"], 0)
        self.assertEqual(stats.events["loss_boundary"], 5)

    def test_reversible_constructor_fixture_without_forward_patch(self):
        def native(**kwargs): return kwargs
        class Model:
            def __init__(self):
                self.config=SimpleNamespace(model_type="qwen2",vocab_size=ce.VOCABULARY)
                self.training=True
            @property
            def loss_function(self): return self.__dict__.get("_loss_function", native)
            @loss_function.setter
            def loss_function(self, value): self._loss_function=value
        original=Model.__init__; stats=RouteStats()
        with contextlib.redirect_stdout(io.StringIO()):
            handle=ConstructorInstallation(Model,native,Model.loss_function,stats)
            first=Model(); second=Model()
            self.assertIsInstance(first.loss_function, MaterializedCELoss)
            self.assertIs(handle.install_model(first), handle.models[first])
            foreign=lambda **kw: None; second.loss_function=foreign
            restored=handle.restore()
        self.assertIs(Model.__init__,original); self.assertIs(first.loss_function,native)
        self.assertIs(second.loss_function,foreign)
        self.assertEqual(restored["model_losses"], {"restored":1,"foreign_loss_preserved":1})
        self.assertEqual(handle.restore()["constructor"],"already_restored")

    def test_telemetry_bounded_and_explicit(self):
        stats=RouteStats(); stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            for i in range(300): stats.record("adapter_native_fallback", reason=str(i))
            stats.record("loss_boundary", denominator_type="Tensor")
            stats.record("triton_forward", rows=12)
            stats.record("triton_backward", rows=12, phase="enqueued_not_completion")
        self.assertLessEqual(len(stats.reasons),129)
        rows=[json.loads(line[len(stats.PREFIX):]) for line in stream.getvalue().splitlines()]
        self.assertEqual(rows[-1]["own_backward_enqueues"],1)
        self.assertEqual(rows[-1]["metadata"]["phase"],"enqueued_not_completion")
        self.assertFalse(stats.snapshot()["telemetry"]["kernel_completion_or_share_claim"])

    def test_flush_survives_killed_child(self):
        # A killed process cannot execute atexit: this specifically proves the
        # already-emitted route line, not eventual summaries, remains readable.
        code = "from solution.telemetry import RouteStats; import time; RouteStats().record('loss_boundary', test='cpu_kill_fixture'); time.sleep(20)"
        root = str(Path(__file__).resolve().parents[1])
        child = subprocess.Popen([sys.executable,"-c",code],cwd=root,stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,text=True,
                                 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=root))
        try:
            ready,_,_=select.select([child.stdout],[],[],5)
            self.assertTrue(ready,"No immediately flushed route line")
            line=child.stdout.readline()
            child.kill(); child.wait(timeout=5)
            record=json.loads(line[len(RouteStats.PREFIX):])
            self.assertEqual(record["event"],"loss_boundary")
            self.assertLess(child.returncode,0)
        finally:
            if child.poll() is None: child.kill(); child.wait(timeout=5)
            child.stdout.close(); child.stderr.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
