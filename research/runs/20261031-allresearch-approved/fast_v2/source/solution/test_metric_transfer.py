"""Exact metric-packet checks; CPU dispatch is not GPU timing evidence."""

import gc
import math
import os
import struct
import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from . import metric_transfer as packets


class _TransferTrace(TorchDispatchMode):
    def __init__(self):
        self.scalar_reads = []
        self.host_packets = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = function(*args, **kwargs)
        if function is torch.ops.aten._local_scalar_dense.default:
            self.scalar_reads.append((str(args[0].dtype), str(args[0].device)))
        elif function is torch.ops.aten._to_copy.default:
            destination = kwargs.get("device")
            if destination is not None and torch.device(destination).type == "cpu":
                self.host_packets.append(args[0].numel() * args[0].element_size())
        return result


def _assert_python_exact(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, float):
        if math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert struct.pack("<d", actual) == struct.pack("<d", expected)
    else:
        assert actual == expected


def _assert_rows_exact(actual, expected):
    assert len(actual) == len(expected) == 2
    for values, references in zip(actual, expected):
        assert len(values) == len(references)
        for value, reference in zip(values, references):
            _assert_python_exact(value, reference)


def _row_case(device="cpu", all_ignored=False):
    labels = torch.tensor([
        [-100, 2, 3, 4, -100], [-100, 1, -100, -100, -100],
        [-100, -100, -100, -100, -100], [-100, 9, 8, -100, -100],
    ], device=device)
    if all_ignored:
        labels.fill_(-100)
    losses = torch.tensor([
        [1.1, 2.2, 3.3, 0., 0.], [2.3, 0., 0., 0., 0.],
        [float("nan"), 0., 0., 0., 0.], [float("inf"), 4., 0., 0., 0.],
    ], device=device, dtype=torch.float32)
    return losses, labels


@pytest.fixture
def rewind_original(monkeypatch):
    from soup_cli.trainer import rewind_hf

    original = rewind_hf.row_losses_from_token_losses
    while getattr(original, "_soup_candidate_metric_packet", False):
        original = original.__wrapped__
    monkeypatch.setattr(rewind_hf, "row_losses_from_token_losses", original)
    return rewind_hf, original


def test_metric_runtime_requires_the_real_pinned_stack():
    # A disabled helper must not silently satisfy the active-path tests.
    assert packets.metric_runtime_supported()
    assert packets._rewind_runtime_supported()


def test_scalar_packet_preserves_large_int64_signed_zero_and_fp32():
    count = torch.tensor(2**53 + 1, dtype=torch.int64)
    entropy = torch.tensor(-0.0, dtype=torch.float32)
    accuracy = torch.tensor(2) / torch.tensor(3)
    expected = (count.item(), entropy.item(), accuracy.item())
    before = torch.get_rng_state().clone()
    trace = _TransferTrace()
    with trace:
        observed = packets.metric_values_to_python(count, entropy, accuracy)
    for value, reference in zip(observed, expected):
        _assert_python_exact(value, reference)
    assert observed[0] == 2**53 + 1 and int(float(2**53 + 1)) != observed[0]
    assert math.copysign(1.0, observed[1]) == -1.0
    assert observed[2] == 0.6666666865348816 and observed[2] != 2 / 3
    assert trace.scalar_reads == [] and trace.host_packets == [16]
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("entropy,accuracy", [
    (float("nan"), 0.0), (float("inf"), float("nan")),
    (-float("inf"), -0.0), (0.0, 0.0),
])
def test_scalar_packet_preserves_nonfinite_and_zero_values(entropy, accuracy):
    tensors = (torch.tensor(0, dtype=torch.int64),
               torch.tensor(entropy, dtype=torch.float32),
               torch.tensor(accuracy, dtype=torch.float32))
    for value, reference in zip(packets.metric_values_to_python(*tensors),
                                (tensor.item() for tensor in tensors)):
        _assert_python_exact(value, reference)


@pytest.mark.parametrize("unsupported", ["integer_dtype", "float_dtype", "rank", "subclass", "bridge"])
def test_scalar_packet_rejects_unsupported_metadata_before_transfer(monkeypatch, unsupported):
    values = [torch.tensor(7, dtype=torch.int64), torch.tensor(1.5), torch.tensor(0.25)]
    if unsupported == "integer_dtype":
        values[0] = values[0].to(torch.int32)
    elif unsupported == "float_dtype":
        values[1] = values[1].double()
    elif unsupported == "rank":
        values[2] = values[2].reshape(1)
    elif unsupported == "subclass":
        class OtherTensor(torch.Tensor):
            pass
        values[0] = values[0].as_subclass(OtherTensor)
    else:
        monkeypatch.setattr(packets, "_numpy_bridge_supported", lambda: False)
    trace = _TransferTrace()
    with trace:
        assert packets.metric_values_to_python(*values) is None
    assert trace.host_packets == [] and trace.scalar_reads == []


@pytest.mark.parametrize("unsupported", ["torch_version", "package_versions", "bridge"])
def test_runtime_guard_preserves_unpatched_rewind(monkeypatch, rewind_original, unsupported):
    module, original = rewind_original
    if unsupported == "torch_version":
        monkeypatch.setattr(torch, "__version__", "2.11.1")
    elif unsupported == "package_versions":
        monkeypatch.setattr(packets, "_package_versions_match", lambda requirements: False)
    else:
        monkeypatch.setattr(packets, "_numpy_bridge_supported", lambda: False)
    assert not packets.metric_runtime_supported()
    assert not packets.patch_rewind_metric_transfer()
    losses, labels = _row_case()
    assert not packets.mark_streamed_token_loss(losses, labels)
    assert module.row_losses_from_token_losses is original


def test_scalar_gathers_precede_packing_at_dataloader_remainder():
    from accelerate import Accelerator
    from types import SimpleNamespace

    class RemainderAccelerator:
        gradient_state = SimpleNamespace(end_of_dataloader=True, remainder=1)

        def __init__(self):
            self.calls = []

        @staticmethod
        def gather(value):
            return value

        def gather_for_metrics(self, value):
            self.calls.append((value.shape, value.dtype))
            return Accelerator.gather_for_metrics(self, value)

    accelerator = RemainderAccelerator()
    entropy = accelerator.gather_for_metrics(torch.tensor(1.234567)).mean()
    tokens = accelerator.gather_for_metrics(torch.tensor(19, dtype=torch.int64)).sum()
    correct = accelerator.gather_for_metrics(torch.tensor(2, dtype=torch.int64))
    count = accelerator.gather_for_metrics(torch.tensor(3, dtype=torch.int64)).sum()
    accuracy = correct.sum() / count
    observed = packets.metric_values_to_python(tokens, entropy, accuracy)
    expected = (tokens.item(), entropy.item(), accuracy.item())
    for value, reference in zip(observed, expected):
        _assert_python_exact(value, reference)
    assert accelerator.calls == [
        (torch.Size([]), torch.float32), (torch.Size([]), torch.int64),
        (torch.Size([]), torch.int64), (torch.Size([]), torch.int64),
    ]
    # A premature packed vector would be truncated by the real implementation.
    assert Accelerator.gather_for_metrics(accelerator, torch.arange(3)).shape == (1,)


@pytest.mark.parametrize("all_ignored", [False, True])
def test_rewind_packet_preserves_real_soup_rows_and_identity(rewind_original, all_ignored):
    module, original = rewind_original
    assert packets.patch_rewind_metric_transfer()
    installed = module.row_losses_from_token_losses
    assert packets.patch_rewind_metric_transfer() and module.row_losses_from_token_losses is installed
    losses, labels = _row_case(all_ignored=all_ignored)
    expected = original(losses, labels)
    trace = _TransferTrace()
    with trace:
        _assert_rows_exact(installed(losses, labels), expected)
    assert len(trace.scalar_reads) == 8 and trace.host_packets == []
    assert packets.mark_streamed_token_loss(losses, labels)
    trace = _TransferTrace()
    before = torch.get_rng_state().clone()
    with trace:
        actual = installed(losses, labels)
    _assert_rows_exact(actual, expected)
    assert trace.scalar_reads == [] and trace.host_packets == [48]
    assert torch.equal(before, torch.get_rng_state())
    assert actual[1][2] == 0 and math.isnan(actual[0][2])
    copied_labels = labels.clone()
    trace = _TransferTrace()
    with trace:
        _assert_rows_exact(installed(losses, copied_labels), expected)
    assert len(trace.scalar_reads) == 8 and trace.host_packets == []


def test_rewind_empty_noncontiguous_and_error_paths_delegate(rewind_original):
    module, original = rewind_original
    assert packets.patch_rewind_metric_transfer()
    empty, labels = torch.empty(0, 5), torch.empty(0, 5, dtype=torch.int64)
    assert not packets.mark_streamed_token_loss(empty, labels)
    assert module.row_losses_from_token_losses(empty, labels) == original(empty, labels) == ([], [])
    noncontiguous = torch.arange(15, dtype=torch.float32).reshape(5, 3).t()
    labels = torch.full((3, 5), -100)
    assert not packets.mark_streamed_token_loss(noncontiguous, labels)
    _assert_rows_exact(module.row_losses_from_token_losses(noncontiguous, labels), original(noncontiguous, labels))
    losses = torch.zeros(3, 5)
    assert packets.mark_streamed_token_loss(losses, labels)
    labels.resize_(4, 5).fill_(-100)
    with pytest.raises(RuntimeError) as expected:
        original(losses, labels)
    with pytest.raises(RuntimeError) as actual:
        module.row_losses_from_token_losses(losses, labels)
    assert str(actual.value) == str(expected.value)


def test_token_loss_marker_does_not_retain_labels():
    losses, labels = torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.int64)
    reference = weakref.ref(labels)
    assert packets.mark_streamed_token_loss(losses, labels)
    del labels
    gc.collect()
    assert reference() is None


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1",
                    reason="requires physical CUDA packet transfers, not CPU/interpreter checks")
@pytest.mark.parametrize("all_ignored", [False, True])
def test_physical_cuda_metric_and_rewind_packets(rewind_original, all_ignored):
    assert packets.metric_runtime_supported() and packets._rewind_runtime_supported()
    count = torch.tensor(2**53 + 1, dtype=torch.int64, device="cuda")
    entropy = torch.tensor(-0.0, dtype=torch.float32, device="cuda")
    accuracy = torch.tensor(2, device="cuda") / torch.tensor(3, device="cuda")
    expected = (count.item(), entropy.item(), accuracy.item())
    trace = _TransferTrace()
    with trace:
        actual = packets.metric_values_to_python(count, entropy, accuracy)
    for value, reference in zip(actual, expected):
        _assert_python_exact(value, reference)
    assert trace.scalar_reads == [] and trace.host_packets == [16]
    module, original = rewind_original
    losses, labels = _row_case(device="cuda", all_ignored=all_ignored)
    expected = original(losses, labels)
    assert packets.patch_rewind_metric_transfer()
    assert packets.mark_streamed_token_loss(losses, labels)
    trace = _TransferTrace()
    with trace:
        actual = module.row_losses_from_token_losses(losses, labels)
    _assert_rows_exact(actual, expected)
    assert trace.scalar_reads == [] and trace.host_packets == [48]
