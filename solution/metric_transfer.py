"""Exact small metric packets for the pinned single-process training path.

TRL 0.29.1 SFTTrainer computes entropy/accuracy in FP32 and obtains Python
scalars only afterward.  Soup 0.77.0 Rewind similarly computes FP32 row means
and int64 row counts before converting each element separately.  Packing the
already-computed scalar bytes keeps that arithmetic and avoids separate host
waits for each value.  It does not move divisions or reductions to Python.

The helpers use one ordered, blocking CPU copy per packet.  CPU execution is
also supported for dispatcher/numerical checks; it is not GPU timing evidence.
No JIT kernel, warmup, pinned buffer cache, asynchronous read, or RNG is used.
"""

from functools import lru_cache, wraps
from importlib.metadata import PackageNotFoundError, version
import struct
import sys
import weakref

import torch


_STREAMED_MARKER = object()
_TOKEN_LOSS_MARKER = "_soup_candidate_metric_token_loss"


@lru_cache(maxsize=None)
def _package_versions_match(requirements):
    try:
        return all(version(name) == expected for name, expected in requirements)
    except PackageNotFoundError:
        return False


def _torch_runtime_supported():
    return sys.byteorder == "little" and torch.__version__.split("+", 1)[0] == "2.11.0"


def metric_runtime_supported():
    return _torch_runtime_supported() and _package_versions_match(
        (("trl", "0.29.1"), ("accelerate", "1.14.0"))
    )


def _rewind_runtime_supported():
    return _torch_runtime_supported() and _package_versions_match((("soup-cli", "0.77.0"),))


def _ordinary_tensor(value):
    return (
        type(value) is torch.Tensor
        and value.layout == torch.strided
        and value.device.type in ("cpu", "cuda")
        and not value.is_conj() and not value.is_neg()
        and value.names == (None,) * value.ndim
    )


def _copy_packet_to_cpu(segments):
    """Bit-preserving concatenation followed by exactly one explicit copy.

    ``copy=True`` makes the same boundary observable in CPU dispatcher tests.
    On CUDA, the blocking copy waits for the packet's producing stream before
    the CPU buffer is read.  No scalar extraction occurs on a device tensor.
    """
    packet = torch.cat(tuple(value.reshape(-1).view(torch.uint8) for value in segments))
    return packet.to(device="cpu", non_blocking=False, copy=True)


def metric_values_to_python(tokens, entropy, accuracy):
    """Return exact Python (int, float, float) from one 16-byte packet.

    Return None for unsupported gathered metadata, before any transfer, so the
    caller can decode its already-gathered values by the original path.
    """
    values = (tokens, entropy, accuracy)
    if (
        not all(_ordinary_tensor(value) and value.ndim == 0 for value in values)
        or tokens.dtype != torch.int64
        or entropy.dtype != torch.float32 or accuracy.dtype != torch.float32
        or not (tokens.device == entropy.device == accuracy.device)
    ):
        return None
    host = _copy_packet_to_cpu(values)
    # The integer is transferred as int64 bits, never via float32/float64.
    # struct widens the already-rounded FP32 values exactly to Python floats.
    return struct.unpack("<qff", host.numpy())


def _row_inputs_supported(token_loss, labels):
    return (
        _ordinary_tensor(token_loss) and _ordinary_tensor(labels)
        and token_loss.ndim == labels.ndim == 2
        and token_loss.shape == labels.shape
        and token_loss.shape[0] > 0 and token_loss.shape[1] >= 2
        and token_loss.dtype == torch.float32 and labels.dtype == torch.int64
        and token_loss.device == labels.device
        and token_loss.is_contiguous()
    )


def mark_streamed_token_loss(token_loss, labels):
    """Identify only our existing per-token-loss protocol to the Rewind hook.

    A weak identity reference does not retain labels beyond their normal life.
    No tensor values or public HeadResult/Soup interfaces are changed.
    """
    if not _rewind_runtime_supported() or not _row_inputs_supported(token_loss, labels):
        return False
    setattr(token_loss, _TOKEN_LOSS_MARKER, (_STREAMED_MARKER, weakref.ref(labels)))
    return True


def _marked_streamed_pair(token_loss, labels):
    marker = getattr(token_loss, _TOKEN_LOSS_MARKER, None)
    return (
        isinstance(marker, tuple) and len(marker) == 2
        and marker[0] is _STREAMED_MARKER
        and isinstance(marker[1], weakref.ReferenceType) and marker[1]() is labels
    )


def _packed_row_losses(token_loss, labels):
    # These are the same operations and ordering as Soup 0.77.0's
    # row_losses_from_token_losses.  In particular, do not mask ignored rows
    # or replace nonfinite means by zero, even when their token count is zero.
    with torch.no_grad():
        tokens = (labels[:, 1:] != -100).sum(dim=1)
        totals = token_loss.float().sum(dim=1).to(tokens.device)
        means = totals / tokens.clamp(min=1).to(totals.dtype)
        host = _copy_packet_to_cpu((tokens, means))
    rows = labels.shape[0]
    values = struct.unpack(f"<{rows}q{rows}f", host.numpy())
    return list(values[rows:]), list(values[:rows])


def patch_rewind_metric_transfer():
    """Install a guarded wrapper without changing Soup source or interfaces."""
    if not _rewind_runtime_supported():
        return False
    try:
        from soup_cli.trainer import rewind_hf
    except ImportError:
        return False
    original = rewind_hf.row_losses_from_token_losses
    if getattr(original, "_soup_candidate_metric_packet", False):
        return True

    @wraps(original)
    def row_losses(token_loss, labels):
        if (
            not _rewind_runtime_supported()
            or not _marked_streamed_pair(token_loss, labels)
            or not _row_inputs_supported(token_loss, labels)
        ):
            return original(token_loss, labels)
        return _packed_row_losses(token_loss, labels)

    row_losses._soup_candidate_metric_packet = True
    rewind_hf.row_losses_from_token_losses = row_losses
    return True
