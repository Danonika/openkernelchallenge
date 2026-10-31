"""Exact small metric packets for the pinned single-process training path.

This helper is ported from the user's verified-connection notebook payload.
TRL computes its FP32 metrics before converting them to Python scalars; Soup
likewise computes FP32 row means and int64 row counts before conversion. Pack
those already-computed bytes so each group needs one ordered, blocking host
copy. Divisions, reductions, scalar gathers, and RNG remain unchanged.

CPU execution supports numerical and dispatcher checks only. It is not GPU
timing evidence. No JIT kernel, pinned-buffer cache, or asynchronous read is
introduced here.
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


@lru_cache(maxsize=1)
def _numpy_bridge_supported():
    """Check the decode bridge once, using only an empty CPU tensor.

    A package can be installed while its NumPy ABI bridge is unavailable. Fail
    back to ordinary scalar conversion before allocating a metric packet in
    either case. This capability check never initializes CUDA.
    """
    try:
        torch.empty(0, dtype=torch.uint8, device="cpu").numpy()
    except (ImportError, RuntimeError, AttributeError):
        return False
    return True


def metric_runtime_supported():
    return (
        _torch_runtime_supported()
        and _numpy_bridge_supported()
        and _package_versions_match((("trl", "0.29.1"), ("accelerate", "1.14.0")))
    )


def _rewind_runtime_supported():
    return (
        _torch_runtime_supported()
        and _numpy_bridge_supported()
        and _package_versions_match((("soup-cli", "0.77.0"),))
    )


def _ordinary_tensor(value):
    return (
        type(value) is torch.Tensor
        and value.layout == torch.strided
        and value.device.type in ("cpu", "cuda")
        and not value.is_conj() and not value.is_neg()
        and value.names == (None,) * value.ndim
    )


def _copy_packet_to_cpu(segments):
    """Concatenate raw bytes and perform exactly one explicit blocking copy."""
    packet = torch.cat(tuple(value.reshape(-1).view(torch.uint8) for value in segments))
    # copy=True keeps the same boundary observable in CPU dispatcher tests.
    return packet.to(device="cpu", non_blocking=False, copy=True)


def metric_values_to_python(tokens, entropy, accuracy):
    """Decode exact Python (int, float, float) from one 16-byte packet.

    Return None before any transfer for unsupported metadata or a missing
    NumPy bridge, so the caller can reuse its already-gathered scalar values.
    """
    values = (tokens, entropy, accuracy)
    if (
        not all(_ordinary_tensor(value) and value.ndim == 0 for value in values)
        or tokens.dtype != torch.int64
        or entropy.dtype != torch.float32 or accuracy.dtype != torch.float32
        or not (tokens.device == entropy.device == accuracy.device)
        or not _numpy_bridge_supported()
    ):
        return None
    host = _copy_packet_to_cpu(values)
    # Preserve int64 bits; widen only the already-rounded FP32 values.
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
    """Mark our compact per-token protocol for the Rewind conversion hook.

    The historical name also covers cached heads using the same protocol. The
    weak identity reference does not extend the normal labels lifetime.
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
    # Same operations and order as Soup 0.77.0 row_losses_from_token_losses.
    # Do not mask ignored rows or replace their nonfinite means by zero.
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
