"""Original full-vocabulary entropy reduction for TRL's no-grad metrics.

The two kernels below were written for this solution.  A tile retains a local
maximum, an exponential sum, and a weighted displacement sum.  Merging those
statistics gives Shannon entropy without creating any vocabulary-sized output.
All original metric masks, aggregation, logging, and training losses stay in TRL.
"""

from __future__ import annotations

import ast
import copy
import functools
import importlib
import inspect
import textwrap
import types

import torch
import triton
import triton.language as tl


@triton.jit
def soup_entropy_partials(
    X,
    Partials,
    VOCAB: tl.constexpr,
    ROWS: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    column = tile * BLOCK + tl.arange(0, BLOCK)
    valid = column < VOCAB
    value = tl.load(X + row * VOCAB + column, valid, other=-float("inf"))
    maximum = tl.max(value, axis=0)
    displacement = value - maximum
    mass = tl.where(valid, tl.exp(displacement), 0.0)
    total = tl.sum(mass, axis=0)
    weighted = tl.sum(tl.where(valid, mass * displacement, 0.0), axis=0)

    # Native log_softmax / exp / product entropy is NaN if a real logit is
    # nonfinite, including -inf (the product includes 0 * -inf).  Do not let
    # a reduction's NaN ordering or padded lanes silently alter that behavior.
    bad = tl.max(
        (valid & ((value != value) | (tl.abs(value) == float("inf")))).to(tl.int32),
        axis=0,
    )
    index = row * TILES + tile
    plane = ROWS * TILES
    tl.store(Partials + index, maximum)
    tl.store(Partials + plane + index, total)
    tl.store(Partials + 2 * plane + index, weighted)
    tl.store(Partials + 3 * plane + index, bad.to(tl.float32))


@triton.jit
def soup_entropy_merge(
    Partials,
    Entropy,
    ROWS: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    valid = tile < TILES
    index = row * TILES + tile
    plane = ROWS * TILES
    local_maximum = tl.load(Partials + index, valid, other=-float("inf"))
    local_total = tl.load(Partials + plane + index, valid, other=0.0)
    local_weighted = tl.load(Partials + 2 * plane + index, valid, other=0.0)
    local_bad = tl.load(Partials + 3 * plane + index, valid, other=0.0)

    maximum = tl.max(local_maximum, axis=0)
    displacement = local_maximum - maximum
    rescale = tl.exp(displacement)
    total = tl.sum(tl.where(valid, rescale * local_total, 0.0), axis=0)
    weighted = tl.sum(
        tl.where(valid, rescale * (local_weighted + displacement * local_total), 0.0),
        axis=0,
    )
    # H = log(sum exp(x-M)) - sum(exp(x-M)*(x-M)) / sum exp(x-M).
    entropy = tl.log(total) - weighted / total
    entropy = tl.where(tl.max(local_bad, axis=0) > 0, float("nan"), entropy)
    tl.store(Entropy + row, entropy)


def _can_use_entropy(logits, chunk_size):
    return (
        isinstance(logits, torch.Tensor)
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim >= 1
        and 0 < logits.shape[-1] <= 262144
        and 0 < logits.numel() <= 2147483647
        and logits.is_contiguous()
        and isinstance(chunk_size, int)
        and chunk_size > 0
        # SFT evaluates metrics inside no_grad, but its logits retain this flag.
        # Differentiable callers must continue to receive the native graph.
        and not (torch.is_grad_enabled() and logits.requires_grad)
    )


def _entropy_forward(logits):
    vocab = logits.shape[-1]
    rows = logits.numel() // vocab
    block = 4096
    tiles = triton.cdiv(vocab, block)
    partials = torch.empty((4, rows, tiles), dtype=torch.float32, device=logits.device)
    entropy = torch.empty(logits.shape[:-1], dtype=torch.float32, device=logits.device)
    with torch.cuda.device(logits.device):
        soup_entropy_partials[(rows, tiles)](
            logits, partials, vocab, rows, tiles, block, num_warps=4
        )
        soup_entropy_merge[(rows,)](
            partials, entropy, rows, tiles, triton.next_power_of_2(tiles), num_warps=4
        )
    return entropy


def _make_entropy_wrapper(original):
    @functools.wraps(original)
    def entropy_from_logits(logits, chunk_size=128):
        if _can_use_entropy(logits, chunk_size):
            return _entropy_forward(logits)
        return original(logits, chunk_size=chunk_size)

    entropy_from_logits._soup_entropy_original = original
    return entropy_from_logits


def _accuracy_without_copy(original):
    """Return a guarded one-expression rewrite of the installed trainer method.

    No trainer implementation is vendored.  The exact existing function is
    parsed and only its ``outputs.logits[..., :-1, :].contiguous()`` expression
    loses the final no-argument call.  Native strided argmax, ties, masks,
    virtual-token handling, gathering, logging, and every other branch remain.
    This is optional and deliberately not installed by default.
    """
    if hasattr(original, "_soup_accuracy_original"):
        return original
    if not isinstance(original, types.FunctionType):
        return None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    except (OSError, TypeError, SyntaxError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    method = tree.body[0]
    if method.name != "compute_loss" or method.decorator_list:
        return None
    expected_args = ["self", "model", "inputs", "return_outputs", "num_items_in_batch"]
    if (
        [arg.arg for arg in method.args.args] != expected_args
        or method.args.posonlyargs
        or method.args.kwonlyargs
        or method.args.vararg
        or method.args.kwarg
        or original.__code__.co_freevars != ("__class__",)
        or original.__closure__ is None
    ):
        return None

    target = ast.parse("outputs.logits[..., :-1, :].contiguous()", mode="eval").body
    target_dump = ast.dump(target, include_attributes=False)
    matches = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "shift_logits"
        and ast.dump(node.value, include_attributes=False) == target_dump
    ]
    # Also require the expected native consumer, since the copy must be only
    # an input preparation for a last-dimension reduction.
    consumer = ast.parse("shift_logits.argmax(dim=-1)", mode="eval").body
    consumer_dump = ast.dump(consumer, include_attributes=False)
    consumers = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call) and ast.dump(node, include_attributes=False) == consumer_dump
    ]
    if len(matches) != 1 or len(consumers) != 1:
        return None

    # A round trip must recover the entire original AST exactly; this guards
    # against accidentally modifying a second expression while editing here.
    before = ast.dump(tree, include_attributes=False)
    replaced = matches[0]
    old_value = replaced.value
    replaced.value = old_value.func.value
    after = ast.dump(tree, include_attributes=False)
    replaced.value = old_value
    if ast.dump(tree, include_attributes=False) != before or before == after:
        return None
    replaced.value = old_value.func.value

    # Compiling inside a class supplies a __class__ closure for zero-argument
    # super().  Reuse the *original* closure below, so super() keeps its normal
    # SFTTrainer resolution instead of referring to this temporary class.
    holder = ast.ClassDef(
        name="_SoupAccuracyCompileHolder", bases=[], keywords=[],
        body=[copy.deepcopy(method)], decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[holder], type_ignores=[]))
    namespace = dict(original.__globals__)
    try:
        exec(compile(module, __file__, "exec"), namespace)
        compiled = namespace[holder.name].__dict__[method.name]
    except (TypeError, ValueError, SyntaxError, NameError):
        return None
    if (
        compiled.__code__.co_freevars != original.__code__.co_freevars
        or compiled.__code__.co_argcount != original.__code__.co_argcount
        or compiled.__code__.co_kwonlyargcount != original.__code__.co_kwonlyargcount
        or compiled.__code__.co_varnames != original.__code__.co_varnames
    ):
        return None
    patched = types.FunctionType(
        compiled.__code__, original.__globals__, original.__name__,
        original.__defaults__, original.__closure__,
    )
    functools.update_wrapper(patched, original)
    patched.__kwdefaults__ = original.__kwdefaults__
    patched._soup_accuracy_original = original
    patched._soup_accuracy_ast_before = before
    patched._soup_accuracy_ast_after = after
    return patched


def install(*, accuracy_no_copy=False):
    """Patch both existing entropy bindings; optionally remove the accuracy copy."""
    unavailable = {"entropy": False, "accuracy_no_copy": False}
    try:
        utils = importlib.import_module("trl.trainer.utils")
        sft = importlib.import_module("trl.trainer.sft_trainer")
    except (ImportError, AttributeError):
        return unavailable

    # Only install against the supported library interface.  Minimal or
    # substituted dependencies must remain untouched, rather than gaining a
    # function or class method which they never provided.  These checks occur
    # before any mutation; no kernel invocation is caught or silenced here.
    if not isinstance(utils, types.ModuleType) or not isinstance(sft, types.ModuleType):
        return unavailable
    existing = getattr(utils, "entropy_from_logits", None)
    bound = getattr(sft, "entropy_from_logits", None)
    trainer = getattr(sft, "SFTTrainer", None)
    method = getattr(trainer, "compute_loss", None) if isinstance(trainer, type) else None
    if (
        not isinstance(existing, types.FunctionType)
        or not isinstance(bound, types.FunctionType)
        or not isinstance(method, types.FunctionType)
        or existing.__module__ != "trl.trainer.utils"
        or existing.__name__ != "entropy_from_logits"
        or method.__module__ != "trl.trainer.sft_trainer"
    ):
        return unavailable
    original = getattr(existing, "_soup_entropy_original", existing)
    bound_original = getattr(bound, "_soup_entropy_original", bound)
    if not isinstance(original, types.FunctionType) or bound_original is not original:
        return unavailable
    try:
        parameters = inspect.signature(original).parameters
    except (TypeError, ValueError):
        return unavailable
    if (
        tuple(parameters) != ("logits", "chunk_size")
        or parameters["logits"].kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameters["chunk_size"].kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameters["chunk_size"].default != 128
    ):
        return unavailable

    if hasattr(existing, "_soup_entropy_original"):
        wrapper = existing
    else:
        wrapper = _make_entropy_wrapper(existing)
    utils.entropy_from_logits = wrapper
    # SFT imports this name at module scope; editing utils alone misses it.
    sft.entropy_from_logits = wrapper

    accuracy_installed = hasattr(method, "_soup_accuracy_original")
    if accuracy_no_copy and not accuracy_installed:
        rewritten = _accuracy_without_copy(method)
        if rewritten is not None:
            trainer.compute_loss = rewritten
            accuracy_installed = True
    return {"entropy": True, "accuracy_no_copy": accuracy_installed}
