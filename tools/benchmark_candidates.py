"""Screen Soup head/NF4/pointwise candidates on a real GPU, outside the official harness.

Examples (run from the starter repository):
  python tools/benchmark_candidates.py --suite head --output validation/head_t4.json
  python tools/benchmark_candidates.py --suite projection
  python tools/benchmark_candidates.py --suite nf4
  python tools/benchmark_candidates.py --suite fused
  python tools/benchmark_candidates.py --suite pointwise
  python tools/benchmark_candidates.py --suite head --chunk-sizes 128,256,512
  python tools/benchmark_candidates.py --suite head --head-projections torch,t4_mma
  python tools/benchmark_candidates.py --suite head --head-projections torch,t4_fused
  python tools/benchmark_candidates.py --list-cases

Each case runs in a fresh process with empty Triton/Inductor caches. First-call
wall time includes that operation's compilation and, for a streamed head, its
FP16 weight-cache creation. It EXCLUDES imports, CUDA context setup, input
generation and quantization. Warm CUDA-event intervals and synchronized wall
intervals are recorded separately. CUDA events include idle gaps between
launches; they are not a sum of kernel durations. Neither timing is an official
whole-training result. Allocator peaks are not nvidia-smi device-memory peaks.

The head reference uses a full FP16 autocast linear, FP32 cross-entropy, and
TRL's actual chunked entropy helper. All positions receive entropy/argmax;
causal shifted labels alone determine CE and hidden gradients. The benchmark
has synthetic data, no decoder, no optimizer, no adapter and no Rewind recorder.
The pointwise suite isolates native and fused RoPE and LoRA scale/add forward
and backward. Its prepared upstream gradients already include --loss-scale;
it does not include LoRA projections, dropout or an optimizer. These synthetic
timing/memory ratios cannot establish contest speedup, quality or admission.

This file is never imported by solution.apply(), changes no production settings,
and never changes the official harness/task. Experimental choices stay explicit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
import traceback


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _csv(kind):
    def parse(value):
        try:
            result = [kind(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
        if not result:
            raise argparse.ArgumentTypeError("at least one comma-separated value is required")
        return list(dict.fromkeys(result))
    return parse


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=("screen", "head", "projection", "nf4", "fused", "pointwise", "all"), default="screen",
                        help="screen = head + projection; pointwise = RoPE + LoRA epilogues; all includes every suite")
    parser.add_argument("--output", type=Path, default=Path("validation/t4_candidates.json"))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lengths", type=_csv(int), default=[128, 512])
    parser.add_argument("--fractions", type=_csv(float), default=[0.25, 0.75],
                        help="fraction of each row's T-1 causal targets supervised")
    parser.add_argument("--chunk-sizes", type=_csv(int), default=[256])
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--projection-rows", type=_csv(int), default=[32, 256])
    parser.add_argument("--fused-rows", type=_csv(int), default=[256, 2048],
                        help="fused forward screening includes the full 4x512-token batch")
    parser.add_argument("--head-projections", type=_csv(str), default=["torch"],
                        help="torch,t4_mma,t4_fused; experimental paths stay explicit and production settings are unchanged")
    parser.add_argument("--loss-scale", type=float, default=65536.0,
                        help="head GradScaler scale / pointwise upstream-gradient scale, held fixed; no optimizer step")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=600, help="seconds per isolated case")
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--allow-other-gpu", action="store_true", help="allow architecture other than SM75; label it explicitly")
    parser.add_argument("--nf4-reuse-output", action="store_true", help="also screen preallocated dequantization outputs")
    parser.add_argument("--list-cases", action="store_true", help="print the exact experiment plan without importing CUDA")
    parser.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--probe-output", type=Path, help=argparse.SUPPRESS)
    return parser


def _cases(args):
    common = {"seed": args.seed, "warmup": args.warmup, "iterations": args.iterations,
              "loss_scale": args.loss_scale, "allow_other_gpu": args.allow_other_gpu}
    cases = []
    if args.suite in ("screen", "projection", "all"):
        for rows in args.projection_rows:
            for variant in ("torch", "t4_mma"):
                cases.append({**common, "kind": "projection", "variant": variant,
                              "rows": rows, "width": args.width, "vocab": args.vocab})
    if args.suite in ("screen", "head", "all"):
        for length in args.lengths:
            for fraction in args.fractions:
                shape = {**common, "kind": "head", "batch": args.batch, "length": length,
                         "fraction": fraction, "width": args.width, "vocab": args.vocab}
                cases.append({**shape, "variant": "dense_reference", "chunk_size": None, "sparse": False})
                for chunk in args.chunk_sizes:
                    for projection in args.head_projections:
                        for sparse in (False, True):
                            cases.append({**shape, "variant": projection, "chunk_size": chunk, "sparse": sparse})
    if args.suite in ("nf4", "all"):
        for rows, width in ((1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)):
            for reuse in ([False, True] if args.nf4_reuse_output else [False]):
                for variant in ("bitsandbytes", "original_triton"):
                    cases.append({**common, "kind": "nf4", "variant": variant,
                                  "rows": rows, "width": width, "reuse_output": reuse,
                                  "compress_statistics": True, "blocksize": 64})
    if args.suite in ("fused", "all"):
        for rows in args.fused_rows:
            for variant in ("torch_projection_triton_statistics", "fused_t4"):
                cases.append({**common, "kind": "fused", "variant": variant,
                              "rows": rows, "width": args.width, "vocab": args.vocab})
    if args.suite in ("pointwise", "all"):
        for length in args.lengths:
            for variant in ("native", "original_triton"):
                cases.append({**common, "kind": "rotary", "variant": variant,
                              "batch": args.batch, "length": length, "q_heads": 12,
                              "kv_heads": 2, "head_dim": 128, "dtype": "float32",
                              "input_layout": "BTHD allocation viewed as BHTD"})
                for projection, outputs in (("q", 1536), ("v", 256)):
                    cases.append({**common, "kind": "lora_epilogue", "variant": variant,
                                  "batch": args.batch, "length": length, "projection": projection,
                                  "output_features": outputs, "input_dtype": "float16",
                                  "output_dtype": "float32", "adapter_scale": 2.0})
    # Deterministic randomized order limits systematic warm-card/order bias. This
    # still is a screening experiment, not the official base/solution ordering.
    random.Random(args.seed).shuffle(cases)
    for index, case in enumerate(cases):
        case["case_id"] = f"{index:03d}_{case['kind']}_{case['variant']}"
    return cases


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _versions():
    result = {"python": sys.version.split()[0]}
    for name in ("torch", "triton", "transformers", "trl", "peft", "accelerate", "bitsandbytes", "soup-cli"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _hardware(torch):
    result = {"cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda}
    if not result["cuda_available"]:
        return result
    properties = torch.cuda.get_device_properties(0)
    result.update(name=properties.name, capability=list(torch.cuda.get_device_capability(0)),
                  total_memory_bytes=properties.total_memory, visible_device_count=torch.cuda.device_count(),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                  allocator_backend=torch.cuda.memory.get_allocator_backend(),
                  allocator_config=os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")))
    result["is_t4_sm75"] = "T4" in properties.name and result["capability"] == [7, 5]
    return result


def _nvidia_snapshot():
    try:
        command = ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem",
                   "--format=csv,noheader,nounits"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return {"fields": command[1].split("=", 1)[1].split(","), "rows": result.stdout.strip().splitlines(),
                "exit_code": result.returncode, "error": result.stderr.strip() or None}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


def _probe_hardware():
    # Inspect CUDA in a short-lived child: a parent CUDA context would otherwise
    # stay resident alongside every benchmark worker and distort available VRAM.
    with tempfile.TemporaryDirectory(prefix="soup_gpu_probe_") as directory:
        output = Path(directory) / "hardware.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--probe-output", str(output)]
        try:
            process = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=60, check=False)
            if process.returncode == 0 and output.exists():
                return json.loads(output.read_text())
            return {"cuda_available": False, "preflight_error": (process.stdout + process.stderr)[-4000:]}
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"cuda_available": False, "preflight_error": str(error)}


def _memory(torch):
    return {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def _summary(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "samples": values, "unit": "ms"}


def _comparisons(results):
    """Compare only matching synthetic cases with successful measurements."""
    def key(case):
        names = {"projection": ("rows", "width", "vocab"),
                 "head": ("batch", "length", "fraction", "width", "vocab", "loss_scale"),
                 "nf4": ("rows", "width", "reuse_output", "compress_statistics", "blocksize"),
                 "fused": ("rows", "width", "vocab"),
                 "rotary": ("batch", "length", "q_heads", "kv_heads", "head_dim", "loss_scale"),
                 "lora_epilogue": ("batch", "length", "projection", "output_features", "loss_scale")}[case["kind"]]
        return (case["kind"], *(case[name] for name in names))

    baseline = {"projection": "torch", "head": "dense_reference", "nf4": "bitsandbytes",
                "fused": "torch_projection_triton_statistics", "rotary": "native", "lora_epilogue": "native"}
    completed = [item for item in results if item["status"] == "measured_numerically_close"]
    references = {key(item["case"]): item for item in completed
                  if item["case"]["variant"] == baseline[item["case"]["kind"]]}
    comparisons = []
    for item in completed:
        case = item["case"]
        reference = references.get(key(case))
        if reference is None or reference is item:
            continue
        elapsed = item["warm_timing"]["synchronized_wall"]["median"]
        peak = item["warm_memory_after"]["peak_allocated_bytes"]
        comparisons.append({"case_id": case["case_id"], "reference_case_id": reference["case"]["case_id"],
                            "isolated_operation_wall_ratio_reference_over_candidate":
                                reference["warm_timing"]["synchronized_wall"]["median"] / elapsed,
                            "allocator_peak_ratio_reference_over_candidate":
                                reference["warm_memory_after"]["peak_allocated_bytes"] / peak,
                            "official_contest_result": False})
    return comparisons


def _error(actual, expected, *, rtol, atol):
    import torch
    actual, expected = actual.float(), expected.float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    difference = actual - expected
    denominator = max(float(torch.linalg.vector_norm(expected.flatten())), 1e-30)
    return {"finite": finite, "max_abs_error": float(difference.abs().max()),
            "mean_abs_error": float(difference.abs().mean()),
            "relative_l2_error": float(torch.linalg.vector_norm(difference.flatten())) / denominator,
            "allclose": finite and torch.allclose(actual, expected, rtol=rtol, atol=atol),
            "rtol": rtol, "atol": atol}


def _make_head(case, torch):
    from solution.streamed_head import HeadResult, streamed_linear_ce
    from trl.trainer.utils import entropy_from_logits
    import torch.nn.functional as F

    batch, length, width, vocab = (case[key] for key in ("batch", "length", "width", "vocab"))
    # Keep the original FP32 frozen head resident for both paths, as k-bit
    # preparation does. Only the streamed path retains its additional FP16 copy.
    weight = torch.randn(vocab, width, device="cuda", dtype=torch.float32).div_(math.sqrt(width))
    hidden = torch.randn(batch, length, width, device="cuda", dtype=torch.float32, requires_grad=True)
    labels = torch.randint(vocab, (batch, length), device="cuda")
    count = max(1, round((length - 1) * case["fraction"]))
    labels[:, :length - count] = -100
    targets = F.pad(labels, (0, 1), value=-100)[:, 1:]
    denominator = (targets != -100).sum()
    scaler = torch.amp.GradScaler("cuda", init_scale=case["loss_scale"], growth_interval=2**30)
    cached_half = None

    def reference_forward(value):
        with torch.autocast("cuda", dtype=torch.float16):
            logits_half = F.linear(value, weight)
        logits = logits_half.float()
        token_loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="none").reshape_as(labels)
        loss = token_loss.sum() / denominator
        with torch.no_grad():
            entropy = entropy_from_logits(logits)
            predictions = logits.argmax(-1)
        return HeadResult(loss, token_loss.detach(), entropy, predictions)

    def operation():
        nonlocal cached_half
        hidden.grad = None
        if case["variant"] == "dense_reference":
            result = reference_forward(hidden)
        else:
            if cached_half is None:
                cached_half = weight.detach().half().contiguous()
            with torch.autocast("cuda", dtype=torch.float16):
                result = streamed_linear_ce(hidden, cached_half, labels, denominator,
                                            chunk_size=case["chunk_size"], projection=case["variant"],
                                            sparse_backward=case["sparse"])
        scaler.scale(result.loss).backward()
        return result

    def snapshot(result, value=hidden):
        return {"loss": result.loss.detach().cpu(), "token_loss": result.token_loss.detach().cpu(),
                "entropy": result.entropy.detach().cpu(), "predictions": result.predictions.detach().cpu(),
                "gradient_unscaled": value.grad.detach().float().cpu() / case["loss_scale"]}

    def reference():
        value = hidden.detach().clone().requires_grad_()
        result = reference_forward(value)
        reference_scaler = torch.amp.GradScaler("cuda", init_scale=case["loss_scale"], growth_interval=2**30)
        reference_scaler.scale(result.loss).backward()
        return snapshot(result, value)

    def validate(actual, expected):
        checks = {key: _error(actual[key], expected[key], rtol=3e-4, atol=3e-4)
                  for key in ("loss", "token_loss", "entropy")}
        checks["gradient_unscaled"] = _error(actual["gradient_unscaled"], expected["gradient_unscaled"],
                                               rtol=0.02, atol=1.5e-6)
        target_cpu = targets.cpu()
        mask = target_cpu != -100
        correct = int(((actual["predictions"] == target_cpu) & mask).sum())
        reference_correct = int(((expected["predictions"] == target_cpu) & mask).sum())
        supervised = int(mask.sum())
        zero_gradients = bool((actual["gradient_unscaled"][~mask] == 0).all())
        return {"floating_point_checks": checks,
                "numerically_close": all(item["allclose"] for item in checks.values()) and zero_gradients,
                "ignored_output_gradients_exact_zero": zero_gradients,
                "prediction_agreement_fraction": float((actual["predictions"] == expected["predictions"]).float().mean()),
                "supervised_tokens": supervised, "total_positions": batch * length,
                "realized_supervised_fraction": supervised / (batch * length),
                "supervised_correct": correct, "reference_supervised_correct": reference_correct,
                "token_accuracy": correct / supervised, "reference_token_accuracy": reference_correct / supervised,
                "note": "Synthetic labels often have zero accuracy. Prediction agreement is reported separately; official metric tolerance and adapter quality remain untested."}

    def cleanup():
        hidden.grad = None

    return operation, snapshot, reference, validate, cleanup


def _make_projection(case, torch):
    from solution.head_matmul_t4 import project_fp16
    x = torch.randn(case["rows"], case["width"], device="cuda", dtype=torch.float16)
    weight = torch.randn(case["vocab"], case["width"], device="cuda", dtype=torch.float16).div_(math.sqrt(case["width"]))

    def operation():
        return torch.mm(x, weight.T) if case["variant"] == "torch" else project_fp16(x, weight)

    def validate(actual, expected):
        error = _error(actual, expected, rtol=3e-3, atol=3e-3)
        return {"projection": error, "numerically_close": error["allclose"]}

    return operation, lambda result: result.cpu(), lambda: torch.mm(x, weight.T).cpu(), validate, lambda: None


def _make_nf4(case, torch):
    import bitsandbytes.functional as F
    from solution.nf4 import dequantize_nf4
    values = torch.randn(case["rows"], case["width"], device="cuda", dtype=torch.float16)
    packed, state = F.quantize_4bit(values, blocksize=case["blocksize"], compress_statistics=True, quant_type="nf4")
    del values
    output = torch.empty(state.shape, device="cuda", dtype=state.dtype) if case["reuse_output"] else None

    def operation():
        if case["variant"] == "bitsandbytes":
            return F.dequantize_4bit(packed, state, out=output)
        return dequantize_nf4(packed, state, out=output)

    def validate(actual, expected):
        error = _error(actual, expected, rtol=0.0, atol=0.0)
        return {"dequantization": error, "bit_exact": torch.equal(actual, expected),
                "numerically_close": error["allclose"],
                "note": "This screening gate requires identical dequantized FP16 values."}

    return operation, lambda result: result.cpu(), lambda: F.dequantize_4bit(packed, state).cpu(), validate, lambda: None


def _make_fused(case, torch):
    from solution.streamed_head import _summarize_logits
    from solution.head_fused_t4 import fused_head_statistics_fp16
    from trl.trainer.utils import entropy_from_logits
    import torch.nn.functional as F

    x = torch.randn(case["rows"], case["width"], device="cuda", dtype=torch.float16)
    weight = torch.randn(case["vocab"], case["width"], device="cuda", dtype=torch.float16).div_(math.sqrt(case["width"]))
    targets = torch.randint(case["vocab"], (case["rows"],), device="cuda")
    targets[:max(1, case["rows"] // 4)] = -100

    def operation():
        with torch.no_grad():
            if case["variant"] == "fused_t4":
                return fused_head_statistics_fp16(x, weight, targets)
            scores = torch.mm(x, weight.T)
            ce = torch.empty(case["rows"], device="cuda", dtype=torch.float32)
            entropy, lse = torch.empty_like(ce), torch.empty_like(ce)
            predictions = torch.empty(case["rows"], device="cuda", dtype=torch.long)
            _summarize_logits(scores, targets, ce, entropy, predictions, lse)
            return ce, entropy, predictions, lse

    def snapshot(result):
        return {key: value.cpu() for key, value in zip(("token_loss", "entropy", "predictions", "logsumexp"), result)}

    def reference():
        with torch.no_grad():
            logits = torch.mm(x, weight.T).float()
            return snapshot((F.cross_entropy(logits, targets, reduction="none"), entropy_from_logits(logits),
                             logits.argmax(-1), torch.logsumexp(logits, -1)))

    def validate(actual, expected):
        checks = {key: _error(actual[key], expected[key], rtol=3e-4, atol=3e-4)
                  for key in ("token_loss", "entropy", "logsumexp")}
        return {"floating_point_checks": checks, "numerically_close": all(item["allclose"] for item in checks.values()),
                "prediction_agreement_fraction": float((actual["predictions"] == expected["predictions"]).float().mean()),
                "note": "Forward-only statistics; this does not benchmark a fused training backward."}

    return operation, snapshot, reference, validate, lambda: None


def _validate_pointwise(actual, expected, fields, note):
    """These pointwise rewrites require exact outputs and exact scaled grads."""
    import torch

    checks = {name: _error(actual[name], expected[name], rtol=0.0, atol=0.0) for name in fields}
    bit_exact = {
        name: (actual[name].dtype == expected[name].dtype
               and actual[name].shape == expected[name].shape
               and torch.equal(actual[name].contiguous().view(torch.uint8),
                               expected[name].contiguous().view(torch.uint8)))
        for name in fields
    }
    same_strides = actual["output_strides"] == expected["output_strides"]
    # TensorIterator may normalize an irrelevant singleton-dimension stride.
    # All offsets and every non-singleton stride still have to match.
    equivalent_layout = (actual["output_shapes"] == expected["output_shapes"] and all(
        all(size == 1 or left == right for size, left, right in zip(shape, actual_stride, expected_stride))
        for shape, actual_stride, expected_stride in zip(
            actual["output_shapes"], actual["output_strides"], expected["output_strides"])
    ))
    return {"floating_point_checks": checks, "bit_exact": bit_exact,
            "output_strides_match": same_strides, "output_layout_equivalent": equivalent_layout,
            "output_strides": actual["output_strides"], "reference_output_strides": expected["output_strides"],
            "numerically_close": all(item["allclose"] for item in checks.values())
                                   and all(bit_exact.values()) and equivalent_layout,
            "note": note}


def _make_rotary(case, torch):
    # Capture the actual pinned native equation; no production patch is applied
    # by this benchmark worker or by importing the optional kernel module.
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    from solution.rotary import fused_apply_rotary_pos_emb

    batch, length, dimension = (case[name] for name in ("batch", "length", "head_dim"))
    query = torch.randn(batch, length, case["q_heads"], dimension, device="cuda").transpose(1, 2).requires_grad_()
    key = torch.randn(batch, length, case["kv_heads"], dimension, device="cuda").transpose(1, 2).requires_grad_()
    angles = torch.rand(batch, length, dimension // 2, device="cuda") * (2 * math.pi)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).contiguous()
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).contiguous()
    # Generate and scale once outside timing, exactly as an upstream scaled
    # gradient arrives at this operation during AMP-managed backward.
    query_grad = (torch.randn_like(query) * 0.03141593).mul_(case["loss_scale"])
    key_grad = (torch.randn_like(key) * 0.03141593).mul_(case["loss_scale"])

    def operation():
        query.grad = key.grad = None
        function = apply_rotary_pos_emb if case["variant"] == "native" else fused_apply_rotary_pos_emb
        result = function(query, key, cos, sin)
        torch.autograd.backward(result, (query_grad, key_grad))
        return result

    def snapshot(result, query_value=query, key_value=key):
        return {"query_output": result[0].detach().cpu(), "key_output": result[1].detach().cpu(),
                "query_gradient_scaled": query_value.grad.detach().cpu(),
                "key_gradient_scaled": key_value.grad.detach().cpu(),
                "output_strides": [list(value.stride()) for value in result],
                "output_shapes": [list(value.shape) for value in result]}

    def reference():
        query_value = query.detach().clone().requires_grad_()
        key_value = key.detach().clone().requires_grad_()
        result = apply_rotary_pos_emb(query_value, key_value, cos, sin)
        torch.autograd.backward(result, (query_grad, key_grad))
        return snapshot(result, query_value, key_value)

    def validate(actual, expected):
        return _validate_pointwise(
            actual, expected, ("query_output", "key_output", "query_gradient_scaled", "key_gradient_scaled"),
            "FP32 Q/K rotation and backward only, using the native Transformers equation and pre-scaled "
            "upstream gradients. Exact values, signed-zero bits and non-singleton output strides are required. "
            "Attention projection, SDPA and full training remain outside this measurement.",
        )

    def cleanup():
        query.grad = key.grad = None

    return operation, snapshot, reference, validate, cleanup


def _make_lora_epilogue(case, torch):
    from solution.lora_epilogue import lora_epilogue

    shape = (case["batch"], case["length"], case["output_features"])
    base = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    adapter = torch.randn_like(base, requires_grad=True)
    upstream = (torch.randn(shape, device="cuda", dtype=torch.float32) * 0.03141593).mul_(case["loss_scale"])

    def native(base_value, adapter_value):
        # The native bnb output upcast, PEFT defensive clone, half-precision
        # scale, and mixed-dtype addition are all inside this timed boundary.
        return base_value.float().clone() + adapter_value * case["adapter_scale"]

    def operation():
        base.grad = adapter.grad = None
        function = native if case["variant"] == "native" else lora_epilogue
        result = function(base, adapter)
        result.backward(upstream)
        return result

    def snapshot(result, base_value=base, adapter_value=adapter):
        return {"output": result.detach().cpu(), "base_gradient_scaled": base_value.grad.detach().cpu(),
                "adapter_gradient_scaled": adapter_value.grad.detach().cpu(),
                "output_strides": [list(result.stride())], "output_shapes": [list(result.shape)]}

    def reference():
        base_value = base.detach().clone().requires_grad_()
        adapter_value = adapter.detach().clone().requires_grad_()
        result = native(base_value, adapter_value)
        result.backward(upstream)
        return snapshot(result, base_value, adapter_value)

    def validate(actual, expected):
        return _validate_pointwise(
            actual, expected, ("output", "base_gradient_scaled", "adapter_gradient_scaled"),
            "LoRA scale/add epilogue and backward only. Inputs are the already computed FP16 base and "
            "adapter projections; the native output cast and clone remain inside timing. Exact output "
            "and both scaled gradients, including signed-zero bits, are required. Quantized GEMM, "
            "dropout, adapter GEMMs and full training remain outside this measurement.",
        )

    def cleanup():
        base.grad = adapter.grad = None

    return operation, snapshot, reference, validate, cleanup


def _worker(case):
    import torch
    torch.set_num_threads(1)
    hardware = _hardware(torch)
    result = {"case": case, "hardware": hardware, "versions": _versions(), "status": "starting"}
    if not hardware["cuda_available"]:
        return {**result, "status": "blocked_no_cuda", "physical_gpu_execution": False}
    if hardware["capability"] != [7, 5] and not case["allow_other_gpu"]:
        return {**result, "status": "blocked_wrong_architecture", "physical_gpu_execution": False}
    torch.cuda.set_device(0)
    torch.manual_seed(case["seed"])
    factories = {"head": _make_head, "projection": _make_projection, "nf4": _make_nf4, "fused": _make_fused,
                 "rotary": _make_rotary, "lora_epilogue": _make_lora_epilogue}
    operation, snapshot, reference, validate, cleanup = factories[case["kind"]](case, torch)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_memory = _memory(torch)
    start = time.perf_counter()
    output = operation()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - start) * 1000
    cold_memory = _memory(torch)
    actual = snapshot(output)
    del output
    cleanup()
    expected = reference()
    validation = validate(actual, expected)
    del actual, expected
    cleanup()
    gc.collect()
    torch.cuda.synchronize()
    result.update(physical_gpu_execution=True, cold_first_operation_wall_ms=cold_ms,
                  cold_memory_before=initial_memory, cold_memory_after=cold_memory, validation=validation)
    if not validation["numerically_close"]:
        result["status"] = "numerical_screen_failed"
        result["warm_timing"] = None
        return result
    for _ in range(case["warmup"]):
        output = operation()
        del output
    cleanup()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    warm_before = _memory(torch)
    wall_ms, event_ms = [], []
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Initialize event objects outside timing. Leave allocations and host dispatch
    # in the operation; sparse nonzero's synchronization is part of its cost.
    start_event.record()
    end_event.record()
    end_event.synchronize()
    for _ in range(case["iterations"]):
        torch.cuda.synchronize()
        start = time.perf_counter()
        start_event.record()
        output = operation()
        end_event.record()
        end_event.synchronize()
        wall_ms.append((time.perf_counter() - start) * 1000)
        event_ms.append(start_event.elapsed_time(end_event))
        del output
    warm_after = _memory(torch)
    cleanup()
    result.update(status="measured_numerically_close", warm_memory_before=warm_before, warm_memory_after=warm_after,
                  warm_timing={"synchronized_wall": _summary(wall_ms), "cuda_event_interval": _summary(event_ms)},
                  official_contest_result=False)
    return result


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.probe_output:
        import torch
        _write(args.probe_output, _hardware(torch))
        return 0
    if args.worker_input:
        if args.worker_output is None:
            parser.error("--worker-output is required for an isolated worker")
        case = json.loads(args.worker_input.read_text())
        try:
            result = _worker(case)
        except Exception as error:
            result = {"case": case, "status": "error", "exception_type": type(error).__name__,
                      "error": str(error), "traceback": traceback.format_exc(), "official_contest_result": False}
        _write(args.worker_output, result)
        return 0 if result["status"] == "measured_numerically_close" else 1
    if min(args.batch, args.width, args.vocab, *args.projection_rows, *args.fused_rows, *args.chunk_sizes) < 1 or min(args.lengths) < 2:
        parser.error("dimensions/chunk sizes must be positive, and sequence lengths must be at least 2")
    if any(not 0 < fraction <= 1 for fraction in args.fractions):
        parser.error("fractions must be in (0, 1]")
    if args.iterations < 1 or args.warmup < 1 or args.timeout < 1 or not math.isfinite(args.loss_scale) or args.loss_scale <= 0:
        parser.error("iterations, warmup, timeout and loss scale must be positive")
    if set(args.head_projections) - {"torch", "t4_mma", "t4_fused"}:
        parser.error("head projections must be torch, t4_mma and/or t4_fused")
    cases = _cases(args)
    if args.list_cases:
        print(json.dumps({"count": len(cases), "cases": cases}, indent=2))
        return 0
    if os.environ.get("TRITON_INTERPRET") == "1":
        parser.error("this benchmark requires physical CUDA execution; unset TRITON_INTERPRET")
    from harness.check_rules import solution_sha256
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Synthetic isolated-kernel/head screening only. No full training, quality, admission or 5% kernel-share result.",
              "status": "running", "versions": _versions(), "hardware": _probe_hardware(),
              "solution_sha256": solution_sha256(REPO / "solution"), "settings_source": (REPO / "solution/settings.py").read_text(),
              "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "argv": sys.argv[1:], "cases_planned": cases, "results": [], "official_contest_result": False,
              "memory_scope": "PyTorch process allocator peaks, including resident inputs/cache. Not external sampled device peak.",
              "timing_scope": "First operation excludes process/import/CUDA/input setup. Warm events include inter-launch idle time.",
              "nvidia_smi_before": _nvidia_snapshot()}
    if not report["hardware"]["cuda_available"]:
        report["status"] = "blocked_no_cuda"
        _write(args.output, report)
        print(f"CUDA is unavailable; no benchmark executed. Diagnostic report: {args.output}", file=sys.stderr)
        return 2
    if report["hardware"]["capability"] != [7, 5] and not args.allow_other_gpu:
        report["status"] = "blocked_wrong_architecture"
        _write(args.output, report)
        print("SM75 is required by default. Use --allow-other-gpu only for explicitly non-T4 screening.", file=sys.stderr)
        return 2
    _write(args.output, report)
    for index, case in enumerate(cases):
        print(f"[{index+1}/{len(cases)}] {case['case_id']}", flush=True)
        with tempfile.TemporaryDirectory(prefix="soup_candidate_") as directory:
            temporary = Path(directory)
            input_path, output_path = temporary / "case.json", temporary / "result.json"
            _write(input_path, case)
            environment = dict(os.environ, TRITON_CACHE_DIR=str(temporary / "triton_cache"),
                               TORCHINDUCTOR_CACHE_DIR=str(temporary / "inductor_cache"), OMP_NUM_THREADS="1")
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-input", str(input_path),
                       "--worker-output", str(output_path)]
            started = time.perf_counter()
            try:
                process = subprocess.run(command, cwd=REPO, env=environment, text=True,
                                         capture_output=True, timeout=args.timeout, check=False)
                result = json.loads(output_path.read_text()) if output_path.exists() else {
                    "case": case, "status": "worker_failed_without_result"}
                result.update(worker_exit_code=process.returncode,
                              worker_log_tail=(process.stdout + process.stderr)[-12000:])
            except subprocess.TimeoutExpired as error:
                result = {"case": case, "status": "timeout", "error": str(error)}
            result["isolated_process_wall_s_including_setup"] = time.perf_counter() - started
        report["results"].append(result)
        _write(args.output, report)
        print(f"  {result['status']}", flush=True)
    successes = sum(item["status"] == "measured_numerically_close" for item in report["results"])
    report.update(status="completed" if successes == len(cases) else "completed_with_failures",
                  measured_numerically_close=successes, failed_or_blocked=len(cases)-successes,
                  isolated_case_comparisons=_comparisons(report["results"]),
                  nvidia_smi_after=_nvidia_snapshot(), finished_utc=datetime.now(timezone.utc).isoformat())
    _write(args.output, report)
    print("\ncase                                           warm wall ms   allocated peak MiB")
    for item in report["results"]:
        if item["status"] == "measured_numerically_close":
            elapsed = item["warm_timing"]["synchronized_wall"]["median"]
            peak = item["warm_memory_after"]["peak_allocated_bytes"] / 2**20
            print(f"{item['case']['case_id']:<46s} {elapsed:12.3f} {peak:20.2f}")
        else:
            print(f"{item['case']['case_id']:<46s} {item['status']}")
    print(f"Written {args.output}: {successes}/{len(cases)} cases measured. This is not an official competition result.")
    return 0 if successes == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
