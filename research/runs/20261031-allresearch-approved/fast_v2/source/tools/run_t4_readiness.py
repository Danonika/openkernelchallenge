"""Fail-fast CUDA readiness and optional unchanged official measurement.

  python tools/run_t4_readiness.py --plan
  python tools/run_t4_readiness.py --output-dir runs/readiness_default
  python tools/run_t4_readiness.py --include-experiments --output-dir runs/readiness_all
  python tools/run_t4_readiness.py --official --expected-commit FULL_40_CHARACTER_SHA

The default tests the active source settings, not every disabled experiment.
--include-experiments additionally checks the optional PTX/storage paths. CUDA
test groups must actually pass: interpreter-only results and silently skipped
mandatory GPU tests cannot satisfy readiness. Some explicitly CPU-interpreter
tests are expected to skip during a physical-GPU run.

TRITON_INTERPRET is removed from child environments. CUDA preflight and tests
run in child processes; this parent never imports torch or retains a CUDA
context. The official stage invokes unchanged task/harness entry points with
two paired repeats. It does not change settings, install dependencies, enable
optimizations, reuse readiness compiler caches or claim hidden-jury admission.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]
STARTER_COMMIT = "c7c2fcd15972ce8b898d8d3ff732ee192c817ceb"
PROTECTED_SHA256 = {
    "harness/_train.py": "45a62642dceab32cf3efcda504de87a9c25ea28b119b1806f56d14ff1146a89d",
    "harness/check_rules.py": "879ab1861674f6e714eadd00aaf044235d98526dd75fab1035deec3401121ef5",
    "harness/common.py": "c83066f24f266dab62ebcc9c2d44f5a65902760b53da912e65ebfe511579c2db",
    "harness/eval_adapter.py": "47cddbb8e4579c69f9535b099ff2371b1c3af08bfbd93d14bf79aadd74d38482",
    "harness/example_reference.py": "6a11055f4a1822e53971c228c6e3e3aa6b88edd9eb6db81a388e26a70a030ceb",
    "harness/run_task.py": "d935a8959f348977313f2eef36870ff2699b9ba013a7978918ff35e148968a79",
    "harness/score.py": "731c9b644dcc4069b0b069616f438064d1f7f376a202feb15201fd910ec688c5",
    "task/prepare_data.py": "a8543f11a55aff80dbdc714a0e62391244a5296616eea2e34dc2eb3ff3ec18f7",
    "task/task.yaml": "dd70c739fcd143e06aef3bb1efcc9386f682bf1de26f30e08409d86a97ff47ce",
    "tests/conftest.py": "a82eabab468ea0ef1e003cccd7c3827ce2f4d00d09c32948a34fb48221f7337a",
    "tests/test_callback.py": "1919303a3670c0a33abbfce9b33eab0913bb64a3a21c3312d86531c32b5bf490",
    "tests/test_config.py": "db808c44470d144b7cf18b7329492f41f76e37cfc24306e7cba0488bca9d787c",
    "tests/test_formats.py": "5d656ed50c59f794e934473d459ea85635f892ba028e192414474e9b5d65919e",
    "tests/test_loader.py": "3dad1bea2d33aab97e52b7cd44888d768bc9519af44fbe6cf3e96c260026632f",
    "tests/test_trainer_init.py": "b7608d62a40983bf7c9e5b270e0e527b641a837e462d6df2909fec88f607e821",
}
PINNED_PACKAGES = {"soup-cli": "0.77.0", "transformers": "5.16.1",
                   "peft": "0.20.0", "trl": "0.29.1", "bitsandbytes": "0.50.2", "accelerate": "1.14.0"}


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _settings():
    source = (REPO / "solution/settings.py").read_text(encoding="utf-8")
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = ast.literal_eval(node.value)
    required = {"HEAD_PROJECTION", "HEAD_MODE", "HEAD_NUMERICS", "CACHED_CE_CHUNK_TOKENS", "CACHED_CE_BACKWARD", "PACKED_METRICS",
                "FUSED_NF4", "NF4_FORWARD_CACHE_BYTES", "NF4_CACHE_BACKWARD", "NF4_DIRECT_LINEAR", "FROZEN_STORAGE", "COMPACT_MLP_ACTIVATIONS",
                "CHUNK_TOKENS", "SPARSE_BACKWARD", "FUSED_ROTARY", "FUSED_LORA_EPILOGUE",
                "FUSED_LORA_DROPOUT", "SHARED_ATTENTION_MASK", "FUSED_RMSNORM", "MLP_PADDING_COMPACTION", "VARLEN_ATTENTION"}
    missing = required - set(values)
    if missing:
        raise ValueError(f"missing fixed source settings: {sorted(missing)}")
    return values


def _protected():
    changed, missing = [], []
    for name, expected in PROTECTED_SHA256.items():
        path = REPO / name
        if not path.is_file():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(name)
    actual = {str(path.relative_to(REPO)) for directory in ("harness", "task", "tests")
              for path in (REPO / directory).rglob("*")
              if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    extra = sorted(actual - set(PROTECTED_SHA256))
    return {"starter_commit": STARTER_COMMIT, "ok": not (changed or missing or extra),
            "changed": changed, "missing": missing, "extra": extra}


def _source_hash():
    # The unchanged official helper is safe to import: it does not initialize CUDA.
    sys.path.insert(0, str(REPO))
    from harness.check_rules import solution_sha256
    return solution_sha256(REPO / "solution")


def _git_info():
    def read(*args):
        result = subprocess.run(["git", "-C", str(REPO), *args], text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    remote = read("remote", "get-url", "origin")
    if remote and "://" in remote:
        parsed = urlsplit(remote)
        remote = f"{parsed.scheme}://{parsed.hostname or ''}{parsed.path}"
    return {"commit": read("rev-parse", "HEAD"), "remote": remote,
            "initial_status": read("status", "--porcelain", "--untracked-files=all")}


def _environment():
    environment = dict(os.environ)
    environment.pop("TRITON_INTERPRET", None)
    environment.pop("SOUP_APPLY_SOLUTION", None)
    # PyTorch's newer alias can override the exact allocator used by the harness.
    environment.pop("PYTORCH_ALLOC_CONF", None)
    environment.update(CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1",
                       TOKENIZERS_PARALLELISM="false")
    return environment


def _probe(output):
    import torch
    versions = {"python": sys.version.split()[0]}
    for name in ("torch", "triton", *PINNED_PACKAGES, "numpy", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    hardware = {"cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda,
                "triton_interpret": os.environ.get("TRITON_INTERPRET"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if hardware["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        hardware.update(name=properties.name, capability=list(torch.cuda.get_device_capability(0)),
                        device_count=torch.cuda.device_count(), total_memory_bytes=properties.total_memory,
                        allocator_backend=torch.cuda.memory.get_allocator_backend())
    test_tools = {}
    for name in ("pytest", "pluggy", "iniconfig", "pygments", "packaging"):
        spec = importlib.util.find_spec(name)
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        test_tools[name] = {"version": version, "module_origin": spec.origin if spec else None}
    _write(output, {"hardware": hardware, "versions": versions, "test_tools": test_tools})


def _stop_child_tree(process):
    """Stop only this runner's child tree, including nested process sessions.

    The official harness deliberately starts training in a new process group;
    killing only the outer harness group would leave that CUDA process alive.
    psutil is an existing Accelerate dependency, not an installed extra here.
    """
    import psutil
    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return
    descendants = {}
    try:
        parent.suspend()
        # Suspend descendants before killing parents so a child cannot fork a
        # new training/compiler process during cancellation. Recheck the tree
        # to catch a fork that preceded a descendant's suspension.
        while True:
            added = False
            for child in parent.children(recursive=True):
                if child.pid not in descendants:
                    descendants[child.pid] = child
                    added = True
                    try:
                        child.suspend()
                    except psutil.NoSuchProcess:
                        pass
            if not added:
                break
    except psutil.NoSuchProcess:
        pass
    finally:
        for child in reversed(list(descendants.values())):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        psutil.wait_procs(list(descendants.values()), timeout=5)


def _run(command, log_path, environment, timeout):
    import psutil  # Fail before starting a child if the normal Accelerate dependency is missing.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN " + " ".join(map(str, command)), flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)

        def relay():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=relay, daemon=True)
        reader.start()
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_child_tree(process)
            returncode = process.wait()
        except BaseException:
            _stop_child_tree(process)
            process.wait()
            reader.join(timeout=10)
            raise
        reader.join(timeout=10)
    return {"command": list(map(str, command)), "returncode": returncode, "timeout": timed_out,
            "wall_s": time.perf_counter() - started, "log": str(log_path)}


def _test_plan(settings, include_experiments):
    files = ["solution/test_kernels.py", "solution/test_integration.py", "solution/test_fusion_cuda.py"]
    required = ["test_swiglu_matches_pytorch", "test_physical_t4_full_fusion_training"]
    excluded = []
    minimum_counts = {"test_physical_t4_full_fusion_training": 2}
    if settings["FUSED_RMSNORM"] or include_experiments:
        files.append("solution/test_rmsnorm.py")
        required += ["test_rmsnorm_forward_backward", "test_rmsnorm_preserves_scaled_native_gradient"]
        minimum_counts.update(test_rmsnorm_forward_backward=6,
                              test_rmsnorm_preserves_scaled_native_gradient=6)
    if settings["HEAD_MODE"] in ("cached", "sparse_cached") or include_experiments:
        files.append("solution/test_cached_head.py")
        required += ["test_cached_summary_and_scaled_native_gradient",
                     "test_saturated_and_nonfinite_rows_keep_native_behavior",
                     "test_all_ignored_denominator_semantics",
                     "test_saves_original_half_payload_and_no_full_float_payload",
                     "test_native_head_is_called_once_and_keeps_its_full_backward_geometry"]
        minimum_counts.update(test_cached_summary_and_scaled_native_gradient=18,
                              test_saturated_and_nonfinite_rows_keep_native_behavior=10,
                              test_all_ignored_denominator_semantics=3)
        if settings["HEAD_NUMERICS"] == "native":
            required += ["test_cached_native_full_vocabulary_guard_boundary",
                         "test_cached_native_statistics_are_bounded_without_custom_work"]
            minimum_counts.update(test_cached_native_full_vocabulary_guard_boundary=4,
                                  test_cached_native_statistics_are_bounded_without_custom_work=1)
    if settings["HEAD_MODE"] == "sparse_cached" or include_experiments:
        files.append("solution/test_sparse_cached_head.py")
        files.append("solution/test_sparse_cached_integration.py")
        required.append("test_sparse_cached_physical_full_vocabulary_scaled_gradient_norm")
        minimum_counts["test_sparse_cached_physical_full_vocabulary_scaled_gradient_norm"] = 4
    if settings["HEAD_MODE"] == "streamed" or include_experiments:
        files.append("solution/test_streamed_head.py")
        required += ["test_loss_metrics_and_scaled_gradient",
                     "test_sparse_backward_matches_dense_backward",
                     "test_saturated_large_offset_rows_use_native_loss_and_scaled_backward",
                     "test_ignored_nonfinite_rows_preserve_native_gradient_and_entropy"]
        minimum_counts.update(test_saturated_large_offset_rows_use_native_loss_and_scaled_backward=8,
                              test_ignored_nonfinite_rows_preserve_native_gradient_and_entropy=8)
        if settings["HEAD_NUMERICS"] == "native":
            required.append("test_native_head_precision_cuda_only")
            minimum_counts["test_native_head_precision_cuda_only"] = 10
    if settings["CACHED_CE_BACKWARD"] == "native_fused" or include_experiments:
        files.append("solution/test_cached_ce_backward.py")
        files.append("solution/test_cached_ce_integration.py")
        required += ["test_cached_ce_consumer_physical_full_vocab_exact_native_scaled_gradients",
                     "test_cached_ce_consumer_random_native_gradient_and_tails",
                     "test_cached_ce_consumer_full_vocab_saturated_peak_columns",
                     "test_cached_ce_consumer_nonfinite_ignored_and_explicit_divisors"]
        minimum_counts.update(test_cached_ce_consumer_random_native_gradient_and_tails=12,
                              test_cached_ce_consumer_full_vocab_saturated_peak_columns=8,
                              test_cached_ce_consumer_nonfinite_ignored_and_explicit_divisors=10)
    if settings["PACKED_METRICS"] or include_experiments:
        files.append("solution/test_metric_transfer.py")
        required.append("test_physical_cuda_metric_and_rewind_packets")
        minimum_counts["test_physical_cuda_metric_and_rewind_packets"] = 2
    if settings["FUSED_LORA_DROPOUT"] or include_experiments:
        files.append("solution/test_lora_dropout.py")
        required += ["test_physical_t4_native_peft_dropout_gradient_and_rng",
                     "test_physical_t4_nf4_lora_dropout_gradient_and_rng"]
        minimum_counts.update(test_physical_t4_nf4_lora_dropout_gradient_and_rng=3)
    if settings["SHARED_ATTENTION_MASK"] or include_experiments:
        files.append("solution/test_shared_attention_mask.py")
        required.append("test_physical_t4_qwen_attention_composition")
    if settings["FUSED_LORA_EPILOGUE"] or include_experiments:
        files.append("solution/test_lora_epilogue.py")
        required += ["test_lora_epilogue_rounding_edges_cuda_only",
                     "test_native_cuda_peft_nf4_epilogue_and_gradscaler"]
        minimum_counts.update(test_native_cuda_peft_nf4_epilogue_and_gradscaler=5)
        if settings["FUSED_LORA_DROPOUT"] or include_experiments:
            files.append("solution/test_lora_joint_cuda.py")
            required.append("test_physical_t4_joint_lora_epilogues_native_nf4")
            minimum_counts["test_physical_t4_joint_lora_epilogues_native_nf4"] = 6
    if settings["FUSED_ROTARY"] or include_experiments:
        files.append("solution/test_rotary.py")
        required += ["test_rotary_cuda_full_task_shape",
                     "test_rotary_cuda_sdpa_cast_and_grouped_attention_boundary"]
        minimum_counts.update(test_rotary_cuda_full_task_shape=2,
                              test_rotary_cuda_sdpa_cast_and_grouped_attention_boundary=2)
    if settings["FUSED_NF4"] or include_experiments:
        files.append("solution/test_nf4.py")
        required += ["test_nf4_matches_native_bitsandbytes_representation",
                     "test_nf4_patched_linear4bit_forward_and_scaled_backward"]
    if settings["NF4_FORWARD_CACHE_BYTES"] or include_experiments:
        files.append("solution/test_nf4_forward_cache.py")
        required.append("test_nf4_forward_cache_physical_linear4bit_exact_forward_and_uncached_backward")
        minimum_counts["test_nf4_forward_cache_physical_linear4bit_exact_forward_and_uncached_backward"] = 2
    if settings["NF4_CACHE_BACKWARD"] or include_experiments:
        files.append("solution/test_nf4_backward_cache.py")
        required.append("test_nf4_backward_cache_physical_linear4bit_exact_two_steps")
        minimum_counts["test_nf4_backward_cache_physical_linear4bit_exact_two_steps"] = 4
    if settings["NF4_DIRECT_LINEAR"] or include_experiments:
        files.append("solution/test_nf4_cached_linear.py")
        required.extend(("test_nf4_cached_linear_physical_exact_native_two_steps",
                         "test_nf4_cached_linear_physical_native_fallback_and_first_bias_cast"))
        minimum_counts.update(test_nf4_cached_linear_physical_exact_native_two_steps=12,
                              test_nf4_cached_linear_physical_native_fallback_and_first_bias_cast=10)
    if settings["COMPACT_MLP_ACTIVATIONS"] or include_experiments:
        files.append("solution/test_activation_storage.py")
        required += ["test_cuda_mixed_pointer_starter_backward_matches_fp32_saved_baseline",
                     "test_cuda_native_nf4_mlp_activates_compact_path_without_guard_overrides",
                     "test_cuda_direct_half_swiglu_preserves_fp32_arithmetic_and_cast_boundaries"]
    if settings["MLP_PADDING_COMPACTION"] or include_experiments:
        files.append("solution/test_mlp_padding.py")
        required.append("test_physical_t4_mlp_padding_native_nf4_gradients_rng_and_metrics")
        minimum_counts["test_physical_t4_mlp_padding_native_nf4_gradients_rng_and_metrics"] = 2
        required.append("test_physical_t4_mlp_padding_row_copy_exact")
        minimum_counts["test_physical_t4_mlp_padding_row_copy_exact"] = 2
    if settings["VARLEN_ATTENTION"] or include_experiments:
        files.extend(("solution/test_varlen_attention.py", "solution/test_varlen_attention_patch.py"))
        required.extend(("test_physical_t4_native_varlen", "test_physical_t4_varlen_real_qwen_registry_composition"))
        minimum_counts.update(test_physical_t4_native_varlen=7,
                              test_physical_t4_varlen_real_qwen_registry_composition=2)
    if settings["FROZEN_STORAGE"] or include_experiments:
        files.append("solution/test_frozen_storage.py")
        required.append("test_cuda_gather_and_real_prepare_storage_reduction")
    if settings["HEAD_PROJECTION"] == "t4_mma" or include_experiments:
        files.append("solution/test_head_matmul_t4.py")
        required += ["test_t4_mma_head_cuda_only", "test_t4_projection_on_cuda",
                     "test_t4_projection_unaligned_contiguous_views_on_cuda"]
    else:
        excluded.append("t4_mma")
    if settings["HEAD_PROJECTION"] == "t4_fused" or include_experiments:
        files.append("solution/test_head_fused_t4.py")
        required += ["test_t4_fused_training_cuda_only", "test_fused_capability_guard_rejects_before_decoder_cuda_only",
                     "test_t4_fused_nonfinite_detection_and_native_fallback_cuda_only",
                     "test_fused_head_statistics_on_cuda", "test_fused_head_logit_gradient_on_cuda",
                     "test_fused_head_logit_gradient_ignored_nonfinite_scale_on_cuda",
                     "test_fused_head_logit_gradient_scales_before_fp16_underflow_on_cuda",
                     "test_fused_head_logit_gradient_full_state_and_compacted_chunks_on_cuda",
                     "test_fused_head_centered_state_preserves_large_common_offset_on_cuda"]
        minimum_counts.update(test_t4_fused_training_cuda_only=4, test_fused_head_statistics_on_cuda=4,
                              test_t4_fused_nonfinite_detection_and_native_fallback_cuda_only=2,
                              test_fused_head_logit_gradient_on_cuda=8,
                              test_fused_head_logit_gradient_ignored_nonfinite_scale_on_cuda=3)
    else:
        excluded += ["t4_fused", "fused_capability"]
    if settings["HEAD_PROJECTION"] == "triton" or include_experiments:
        files.append("solution/test_head_matmul.py")
    else:
        excluded.append("original_projection_forward_and_backward_path")
    expression = " and ".join(f"not {item}" for item in excluded)
    return {"files": files, "required_gpu_test_prefixes": required, "pytest_expression": expression,
            "required_gpu_minimum_counts": minimum_counts, "include_experiments": include_experiments}


def _junit(path, required=(), minimum_counts=None):
    if not path.is_file():
        return {"ok": False, "error": "pytest did not write its JUnit report"}
    root = ET.parse(path).getroot()
    cases = []
    evidence = []
    for node in root.iter("testcase"):
        status = "passed"
        for tag in ("skipped", "failure", "error"):
            if node.find(tag) is not None:
                status = tag
                break
        cases.append({"name": node.attrib.get("name", ""), "class": node.attrib.get("classname", ""), "status": status})
        for property_node in node.findall("./properties/property"):
            name, value = property_node.attrib.get("name", ""), property_node.attrib.get("value", "")
            if name.startswith("soup_") and len(value.encode()) <= 128 * 1024:
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
                evidence.append({"test": node.attrib.get("name", ""), "name": name, "value": value})
    groups = []
    minimum_counts = minimum_counts or {}
    for prefix in required:
        matches = [case for case in cases if case["name"].startswith(prefix)]
        minimum = minimum_counts.get(prefix, 1)
        groups.append({"prefix": prefix, "count": len(matches), "required_minimum": minimum,
                       "ok": len(matches) >= minimum and all(case["status"] == "passed" for case in matches),
                       "not_passed": [case for case in matches if case["status"] != "passed"]})
    counts = {status: sum(case["status"] == status for case in cases)
              for status in ("passed", "skipped", "failure", "error")}
    return {"ok": bool(cases) and not counts["failure"] and not counts["error"]
                  and all(group["ok"] for group in groups), "counts": counts, "required_gpu_groups": groups,
            "skipped_tests": [case for case in cases if case["status"] == "skipped"],
            "test_evidence": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/readiness"))
    parser.add_argument("--include-experiments", action="store_true")
    parser.add_argument("--official", action="store_true", help="after readiness, run unchanged two-repeat official training/evaluation/checks")
    parser.add_argument("--official-out", type=Path, default=Path("runs/final_candidate"))
    parser.add_argument("--expected-commit", help="full frozen team commit; required for --official")
    parser.add_argument("--local-snapshot", action="store_true",
                        help="benchmark a clean local source snapshot; not a team submission")
    parser.add_argument("--allow-version-mismatch", action="store_true", help="allow different officially pinned packages for development tests only; cannot be combined with --official")
    parser.add_argument("--test-timeout", type=int, default=1800)
    parser.add_argument("--train-timeout", type=int, default=900, help="per-run timeout passed unchanged to official harness")
    parser.add_argument("--plan", action="store_true", help="show source/settings/test plan; do not access CUDA or execute tests")
    parser.add_argument("--probe-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe_output:
        _probe(args.probe_output)
        return 0
    if args.test_timeout < 1 or args.train_timeout < 1:
        parser.error("timeouts must be positive")
    if args.expected_commit and not re.fullmatch(r"[0-9a-fA-F]{40}", args.expected_commit):
        parser.error("--expected-commit must be a full 40-character Git commit")
    if args.official and (not args.expected_commit or args.allow_version_mismatch):
        parser.error("--official requires --expected-commit and the package pins from the official notebook")
    if args.official and os.environ.get("SOUP_READINESS_TEST_DEPS"):
        parser.error("offline test dependencies are scoped to readiness; launch measurement from the original environment")
    settings, integrity, git = _settings(), _protected(), _git_info()
    plan = _test_plan(settings, args.include_experiments)
    if args.plan:
        print(json.dumps({"settings": settings, "protected_source": integrity, "git": git, "test_plan": plan,
                          "official_requested": args.official}, indent=2))
        return 0 if integrity["ok"] else 1
    output = args.output_dir.resolve()
    report_path = output / "readiness.json"
    if report_path.exists():
        parser.error("readiness.json already exists; choose a new --output-dir to retain earlier evidence")
    if args.official and args.official_out.exists() and any(args.official_out.iterdir()):
        parser.error("official output directory is not empty; choose a new --official-out to retain earlier evidence")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "preflight",
              "settings": settings, "protected_source": integrity, "git": git, "test_plan": plan,
              "solution_sha256": _source_hash(), "stages": [], "official_requested": args.official,
              "official_checks_passed": False, "hidden_jury_admission": "not_established",
              "source_kind": "local_snapshot" if args.local_snapshot else "team_checkout",
              "interpreter_removed_from_child_environment": True}
    environment = _environment()
    errors = []
    if not integrity["ok"]:
        errors.append("protected starter sources differ from the reviewed official commit")
    if args.expected_commit and git["commit"] != args.expected_commit.lower():
        errors.append("checkout HEAD does not equal the requested frozen commit")
    if args.official:
        if git["initial_status"] != "":
            errors.append("official run requires a clean, committed checkout before generated outputs")
        if not args.local_snapshot and (not git["remote"] or
                "makazhanalpamys/soup-kernel-challenge" in git["remote"].lower()):
            errors.append("use the platform-created team repository, not the official starter remote")
    _write(report_path, report)
    try:
        with tempfile.TemporaryDirectory(prefix="soup_readiness_") as cache:
            environment.update(TRITON_CACHE_DIR=str(Path(cache) / "triton"),
                               TORCHINDUCTOR_CACHE_DIR=str(Path(cache) / "inductor"))
            probe_path = output / "preflight.json"
            stage = _run([sys.executable, str(Path(__file__).resolve()), "--probe-output", str(probe_path)],
                         output / "preflight.log", environment, 60)
            report["stages"].append({"name": "cuda_preflight", **stage})
            if stage["returncode"] or not probe_path.exists():
                errors.append("CUDA preflight process failed")
                probe = {}
            else:
                probe = json.loads(probe_path.read_text())
            report["preflight"] = probe
            hardware, versions = probe.get("hardware", {}), probe.get("versions", {})
            if (not hardware.get("cuda_available") or "T4" not in hardware.get("name", "")
                    or hardware.get("capability") != [7, 5] or hardware.get("device_count") != 1
                    or hardware.get("cuda_visible_devices") != "0"):
                errors.append("requires exactly one visible Tesla T4 (GPU 0), SM75, with real CUDA")
            mismatches = [f"{name}: expected {expected}, found {versions.get(name)}"
                          for name, expected in PINNED_PACKAGES.items() if versions.get(name) != expected]
            report["version_mismatches"] = mismatches
            deviations = []
            if versions.get("triton") != "3.6.0":
                deviations.append(f"Triton differs from the audited 3.6.0: {versions.get('triton')}")
            if not str(versions.get("torch", "")).startswith("2.11."):
                deviations.append(f"PyTorch differs from the audited 2.11.x: {versions.get('torch')}")
            if not str(versions.get("python", "")).startswith("3.13."):
                deviations.append(f"Python differs from the described Kaggle image 3.13.x: {versions.get('python')}")
            report["audit_deviations_not_organizer_package_pins"] = deviations
            if deviations:
                print("AUDIT CONTEXT: " + "; ".join(deviations), flush=True)
            if mismatches and not args.allow_version_mismatch:
                errors.append("installed packages do not match the official notebook pins")
            if not versions.get("pytest"):
                errors.append("pytest is unavailable in the readiness subprocess")
            if errors:
                report.update(status="blocked_preflight", errors=errors, physical_cuda_tests_executed=False)
                _write(report_path, report)
                print("Preflight blocked: " + "; ".join(errors), file=sys.stderr)
                return 2
            gpu_xml = output / "candidate_cuda.xml"
            command = [sys.executable, "-m", "pytest", *plan["files"], "-q", "-ra", f"--junitxml={gpu_xml}"]
            if plan["pytest_expression"]:
                command += ["-k", plan["pytest_expression"]]
            stage = _run(command, output / "candidate_cuda.log", environment, args.test_timeout)
            report["stages"].append({"name": "candidate_cuda_tests", **stage})
            report["candidate_cuda"] = _junit(gpu_xml, plan["required_gpu_test_prefixes"], plan["required_gpu_minimum_counts"])
            report["physical_cuda_tests_executed"] = True
            _write(report_path, report)
            if stage["returncode"] or not report["candidate_cuda"]["ok"]:
                report["status"] = "candidate_cuda_tests_failed_or_skipped"
                _write(report_path, report)
                return 1
            regression_xml = output / "official_regressions.xml"
            stage = _run([sys.executable, "-m", "pytest", "tests", "-q", f"--junitxml={regression_xml}"],
                         output / "official_regressions.log", dict(environment, SOUP_APPLY_SOLUTION="1"), args.test_timeout)
            report["stages"].append({"name": "official_regressions", **stage})
            report["official_regressions"] = _junit(regression_xml)
            counts = report["official_regressions"].get("counts", {})
            if stage["returncode"] or not report["official_regressions"]["ok"] or counts.get("passed") != 79 or counts.get("skipped"):
                report["status"] = "official_regressions_failed_or_incomplete"
                _write(report_path, report)
                return 1
            if _source_hash() != report["solution_sha256"] or not _protected()["ok"]:
                report["status"] = "source_changed_during_readiness"
                _write(report_path, report)
                return 1
            report["status"] = "cuda_readiness_passed_official_training_not_run"
            _write(report_path, report)
            if not args.official:
                print(f"CUDA readiness passed; full training/quality/5% share remain unmeasured. Report: {report_path}")
                return 0
            if not all((REPO / name).exists() for name in ("data/train.jsonl", "data/eval_public.jsonl")):
                stage = _run([sys.executable, "task/prepare_data.py"], output / "prepare_data.log", environment, 1800)
                report["stages"].append({"name": "prepare_public_data", **stage})
                if stage["returncode"]:
                    report["status"] = "public_data_preparation_failed"
                    _write(report_path, report)
                    return 1
            official_out = args.official_out.resolve()
            stage = _run([sys.executable, "harness/run_task.py", "--repeats", "2", "--out", str(official_out),
                          "--timeout", str(args.train_timeout)], output / "official_training.log", environment,
                         None)  # The unchanged harness owns per-run timeouts and child-process cleanup.
            report["stages"].append({"name": "official_paired_training", **stage})
            if stage["returncode"]:
                report["status"] = "official_training_command_failed"
                _write(report_path, report)
                return 1
            if _source_hash() != report["solution_sha256"] or not _protected()["ok"]:
                report["status"] = "source_changed_during_official_training"
                _write(report_path, report)
                return 1
            stage = _run([sys.executable, "harness/check_rules.py", "--result", str(official_out / "result.json")],
                         output / "official_checks.log", environment, args.test_timeout)
            report["stages"].append({"name": "official_admission_checks", **stage})
            report["official_result"] = str(official_out / "result.json")
            report["official_checks"] = str(official_out / "checks.json")
            checks = json.loads((official_out / "checks.json").read_text()) if (official_out / "checks.json").is_file() else {}
            report["official_checks_passed"] = (stage["returncode"] == 0 and len(checks.get("checks", [])) == 10
                                                and all(item.get("ok") is True for item in checks["checks"]))
            if not report["official_checks_passed"]:
                report["status"] = "official_checks_failed"
                _write(report_path, report)
                return 1
            stage = _run([sys.executable, "harness/score.py", str(official_out / "result.json")],
                         output / "official_score.log", environment, 60)
            report["stages"].append({"name": "official_score_display", **stage})
            report["status"] = "public_official_checks_passed" if stage["returncode"] == 0 else "score_display_failed"
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write(report_path, report)
            print(f"Report: {report_path}; hidden-jury results and rank are not established.")
            return 0 if stage["returncode"] == 0 else 1
    except KeyboardInterrupt:
        report.update(status="cancelled", finished_utc=datetime.now(timezone.utc).isoformat())
        _write(report_path, report)
        return 130
    except Exception as error:
        report.update(status="runner_error", error=str(error), traceback=traceback.format_exc())
        _write(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
