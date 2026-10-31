"""Synthetic evidence fixtures test refusal logic, never GPU performance.

These files are created exclusively under pytest temporary directories. Their
invented timings/check results are not real challenge runs or deliverables.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


_SPEC = importlib.util.spec_from_file_location("verify_record", Path(__file__).with_name("verify_record.py"))
verify = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify)


@pytest.fixture
def evidence(tmp_path):
    repo = tmp_path / "synthetic_repo"
    for name in verify.PROTECTED_SHA256:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(verify.REPO / name, target)
    (repo / "solution").mkdir()
    (repo / "solution/kernel.py").write_text(
        "import triton\nimport triton.language as tl\n"
        "@triton.jit\ndef fixture_original_reduction(x, y, WIDTH: tl.constexpr):\n"
        "    row = tl.program_id(0)\n    col = tl.arange(0, WIDTH)\n"
        "    vals = tl.load(x + row * WIDTH + col)\n"
        "    mean = tl.sum(vals, axis=0) / WIDTH\n"
        "    tl.store(y + row, mean)\n"
    )
    result = {"fixture_kind": "synthetic verifier test, not a measured benchmark",
              "task": {"repeats": 2, "only": None, "seed": 1},
              "packages_changed": [], "runs": {"base": [], "solution": []},
              "evals": {side: {"exit_code": 0, "mean_loss": 2.0} for side in ("base", "solution")}}
    for side, times in (("base", (76., 78.)), ("solution", (31., 33.))):
        for index, wall in enumerate(times, 1):
            loop = wall - 10
            metrics = {"exit_code": 0, "solution": side == "solution", "steps": 64, "timed_steps": 62,
                       "train_loop_s": loop, "timed_loop_s": loop - 1, "cuda_device": "Tesla T4",
                       "device_count": 1, "visible_devices": "0", "alloc_conf": "expandable_segments:True",
                       "alloc_conf_expected": "expandable_segments:True", "torch_max_reserved_mib": 8000,
                       "triton_watch_error": None, "banned_modules_loaded": [],
                       "triton_launches": {}, "triton_files": {}, "own_kernel_ms": {},
                       "trainer": {"class": "RewindSFTTrainer", "fp16": True, "bf16": False,
                                   "per_device_train_batch_size": 4, "gradient_accumulation_steps": 1}}
            if side == "solution":
                metrics.update(triton_launches={"fixture_original_reduction": 64},
                               triton_files={"fixture_original_reduction": "solution/kernel.py"},
                               own_kernel_ms={"fixture_original_reduction": (loop - 1) * 70})
            result["runs"][side].append({
                "run": f"{side}_{index}", "side": side, "exit_code": 0, "wall_s": wall,
                "gpu_peak_mib": 10000 if side == "base" else 8500, "gpu_start_mib": 0,
                "metrics": metrics, "global_step": 64,
                "logged_metrics": ["loss", "entropy", "num_tokens", "mean_token_accuracy"],
                "adapter_tensors": {"q_proj.lora_A.weight": ["F32", [16, 1536]]},
                "adapter_config": {"r": 16, "lora_alpha": 32, "lora_dropout": .05},
                "log": {"loss": [2.] * 64, "entropy": [2.3] * 64, "mean_token_accuracy": [.4] * 64,
                        "num_tokens": [100 * (i + 1) for i in range(64)]},
                "rewind": [[[4 * i + j for j in range(4)], [20] * 4, [2.] * 4] for i in range(64)],
            })
    official = verify._official_helpers(repo)
    checks = {"solution_sha256": official.solution_sha256(repo / "solution"),
              "checks": [{"check": name, "ok": True, "detail": "79 passed (synthetic fixture)" if index == 9 else "synthetic fixture"}
                         for index, name in enumerate(verify.CHECK_NAMES)], "notes": []}
    path = repo / "runs/final/result.json"
    path.parent.mkdir(parents=True)

    def write(*, refresh_summary=True, refresh_result_hash=True):
        if refresh_summary:
            import statistics
            summary = {}
            for side, runs in result["runs"].items():
                summary[side] = {"ok": True, "runs": len(runs),
                                 "wall_s": round(statistics.median(r["wall_s"] for r in runs), 2),
                                 "gpu_peak_mib": max(r["gpu_peak_mib"] for r in runs),
                                 "train_loop_s": round(statistics.median(r["metrics"]["train_loop_s"] for r in runs), 2),
                                 "eval_loss": result["evals"][side]["mean_loss"]}
            summary["ratios"] = {"speedup": round(summary["base"]["wall_s"] / summary["solution"]["wall_s"], 4),
                                  "memory_ratio": round(summary["base"]["gpu_peak_mib"] / summary["solution"]["gpu_peak_mib"], 4),
                                  "quality_delta_pct": 0.}
            result["summary"] = summary
        path.write_text(json.dumps(result))
        if refresh_result_hash:
            checks["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_name("checks.json").write_text(json.dumps(checks))
    write()
    return repo, path, result, checks, write


def assess(evidence, **kwargs):
    repo, path, *_ = evidence
    return verify.verify_result(path, repo_root=repo, **kwargs)


def reject(report, text):
    assert report["target_met_under_official_public_protocol"] is False
    assert any(text in reason for reason in report["reasons"]), report


def test_32_second_target_passes_without_world_record_claim(evidence):
    report = assess(evidence, record_seconds=33, reference_baseline_seconds=77)
    assert report["target_met_under_official_public_protocol"] is True, report
    assert report["passes_recorded_public_automatic_gates"] is True
    assert report["proven_world_record"] is False
    assert report["measurements"]["speedup_vs_paired_baseline"] == 77 / 32
    assert report["provenance"]["per_run_source_hashes_recorded"] is False


@pytest.mark.parametrize("target", [32., 31.99])
def test_tie_and_slow_result_do_not_qualify(evidence, target):
    report = assess(evidence, record_seconds=target)
    reject(report, "not strictly below")
    assert report["passes_recorded_public_automatic_gates"] is True


def test_actual_paired_baseline_overrides_user_reference(evidence):
    _, _, result, _, write = evidence
    for run in result["runs"]["base"]:
        run["wall_s"] += 3
    write()
    report = assess(evidence, reference_baseline_seconds=77)
    assert report["target_met_under_official_public_protocol"] is True, report
    assert report["measurements"]["speedup_vs_paired_baseline"] == 80 / 32


def test_exact_result_bytes_are_bound_even_for_whitespace_changes(evidence):
    _, path, *_ = evidence
    path.write_bytes(path.read_bytes() + b"\n")
    reject(assess(evidence), "exact result.json bytes")


def test_current_solution_change_invalidates_prior_checks(evidence):
    repo, *_ = evidence
    with (repo / "solution/kernel.py").open("a") as f:
        f.write("\n# a new candidate, not the checked source\n")
    reject(assess(evidence), "current solution source hash differs")


def test_single_run_cannot_claim_target(evidence):
    _, _, result, _, write = evidence
    result["task"]["repeats"] = 1
    for side in result["runs"]:
        result["runs"][side] = result["runs"][side][:1]
    write()
    reject(assess(evidence), "at least two matched")


def test_missing_checks_fail_closed(evidence):
    _, path, *_ = evidence
    path.with_name("checks.json").unlink()
    reject(assess(evidence), "cannot verify official evidence")


@pytest.mark.parametrize("kind", ["string_true", "failed", "duplicate_name", "missing", "skipped_tests"])
def test_all_checks_must_be_complete_and_genuine_booleans(evidence, kind):
    _, _, _, checks, write = evidence
    if kind == "string_true":
        checks["checks"][2]["ok"] = "true"
    elif kind == "failed":
        checks["checks"][2]["ok"] = False
    elif kind == "duplicate_name":
        checks["checks"][2]["check"] = checks["checks"][1]["check"]
    elif kind == "missing":
        checks["checks"].pop()
    else:
        checks["checks"][-1]["detail"] = "skipped (--skip-tests)"
    write()
    reject(assess(evidence), "79 tests" if kind == "skipped_tests" else "all ten distinct")


def test_false_summary_is_detected_even_with_matching_result_digest(evidence):
    _, _, result, _, write = evidence
    result["summary"]["solution"]["wall_s"] = 1
    write(refresh_summary=False)
    reject(assess(evidence), "summary wall_s disagrees")


@pytest.mark.parametrize("kind", ["device", "allocator", "busy", "share", "launch", "eval", "outputs"])
def test_raw_evidence_is_rechecked_instead_of_trusting_green_check_flags(evidence, kind):
    _, _, result, _, write = evidence
    run = result["runs"]["solution"][0]
    expected = {"device": "one T4", "allocator": "one T4", "busy": "idle at start",
                "share": "less than 5%", "launch": "no eligible original kernel",
                "eval": "more than 0.5%", "outputs": "official output comparison"}[kind]
    if kind == "device": run["metrics"]["cuda_device"] = "NVIDIA H100"
    elif kind == "allocator": run["metrics"]["alloc_conf"] = "backend:cudaMallocAsync"
    elif kind == "busy": run["gpu_start_mib"] = 1200
    elif kind == "share": run["metrics"]["own_kernel_ms"]["fixture_original_reduction"] = 1
    elif kind == "launch": run["metrics"]["triton_launches"].clear()
    elif kind == "eval": result["evals"]["solution"]["mean_loss"] = 2.2
    else: run["log"]["num_tokens"][0] += 1
    write()
    reject(assess(evidence), expected)


@pytest.mark.parametrize("invalid", [0, -1, True, float("inf"), float("nan")])
def test_nonpositive_nonfinite_and_boolean_timings_fail(evidence, invalid):
    _, _, result, _, write = evidence
    result["runs"]["solution"][0]["wall_s"] = invalid
    write(refresh_summary=False)
    report = assess(evidence)
    reject(report, "positive finite numbers")
    json.dumps(report, allow_nan=False)


def test_early_nonfinite_grad_norm_is_not_a_new_gate(evidence):
    _, _, result, _, write = evidence
    for runs in result["runs"].values():
        for run in runs:
            run["log"]["grad_norm"] = [float("inf"), *([1.] * 63)]
    write()
    assert assess(evidence)["target_met_under_official_public_protocol"] is True


def test_genuine_early_only_kernel_contributes_zero_instead_of_invalidating_share(evidence):
    repo, _, result, checks, write = evidence
    path = repo / "solution/kernel.py"
    source = path.read_text()
    path.write_text(source + "\n" + source[source.index("@triton.jit"):].replace(
        "fixture_original_reduction", "fixture_early_only_reduction"))
    checks["solution_sha256"] = verify._official_helpers(repo).solution_sha256(repo / "solution")
    for run in result["runs"]["solution"]:
        run["metrics"]["triton_launches"]["fixture_early_only_reduction"] = 1
        run["metrics"]["triton_files"]["fixture_early_only_reduction"] = "solution/kernel.py"
    write()
    assert assess(evidence)["target_met_under_official_public_protocol"] is True


def test_protected_source_tamper_is_rejected_before_import(evidence):
    repo, *_ = evidence
    with (repo / "harness/common.py").open("a") as f:
        f.write("\nraise AssertionError('must never execute this changed file')\n")
    reject(assess(evidence), "protected source differs")


def test_optional_readiness_observation_is_bound_to_source_and_result(evidence):
    repo, path, _, checks, _ = evidence
    ready = {"solution_sha256": checks["solution_sha256"], "status": "public_official_checks_passed",
             "protected_source": {"ok": True}, "official_checks_passed": True,
             "official_result": str(path), "source_kind": "local_snapshot"}
    ready_path = repo / "readiness.json"
    ready_path.write_text(json.dumps(ready))
    report = assess(evidence, readiness_report=ready_path)
    assert report["target_met_under_official_public_protocol"] is True, report
    assert report["provenance"]["readiness"]["valid"] is True
    assert report["provenance"]["per_run_source_hashes_recorded"] is False
    ready["solution_sha256"] = "0" * 64
    ready_path.write_text(json.dumps(ready))
    reject(assess(evidence, readiness_report=ready_path), "readiness report does not match")


def test_cli_writes_and_prints_same_verdict_without_gpu(evidence, monkeypatch, capsys):
    repo, path, *_ = evidence
    monkeypatch.setattr(verify, "REPO", repo)
    output = repo / "runs/final/target_verdict.json"
    assert verify.main([str(path), "--record-seconds", "33", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text())


def test_cli_refuses_to_overwrite_its_evidence(evidence, monkeypatch, capsys):
    repo, path, *_ = evidence
    monkeypatch.setattr(verify, "REPO", repo)
    before = path.read_bytes()
    assert verify.main([str(path), "--output", str(path)]) == 1
    assert path.read_bytes() == before
    assert "must not replace" in capsys.readouterr().out
