"""Read-only validation of downloaded official one-repeat baseline artifacts.

Does not train, change the official harness, access credentials, or claim admission.
Run: python validate_baseline.py /path/to/kaggle/downloads [--out evidence.json]
"""
import argparse
import json
import math
from pathlib import Path

PINS = {"soup-cli": "0.77.0", "transformers": "5.16.1", "peft": "0.20.0",
        "trl": "0.29.1", "bitsandbytes": "0.50.2", "accelerate": "1.14.0"}
COMMIT = "c7c2fcd15972ce8b898d8d3ff732ee192c817ceb"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(result, environment=None, wrapper=None):
    failures = []
    def check(condition, message):
        if not condition:
            failures.append(message)

    task = result.get("task", {})
    runs = result.get("runs", {}).get("base", [])
    evaluation = result.get("evals", {}).get("base", {})
    summary = result.get("summary", {}).get("base", {})
    versions = result.get("versions", {})
    check(task.get("only") == "base", "Expected base-only run")
    check(task.get("repeats") == 1, "Expected exactly one provisional repeat")
    check(task.get("seed") == 1, "Expected public seed 1")
    check(len(runs) == 1, "Expected exactly one recorded base run")
    check(not result.get("runs", {}).get("solution"), "Unexpected solution run")
    check(not result.get("packages_changed"), "Packages changed during measurement")
    for name, expected in PINS.items():
        check(versions.get(name) == expected, f"Version mismatch {name}: {versions.get(name)!r}")
    if environment is not None:
        check(environment.get("starter_commit") == COMMIT, "Unexpected official starter commit")
    if wrapper is not None:
        check(wrapper.get("status") == "completed", "Outer wrapper did not complete")
    check(summary.get("ok") is True, "Base summary not successful")
    check(evaluation.get("exit_code") == 0, "Evaluation exit is not 0")
    check(finite(evaluation.get("mean_loss")) and evaluation.get("mean_loss", 0) > 0,
          "Evaluation loss absent or invalid")
    check(evaluation.get("rows") == 256, "Expected 256 public evaluation rows")
    check(finite(evaluation.get("tokens")) and evaluation.get("tokens", 0) > 0,
          "Evaluation answer-token count absent or invalid")
    measurements = []
    for run in runs:
        name, m = run.get("run", "unknown"), run.get("metrics", {})
        check(run.get("exit_code") == 0, f"{name}: training exit is not 0")
        check(m.get("exit_code") == 0, f"{name}: metrics exit is not 0")
        check(m.get("steps") == 64, f"{name}: expected 64 training steps")
        check(run.get("global_step") == 64, f"{name}: saved trainer global step is not 64")
        check(m.get("solution") is False, f"{name}: unexpectedly applied solution")
        check("T4" in str(m.get("cuda_device")), f"{name}: actual CUDA hardware is not T4")
        check(m.get("device_count") == 1 and m.get("visible_devices") == "0",
              f"{name}: expected exactly one visible CUDA device, GPU 0")
        check(m.get("alloc_conf") == "expandable_segments:True", f"{name}: allocator mismatch")
        check(not m.get("banned_modules_loaded"), f"{name}: banned runtime modules loaded")
        start, peak, reserved = (run.get("gpu_start_mib"), run.get("gpu_peak_mib"),
                                 m.get("torch_max_reserved_mib"))
        check(finite(start) and start <= 1000, f"{name}: GPU idle start not established")
        check(finite(peak) and peak > 0, f"{name}: GPU memory peak absent or invalid")
        check(finite(peak) and finite(reserved) and reserved > 0 and peak >= .97 * reserved,
              f"{name}: nvidia-smi / reserved-memory agreement not established")
        check(finite(run.get("wall_s")) and run.get("wall_s", 0) > 0,
              f"{name}: wall time absent or invalid")
        check(bool(run.get("adapter_tensors")), f"{name}: saved adapter tensors absent")
        check(bool(run.get("adapter_config")), f"{name}: saved adapter config absent")
        check(run.get("rewind") is not None, f"{name}: rewind output absent")
        for key in ("loss", "num_tokens", "entropy", "mean_token_accuracy", "grad_norm"):
            values = run.get("log", {}).get(key, [])
            check(len(values) == 64 and all(finite(v) for v in values),
                  f"{name}: {key} log missing or incomplete")
        measurements.append({"run": name, "wall_s": run.get("wall_s"),
                             "train_loop_s": m.get("train_loop_s"), "gpu_peak_mib": peak,
                             "gpu_start_mib": start, "torch_max_reserved_mib": reserved,
                             "cuda_device": m.get("cuda_device"), "steps": m.get("steps")})
    return {"status": "provisional_baseline_validated" if not failures else "validation_failed",
            "full_admission": False, "score_points_claimed": None,
            "scope": "Official public base-only single-repeat smoke; not a solution comparison",
            "failures": failures, "measurements": measurements,
            "eval_loss": evaluation.get("mean_loss"), "eval_rows": evaluation.get("rows"),
            "eval_tokens": evaluation.get("tokens"), "versions": versions,
            "warning": "Wrapper completion or Kaggle COMPLETE alone does not prove training/evaluation succeeded."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    def find_json(name):
        paths = list(args.root.rglob(name))
        if len(paths) > 1:
            raise ValueError(f"Ambiguous {name}: {len(paths)} paths")
        return json.loads(paths[0].read_text()) if paths else None
    result = find_json("result.json")
    wrapper = find_json("baseline_session_status.json")
    if result is None:
        evidence = {"status": "no_result_artifact", "full_admission": False,
                    "score_points_claimed": None, "wrapper": wrapper,
                    "reason": "No official result.json; training and evaluation unestablished"}
    else:
        evidence = validate(result, find_json("baseline_environment.json"), wrapper)
    rendered = json.dumps(evidence, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
