"""Reproduce the 576-case CPU/interpreter exceptional-row head audit.

Run outside the official measured process, from the repository root:
    TRITON_INTERPRET=1 OMP_NUM_THREADS=1 python tools/check_exceptional_rows.py

This is the original executed audit made reproducible, not additional coverage.
Every tensor stays on CPU. The tool does not install the solution, run inline
PTX, measure performance, or claim physical CUDA or official-harness evidence.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import warnings


SCENARIO_AXES = {
    "kind": [
        "finite", "hidden_nan", "hidden_posinf", "hidden_neginf",
        "weight_nan", "weight_posinf", "weight_neginf", "negative_inf_only",
    ],
    "all_ignored": [False, True],
    "upstream": ["0", "1", "65536", "+inf", "-inf", "nan"],
    "denominator": [None, 3, 0],
    "sparse": [False, True],
}
TOLERANCES = {"rtol": 0.005, "atol": 0.02, "equal_nan": True}
FIELDS = ("loss", "token_loss", "entropy", "predictions", "gradient")


def _json_values(value):
    """Encode exceptional tensor values in portable, strict JSON."""
    if isinstance(value, list):
        return [_json_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
    return value


def _source_hashes(repo):
    return {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in ("solution/streamed_head.py", "tools/check_exceptional_rows.py")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("validation/exceptional_rows.json"))
    args = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") != "1":
        parser.error("set TRITON_INTERPRET=1 before starting Python; this audit executes only on CPU")

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    import torch
    import torch.nn.functional as F
    import triton
    from solution.streamed_head import streamed_linear_ce

    def reference(hidden, weight, labels, denominator):
        scores = F.linear(hidden.half(), weight.half()).float()
        targets = F.pad(labels, (0, 1), value=-100)[:, 1:]
        token_loss = F.cross_entropy(scores.flatten(0, 1), targets.flatten(), reduction="none")
        divisor = ((targets != -100).sum() if denominator is None
                   else torch.as_tensor(denominator, device="cpu"))
        logp = scores.log_softmax(-1)
        return (token_loss.sum() / divisor, token_loss.reshape_as(labels),
                -(logp.exp() * logp).sum(-1), scores.argmax(-1))

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    hashes_before = _source_hashes(repo)
    failures = []
    attempted_cases = completed_cases = field_comparisons = failed_cases = 0
    axes = tuple(SCENARIO_AXES)
    # Preserve the exact fixture values, scenario order, projection, chunk
    # size, native reference, backward scale, and allclose tolerances used by
    # the original 576-case inline audit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for values in itertools.product(*(SCENARIO_AXES[axis] for axis in axes)):
            scenario = dict(zip(axes, values))
            kind, all_ignored, upstream_text, denominator, sparse = values
            upstream = float(upstream_text)
            attempted_cases += 1
            previous_failures = len(failures)
            try:
                hidden = torch.tensor([[[-1., 1.], [-2., 1.], [-3., 1.]]], device="cpu")
                weight = torch.tensor([[1., 0.], [0., 1.], [1., -1.]], device="cpu")
                nonfinite = {"nan": float("nan"), "posinf": float("inf"), "neginf": -float("inf")}
                if kind.startswith("hidden_"):
                    hidden[0, 1, 0] = nonfinite[kind.split("_")[-1]]
                elif kind.startswith("weight_"):
                    weight[0, 0] = nonfinite[kind.split("_")[-1]]
                elif kind == "negative_inf_only":
                    weight[0, 0] = float("inf")
                hidden.requires_grad_()
                reference_hidden = hidden.detach().clone().requires_grad_()
                labels = (torch.full((1, 3), -100, dtype=torch.long, device="cpu") if all_ignored
                          else torch.tensor([[-100, 1, -100]], device="cpu"))
                actual = streamed_linear_ce(
                    hidden, weight, labels, denominator, chunk_size=2, sparse_backward=sparse,
                )
                expected = reference(reference_hidden, weight, labels, denominator)
                (actual.loss * upstream).backward()
                (expected[0] * upstream).backward()
                completed_cases += 1
                observations = (
                    ("loss", actual.loss, expected[0]),
                    ("token_loss", actual.token_loss, expected[1]),
                    ("entropy", actual.entropy, expected[2]),
                    ("predictions", actual.predictions, expected[3]),
                    ("gradient", hidden.grad, reference_hidden.grad),
                )
                for name, observed, wanted in observations:
                    field_comparisons += 1
                    if not torch.allclose(observed, wanted, **TOLERANCES):
                        failures.append({
                            "scenario": scenario, "field": name,
                            "actual": _json_values(observed.detach().tolist()),
                            "expected": _json_values(wanted.detach().tolist()),
                        })
            except Exception as error:
                failures.append({"scenario": scenario, "field": "execution",
                                 "exception": type(error).__name__, "message": str(error)})
            if len(failures) != previous_failures:
                failed_cases += 1

    hashes_after = _source_hashes(repo)
    source_changes = [name for name in hashes_before if hashes_before[name] != hashes_after[name]]
    planned_cases = math.prod(len(values) for values in SCENARIO_AXES.values())
    passed = not failures and not source_changes and completed_cases == planned_cases
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "audit_wall_seconds": time.perf_counter() - started,
        "scope": "CPU native PyTorch reference and Triton interpreter arithmetic only; no physical CUDA, PTX execution, performance measurement, or official training run.",
        "physical_gpu_execution": False,
        "device": "cpu",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torch_git": torch.version.git_version, "triton": triton.__version__},
        "environment": {"TRITON_INTERPRET": os.environ.get("TRITON_INTERPRET"),
                        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                        "torch_num_threads": torch.get_num_threads()},
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "source_changes_during_run": source_changes,
        "scenario_axes": SCENARIO_AXES,
        "fixture": {
            "hidden": [[[-1., 1.], [-2., 1.], [-3., 1.]]],
            "weight": [[1., 0.], [0., 1.], [1., -1.]],
            "mixed_labels": [[-100, 1, -100]],
            "chunk_size": 2, "projection": "torch",
            "note": "negative_inf_only intentionally repeats the weight_posinf fixture from the original audit: negative hidden values yield one -Inf logit and finite alternatives, while zero ignored dlogits times Inf can make dH NaN. The 576 count is an execution count, not 576 unique input fixtures.",
        },
        "checks": {"comparison": "torch.allclose", "fields": list(FIELDS),
                   "tolerances_for_every_field": TOLERANCES,
                   "nonfinite_values_in_failure_arrays": "nan, +inf and -inf strings"},
        "tested_counts": {"planned_cases": planned_cases, "attempted_cases": attempted_cases,
                          "completed_cases": completed_cases, "field_comparisons": field_comparisons,
                          "failed_cases": failed_cases, "failed_fields_or_exceptions": len(failures)},
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], **report["tested_counts"],
                      "report": str(args.output.resolve()), "physical_gpu_execution": False}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
