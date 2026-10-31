"""Unchanged original project gate helpers from verify_ce.py82d6635b.
No third-party kernel code; no execution on import.
"""
import math
import torch

GATES = {
    "status": "declared_before_any_GPU_execution",
    "loss_atol": 1e-6, "loss_rtol": 1e-4,
    "nearzero_threshold": 1e-3, "nearzero_relative_limit": 0.01,
    "exact_zero_reference_requires_zero": True,
    "gradient_atol": 2.0 ** -24, "gradient_rtol": 2e-3,
    "gradient_relative_L2_max": 1e-3,
    "same_nonfinite_positions": True,
    "meaning": "Engineering diagnostics only; not official task or admission tolerances.",
}


def _number(value):
    value = float(value)
    if math.isfinite(value):
        return value
    return "NaN" if math.isnan(value) else "+Inf" if value > 0 else "-Inf"


def _scalar_check(actual, reference):
    a, r = float(actual), float(reference)
    if not math.isfinite(a) or not math.isfinite(r):
        same = (math.isnan(a) and math.isnan(r)) or a == r
        return {"passed": same, "actual": _number(a), "reference": _number(r), "absolute": None, "relative": None}
    absolute = abs(a - r)
    relative = absolute / max(abs(r), 1e-12)
    passed = absolute <= GATES["loss_atol"] + GATES["loss_rtol"] * abs(r)
    if 0 < abs(r) < GATES["nearzero_threshold"]:
        passed = passed and relative <= GATES["nearzero_relative_limit"]
    if r == 0:
        passed = passed and a == 0
    return {"passed": passed, "actual": a, "reference": r, "absolute": absolute, "relative": relative,
            "exact": a == r}


def _gradient_check(actual, reference):
    # Bounded chunks: never allocate an additional full-logits boolean/double array.
    a, r = actual.reshape(-1), reference.reshape(-1)
    class_bad = finite_different = outside = signed_zero_different = 0
    max_abs = max_relative = sq_error = sq_reference = 0.0
    for start in range(0, r.numel(), 262144):
        aa, rr = a[start:start + 262144], r[start:start + 262144]
        mismatched = ((aa.isnan() != rr.isnan()) | (aa.isposinf() != rr.isposinf())
                      | (aa.isneginf() != rr.isneginf()))
        class_bad += int(mismatched.sum().item())
        finite = aa.isfinite() & rr.isfinite()
        zeros = (aa == 0) & (rr == 0)
        signed_zero_different += int((zeros & (aa.signbit() != rr.signbit())).sum().item())
        af, rf = aa[finite].double(), rr[finite].double()
        delta = (af - rf).abs()
        finite_different += int((af != rf).sum().item())
        outside += int((delta > GATES["gradient_atol"] + GATES["gradient_rtol"] * rf.abs()).sum().item())
        if delta.numel():
            max_abs = max(max_abs, float(delta.max().item()))
            max_relative = max(max_relative, float((delta / rf.abs().clamp_min(1e-12)).max().item()))
            sq_error += float(delta.square().sum().item())
            sq_reference += float(rf.square().sum().item())
    relative_l2 = math.sqrt(sq_error / sq_reference) if sq_reference else (0.0 if not sq_error else float("inf"))
    return {"passed": class_bad == 0 and outside == 0 and relative_l2 <= GATES["gradient_relative_L2_max"],
            "nonfinite_class_mismatches": class_bad, "finite_exact_mismatches": finite_different,
            "signed_zero_mismatches": signed_zero_different,
            "outside_element_gate": outside, "max_absolute": max_abs, "max_relative_floor_1e_minus12": max_relative,
            "relative_L2": _number(relative_l2)}

