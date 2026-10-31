"""Explicit candidate arithmetic stage, separate from native approximation.

The native CUDA sparse witness disproved an earlier CPU-derived assumption.
No native hidden accumulator is exposed or inferred here. Tolerances are the
unchanged predeclared engineering gate, not a fixed-task quality decision.
"""
import torch
from .fixtures import exact, engineering_close


def verify_bias_stage(actual, native, fp32_reference_without_bias, bias,
                      native_materialized_without_bias):
    if fp32_reference_without_bias.dtype != torch.float32 or bias.dtype != torch.float16:
        raise RuntimeError('Explicit reference stage dtype mismatch')
    intended = (fp32_reference_without_bias + bias.float()).half()
    postround = fp32_reference_without_bias.half() + bias
    if torch.equal(intended, postround):
        raise RuntimeError('Candidate staged-bias negative witness became vacuous')
    # Verify the candidate's intended arithmetic, not an assumed CUDA epilogue.
    exact(actual, intended)
    numeric = engineering_close(actual, native)
    return {
        'candidate_fp32_bias_before_half_exact': True,
        'candidate_postround_negative_witness_nonvacuous': True,
        'native_comparison': numeric,
        'observations': {
            'native_equals_candidate': bool(torch.equal(native, actual)),
            'native_equals_explicit_fp32_bias_before_half': bool(torch.equal(native, intended)),
            'native_equals_materialized_half_then_bias': bool(torch.equal(native, native_materialized_without_bias + bias)),
            'native_unique': torch.unique(native.detach()).float().cpu().tolist(),
            'candidate_unique': torch.unique(actual.detach()).float().cpu().tolist(),
            'fp32_reference_without_bias_unique': torch.unique(fp32_reference_without_bias.detach()).cpu().tolist(),
            'intended_unique': torch.unique(intended.detach()).float().cpu().tolist(),
            'postround_unique': torch.unique(postround.detach()).float().cpu().tolist(),
        },
        'native_internal_accumulator_observed': False,
        'task_quality': 'UNKNOWN',
    }
