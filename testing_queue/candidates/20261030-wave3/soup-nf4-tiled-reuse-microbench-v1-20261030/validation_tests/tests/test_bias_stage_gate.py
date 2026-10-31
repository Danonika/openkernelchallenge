"""CPU test of gate semantics only; target values are source-bound separately."""
import hashlib, json, sys, zipfile
from pathlib import Path
import pytest, torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solution.bias_stage_gate import verify_bias_stage


def inputs():
    before=torch.full((2,3),1.+float(torch.tensor(.0004,dtype=torch.float16)),dtype=torch.float32)
    bias=torch.full((3,),.0002,dtype=torch.float16)
    candidate=(before+bias.float()).half()
    native=torch.ones((2,3),dtype=torch.float16)
    return candidate,native,before,bias,native.clone()


def test_observed_native_mismatch_does_not_invalidate_intended_candidate_stage():
    report=verify_bias_stage(*inputs())
    assert report['candidate_fp32_bias_before_half_exact']
    assert report['candidate_postround_negative_witness_nonvacuous']
    assert report['observations']['native_equals_candidate'] is False
    assert report['observations']['native_equals_materialized_half_then_bias'] is True
    assert report['native_internal_accumulator_observed'] is False
    assert report['native_comparison']['atol']==.003 and report['native_comparison']['rtol']==.002
    assert report['task_quality']=='UNKNOWN' and not torch.cuda.is_initialized()


@pytest.mark.parametrize('failure',['postround_candidate','native_far','native_nan','vacuous','fp16_reference'])
def test_revised_gate_remains_nonvacuous_and_failclosed(failure):
    values=list(inputs())
    if failure=='postround_candidate':values[0]=values[2].half()+values[3]
    if failure=='native_far':values[1].fill_(2.)
    if failure=='native_nan':values[1].fill_(float('nan'))
    if failure=='vacuous':values[3].zero_()
    if failure=='fp16_reference':values[2]=values[2].half()
    with pytest.raises((RuntimeError,AssertionError)):verify_bias_stage(*values)


def test_reference_kernel_and_predeclared_tolerances_unchanged_from_frozen_v3():
    frozen=Path('/workspace/shared/soup_nf4_fused_forward_2202_v3/solution')
    for name in ('fixtures.py','nf4_format.py','operand_probe.py','native_source_manifest.json'):
        assert (ROOT/'solution'/name).read_bytes()==(frozen/name).read_bytes()
    assert (ROOT/'solution/reference_kernel.py').read_bytes()==(frozen/'kernel.py').read_bytes()
    original_normalization=Path('/workspace/shared/soup_nf4_parameter_bias_research_v1/solution')
    for name in ('parameter_bias.py',):
        assert (ROOT/'solution'/name).read_bytes()==(original_normalization/name).read_bytes()


def test_revised_gate_provenance_is_bound_to_actual_target_not_cpu_predictions():
    provenance=json.loads((ROOT/'solution/bias_gate_provenance.json').read_text())
    p=Path('/workspace/shared/soup_server_jobs_20261030')/(provenance['job_id']+'-outputs.zip')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==provenance['output_zip_sha256']
    with zipfile.ZipFile(p) as z:
        raw=z.read(next(n for n in z.namelist() if n.endswith('/raw/preflight.json')))
        assert hashlib.sha256(raw).hexdigest()==provenance['diagnostic_report_sha256']
    report=json.loads(raw);obs=report['observations']
    assert report['status']=='COMPLETED_DIAGNOSTIC_ONLY' and not report['candidate_validated']
    assert obs['route']['counts']['custom']==2 and obs['inputs']['operand_exact']['passed']
    for name in ('native_with_bias','native_explicit_Flinear_with_bias'):
        assert obs['outputs'][name]['unique_values']==[1.0]
    for name in ('candidate_with_bias','fp32_acc_plus_bias_then_half'):
        assert obs['outputs'][name]['unique_values']==[1.0009765625]
    assert obs['comparisons']['candidate_vs_native_exact']['passed'] is False
    assert obs['comparisons']['candidate_vs_native_fixed_engineering']['passed'] is True
    assert provenance['native_internal_accumulator_not_observed'] is True


def test_module_gate_is_required_before_timing_and_reports_real_activation():
    preflight=(ROOT/'solution/web_preflight.py').read_text()
    driver=(ROOT/'microbench/driver.py').read_text()
    assert preflight.index('module_checks=module_run()')<preflight.index("return {'rows':rows")
    assert driver.index("read(run/'preflight.json').get('status')!='PASSED_PREFLIGHT_ONLY'")<driver.index("for method in status['methods_in_execution_order']")
    source=(ROOT/'solution/module_preflight.py').read_text()
    assert "g.COUNTS['custom']!=count+1" in source
    assert "len(rows)!=12" in source
    assert 'module(a)' in source and 'exact(x,y)' in source
    assert 'not captured fixed-task q/v activations' in source
