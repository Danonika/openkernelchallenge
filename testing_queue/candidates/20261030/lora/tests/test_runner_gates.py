"""Synthetic malformed/partial output gate tests; no measured result is generated."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('prepared_runner',root/'run_prepared.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def synthetic(tmp_path):
    adapter=tmp_path/'synthetic_adapter';adapter.mkdir()
    for name in ('adapter_model.safetensors','adapter_config.json'):(adapter/name).write_text('synthetic-not-real-weights')
    value={'runs':{},'evals':{},'packages_changed':[],'summary':{'ratios':{'speedup':1.0}}}
    for side in ('base','solution'):
        value['runs'][side]=[]
        for i in range(2):
            value['runs'][side].append({'run':f'{side}_{i+1}','exit_code':0,'global_step':64,'wall_s':100.,'gpu_peak_mib':1000,
                'adapter':str(adapter),'metrics':{'exit_code':0,'steps':64,'timed_steps':62,'timed_loop_s':10.,
                'own_kernel_ms':{runner.KERNEL:100.},'triton_launches':{runner.KERNEL:64},
                'triton_files':{runner.KERNEL:'solution/epilogue.py'},'triton_watch_error':None}})
        value['evals'][side]={'exit_code':0,'mean_loss':1.0}
    (tmp_path/'result.json').write_text(json.dumps(value))
    return value


def test_gate_does_not_assign_score_or_admission_even_complete_synthetic(tmp_path):
    value=synthetic(tmp_path);a=runner.assess(value,tmp_path)
    assert a['status']=='COMPLETE_MEASUREMENT'
    assert a['score'] is None and a['admission']=='UNKNOWN' and not a['own_share_gate']


def test_each_nested_failure_suppresses_comparison_and_share(tmp_path):
    value=synthetic(tmp_path)
    mutations=[
        lambda v:v['runs']['solution'][0].update(exit_code=-9),
        lambda v:v['runs']['solution'][0]['metrics'].update(exit_code=1),
        lambda v:v['runs']['solution'][0]['metrics'].update(steps=63),
        lambda v:v['runs']['solution'][0].update(global_step=63),
        lambda v:v['evals']['solution'].update(exit_code=1),
        lambda v:v['evals']['solution'].update(mean_loss=float('nan')),
        lambda v:v['runs']['solution'][0]['metrics'].update(own_kernel_ms={}),
        lambda v:v['runs']['solution'][0].update(gpu_peak_mib=0),
        lambda v:v.update(packages_changed=['changed']),
    ]
    for mutate in mutations:
        case=copy.deepcopy(value);mutate(case);a=runner.assess(case,tmp_path)
        assert a['status']=='FAILED_OR_INCOMPLETE'
        assert a['score'] is None and a['admission']=='UNKNOWN'
        assert a['own_share_min'] is None and a['ratios'] is None
