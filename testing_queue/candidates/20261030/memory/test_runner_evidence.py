"""Synthetic fail-closed reporting tests; never launches a process or GPU."""
import json
from pathlib import Path
import tempfile
from notebook_runner import summarize_evidence


def good_run(side, number):
    return {'run': f'{side}_{number}', 'exit_code': 0, 'wall_s': 100.0,
            'gpu_peak_mib': 12000, 'global_step': 64,
            'adapter_config': {'r': 16}, 'adapter_tensors': {'lora_A': ['F32', [16, 64]]},
            'metrics': {'steps': 64, 'exit_code': 0, 'timed_steps': 62,
                        'timed_loop_s': 80.0, 'own_kernel_ms': {'soup_mem001_restore_half_to_fp32': 0.01}}}


def main():
    results=[]
    for case in ('complete_component', 'native_timeout', 'missing_eval', 'missing_final_routes', 'wrapper_no_result'):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'runs').mkdir()
            raw={'packages_changed': [], 'runs': {s: [good_run(s, i) for i in (1,2)] for s in ('base','solution')},
                 'evals': {s: {'exit_code':0,'mean_loss':2.0} for s in ('base','solution')},
                 'summary': {'ratios': {'speedup':1.0,'memory_ratio':1.0}}}
            for i in (1,2):
                path=root/'runs'/f'solution_{i}';path.mkdir()
                log='SOUP_MEMORY_VS1_PROGRESS '+json.dumps({'cumulative_lower_bound':{'restores':4},'completed_training':False})+'\n'
                if case!='missing_final_routes':
                    log+='SOUP_MEMORY_VS1_RUNTIME '+json.dumps({'encoded_operands':100,'restores':100})+'\n'
                (path/'train.log').write_text(log)
            if case=='native_timeout': raw['runs']['solution'][1]['exit_code']=-9
            if case=='missing_eval': raw['evals'].pop('solution')
            if case!='wrapper_no_result': (root/'runs/result.json').write_text(json.dumps(raw))
            report=summarize_evidence(root/'runs/result.json',root)
            assert report['score'] is None and report['admission']=='UNKNOWN'
            if case=='complete_component': assert report['status']=='COMPLETE_COMPONENT_MEASUREMENT'
            else: assert report['status']=='FAILED_OR_INCOMPLETE' and report['ratios'] is None
            assert len(report['timeout_surviving_logs'])==2
            results.append({'case':case,'passed':True,'status':report['status']})
    result={'scope':'synthetic reporter fixtures only','passed':True,'cases':results,'gpu_executed':False}
    Path(__file__).with_name('runner_fixture_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
