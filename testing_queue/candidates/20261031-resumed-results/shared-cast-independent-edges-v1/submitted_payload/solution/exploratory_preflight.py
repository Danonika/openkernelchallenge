"""Distinct exploratory policy over immutable strict-v3 fixture and thresholds.

Only explicitly listed finite LoRA gradient/nominal parameter-update numerical
comparisons are observations. Failed comparisons stay passed:false. Loss,
outputs, structure, sources, task switches, RNG, scaler, nonfinite classes,
route activation and all other checks remain blocking. This is not admission.
"""
import argparse,hashlib,json,math,traceback
from pathlib import Path
from . import web_preflight as strict

STRICT_HELPER_SHA256='6159dd11201aa412cdff8c7d562dfc1ce6c7f77ab07de23a9dd8099f8839e8c4'
STATUS='STRUCTURAL_CHECKS_COMPLETE_NUMERICAL_RESULTS_RECORDED'
PARAMETERS=tuple('base_model.model.model.layers.0.self_attn.'+projection+'.lora_'+side+'.default.weight'
                 for projection in ('q_proj','v_proj') for side in ('A','B'))
GRADIENT_NAMES=frozenset(group+'/'+name for group in ('scaled','unscaled') for name in PARAMETERS)
UPDATE_NAMES=frozenset('update/'+name for name in PARAMETERS)
POLICY={
 'scope':'EXPLORATORY_SINGLE_TASK_SCREEN_NOT_JURY_VALIDATION',
 'numerical_thresholds_unchanged':True,
 'warning_only_comparisons':{'gradients':sorted(GRADIENT_NAMES),'parameters':sorted(UPDATE_NAMES)},
 'blocking':'loss; entropy/accuracy/Rewind/gather outputs; token/order/argmax; shape/dtype/nonfinite classes; source/ownership/task/RNG/scaler/count/route; all unlisted comparisons',
 'nominal_fixture_optimizer_not_full_task_scheduler':True,
 'old_failures_relabelled':False,'score':None,'admission':'UNKNOWN'}


def finite_statistics(actual,expected):
    import torch
    finite=actual.isfinite()&expected.isfinite()
    a=actual[finite].double();b=expected[finite].double();delta=(a-b).abs()
    maxabs=float(delta.max()) if delta.numel() else 0.
    refmax=float(b.abs().max()) if b.numel() else 0.
    error_norm=float(delta.norm());reference_norm=float(b.norm())
    relative=error_norm/max(reference_norm,1e-30) if b.numel() else 0.
    values={'max_abs':maxabs,'reference_max_abs':refmax,'error_l2':error_norm,
            'reference_l2':reference_norm,'relative_l2':relative}
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError('Nonfinite derived numerical statistic; exploratory policy cannot waive it')
    return a,b,delta,values


def comparison_examples(a,b,mask,limit=8):
    return [{'finite_flat_index':int(i),'actual':float(a[i]),'reference':float(b[i])}
            for i in mask.nonzero().reshape(-1)[:limit]]


class ObservationPolicy:
    def __init__(self,progress):
        self.progress=progress;self.observations=[];self.original_metric=strict.metric;self.original_gradient=strict.gradient
    def remember(self,name,kind,passed,values,tolerances,examples=()):
        record={'name':name,'classification':kind,'passed':bool(passed),'blocking':False,
                'scale':self.progress.get('current_phase',{}).get('scale'),
                'values':values,'original_tolerances':tolerances,'failing_examples':list(examples)}
        self.observations.append(record)
        return record
    def gradient(self,actual,expected,name):
        if name not in GRADIENT_NAMES:raise RuntimeError('Unlisted gradient comparison is not warning-authorized: '+str(name))
        strict.classes(actual,expected,name)
        a,b,delta,values=finite_statistics(actual,expected)
        # Exactly the frozen strict-v3 gradient inequalities; no exception
        # interception and no threshold fitted to observed discrepancies.
        abs_limit=.005*values['reference_max_abs']+1e-7
        passed=values['relative_l2']<=.001 and values['max_abs']<=abs_limit
        return self.remember(name,'finite_lora_gradient_numeric',passed,values,
            {'relative_l2_max':.001,'max_abs_reference_multiplier':.005,'max_abs_atol':1e-7,'actual_max_abs_limit':abs_limit},
            comparison_examples(a,b,delta>abs_limit))
    def metric(self,actual,expected,name,rtol=1e-4,atol=1e-6):
        if name not in UPDATE_NAMES:
            # Loss, all output metrics, clip norm and every unlisted check
            # call the exact original blocker; no broad catch exists.
            return self.original_metric(actual,expected,name,rtol=rtol,atol=atol)
        if rtol!=.005 or atol!=1e-6:raise RuntimeError('Original update threshold changed')
        import torch
        strict.classes(actual,expected,name)
        actual=torch.as_tensor(actual,dtype=torch.float64);expected=torch.as_tensor(expected,dtype=torch.float64)
        strict.classes(actual,expected,name)
        a,b,delta,values=finite_statistics(actual,expected)
        # torch.testing.assert_close's finite elementwise asymmetric tolerance.
        failed=delta>atol+rtol*b.abs();passed=not bool(failed.any())
        values['violating_elements']=int(failed.sum());values['finite_elements']=a.numel()
        return self.remember(name,'finite_nominal_lora_parameter_update_numeric',passed,values,
            {'rtol':rtol,'atol':atol,'criterion':'abs(actual-reference)<=atol+rtol*abs(reference)'},
            comparison_examples(a,b,failed))
    def execute(self,save):
        if strict.metric is not self.original_metric or strict.gradient is not self.original_gradient:
            raise RuntimeError('Strict helper bindings changed before explicit policy install')
        strict.metric=self.metric;strict.gradient=self.gradient
        try:return strict.run(self.progress,save)
        finally:strict.metric=self.original_metric;strict.gradient=self.original_gradient


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args();out=Path(args.out)
    report={'status':'BLOCKED_EXPLORATORY_STRUCTURAL_PREFLIGHT','progress':{},'policy':POLICY,
            'actual_target_execution':True,'fixed_task_training':False,'score':None,'admission':'UNKNOWN'}
    policy=ObservationPolicy(report['progress']);report['engineering_numeric_observations']=policy.observations
    def save():
        pending=out.with_name(out.name+'.pending');pending.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');pending.replace(out)
    save()
    try:
        if hashlib.sha256(Path(strict.__file__).read_bytes()).hexdigest()!=STRICT_HELPER_SHA256:
            raise RuntimeError('Exact strict-v3 helper source binding changed')
        report['checks']=policy.execute(save)
        report['status']=STATUS
    except BaseException as error:
        # Terminal error reporting only; no exception is converted into success.
        report.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        report['engineering_numeric_failures']=sum(not x['passed'] for x in policy.observations)
        report['original_thresholds_changed']=False;report['numerical_pass_claim']=False;save()
    print(json.dumps({'status':report['status'],'out':str(out),'engineering_numeric_failures':report['engineering_numeric_failures']}),flush=True)
    raise SystemExit(0 if report['status']==STATUS else 1)
if __name__=='__main__':main()
