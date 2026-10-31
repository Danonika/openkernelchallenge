"""Bounded component-only target gate; preserves partial case records on failure."""
import argparse,hashlib,inspect,json,os,traceback
from pathlib import Path
EXPECTED_QWEN_BLOB='b4fed98fafae8e4ee39be3b8cd766b1c85b079cd'

def run(report):
    if os.environ.get('TRITON_INTERPRET'):raise RuntimeError('Interpreter is not target execution')
    import torch,triton,transformers
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    from .residual_norm import fused_residual_norm,supported
    from .contracts import compare,native,fixture,extreme_diagnostics,H
    assert torch.__version__=='2.11.0+cu128' and triton.__version__=='3.6.0' and transformers.__version__=='5.16.1'
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='Tesla T4' and torch.cuda.get_device_capability(0)==(7,5)
    raw=Path(inspect.getfile(qwen)).read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    assert blob==EXPECTED_QWEN_BLOB
    report.update(gpu_execution_attempted=True,native_source_blob=blob,ordinary=[],structural=[],extreme_diagnostics=[])
    # The native eager expression is checked against the actual installed class.
    a,b,w=fixture(7,device='cuda');module=qwen.Qwen2RMSNorm(H).cuda();module.weight=torch.nn.Parameter(w,requires_grad=False)
    expected=module(a+b);ref=native(a,b,w)[0]
    assert torch.equal(expected,ref)
    up=torch.ones_like(ref)*128.
    ga=torch.autograd.grad(expected,(a,b),up);gb=torch.autograd.grad(ref,(a,b),up)
    assert all(torch.equal(x,y) for x,y in zip(ga,gb))
    report['structural'].append('native_expression_matches_installed_qwen_forward_backward')
    for rows in (1,7,2048):
        for scale in (.01,1.,10.):
            report['current_case']={'rows':rows,'activation_scale':scale,'upstream_scale':65536.}
            report['ordinary'].append(compare(fused_residual_norm,rows,scale,65536.,device='cuda'))
    for alias in ('independent','same','overlap'):
        for use in ('norm_only','residual_only','both'):
            report['current_case']={'alias':alias,'use':use}
            report['ordinary'].append(compare(fused_residual_norm,7,1.,128.,device='cuda',alias=alias,use=use))
    report['ordinary'].append(compare(fused_residual_norm,7,1.,1.,device='cuda',only_a=True))
    report['ordinary'].append(compare(fused_residual_norm,7,0.,1.,device='cuda'))
    a,b,w=fixture(7,device='cuda')
    assert supported(a,b,w,1e-6)
    for triple,eps in (((a.half(),b.half(),w),1e-6),((a,b,w.clone().requires_grad_()),1e-6),((a[:,::2],b[:,::2],w[:768]),1e-6),((a,b,w),1e-5)):
        assert not supported(*triple,eps)
        try:fused_residual_norm(*triple,eps)
        except ValueError:pass
        else:raise AssertionError('Unsupported contract accepted')
    report['structural'].append('dtype_trainable_weight_layout_epsilon_negatives_rejected')
    # Saved outputs and weight retain native version-counter protection.
    for target in ('residual','weight'):
        a,b,w=fixture(1,device='cuda');y,z=fused_residual_norm(a,b,w)
        with torch.no_grad():(z if target=='residual' else w).add_(1)
        try:torch.autograd.grad(y,(a,b),torch.ones_like(y))
        except RuntimeError as exc:assert 'modified by an inplace operation' in str(exc)
        else:raise AssertionError('Saved '+target+' mutation not detected')
    report['structural'].append('saved_residual_and_weight_mutation_rejected')
    saved=[]
    a,b,w=fixture(1,device='cuda')
    with torch.no_grad(),torch.autograd.graph.saved_tensors_hooks(lambda x:saved.append(x) or x,lambda x:x):y,z=fused_residual_norm(a,b,w)
    assert not saved and not y.requires_grad and not z.requires_grad
    report['structural'].append('no_grad_saves_nothing')
    report['current_case']='extreme_diagnostics_not_universal_equivalence_gate'
    report['extreme_diagnostics']=extreme_diagnostics(fused_residual_norm,'cuda')
    # Unused versus connected-zero norm paths must preserve nonfinite classes.
    rows=report['extreme_diagnostics']
    for index in (1,4):assert rows[index]['all_one'] and rows[index]['nan']==0
    for index in (2,5):assert rows[index]['nan']==H
    # The cancellation witness is reported, never concealed or used as an
    # assertion of full native-graph equivalence. No production integration yet.
    # New mixed-input cases preserve the native FP32 -> FP16 add adjoint edge.
    from .branch_capture import compare as mixed_close
    report['mixed']=[]
    for rows_count in (1,7,2048):
        for activation_scale in (.01,1.,10.):
            for upstream_scale in (128.,65536.):
                a,b,w=fixture(rows_count,activation_scale,device='cuda')
                b=b.detach().half().requires_grad_();dy=torch.full_like(a,.7)*upstream_scale;dz=torch.full_like(a,-.2)*upstream_scale
                report['current_case']={'mixed_rows':rows_count,'activation_scale':activation_scale,'upstream_scale':upstream_scale}
                y,z=fused_residual_norm(a,b,w);ey,ez=native(a,b,w)
                ga,gb=torch.autograd.grad((y,z),(a,b),(dy,dz));na,nb=torch.autograd.grad((ey,ez),(a,b),(dy,dz))
                assert torch.equal(z,ez) and ga.dtype==torch.float32 and gb.dtype==torch.float16
                report['mixed'].append({'rows':rows_count,'activation_scale':activation_scale,'upstream_scale':upstream_scale,
                    'forward':mixed_close(y,ey,forward=True),'a_gradient':mixed_close(ga,na,scale=upstream_scale),
                    'b_gradient':mixed_close(gb,nb,scale=upstream_scale),'native_input_output_adjoint_dtypes':True,'residual_exact':True})
    report['current_case']='actual_tiny_native_qwen_peft_amp_branch'
    from .native_branch_fixture import run as native_branch
    report['native_branch']=native_branch(fused_residual_norm)
    torch.cuda.synchronize();report['gpu_executed']=True

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    report={'status':'FAILED_PREFLIGHT_ONLY','scope':'MICROBENCH_ONLY','training_executed':False,'complete_fixed_task':False,'score':None,'admission':'UNKNOWN','gpu_executed':False}
    try:run(report);report['status']='PASSED_PREFLIGHT_ONLY'
    except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc())
    Path(args.out).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'ordinary_cases':len(report.get('ordinary',[])),'output':args.out}))
    raise SystemExit(report['status']!='PASSED_PREFLIGHT_ONLY')
if __name__=='__main__':main()
