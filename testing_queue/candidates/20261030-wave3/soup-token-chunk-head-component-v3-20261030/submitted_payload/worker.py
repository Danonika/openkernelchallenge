"""Bounded standalone head component: no model load, installer, optimizer or training.
Native references are the pinned ForCausalLMLoss, TRL entropy and Soup row_losses.
Only isolated head+required-metric+adjoint CUDA timings and allocator peaks count.
"""
import argparse,collections,gc,hashlib,inspect,json,os,statistics,time,traceback,weakref
from types import SimpleNamespace
from pathlib import Path


def write(path,value):
    tmp=path.with_name(path.name+'.pending');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['numerical','native','chunk'],required=True);p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);assert not out.exists()
    report=dict(status='FAILED',scope='HEAD_COMPONENT_PREFLIGHT_ONLY',method=args.method,training_executed=False,complete_fixed_task=False,score=None,admission='UNKNOWN',original_triton_credit=False,rows=[])
    write(out,report)
    try:
        import torch,triton,transformers
        import torch.nn.functional as F
        from transformers.loss.loss_utils import ForCausalLMLoss
        from trl.trainer.utils import entropy_from_logits
        from soup_cli.trainer.rewind_hf import row_losses
        from solution.token_chunk_head import token_chunk_boundary
        from shared_consumer import bridge_state
        from shared_consumer.consumer import streaming_sink,publish_metrics,native_order_row_losses
        from accelerate.utils.operations import convert_to_fp32
        from transformers.modeling_outputs import CausalLMOutputWithPast
        assert torch.__version__=='2.11.0+cu128' and triton.__version__=='3.6.0' and transformers.__version__=='5.16.1'
        assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='Tesla T4' and torch.cuda.get_device_capability(0)==(7,5)
        assert not os.environ.get('TRITON_INTERPRET')
        expected=json.loads((Path(__file__).parent/'native_source_manifest.json').read_text())
        for fn in (ForCausalLMLoss,entropy_from_logits,row_losses,convert_to_fp32):
            path=Path(inspect.getfile(fn));assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[path.name]
        flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        report.update(torch=torch.__version__,triton=triton.__version__,gpu=torch.cuda.get_device_name(0),precision_flags_unmodified=flags,native_sources=expected)
        g=torch.Generator(device='cuda').manual_seed(810923)
        w=torch.nn.Parameter(torch.randn(151936,1536,device='cuda',dtype=torch.float32,generator=g)*.025,requires_grad=False)
        owner=torch.nn.Module()
        class Gather:
            def gather_for_metrics(self,value):return value
        assert not w.requires_grad
        def fixture(b,t,ignored=False):
            h=(torch.randn(b,t,1536,device='cuda',generator=g)*.2).requires_grad_()
            labels=torch.randint(0,151936,(b,t),device='cuda',generator=g,dtype=torch.int64)
            labels[:,0]=-100;labels[:,::5]=-100
            if ignored:labels.fill_(-100)
            mask=torch.ones_like(labels);mask[:,-1]=0
            return h,labels,mask
        def run(method,h,labels,mask,scale,denominator,details=False):
            if method=='native':
                # Actual native CUDA autocast, not merely an explicit-cast reference.
                with torch.autocast('cuda',dtype=torch.float16):
                    raw=F.linear(h,w)
                    loss=ForCausalLMLoss(raw,labels,vocab_size=151936,num_items_in_batch=denominator)
                assert raw.dtype==torch.float16 and loss.dtype==torch.float32
                public=raw.float()
                with torch.no_grad():
                    ent=entropy_from_logits(public)
                    entropy=((ent*mask).sum()/mask.sum()).item()
                    shift=public[:,:-1,:].contiguous()
                    predictions=shift.argmax(-1);target=labels[:,1:];valid=target!=-100
                    correct=((predictions==target)&valid).sum();total=valid.sum()
                    accuracy=(correct/total).item() if total>0 else 0.
                    del shift
                    rows,counts=row_losses(public,labels)
                    nll=None
                    if details:
                        nll=torch.stack([F.cross_entropy(public[i,:-1],labels[i,1:],ignore_index=-100,reduction='none') for i in range(labels.shape[0])])
                del public,raw
            else:
                trainer=SimpleNamespace(_metrics={'train':collections.defaultdict(list)},_total_train_tokens=0,accelerator=Gather())
                with bridge_state.scope(owner,labels,vocabulary=151936) as transaction:
                    transaction.producer_owner_ref=weakref.ref(owner)
                    sink=streaming_sink(owner,labels,vocabulary=151936)
                    assert sink is not None
                    with torch.autocast('cuda',dtype=torch.float16):
                        loss,s=token_chunk_boundary(h,w,labels,chunk=128,denominator=denominator,summary_sink=sink)
                    outputs=convert_to_fp32(CausalLMOutputWithPast(loss=loss,logits=None))
                    assert transaction.bind_outputs(loss,outputs) and outputs.loss is loss
                    publish_metrics(trainer,outputs,labels,mask,entropy_from_logits,transaction)
                    saved_nll=transaction.take('token_nll',None,labels)
                    assert saved_nll is s.token_nll
                    rows,counts=native_order_row_losses(saved_nll,labels)
                    assert transaction.consumed==set(bridge_state._FIELDS)
                entropy=trainer._metrics['train']['entropy'][0]
                accuracy=trainer._metrics['train']['mean_token_accuracy'][0]
                ent=s.entropy;predictions=s.predictions[:,:-1];nll=s.token_nll[:,:-1] if details else None
            # Actual GradScaler.scale supplies 65536, without an optimizer/step.
            # Other cases retain their genuine arbitrary upstream scalar.
            if scale==65536.:
                scaler=torch.amp.GradScaler('cuda',init_scale=65536.)
                scaled=scaler.scale(loss)
                (dh,)=torch.autograd.grad(scaled,h)
                assert scaler.get_scale()==65536.
            else:
                (dh,)=torch.autograd.grad(loss,h,torch.tensor(scale,device='cuda',dtype=torch.float32))
            return dict(loss=loss.detach(),dh=dh,entropy=entropy,accuracy=accuracy,rows=rows,counts=counts,predictions=predictions,token_entropy=ent,token_nll=nll)
        def cpu(value):
            return {k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in value.items()}
        def compare(a,b,name):
            if not isinstance(a,torch.Tensor):a=torch.tensor(a)
            if not isinstance(b,torch.Tensor):b=torch.tensor(b)
            assert torch.equal(torch.isnan(a),torch.isnan(b)),name+' NaN class'
            assert torch.equal(torch.isposinf(a),torch.isposinf(b)) and torch.equal(torch.isneginf(a),torch.isneginf(b)),name+' infinity class'
            finite=torch.isfinite(a)&torch.isfinite(b)
            af=a[finite].float();bf=b[finite].float();delta=(af-bf).abs()
            maxabs=delta.max().item() if delta.numel() else 0.
            refmax=bf.abs().max().item() if bf.numel() else 0.
            rel_l2=(delta.norm()/bf.norm().clamp(min=1e-30)).item() if delta.numel() else 0.
            if name=='dh':
                assert rel_l2<=.001 and maxabs<=.005*refmax+1e-7,(name,rel_l2,maxabs,refmax)
            else:torch.testing.assert_close(a,b,rtol=1e-5,atol=3e-6,equal_nan=True)
            return dict(max_abs=maxabs,relative_l2=rel_l2,reference_abs_max=refmax,exact_values=bool(torch.equal(a,b)),zero_sign_disagreements=int(((a==0)&(b==0)&(torch.signbit(a)!=torch.signbit(b))).sum()))
        if args.method=='numerical':
            cases=[(4,133,1.,False,True,False),(4,512,65536.,False,True,False),(1,3,.125,False,False,False),(1,3,65536.,True,False,False),(1,2,65536.,False,False,True)]
            for b,t,scale,ignored,external_denom,overflow in cases:
                h,labels,mask=fixture(b,t,ignored);denom=torch.tensor(b*t,device='cuda',dtype=torch.int64) if external_denom else None
                if overflow:
                    with torch.no_grad():
                        w.zero_();w[0,0]=1.;w[1,0]=-1.;h.zero_();labels.fill_(-100);labels[0,1]=0
                rng=torch.cuda.get_rng_state().clone();versions=(h._version,w._version,labels._version,mask._version)
                native=cpu(run('native',h,labels,mask,scale,denom,True));candidate=cpu(run('chunk',h,labels,mask,scale,denom,True))
                if overflow:assert not torch.isfinite(native['dh']).all(),'Actual scaled native overflow witness did not trigger'
                checks={key:compare(candidate[key],native[key],key) for key in ('loss','dh','entropy','accuracy','rows','token_entropy','token_nll')}
                assert candidate['counts']==native['counts']
                assert torch.equal(candidate['predictions'],native['predictions']),'argmax differs'
                assert torch.equal(rng,torch.cuda.get_rng_state()) and versions==(h._version,w._version,labels._version,mask._version)
                report['rows'].append(dict(B=b,T=t,V=151936,H=1536,scale=scale,all_ignored=ignored,external_denominator=external_denom,scaled_overflow_witness=overflow,actual_cuda_gradscaler=scale==65536.,checks=checks,exact_counts_and_first_argmax=True,rng_and_versions_unchanged=True))
                write(out,report);del h,labels,mask,native,candidate
            report['numerical_contract']='Strict component gates, not the task quality gates; gradient relative L2 <=0.001 and normalized max <=0.005 plus1e-7; metrics rtol1e-5/atol3e-6; exact argmax/counts/NaN/Inf classes. Full-V actual-GradScaler overflow classification is checked; tiny CPU overflow/underflow witnesses are separately preserved. Shared consumer files are byte-identical to the common materialized/streaming consumer, exercised through normalized-loss identity across actual Accelerate conversion; no production/model installer exists.'
        else:
            # Each selected method has its own fresh process. Inputs allocated before peaks/timing.
            for t in (133,512):
                h,labels,mask=fixture(4,t);denom=torch.tensor(4*t,device='cuda',dtype=torch.int64)
                torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                before=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
                started=time.perf_counter();value=run(args.method,h,labels,mask,65536.,denom);torch.cuda.synchronize();cold=time.perf_counter()-started
                assert torch.isfinite(value['loss']) and torch.isfinite(value['dh']).all()
                peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
                del value;gc.collect()
                for _ in range(1):
                    value=run(args.method,h,labels,mask,65536.,denom);del value
                torch.cuda.synchronize();samples=[]
                for _ in range(3):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record();value=run(args.method,h,labels,mask,65536.,denom);end.record();end.synchronize()
                    samples.append(begin.elapsed_time(end));del value
                observed_peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
                report['rows'].append(dict(B=4,T=t,V=151936,H=1536,token_chunk=128,scale=65536.,cold_first_shape_call_wall_s=cold,warm_cuda_event_ms=samples,warm_median_ms=statistics.median(samples),input_baseline_bytes=before,component_first_call_absolute_peak_bytes=peak,component_observed_absolute_peak_bytes=observed_peak,component_incremental_allocated_peak_bytes=observed_peak['allocated']-before['allocated']))
                write(out,report);del h,labels,mask,denom
            report['measurement_contract']='Cold synchronized head+metrics+adjoint wall includes dispatch, allocation, CUDA lazy initialization and any libraries first-use; no Triton compile attribution. Warm stream events include Python gaps and required metric host syncs. Peak is this isolated synthetic component INCLUDING its frozen W/input allocations, NOT full model/training peak or external nvidia-smi admission measurement. Native/chunk process order fixed; no task-speed claim. Native TRL entropy already uses128 token chunks. No actual model weights/data loaded.'
        assert flags==dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        report['status']='PASSED_HEAD_COMPONENT_PREFLIGHT_ONLY'
    except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
    finally:write(out,report)
    print(json.dumps(dict(method=args.method,status=report['status'],rows=len(report['rows']))),flush=True)
    raise SystemExit(0 if report['status']=='PASSED_HEAD_COMPONENT_PREFLIGHT_ONLY' else 1)
if __name__=='__main__':main()
