"""Fresh-process q/v-shaped forward microbench, never a training benchmark."""
import argparse,contextlib,json,os,time,traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['native','fused','dense'],required=True)
    p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    report={'status':'FAILED_MICROBENCH_ONLY','scope':'MICROBENCH_ONLY','method':args.method,
        'gpu_executed':False,'training_executed':False,'complete_fixed_task':False,'rows':[],
        'score':None,'admission':'UNKNOWN','method_variants':{'native':'bnb native dispatcher/fallback',
            'fused':'frozen original64x64 four-warp fused NF4 control','dense':'fresh native W16 materialization plus original same-mapping denseMMA, with per-call event instrumentation'},
        'measurement_contract':{'cold':'synchronized first wrapper call per shape; includes needed JIT, allocation, dispatch and device work, not pure compile',
            'warm':'12 separate CUDA event stream intervals after 3 warmup calls; Python launch gaps included',
            'order':'native then candidate in separate processes and private caches; not counterbalanced',
            'inputs':'synthetic scaled random q/v shapes; not captured task activation distributions',
            'no_task_speed_or_share_inference':True}}
    try:
        if os.environ.get('TRITON_INTERPRET'):raise RuntimeError('Interpreter forbidden')
        cache=Path(os.environ['TRITON_CACHE_DIR'])
        if not cache.is_dir() or any(cache.iterdir()):raise RuntimeError('Private empty initial cache required')
        import torch
        from solution.fixtures import EXPECTED,SHAPES,fixture,call,engineering_close
        from solution import guarded_forward as dense_guard,fused_guard
        g=fused_guard if args.method=='fused' else dense_guard
        ctx=lambda:g.temporary_native_fallback(EXPECTED) if args.method!='native' else contextlib.nullcontext()
        for m,n in SHAPES:
            data=fixture(m,n,True);before=g.COUNTS['custom'];stage_before=len(dense_guard.STAGE_EVENTS)
            with ctx():
                torch.cuda.synchronize();start=time.perf_counter();actual=call(data)
                torch.cuda.synchronize();cold=time.perf_counter()-start
            expected=call(data);numeric=engineering_close(actual,expected)
            del actual,expected
            if args.method!='native' and g.COUNTS['custom']!=before+1:raise RuntimeError('Cold candidate inactive')
            pairs=[]
            with ctx():
                for _ in range(3):call(data)
                torch.cuda.synchronize()
                for _ in range(12):
                    start=torch.cuda.Event(enable_timing=True);stop=torch.cuda.Event(enable_timing=True)
                    start.record();call(data);stop.record();pairs.append((start,stop))
                torch.cuda.synchronize()
            samples=[a.elapsed_time(b) for a,b in pairs]
            if args.method!='native' and g.COUNTS['custom']!=before+16:raise RuntimeError('Warm candidate inactive')
            row={'M':m,'N':n,'K':1536,'bias':True,'cold_first_wrapper_wall_s':cold,
                 'warm_forward_cuda_event_ms':samples,'numerics':numeric,'counts':dict(g.COUNTS)}
            if args.method=='dense':
                stages=dense_guard.STAGE_EVENTS[stage_before:]
                if len(stages)!=16:raise RuntimeError('Fresh materialization per measured call not established')
                def describe(pair):
                    begin,middle,end,nbytes=pair
                    return {'native_decode_event_ms':begin.elapsed_time(middle),'dense_MMA_event_ms':middle.elapsed_time(end),
                            'per_call_W16_bytes':nbytes,'scope':'instrumented stream intervals include launch/Python/JIT gaps, not pure kernel time'}
                row['cold_stages']=describe(stages[0]);row['warm_stages']=[describe(x) for x in stages[-12:]]
                row['fresh_materializations_this_shape']=16;row['weight_cache']=False
            report['rows'].append(row);out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if len(report['rows'])!=6:raise RuntimeError('Incomplete fixture set')
        report.update(status='PASSED_MICROBENCH_ONLY',gpu_executed=True)
    except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc())
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'method':args.method,'rows':len(report['rows'])}),flush=True)
    raise SystemExit(report['status']!='PASSED_MICROBENCH_ONLY')
if __name__=='__main__':main()
