"""MICROBENCH_ONLY. No model/trainer/harness/optimizer, network or installations.

Cold first-call wall includes compilation where needed, dispatch, allocations,
execution and synchronization: NOT a pure compilation timer. Warm CUDA-event
intervals include stream idle gaps between Python launches, not profiler kernel
sums. Exactly the original native/candidate public rotary wrapper is timed.
"""
import argparse, hashlib, inspect, json, os, statistics, sys, time, traceback
from pathlib import Path

TS = (133, 235, 317, 485, 512)
WARMUP = 3
GROUPS = 5
REPETITIONS = 8

def write(path, value):
    temp = path.with_name(path.name + '.pending')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=('native', 'old', 'runtime'), required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.exists(): raise FileExistsError(out)
    report = dict(status='FAILED', scope='MICROBENCH_ONLY', method=args.method,
                  gpu_executed=False, training_executed=False, complete_fixed_task=False,
                  score=None, admission='UNKNOWN', rows=[],
                  measurement_contract={
                    'cold':'synchronized first call per shape in this fresh process and private empty Triton cache; includes compile when needed, allocations, dispatch and device work',
                    'warm':'CUDA-event stream interval per repeated wrapper call, with Python launch gaps; no pure-kernel or training-wall claim',
                    'cache':'not reset between shapes; runtime keys are intended to be reused',
                    'order':'single fixed native/old/runtime process order; no statistical counterbalanced performance claim',
                    'fixed_repetitions':REPETITIONS,'groups':GROUPS,'warmup_calls':WARMUP,
                    'coefficient_batch':1,'coefficient_batch_is_observed':False,
                    'shape_provenance':'T values observed in original source-bound training log; paired Q/K synthetic fixtures use task H12/H2 geometry',
                    'backward_layout':'Q projection-order and K contiguous, both layout families observed in original job; no claim every T/layout pair was recorded',
                    'telemetry':'original candidate telemetry remains enabled and its host overhead is included where it occurs'})
    write(out, report)
    try:
        assert not os.environ.get('TRITON_INTERPRET'), 'Interpreter forbidden for GPU microbenchmark'
        cache = Path(os.environ['TRITON_CACHE_DIR'])
        assert cache.is_dir() and not any(cache.iterdir()), 'Private empty cache required before importing Triton'
        import torch, triton, transformers
        from transformers.models.qwen2 import modeling_qwen2 as native_module
        from solution.cpu_contract import inputs, exact
        from solution import rotary as runtime_module
        from old_solution import rotary as old_module
        assert torch.__version__ == '2.11.0+cu128' and torch.version.cuda == '12.8'
        assert triton.__version__ == '3.6.0' and transformers.__version__ == '5.16.1'
        assert torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == 'Tesla T4'
        assert torch.cuda.get_device_capability(0) == (7,5)
        raw = Path(inspect.getfile(native_module)).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        assert blob == runtime_module.EXPECTED_QWEN_BLOB == old_module.EXPECTED_QWEN_BLOB
        native = native_module.apply_rotary_pos_emb
        assert not getattr(native, '_original_rotary_candidate', False), 'Native callable already patched'
        module = {'old':old_module, 'runtime':runtime_module}.get(args.method)
        fn = native if module is None else module.fused_apply_rotary_pos_emb
        def key_counts():
            if module is None: return None
            return {name:sum(len(entry[0]) for entry in getattr(module,name).device_caches.values())
                    for name in ('rotary_apply_f32','rotary_adjoint_f32')}
        def wall(call):
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = call()
            torch.cuda.synchronize()
            return value, time.perf_counter() - start
        def warm(call):
            for _ in range(WARMUP): call()
            torch.cuda.synchronize()
            samples = []
            for _ in range(GROUPS):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(REPETITIONS): call()
                end.record(); end.synchronize()
                samples.append(begin.elapsed_time(end) / REPETITIONS)
            assert all(v > 0 for v in samples)
            return dict(cuda_event_ms_per_call=samples, median_ms=statistics.median(samples),
                        min_ms=min(samples), max_ms=max(samples))
        # Initialize CUDA context and a tiny unrelated eager operation before timing.
        # No RoPE candidate kernel and no representative operation is prewarmed.
        scratch = torch.ones(16, device='cuda'); scratch.add_(1); torch.cuda.synchronize(); del scratch
        report['gpu_executed']=True; write(out,report)
        for t in TS:
            q,k,c,s = inputs(b=4,t=t,cb=1,device='cuda')
            gs = (torch.empty_strided(q.shape,q.stride(),device='cuda').fill_(65536),
                  torch.full(k.shape,-65536.,device='cuda'))
            assert q.stride() == (t*1536,128,1536,1)
            assert k.stride() == (t*256,128,256,1)
            assert gs[0].stride() == q.stride() and gs[1].stride() == (t*256,t*128,128,1)
            versions = [x._version for x in (q,k,c,s)]
            snapshots = [x.detach().clone() for x in (q,k,c,s)]
            rng = torch.cuda.get_rng_state().clone()
            before = key_counts()
            y, forward_s = wall(lambda: fn(q,k,c,s))
            after_forward = key_counts()
            backward = lambda: torch.autograd.grad(y,(q,k),gs,retain_graph=True)
            dy, backward_s = wall(backward)
            after_backward = key_counts()
            # Gates occur after this shape's first-call timing, before any warm timing.
            expected = native(q,k,c,s)
            expected_grad = torch.autograd.grad(expected,(q,k),gs)
            for actual, wanted in zip((*y,*dy),(*expected,*expected_grad)): exact(actual,wanted)
            for actual, source in zip(y,(q,k)):
                assert actual.data_ptr() != source.data_ptr()
            if module is not None:
                for actual, source in zip(y,(q,k)): assert actual.stride() == source.stride()
            for tensor, version, snap in zip((q,k,c,s),versions,snapshots):
                assert tensor._version == version
                exact(tensor,snap)
            assert torch.equal(rng,torch.cuda.get_rng_state())
            row = dict(T=t,B=4,Hq=12,Hk=2,D=128,coefficient_batch=1,
                       input_strides=[list(q.stride()),list(k.stride())],
                       upstream_strides=[list(g.stride()) for g in gs],
                       exact_value_class_signedzero_forward_backward=True,
                       cold_first_call_wall_s={'forward':forward_s,'backward':backward_s},
                       jit_cache_entries={'before':before,'after_forward':after_forward,'after_backward':after_backward})
            row['warm_forward'] = warm(lambda: fn(q,k,c,s))
            row['warm_backward'] = warm(backward)
            assert torch.equal(rng,torch.cuda.get_rng_state())
            report['rows'].append(row); write(out,report)
            del y,dy,expected,expected_grad,backward,q,k,c,s,gs,snapshots
        torch.cuda.synchronize()
        report.update(status='PASSED_MICROBENCH_ONLY',gpu_executed=True,
                      torch=torch.__version__,triton=triton.__version__,native_source_blob=blob,
                      final_jit_cache_entries=key_counts())
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        write(out,report)
    print(json.dumps({'scope':'MICROBENCH_ONLY','method':args.method,'status':report['status'],
                      'rows':len(report['rows']),'score':None,'admission':'UNKNOWN'}),flush=True)
    raise SystemExit(0 if report['status']=='PASSED_MICROBENCH_ONLY' else 1)

if __name__ == '__main__': main()
