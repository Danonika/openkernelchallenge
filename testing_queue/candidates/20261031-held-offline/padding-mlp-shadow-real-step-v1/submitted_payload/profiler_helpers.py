"""Bounded, data-only profiler summaries and identity-owned SDPA observation.
No backend selection, tensor copies/retention, arithmetic or optimizer work.
"""
import hashlib,json,math
from contextlib import contextmanager

def tensor_metadata(t):
    if t is None:return None
    return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'requires_grad':bool(t.requires_grad)}

def tensor_digest(t):
    raw=t.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()
    return {**tensor_metadata(t),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}

@contextmanager
def observe_sdpa(torch, report):
    """Only Python boundary metadata; internal autocast/kernel dtype is NOT inferred."""
    functional=torch.nn.functional
    original=functional.scaled_dot_product_attention
    report.update(call_count=0,signatures=[],restored=False,
        scope='Python SDPA boundary, not internal autocast/kernel operand dtype',
        tensor_values_copied=False,tensors_retained=False,arguments_or_outputs_modified=False)
    signatures={}
    def observed(*args,**kwargs):
        # Read only actual supplied operands/default scalar controls, never tensor values.
        def operand(index,name,default=None):return args[index] if len(args)>index else kwargs.get(name,default)
        q,k,v=[operand(i,n) for i,n in enumerate(('query','key','value'))]
        mask=operand(3,'attn_mask')
        entry={'query':tensor_metadata(q),'key':tensor_metadata(k),'value':tensor_metadata(v),
            'mask':tensor_metadata(mask),'dropout_p':float(operand(4,'dropout_p',0.0)),
            'is_causal':bool(operand(5,'is_causal',False)),
            'scale':kwargs.get('scale'),'enable_gqa':bool(kwargs.get('enable_gqa',False)),
            'autocast_enabled_cuda':bool(torch.is_autocast_enabled('cuda')),
            'autocast_dtype_cuda':str(torch.get_autocast_dtype('cuda'))}
        result=original(*args,**kwargs)
        entry['returned']=tensor_metadata(result)
        key=json.dumps(entry,sort_keys=True,allow_nan=False)
        report['call_count']+=1
        if key not in signatures:
            if len(signatures)>=16:raise RuntimeError('SDPA metadata signature bound exceeded')
            signatures[key]={**entry,'count':0};report['signatures'].append(signatures[key])
        signatures[key]['count']+=1
        return result
    functional.scaled_dot_product_attention=observed
    try:yield report
    finally:
        if functional.scaled_dot_product_attention is not observed:
            report['ownership_failure']=True
            raise RuntimeError('SDPA observer ownership changed; foreign binding not overwritten')
        functional.scaled_dot_product_attention=original
        report['restored']=functional.scaled_dot_product_attention is original

def number(value):
    value=float(value)
    if not math.isfinite(value) or value<0:raise ValueError('Invalid profiler time')
    return value

def bounded_table(rows,field,limit):
    rows=sorted(rows,key=lambda r:(-r[field],r['name']))
    return {'total_unique_names':len(rows),'total_calls':sum(r['count'] for r in rows),
        'total_'+field:sum(r[field] for r in rows),'retained_rows':rows[:limit],
        'omitted_unique_names':max(0,len(rows)-limit),
        'omitted_'+field:sum(r[field] for r in rows[limit:])}

def aggregate_events(events,cuda_type,cpu_type):
    kernels=[];operators=[];attention=[]
    for event in events:
        name=str(event.key)
        if len(name)>4096:raise ValueError('Profiler name bound exceeded')
        count=int(event.count)
        if count<1:raise ValueError('Invalid profiler count')
        if getattr(event,'is_user_annotation',False) or name.startswith('ProfilerStep'):continue
        device=number(getattr(event,'self_device_time_total',getattr(event,'self_cuda_time_total',0)))
        cpu=number(getattr(event,'self_cpu_time_total',0))
        if event.device_type==cuda_type:
            row={'name':name,'count':count,'self_cuda_us':device};kernels.append(row)
        elif event.device_type==cpu_type:
            row={'name':name,'count':count,'self_cpu_us':cpu,'self_correlated_cuda_us':device};operators.append(row)
        else:continue
        if any(s in name.lower() for s in ('attention','sdpa','softmax','bmm')):
            attention.append({'device_type':str(event.device_type),**row})
    if not kernels or sum(r['self_cuda_us'] for r in kernels)<=0:
        raise RuntimeError('CUDA profiling unsupported or no usable CUDA events; stop without fallback')
    if len(attention)>160:raise ValueError('Attention aggregate bound exceeded')
    return {'cuda_kernels':bounded_table(kernels,'self_cuda_us',120),
        'cpu_operators':bounded_table(operators,'self_cpu_us',100),
        'cpu_operators_by_correlated_cuda':bounded_table(operators,'self_correlated_cuda_us',100),
        'attention_related_operator_and_kernel_rows':attention,
        'limits':['One cold instrumented forward/backward; no steady-state/task throughput claim.',
          'CUDA table sums leaf CUDA event self times; CPU correlated CUDA is a separate attribution view, never added to CUDA totals.',
          'Kernel/operator names identify observed paths; dtype strings in names do not prove every internal operand dtype.',
          'No shapes, stacks, FLOPs or memory profiler collection; no raw trace or tensor values exported.']}
