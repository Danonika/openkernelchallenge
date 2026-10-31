"""Sync-count/source and negative classifier fixtures; no GPU/time claims."""
import ast
from dataclasses import FrozenInstanceError
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from solution import mixed_rows as new
from solution import materialized_ce as ce

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/workspace/shared/soup_next_wave_20261030/ce_rowmix_v1')
spec=importlib.util.spec_from_file_location('old_ce_mixed_reference',OLD/'solution/mixed_rows.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def record_calls(fn):
    calls={'item':0,'tolist':0,'boolean_index':0,'nonzero':0}
    item,tolist,getitem,nonzero=torch.Tensor.item,torch.Tensor.tolist,torch.Tensor.__getitem__,torch.nonzero
    def counted_item(self,*args,**kwargs):calls['item']+=1;return item(self,*args,**kwargs)
    def counted_tolist(self,*args,**kwargs):calls['tolist']+=1;return tolist(self,*args,**kwargs)
    def counted_getitem(self,key):
        if isinstance(key,torch.Tensor) and key.dtype==torch.bool:calls['boolean_index']+=1
        return getitem(self,key)
    def counted_nonzero(*args,**kwargs):calls['nonzero']+=1;return nonzero(*args,**kwargs)
    with patch.object(torch.Tensor,'item',counted_item),patch.object(torch.Tensor,'tolist',counted_tolist), \
         patch.object(torch.Tensor,'__getitem__',counted_getitem),patch.object(torch,'nonzero',counted_nonzero):
        out=fn()
    return out,calls


def test_four_separate_explicit_reads_become_one_no_boolean_gather():
    z=torch.zeros(8,17,dtype=torch.float16);z[1,0]=12.;z[5,0]=30.
    target=torch.tensor([1,0,-100,1,1,0,1,1])
    f=z.float();maximum=f.amax(1);logsum=(f-maximum[:,None]).exp().sum(1).log()
    selected=(f.gather(1,target.clamp_min(0)[:,None]).flatten()-maximum)-logsum
    selected[target==-100]=0.;guard=((target!=-100)&(selected>=-ce.NEAR_CERTAIN_NLL)).int()
    def context():return SimpleNamespace(save_for_backward=lambda *args:None,set_materialize_grads=lambda *args:None)
    def before():
        assert bool(torch.any(guard).item())
        rows,reason=old.partition_rows(selected,target,guard);assert reason is None
        return old.MixedRowSum.forward(context(),z,target,maximum,logsum,selected,guard,rows,None)
    def after():
        part,reason=new.partition_rows(selected,target,guard);assert reason is None
        return new.MixedRowSum.forward(context(),z,target,maximum,logsum,selected,guard,part,None)
    with torch.no_grad():
        a,a_calls=record_calls(before);b,b_calls=record_calls(after)
    assert torch.equal(a,b)
    assert a_calls==dict(item=4,tolist=0,boolean_index=1,nonzero=1)
    assert b_calls==dict(item=0,tolist=1,boolean_index=0,nonzero=1)
    (ROOT/'validation/classification_reads.json').write_text(json.dumps({
        'status':'CPU_CALL_COUNT_PASS','before':a_calls,'after':b_calls,
        'scope':'Python-visible classifier/forward calls only. CPU instrumentation is NOT a GPU synchronization count or timing.',
        'unchanged_extra_work':'denominator read, first global flags read, backward q finite read, dynamic native nonzero, native subset replay',
        'no_GPU_executed':True},indent=2)+'\n')


@pytest.mark.parametrize('case',['mixed','all_safe','all_near','all_ignored','nan_supervised','inf_supervised','neginf_supervised','nan_ignored'])
def test_classifier_full_native_guards_and_counts_one_read(case):
    target=torch.tensor([0,0,-100,0,0,-100]);selected=torch.tensor([-2.,-0.0001,0.,-3.,-0.001,0.])
    if case=='all_safe':selected[target!=-100]=-2.
    if case=='all_near':selected[target!=-100]=0.
    if case=='all_ignored':target.fill_(-100);selected.zero_()
    if case in ('nan_supervised','inf_supervised','neginf_supervised'):
        selected[0]={'nan_supervised':float('nan'),'inf_supervised':float('inf'),'neginf_supervised':-float('inf')}[case]
    if case=='nan_ignored':selected[2]=float('nan')
    guard=((target!=-100)&((selected>=-ce.NEAR_CERTAIN_NLL)|~selected.isfinite())).int()
    (part,reason),calls=record_calls(lambda:new.partition_rows(selected,target,guard))
    assert calls['item']==0 and calls['tolist']==1 and calls['boolean_index']==0
    if case in ('nan_supervised','inf_supervised','neginf_supervised'):
        assert part is None and reason=='invalid_selected_logp'
    elif case=='all_near':assert part is None and reason=='all_supervised_rows_near_certain'
    elif case=='all_ignored':assert part is None and reason=='no_supervised_rows'
    else:
        assert reason is None
        expected=torch.nonzero(guard,as_tuple=False).flatten()
        assert part.native_supervised_rows==expected.numel()
        assert part.ignored_rows==int((target==-100).sum())
        assert part.safe_supervised_rows==target.numel()-expected.numel()-part.ignored_rows
        if case=='all_safe':assert part.native_rows is None and calls['nonzero']==0
        else:assert torch.equal(part.native_rows,expected) and calls['nonzero']==1
        with pytest.raises(FrozenInstanceError):part.safe_supervised_rows=100
    if reason is not None:assert calls['nonzero']==0


def test_exhaustive_small_masks_and_permutation_match_prior_route():
    for bits in range(256):
        target=torch.tensor([0,0,-100,0,0,-100,0,0])
        selected=torch.tensor([-0.0001 if bits&(1<<i) else -2. for i in range(8)])
        selected[target==-100]=0.
        for order in (torch.arange(8),torch.tensor([7,2,4,1,0,6,5,3])):
            y,s=target[order],selected[order]
            guard=((y!=-100)&(s>=-ce.NEAR_CERTAIN_NLL)).int()
            a,why_a=old.partition_rows(s,y,guard);b,why_b=new.partition_rows(s,y,guard)
            assert why_a==why_b
            if why_a is None:
                assert torch.equal(a, b.native_rows if b.native_rows is not None else torch.empty(0,dtype=torch.long))
                assert b.native_supervised_rows==a.numel()
                assert b.safe_supervised_rows==int(((y!=-100)&(guard==0)).sum())


def test_all_kernel_arithmetic_and_backward_ASTs_are_identical():
    def kernels(path):
        tree=ast.parse(path.read_text())
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree)
                if isinstance(n,ast.FunctionDef) and any(ast.unparse(d)=='triton.jit' for d in n.decorator_list)}
    for filename in ('materialized_ce.py','mixed_rows.py'):
        assert kernels(OLD/'solution'/filename)==kernels(ROOT/'solution'/filename)
    def backward(path):
        return ast.dump(next(n for n in ast.walk(ast.parse(path.read_text()))
                            if isinstance(n,ast.FunctionDef) and n.name=='backward'),include_attributes=False)
    assert backward(OLD/'solution/mixed_rows.py')==backward(ROOT/'solution/mixed_rows.py')
    for filename in ('numerical_gates.py','verify_integration.py','verify_rowmix.py','telemetry.py','adapter.py','__init__.py'):
        assert (OLD/'solution'/filename).read_bytes()==(ROOT/'solution'/filename).read_bytes()
