"""Focused negative-view/native-fallback/source-identity CPU checks only."""
import ast,hashlib,json,os,sys
from pathlib import Path
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TRITON_INTERPRET']='1';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import pytest
from solution.epilogue import scale_add,_ScaleAdd


@pytest.mark.parametrize('negative_operand',['base','delta'])
def test_negative_view_falls_back_before_capability_query(negative_operand):
    b=torch.randn(2,256,dtype=torch.float32,requires_grad=True)
    d=torch.randn(2,256,dtype=torch.float16,requires_grad=True)
    if negative_operand=='base':b=torch._neg_view(b)
    else:d=torch._neg_view(d)
    expected=b+d*2.
    with patch.object(torch.Tensor,'is_cuda',new=property(lambda self:True)), \
         patch.object(torch.cuda,'get_device_capability',side_effect=AssertionError('CUDA query forbidden')), \
         patch.object(_ScaleAdd,'apply',side_effect=AssertionError('Kernel forbidden on lazy view')):
        actual=scale_add(b,d,2.)
    assert torch.equal(actual,expected)
    got=torch.autograd.grad(actual,(b,d),torch.ones_like(actual),retain_graph=True)
    ref=torch.autograd.grad(expected,(b,d),torch.ones_like(expected))
    assert all(torch.equal(x,y) for x,y in zip(got,ref))


def test_plain_metadata_control_and_noncontiguous_native():
    b=torch.randn(2,256);d=torch.randn(2,256,dtype=torch.float16);token=object()
    with patch.object(torch.Tensor,'is_cuda',new=property(lambda self:True)), \
         patch.object(torch.cuda,'get_device_capability',return_value=(7,5)), \
         patch.object(_ScaleAdd,'apply',return_value=token):
        assert scale_add(b,d,2.) is token
    b=torch.randn(256,2).t();d=torch.randn(256,2,dtype=torch.float16).t()
    assert b.shape==(2,256) and not b.is_contiguous()
    with patch.object(torch.Tensor,'is_cuda',new=property(lambda self:True)), \
         patch.object(_ScaleAdd,'apply',side_effect=AssertionError('noncontiguous kernel')):
        assert torch.equal(scale_add(b,d,2.),b+d*2.)


def test_numeric_and_source_ownership_bodies_unchanged():
    frozen=Path('/redacted/local/soup_six_release_20261030T2026Z/soup-six-candidates/lora/solution')
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old=functions(frozen/'epilogue.py');new=functions(ROOT/'solution/epilogue.py')
    for key in ('lora_half_scale_add','_ScaleAdd'):assert old[key]==new[key]
    assert (frozen/'patch.py').read_bytes()==(ROOT/'solution/patch.py').read_bytes()
