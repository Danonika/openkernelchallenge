"""Focused CPU guard fixtures. Fake is_cuda metadata only, no CUDA query allowed."""
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TRITON_INTERPRET']='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from solution.packed_nf4 import raw_supported


def test_lazy_metadata_rejected():
    shape=(256,1536);n=shape[0]*shape[1]
    packed=torch.zeros(n//2,1,dtype=torch.uint8)
    data=[torch.zeros(n//64,dtype=torch.uint8),torch.linspace(-1,1,256),
          torch.ones((n//64+255)//256),torch.tensor([.1])]
    results=[]
    with patch.object(torch.Tensor,'is_cuda',new=property(lambda self:True)), \
         patch.object(torch.cuda,'get_device_capability',return_value=(7,5)):
        assert raw_supported(packed,shape,*data,torch.float16)
    results.append({'field':'ordinary_metadata_control','accepted_under_explicit_CPU_metadata_shim':True})
    for index in range(4):
        metadata=list(data);metadata[index]=torch._neg_view(metadata[index])
        assert metadata[index].is_neg() and metadata[index].is_contiguous()
        with patch.object(torch.Tensor,'is_cuda',new=property(lambda self:True)), \
             patch.object(torch.cuda,'get_device_capability',side_effect=AssertionError('CUDA query forbidden')):
            assert not raw_supported(packed,shape,*metadata,torch.float16)
        results.append({'field':['qabs','code2','abs2','offset'][index],'lazy_negative_rejected':True})
    flagged=torch._neg_view(packed)
    with patch.object(torch.Tensor,'is_cuda',new=property(lambda self:True)), \
         patch.object(torch.cuda,'get_device_capability',side_effect=AssertionError('CUDA query forbidden')):
        assert not raw_supported(flagged,shape,*data,torch.float16)
    results.append({'field':'packed','lazy_negative_rejected':True})
    (ROOT/'validation/guard_results.json').write_text(json.dumps({'status':'CPU_METADATA_GUARD_PASS','gpu_executed':False,'metadata_shim':True,'cases':results},indent=2)+'\n')


def test_numeric_kernel_ast_unchanged():
    original=Path('/workspace/shared/soup_six_release_20261030T2026Z/soup-six-candidates/nf4/solution/packed_nf4.py')
    current=ROOT/'solution/packed_nf4.py'
    def kernels(path):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node,ast.FunctionDef) and any(ast.unparse(d)=='triton.jit' for d in node.decorator_list)}
    assert kernels(original)==kernels(current)
    assert set(kernels(current))=={'_nf4_value','soup_nf4_blockpair_decode'}
