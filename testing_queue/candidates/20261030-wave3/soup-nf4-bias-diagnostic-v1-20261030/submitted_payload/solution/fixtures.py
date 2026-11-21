"""Small synthetic q/v-shaped inputs, not captured task activations."""
from pathlib import Path
import json
import torch
import bitsandbytes as bnb
from bitsandbytes import functional as F

EXPECTED=json.loads((Path(__file__).with_name('native_source_manifest.json')).read_text())
SHAPES=((65,256),(532,256),(2048,256),(65,1536),(532,1536),(2048,1536))

def fixture(m,n,with_bias=True):
    torch.manual_seed(1729+n)
    w=torch.randn(n,1536,device='cuda',dtype=torch.float16)*.1
    packed,state=F.quantize_4bit(w,blocksize=64,compress_statistics=True,quant_type='nf4')
    a=torch.randn(m,1536,device='cuda',dtype=torch.float16)*.1
    bias=torch.randn(n,device='cuda',dtype=torch.float16)*.03 if with_bias else None
    return a,packed,state,bias

def call(data):
    a,packed,state,bias=data
    return bnb.matmul_4bit(a,packed.t(),quant_state=state,bias=bias)

def classes(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:raise RuntimeError('Shape or dtype mismatch')
    for fn in (torch.isnan,torch.isposinf,torch.isneginf):
        if not torch.equal(fn(a),fn(b)):raise RuntimeError('Nonfinite classes differ')

def exact(a,b):
    classes(a,b)
    finite=torch.isfinite(a)&torch.isfinite(b)
    if not torch.equal(a[finite],b[finite]):raise RuntimeError('Finite values differ')
    if not torch.equal(torch.signbit(a[a==0]),torch.signbit(b[b==0])):raise RuntimeError('Signed zeros differ')

def engineering_close(a,b):
    classes(a,b)
    finite=torch.isfinite(a)&torch.isfinite(b)
    torch.testing.assert_close(a[finite],b[finite],atol=.003,rtol=.002)
    delta=(a[finite].float()-b[finite].float()).abs()
    return {'max_abs_error':float(delta.max().item()) if delta.numel() else 0.,
            'atol':.003,'rtol':.002,'criterion':'fixed synthetic engineering gate, not task quality'}
