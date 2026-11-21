"""Preflight-only direct W16 observation; not part of measured forward."""
import triton
import triton.language as tl
from .nf4_format import _nf4_value

@triton.jit
def operand_probe(PACKED,QABS,CODE2,ABS2,OFFSET,OUT,E:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    q=tl.load(PACKED+i//2,i<E,other=0).to(tl.uint32)
    code=tl.where((i&1)==0,q>>4,q&15)
    block=i//64
    index=tl.load(QABS+block,i<E,other=0).to(tl.int32)
    cv=tl.load(CODE2+index,i<E,other=0).to(tl.float32)
    scale2=tl.load(ABS2+block//256,i<E,other=0).to(tl.float32)
    product=cv*scale2
    scale=product+tl.load(OFFSET).to(tl.float32)
    w=(_nf4_value(code)*scale).to(tl.float16,fp_downcast_rounding='rtne')
    tl.store(OUT+i,w,i<E)
