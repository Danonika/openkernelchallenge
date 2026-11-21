"""Original single-warp register-fragment feasibility probe, not an installer.

Instruction and fragment mapping from NVIDIA PTX ISA m16n8k8 documentation,
not a copied vendor GEMM. Plain tl.dot no longer selects SM75 MMA in Triton3.6.
Target preflight must verify 1D lane layout and exact fragment mapping before use.
"""
import triton
import triton.language as tl
from .nf4_format import _nf4_value


@triton.jit
def _pack_half2(a, b):
    return a.to(tl.uint16, bitcast=True).to(tl.uint32) | (b.to(tl.uint16, bitcast=True).to(tl.uint32) << 16)


@triton.jit
def _mma1688(a0, a1, b0, c0, c1, c2, c3):
    return tl.inline_asm_elementwise(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5}, {$6}, {$7,$8,$9,$10};",
        constraints="=f,=f,=f,=f,r,r,r,f,f,f,f",
        args=[a0,a1,b0,c0,c1,c2,c3],dtype=(tl.float32,tl.float32,tl.float32,tl.float32),
        is_pure=True,pack=1)


@triton.jit(do_not_specialize=['M'])
def soup_nf4_mma_laneprobe(A, PACKED, QABS, CODE2, ABS2, OFFSET, Bias, Out,
                          M, N:tl.constexpr, K:tl.constexpr, BIAS:tl.constexpr):
    tl.static_assert(K % 64 == 0)
    # One warp + 32 scalar logical elements. Codegen gate rejects another layout.
    lane=tl.arange(0,32)
    group=lane//4
    pair=lane%4
    m0=tl.program_id(0)*16+group
    n0=tl.program_id(1)*8+group
    c0=tl.full((32,),0,tl.float32);c1=tl.full((32,),0,tl.float32)
    c2=tl.full((32,),0,tl.float32);c3=tl.full((32,),0,tl.float32)
    for block in tl.range(0,K//64,loop_unroll_factor=1):
        scale_block=n0*(K//64)+block
        qi=tl.load(QABS+scale_block,n0<N,other=0).to(tl.int32)
        cv=tl.load(CODE2+qi,n0<N,other=0).to(tl.float32)
        ab=tl.load(ABS2+scale_block//256,n0<N,other=0).to(tl.float32)
        product=cv*ab
        scale=product+tl.load(OFFSET).to(tl.float32)
        for chunk in tl.range(0,8,loop_unroll_factor=1):
            k0=block*64+chunk*8+pair*2
            a00=tl.load(A+m0*K+k0,m0<M,other=0).to(tl.float16)
            a01=tl.load(A+m0*K+k0+1,m0<M,other=0).to(tl.float16)
            a10=tl.load(A+(m0+8)*K+k0,m0+8<M,other=0).to(tl.float16)
            a11=tl.load(A+(m0+8)*K+k0+1,m0+8<M,other=0).to(tl.float16)
            byte=tl.load(PACKED+(n0*K+k0)//2,n0<N,other=0).to(tl.uint32)
            b0=(_nf4_value(byte>>4)*scale).to(tl.float16,fp_downcast_rounding='rtne')
            b1=(_nf4_value(byte&15)*scale).to(tl.float16,fp_downcast_rounding='rtne')
            c0,c1,c2,c3=_mma1688(_pack_half2(a00,a01),_pack_half2(a10,a11),_pack_half2(b0,b1),c0,c1,c2,c3)
    col=tl.program_id(1)*8+pair*2
    if BIAS:
        bias0=tl.load(Bias+col,col<N,other=0).to(tl.float32)
        bias1=tl.load(Bias+col+1,col+1<N,other=0).to(tl.float32)
        c0=c0+bias0;c1=c1+bias1;c2=c2+bias0;c3=c3+bias1
    tl.store(Out+m0*N+col,c0.to(tl.float16), (m0<M)&(col<N))
    tl.store(Out+m0*N+col+1,c1.to(tl.float16),(m0<M)&(col+1<N))
    tl.store(Out+(m0+8)*N+col,c2.to(tl.float16),(m0+8<M)&(col<N))
    tl.store(Out+(m0+8)*N+col+1,c3.to(tl.float16),(m0+8<M)&(col+1<N))
