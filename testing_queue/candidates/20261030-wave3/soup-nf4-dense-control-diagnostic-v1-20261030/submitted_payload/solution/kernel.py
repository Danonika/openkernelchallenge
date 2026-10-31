"""Original dense-W16 control: same tiled MMA/reduction, no inner-loopNF4decode.

W16 is rematerialized by native dequantization INSIDE every measured wrapper.
No dense matrix cache; no vendor MMA implementation copied.
"""
import triton
import triton.language as tl
@triton.jit
def _mma1688(a0,a1,b0,c0,c1,c2,c3):
    return tl.inline_asm_elementwise(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5}, {$6}, {$7,$8,$9,$10};",
        constraints="=f,=f,=f,=f,r,r,r,f,f,f,f",
        args=[a0,a1,b0,c0,c1,c2,c3],dtype=(tl.float32,tl.float32,tl.float32,tl.float32),
        is_pure=True,pack=1)


@triton.jit(do_not_specialize=['M'])
def soup_dense_mma_control(A,WEIGHT,Bias,Out,
                             M,N:tl.constexpr,K:tl.constexpr,BIAS:tl.constexpr):
    tl.static_assert(K%64==0)
    thread=tl.arange(0,128)
    warp=thread//32;lane=thread%32
    group=lane//4;pair=lane%4
    m0=tl.program_id(0)*64+(warp//2)*32+group
    n0=tl.program_id(1)*64+(warp%2)*32+group
    A2=tl.cast(A,tl.pointer_type(tl.uint32))
    W2=tl.cast(WEIGHT,tl.pointer_type(tl.uint32))
    c00_0=tl.full((128,),0,tl.float32);c00_1=tl.full((128,),0,tl.float32);c00_2=tl.full((128,),0,tl.float32);c00_3=tl.full((128,),0,tl.float32)
    c01_0=tl.full((128,),0,tl.float32);c01_1=tl.full((128,),0,tl.float32);c01_2=tl.full((128,),0,tl.float32);c01_3=tl.full((128,),0,tl.float32)
    c02_0=tl.full((128,),0,tl.float32);c02_1=tl.full((128,),0,tl.float32);c02_2=tl.full((128,),0,tl.float32);c02_3=tl.full((128,),0,tl.float32)
    c03_0=tl.full((128,),0,tl.float32);c03_1=tl.full((128,),0,tl.float32);c03_2=tl.full((128,),0,tl.float32);c03_3=tl.full((128,),0,tl.float32)
    c10_0=tl.full((128,),0,tl.float32);c10_1=tl.full((128,),0,tl.float32);c10_2=tl.full((128,),0,tl.float32);c10_3=tl.full((128,),0,tl.float32)
    c11_0=tl.full((128,),0,tl.float32);c11_1=tl.full((128,),0,tl.float32);c11_2=tl.full((128,),0,tl.float32);c11_3=tl.full((128,),0,tl.float32)
    c12_0=tl.full((128,),0,tl.float32);c12_1=tl.full((128,),0,tl.float32);c12_2=tl.full((128,),0,tl.float32);c12_3=tl.full((128,),0,tl.float32)
    c13_0=tl.full((128,),0,tl.float32);c13_1=tl.full((128,),0,tl.float32);c13_2=tl.full((128,),0,tl.float32);c13_3=tl.full((128,),0,tl.float32)
    for block in tl.range(0,K//64,loop_unroll_factor=1):
        ns0=n0+0
        ns1=n0+8
        ns2=n0+16
        ns3=n0+24
        for chunk in tl.range(0,8,loop_unroll_factor=1):
            k0=block*64+chunk*8+pair*2
            a00=tl.load(A2+((m0+0)*K+k0)//2,m0+0<M,other=0)
            a01=tl.load(A2+((m0+8)*K+k0)//2,m0+8<M,other=0)
            a10=tl.load(A2+((m0+16)*K+k0)//2,m0+16<M,other=0)
            a11=tl.load(A2+((m0+24)*K+k0)//2,m0+24<M,other=0)
            b0=tl.load(W2+(ns0*K+k0)//2,ns0<N,other=0)
            b1=tl.load(W2+(ns1*K+k0)//2,ns1<N,other=0)
            b2=tl.load(W2+(ns2*K+k0)//2,ns2<N,other=0)
            b3=tl.load(W2+(ns3*K+k0)//2,ns3<N,other=0)
            c00_0,c00_1,c00_2,c00_3=_mma1688(a00,a01,b0,c00_0,c00_1,c00_2,c00_3)
            c01_0,c01_1,c01_2,c01_3=_mma1688(a00,a01,b1,c01_0,c01_1,c01_2,c01_3)
            c02_0,c02_1,c02_2,c02_3=_mma1688(a00,a01,b2,c02_0,c02_1,c02_2,c02_3)
            c03_0,c03_1,c03_2,c03_3=_mma1688(a00,a01,b3,c03_0,c03_1,c03_2,c03_3)
            c10_0,c10_1,c10_2,c10_3=_mma1688(a10,a11,b0,c10_0,c10_1,c10_2,c10_3)
            c11_0,c11_1,c11_2,c11_3=_mma1688(a10,a11,b1,c11_0,c11_1,c11_2,c11_3)
            c12_0,c12_1,c12_2,c12_3=_mma1688(a10,a11,b2,c12_0,c12_1,c12_2,c12_3)
            c13_0,c13_1,c13_2,c13_3=_mma1688(a10,a11,b3,c13_0,c13_1,c13_2,c13_3)
    col0=tl.program_id(1)*64+(warp%2)*32+pair*2
    col0=col0+0
    if BIAS:
        bias00=tl.load(Bias+col0,col0<N,other=0).to(tl.float32)
        bias01=tl.load(Bias+col0+1,col0+1<N,other=0).to(tl.float32)
        c00_0=c00_0+bias00
        c00_1=c00_1+bias01
        c00_2=c00_2+bias00
        c00_3=c00_3+bias01
        c10_0=c10_0+bias00
        c10_1=c10_1+bias01
        c10_2=c10_2+bias00
        c10_3=c10_3+bias01
    col1=col0+8
    if BIAS:
        bias10=tl.load(Bias+col1,col1<N,other=0).to(tl.float32)
        bias11=tl.load(Bias+col1+1,col1+1<N,other=0).to(tl.float32)
        c01_0=c01_0+bias10
        c01_1=c01_1+bias11
        c01_2=c01_2+bias10
        c01_3=c01_3+bias11
        c11_0=c11_0+bias10
        c11_1=c11_1+bias11
        c11_2=c11_2+bias10
        c11_3=c11_3+bias11
    col2=col0+16
    if BIAS:
        bias20=tl.load(Bias+col2,col2<N,other=0).to(tl.float32)
        bias21=tl.load(Bias+col2+1,col2+1<N,other=0).to(tl.float32)
        c02_0=c02_0+bias20
        c02_1=c02_1+bias21
        c02_2=c02_2+bias20
        c02_3=c02_3+bias21
        c12_0=c12_0+bias20
        c12_1=c12_1+bias21
        c12_2=c12_2+bias20
        c12_3=c12_3+bias21
    col3=col0+24
    if BIAS:
        bias30=tl.load(Bias+col3,col3<N,other=0).to(tl.float32)
        bias31=tl.load(Bias+col3+1,col3+1<N,other=0).to(tl.float32)
        c03_0=c03_0+bias30
        c03_1=c03_1+bias31
        c03_2=c03_2+bias30
        c03_3=c03_3+bias31
        c13_0=c13_0+bias30
        c13_1=c13_1+bias31
        c13_2=c13_2+bias30
        c13_3=c13_3+bias31
    tl.store(Out+(m0+0)*N+col0+0,c00_0.to(tl.float16),(m0+0<M)&(col0+0<N))
    tl.store(Out+(m0+0)*N+col0+1,c00_1.to(tl.float16),(m0+0<M)&(col0+1<N))
    tl.store(Out+(m0+8)*N+col0+0,c00_2.to(tl.float16),(m0+8<M)&(col0+0<N))
    tl.store(Out+(m0+8)*N+col0+1,c00_3.to(tl.float16),(m0+8<M)&(col0+1<N))
    tl.store(Out+(m0+0)*N+col1+0,c01_0.to(tl.float16),(m0+0<M)&(col1+0<N))
    tl.store(Out+(m0+0)*N+col1+1,c01_1.to(tl.float16),(m0+0<M)&(col1+1<N))
    tl.store(Out+(m0+8)*N+col1+0,c01_2.to(tl.float16),(m0+8<M)&(col1+0<N))
    tl.store(Out+(m0+8)*N+col1+1,c01_3.to(tl.float16),(m0+8<M)&(col1+1<N))
    tl.store(Out+(m0+0)*N+col2+0,c02_0.to(tl.float16),(m0+0<M)&(col2+0<N))
    tl.store(Out+(m0+0)*N+col2+1,c02_1.to(tl.float16),(m0+0<M)&(col2+1<N))
    tl.store(Out+(m0+8)*N+col2+0,c02_2.to(tl.float16),(m0+8<M)&(col2+0<N))
    tl.store(Out+(m0+8)*N+col2+1,c02_3.to(tl.float16),(m0+8<M)&(col2+1<N))
    tl.store(Out+(m0+0)*N+col3+0,c03_0.to(tl.float16),(m0+0<M)&(col3+0<N))
    tl.store(Out+(m0+0)*N+col3+1,c03_1.to(tl.float16),(m0+0<M)&(col3+1<N))
    tl.store(Out+(m0+8)*N+col3+0,c03_2.to(tl.float16),(m0+8<M)&(col3+0<N))
    tl.store(Out+(m0+8)*N+col3+1,c03_3.to(tl.float16),(m0+8<M)&(col3+1<N))
    tl.store(Out+(m0+16)*N+col0+0,c10_0.to(tl.float16),(m0+16<M)&(col0+0<N))
    tl.store(Out+(m0+16)*N+col0+1,c10_1.to(tl.float16),(m0+16<M)&(col0+1<N))
    tl.store(Out+(m0+24)*N+col0+0,c10_2.to(tl.float16),(m0+24<M)&(col0+0<N))
    tl.store(Out+(m0+24)*N+col0+1,c10_3.to(tl.float16),(m0+24<M)&(col0+1<N))
    tl.store(Out+(m0+16)*N+col1+0,c11_0.to(tl.float16),(m0+16<M)&(col1+0<N))
    tl.store(Out+(m0+16)*N+col1+1,c11_1.to(tl.float16),(m0+16<M)&(col1+1<N))
    tl.store(Out+(m0+24)*N+col1+0,c11_2.to(tl.float16),(m0+24<M)&(col1+0<N))
    tl.store(Out+(m0+24)*N+col1+1,c11_3.to(tl.float16),(m0+24<M)&(col1+1<N))
    tl.store(Out+(m0+16)*N+col2+0,c12_0.to(tl.float16),(m0+16<M)&(col2+0<N))
    tl.store(Out+(m0+16)*N+col2+1,c12_1.to(tl.float16),(m0+16<M)&(col2+1<N))
    tl.store(Out+(m0+24)*N+col2+0,c12_2.to(tl.float16),(m0+24<M)&(col2+0<N))
    tl.store(Out+(m0+24)*N+col2+1,c12_3.to(tl.float16),(m0+24<M)&(col2+1<N))
    tl.store(Out+(m0+16)*N+col3+0,c13_0.to(tl.float16),(m0+16<M)&(col3+0<N))
    tl.store(Out+(m0+16)*N+col3+1,c13_1.to(tl.float16),(m0+16<M)&(col3+1<N))
    tl.store(Out+(m0+24)*N+col3+0,c13_2.to(tl.float16),(m0+24<M)&(col3+0<N))
    tl.store(Out+(m0+24)*N+col3+1,c13_3.to(tl.float16),(m0+24<M)&(col3+1<N))
