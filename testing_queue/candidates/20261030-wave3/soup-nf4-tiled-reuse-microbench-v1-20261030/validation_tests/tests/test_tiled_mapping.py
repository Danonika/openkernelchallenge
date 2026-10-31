"""CPU address/bit-pattern/reuse proofs only, not actual PTX execution."""
import ast,collections,json,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]


def coords(thread,rm,cn,reg,pm=0,pn=0):
    warp,lane=divmod(thread,32);group,pair=divmod(lane,4)
    return (pm*64+(warp//2)*32+group+rm*16+(reg//2)*8,
            pn*64+(warp%2)*32+pair*2+cn*8+reg%2)


@pytest.mark.parametrize('shape',[(1,1),(17,9),(33,17),(65,65),(129,129),(65,256),(532,256)])
def test_complete_masked_output_bijection(shape):
    m,n=shape;writes=[]
    for pm in range((m+63)//64):
      for pn in range((n+63)//64):
       for thread in range(128):
        for rm in range(2):
         for cn in range(4):
          for reg in range(4):
           row,col=coords(thread,rm,cn,reg,pm,pn)
           if row<m and col<n:writes.append((row,col))
    assert len(writes)==len(set(writes))==m*n


def test_all_half_bit_patterns_survive_packed_load_without_arithmetic():
    bits=np.arange(65536,dtype=np.uint16)
    words=bits.view(np.uint32)
    low=(words&65535).astype(np.uint16);high=(words>>16).astype(np.uint16)
    assert np.array_equal(low,bits[::2]) and np.array_equal(high,bits[1::2])
    # Includes signed zeros, subnormals, infinities and every NaN payload.
    assert words.size==32768


def test_warp_A_B_fragment_reuse_and_mma_operand_coverage():
    # Integers are exact half data, so each public instruction fragment is
    # reconstructed independently and checked against its mathematical tile.
    A=np.arange(64*8,dtype=np.uint16).reshape(64,8)
    B=(np.arange(64*8,dtype=np.uint16).reshape(64,8)+512)
    Awords=A.reshape(-1).view(np.uint32)
    a_uses=collections.Counter();b_uses=collections.Counter()
    for warp in range(4):
      for rm in range(2):
       for cn in range(4):
        af=np.zeros((16,8),dtype=np.uint16);bf=np.zeros((8,8),dtype=np.uint16)
        for lane in range(32):
         group,pair=divmod(lane,4)
         for ar in range(2):
          row=(warp//2)*32+group+rm*16+ar*8;col=pair*2
          packed=Awords[(row*8+col)//2]
          af[group+ar*8,col:col+2]=[packed&65535,packed>>16]
          a_uses[(warp,lane,rm,ar)]+=1
         n=(warp%2)*32+group+cn*8
         bf[pair*2:pair*2+2,group]=B[n,pair*2:pair*2+2]
         b_uses[(warp,lane,cn)]+=1
        ar0=(warp//2)*32+rm*16;bn0=(warp%2)*32+cn*8
        assert np.array_equal(af,A[ar0:ar0+16])
        assert np.array_equal(bf,B[bn0:bn0+8].T)
    assert set(a_uses.values())=={4} and set(b_uses.values())=={2}


def test_original_staged_decoder_matches_reused_B_coordinates():
    sys.path.insert(0,'/workspace/shared/soup_cycle/nf4_matmul_cycle9')
    from nf4_reference import NF4,PackedNF4,pack_indices,decode_materialized,f32
    n,k=129,128;blocks=n*k//64
    state=PackedNF4(n,k,pack_indices([(i*7+3)%16 for i in range(n*k)]),
                    bytes((i*11)%256 for i in range(blocks)),
                    tuple(f32(-.7+i*.007) for i in range(256)),
                    tuple(f32(.3+i*.11) for i in range((blocks+255)//256)),f32(.2))
    expected=np.array(decode_materialized(state),dtype=np.float16).reshape(n,k)
    packed=np.frombuffer(state.packed,dtype=np.uint8);qabs=np.frombuffer(state.scale_indices,dtype=np.uint8)
    for pn in range((n+63)//64):
     for warp_n in range(2):
      for group in range(8):
       for cn in range(4):
        row=pn*64+warp_n*32+group+cn*8
        if row>=n:continue
        for block in range(k//64):
         sb=row*(k//64)+block
         scale=np.float32(np.float32(state.code2[qabs[sb]])*np.float32(state.abs2[sb//256]))
         scale=np.float32(scale+np.float32(state.offset))
         for chunk in range(8):
          for pair in range(4):
           col=block*64+chunk*8+pair*2;byte=int(packed[(row*k+col)//2])
           got=np.array([np.float32(np.float32(NF4[byte>>4])*scale),np.float32(np.float32(NF4[byte&15])*scale)],dtype=np.float16)
           assert np.array_equal(got.view(np.uint16),expected[row,col:col+2].view(np.uint16))


def test_explicit_source_reuses_operands_and_preserves_reduction_sequence():
    source=(ROOT/'solution/kernel.py').read_text();tree=ast.parse(source)
    kernel=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='soup_nf4_mma_tiled_reuse')
    calls=[x for x in ast.walk(kernel) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='_mma1688']
    assert len(calls)==8
    assert collections.Counter(ast.unparse(x.args[0]) for x in calls)=={'a00':4,'a10':4}
    assert collections.Counter(ast.unparse(x.args[1]) for x in calls)=={'a01':4,'a11':4}
    assert collections.Counter(ast.unparse(x.args[2]) for x in calls)=={'b0':2,'b1':2,'b2':2,'b3':2}
    assert source.count('tl.load(A2+')==4 and source.count('tl.load(PACKED+')==4
    assert source.count('loop_unroll_factor=1')==2
    assert 'thread=tl.arange(0,128)' in source
    assert 'tl.cast(A,tl.pointer_type(tl.uint32))' in source
    assert source.count('tl.store(')==32
