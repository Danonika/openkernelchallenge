"""Tiny CPU fragment-map and staged-bias proof, not instruction/GPU execution."""
from pathlib import Path
import json,hashlib,sys,ast
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/workspace/shared/soup_cycle/nf4_matmul_cycle9')
from nf4_reference import NF4,PackedNF4,pack_indices,decode_materialized,f32

torch.set_num_threads(2)

def fixture(n,k):
    indices=[(i*7+3)%16 for i in range(n*k)]
    blocks=(n*k+63)//64
    return PackedNF4(n,k,pack_indices(indices),bytes((i*11)%256 for i in range(blocks)),
                     tuple(f32(-.7+i*.007) for i in range(256)),
                     tuple(f32(.3+i*.11) for i in range((blocks+255)//256)),f32(.2))

def unpack(words):
    return words.view(np.uint16).view(np.float16).reshape(-1,2)

records=[]
for n,k in [(9,64),(17,128),(129,128)]:
    state=fixture(n,k)
    weights=np.array(decode_materialized(state),dtype=np.float16).reshape(n,k)
    # Independent scalar staged decoder from packed arrays, matching each B fragment.
    packed=np.frombuffer(state.packed,dtype=np.uint8)
    qabs=np.frombuffer(state.scale_indices,dtype=np.uint8)
    code=np.array(state.code2,dtype=np.float32);abs2=np.array(state.abs2,dtype=np.float32)
    for n_tile in range((n+7)//8):
      for block in range(k//64):
       for chunk in range(8):
        reconstructed=np.zeros((8,8),dtype=np.float16)
        visited=set()
        for lane in range(32):
          group,pair=divmod(lane,4);row=n_tile*8+group;col=block*64+chunk*8+pair*2
          if row<n:
            scale_block=row*(k//64)+block
            scale=np.float32(np.float32(code[qabs[scale_block]]*abs2[scale_block//256])+np.float32(state.offset))
            byte=packed[(row*k+col)//2]
            pairvals=np.array([np.float32(np.float32(NF4[int(byte)>>4])*scale),np.float32(np.float32(NF4[int(byte)&15])*scale)],dtype=np.float16)
            word=pairvals.view(np.uint32)
            actual=unpack(word).flatten()
            assert np.array_equal(actual.view(np.uint16),weights[row,col:col+2].view(np.uint16))
            reconstructed[pair*2:pair*2+2,group]=actual
          for j in range(2):visited.add((pair*2+j,group))
        assert len(visited)==64
        valid=min(8,n-n_tile*8)
        assert np.array_equal(reconstructed[:,:valid].view(np.uint16),weights[n_tile*8:n_tile*8+valid,block*64+chunk*8:block*64+chunk*8+8].T.copy().view(np.uint16))
    records.append({'N':n,'K':k,'decoded_operand_fragments_finite_bits_exact':True,'nested_superblock_boundary_covered':n*k>16384})

# A fragments and C fragment coordinates independently cover their full rectangles.
A=np.arange(16*8,dtype=np.float16).reshape(16,8)/8
fragment_A=np.empty_like(A); seen_A=set();seen_C=set()
for lane in range(32):
 group,pair=divmod(lane,4)
 for reg in range(4):
  row=group+(8 if reg>=2 else 0);col=pair*2+reg%2
  fragment_A[row,col]=A[row,col];seen_A.add((row,col))
  seen_C.add((row,col))
assert len(seen_A)==len(seen_C)==128 and np.array_equal(fragment_A,A)
for m,n in [(1,1),(17,9),(33,17)]:
 writes=[]
 for mi in range((m+15)//16):
  for ni in range((n+7)//8):
   for lane in range(32):
    group,pair=divmod(lane,4)
    for reg in range(4):
     row=mi*16+group+(8 if reg>=2 else 0);col=ni*8+pair*2+reg%2
     if row<m and col<n:writes.append((row,col))
 assert len(writes)==m*n and len(set(writes))==m*n

# Required non-equivalence witness: a half matmul result cannot be narrowed before bias.
a=torch.zeros(1,64,dtype=torch.float16);a[0,0]=1.;a[0,1]=.0004
w=torch.ones(1,64,dtype=torch.float16);bias=torch.tensor([.0002],dtype=torch.float16)
native=torch.nn.functional.linear(a,w,bias)
fused=((a.float()@w.float().T)+bias.float()).half()
wrong=(a@w.T)+bias
assert torch.equal(native,fused) and not torch.equal(native,wrong)
report={'status':'CPU_FRAGMENT_LAYOUT_AND_STAGED_BIAS_PASS','torch':torch.__version__,
        'kernel_source_sha256':hashlib.sha256((ROOT/'prototype/mma_probe.py').read_bytes()).hexdigest(),
        'operand_cases':records,'A_fragment_coordinate_bijection':True,'C_fragment_coordinate_bijection':True,
        'masked_output_write_bijections':[[1,1],[17,9],[33,17]],
        'bias_negative_witness':{'native_Flinear':native.item(),'FP32_acc_plus_bias_then_half':fused.item(),'incorrect_half_acc_then_bias':wrong.item()},
        'limits':['CPU packing/address/rounding proof only; inline PTX was not executed','No target3.6 compile, physical lane execution, GPU reduction order or speed proof','Single-warp16x8 geometry prioritizes auditability; not tuned or assumed fast'],
        'gpu_executed':False,'admission':'UNKNOWN','score':None}
(ROOT/'validation/cpu_lane_proof.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
