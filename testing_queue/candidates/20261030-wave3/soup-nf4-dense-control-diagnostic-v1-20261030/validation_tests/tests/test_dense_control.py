"""CPU metadata/source/no-cache proof. Native device decode is mocked explicitly."""
import ast,hashlib,sys
from pathlib import Path
from unittest.mock import patch
import torch,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solution.materialize import materialize_native_w16,verify_materialize_owners
from bitsandbytes import functional as F


def test_fresh_native_metadata_materialization_each_call_no_weight_cache():
    a=torch.zeros(65,1536,dtype=torch.float16);b=torch.zeros(1,256*1536//2,dtype=torch.uint8)
    shape=[256,1536];ab=torch.ones(24);qi=torch.zeros(6144,dtype=torch.uint8);code=torch.zeros(256);off=torch.zeros(())
    calls=[];outputs=[]
    def native(packed,quant_state):
        q=quant_state
        assert packed.shape==(256*1536//2,1) and packed.data_ptr()==b.data_ptr()
        assert q.absmax is qi and q.shape is shape and q.offset is off and q.dtype==torch.float16
        assert q.quant_type=='nf4' and q.blocksize==64 and q.nested
        assert q.state2.absmax is ab and q.state2.code is code and q.state2.dtype==torch.float32
        assert q.state2.blocksize==256 and not q.state2.nested
        calls.append(q);return torch.zeros(shape,dtype=torch.float16)
    with patch.object(F,'dequantize_4bit',native):
        for _ in range(3):outputs.append(materialize_native_w16(a,b,shape,ab,64,'nf4',qi,code,off))
    assert len(calls)==3 and len({id(x) for x in calls})==3 and len({x.data_ptr() for x in outputs})==3
    assert not torch.cuda.is_initialized()


def test_native_materialization_owner_gate_rejects_foreign():
    verify_materialize_owners()
    with patch.object(F,'dequantize_4bit',lambda *a,**k:None):
        with pytest.raises(RuntimeError,match='source/owner'):verify_materialize_owners()


def test_frozen_fused_control_and_parameter_math_unchanged():
    frozen=Path('/redacted/local/soup_nf4_tiled_reuse_2320_v1/solution')
    assert (ROOT/'solution/fused_kernel.py').read_bytes()==(frozen/'kernel.py').read_bytes()
    assert (ROOT/'solution/fused_guard.py').read_text()==(frozen/'guarded_forward.py').read_text().replace('from .kernel import','from .fused_kernel import')
    for name in ('parameter_bias.py','module_preflight.py','fixtures.py','bias_stage_gate.py','bias_gate_provenance.json','nf4_format.py'):
        assert (ROOT/'solution'/name).read_bytes()==(frozen/name).read_bytes()


def test_dense_math_removes_decode_not_changes_mma_order():
    dense=(ROOT/'solution/kernel.py').read_text();fused=(ROOT/'solution/fused_kernel.py').read_text()
    def instructions(source):
        tree=ast.parse(source)
        return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Name) and n.func.id=='_mma1688']
    assert instructions(dense)==instructions(fused)
    assert dense.count('tl.load(W2+')==4 and dense.count('tl.load(A2+')==4
    assert '_nf4_value' not in dense and '_pack_half2' not in dense
    assert dense.count('loop_unroll_factor=1')==2
    assert dense[dense.index('    col0='):]==fused[fused.index('    col0='):]


def test_materialization_inside_each_wrapper_and_stages_do_not_retain_weight():
    source=(ROOT/'solution/guarded_forward.py').read_text()
    assert source.index('weight=materialize_native_w16(')<source.index('compiled=soup_dense_mma_control[')
    assert 'STAGE_EVENTS.append((begin,middle,end,weight.numel()*weight.element_size()))' in source
    assert source.count('materialize_native_w16(')==1
    worker=(ROOT/'microbench/worker.py').read_text();driver=(ROOT/'microbench/driver.py').read_text()
    assert "['native','fused','dense']" in driver and 'len(stages)!=16' in worker
    assert "row['weight_cache']=False" in worker
    assert 'F.dequantize' not in worker and 'materialize_native_w16' not in worker
