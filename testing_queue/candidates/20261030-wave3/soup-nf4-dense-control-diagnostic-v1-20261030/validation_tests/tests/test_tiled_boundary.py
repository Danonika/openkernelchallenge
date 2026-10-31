"""New packed-load alignment boundary plus immutable candidate/reference claims."""
import ast,hashlib,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,PropertyMock
import torch,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solution import guarded_forward as g


def test_contiguous_misaligned_half_view_falls_back_with_original_inputs():
    n,k,m=256,1536,65;blocks=n*k//64
    a=torch.zeros(m*k+1,dtype=torch.float16)[1:].reshape(m,k)
    assert a.is_contiguous() and a.data_ptr()%4==2
    args=[a,torch.zeros(n*k//2,1,dtype=torch.uint8),[n,k],torch.ones((blocks+255)//256),64,'nf4',
          torch.zeros(n,dtype=torch.float16),torch.zeros(blocks,dtype=torch.uint8),torch.zeros(256),torch.zeros(())]
    token=object();calls=[]
    def native(*args):calls.append(args);return token
    with patch.object(torch.Tensor,'device',new_callable=PropertyMock,return_value=torch.device('cuda:0')),patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(major=7,minor=5,multi_processor_count=40)):
        assert g.reason(*args)=='activation_half2_alignment'
        assert g.make_replacement(native)(*args) is token
    assert len(calls)==1 and all(x is y for x,y in zip(calls[0],args))
    assert not torch.cuda.is_initialized()



def test_exact_frozen_reference_required_before_timing_not_measured_worker():
    preflight=(ROOT/'solution/web_preflight.py').read_text()
    assert preflight.index('exact(actual,original)')<preflight.index('error=engineering_close(actual,expected)')
    assert "'frozen_original_MMA_output_exact':True" in preflight
    assert 'warpsPerCTA = [4]' in preflight and "c.n_spills!=0" in preflight
    assert 'original_reference' not in (ROOT/'microbench/worker.py').read_text()
    assert 'original_reference' not in (ROOT/'solution/module_preflight.py').read_text()
