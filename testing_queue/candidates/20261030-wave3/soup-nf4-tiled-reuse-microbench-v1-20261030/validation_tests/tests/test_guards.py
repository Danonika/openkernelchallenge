"""Metadata-only CPU tests, never launch a CUDA kernel."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch,PropertyMock
import torch,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.guarded_forward import reason,make_replacement

def args(n=256,m=65):
    blocks=n*1536//64
    return [torch.zeros(m,1536,dtype=torch.float16),torch.zeros(n*1536//2,1,dtype=torch.uint8),(n,1536),
            torch.ones((blocks+255)//256),64,'nf4',torch.zeros(n,dtype=torch.float16),
            torch.zeros(blocks,dtype=torch.uint8),torch.zeros(256),torch.zeros(())]

@pytest.mark.parametrize('n',[256,1536])
@pytest.mark.parametrize('m',[65,532,2048])
def test_guard_accept_metadata_only(n,m):
    a=args(n,m)
    with patch.object(torch.Tensor,'device',new_callable=PropertyMock,return_value=torch.device('cuda:0')),patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(major=7,minor=5,multi_processor_count=40)):
        assert reason(*a) is None
        a[0].requires_grad_();assert reason(*a)=='direct_autograd_use'
        with torch.no_grad():assert reason(*a) is None

@pytest.mark.parametrize('case',['lazy_a','lazy_bias','lazy_abs','subclass','m64','m2049','n512','fp32','bias_shape','missing_nested','scale_grad','strided','wrong_block','architecture'])
def test_failclosed(case):
    a=args()
    if case=='lazy_a':a[0]=torch._neg_view(a[0])
    elif case=='lazy_bias':a[6]=torch._neg_view(a[6])
    elif case=='lazy_abs':a[3]=torch._neg_view(a[3])
    elif case=='subclass':a[0]=torch.nn.Parameter(a[0],requires_grad=False)
    elif case=='m64':a=args(m=64)
    elif case=='m2049':a=args(m=2049)
    elif case=='n512':a=args(n=512)
    elif case=='fp32':a[0]=a[0].float()
    elif case=='bias_shape':a[6]=a[6][:1]
    elif case=='missing_nested':a[7]=None
    elif case=='scale_grad':a[3].requires_grad_()
    elif case=='strided':a[0]=torch.zeros(65,3072,dtype=torch.float16)[:,::2]
    elif case=='wrong_block':a[4]=32
    prop=SimpleNamespace(major=8 if case=='architecture' else 7,minor=5,multi_processor_count=40)
    calls=[];token=object()
    def native(*aa):calls.append(aa);return token
    with patch.object(torch.Tensor,'device',new_callable=PropertyMock,return_value=torch.device('cuda:0')),patch.object(torch.cuda,'get_device_properties',return_value=prop):
        assert reason(*a) is not None
        assert make_replacement(native)(*a) is token
    assert len(calls)==1 and all(x is y for x,y in zip(calls[0],a))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('key',['fallback','backward'])
def test_preinstalled_wrapped_native_owner_rejected(key):
    import functools,inspect,hashlib
    from solution.guarded_forward import verify_native_owner
    def original(*args):return args
    original.__module__,original.__qualname__={
        'fallback':('bitsandbytes.backends.cuda.ops','_dequant_linear_fallback'),
        'backward':('bitsandbytes.autograd._functions','MatMul4Bit.backward')}[key]
    expected={key:hashlib.sha256(inspect.getsource(original).encode()).hexdigest()}
    verify_native_owner(key,original,expected)
    @functools.wraps(original)
    def foreign(*args):return original(*args)
    # Reproduces v1 weakness: source hashing alone unwraps to the approved body.
    assert hashlib.sha256(inspect.getsource(foreign).encode()).hexdigest()==expected[key]
    with pytest.raises(RuntimeError,match='Foreign native callable owner'):
        verify_native_owner(key,foreign,expected)
    original.__module__='foreign.module'
    with pytest.raises(RuntimeError,match='Foreign native callable owner'):
        verify_native_owner(key,original,expected)


def test_actual_int_list_dispatch_schema_accepts_native_shape_metadata():
    # Exact shape schema used by bitsandbytes::gemm_4bit; no CUDA execution.
    lib=torch.library.Library('nf4_shape_guard_v3','DEF')
    lib.define('check(Tensor a, int[] shapeB) -> Tensor')
    observed=[]
    @torch.library.impl(lib,'check','CPU')
    def check(a,shapeB):
        observed.append(type(shapeB))
        values=args();values[2]=shapeB
        with patch.object(torch.Tensor,'device',new_callable=PropertyMock,return_value=torch.device('cuda:0')),patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(major=7,minor=5,multi_processor_count=40)):
            assert reason(*values) is None
        return a.clone()
    torch.ops.nf4_shape_guard_v3.check(torch.zeros(1),(256,1536))
    torch.ops.nf4_shape_guard_v3.check(torch.zeros(1),torch.Size((256,1536)))
    assert observed==[list,list]
    class ForeignList(list):pass
    values=args();values[2]=ForeignList((256,1536))
    assert reason(*values)=='shape_metadata'
    assert not torch.cuda.is_initialized()
