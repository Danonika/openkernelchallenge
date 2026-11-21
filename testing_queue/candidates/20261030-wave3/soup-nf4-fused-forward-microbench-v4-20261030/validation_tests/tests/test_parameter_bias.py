"""CPU ownership proof; GPU GEMM/dequant are explicitly mocked, never launched."""
import hashlib,inspect,json,sys
from pathlib import Path
from unittest.mock import patch
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solution.parameter_bias import normalized_native_bias


def same(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8))


@pytest.mark.parametrize('requires_grad',[False,True])
def test_exact_parameter_alias_only_inside_nograd(requires_grad):
    bias=torch.nn.Parameter(torch.tensor([1.,-0.,2.],dtype=torch.float16),requires_grad=requires_grad)
    raw,why=normalized_native_bias(bias)
    assert raw is None and why=='parameter_bias_outside_native_nograd_forward'
    with torch.no_grad():raw,why=normalized_native_bias(bias)
    assert why is None and type(raw) is torch.Tensor and not raw.requires_grad
    assert raw.data_ptr()==bias.data_ptr() and raw.shape==bias.shape and raw.stride()==bias.stride()
    same(raw,bias);assert bias.requires_grad==requires_grad


@pytest.mark.parametrize('case',['custom_parameter','custom_tensor','lazy_negative','lazy_conjugate','strided','foreign_detach'])
def test_exclusions(case):
    class ForeignParameter(torch.nn.Parameter):pass
    class ForeignTensor(torch.Tensor):pass
    bias=torch.nn.Parameter(torch.zeros(8,dtype=torch.float16),requires_grad=False)
    if case=='custom_parameter':bias=ForeignParameter(bias)
    if case=='custom_tensor':bias=bias.detach().as_subclass(ForeignTensor)
    if case=='lazy_negative':bias=torch.nn.Parameter(torch._neg_view(bias.detach()),requires_grad=False)
    if case=='lazy_conjugate':bias=torch.nn.Parameter(torch.ones(8,dtype=torch.complex64).conj(),requires_grad=False)
    if case=='strided':bias=torch.nn.Parameter(torch.zeros(16,dtype=torch.float16)[::2],requires_grad=False)
    with torch.no_grad():
        if case=='foreign_detach':
            with patch.object(torch.Tensor,'detach',lambda x:x):raw,why=normalized_native_bias(bias)
        else:raw,why=normalized_native_bias(bias)
    assert raw is None and why is not None


def test_full_guard_uses_alias_and_fallback_keeps_original_parameter():
    from types import SimpleNamespace
    from unittest.mock import PropertyMock
    from solution import guarded_forward as g
    n,k,m=256,1536,65;blocks=n*k//64
    bias=torch.nn.Parameter(torch.zeros(n,dtype=torch.float16))
    args=[torch.zeros(m,k,dtype=torch.float16),torch.zeros(n*k//2,1,dtype=torch.uint8),[n,k],
          torch.ones((blocks+255)//256),64,'nf4',bias,torch.zeros(blocks,dtype=torch.uint8),torch.zeros(256),torch.zeros(())]
    native_calls=[];launched=[];token=object()
    def native(*a):native_calls.append(a);return token
    original_empty=torch.empty
    def cpu_empty(*a,**kw):kw.pop('device',None);return original_empty(*a,**kw)
    class FakeKernel:
        def __getitem__(self,grid):
            def launch(*a,**kw):
                assert type(a[6]) is torch.Tensor and a[6].data_ptr()==bias.data_ptr()
                assert not a[6].requires_grad and kw['enable_fp_fusion'] is False
                launched.append(grid);return object()
            return launch
    replacement=g.make_replacement(native)
    with patch.object(torch.Tensor,'device',new_callable=PropertyMock,return_value=torch.device('cuda:0')),patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(major=7,minor=5,multi_processor_count=40)),patch.object(torch,'empty',cpu_empty),patch.object(g,'soup_nf4_mma_laneprobe',FakeKernel()):
        assert replacement(*args) is token
        assert native_calls[-1][6] is bias
        with torch.no_grad():out=replacement(*args)
    assert out.shape==(m,n) and launched==[(5,32)] and len(native_calls)==1
    assert bias.requires_grad and not torch.cuda.is_initialized()


@pytest.mark.parametrize('activation_shape',[(3,8),(2,3,8)])
@pytest.mark.parametrize('entry',['autograd_function','actual_Linear4bit_module'])
@pytest.mark.parametrize('scale',[1.,128.,65536.])
@pytest.mark.parametrize('bias_requires_grad',[False,True])
def test_actual_native_matmul_methods_keep_parameter_gradient_owner(scale,bias_requires_grad,entry,activation_shape):
    # Actual installed bnb MatMul4Bit.forward/backward are used; only GPU op and
    # dequant calls are replaced by tiny CPU tensors. Not an actual NF4/GPU test.
    from types import SimpleNamespace
    from bitsandbytes.autograd._functions import MatMul4Bit
    from bitsandbytes import functional as F
    expected=json.loads((ROOT/'solution/native_source_manifest.json').read_text())
    assert hashlib.sha256(inspect.getsource(MatMul4Bit.backward).encode()).hexdigest()==expected['backward']
    backward=MatMul4Bit.backward
    torch.manual_seed(51);w=torch.randn(5,8,dtype=torch.float16)*.05
    base_a=torch.randn(activation_shape,dtype=torch.float16)*.1;base_bias=torch.randn(5,dtype=torch.float16)*.02
    packed=torch.zeros(20,1,dtype=torch.uint8)
    state=SimpleNamespace(shape=(5,8),nested=False,absmax=torch.ones(1),blocksize=64,quant_type='nf4')
    def one(normalize):
        a=base_a.clone().requires_grad_();bias=torch.nn.Parameter(base_bias.clone(),requires_grad=bias_requires_grad)
        module=None
        if entry=='actual_Linear4bit_module':
            from bitsandbytes.nn import Linear4bit,Params4bit
            module=Linear4bit(8,5,bias=True,compute_dtype=torch.float16,quant_type='nf4')
            module.train();module.bias=bias
            module.weight=Params4bit(packed,requires_grad=False,quant_state=state,bnb_quantized=True)
        original_id=id(bias);calls=[]
        def gemm(A,B,shape,absmax,blocksize,quant_type,bias=None):
            assert not torch.is_grad_enabled() and type(bias) is torch.nn.Parameter
            assert type(B) is torch.Tensor
            raw,why=normalized_native_bias(bias) if normalize else (bias,None)
            assert why is None and raw.data_ptr()==bias.data_ptr()
            calls.append(type(raw).__name__)
            return torch.nn.functional.linear(A,w,raw)
        before=torch.get_rng_state().clone()
        with patch.object(torch.ops.bitsandbytes.gemm_4bit,'default',gemm),patch.object(F,'dequantize_4bit',lambda B,state:w):
            out=module(a) if module is not None else MatMul4Bit.apply(a,packed,None,bias,state)
            upstream=torch.linspace(-.02,.02,out.numel()).reshape_as(out).half()*scale
            out.backward(upstream)
        assert id(bias)==original_id and MatMul4Bit.backward is backward
        assert torch.equal(before,torch.get_rng_state())
        assert calls==(['Tensor'] if normalize else ['Parameter'])
        return out.detach(),a.grad,bias.grad
    original,normalized=one(False),one(True)
    for a,b in zip(original,normalized):
        if a is None:assert b is None
        else:same(a,b)
    assert not torch.cuda.is_initialized()
