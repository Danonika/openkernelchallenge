"""CPU synthetic ownership tests, not PEFT/bitsandbytes/GPU integration."""
import functools
import importlib
import sys
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import pytest
import solution
from solution.lifecycle import Ownership


def fixture(fail=None):
    events=[]
    def native_fwd(x): return x+1
    def native_fun(x): return x+2
    ops=ModuleType('bitsandbytes.backends.cuda.ops');ops._dequant_linear_fallback=native_fwd
    fun=ModuleType('bitsandbytes.functional');fun.dequantize_4bit=native_fun
    class Layer:
        def forward(self,x): return ops._dequant_linear_fallback(x)+fun.dequantize_4bit(x)
    class Qwen:
        def __init__(self): self.training=True
    original_lora=Layer.forward;original_qwen=Qwen.__init__
    nf4=SimpleNamespace(_MARKER='_synthetic_nf4')
    def nf4_install():
        events.append('nf4')
        if fail=='nf4':return {'patched':False,'reason':'synthetic_refusal'}
        @functools.wraps(native_fwd)
        def fwd(x): return native_fwd(x)
        @functools.wraps(native_fun)
        def functional(x): return native_fun(x)
        setattr(fwd,nf4._MARKER,True);setattr(functional,nf4._MARKER,True)
        ops._dequant_linear_fallback=fwd
        if fail=="nf4_partial":raise RuntimeError("synthetic partial NF4 installation")
        fun.dequantize_4bit=functional
        return {'patched':True,'already_patched':False}
    nf4.install=nf4_install
    lora=SimpleNamespace(_OWNER=None)
    def lora_install():
        events.append('lora')
        assert ops._dequant_linear_fallback is not native_fwd
        if fail=='lora': raise RuntimeError('synthetic source mismatch')
        @functools.wraps(original_lora)
        def wrapped(self,x): return original_lora(self,x)
        Layer.forward=wrapped;lora._OWNER=(Layer,original_lora,wrapped)
    lora.install=lora_install
    ce=SimpleNamespace(_handle=None)
    def ce_apply():
        if ce._handle is not None and ce._handle.active:return True
        events.append('ce')
        if fail=='ce':return False
        @functools.wraps(original_qwen)
        def wrapped(self): original_qwen(self)
        Qwen.__init__=wrapped;ce._handle=SimpleNamespace(active=True,hook=wrapped)
        return True
    def ce_restore():
        if ce._handle is None:return {'constructor':'not_installed','model_losses':{}}
        if not ce._handle.active:return {'constructor':'already_restored','model_losses':{}}
        state='foreign_constructor_preserved'
        if Qwen.__init__ is ce._handle.hook:Qwen.__init__=original_qwen;state='restored'
        ce._handle.active=False
        return {'constructor':state,'model_losses':{}}
    ce.apply=ce_apply;ce.restore=ce_restore;ce.diagnostics=lambda:{'events':{}}
    bnb=ModuleType('bitsandbytes');backends=ModuleType('bitsandbytes.backends');cuda=ModuleType('bitsandbytes.backends.cuda')
    bnb.functional=fun;bnb.backends=backends;backends.cuda=cuda;cuda.ops=ops
    peft=ModuleType('peft');tuners=ModuleType('peft.tuners');lt=ModuleType('peft.tuners.lora');pb=ModuleType('peft.tuners.lora.bnb');pb.Linear4bit=Layer
    peft.tuners=tuners;tuners.lora=lt;lt.bnb=pb
    tf=ModuleType('transformers');models=ModuleType('transformers.models');q2=ModuleType('transformers.models.qwen2');mq=ModuleType('transformers.models.qwen2.modeling_qwen2');mq.Qwen2ForCausalLM=Qwen
    tf.models=models;models.qwen2=q2;q2.modeling_qwen2=mq
    modules={m.__name__:m for m in (bnb,backends,cuda,ops,fun,peft,tuners,lt,pb,tf,models,q2,mq)}
    for name,obj in (('nf4',nf4),('lora',lora)):
        package=ModuleType('solution.'+name+'_component')
        key='routes' if name=='nf4' else 'patch';setattr(package,key,obj)
        modules[package.__name__]=package;modules[package.__name__+'.'+key]=obj
    modules['solution.ce_component']=ce
    return locals()


@pytest.fixture
def install_fixture():
    installed=[]
    def enter(fail=None):
        f=fixture(fail);contexts=[patch.dict(sys.modules,f['modules']),
              patch.object(solution,'version',lambda name:solution._EXPECTED[name]),
              patch.object(solution,'_ownership',None),patch.object(solution,'_active',False),
              patch.object(solution,'ce_component',f['ce'],create=True),
              patch.object(solution,'nf4_component',f['modules']['solution.nf4_component'],create=True),
              patch.object(solution,'lora_component',f['modules']['solution.lora_component'],create=True)]
        for c in contexts:c.start()
        installed.append(contexts)
        return f
    yield enter
    for contexts in reversed(installed):
        for c in reversed(contexts):c.stop()


def assert_native(f):
    assert f['ops']._dequant_linear_fallback is f['native_fwd']
    assert f['fun'].dequantize_4bit is f['native_fun']
    assert f['Layer'].forward is f['original_lora']
    assert f['Qwen'].__init__ is f['original_qwen']


def test_disjoint_order_nested_fallback_and_idempotence(install_fixture):
    f=install_fixture();assert solution.apply() is True
    assert f['events']==['nf4','lora','ce'];assert f['Layer']().forward(5)==13
    assert solution.apply() is True and f['events']==['nf4','lora','ce']
    result=solution.restore();assert result['constructor']=='restored';assert_native(f)
    assert f['lora']._OWNER is None


@pytest.mark.parametrize('failed',['nf4','nf4_partial','lora','ce'])
def test_all_or_nothing_refusal_rolls_back(failed,install_fixture):
    f=install_fixture(failed)
    with pytest.raises(RuntimeError):solution.apply()
    assert_native(f);assert not solution._active;assert f['lora']._OWNER is None


def test_foreign_patch_is_preserved_and_reapply_refused(install_fixture):
    f=install_fixture();solution.apply()
    foreign=lambda x:-100
    f['fun'].dequantize_4bit=foreign
    with pytest.raises(RuntimeError,match='ownership changed'):solution.apply()
    result=solution.restore()
    assert f['fun'].dequantize_4bit is foreign
    assert result['composition_restoration']['nf4_functional:dequantize_4bit']=='foreign_preserved'
    assert f['ops']._dequant_linear_fallback is f['native_fwd']
    assert f['Layer'].forward is f['original_lora']


def test_preexisting_component_and_overlapping_bindings_refused(install_fixture):
    f=install_fixture();setattr(f['native_fwd'],f['nf4']._MARKER,True)
    with pytest.raises(RuntimeError,match='Pre-existing'):solution.apply()
    assert_native(f)
    obj=SimpleNamespace(call=object());previous=object();own=Ownership()
    own.add(obj,'call',previous,obj.call,'one')
    with pytest.raises(RuntimeError,match='Overlapping'):own.add(obj,'call',previous,obj.call,'two')
