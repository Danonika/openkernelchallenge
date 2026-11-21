"""CPU native-body fixture, no GPU/task/data/model download.

Exact installed HF/SFT method AST bodies are executed in lightweight classes to
avoid installing missing dataset dependencies. This is not full Trainer or PEFT.
"""
import ast,collections,copy,contextlib,sys,warnings
from pathlib import Path
from types import MethodType,SimpleNamespace
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,'/redacted/local/soup_cycle/accuracy_no_shift_copy_control/trl_target')
from accelerate.utils.operations import convert_outputs_to_fp32,convert_to_fp32,ConvertOutputsToFp32,recursively_apply
from torch.amp.autocast_mode import autocast_decorator
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM,Qwen2Config
from transformers.loss.loss_utils import ForCausalLMLoss
from trl.trainer.utils import entropy_from_logits
from soup_cli.trainer.rewind_hf import make_rewind_trainer_class,RewindState,row_losses
from solution.scoped_head import Owners,Installation,prepared_chain_reason,_ACTIVE
from solution.telemetry import Telemetry

def native_body(path,clsname,newname,base):
    tree=ast.parse(Path(path).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==clsname)
    method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute_loss'))
    tree=ast.Module(body=[ast.ClassDef(name=newname,bases=[ast.Name(id='BASE',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])],type_ignores=[])
    ast.fix_missing_locations(tree)
    scope={'BASE':base,'torch':torch,'nn':torch.nn,'Any':object,'entropy_from_logits':entropy_from_logits,'warnings':warnings}
    exec(compile(tree,str(path),'exec'),scope)
    return scope[newname]
HF=native_body('/redacted/local/soup_cloud_env/venv/lib/python3.12/site-packages/transformers/trainer.py','Trainer','HF',object)
SFT=native_body('/redacted/local/soup_cycle/accuracy_no_shift_copy_control/trl_target/trl/trainer/sft_trainer.py','SFTTrainer','SFT',HF)
Rewind=make_rewind_trainer_class(SFT)
class Tuner(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,**kw):return self.model(**kw)
class PeftFixture(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.base_model=Tuner(model)
        self.active_peft_config=SimpleNamespace(is_prompt_learning=False,peft_type='LORA',alora_invocation_tokens=None)
    def get_base_model(self):return self.base_model.model
    def forward(self,**kw):return self.base_model(**kw)
class Gather:
    num_processes=1;parallelism_config=None;native_amp=True;mixed_precision='fp16';device=torch.device('cpu')
    def __init__(self):self.values=[]
    def gather_for_metrics(self,x):self.values.append(x.detach().clone());return x
class Sink:
    def __init__(self):self.rows=[]
    def record_batch(self,**kw):self.rows.append(kw)

def setup():
    torch.manual_seed(711)
    qwen=Qwen2ForCausalLM(Qwen2Config(vocab_size=13,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,tie_word_embeddings=True,attention_dropout=.05,attn_implementation='eager'))
    qwen.requires_grad_(False);qwen.model.layers[0].self_attn.q_proj.weight.requires_grad_(True);qwen.model.layers[0].self_attn.v_proj.weight.requires_grad_(True)
    model=PeftFixture(qwen);model.train();model._original_forward=model.forward
    amp=torch.autocast('cpu',dtype=torch.float16)(PeftFixture.forward)
    model.forward=MethodType(convert_outputs_to_fp32(amp),model)
    t=Rewind.__new__(Rewind);t.model=model;t.model_wrapped=model;t.accelerator=Gather()
    t.args=SimpleNamespace(use_liger_kernel=False,loss_type='nll',padding_free=False,past_index=-1,average_tokens_across_devices=False,n_gpu=0)
    t.num_virtual_tokens=0;t.aux_loss_enabled=False;t.compute_loss_func=None;t.label_smoother=None;t.model_accepts_loss_kwargs=True
    t._metrics={'train':collections.defaultdict(list),'eval':collections.defaultdict(list)};t._total_train_tokens=0;t.state=SimpleNamespace(global_step=0)
    sink=Sink();state=RewindState(sink);state.on_index(3);state.on_index(8);t._soup_rewind_state=state
    inputs={'input_ids':torch.tensor([[1,2,3,4],[2,5,5,6]]),'labels':torch.tensor([[1,2,3,4],[2,-100,5,6]]),'attention_mask':torch.ones(2,4,dtype=torch.int64)}
    owners=Owners(Rewind,SFT,HF,PeftFixture,Qwen2ForCausalLM,Tuner,
        Rewind.compute_loss,SFT.compute_loss,HF.compute_loss,PeftFixture.forward,Qwen2ForCausalLM.forward,Tuner.forward,
        ForCausalLMLoss,entropy_from_logits,row_losses,PeftFixture.get_base_model,torch.nn.Linear.forward,ConvertOutputsToFp32,
        ConvertOutputsToFp32.__call__,convert_to_fp32,recursively_apply,convert_outputs_to_fp32(lambda:None).__code__,
        autocast_decorator(None,lambda:None).__code__,device_type='cpu',vocabulary=13,batch_size=2)
    return t,model,inputs,owners,sink

def same(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    if a.is_floating_point():
        assert torch.equal(a.isnan(),b.isnan()) and torch.equal(a.isposinf(),b.isposinf()) and torch.equal(a.isneginf(),b.isneginf())
        f=torch.isfinite(a)&torch.isfinite(b);assert torch.equal(a[f],b[f]);assert torch.equal(a.signbit()[a==0],b.signbit()[b==0])
    else:assert torch.equal(a,b)

@pytest.mark.parametrize('scale,infinite',[(1.,False),(128.,False),(65536.,False),(128.,True)])
def test_native_body_loss_metrics_rewind_gradscaler(scale,infinite):
    results=[]
    for patched in [False,True]:
        t,m,inputs,o,sink=setup();stats=Telemetry(False);h=Installation(o,stats) if patched else None
        params={n:p for n,p in m.named_parameters() if p.requires_grad};opt=torch.optim.AdamW(params.values(),lr=.001)
        scaler=torch.amp.GradScaler('cpu',init_scale=scale,growth_interval=100)
        try:
            assert prepared_chain_reason(m,o) is None
            loss=t.compute_loss(m,inputs,num_items_in_batch=torch.tensor(5))
            (scaler.scale(loss)*(torch.tensor(float('inf')) if infinite else 1)).backward()
            scaled={n:p.grad.detach().clone() for n,p in params.items()};scaler.unscale_(opt)
            unscaled={n:p.grad.detach().clone() for n,p in params.items()};scaler.step(opt);scaler.update()
            assert not m.get_base_model().lm_head._forward_hooks and _ACTIVE.get() is None
            results.append({'loss':loss.detach(),'scaled':scaled,'unscaled':unscaled,'params':{n:p.detach().clone() for n,p in params.items()},'metrics':dict(t._metrics['train']),'rewind':sink.rows,'scaler':scaler.state_dict(),'rng':torch.random.get_rng_state(),'stats':stats.snapshot()})
            if patched:assert stats.counts['shared_cast']==stats.counts['owned_loss_return']==1 and not stats.counts['native_fallback']
        finally:
            if h:assert h.restore()
    a,b=results;same(a['loss'],b['loss']);same(a['rng'],b['rng'])
    for k in ['scaled','unscaled','params']:
        for n in a[k]:same(a[k][n],b[k][n])
    assert a['metrics']==b['metrics'] and a['rewind']==b['rewind'] and a['scaler']==b['scaler']
    assert a['rewind'][0]['rows']==[3,8] and a['rewind'][0]['row_tokens']==[3,2]

@pytest.mark.parametrize('case',['outputs','eval','custom_loss','label_smoother','aux','foreign_head_hook','backward_pre_hook','global_hook','global_backward_pre','changed_linear','changed_converter','wrong_amp','unprepared','extra_input','nonnative_loss','sft_changed','rewind_rows_changed','entropy_changed','foreign_get_base','compiled_head','nested'])
def test_pre_forward_fallbacks(case):
    t,m,inputs,o,sink=setup();stats=Telemetry(False);h=Installation(o,stats);cleanup=[];kwargs={}
    try:
        if case=='outputs':kwargs['return_outputs']=True
        elif case=='eval':m.eval()
        elif case=='custom_loss':t.compute_loss_func=lambda outputs,labels,**kw:outputs.loss
        elif case=='label_smoother':t.label_smoother=lambda outputs,labels,**kw:outputs.loss
        elif case=='aux':t.aux_loss_enabled=True
        elif case=='foreign_head_hook':cleanup.append(m.get_base_model().lm_head.register_forward_hook(lambda m,a,o:o).remove)
        elif case=='backward_pre_hook':cleanup.append(m.get_base_model().lm_head.register_full_backward_pre_hook(lambda m,g:g).remove)
        elif case=='global_hook':cleanup.append(torch.nn.modules.module.register_module_forward_hook(lambda m,a,o:o).remove)
        elif case=='global_backward_pre':cleanup.append(torch.nn.modules.module.register_module_full_backward_pre_hook(lambda m,g:g).remove)
        elif case=='changed_linear':
            original=torch.nn.Linear.forward;torch.nn.Linear.forward=lambda self,x:original(self,x);cleanup.append(lambda:setattr(torch.nn.Linear,'forward',original))
        elif case=='changed_converter':m.forward=lambda **kw:m._original_forward(**kw)
        elif case=='wrong_amp':m.forward=MethodType(convert_outputs_to_fp32(torch.autocast('cpu',dtype=torch.bfloat16)(PeftFixture.forward)),m)
        elif case=='unprepared':m.forward=m._original_forward;del m._original_forward
        elif case=='extra_input':inputs['use_cache']=False
        elif case=='nonnative_loss':m.get_base_model().loss_function=lambda **kw:ForCausalLMLoss(**kw)
        elif case=='sft_changed':
            SFT.compute_loss=lambda *a,**kw:o.sft_loss(*a,**kw);cleanup.append(lambda:setattr(SFT,'compute_loss',o.sft_loss))
        elif case=='rewind_rows_changed':
            gl=o.rewind_loss.__globals__;gl['row_losses']=lambda *a,**kw:row_losses(*a,**kw);cleanup.append(lambda:gl.__setitem__('row_losses',row_losses))
        elif case=='entropy_changed':
            gl=o.sft_loss.__globals__;gl['entropy_from_logits']=lambda *a,**kw:entropy_from_logits(*a,**kw);cleanup.append(lambda:gl.__setitem__('entropy_from_logits',entropy_from_logits))
        elif case=='foreign_get_base':m.get_base_model=lambda:m.base_model.model
        elif case=='compiled_head':m.get_base_model().lm_head._compiled_call_impl=lambda *a:None
        elif case=='nested':
            token=_ACTIVE.set(object());cleanup.append(lambda:_ACTIVE.reset(token))
        reason,_=h.guard(t,m,inputs,kwargs.get('return_outputs',False));assert reason is not None

        assert stats.counts['shared_cast']==0
    finally:
        for f in reversed(cleanup):f()
        assert h.restore()


def test_return_outputs_executes_native_and_no_stale_hook():
    t,m,inputs,o,sink=setup();stats=Telemetry(False);h=Installation(o,stats)
    try:
        loss,out=t.compute_loss(m,inputs,return_outputs=True,num_items_in_batch=5)
        assert out.logits.dtype==torch.float32 and stats.counts['shared_cast']==0
        assert not m.get_base_model().lm_head._forward_hooks and _ACTIVE.get() is None
        loss.backward()
    finally:assert h.restore()


def test_exception_cleanup():
    t,m,inputs,o,sink=setup();stats=Telemetry(False);h=Installation(o,stats)
    def explode(*args):raise ValueError('native decoder fixture exception')
    bad=m.get_base_model().model.layers[0].register_forward_pre_hook(explode)
    try:
        with pytest.raises(ValueError,match='native decoder'):t.compute_loss(m,inputs,num_items_in_batch=5)
        assert not m.get_base_model().lm_head._forward_hooks and _ACTIVE.get() is None
    finally:bad.remove();assert h.restore()


def test_foreign_restore_preserved():
    t,m,inputs,o,sink=setup();stats=Telemetry(False);h=Installation(o,stats)
    foreign=lambda *a,**kw:None;Rewind.compute_loss=foreign
    try:assert not h.restore() and Rewind.compute_loss is foreign
    finally:Rewind.compute_loss=o.rewind_loss

def native_linear_with_docstring(self,input):
    """Runs the forward pass (present in target Torch2.11)."""
    return F.linear(input,self.weight,self.bias)

def native_linear_without_docstring(self,input):
    return F.linear(input,self.weight,self.bias)

def foreign_linear_extra_work(self,input):
    value=F.linear(input,self.weight,self.bias)
    return value

@pytest.mark.parametrize('fn,accepted',[(native_linear_with_docstring,True),(native_linear_without_docstring,True),(foreign_linear_extra_work,False)])
def test_exact_installer_body_check_accepts_only_optional_native_docstring(fn,accepted):
    import inspect
    tree=ast.parse((ROOT/'solution/install.py').read_text())
    method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install_pinned')
    begin=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='node' for t in n.targets))
    end=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='owners' for t in n.targets))
    part=ast.Module(body=method.body[begin:end],type_ignores=[])
    scope={'linear':fn,'inspect':inspect,'ast':ast,'textwrap':__import__('textwrap')}
    if accepted:exec(compile(part,'exact_installer_body_guard','exec'),scope)
    else:
        with pytest.raises(RuntimeError,match='Foreign Linear forward body'):exec(compile(part,'exact_installer_body_guard','exec'),scope)


@pytest.mark.parametrize('name,fn',[
    ('entropy',entropy_from_logits),('rewind_rows',row_losses),
    ('converter_call',ConvertOutputsToFp32.__call__),('converter_float',convert_to_fp32),
    ('converter_factory',convert_outputs_to_fp32),('converter_recurse',recursively_apply),
    ('autocast_decorator',autocast_decorator)])
def test_preinstalled_wrapper_cannot_be_blessed_by_capture(name,fn):
    from functools import wraps
    from solution.install import verify_callable
    verify_callable(name,fn)
    @wraps(fn)
    def foreign(*a,**kw):return fn(*a,**kw)
    with pytest.raises(RuntimeError,match='Foreign callable wrapper'):
        verify_callable(name,foreign)


def test_web_preflight_snapshot_survives_asymmetric_native_output_probe():
    from solution.web_preflight import snapshot_observations
    t,m,inputs,o,sink=setup();stats=Telemetry(False);h=Installation(o,stats)
    try:
        t.compute_loss(m,inputs,num_items_in_batch=5)
        shallow=dict(t._metrics['train'])
        frozen=snapshot_observations(t._metrics['train'],sink.rows,{'scale':1.0})
        expected=copy.deepcopy(frozen)
        for row in [3,8]:t._soup_rewind_state.on_index(row)
        t.compute_loss(m,inputs,return_outputs=True,num_items_in_batch=5)
        assert len(shallow['entropy'])==len(shallow['mean_token_accuracy'])==2
        assert shallow['num_tokens']==frozen['metrics']['num_tokens']
        assert all(len(v)==1 for v in frozen['metrics'].values())
        assert len(sink.rows)==2 and len(frozen['rewind'])==1
        assert frozen==expected
        assert not t._soup_rewind_state.failed and not t._soup_rewind_state.dropped
        assert stats.counts['shared_cast']==1 and stats.counts['native_fallback']==1
    finally:assert h.restore()
