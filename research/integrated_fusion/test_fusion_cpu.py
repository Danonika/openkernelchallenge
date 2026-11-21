"""CPU mathematical gates for the new *integrated* original Soup candidate.

These are NOT T4/QLoRA parity or the official Soup 79 regression tests.
"""
import ast
import importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
PACK_PATH=ROOT/'solution'/'packed_decoder.py'
spec=importlib.util.spec_from_file_location('candidate_packed',PACK_PATH)
pack=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


def test_source_integrated_apply():
    entry=ast.parse((ROOT/'solution'/'__init__.py').read_text())
    funcs=[x for x in entry.body if isinstance(x,ast.FunctionDef) and x.name=='apply']
    assert len(funcs)==1 and len(funcs[0].args.args)==0
    calls={x.func.id for x in ast.walk(funcs[0]) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)}
    expected={'patch_bitsandbytes_nf4','patch_frozen_storage','patch_qwen2_mlp','patch_compact_mlp_activations','patch_frozen_mlp','patch_mlp_padding','patch_qwen2_rmsnorm','patch_streamed_training','install_decoder_packing'}
    assert calls >= expected
    integration=(ROOT/'solution'/'integration.py').read_text()
    assert 'with labelled_training(current_inputs["labels"]):' in integration
    assert 'Trainer.compute_loss(' in integration
    source=(ROOT/'solution'/'packed_decoder.py').read_text()
    assert "if (counters['dropout'] != len(dropout_ids)" in source
    for f in (ROOT/'solution').glob('*.py'):
        ast.parse(f.read_text(),filename=f.name)


def test_segmented_attention_cpu_numerics():
    for lengths in ([5,11,8,15],[3,5,2],[1,4],[7,7,7]):
        batch=len(lengths); width=max(lengths)
        mask=(torch.arange(width)[None,:]<torch.tensor(lengths)[:,None]).long()
        assert pack._valid_right_padded(mask)
        indices,pos,bounds=pack._batch_plan(mask)
        assert len(indices)==sum(lengths)
        torch.manual_seed(73)
        q=torch.randn(batch,4,width,16,requires_grad=True)
        k=torch.randn(batch,2,width,16,requires_grad=True)
        v=torch.randn(batch,2,width,16,requires_grad=True)
        dense=[]
        for row, length in enumerate(lengths):
            dense.append(F.scaled_dot_product_attention(
                q[row:row+1,:,:length],k[row:row+1,:,:length],v[row:row+1,:,:length],
                is_causal=True,enable_gqa=True))
        expected=torch.cat(dense,dim=-2)
        qpacked=q.permute(0,2,1,3).reshape(batch*width,4,16).index_select(0,indices).transpose(0,1).unsqueeze(0)
        kpacked=k.permute(0,2,1,3).reshape(batch*width,2,16).index_select(0,indices).transpose(0,1).unsqueeze(0)
        vpacked=v.permute(0,2,1,3).reshape(batch*width,2,16).index_select(0,indices).transpose(0,1).unsqueeze(0)
        actual=torch.cat([F.scaled_dot_product_attention(
            qpacked[:,:,lo:hi],kpacked[:,:,lo:hi],vpacked[:,:,lo:hi],
            is_causal=True,enable_gqa=True) for lo,hi in bounds],dim=-2)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6)
        gout=torch.randn_like(actual)
        expected_grads=torch.autograd.grad(expected,(q,k,v),gout,retain_graph=True)
        actual_grads=torch.autograd.grad(actual,(q,k,v),gout)
        for actual_grad,expected_grad in zip(actual_grads,expected_grads):
            torch.testing.assert_close(actual_grad,expected_grad,rtol=2e-5,atol=5e-6)


def test_native_padded_dropout_fp16_and_fp32_rng():
    for dtype in [torch.float16,torch.float32]:
        for seed in [1,3,9,89,333]:
            b,t,d=4,7,64
            mask=(torch.arange(t)[None,:]<torch.tensor([3,7,1,5])[:,None]).long()
            indices,_,_=pack._batch_plan(mask)
            full=torch.randn(b,t,d,dtype=dtype)
            packed=full.reshape(b*t,d).index_select(0,indices).view(1,-1,d)
            plan=(b,t,indices,{}, {'dropout':0})
            torch.manual_seed(seed)
            expected=F.dropout(full,p=.05,training=True).reshape(b*t,d).index_select(0,indices).view_as(packed)
            expected_rng=torch.get_rng_state().clone()
            torch.manual_seed(seed)
            actual=pack._native_geometry_dropout(packed,plan,.05)
            assert torch.equal(torch.get_rng_state(),expected_rng), (dtype,seed,'RNG')
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            assert plan[-1]['dropout']==1


def test_fp16_output_rmsnorm_math_and_gradient():
    for n,d in [(1,128),(5,1536),(100,1536),(1,8960)]:
        torch.manual_seed(103+n+d)
        w=(torch.randn(d,dtype=torch.float32)*.1+1).detach()
        x=torch.randn(n,d,dtype=torch.float32,requires_grad=True)
        eps=1e-6
        out=(x*torch.rsqrt(x.square().mean(dim=-1,keepdim=True)+eps)*w).half()
        grad=torch.randn_like(out)
        dx,=torch.autograd.grad(out,x,grad)
        with torch.no_grad():
            inv=torch.rsqrt(x.detach().square().mean(dim=-1,keepdim=True)+eps)
            upstream=grad.float()*w
            expected=upstream*inv + (-.5*(upstream*x.detach()).sum(-1,keepdim=True)*inv.pow(3)/d)*(2*x.detach())
        torch.testing.assert_close(dx,expected,rtol=2e-5,atol=5e-6)


class _SaveUpOracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx,gate,up,g_input,w_gate,out_dtype):
        ctx.save_for_backward(up,g_input,w_gate)
        ctx.kind=out_dtype
        a=gate.float()
        return (a*torch.sigmoid(a)*up.float()).to(out_dtype)
    @staticmethod
    def backward(ctx,dout):
        up,g_input,w=ctx.saved_tensors
        gate=F.linear(g_input.detach(),w).float()
        sig=torch.sigmoid(gate)
        du=(dout.float()*gate*sig).to(up.dtype)
        dg=(dout.float()*up.float()*sig*(1+gate*(1-sig))).to(up.dtype)
        return dg,du,None,None,None


def test_frozen_mlp_saveup_both_input_types():
    for input_dtype in (torch.float16,torch.float32):
        for seed in range(15):
            torch.manual_seed(seed)
            d=32; inter=64; n=5
            weight_g=(torch.randn(inter,d)*0.03).half()
            weight_u=(torch.randn(inter,d)*0.03).half()
            weight_down=(torch.randn(d,inter)*0.03).half()
            x=(torch.randn(n,d)*0.1).to(input_dtype).requires_grad_(True)
            wg=x.half(); wu=x.half()
            gate=F.linear(wg,weight_g)
            up=F.linear(wu,weight_u)
            out_dtype=torch.float16 if input_dtype==torch.float16 else torch.float32
            ref_act=(gate.float()*torch.sigmoid(gate.float())*up.float()).to(out_dtype)
            ref_out=F.linear(ref_act.half(),weight_down).to(out_dtype)
            g=torch.randn_like(ref_out)
            dx0,=torch.autograd.grad(ref_out,x,g,retain_graph=True)
            actual_act=_SaveUpOracle.apply(gate,up,wg,weight_g,out_dtype)
            actual_out=F.linear(actual_act.half(),weight_down).to(out_dtype)
            dx1,=torch.autograd.grad(actual_out,x,g)
            torch.testing.assert_close(actual_out,ref_out,rtol=0,atol=0)
            torch.testing.assert_close(dx1,dx0,rtol=2.5e-3,atol=3e-5)
