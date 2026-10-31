"""Bounded diagnostic hooks only; never modifies native module outputs/gradients."""
import torch

class BranchCapture:
    def __init__(self,layer):
        self.layer=layer;self.handles=[];self.tensors={};self.counts={};self.gradients={}
    def save(self,key,tensor):
        if type(tensor) is not torch.Tensor:raise TypeError('Native branch tensor required: '+key)
        self.counts[key]=self.counts.get(key,0)+1
        if self.counts[key]!=1:raise RuntimeError('Expected one native decoder call: '+key)
        self.tensors[key]=tensor.detach().clone()
        if tensor.requires_grad:
            def observe(gradient):
                if key in self.gradients:raise RuntimeError('Repeated branch gradient: '+key)
                self.gradients[key]=gradient.detach().clone()
                return None
            self.handles.append(tensor.register_hook(observe))
    def __enter__(self):
        # with_kwargs catches the actual Qwen model's keyword-only hidden input.
        def before(module,args,kwargs):self.save('a',args[0] if args else kwargs['hidden_states'])
        def attn(module,args,output):self.save('b',output[0])
        def norm_before(module,args):self.save('z',args[0])
        def norm_after(module,args,output):self.save('y',output)
        def decoder_after(module,args,output):self.save('output',output)
        self.handles += [self.layer.register_forward_pre_hook(before,with_kwargs=True),
            self.layer.self_attn.register_forward_hook(attn),
            self.layer.post_attention_layernorm.register_forward_pre_hook(norm_before),
            self.layer.post_attention_layernorm.register_forward_hook(norm_after),
            self.layer.register_forward_hook(decoder_after)]
        return self
    def __exit__(self,*exc):
        for handle in reversed(self.handles):handle.remove()
        self.handles.clear()
    def require(self):
        if self.counts!={k:1 for k in ('a','b','z','y','output')}:raise RuntimeError('Native capture incomplete')
        if not set(('b','z','y','output'))<=set(self.gradients):raise RuntimeError('Real native branch cotangents absent')
        return self

def summary(tensor):
    finite=torch.isfinite(tensor);values=tensor[finite].float()
    return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'finite':int(finite.sum()),
        'nan':int(tensor.isnan().sum()),'posinf':int(tensor.isposinf().sum()),'neginf':int(tensor.isneginf().sum()),
        'absolute_max':float(values.abs().max()) if values.numel() else None,
        'rms':float(values.square().mean().sqrt()) if values.numel() else None}

def compare(actual,expected,*,scale=1.,forward=False):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    for fn in (torch.isnan,torch.isposinf,torch.isneginf):assert torch.equal(fn(actual),fn(expected)),'nonfinite classes changed'
    finite=torch.isfinite(actual)&torch.isfinite(expected)
    a=actual[finite].float()/scale;b=expected[finite].float()/scale
    torch.testing.assert_close(a,b,rtol=1e-4 if forward else 1e-3,atol=1e-5 if forward else 2e-5)
    return {'max_abs_scaled':float((a-b).abs().max().detach()) if a.numel() else 0.,'scale':scale,
        'nonfinite_class_exact':True,'zero_sign_disagreements':int(((actual==0)&(expected==0)&(torch.signbit(actual)!=torch.signbit(expected))).sum())}

def replay(capture,weight,fn,scale):
    capture.require();a=capture.tensors['a'].clone().requires_grad_();b=capture.tensors['b'].clone().requires_grad_()
    y,z=fn(a,b,weight)
    assert torch.equal(z,capture.tensors['z']),'Native mixed residual changed'
    forward=compare(y,capture.tensors['y'],forward=True)
    ga,gb=torch.autograd.grad((y,z),(a,b),(capture.gradients['y'],capture.gradients['output']))
    # Native z receives both norm and identity-bypass contributions. Native b
    # has no second use, so its captured adjoint proves the FP32->FP16 edge.
    grads={'a_add_only':compare(ga,capture.gradients['z'],scale=scale),
           'b_attention_output':compare(gb,capture.gradients['b'],scale=scale)}
    return {'forward':forward,'residual_exact':True,'gradients':grads,
        'activations':{k:summary(v) for k,v in capture.tensors.items()},
        'incoming_cotangents':{k:summary(capture.gradients[k]) for k in ('y','output')},
        'native_total_z_gradient':summary(capture.gradients['z'])}
