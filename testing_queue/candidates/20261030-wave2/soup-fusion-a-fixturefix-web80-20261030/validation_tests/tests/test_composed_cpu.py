"""Small composed CPU interpreter/native-fallback witness, NOT T4/PEFT integration."""
import ast,copy,os
os.environ['TRITON_INTERPRET']='1'
import torch
from solution.nf4_component.packed_nf4 import launch_into
from solution.lora_component.epilogue import _ScaleAdd
from solution.lora_component.patch import transform_forward
from solution.ce_component.materialized_ce import materialized_causal_loss,native_causal_loss,Diagnostics

torch.set_num_threads(2)
NF4=torch.tensor([-1.,-.6961928009986877,-.5250730514526367,-.39491748809814453,-.28444138169288635,-.18477343022823334,-.09105003625154495,0.,.07958029955625534,.16093020141124725,.24611230194568634,.33791524171829224,.44070982933044434,.5626170039176941,.7229568362236023,1.],dtype=torch.float32)
FORWARD='''
def forward(self, x):
    result = self.base(x).clone()
    lora_A, lora_B, dropout, scaling = self.a, self.b, self.drop, 2.0
    output = lora_B(lora_A(dropout(x))) * scaling
    result = result + output
    return result
'''

def exact(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.equal(a.isnan(),b.isnan()) and torch.equal(a.isposinf(),b.isposinf()) and torch.equal(a.isneginf(),b.isneginf())
    finite=a.isfinite()&b.isfinite();assert torch.equal(a[finite],b[finite])
    zeros=finite&(a==0)&(b==0);assert torch.equal(a.signbit()[zeros],b.signbit()[zeros])


def test_nf4_lora_ce_combined_values_scaled_vjp_rng_and_denominator():
    torch.manual_seed(1103)
    class Branch(torch.nn.Module):
        def __init__(self,width,weight):
            super().__init__();self.base=torch.nn.Linear(32,width,bias=False).half()
            self.base.weight.data.copy_(weight);self.base.weight.requires_grad_(False)
            self.a=torch.nn.Linear(32,16,bias=False).half();self.b=torch.nn.Linear(16,width,bias=False).half();self.drop=torch.nn.Dropout(.05)
    native=[];candidate=[]
    for width in (32,8):
        n=width*32;packed=torch.randint(0,256,(n//2,1),dtype=torch.uint8)
        qabs=torch.randint(0,256,(n//64,),dtype=torch.uint8)
        code=torch.linspace(-.5,.5,256);abs2=torch.tensor([.2]);offset=torch.tensor(.4)
        nibble=torch.stack((packed.reshape(-1)>>4,packed.reshape(-1)&15),dim=1).reshape(-1).long()
        scale=code[qabs.long()]*abs2[torch.arange(n//64)//256]+offset
        expected=(NF4[nibble]*scale[torch.arange(n)//64]).half().reshape(width,32)
        actual=torch.empty_like(expected);launch_into(packed,qabs,code,abs2,offset,actual)
        exact(actual,expected)
        branch=Branch(width,expected);native.append(branch);cloned=copy.deepcopy(branch);cloned.base.weight.data.copy_(actual);candidate.append(cloned)
    ns={};exec(compile(ast.parse(FORWARD),'<native-synthetic>','exec'),ns)
    cs={'_soup_original_scale_add':lambda b,d,s:_ScaleAdd.apply(b,d)}
    exec(compile(transform_forward(FORWARD),'<composed-synthetic>','exec'),cs)
    data=torch.randn(4,3,32,dtype=torch.float16);labels=torch.tensor([[0,1,39],[0,12,-100],[0,-100,38],[0,25,26]])
    denominator=torch.tensor(11,dtype=torch.int64);version=denominator._version
    rng=torch.random.get_rng_state().clone()
    def forward_backward(layers,fn,candidate_route):
        masks=[];handles=[layer.drop.register_forward_hook(lambda module,args,out:masks.append((out!=0).clone())) for layer in layers]
        x=data.clone().requires_grad_();outputs=[fn(layer,x) for layer in layers];logits=torch.cat(outputs,dim=-1)
        diag=Diagnostics()
        loss=(materialized_causal_loss(logits,labels,denominator,diagnostics=diag) if candidate_route else native_causal_loss(logits,labels,denominator))
        loss.backward(torch.tensor(65536.,dtype=torch.float32))
        for h in handles:h.remove()
        return logits.detach(),loss.detach(),x.grad,[p.grad for layer in layers for p in layer.parameters() if p.requires_grad],masks,torch.random.get_rng_state().clone(),diag
    a=forward_backward(native,ns['forward'],False);torch.random.set_rng_state(rng)
    b=forward_backward(candidate,cs['forward'],True)
    for x,y in zip(a[:3],b[:3]):exact(x,y)
    for x,y in zip(a[3],b[3]):exact(x,y)
    for x,y in zip(a[4],b[4]):assert torch.equal(x,y)
    assert not torch.equal(b[4][0],b[4][1])
    assert torch.equal(a[5],b[5]);assert denominator._version==version and denominator.item()==11
    assert b[6].events==[{'event':'native_forward_fallback','reason':'not_cuda'}]
