import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from solution.contracts import native,CpuMixed,fixture,extreme_diagnostics
from solution.branch_capture import BranchCapture,replay,compare
class Attn(torch.nn.Module):
 def forward(self,x):return (x.half()*.37,None)
class Norm(torch.nn.Module):
 def __init__(self,h):super().__init__();self.weight=torch.nn.Parameter(torch.ones(h),requires_grad=False)
 def forward(self,x):return self.weight*(x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6))
class Layer(torch.nn.Module):
 def __init__(self):super().__init__();self.self_attn=Attn();self.post_attention_layernorm=Norm(1536)
 def forward(self,hidden_states):
  b,_=self.self_attn(hidden_states);z=hidden_states+b;y=self.post_attention_layernorm(z);return z+torch.nn.functional.silu(y).half()
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def test_mixed_scales_and_branches(self):
  for scale in (1.,128.,65536.):
   for use in ('both','norm_only','residual_only'):
    a,b,w=fixture(7);b=b.detach().half().requires_grad_();y,z=CpuMixed.apply(a,b,w);ey,ez=native(a,b,w)
    dy=torch.linspace(-.7,.9,a.numel()).reshape_as(a)*scale;dz=torch.full_like(a,.1)*scale
    select=(0,1) if use=='both' else (0,) if use=='norm_only' else (1,)
    g=torch.autograd.grad(tuple((y,z)[i] for i in select),(a,b),tuple((dy,dz)[i] for i in select));eg=torch.autograd.grad(tuple((ey,ez)[i] for i in select),(a,b),tuple((dy,dz)[i] for i in select))
    self.assertEqual((g[0].dtype,g[1].dtype),(torch.float32,torch.float16))
    for x,e in zip(g,eg):compare(x,e,scale=scale)
 def test_capture_real_branch_cotangents(self):
  layer=Layer();a,_,w=fixture(7)
  with BranchCapture(layer) as cap:
   out=layer(hidden_states=a);out.square().mean().backward()
  result=replay(cap,layer.post_attention_layernorm.weight,CpuMixed.apply,1.)
  self.assertEqual(result['activations']['b']['dtype'],'torch.float16');self.assertTrue(result['residual_exact'])
  for m in layer.modules():self.assertFalse(m._forward_hooks);self.assertFalse(m._forward_pre_hooks)
 def test_capture_rejects_duplicate(self):
  layer=Layer();a,_,_=fixture(1)
  with self.assertRaisesRegex(RuntimeError,'one native'):
   with BranchCapture(layer):layer(a);layer(a)
  for m in layer.modules():self.assertFalse(m._forward_hooks);self.assertFalse(m._forward_pre_hooks)
 def test_capture_cannot_forge_missing_backward(self):
  layer=Layer();a,_,_=fixture(1)
  with BranchCapture(layer) as cap:layer(a)
  with self.assertRaisesRegex(RuntimeError,'cotangents absent'):cap.require()
 def test_extreme_association_retained(self):
  r=extreme_diagnostics(CpuMixed.apply);self.assertEqual((r[0]['min'],r[3]['min']),(0.,1.))
 def test_mixed_edge_rounding_not_precast(self):
  a=torch.full((1,1536),512.,requires_grad=True);b=torch.full_like(a,512.,dtype=torch.float16,requires_grad=True);w=torch.ones(1536)
  y,z=CpuMixed.apply(a,b,w);g=torch.autograd.grad(z,(a,b),torch.full_like(z,65536.))
  self.assertTrue(torch.isfinite(g[0]).all());self.assertTrue(torch.isposinf(g[1]).all())
if __name__=='__main__':unittest.main(verbosity=2)
