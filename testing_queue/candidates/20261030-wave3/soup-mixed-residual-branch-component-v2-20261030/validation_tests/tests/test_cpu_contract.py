import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from solution.contracts import CpuBoundary,compare,extreme_diagnostics,fixture,H
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def test_realistic_scaled_stages(self):
  for rows in (1,7,2048):
   for scale in (.01,1.,10.):
    with self.subTest(rows=rows,scale=scale):compare(CpuBoundary.apply,rows,scale,65536.)
 def test_alias_and_branch_edges(self):
  for alias in ('independent','same','overlap'):
   for use in ('norm_only','residual_only','both'):
    with self.subTest(alias=alias,use=use):compare(CpuBoundary.apply,7,1.,128.,alias=alias,use=use)
 def test_one_required_input(self):compare(CpuBoundary.apply,7,only_a=True)
 def test_negative_native_association_is_retained(self):
  r=extreme_diagnostics(CpuBoundary.apply)
  self.assertEqual((r[0]['min'],r[3]['min']),(0.,1.))
  for i in (1,4):self.assertTrue(r[i]['all_one']);self.assertEqual(r[i]['nan'],0)
  for i in (2,5):self.assertEqual(r[i]['nan'],H)
 def test_mutated_saved_residual_rejected(self):
  a,b,w=fixture(1);y,z=CpuBoundary.apply(a,b,w)
  with torch.no_grad():z.add_(1)
  with self.assertRaisesRegex(RuntimeError,'modified by an inplace operation'):torch.autograd.grad(y,(a,b),torch.ones_like(y))
 def test_weight_version_rejected(self):
  a,b,w=fixture(1);y,z=CpuBoundary.apply(a,b,w)
  w.add_(1)
  with self.assertRaisesRegex(RuntimeError,'modified by an inplace operation'):torch.autograd.grad(y,(a,b),torch.ones_like(y))
 def test_no_grad_output_contract(self):
  a,b,w=fixture(1)
  with torch.no_grad():y,z=CpuBoundary.apply(a,b,w)
  self.assertFalse(y.requires_grad);self.assertFalse(z.requires_grad);self.assertEqual(y.dtype,torch.float32)
 def test_native_rng_unchanged(self):
  before=torch.get_rng_state().clone();compare(CpuBoundary.apply,7);self.assertTrue(torch.equal(before,torch.get_rng_state()))
if __name__=='__main__':unittest.main(verbosity=2)
