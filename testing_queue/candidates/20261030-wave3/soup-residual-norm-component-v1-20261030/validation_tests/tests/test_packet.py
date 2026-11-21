import ast,base64,hashlib,json,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PACKET=ROOT/'microbench_packet_v1';KIT=Path('/redacted/local/soup_web_ready_20261030_v2/support')
class Tests(unittest.TestCase):
 def test_protected_helpers(self):
  for n in ('process_guard.py','owned_process_tree.py','runtime_probe.py','official_manifest.json'):self.assertEqual((PACKET/n).read_bytes(),(KIT/n).read_bytes())
  def defs(p):return {n.name:ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  old=defs(KIT/'web_runner.py');new=defs(PACKET/'reviewed_support.py')
  for name in ('sha','read','write','verify_official','idle_check','export_small','Exporter'):self.assertEqual(old[name],new[name])
 def test_payload_binding(self):
  nb=json.loads((ROOT/'residual_norm_MICROBENCH_ONLY_v1.ipynb').read_text());tree=ast.parse(''.join(nb['cells'][1]['source']));values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PAYLOAD','PAYLOAD_SHA256')}
  self.assertEqual(set(values['PAYLOAD']),set(values['PAYLOAD_SHA256']))
  for n,b in values['PAYLOAD'].items():
   raw=base64.b64decode(b,validate=True);self.assertEqual(raw,(PACKET/n).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),values['PAYLOAD_SHA256'][n])
 def test_old_original_comparator_immutable(self):self.assertEqual((ROOT/'old_solution/rmsnorm.py').read_bytes(),Path('/redacted/local/soup_gpu_trials/trial001/solution/rmsnorm.py').read_bytes())
 def test_source_zip_binding(self):
  manifest=json.loads((ROOT/'source_manifest.json').read_text())
  with zipfile.ZipFile(ROOT/'candidate.zip') as z:
   self.assertEqual(set(z.namelist()),set(manifest))
   for n,h in manifest.items():self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),h);self.assertEqual(z.read(n),(ROOT/n).read_bytes())
 def test_no_training_or_install(self):
  for p in (ROOT/'microbench/driver.py',ROOT/'microbench/worker.py',ROOT/'solution/residual_norm.py'):
   tree=ast.parse(p.read_text());names=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
   for forbidden in ('install_bundle','patch_qwen2_norm','frozen_rms32.patch_qwen2_norm','torch.compile','torch.cuda.CUDAGraph','torch.set_default_dtype'):self.assertFalse(any(n.endswith(forbidden) for n in names))
   self.assertNotIn('run_task.py',p.read_text());self.assertNotIn('from_pretrained',p.read_text())
 def test_fixed_caps_and_unique_paths(self):
  source=(ROOT/'microbench/driver.py').read_text();self.assertIn('timeout=80',source);self.assertIn('timeout=40',source);self.assertNotIn('timeout=900',source)
  names=[m+s for m in ('runtime_probe','preflight','native','standalone','fused') for s in ('.json','.log','.log.process.json')];self.assertEqual(len(names),len(set(names)))
 def test_no_fp16_output_or_grad_forcing(self):
  text=(ROOT/'solution/residual_norm.py').read_text();self.assertNotIn('float16',text);self.assertNotIn('requires_grad_',text);self.assertIn('set_materialize_grads(False)',text);self.assertIn('enable_fp_fusion=False',text)
 def test_extreme_witness_is_disclosed(self):
  nb=json.loads((ROOT/'residual_norm_MICROBENCH_ONLY_v1.ipynb').read_text());s=''.join(nb['cells'][0]['source']);self.assertIn('native0 vs atomic1',s);self.assertIn('universal equality is NOT claimed',s)
if __name__=='__main__':unittest.main()
