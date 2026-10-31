import ast,base64,hashlib,json,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PACKET=ROOT/'microbench_packet_v1';KIT=Path('/workspace/shared/soup_web_ready_20261030_v2/support')
COMMON=Path('/workspace/shared/soup_head_metric_bridge_v2/solution')
class Tests(unittest.TestCase):
 def test_protected_helpers(self):
  for n in ('process_guard.py','owned_process_tree.py','runtime_probe.py','official_manifest.json'):self.assertEqual((PACKET/n).read_bytes(),(KIT/n).read_bytes())
  def defs(p):return {n.name:ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  old=defs(KIT/'web_runner.py');new=defs(PACKET/'reviewed_support.py')
  for name in ('sha','read','write','verify_official','idle_check','export_small','Exporter'):self.assertEqual(old[name],new[name])
 def test_payload_binding(self):
  nb=json.loads((ROOT/'token_chunk_ce_COMPONENT_ONLY_v1.ipynb').read_text());tree=ast.parse(''.join(nb['cells'][1]['source']));values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PAYLOAD','PAYLOAD_SHA256')}
  self.assertEqual(set(values['PAYLOAD']),set(values['PAYLOAD_SHA256']))
  for n,b in values['PAYLOAD'].items():
   raw=base64.b64decode(b,validate=True);self.assertEqual(raw,(PACKET/n).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),values['PAYLOAD_SHA256'][n])
  for n,h in json.loads((PACKET/'payload_manifest.json').read_text()).items():self.assertEqual(hashlib.sha256((PACKET/n).read_bytes()).hexdigest(),h)
 def test_shared_common_consumer_is_byte_identical(self):
  manifest=json.loads((ROOT/'shared_consumer_manifest.json').read_text())
  for n,h in manifest.items():
   self.assertEqual((PACKET/'shared_consumer'/n).read_bytes(),(COMMON/n).read_bytes());self.assertEqual(hashlib.sha256((COMMON/n).read_bytes()).hexdigest(),h)
 def test_native_sources_bound_to_recovered_manifest(self):
  expected=json.loads((Path('/workspace/shared/soup_streaming_head_research_20261030/source_bindings.json')).read_text());target=json.loads((PACKET/'native_source_manifest.json').read_text())
  for key in ('native_loss','trl_entropy','soup_rewind','accelerate_operations'):
   p=Path(expected[key]['path']);self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),expected[key]['sha256']);self.assertEqual(target[p.name],expected[key]['sha256'])
 def test_source_zip_binding(self):
  manifest=json.loads((ROOT/'source_manifest.json').read_text())
  with zipfile.ZipFile(ROOT/'candidate.zip') as z:
   self.assertEqual(set(z.namelist()),set(manifest))
   for n,h in manifest.items():self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),h);self.assertEqual(z.read(n),(ROOT/n).read_bytes())
 def test_no_training_or_install_or_precision_change(self):
  for p in (ROOT/'microbench/driver.py',ROOT/'microbench/worker.py',ROOT/'solution/token_chunk_head.py'):
   tree=ast.parse(p.read_text());names=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
   for forbidden in ('install_bundle','patch_qwen2_norm','torch.compile','torch.cuda.CUDAGraph','torch.set_default_dtype','torch.set_float32_matmul_precision','ConsumerInstallation'):self.assertFalse(any(n.endswith(forbidden) for n in names))
   for forbidden in ('run_task.py','from_pretrained','torch.optim'):self.assertNotIn(forbidden,p.read_text())
 def test_fixed_caps_native_amp_and_real_gradscaler(self):
  source=(ROOT/'microbench/driver.py').read_text();self.assertIn('timeout=80',source);self.assertIn('timeout=40',source);self.assertNotIn('timeout=900',source)
  worker=(ROOT/'microbench/worker.py').read_text();self.assertIn("torch.amp.GradScaler('cuda',init_scale=65536.)",worker);self.assertIn('scaler.scale(loss)',worker);self.assertIn("with torch.autocast('cuda',dtype=torch.float16)",worker)
  self.assertIn("['native','old_chunk','ce_chunk']",source);self.assertIn("--method','numerical'",source)
 def test_first_call_and_observed_peak_distinguished(self):
  source=(ROOT/'microbench/worker.py').read_text();self.assertIn('component_first_call_absolute_peak_bytes=peak',source);self.assertIn('component_observed_absolute_peak_bytes=observed_peak',source)
  self.assertIn('torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()',source)
 def test_scopes_are_explicit(self):
  nb=json.loads((ROOT/'token_chunk_ce_COMPONENT_ONLY_v1.ipynb').read_text());s=''.join(nb['cells'][0]['source'])
  for expected in ('not full model/training memory','29s/3GB evidence','No duplicate consumer','No production integration is bundled'):self.assertIn(expected,s)
 def test_frozen_prior_head_and_original_eight_kernel_modules(self):
  old=Path('/workspace/shared/soup_streaming_head_research_20261030/target_component_v2/solution')
  for n in ('__init__.py','token_chunk_head.py'):self.assertEqual((ROOT/'old_solution'/n).read_bytes(),(old/n).read_bytes())
  manifest=json.loads((ROOT/'tile_source_manifest.json').read_text())
  for n,h in manifest.items():
   if n=='solution/__init__.py':continue
   self.assertEqual(hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),h)
   self.assertEqual((ROOT/n).read_bytes(),(PACKET/n).read_bytes())
 def test_original_gates_declared_before_target_and_real_recompute_contexts(self):
  s=(ROOT/'microbench/worker.py').read_text()
  self.assertIn("rtol=.01 if name in ('entropy','token_entropy') else 1e-4",s)
  self.assertIn("assert not torch.is_autocast_enabled('cuda')",s)
  self.assertIn("a=F.linear(hh[start:start+128],ww)",s)
  self.assertIn("'mixed_near_certain'",s);self.assertIn("'masked_nonfinite'",s)
  self.assertIn('report[\'tile_checks\']=verify_tile',s)
 def test_saved_state_and_single_global_normalization(self):
  s=(ROOT/'solution/token_chunk_head.py').read_text()
  self.assertIn('ctx.save_for_backward(*saved)',s)
  self.assertIn("total=F.nll_loss(-nll[:,None],internal_target,reduction='sum'",s)
  self.assertEqual(s.count('normalized=grad_loss/divisor'),1)
  self.assertEqual(s.count('prepare_upstream(normalized)'),1)
  self.assertIn('tile.state.tensors',s)
if __name__=='__main__':unittest.main()
