import ast,base64,copy,hashlib,importlib.util,json, pathlib,sys,unittest,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];PACKET=ROOT/'packet';KIT=pathlib.Path('/redacted/local/soup_web_ready_20261030_v2/support')
sys.path.insert(0,str(PACKET));import driver
class Tests(unittest.TestCase):
 def test_protected_helpers_byte_exact(self):
  for n in ('process_guard.py','owned_process_tree.py','runtime_probe.py','official_manifest.json','prepare_asset_manifest.py','evidence_reader.py'):self.assertEqual((PACKET/n).read_bytes(),(KIT/n).read_bytes())
  self.assertEqual((PACKET/'reviewed_support.py').read_bytes(),(KIT/'web_runner.py').read_bytes())
 def test_embedded_packet_hashes(self):
  nb=json.loads((ROOT/'stage_timing_DIAGNOSTIC_ONLY_v1.ipynb').read_text());code=''.join(nb['cells'][1]['source']);vals={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PAYLOAD','PAYLOAD_SHA256')}
  for n,b in vals['PAYLOAD'].items():raw=base64.b64decode(b,validate=True);self.assertEqual(raw,(PACKET/n).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),vals['PAYLOAD_SHA256'][n])
 def test_source_zip_manifest(self):
  m=json.loads((ROOT/'source_manifest.json').read_text())
  with zipfile.ZipFile(ROOT/'candidate.zip') as z:
   self.assertEqual(set(z.namelist()),set(m))
   for n,h in m.items():self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),h);self.assertEqual(z.read(n),(ROOT/n).read_bytes())
 def test_diagnostic_command_not_paired_or_shortened(self):
  tree=ast.parse((PACKET/'driver.py').read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='command' for t in n.targets));strings=[x.value for x in node.value.elts if isinstance(x,ast.Constant)];self.assertEqual(strings,['-B','--repeats','1','--seed','1','--timeout','80','--only','solution','--out']);self.assertNotIn('--profile',strings)
 def test_no_model_arithmetic_or_execution_hooks(self):
  text=(ROOT/'solution/stage_probe.py').read_text();tree=ast.parse(text)
  for n in ast.walk(tree):
   if isinstance(n,ast.Call):
    name=ast.unparse(n.func)
    self.assertNotIn(name,('os._exit','sys.exit','torch.cuda.synchronize','torch.compile','subprocess.run','open','eval','exec'))
  self.assertNotIn('setprofile',text);self.assertNotIn('add_callback',text);self.assertNotIn('manual_seed',text)
 def sample(self):
  j=pathlib.Path('/redacted/local/soup_server_jobs_20261030/redacted-job-outputs.zip')
  with zipfile.ZipFile(j) as z:
   n=next(n for n in z.namelist() if n.endswith('/result.json'));raw=json.loads(z.read(n))
  row=raw['runs']['base'][0];row.update(run='solution_1',side='solution');row['metrics']['solution']=True;row['metrics']['own_kernel_ms']={}
  raw['runs']={'solution':[row]};raw['evals']={'solution':raw['evals']['base']};raw['task'].update(only='solution',repeats=1)
  labels=('apply','soup_setup','model_setup','trainer_init','trainer_train','adapter_save','soup_train_and_final_save');log='\n'.join('SOUP_STAGE_TIMING '+json.dumps({'stage':x,'event':'return','count':1}) for x in labels)
  return raw,{'exit_code':0,'timed_out':False,'interrupted':False,'cleanup_confirmed':True},log
 def test_full_synthetic_single_diagnostic_only(self):self.assertEqual(driver.diagnostic_errors(*self.sample()),[])
 def test_internal64_with_fullcommand_timeout_fails(self):
  raw,proc,log=self.sample();self.assertIn('native_deadline',driver.diagnostic_errors(raw,proc,log+'\nTIMEOUT after 80 s'))
 def test_unclean_process_fails(self):
  raw,proc,log=self.sample();proc['cleanup_confirmed']=False;self.assertIn('harness_process_incomplete',driver.diagnostic_errors(raw,proc,log))
 def test_missing_eval_fails(self):
  raw,proc,log=self.sample();raw['evals']={};self.assertIn('native_public_evaluation_missing',driver.diagnostic_errors(raw,proc,log))
 def test_missing_native_outputs_fails(self):
  raw,proc,log=self.sample();raw['runs']['solution'][0]['rewind']=[];self.assertIn('rewind_missing_or_malformed',driver.diagnostic_errors(raw,proc,log))
if __name__=='__main__':unittest.main(verbosity=2)
