"""CPU/static-only web packet guards. No GPU, notebook execution or API use."""
from pathlib import Path
import ast,base64,copy,hashlib,json,sys,tempfile,zipfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'support'))
import web_runner as w
results=[]
sha=lambda raw:hashlib.sha256(raw).hexdigest()
raw=(ROOT/'candidate.ipynb').read_bytes();nb=json.loads(raw);assert len(raw)<8*1024**2
payload=expected={}
for i,c in enumerate(nb['cells']):
 if c['cell_type']!='code':continue
 assert c['execution_count'] is None and c['outputs']==[]
 code=''.join(c['source']);compile(code,f'candidate:{i}','exec')
 for node in ast.parse(code).body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
   if node.targets[0].id=='PAYLOAD':payload=ast.literal_eval(node.value)
   if node.targets[0].id=='PAYLOAD_SHA256':expected=ast.literal_eval(node.value)
   if node.targets[0].id=='MODE':assert ast.literal_eval(node.value)=='memory'
assert payload and set(payload)==set(expected)
for name,b64 in payload.items():
 file=next((p for p in (ROOT/'support'/name,ROOT/'payloads'/name,ROOT/name) if p.is_file()),None)
 data=base64.b64decode(b64,validate=True);assert file and data==file.read_bytes() and sha(data)==expected[name]
results.append({'test':'self_contained_notebook_exact_payload','bytes':len(raw),'pass':True})
for p in list((ROOT/'support').glob('*.py'))+list((ROOT/'solution').glob('*.py')):compile(p.read_bytes(),str(p),'exec')
manifest=json.loads((ROOT/'source_manifest.json').read_text())
assert (ROOT/'candidate.zip').read_bytes()==(ROOT/'payloads/memory.zip').read_bytes()
assert manifest==json.loads((ROOT/'payloads/memory_manifest.json').read_text())
with zipfile.ZipFile(ROOT/'candidate.zip') as z:
 assert set(z.namelist())==set(manifest)
 for name,h in manifest.items():assert sha(z.read(name))==h and z.read(name)==(ROOT/name).read_bytes()
sys.path.insert(0,'/redacted/local/soup_cycle/test_queue')
from bundle_candidate import bundle
new,files=bundle(ROOT/'solution');assert new==(ROOT/'candidate.zip').read_bytes() and files==manifest
assert sha(new)=='055183cdd11a254e3462f25b95b4c0ec87a556cc58cb9bcc08e77f5390548979'
results.append({'test':'exact_reviewed_source_and_deterministic_zip','pass':True})
original=Path('/redacted/local/soup_web_ready_20261030/support')
for name in ('runtime_probe.py','process_guard.py','owned_process_tree.py','prepare_asset_manifest.py','official_manifest.json'):
 assert (ROOT/'support'/name).read_bytes()==(original/name).read_bytes()
results.append({'test':'accepted_runtime_asset_guardian_helpers_unchanged','pass':True})
protected=json.loads((ROOT/'support/official_manifest.json').read_text())
with tempfile.TemporaryDirectory() as d:
 repo=Path(d)
 for name in protected:
  target=repo/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((Path('/redacted/local/soup_cloud_env/source')/name).read_bytes())
 assert not (repo/'.git').exists();w.verify_official(repo,protected)
 bad=repo/'harness/extra.py';bad.write_text('pass')
 try:w.verify_official(repo,protected)
 except RuntimeError:pass
 else:raise AssertionError('Unexpected official source accepted')
 bad.unlink();(repo/next(iter(protected))).write_text('tampered')
 try:w.verify_official(repo,protected)
 except RuntimeError:pass
 else:raise AssertionError('Changed official bytes accepted')
results.append({'test':'gitless_hash_gate_and_negatives','pass':True})
with tempfile.TemporaryDirectory() as d:
 repo=Path(d)/'repo';repo.mkdir();(repo/'solution').mkdir();(repo/'solution/old.py').write_text('old');(repo/'protected').write_text('same')
 w.install_bundle(repo,ROOT/'payloads/memory.zip',manifest,Path(d)/'backup')
 assert (repo/'protected').read_text()=='same' and (Path(d)/'backup/old.py').read_text()=='old'
 assert all(sha((repo/name).read_bytes())==h for name,h in manifest.items())
results.append({'test':'solution_only_overlay_preserves_prior_bytes','pass':True})

def fixture():
 data={'task':{'seed':1,'repeats':2,'only':None},'packages_changed':[],'runs':{},'evals':{},'summary':{'synthetic_only':True}};logs={}
 for side in ('base','solution'):
  data['runs'][side]=[];data['evals'][side]={'exit_code':0,'mean_loss':1.}
  for i in (1,2):
   name=f'{side}_{i}';m={'exit_code':0,'steps':64,'device_count':1,'cuda_device':'Tesla T4','trainer':{'fp16':True,'bf16':False,'per_device_train_batch_size':4,'gradient_accumulation_steps':1,'seed':1,'max_length':512,'torch_compile':False,'gradient_checkpointing':False},'timed_steps':62,'timed_loop_s':30.,'own_kernel_ms':{'soup_mem001_restore_half_to_fp32':1000.}}
   data['runs'][side].append({'run':name,'side':side,'adapter_config':{'r':16,'lora_alpha':32,'base_model_name_or_path':'Qwen/Qwen2.5-1.5B-Instruct'},'adapter_tensors':{'fixture.lora_A':['F32',[16,1536]]},'logged_metrics':['loss','num_tokens','entropy','mean_token_accuracy','grad_norm'],'log':{k:[1.]*64 for k in ('loss','num_tokens','entropy','mean_token_accuracy','grad_norm')},'rewind':[[[0,1,2,3],[1,1,1,1],[1.,1.,1.,1.]] for _ in range(64)],'exit_code':0,'global_step':64,'wall_s':70.,'gpu_peak_mib':1000.,'metrics':m})
   logs[name]='SOUP_MEMORY_VS1_RUNTIME '+json.dumps({'encoded_operands':3584,'restores':3584})+'\n'
 return data,logs
process={'exit_code':0,'timed_out':False,'cleanup_confirmed':True}
data,logs=fixture();good=w.normalize(data,'memory',process,logs)
assert good['complete_fixed_task'] and good['score'] is None and good['admission']=='UNKNOWN'
negative_cases=[]
for failure in ('native_timeout','missing_eval','partial_steps','wrapper_timeout','cleanup_unknown','missing_kernel','missing_route','progress_only','malformed_final','zero_restores','bool_counter','short_timed_window','missing_adapter','missing_metrics','missing_rewind','wrong_only','extra_side','interrupted'):
 broken=copy.deepcopy(data);p=dict(process);l=dict(logs);row=broken['runs']['solution'][0]
 if failure=='missing_adapter':row.pop('adapter_tensors')
 if failure=='missing_metrics':row['log'].pop('num_tokens')
 if failure=='missing_rewind':row['rewind']=None
 if failure=='wrong_only':broken['task']['only']='solution'
 if failure=='extra_side':broken['runs']['unexpected']=[]
 if failure=='interrupted':p['interrupted']=True
 if failure=='native_timeout':row['exit_code']=-9
 if failure=='missing_eval':broken['evals']['base']={}
 if failure=='partial_steps':row['metrics']['steps']=63
 if failure=='wrapper_timeout':p['timed_out']=True
 if failure=='cleanup_unknown':p['cleanup_confirmed']=False
 if failure=='missing_kernel':row['metrics']['own_kernel_ms']={}
 if failure=='missing_route':l[row['run']]=''
 if failure=='progress_only':l[row['run']]='SOUP_MEMORY_VS1_PROGRESS '+json.dumps({'encoded_operands':3584,'restores':3584})
 if failure=='malformed_final':l[row['run']]='SOUP_MEMORY_VS1_RUNTIME {"restores":'
 if failure=='zero_restores':l[row['run']]='SOUP_MEMORY_VS1_RUNTIME '+json.dumps({'encoded_operands':3584,'restores':0})
 if failure=='bool_counter':l[row['run']]='SOUP_MEMORY_VS1_RUNTIME '+json.dumps({'encoded_operands':True,'restores':True})
 if failure=='short_timed_window':row['metrics']['timed_steps']=61
 r=w.normalize(broken,'memory',p,l)
 assert not r['complete_fixed_task'] and r['complete_measurements'] is None and r['score'] is None and r['admission']=='UNKNOWN',failure
 negative_cases.append(failure)
results.append({'test':'complete_vs_failed_route_normalization','negative_cases':negative_cases,'pass':True})
with tempfile.TemporaryDirectory() as d:
 run=Path(d)/'run';out=Path(d)/'out';run.mkdir();out.mkdir();(run/'good.log').write_text('flushed')
 (run/'adapter').mkdir();(run/'adapter/weights.json').write_text('no export');(run/'large.log').write_bytes(b'x'*(4*1024**2))
 w.export_small(run,out);assert (out/'raw/good.log').read_text()=='flushed' and not (out/'raw/adapter').exists() and not (out/'raw/large.log').exists()
 prior=w.EXPORT_BUDGET;w.EXPORT_BUDGET=1;(run/'over.log').write_text('too big');w.export_small(run,out)
 assert 'over.log' in w.EXPORT_OMITTED and not (out/'raw/over.log').exists();w.EXPORT_BUDGET=prior
results.append({'test':'small_evidence_export_exclusions_and_bound','pass':True})
source=(ROOT/'support/web_runner.py').read_text()
assert w.NATIVE_TIMEOUT==80 and "'--timeout','80'" in source
assert 'git archive' not in source and "'git'" not in source
assert "'SOUP_REPO'" in source and "'SOUP_RESULTS'" in source and 'PATH=' not in source
assert w.FUSIONS['memory']['preflight']=='solution.verify_memory' and w.FUSIONS['memory']['preflight_status']=='PASS'
results.append({'test':'web_interface_gitless_80s_preflight_contract','pass':True})
report={'status':'CPU_STATIC_PASS','checks':results,'source_zip_sha256':sha(new),'notebook_sha256':sha(raw),
 'gpu_executed':False,'notebooks_executed':False,'authenticated_requests':False,'token_access':False,'job_submitted':False,
 'actual_web_runtime_executed':False,'tied_storage_claim':False,'score':None,'admission':'UNKNOWN'}
(ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
