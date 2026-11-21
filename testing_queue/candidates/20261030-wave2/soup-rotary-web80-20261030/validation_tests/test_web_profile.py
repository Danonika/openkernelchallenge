"""Offline CPU/static fixtures; never calls real GPU preflight or notebook."""
import ast,base64,copy,hashlib,importlib.util,json,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=Path('/redacted/local/soup_web_ready_20261030')
sys.path.insert(0,str(ROOT/'web_adapter'))
import web_runner as w
checks=[]
def done(name):checks.append({'test':name,'pass':True})
manifest=json.loads((ROOT/'source_manifest.json').read_text())
frozen=json.loads((ROOT/'frozen_source_manifest.json').read_text())
for name,h in frozen.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
assert set(manifest)-set(frozen)=={'solution/web_preflight.py'}
with zipfile.ZipFile(ROOT/'candidate.zip') as z:
 assert z.namelist()==sorted(manifest)
 for name,h in manifest.items():assert z.read(name)==(ROOT/name).read_bytes() and hashlib.sha256(z.read(name)).hexdigest()==h
assert (ROOT/'candidate.zip').read_bytes()==(ROOT/'web_adapter/rotary.zip').read_bytes()
done('five_original_source_files_exact_only_new_preflight_adapter')
nb=json.loads((ROOT/'candidate.ipynb').read_text());payload={};hashes={}
for i,cell in enumerate(nb['cells']):
 if cell['cell_type']!='code':continue
 assert cell['outputs']==[] and cell['execution_count'] is None
 code=''.join(cell['source']);compile(code,f'cell{i}','exec')
 for node in ast.parse(code).body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
   if node.targets[0].id=='PAYLOAD':payload=ast.literal_eval(node.value)
   if node.targets[0].id=='PAYLOAD_SHA256':hashes=ast.literal_eval(node.value)
assert set(payload)==set(hashes)
for name,b64 in payload.items():
 data=base64.b64decode(b64,validate=True);assert data==(ROOT/'web_adapter'/name).read_bytes();assert hashlib.sha256(data).hexdigest()==hashes[name]
assert len((ROOT/'candidate.ipynb').read_bytes())<8*1024**2
done('cleared_self_contained_notebook_exact_payloads')
for name in ('runtime_probe.py','process_guard.py','owned_process_tree.py','prepare_asset_manifest.py','official_manifest.json'):
 assert (ROOT/'web_adapter'/name).read_bytes()==(BASE/'support'/name).read_bytes()
source=(ROOT/'web_adapter/web_runner.py').read_text();base=(BASE/'support/web_runner.py').read_text()
new_tree=ast.parse(source.replace("choices=['baseline',*FUSIONS]","choices=['baseline','fusion_a','fusion_b']"));old_tree=ast.parse(base)
newf={n.name:ast.dump(n,include_attributes=False) for n in new_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};oldf={n.name:ast.dump(n,include_attributes=False) for n in old_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert set(newf)==set(oldf)
for name in oldf:
 if name!='normalize':assert newf[name]==oldf[name],name
assert w.NATIVE_TIMEOUT==80 and "'--timeout','80'" in source
assert 'torch' not in sys.modules
done('only_profile_normalization_specialization_shared_execution_contract_exact')
with tempfile.TemporaryDirectory() as d:
 root=Path(d);repo=root/'repo';repo.mkdir();protected=json.loads((ROOT/'web_adapter/official_manifest.json').read_text())
 for name in protected:
  target=repo/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((Path('/redacted/local/soup_cloud_env/source')/name).read_bytes())
 assert not (repo/'.git').exists();w.verify_official(repo,protected)
 (repo/'solution').mkdir();(repo/'solution/previous.py').write_text('old')
 w.install_bundle(repo,ROOT/'web_adapter/rotary.zip',manifest,root/'backup')
 w.verify_official(repo,protected);assert (root/'backup/previous.py').read_text()=='old'
 for name in manifest:assert (repo/name).read_bytes()==(ROOT/name).read_bytes()
 (repo/'harness/extra.py').write_text('pass')
 try:w.verify_official(repo,protected)
 except RuntimeError:pass
 else:raise AssertionError('extra protected source accepted')
done('gitless_hash_guard_and_solution_only_overlay')

node=next(n for n in ast.parse((BASE/'test_preparation.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='fixture')
ns={'w':w};exec(compile(ast.Module(body=[node],type_ignores=[]),'fixture','exec'),ns)
def fixture():
 data,logs=ns['fixture']('rotary')
 events=[{'event':'patch_installed','count':1,'source_blob':'b4fed98fafae8e4ee39be3b8cd766b1c85b079cd'},{'event':'forward_launch_returned','count':1,'dtype':'torch.float32'},{'event':'backward_launch_returned','count':1}]
 for name in logs:logs[name]='\n'.join('ROTARY_EVIDENCE '+json.dumps(e) for e in events)
 return data,logs
process={'exit_code':0,'timed_out':False,'cleanup_confirmed':True}
data,logs=fixture();good=w.normalize(data,'rotary',process,logs);assert good['complete_fixed_task'] and good['score'] is None and good['admission']=='UNKNOWN'
for failure in ('native_timeout','missing_eval','partial_steps','wrapper_timeout','interrupted','uncertain_cleanup','missing_adapter','missing_rewind','wrong_precision','wrong_only','missing_apply_timing','missing_adjoint_timing','missing_patch','missing_forward','missing_backward','wrong_source_blob','wrong_dtype','zero_count','nan_timing','bare_substring'):
 data,logs=fixture();p=copy.deepcopy(process);row=data['runs']['solution'][0]
 if failure=='native_timeout':row['exit_code']=-9
 if failure=='missing_eval':data['evals']['solution']={}
 if failure=='partial_steps':row['metrics']['steps']=63
 if failure=='wrapper_timeout':p['timed_out']=True
 if failure=='interrupted':p['interrupted']=True
 if failure=='uncertain_cleanup':p['cleanup_confirmed']=False
 if failure=='missing_adapter':row.pop('adapter_tensors')
 if failure=='missing_rewind':row['rewind']=None
 if failure=='wrong_precision':row['metrics']['trainer']['bf16']=True
 if failure=='wrong_only':data['task']['only']='solution'
 if failure=='missing_apply_timing':row['metrics']['own_kernel_ms'].pop('rotary_apply_f32')
 if failure=='missing_adjoint_timing':row['metrics']['own_kernel_ms'].pop('rotary_adjoint_f32')
 if failure=='nan_timing':row['metrics']['own_kernel_ms']['rotary_apply_f32']=float('nan')
 if failure in ('missing_patch','missing_forward','missing_backward'):
  token={'missing_patch':'patch_installed','missing_forward':'forward_launch_returned','missing_backward':'backward_launch_returned'}[failure]
  logs[row['run']]='\n'.join(line for line in logs[row['run']].splitlines() if token not in line)
 if failure=='wrong_source_blob':logs[row['run']]=logs[row['run']].replace('b4fed98fafae8e4ee39be3b8cd766b1c85b079cd','incorrect')
 if failure=='wrong_dtype':logs[row['run']]=logs[row['run']].replace('torch.float32','torch.float16')
 if failure=='zero_count':logs[row['run']]=logs[row['run']].replace('"count": 1','"count": 0')
 if failure=='bare_substring':logs[row['run']]='ROTARY_EVIDENCE patch_installed forward_launch_returned backward_launch_returned'
 out=w.normalize(data,'rotary',p,logs)
 assert not out['complete_fixed_task'] and out['complete_measurements'] is None and out['score'] is None,(failure,out)
 done('normalize_rejects_'+failure)
done('positive_normalization_only_with_complete_fixed_task_and_structured_routes')

sys.path.insert(0,str(ROOT));from solution import web_preflight as adapter
from solution import gpu_preflight as original
assert 'torch' not in sys.modules
saved=original.main
for failure in ('success','nonzero','wrong_status','gpu_false','missing_report','malformed_report','exception','interrupt'):
 with tempfile.TemporaryDirectory() as d:
  destination=Path(d)/'preflight.json';argv=sys.argv
  def fake():
   assert sys.argv[-2]=='--out';details=Path(sys.argv[-1]);details.mkdir(exist_ok=False)
   if failure=='exception':raise RuntimeError('fake')
   if failure=='interrupt':raise KeyboardInterrupt()
   if failure=='missing_report':raise SystemExit(0)
   if failure=='malformed_report':(details/'preflight.json').write_text('{');raise SystemExit(0)
   report={'status':'WRONG' if failure=='wrong_status' else 'PASSED_PREFLIGHT_ONLY','gpu_executed':failure!='gpu_false','score':99,'admission':'PASS'}
   (details/'preflight.json').write_text(json.dumps(report));raise SystemExit(1 if failure=='nonzero' else 0)
  original.main=fake
  code=adapter.run(destination);report=json.loads(destination.read_text())
  assert code==(0 if failure=='success' else 1) and (report['status']=='PASSED_PREFLIGHT_ONLY')==(failure=='success')
  assert report['score'] is None and report['admission']=='UNKNOWN' and sys.argv is argv
  try:adapter.run(destination)
  except FileExistsError:pass
  else:raise AssertionError('existing output overwritten')
 done('preflight_file_adapter_'+failure)
original.main=saved
assert 'torch' not in sys.modules
report={'status':'CPU_STATIC_PASS','checks':checks,'check_count':len(checks),'source_zip_sha256':hashlib.sha256((ROOT/'candidate.zip').read_bytes()).hexdigest(),'notebook_sha256':hashlib.sha256((ROOT/'candidate.ipynb').read_bytes()).hexdigest(),'gpu_executed':False,'real_preflight_executed':False,'authenticated_requests':False,'submitted':False}
(ROOT/'validation/web_profile_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
