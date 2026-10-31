"""CPU/static generated full-task packet review; no Torch import/GPU/submission."""
import ast,base64,copy,hashlib,importlib.util,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
KIT=Path('/workspace/shared/soup_web_ready_20261030_v2');BASE=KIT.with_name('soup_web_ready_20261030')
checks=[]
def done(name):checks.append({'name':name,'passed':True})
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
receipt=json.loads((ROOT/'web_build_receipt.json').read_text())
nbpath=Path(receipt['notebook']);assert sha(nbpath)==receipt['notebook_sha256']
assert sha(ROOT/'candidate.zip')==receipt['source_zip_sha256']=='0d02a6c69c63351f63d12ef706cf5a2f9572dd4ad6d2cc28578e80f81ef1955e'
manifest=json.loads((ROOT/'source_manifest.json').read_text())
with zipfile.ZipFile(ROOT/'candidate.zip') as z:
 assert set(z.namelist())==set(manifest)
 for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h==sha(ROOT/n)
done('source_zip_and_source_bytes_match_target_preflight_and_microbench')
for name in ('runtime_probe.py','process_guard.py','owned_process_tree.py','prepare_asset_manifest.py','official_manifest.json','evidence_reader.py'):
 assert (ROOT/'web_adapter'/name).read_bytes()==(KIT/'support'/name).read_bytes()
done('protected_helpers_byte_identical')
def functions(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
a=functions((KIT/'support/web_runner.py').read_text())
generated=(ROOT/'web_adapter/web_runner.py').read_text()
b=functions(generated.replace("choices=['baseline',*FUSIONS]","choices=['baseline','fusion_a','fusion_b']"))
assert a.keys()==b.keys()
for name in a:
 if name!='normalize':assert a[name]==b[name],name
done('all_execution_source_runtime_cap_export_process_function_ASTs_unchanged')
nb=json.loads(nbpath.read_text());seed=json.loads((KIT/'workflow_seed.ipynb').read_text())
assert len(nb['cells'])==len(seed['cells'])==4
assert nb['cells'][3]==seed['cells'][3]
for i,cell in enumerate(nb['cells']):
 if cell['cell_type']=='code':
  assert cell['outputs']==[] and cell['execution_count'] is None
  compile(''.join(cell['source']),'cell'+str(i),'exec')
target_ast=ast.parse(''.join(nb['cells'][1]['source']));seed_ast=ast.parse(''.join(seed['cells'][1]['source']))
seed_values={n.targets[0].id:n.value for n in seed_ast.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
values={}
for n in target_ast.body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('MODE','PAYLOAD','PAYLOAD_SHA256'):
  name=n.targets[0].id;values[name]=ast.literal_eval(n.value);n.value=seed_values[name]
assert ast.dump(target_ast,include_attributes=False)==ast.dump(seed_ast,include_attributes=False)
assert values['MODE']=='rotary_runtime_dims'
for n,raw in values['PAYLOAD'].items():
 assert base64.b64decode(raw,validate=True)==(ROOT/'web_adapter'/n).read_bytes()
 assert sha(ROOT/'web_adapter'/n)==values['PAYLOAD_SHA256'][n]
done('notebook_launch_workflow_and_exact_encoded_payload')
sys.path.insert(0,str(ROOT/'web_adapter'))
spec=importlib.util.spec_from_file_location('full_rope_runner',ROOT/'web_adapter/web_runner.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
w.verify_official(Path('/workspace/shared/soup_cloud_env/source'),json.loads((KIT/'support/official_manifest.json').read_text()))
assert w.NATIVE_TIMEOUT==80 and "'--timeout','80'" in generated and "'--repeats','1' if a.mode=='baseline' else '2'" in generated
done('official_task_bytes_unchanged_and_paired_80s_cap_preserved')
fixture=next(n for n in ast.parse((BASE/'test_preparation.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='fixture')
ns={'w':w};exec(compile(ast.Module(body=[fixture],type_ignores=[]),'trusted_local_fixture','exec'),ns)
mode='rotary_runtime_dims';process={'exit_code':0,'timed_out':False,'cleanup_confirmed':True}
records=[{'count':1,'event':'patch_installed','source_blob':'b4fed98fafae8e4ee39be3b8cd766b1c85b079cd'},
         {'count':1,'event':'forward_launch_returned','dtype':'torch.float32'},
         {'count':1,'event':'backward_launch_returned'}]
valid=''.join('ROTARY_EVIDENCE '+json.dumps(r)+'\n' for r in records)
data,logs=ns['fixture'](mode);logs={n:valid for n in logs}
assert w.normalize(data,mode,process,logs)['complete_fixed_task']
done('synthetic_fixture_complete_only_with_all_owned_kernels_routes_and_outputs')
for failure in ('native_timeout','missing_eval','partial_steps','wrapper_timeout','missing_kernel','missing_route','missing_adapter','missing_metrics','missing_rewind','wrong_only','extra_side','interrupted','uncertain_cleanup','wrong_source_blob','wrong_dtype','bool_route_count'):
 d=copy.deepcopy(data);p=copy.deepcopy(process);l=copy.deepcopy(logs);row=d['runs']['solution'][0]
 if failure=='native_timeout':row['exit_code']=-9
 if failure=='missing_eval':d['evals']['solution']={}
 if failure=='partial_steps':row['metrics']['steps']=63
 if failure=='wrapper_timeout':p['timed_out']=True
 if failure=='missing_kernel':row['metrics']['own_kernel_ms'].pop('rotary_adjoint_f32')
 if failure=='missing_route':l[row['run']]=''
 if failure=='missing_adapter':row.pop('adapter_tensors')
 if failure=='missing_metrics':row['log'].pop('num_tokens')
 if failure=='missing_rewind':row['rewind']=None
 if failure=='wrong_only':d['task']['only']='solution'
 if failure=='extra_side':d['runs']['unexpected']=[]
 if failure=='interrupted':p['interrupted']=True
 if failure=='uncertain_cleanup':p['cleanup_confirmed']=False
 if failure in ('wrong_source_blob','wrong_dtype','bool_route_count'):
  bad=copy.deepcopy(records)
  if failure=='wrong_source_blob':bad[0]['source_blob']='wrong'
  if failure=='wrong_dtype':bad[1]['dtype']='torch.float16'
  if failure=='bool_route_count':bad[2]['count']=True
  l[row['run']]=''.join('ROTARY_EVIDENCE '+json.dumps(r)+'\n' for r in bad)
 result=w.normalize(d,mode,p,l)
 assert not result['complete_fixed_task'] and result['complete_measurements'] is None and result['score'] is None and result['admission']=='UNKNOWN'
 done('reject_'+failure)
assert 'torch' not in sys.modules
report={'status':'CPU_STATIC_PACKET_PASS','check_count':len(checks),'checks':checks,'notebook_sha256':sha(nbpath),'source_zip_sha256':sha(ROOT/'candidate.zip'),'gpu_executed':False,'notebook_executed':False,'submitted':False,'score':None,'admission':'UNKNOWN'}
(ROOT/'validation').mkdir(exist_ok=True);(ROOT/'validation/packet_review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
