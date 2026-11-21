"""Focused CPU/static web80 checks; never executes notebook/probe/model/auth."""
import ast,base64,copy,hashlib,json,sys,tempfile,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'web_adapter'))
import web_runner as w
checks=[]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(name,details=None):checks.append({'name':name,'passed':True,'details':details or {}})
receipt=json.loads((R/'web_build_receipt.json').read_text())
assert sha(R/'reference/accepted_web_runner.py')==receipt['adapter_base_sha256']=='cc06088f35a870b4912b9a78c8938ff330c833ecd63b04816c50ee2b6e54d01c'
assert sha(R/'web_adapter/web_runner.py')==receipt['generated_runner_sha256']
# Compare all non-profile/control behavior against the accepted shared source.
class RemoveCoverage(ast.NodeTransformer):
 def visit_Assign(self,node):
  if any(isinstance(t,ast.Name) and t.id=='FUSIONS' for t in node.targets):return None
  return self.generic_visit(node)
 def visit_keyword(self,node):
  if node.arg=='choices':node.value=ast.Constant(value='PROFILE_CHOICES')
  return self.generic_visit(node)
 def visit_FunctionDef(self,node):
  if node.name=='normalize':
   for sub in ast.walk(node):
    if isinstance(sub,ast.If) and ast.unparse(sub.test)=="side == 'solution'":
     # Retain common log/own assignments and timed-window gate; remove only the
     # candidate-specific coverage selection block specialized by the builder.
     assert len(sub.body)>=3 and isinstance(sub.body[0],ast.Assign) and isinstance(sub.body[1],ast.Assign)
     assert isinstance(sub.body[2],ast.If) and 'timed_steps' in ast.unparse(sub.body[2])
     sub.body=sub.body[:3]
  return self.generic_visit(node)
a=RemoveCoverage().visit(ast.parse((R/'reference/accepted_web_runner.py').read_text()))
b=RemoveCoverage().visit(ast.parse((R/'web_adapter/web_runner.py').read_text()))
assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
assert w.NATIVE_TIMEOUT==80 and "'--timeout','80'" in (R/'web_adapter/web_runner.py').read_text()
record('accepted_shared_cap_runtime_hash_output_guardian_export_contract_unchanged')
manifest=json.loads((R/'source_manifest.json').read_text());raw=(R/'candidate.zip').read_bytes()
with zipfile.ZipFile(R/'candidate.zip') as z:
 assert {name:hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()}==manifest
assert (R/'web_adapter/ce_rowmix_v1.zip').read_bytes()==raw
assert json.loads((R/'web_adapter/ce_rowmix_v1_manifest.json').read_text())==manifest
actual={'solution/'+str(p.relative_to(R/'solution')):sha(p) for p in (R/'solution').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
assert actual==manifest
sys.path.insert(0,'/redacted/local/soup_cycle/test_queue');import bundle_candidate
assert bundle_candidate.bundle(R/'solution')==(raw,manifest)
record('exact_source_zip_manifest_and_determinism',{'source_sha256':sha(R/'candidate.zip'),'files':len(manifest)})
nb=json.loads((R/'candidate.ipynb').read_text());literal={}
assert len((R/'candidate.ipynb').read_bytes())<8*1024**2
for cell in nb['cells']:
 if cell['cell_type']=='code':
  assert cell['outputs']==[] and cell['execution_count'] is None
  compile(''.join(cell['source']),'cell','exec')
  for n in ast.parse(''.join(cell['source'])).body:
   if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
    try:literal[n.targets[0].id]=ast.literal_eval(n.value)
    except (TypeError,ValueError):pass
assert literal['MODE']=='ce_rowmix_v1'
for name,b64 in literal['PAYLOAD'].items():
 data=base64.b64decode(b64,validate=True)
 assert data==(R/'web_adapter'/name).read_bytes() and hashlib.sha256(data).hexdigest()==literal['PAYLOAD_SHA256'][name]
assert sha(R/'candidate.ipynb')==receipt['notebook_sha256']
record('cleared_notebook_exact_embedded_payload')
for p in R.rglob('*.py'):
 if '__pycache__' not in p.parts:compile(p.read_bytes(),str(p),'exec')
record('all_python_syntax')
with tempfile.TemporaryDirectory(prefix='ce-web-gitless-') as directory:
 repo=Path(directory)/'repo';repo.mkdir();official=json.loads((R/'web_adapter/official_manifest.json').read_text())
 for name in official:
  dest=repo/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((Path('/redacted/local/soup_cloud_env/source')/name).read_bytes())
 assert not (repo/'.git').exists();w.verify_official(repo,official)
 (repo/'solution').mkdir();(repo/'solution/old.py').write_text('retained previous source')
 w.install_bundle(repo,R/'web_adapter/ce_rowmix_v1.zip',manifest,Path(directory)/'backup')
 w.verify_official(repo,official)
 assert (Path(directory)/'backup/old.py').read_text()=='retained previous source'
 assert all(sha(repo/name)==digest for name,digest in manifest.items())
 (repo/'harness/unexpected.py').write_text('pass')
 try:w.verify_official(repo,official)
 except RuntimeError:pass
 else:raise AssertionError('Unexpected protected file accepted')
record('gitless_hash_gate_exact_solution_overlay_and_protected_extra_rejection')
profile=w.FUSIONS['ce_rowmix_v1'];assert profile['requires_original_kernels'] is True and len(profile['kernels'])==5
assert profile['preflight']=='solution.verify_integration' and profile['preflight_status']=='passed_gpu_integration_preflight'
record('standalone_ce_preflight_five_kernels_and_structured_routes')

def fixture():
 metrics_keys=('loss','num_tokens','entropy','mean_token_accuracy','grad_norm')
 result={'task':{'seed':1,'repeats':2,'only':None},'packages_changed':[],'runs':{},'evals':{},'summary':{'synthetic_fixture_only':True}}
 logs={}
 for side in ('base','solution'):
  result['runs'][side]=[];result['evals'][side]={'exit_code':0,'mean_loss':2.}
  for index in range(1,3):
   name=f'{side}_{index}'
   result['runs'][side].append({'run':name,'side':side,'exit_code':0,'global_step':64,'wall_s':70.,'gpu_peak_mib':1000.,'adapter_config':{'r':16,'lora_alpha':32,'base_model_name_or_path':'Qwen/Qwen2.5-1.5B-Instruct'},'adapter_tensors':{'fixture.lora_A':['F32',[16,1536]]},'logged_metrics':list(metrics_keys),'log':{k:[1.]*64 for k in metrics_keys},'rewind':[[[0,1,2,3],[1,1,1,1],[1.,1.,1.,1.]] for _ in range(64)],'metrics':{'exit_code':0,'steps':64,'device_count':1,'cuda_device':'Tesla T4','timed_steps':62,'timed_loop_s':30.,'own_kernel_ms':{k:1. for k in profile['kernels']},'trainer':{'fp16':True,'bf16':False,'per_device_train_batch_size':4,'gradient_accumulation_steps':1,'seed':1,'max_length':512,'torch_compile':False,'gradient_checkpointing':False}}})
   logs[name]='\n'.join('SOUP_CE_V3_EVENT '+json.dumps({'event':event,field:1}) for event,field in [('triton_forward','own_forward_routes'),('triton_backward','own_backward_enqueues')])
 return result,logs
process={'exit_code':0,'timed_out':False,'cleanup_confirmed':True,'interrupted':False}
base,logs=fixture();good=w.normalize(base,'ce_rowmix_v1',process,logs)
assert good['complete_fixed_task'] and good['score'] is None and good['admission']=='UNKNOWN'
failures=['native_timeout','nested_timeout','partial_steps','missing_adapter','missing_rewind','missing_metrics','bad_precision','no_eval','nan_eval','wrong_only','extra_side','wrong_repeat','missing_package_inventory','outer_timeout','interrupted','uncertain_cleanup','install_only','invalid_json_route','wrong_event','zero_route','nonfinite_route']+['kernel:'+k for k in profile['kernels']]
for failure in failures:
 data=copy.deepcopy(base);observed=dict(logs);proc=dict(process);row=data['runs']['solution'][0]
 if failure=='native_timeout':row['exit_code']=-9
 elif failure=='nested_timeout':row['metrics']['exit_code']=-9
 elif failure=='partial_steps':row['metrics']['steps']=63
 elif failure=='missing_adapter':row.pop('adapter_tensors')
 elif failure=='missing_rewind':row['rewind']=None
 elif failure=='missing_metrics':row['log'].pop('entropy')
 elif failure=='bad_precision':row['metrics']['trainer']['bf16']=True
 elif failure=='no_eval':data['evals']['solution']={}
 elif failure=='nan_eval':data['evals']['solution']['mean_loss']=float('nan')
 elif failure=='wrong_only':data['task']['only']='solution'
 elif failure=='extra_side':data['runs']['extra']=[]
 elif failure=='wrong_repeat':data['runs']['solution'].pop()
 elif failure=='missing_package_inventory':data.pop('packages_changed')
 elif failure=='outer_timeout':proc['timed_out']=True
 elif failure=='interrupted':proc['interrupted']=True
 elif failure=='uncertain_cleanup':proc['cleanup_confirmed']=False
 elif failure=='install_only':observed[row['run']]='SOUP_CE_V3_INSTALLED loss_boundary_only'
 elif failure=='invalid_json_route':observed[row['run']]='SOUP_CE_V3_EVENT triton_forward triton_backward'
 elif failure=='wrong_event':observed[row['run']]='SOUP_CE_V3_EVENT {"event":"installed","own_forward_routes":1,"own_backward_enqueues":1}'
 elif failure=='zero_route':observed[row['run']]=observed[row['run']].replace(': 1',': 0')
 elif failure=='nonfinite_route':observed[row['run']]=observed[row['run']].replace(': 1',': NaN')
 elif failure.startswith('kernel:'):row['metrics']['own_kernel_ms'].pop(failure.split(':',1)[1])
 result=w.normalize(data,'ce_rowmix_v1',proc,observed)
 assert not result['complete_fixed_task'] and result['complete_measurements'] is None and result['score'] is None and result['admission']=='UNKNOWN',failure
record('result_normalization_positive_and_negative_fixtures',{'negative_cases':len(failures),'cases':failures,'synthetic_not_measurements':True})
with tempfile.TemporaryDirectory(prefix='ce-web-export-') as directory:
 run=Path(directory)/'run';export=Path(directory)/'export';run.mkdir();export.mkdir()
 (run/'train.log').write_text(logs['solution_1']);(run/'adapter').mkdir();(run/'adapter/payload.json').write_text('not exportable')
 (run/'triton_cache').mkdir();(run/'triton_cache/compiled.json').write_text('not exportable')
 (run/'large.log').write_bytes(b'x'*(4*1024**2));w.export_small(run,export)
 assert (export/'raw/train.log').read_text()==logs['solution_1']
 assert not (export/'raw/adapter').exists() and not (export/'raw/triton_cache').exists() and not (export/'raw/large.log').exists()
 assert 'large.log' in w.EXPORT_OMITTED
record('small_export_preserves_flushed_routes_excludes_weights_and_cache')
report={'status':'CPU_STATIC_WEB_PACKET_PASS','checks':checks,'gpu_executed':False,'notebook_executed':False,'auth_requests':False,'submitted':False,'baseline_feedback':'PARENT_REPORTED_COMPLETED; THIS_PACKET_NOT_RUN','score':None,'admission':'UNKNOWN'}
(R/'validation/web_packet_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
