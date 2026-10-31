"""CPU/static-only fixtures; no GPU probe, model execution, job or auth request."""
from pathlib import Path
import ast,base64,copy,hashlib,importlib.util,json,sys,tempfile,zipfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'support'))
import web_runner as w
results=[]
for name in ('00_baseline.ipynb','01_fusion_a.ipynb','02_fusion_b.ipynb'):
 raw=(ROOT/name).read_bytes();nb=json.loads(raw);assert len(raw)<8*1024**2
 payload={}
 for i,c in enumerate(nb['cells']):
  if c['cell_type']!='code':continue
  assert c['execution_count'] is None and c['outputs']==[]
  code=''.join(c['source']);compile(code,f'{name}:{i}','exec')
  for node in ast.parse(code).body:
   if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='PAYLOAD':payload=ast.literal_eval(node.value)
 assert payload
 for n,b64 in payload.items():
  location=ROOT/'support'/n
  if not location.is_file():location=ROOT/'payloads'/n
  if not location.is_file():location=ROOT/n
  assert base64.b64decode(b64,validate=True)==location.read_bytes()
 results.append({'test':'self_contained_notebook','file':name,'bytes':len(raw),'pass':True})
for p in (ROOT/'support').glob('*.py'):compile(p.read_bytes(),str(p),'exec')
# Prepared source works without .git; exact hash and added-file negatives fail.
manifest=json.loads((ROOT/'support/official_manifest.json').read_text())
with tempfile.TemporaryDirectory() as d:
 repo=Path(d)
 for name in manifest:
  target=repo/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((Path('/workspace/shared/soup_cloud_env/source')/name).read_bytes())
 assert not (repo/'.git').exists();w.verify_official(repo,manifest)
 bad=repo/'harness/extra.py';bad.write_text('pass')
 try:w.verify_official(repo,manifest)
 except RuntimeError:pass
 else:raise AssertionError('Unexpected protected source accepted')
 bad.unlink();first=next(iter(manifest));(repo/first).write_text('tampered')
 try:w.verify_official(repo,manifest)
 except RuntimeError:pass
 else:raise AssertionError('Hash drift accepted')
results.append({'test':'gitless_prepared_hash_gate_and_negatives','pass':True})
# Exact bundle staging changes only solution and retains previous source.
for mode in ('fusion_a','fusion_b'):
 with tempfile.TemporaryDirectory() as d:
  repo=Path(d)/'repo';repo.mkdir();(repo/'solution').mkdir();(repo/'solution/old.py').write_text('old')
  (repo/'protected').write_text('unchanged')
  w.install_bundle(repo,ROOT/'payloads'/(mode+'.zip'),json.loads((ROOT/'payloads'/(mode+'_manifest.json')).read_text()),Path(d)/'backup')
  assert (repo/'protected').read_text()=='unchanged' and (Path(d)/'backup/old.py').read_text()=='old'
 results.append({'test':'exact_bundle_staging','mode':mode,'pass':True})
# Normalizer never assigns full comparison to native timeout, wrapper success,
# missing eval, missing steps, routes, or owned timing.
def fixture(mode):
 data={'task':{'seed':1,'repeats':1 if mode=='baseline' else 2,'only':'base' if mode=='baseline' else None},'packages_changed':[],'runs':{},'evals':{},'summary':{'fixture_only':True}}
 logs={};sides=['base'] if mode=='baseline' else ['base','solution']
 for side in sides:
  data['runs'][side]=[];data['evals'][side]={'exit_code':0,'mean_loss':1.}
  for i in range(1 if mode=='baseline' else 2):
   name=f'{side}_{i+1}';m={'exit_code':0,'steps':64,'device_count':1,'cuda_device':'Tesla T4','trainer':{'fp16':True,'bf16':False,'per_device_train_batch_size':4,'gradient_accumulation_steps':1,'seed':1,'max_length':512,'torch_compile':False,'gradient_checkpointing':False},'timed_steps':62,'timed_loop_s':30.,'own_kernel_ms':{k:1000. for k in w.FUSIONS.get(mode,{}).get('kernels',[])}}
   data['runs'][side].append({'run':name,'side':side,'adapter_config':{'r':16,'lora_alpha':32,'base_model_name_or_path':'Qwen/Qwen2.5-1.5B-Instruct'},'adapter_tensors':{'fixture.lora_A':['F32',[16,1536]]},'logged_metrics':['loss','num_tokens','entropy','mean_token_accuracy','grad_norm'],'log':{k:[1.]*64 for k in ('loss','num_tokens','entropy','mean_token_accuracy','grad_norm')},'rewind':[[[0,1,2,3],[1,1,1,1],[1.,1.,1.,1.]] for _ in range(64)],'exit_code':0,'global_step':64,'wall_s':70.,'gpu_peak_mib':1000.,'metrics':m})
   logs[name]='SOUP_MEMORY_VS1_RUNTIME native_sum_backward_return backward_launch_returned SOUP_CE_V3_EVENT triton_backward SOUP_LORA_EPILOGUE launch_submitted SOUP_NF4_ROUTE'
 return data,logs
process={'exit_code':0,'timed_out':False,'cleanup_confirmed':True}
for mode in ('baseline','fusion_a','fusion_b'):
 data,logs=fixture(mode);good=w.normalize(data,mode,process,logs);assert good['complete_fixed_task'] and good['score'] is None and good['admission']=='UNKNOWN'
 for failure in ('native_timeout','missing_eval','partial_steps','wrapper_timeout','missing_kernel','missing_route','missing_adapter','missing_metrics','missing_rewind','wrong_only','extra_side','interrupted'):
  if mode=='baseline' and failure in ('missing_kernel','missing_route','missing_adapter','missing_metrics','missing_rewind','wrong_only','extra_side','interrupted'):continue
  broken=copy.deepcopy(data);p=dict(process);l=dict(logs);row=broken['runs']['base' if mode=='baseline' else 'solution'][0]
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
  if failure=='missing_kernel':row['metrics']['own_kernel_ms']={}
  if failure=='missing_route':l[row['run']]=''
  r=w.normalize(broken,mode,p,l);assert not r['complete_fixed_task'] and r['complete_measurements'] is None and r['score'] is None and r['admission']=='UNKNOWN'
 results.append({'test':'truthful_result_normalization','mode':mode,'pass':True})
with tempfile.TemporaryDirectory() as d:
 run=Path(d)/'run';export=Path(d)/'out';run.mkdir();export.mkdir()
 (run/'good.log').write_text('flushed evidence')
 (run/'adapter').mkdir();(run/'adapter/secret.json').write_text('not an export')
 (run/'large.log').write_bytes(b'x'*(4*1024**2))
 w.export_small(run,export)
 assert (export/'raw/good.log').read_text()=='flushed evidence'
 assert not (export/'raw/adapter').exists() and not (export/'raw/large.log').exists()
 old=w.EXPORT_BUDGET;w.EXPORT_BUDGET=1
 (run/'budget.log').write_text('exceeds remaining export budget');w.export_small(run,export)
 assert 'budget.log' in w.EXPORT_OMITTED and not (export/'raw/budget.log').exists();w.EXPORT_BUDGET=old
results.append({'test':'small_export_scope_and_budget','pass':True})
source=(ROOT/'support/web_runner.py').read_text()
assert "'--timeout','80'" in source and 'git_output' not in source and 'git archive' not in source
assert w.NATIVE_TIMEOUT==80
report={'status':'CPU_STATIC_PASS','checks':results,'gpu_executed':False,'notebooks_executed':False,'authenticated_requests':False,'job_submitted':False,'actual_web_runtime_executed':False}
(ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
