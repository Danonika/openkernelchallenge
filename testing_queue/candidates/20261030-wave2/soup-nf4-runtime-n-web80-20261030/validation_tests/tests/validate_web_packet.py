"""Static/CPU-only web contract tests; never invokes a notebook or GPU probe."""
import ast,base64,copy,hashlib,json,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'web_adapter'))
import web_runner as w
sys.path.insert(0,'/redacted/local/soup_cycle/test_queue')
from bundle_candidate import bundle
rows=[]
def check(name,yes):
 assert yes,name;rows.append({'check':name,'passed':True})
for p in ROOT.rglob('*.py'):compile(p.read_text(),str(p),'exec')
check('all_python_syntax',True)
raw,manifest=bundle(ROOT/'solution')
check('deterministic_source_bundle',raw==(ROOT/'candidate.zip').read_bytes() and bundle(ROOT/'solution')[0]==raw)
check('exact_source_manifest',manifest==json.loads((ROOT/'source_manifest.json').read_text()))
nb=json.loads((ROOT/'candidate.ipynb').read_text());payload=None
for i,c in enumerate(nb['cells']):
 if c['cell_type']!='code':continue
 check(f'cell_{i}_cleared',c['outputs']==[] and c['execution_count'] is None)
 text=''.join(c['source']);compile(text,str(i),'exec')
 for node in ast.parse(text).body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='PAYLOAD':payload=ast.literal_eval(node.value)
check('payload_exists',bool(payload))
for name,encoded in payload.items():
 path=ROOT/'web_adapter'/name
 if name=='nf4_runtime_n.zip':path=ROOT/'candidate.zip'
 elif name=='nf4_runtime_n_manifest.json':path=ROOT/'source_manifest.json'
 elif not path.exists():path=ROOT/name
 check('embedded_'+name,base64.b64decode(encoded,validate=True)==path.read_bytes())
check('notebook_upload_size',len((ROOT/'candidate.ipynb').read_bytes())<8*1024**2)
check('native_cap80',w.NATIVE_TIMEOUT==80 and "'--timeout','80'" in (ROOT/'web_adapter/web_runner.py').read_text())
check('no_manual_path_fallback',"os.environ.get('SOUP_REPO')" in (ROOT/'web_adapter/web_runner.py').read_text())
check('accepted_shared_generator_used',json.loads((ROOT/'web_build_receipt.json').read_text())['source_zip_sha256']==hashlib.sha256(raw).hexdigest())
check('declared_original_kernel_required',w.FUSIONS['nf4_runtime_n']['requires_original_kernels'] is True)
with tempfile.TemporaryDirectory() as tmp:
 tmp=Path(tmp);repo=tmp/'repo';repo.mkdir();(repo/'solution').mkdir();(repo/'solution/old.py').write_text('original')
 file=tmp/'nf4_runtime_n.zip';file.write_bytes(raw)
 w.install_bundle(repo,file,manifest,tmp/'backup')
 check('exact_bundle_overlay_only',all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in manifest.items()) and (tmp/'backup/old.py').read_text()=='original')

def fixture():
 result={'task':{'seed':1,'repeats':2,'only':None},'packages_changed':[],'runs':{},'evals':{},'summary':{'fixture_only':True}}
 logs={}
 for side in ('base','solution'):
  result['runs'][side]=[];result['evals'][side]={'exit_code':0,'mean_loss':1.}
  for i in (1,2):
   name=f'{side}_{i}'
   m={'exit_code':0,'steps':64,'device_count':1,'cuda_device':'Tesla T4','trainer':{'fp16':True,'bf16':False,'per_device_train_batch_size':4,'gradient_accumulation_steps':1,'seed':1,'max_length':512,'torch_compile':False,'gradient_checkpointing':False},'timed_steps':62,'timed_loop_s':30.,'own_kernel_ms':{'soup_nf4_blockpair_runtime_n':1000.}}
   result['runs'][side].append({'run':name,'side':side,'adapter_config':{'r':16,'lora_alpha':32,'base_model_name_or_path':'Qwen/Qwen2.5-1.5B-Instruct'},'adapter_tensors':{'fixture.lora_A':['F32',[16,1536]]},'logged_metrics':['loss','num_tokens','entropy','mean_token_accuracy','grad_norm'],'log':{k:[1.]*64 for k in ('loss','num_tokens','entropy','mean_token_accuracy','grad_norm')},'rewind':[[[0,1,2,3],[1,1,1,1],[1.,1.,1.,1.]] for _ in range(64)],'exit_code':0,'global_step':64,'wall_s':70.,'gpu_peak_mib':1000.,'metrics':m})
   logs[name]='\n'.join('SOUP_NF4_ROUTE '+json.dumps({'candidate':'nf4_runtime_n_v2','route':r,'calls':1,'shape':[1536,1536],'dtype':'torch.float16','kernel':'soup_nf4_blockpair_runtime_n'}) for r in ('forward','functional'))
 return result,logs
process={'exit_code':0,'timed_out':False,'interrupted':False,'cleanup_confirmed':True}
data,logs=fixture();r=w.normalize(data,'nf4_runtime_n',process,logs)
check('complete_synthetic_pair_not_admission',r['complete_fixed_task'] and r['score'] is None and r['admission']=='UNKNOWN')
for failure in ('native_timeout','partial_steps','missing_adapter','missing_metrics','missing_rewind','missing_eval','missing_kernel','missing_routes','missing_forward','missing_functional','wrong_candidate','malformed_routes','boolean_calls','outer_timeout','interrupted','wrong_only','extra_side','packages_unrecorded'):
 d=copy.deepcopy(data);l=dict(logs);p=dict(process);row=d['runs']['solution'][0]
 if failure=='native_timeout':row['exit_code']=-9
 if failure=='partial_steps':row['metrics']['steps']=63
 if failure=='missing_adapter':row.pop('adapter_tensors')
 if failure=='missing_metrics':row['log'].pop('loss')
 if failure=='missing_rewind':row['rewind']=None
 if failure=='missing_eval':d['evals']['base']={}
 if failure=='missing_kernel':row['metrics']['own_kernel_ms']={}
 if failure=='missing_routes':l[row['run']]=''
 if failure=='missing_forward':l[row['run']]='\n'.join(x for x in l[row['run']].splitlines() if '"forward"' not in x)
 if failure=='missing_functional':l[row['run']]='\n'.join(x for x in l[row['run']].splitlines() if '"functional"' not in x)
 if failure=='wrong_candidate':l[row['run']]=l[row['run']].replace('nf4_runtime_n_v2','other')
 if failure=='malformed_routes':l[row['run']]='SOUP_NF4_ROUTE {truncated'
 if failure=='boolean_calls':l[row['run']]=l[row['run']].replace('"calls": 1','"calls": true')
 if failure=='outer_timeout':p['timed_out']=True
 if failure=='interrupted':p['interrupted']=True
 if failure=='wrong_only':d['task']['only']='solution'
 if failure=='extra_side':d['runs']['surprise']=[]
 if failure=='packages_unrecorded':d.pop('packages_changed')
 out=w.normalize(d,'nf4_runtime_n',p,l)
 check(failure+'_fails_closed',not out['complete_fixed_task'] and out['complete_measurements'] is None and out['score'] is None and out['admission']=='UNKNOWN')

wrapped={k:'\n'.join('\x1b[2K'+line.replace(', ', ',\n') for line in v.splitlines()) for k,v in logs.items()}
check('V2_wrapped_metadata_recovers_only_evidence',w.normalize(data,'nf4_runtime_n',process,wrapped)['complete_fixed_task'])
for label,transform in [('duplicate_key',lambda x:x.replace('"calls": 1','"calls": 1, "calls": 2')),
                        ('nan_count',lambda x:x.replace('"calls": 1','"calls": NaN')),
                        ('zero_count',lambda x:x.replace('"calls": 1','"calls": 0')),
                        ('unanchored',lambda x:x.replace('SOUP_NF4_ROUTE ','untrustedtext SOUP_NF4_ROUTE '))]:
 bad={k:transform(v) for k,v in logs.items()}
 check(label+'_fails_closed',not w.normalize(data,'nf4_runtime_n',process,bad)['complete_fixed_task'])
report={'status':'CPU_STATIC_WEB_PACKET_PASS','checks':rows,'count':len(rows),'gpu_executed':False,'notebook_executed':False,'authenticated_requests':False,'submitted':False,'score':None,'admission':'UNKNOWN','candidate_sha256':hashlib.sha256(raw).hexdigest(),'notebook_sha256':hashlib.sha256((ROOT/'candidate.ipynb').read_bytes()).hexdigest()}
(ROOT/'validation/web_packet_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
