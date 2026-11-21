"""Data-only source/ZIP/notebook/accepted-runner negative fixtures."""
import ast,base64,copy,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;KIT=Path('/redacted/local/soup_web_ready_20261030_v2/support')
MODE='head_shared_cast_v4';checks=[]
def good(name):checks.append(name)
manifest=json.loads((ROOT/'source_manifest.json').read_text())
with zipfile.ZipFile(ROOT/'candidate.zip') as z:
 assert set(z.namelist())==set(manifest) and len(z.namelist())==len(manifest)
 for name,digest in manifest.items():
  raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==digest and raw==(ROOT/name).read_bytes();compile(raw,name,'exec')
good('exact_source_zip_manifest')
nb=json.loads((ROOT/'candidate.ipynb').read_text());values={}
for i,c in enumerate(nb['cells']):
 if c['cell_type']=='code':
  assert c['execution_count'] is None and not c['outputs'];code=''.join(c['source']);compile(code,'cell'+str(i),'exec')
  for n in ast.parse(code).body:
   if isinstance(n,ast.Assign) and len(n.targets)==1 and getattr(n.targets[0],'id','') in ['MODE','PAYLOAD','PAYLOAD_SHA256']:
    values[n.targets[0].id]=ast.literal_eval(n.value)
assert values['MODE']==MODE
for n,b in values['PAYLOAD'].items():
 raw=base64.b64decode(b,validate=True);assert raw==(ROOT/'web_adapter'/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==values['PAYLOAD_SHA256'][n]
assert (ROOT/'web_adapter'/f'{MODE}.zip').read_bytes()==(ROOT/'candidate.zip').read_bytes()
good('notebook_embedded_payload_exact_clear_outputs')
for n in ['runtime_probe.py','process_guard.py','owned_process_tree.py','prepare_asset_manifest.py','official_manifest.json','evidence_reader.py']:
 assert (ROOT/'web_adapter'/n).read_bytes()==(KIT/n).read_bytes()
base=ast.parse((KIT/'web_runner.py').read_text());cur=ast.parse((ROOT/'web_adapter/web_runner.py').read_text())
base_nodes={n.name:n for n in base.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))};cur_nodes={n.name:n for n in cur.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
assert set(base_nodes)==set(cur_nodes)
for n in base_nodes:
 if n not in ['normalize','main']:assert ast.dump(base_nodes[n])==ast.dump(cur_nodes[n]),n
# The generated main must equal the previously accepted V2 head main exactly.
accepted=ast.parse(Path('/redacted/local/soup_web_wave3_20261030/head_telemetry_v2/web_adapter/web_runner.py').read_text())
accepted_main=next(n for n in accepted.body if isinstance(n,ast.FunctionDef) and n.name=='main')
assert ast.dump(cur_nodes['main'])==ast.dump(accepted_main)
good('protected_source_runtime_process_cap_export_workflow_unchanged')
sys.path.insert(0,str(ROOT/'web_adapter'));import web_runner as w
assert w.NATIVE_TIMEOUT==80 and w.FUSIONS[MODE]['kernels']==[] and not w.FUSIONS[MODE]['requires_original_kernels']
fixture_tree=ast.parse(Path('/redacted/local/soup_web_wave2_20261030/head/test_preparation.py').read_text())
fixture=next(n for n in fixture_tree.body if isinstance(n,ast.FunctionDef) and n.name=='fixture');scope={'json':json}
exec(compile(ast.Module(body=[fixture],type_ignores=[]),'synthetic_complete_fixture','exec'),scope)
data,logs=scope['fixture']();proc={'exit_code':0,'timed_out':False,'interrupted':False,'cleanup_confirmed':True}
log='\n'.join('SOUP_HEAD_SHARED '+json.dumps({'event':e,'count':64}) for e in ['shared_cast','owned_loss_return'])
logs={'solution_1':log,'solution_2':log}
assert w.normalize(data,MODE,proc,logs)['complete_fixed_task']
negative=0
for count in [0,1,32,63,65,'64',64.0,True,None]:
 bad='\n'.join('SOUP_HEAD_SHARED '+json.dumps({'event':e,'count':count}) for e in ['shared_cast','owned_loss_return'])
 result=w.normalize(data,MODE,proc,{'solution_1':bad,'solution_2':bad});assert not result['complete_fixed_task'] and result['complete_measurements'] is None and result['score'] is None;negative+=1
for key,value in [('exit_code',1),('timed_out',True),('interrupted',True),('cleanup_confirmed',False)]:
 bad=dict(proc);bad[key]=value;assert not w.normalize(data,MODE,bad,logs)['complete_fixed_task'];negative+=1
for missing in ['shared_cast','owned_loss_return']:
 bad='\n'.join(x for x in log.splitlines() if missing not in x);assert not w.normalize(data,MODE,proc,{'solution_1':bad,'solution_2':bad})['complete_fixed_task'];negative+=1
good('synthetic_complete_and_'+str(negative)+'_negative_route_cleanup_count_controls')
assert 'torch' not in sys.modules and 'triton' not in sys.modules
result={'status':'PASS_CPU_PACKET_ONLY','checks':checks,'negative_cases':negative,'gpu_executed':False,
 'notebook_sha256':hashlib.sha256((ROOT/'candidate.ipynb').read_bytes()).hexdigest(),
 'source_zip_sha256':hashlib.sha256((ROOT/'candidate.zip').read_bytes()).hexdigest(),
 'performance':'UNMEASURED','admission':'BLOCKED_ZERO_OWNED_TRITON'}
(ROOT/'packet_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
