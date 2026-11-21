"""Validate packet integrity/source contracts and cheap CPU infrastructure fixtures.
No torch import, CUDA query/compile/launch, model execution, or network access.
"""
import ast
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT=Path(__file__).resolve().parent
QUEUE=Path('/redacted/local/soup_cycle/test_queue')
OFFICIAL=Path('/redacted/local/soup_cloud_env/source')
checks=[]
def sha_bytes(raw): return hashlib.sha256(raw).hexdigest()
def sha(p): return sha_bytes(Path(p).read_bytes())
def record(name,details): checks.append({'name':name,'passed':True,'details':details})
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

for file in sorted(ROOT.rglob('*.py')):
    if '__pycache__' not in file.parts: compile(file.read_text(),str(file),'exec')
record('all_python_syntax',{'files':len(list(ROOT.rglob('*.py')))})
source_manifest=json.loads((ROOT/'source_manifest.json').read_text())
actual={'solution/'+str(p.relative_to(ROOT/'solution')):sha(p) for p in sorted((ROOT/'solution').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
assert source_manifest==actual
raw=(ROOT/'candidate.zip').read_bytes()
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    assert set(z.namelist())==set(source_manifest)
    assert {name:sha_bytes(z.read(name)) for name in z.namelist()}==source_manifest
    assert all(i.date_time==(2026,1,1,0,0,0) for i in z.infolist())
if QUEUE.exists():
    sys.path.insert(0,str(QUEUE));import bundle_candidate
    regenerated,files=bundle_candidate.bundle(ROOT/'solution')
    assert regenerated==raw and files==source_manifest
record('exact_and_deterministic_solution_zip',{'files':len(actual),'sha256':sha_bytes(raw)})
nb=json.loads((ROOT/'candidate.ipynb').read_text());assert nb['nbformat']==4
literal={}
for cell in nb['cells']:
    if cell['cell_type']=='code':
        assert cell['outputs']==[] and cell['execution_count'] is None
        tree=ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                try: literal[node.targets[0].id]=ast.literal_eval(node.value)
                except (ValueError,TypeError): pass
assert base64.b64decode(literal['BUNDLE_B64'],validate=True)==raw
assert literal['BUNDLE_SHA256']==sha_bytes(raw)
assert literal['SOURCE_MANIFEST']==actual
assert literal['RUNNER_SOURCE']==(ROOT/'notebook_runner.py').read_text()
assert literal['CONFIG']['execute_gpu'] is False
record('notebook_syntax_cleared_and_embedded_bytes',{'cells':len(nb['cells']),'code_cells':sum(c['cell_type']=='code' for c in nb['cells'])})
contract=json.loads((ROOT/'numeric_contract.json').read_text())
tree=ast.parse((ROOT/'solution/materialized_ce.py').read_text())
functions={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
kernels={n:sha_bytes(ast.dump(node,include_attributes=False).encode()) for n,node in functions.items() if n.startswith('soup_ce_exp_')}
assert kernels==contract['kernel_ast_sha256'] and len(kernels)==5
assert sha_bytes(ast.dump(functions['native_causal_loss'],include_attributes=False).encode())==contract['native_causal_loss_ast_sha256']
assert sha(ROOT/'solution/numerical_gates.py')==contract['numerical_gates_sha256']

fn=functions['materialized_causal_loss'];topifs={ast.unparse(n.test):n for n in fn.body if isinstance(n,ast.If)}
assert all(isinstance(topifs[key].body[0],ast.Return) for key in ('invalid','near_certain'))
assert '_denominator_metadata' not in (ROOT/'solution/adapter.py').read_text()
assert '.item()' not in '\n'.join(line for line in (ROOT/'solution/adapter.py').read_text().splitlines() if not line.lstrip().startswith('#'))
record('numeric_contract_unchanged',{'kernel_count':5,'native_expression_ast_unchanged':True,'fixed_gates_bytes_unchanged':True,'fallback_graph_context_preserved':True,'adapter_duplicate_item_removed':True})
official_manifest=json.loads((ROOT/'official_manifest.json').read_text())
assert literal['OFFICIAL_MANIFEST']==official_manifest
if OFFICIAL.exists():
    assert all(sha(OFFICIAL/p)==h for p,h in official_manifest.items())
    sys.path.insert(0,str(OFFICIAL/'harness'));check_rules=load_module('official_ce_static_check_rules',OFFICIAL/'harness/check_rules.py')
    own,copies,lookalikes,banned,broken=check_rules.source_kernels(ROOT/'solution')
    assert len(own)==5 and not copies and not banned and not broken
    record('official_static_source_scan',{'own_named_kernels':list(own),'copies':copies,'lookalikes':lookalikes,'banned':banned,'broken':broken,'not_jury_originality_approval':True})
record('official_source_manifest',{'files':len(official_manifest),'matches_local_official':OFFICIAL.exists()})
runner=load_module('ce_prepared_runner',ROOT/'notebook_runner.py')
assert runner.validate_zip(raw,actual)
assert runner.completed_result({})[0] is False

run={'exit_code':0,'metrics':{'exit_code':0,'steps':64},'global_step':64,'adapter_config':{'fixture':True},'adapter_tensors':{'fixture':[]},'wall_s':1,'gpu_peak_mib':1}
fixture={'task':{'repeats':2,'seed':1,'only':None},'runs':{'base':[run.copy(),run.copy()],'solution':[run.copy(),run.copy()]},'evals':{'base':{'exit_code':0,'mean_loss':2.},'solution':{'exit_code':0,'mean_loss':2.}},'packages_changed':[]}
assert runner.completed_result(fixture)[0]
fixture['runs']['solution'][0]['exit_code']=-9
assert not runner.completed_result(fixture)[0]
fixture['runs']['solution'][0]['exit_code']=0;fixture['evals']['solution']['mean_loss']=float('nan')
assert not runner.completed_result(fixture)[0]
record('completion_fail_closed_fixtures',{'empty_rejected':True,'native_timeout_with_nested_64_rejected':True,'nonfinite_quality_rejected':True})
with tempfile.TemporaryDirectory(prefix='ce-cpu-validation-') as temp:
    tmp=Path(temp);log=tmp/'route.log'
    log.write_text('progress\rSOUP_CE_V3_EVENT {"event":"loss_boundary","sequence":1}\nSOUP_CE_V3_EVENT {"event":')
    parsed=runner.parse_routes(log)
    assert len(parsed['records'])==1 and parsed['partial_lines']==1
    assert runner.parse_routes(tmp/'missing')['status']=='UNKNOWN'

    pidfile=tmp/'descendant.pid'
    code="import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(20)"
    status,wall=runner.execute([sys.executable,'-c',code,str(pidfile)],cwd=tmp,env=None,logfile=tmp/'outer.log',timeout=0.5,quarantine_path=tmp/'blocked')
    assert status==-9 and pidfile.exists() and (tmp/'blocked').exists()
    cleanup=json.loads((tmp/'outer.log.cleanup.json').read_text())
    assert cleanup['verified_known_tree_terminated']
    descendant=int(pidfile.read_text());stat=Path(f'/proc/{descendant}/stat')
    if stat.exists(): assert stat.read_text().rsplit(')',1)[1].split()[0] in ('Z','X')
with tempfile.TemporaryDirectory(prefix='ce-public-cache-fixture-') as temp:
    cache=Path(temp);root=cache/'models--Qwen--Qwen2.5-1.5B-Instruct'
    snapshot=root/'snapshots'/runner.MODEL_REVISION
    for p in (root/'refs',root/'blobs',snapshot): p.mkdir(parents=True,exist_ok=True)
    (root/'refs/main').write_text(runner.MODEL_REVISION)
    (root/'blobs/config').write_text('{"fixture":true}')
    (snapshot/'config.json').symlink_to('../../blobs/config')
    assert runner.public_model_manifest(cache)['revision']==runner.MODEL_REVISION
    (root/'refs/main').write_text('0'*40)
    try: runner.public_model_manifest(cache)
    except RuntimeError: pass
    else: raise AssertionError('Drifted model main accepted')
    (root/'refs/main').write_text(runner.MODEL_REVISION)
    (snapshot/'config.json').unlink();(snapshot/'config.json').symlink_to('/etc/hostname')
    try: runner.public_model_manifest(cache)
    except RuntimeError: pass
    else: raise AssertionError('Escaped model blob accepted')
record('pinned_public_model_cache_cpu_fixtures',{'wrong_main_rejected':True,'escaped_blob_rejected':True})
record('route_parser_and_owned_timeout_cleanup_cpu_fixture',{'partial_line_ignored':True,'missing_unknown':True,'nested_session_child_terminated':True})
sub=json.loads((ROOT/'submission.template.json').read_text())
assert sub['baseline_id']=='REPLACE_WITH_MEASURED_GCP_BASELINE_ID' and sub['artifact_kind']=='solution_bundle' and sub['entrypoint']=='soup_screen_v1'
record('queue_template_unresolved_baseline',{'submitted':False,'baseline_fabricated':False})
output={'status':'PASS_STATIC_AND_CPU_INFRASTRUCTURE_ONLY','checks':checks,'gpu_queried':False,'gpu_executed':False,
        'performance_claim':False,'official_admission':'UNKNOWN','score':None}
(ROOT/'validation/static_checks.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
