#!/usr/bin/env python3
"""Local static validation only; no GPU, model, queue, install or network action."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import py_compile
import sys
import zipfile
root=Path(__file__).resolve().parent
sys.path.insert(0,'/redacted/local/soup_cycle/test_queue')
from bundle_candidate import bundle
python_files=sorted(root.rglob('*.py'))
for p in python_files:
    ast.parse(p.read_text(),filename=str(p))
    py_compile.compile(str(p),doraise=True)
notebook=json.loads((root/'candidate.ipynb').read_text())
assert notebook['nbformat']==4
code_cells=0
for c in notebook['cells']:
    if c['cell_type']=='code':
        assert c['execution_count'] is None and c['outputs']==[]
        compile(''.join(c['source']),'<notebook>','exec');code_cells+=1
payload_cell=''.join(notebook['cells'][2]['source'])
tree=ast.parse(payload_cell)
payload=ast.literal_eval(tree.body[0].value)
expected=ast.literal_eval(tree.body[1].value)
for name,source in payload.items():
    assert (root/name).read_text()==source
    assert hashlib.sha256(source.encode()).hexdigest()==expected[name]
raw,files=bundle(root/'solution')
assert raw==(root/'candidate.zip').read_bytes()
assert files==json.loads((root/'candidate-files.json').read_text())
with zipfile.ZipFile(root/'candidate.zip') as z:
    assert sorted(z.namelist())==sorted(files)
    for name,sha in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
spec=json.loads((root/'submission.template.json').read_text())
assert set(spec)=={'candidate_id','backend','baseline_id','entrypoint','artifact_kind','priority','contains_secrets'}
assert spec['baseline_id']=='REPLACE_WITH_MEASURED_GCP_BASELINE_ID'
assert spec['entrypoint']=='soup_screen_v1' and spec['artifact_kind']=='solution_bundle' and spec['contains_secrets'] is False
sys.path.insert(0,'/redacted/local/soup_cloud_env/source/harness')
import check_rules
own,copies,lookalikes,banned,broken=check_rules.source_kernels(root/'solution')
assert set(own)=={'lora_half_scale_add'} and not copies and not banned and not broken
# Access metadata only, not device APIs; no CUDA initialization.
import importlib.metadata as md
log=(root/'cpu-tests.log').read_text()
assert '13 passed' in log
record={'status':'PREPARED_T4_UNTESTED','static_checks':'PASS','cpu_tests':'13 passed',
        'python':platform.python_version(),'torch':md.version('torch'),'triton':md.version('triton'),
        'pinned_target_stack':False,'gpu_used':False,'cuda_or_sm75_compile':'NOT_RUN',
        'actual_peft_integration':'NOT_RUN','full_soup_harness':'NOT_RUN','official_tests':'NOT_RUN',
        'official_checks':'NOT_RUN','quality':'UNMEASURED','own_share':'UNMEASURED',
        'performance':'UNMEASURED','admission':'UNKNOWN','score':None,
        'source_python_files':len(python_files),'notebook_code_cells':code_cells,
        'notebook_outputs_cleared':True,'embedded_source_matches':True,
        'deterministic_zip_rebuild_matches':True,'zip_sha256':hashlib.sha256(raw).hexdigest(),
        'official_static_source_scan':{'own_kernel_names':sorted(own),'example_copies':copies,
                                      'example_lookalikes':lookalikes,'banned_imports':banned,'broken':broken},
        'cpu_test_log_sha256':hashlib.sha256(log.encode()).hexdigest(),
        'tests_scope':['CPU interpreter forward/VJP/tails/casts','nonfinite/signed-zero/overflow','native CPU fallback',
                       'second derivative','synthetic AST rejection/order','synthetic separate q/v dropout/RNG/gradients',
                       'synthetic complete-result no-score gate','nine synthetic partial/failure suppression cases','normal process cleanup','nested-session timeout cleanup','normally exiting orphan parent cleanup','SIGTERM cleanup retains quarantine','quarantine owner/foreign-token protection']}
(root/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2), flush=True)
files_out={}
for p in sorted(root.rglob('*')):
    if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts and not p.name.endswith(('.pyc','.pyo')):
        files_out[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'candidate_id':spec['candidate_id'],'status':'PREPARED_T4_UNTESTED','official_commit':'c7c2fcd15972ce8b898d8d3ff732ee192c817ceb',
          'zip_sha256':record['zip_sha256'],'solution_files':files,'files':files_out,
          'baseline_id':None,'queue_submitted':False,'gpu_run':False,'score':None,'admission':'UNKNOWN'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
