"""Static review validation only. No package imports, GPU execution or network."""
import ast,base64,hashlib,importlib.util,json,sys,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'source_manifest.json').read_text())
with zipfile.ZipFile(root/'candidate.zip') as z:
    assert set(z.namelist())==set(manifest)
    assert not any('__pycache__' in n or n.endswith(('.pyc','.pyo')) for n in z.namelist())
    for n,wanted in manifest.items():
        assert hashlib.sha256(z.read(n)).hexdigest()==wanted==sha(root/n)
        if n.endswith('.py'):compile(z.read(n),n,'exec')
nb=json.loads((root/'candidate.ipynb').read_text())
assert nb['nbformat']==4
payload={};compiled_cells=0
for index,c in enumerate(nb['cells']):
    if c['cell_type']!='code':continue
    assert c['outputs']==[] and c['execution_count'] is None
    text=''.join(c['source']);compile(text,f'cell_{index}','exec');compiled_cells+=1
    for node in ast.parse(text).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='PAYLOAD':
            payload=ast.literal_eval(node.value)
assert payload
for n,b64 in payload.items():assert base64.b64decode(b64,validate=True)==(root/n).read_bytes()
for p in root.rglob('*.py'):compile(p.read_bytes(),str(p),'exec')

sys.path.insert(0,'/redacted/local/soup_cycle/test_queue')
from bundle_candidate import bundle
raw,expected=bundle(root/'solution')
assert raw==(root/'candidate.zip').read_bytes() and expected==manifest
from soup_queue import validate_artifact
validate_artifact(raw,'solution_bundle')
template=json.loads((root/'submission.template.json').read_text())
assert template['baseline_id']=='REPLACE_WITH_MEASURED_GCP_BASELINE_ID'
assert template['entrypoint']=='soup_screen_v1' and template['backend']=='GCP'

sys.path.insert(0,'/redacted/local/soup_cloud_env/source/harness')
from check_rules import source_kernels
own,copies,lookalikes,banned,broken=source_kernels(root/'solution')
assert not copies and not banned and not broken
assert {'rotary_apply_f32','rotary_adjoint_f32'}<=set(own)
report=dict(status='STATIC_PASS',compiled_code_cells=compiled_cells,solution_source_files=len(manifest),
            source_sha256=sha(root/'candidate.zip'),notebook_sha256=sha(root/'candidate.ipynb'),
            notebook_embedded_bytes_match=True,deterministic_queue_bundle=True,queue_artifact_schema_pass=True,
            official_static_owned_kernel_names=sorted(own),example_copies=copies,example_lookalikes=lookalikes,
            banned_imports=banned,broken_source=broken,gpu_executed=False,score=None,admission='UNKNOWN')
print(json.dumps(report,indent=2))
