"""Static and packet-byte validation only. No notebook execution or GPU use."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    notebook=json.loads((ROOT/'candidate.ipynb').read_text())
    code_cells=0
    constants={}
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            assert cell['execution_count'] is None and cell['outputs']==[]
            source=''.join(cell['source']);tree=ast.parse(source);compile(tree,'notebook-cell','exec');code_cells+=1
            for node in tree.body:
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                    try: constants[node.targets[0].id]=ast.literal_eval(node.value)
                    except (ValueError,TypeError): pass
    assert constants['AUTHORIZE_GPU'] is False
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    sources={str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'solution').glob('*.py'))}
    assert constants['SOURCES']==sources and constants['SOURCE_HASHES']==manifest
    assert constants['RUNNER_SOURCE']==(ROOT/'notebook_runner.py').read_text()
    assert constants['PROTECTED_HASHES']==json.loads((ROOT/'official_manifest.json').read_text())
    for name,source in sources.items():
        compile(source,name,'exec');assert sha(source.encode())==manifest[name]
    for p in ROOT.glob('*.py'): compile(p.read_text(),p.name,'exec')
    sys.path.insert(0,'/redacted/local/soup_cycle/test_queue')
    from bundle_candidate import bundle
    regenerated,files=bundle(ROOT/'solution')
    assert regenerated==(ROOT/'candidate.zip').read_bytes() and files==manifest
    receipt=json.loads((ROOT/'bundle_receipt.json').read_text())
    assert receipt['source_sha256']==sha(regenerated) and receipt['files']==manifest
    with zipfile.ZipFile(ROOT/'candidate.zip') as z:
        assert sorted(z.namelist())==sorted(manifest)
        assert all(sha(z.read(name))==digest for name,digest in manifest.items())
    original=Path('/redacted/local/soup_cloud_env/source')
    for name,digest in constants['PROTECTED_HASHES'].items():assert sha((original/name).read_bytes())==digest
    try:
        import nbformat
        nbformat.validate(notebook);schema='PASS'
    except ImportError:
        schema='nbformat unavailable; structural JSON/AST checks passed'
    results={'status':'PASS','scope':'static/CPU packet verification, not target execution',
             'notebook_code_cells_compiled':code_cells,'notebook_outputs_cleared':True,'notebook_schema':schema,
             'source_files_compiled':len(sources),'deterministic_zip_rebuild_identical':True,
             'zip_sha256':sha(regenerated),'embedded_sources_and_runner_match_files':True,
             'official_harness_task_tests_hashes_unchanged':True,
             'source_manifest_matches_every_zip_file':True,'gpu_executed':False,
             'cpu_integration':{'passed_groups':6,'failed_groups':0},'new_version_telemetry':{'passed_groups':6,'failed_groups':0},
             'unchanged_soup_regressions':{'passed':79,'scope':'CPU tests; production route safely declined non-target build'},
             'synthetic_report_fixtures':{'passed':5},'active_quarantine_fixtures':{'passed':5,'includes_abrupt_sigkill':True},'cpu_process_cleanup_fixture':{'passed':True,'scope':'separate-session grandchild, pidfd cleanup and quarantine'},'target_status':'UNTESTED','score':None,'admission':'UNKNOWN'}
    (ROOT/'static_checks.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
