"""CPU/static packet tests. Never import Torch/Triton or execute any GPU child."""
import ast,base64,hashlib,importlib.util,json,shutil,sys,zipfile
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parent
PACKET=ROOT/'microbench_packet_v3'
KIT=Path('/redacted/local/soup_web_ready_20261030_v2/support')

def ast_named(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

def test_notebook_payload_and_identity():
    receipt=json.loads((ROOT/'microbench_build_receipt_v3.json').read_text())
    raw=Path(receipt['notebook']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==receipt['notebook_sha256']
    nb=json.loads(raw);assert nb['nbformat']==4 and len(nb['cells'])==3
    for c in nb['cells']:
        if c['cell_type']=='code':
            assert c['execution_count'] is None and c['outputs']==[]
            compile(''.join(c['source']),'cell','exec')
    assignments={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(''.join(nb['cells'][1]['source'])).body
                 if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
                 and n.targets[0].id in ('PAYLOAD','PAYLOAD_SHA256')}
    for name,value in assignments['PAYLOAD'].items():
        decoded=base64.b64decode(value,validate=True)
        assert decoded==(PACKET/name).read_bytes()
        assert hashlib.sha256(decoded).hexdigest()==assignments['PAYLOAD_SHA256'][name]
    assert not (ROOT/'web_adapter').exists(),'No full-task notebook may have been issued'

def test_reviewed_protected_support_and_functions_unchanged():
    for name in ('process_guard.py','owned_process_tree.py','runtime_probe.py','official_manifest.json'):
        assert (PACKET/name).read_bytes()==(KIT/name).read_bytes()
    original,extracted=ast_named(KIT/'web_runner.py'),ast_named(PACKET/'reviewed_support.py')
    assert set(extracted)=={'sha','read','write','verify_official','idle_check','export_small','Exporter'}
    for name,node in extracted.items():assert node==original[name]

def test_only_kernel_signature_and_decorators_changed():
    old=ast.parse((PACKET/'old_solution/rotary.py').read_text())
    new=ast.parse((PACKET/'solution/rotary.py').read_text())
    assert len(old.body)==len(new.body)
    changed=[]
    for a,b in zip(old.body,new.body):
        if ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False):continue
        assert isinstance(a,ast.FunctionDef) and a.name in ('rotary_apply_f32','rotary_adjoint_f32')
        assert ast.dump(ast.Module(body=a.body,type_ignores=[]))==ast.dump(ast.Module(body=b.body,type_ignores=[]))
        assert a.name==b.name;changed.append(a.name)
    assert changed==['rotary_apply_f32','rotary_adjoint_f32']
    with zipfile.ZipFile(ROOT/'candidate.zip') as archive:
        manifest=json.loads((ROOT/'source_manifest.json').read_text())
        assert set(archive.namelist())==set(manifest)
        for n,h in manifest.items():assert hashlib.sha256(archive.read(n)).hexdigest()==h

def test_bounded_no_training_no_installs_cache_exclusion():
    driver=(PACKET/'driver.py').read_text();worker=(PACKET/'worker.py').read_text()
    for text in (driver,worker):
        tree=ast.parse(text)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('train','step','backward','from_pretrained') for n in calls)
        assert all(token not in text for token in ('run_task.py','pip install','http://','https://','torch.optim'))
    tree=ast.parse(driver)
    for n in ast.walk(tree):
        if isinstance(n,ast.Call):
            for kw in n.keywords:
                if kw.arg=='timeout':assert isinstance(kw.value,ast.Constant) and 0<kw.value.value<=80
    assert "run/'triton_cache'/method" in driver and "run/'inductor_cache'/method" in driver
    assert "'triton_cache'" in (PACKET/'reviewed_support.py').read_text()
    assert 'TRITON_CACHE_DIR' in worker and 'not any(cache.iterdir())' in worker
    assert 'retain_graph=True' in worker and 'before any warm timing' in worker

@pytest.mark.parametrize('failure', [None,'probe','preflight','old','payload'])
def test_orchestration_fail_closed_without_any_gpu_child(tmp_path,monkeypatch,failure):
    packet=tmp_path/'packet';shutil.copytree(PACKET,packet)
    sys.path.insert(0,str(packet))
    spec=importlib.util.spec_from_file_location('tested_micro_driver',packet/'driver.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.path.remove(str(packet))
    realpath=Path
    work=tmp_path/'work';work.mkdir();results=tmp_path/'results';results.mkdir()
    monkeypatch.setattr(module,'Path',lambda p:work if str(p)=='/kaggle/working' else realpath(p))
    monkeypatch.setattr(module,'LOCK',tmp_path/'lock');monkeypatch.setattr(module,'MARKER',tmp_path/'marker')
    monkeypatch.setenv('SOUP_REPO','/redacted/local/soup_cloud_env/source')
    monkeypatch.setenv('SOUP_RESULTS',str(results))
    calls=[];cleanup=[];artifacts=[]
    monkeypatch.setattr(module.tree,'prepare',lambda:cleanup.append('prepare'))
    monkeypatch.setattr(module.tree,'begin_quarantine',lambda p:cleanup.append('begin'))
    monkeypatch.setattr(module.tree,'quarantine',lambda reason:cleanup.append('quarantine'))
    monkeypatch.setattr(module.tree,'finish_quarantine',lambda:cleanup.append('finish'))
    monkeypatch.setattr(module,'idle_check',lambda *a:dict(exit_code=0,cleanup_confirmed=True))
    def fake_run(command,**kwargs):
        assert kwargs['timeout']<=80
        receipt=Path(str(kwargs['log'])+'.process.json')
        if receipt.exists():raise RuntimeError('Process receipt already exists; refuse to overwrite')
        receipt.write_text('{}')
        if str(command[2]).endswith('runtime_probe.py'):
            label='probe';output=Path(command[-1]);data={'passed':failure!='probe'}
        elif 'solution.web_preflight' in command:
            label='preflight';output=Path(command[-1]);data={'status':'PASSED_PREFLIGHT_ONLY'}
        else:
            label=command[command.index('--method')+1];output=Path(command[-1])
            data={'status':'PASSED_MICROBENCH_ONLY','rows':[{}]*5,'training_executed':False,'complete_fixed_task':False}
            cache=Path(kwargs['env']['TRITON_CACHE_DIR']);assert cache.is_dir() and not any(cache.iterdir())
        if label in ('native','old','runtime') and output.exists():raise FileExistsError(output)
        artifacts.extend(str(p) for p in (kwargs['log'],receipt,output))
        calls.append(label);output.write_text(json.dumps(data))
        return dict(exit_code=1 if failure==label else 0,cleanup_confirmed=True,timed_out=failure==label)
    monkeypatch.setattr(module.process_guard,'run',fake_run)
    if failure=='payload':(packet/'worker.py').write_text('raise RuntimeError("tampered")\n')
    with pytest.raises(SystemExit) as ex:module.main()
    assert ex.value.code==(0 if failure is None else 1)
    status=json.loads(next(results.glob('*/status.json')).read_text())
    assert status['score'] is None and status['admission']=='UNKNOWN' and not status['training_executed']
    assert not status['complete_fixed_task']
    if failure is None:
        assert status['status']=='PASSED_MICROBENCH_ONLY'
        assert calls==['probe','preflight','native','old','runtime']
        assert len(artifacts)==15 and len(set(artifacts))==15
        names={Path(p).name for p in artifacts}
        for role in ('runtime_probe','native','old','runtime'):
            assert {role+'.json',role+'.log',role+'.log.process.json'}<=names
        assert cleanup==['prepare','begin','finish']
    elif failure=='payload':assert calls==[] and cleanup==[]
    else:
        assert status['status']=='BLOCKED_FAILED_OR_CENSORED_MICROBENCH_ONLY'
        assert cleanup[-2:]==['quarantine','finish']
        assert calls[-1]==failure

def test_v2_real_collision_is_reproduced_without_child_execution(tmp_path,monkeypatch):
    """The previous mock omitted artifact existence checks; this reproduces the real failure."""
    monkeypatch.setattr(sys.modules[__name__],'PACKET',ROOT/'microbench_packet_v2')
    with pytest.raises(AssertionError):
        test_orchestration_fail_closed_without_any_gpu_child(tmp_path,monkeypatch,None)
    status=json.loads(next((tmp_path/'results').glob('*/status.json')).read_text())
    assert status['status']=='BLOCKED_FAILED_OR_CENSORED_MICROBENCH_ONLY'
    assert status['error']=="RuntimeError('Process receipt already exists; refuse to overwrite')"
    assert status['processes']['native']['exit_code']==0
    assert status['processes']['old']['exit_code']==0
    # The original probe survives under 'runtime'; no benchmark report overwrote it.
    probe=json.loads(next((tmp_path/'work').glob('*/runtime.json')).read_text())
    assert probe=={'passed':True}

def test_v3_delta_is_only_three_probe_artifact_names():
    before=ast.parse((ROOT/'microbench_packet_v2/driver.py').read_text())
    after=ast.parse((ROOT/'microbench_packet_v3/driver.py').read_text())
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            mapping={'runtime_probe.json':'runtime.json','runtime_probe.log':'runtime.log','runtime_probe':'runtime'}
            if isinstance(node.value,str) and node.value in mapping:
                return ast.copy_location(ast.Constant(mapping[node.value]),node)
            return node
    assert ast.dump(before,include_attributes=False)==ast.dump(Normalize().visit(after),include_attributes=False)
    for name in ('worker.py','runtime_probe.py','reviewed_support.py','process_guard.py','owned_process_tree.py','source_manifest.json'):
        assert (ROOT/'microbench_packet_v2'/name).read_bytes()==(ROOT/'microbench_packet_v3'/name).read_bytes()
    assert (ROOT/'microbench_packet_v2/solution/rotary.py').read_bytes()==(ROOT/'microbench_packet_v3/solution/rotary.py').read_bytes()
