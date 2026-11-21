import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile
ROOT = Path(__file__).resolve().parents[1]
Q = Path('/redacted/local/soup_cycle/test_queue')
sys.path.insert(0,str(Q))
from bundle_candidate import bundle
checks = []

def check(name, value):
    checks.append({'check':name, 'passed':bool(value)})
    assert value, name

python_files = sorted(ROOT.rglob('*.py'))
for p in python_files:
    compile(p.read_text(),str(p),'exec')
check('all_python_syntax', True)
nb = json.loads((ROOT/'candidate.ipynb').read_text())
check('notebook_format_4',nb['nbformat']==4)
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code':
        compile(''.join(c['source']),f'notebook-cell-{i}','exec')
        check(f'cell_{i}_outputs_cleared',c['outputs']==[] and c['execution_count'] is None)
source_cell = next(c for c in nb['cells'] if ''.join(c['source']).startswith('PAYLOAD = '))
tree = ast.parse(''.join(source_cell['source']))
payload = ast.literal_eval(tree.body[0].value)
for name, body in payload.items():
    check('embedded_'+name,body==(ROOT/name).read_text())
driver_cell = next(c for c in nb['cells'] if ''.join(c['source']).startswith('"""Prepared-assets-only'))
check('embedded_driver_matches',''.join(driver_cell['source'])==(ROOT/'notebook_driver.py').read_text())
raw, files=bundle(ROOT/'solution')
check('deterministic_bundle',raw==(ROOT/'candidate.zip').read_bytes() and bundle(ROOT/'solution')[0]==raw)
manifest=json.loads((ROOT/'source_manifest.json').read_text())
check('source_manifest',manifest==files)
with zipfile.ZipFile(ROOT/'candidate.zip') as z:
    check('zip_exact_source_set',sorted(z.namelist())==sorted(files))
    for name in z.namelist():
        check('zip_hash_'+name,hashlib.sha256(z.read(name)).hexdigest()==files[name])
official=Path('/redacted/local/soup_cloud_env/source')
frozen=json.loads((ROOT/'official_manifest.json').read_text())
check('official_commit',subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()=='c7c2fcd15972ce8b898d8d3ff732ee192c817ceb')
check('official_source_unchanged',all(hashlib.sha256((official/p).read_bytes()).hexdigest()==h for p,h in frozen.items()))
for p in (ROOT/'solution').glob('*.py'):
    tree=ast.parse(p.read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                check('native_backward_not_replaced_'+p.name, not (isinstance(target,ast.Attribute) and target.attr=='backward'))
check('no_copied_old_decoder', (ROOT/'solution/packed_nf4.py').read_bytes() != Path('/redacted/local/soup_gpu_trials/trial002/solution/nf4_decode.py').read_bytes())
check('submission_baseline_unresolved',json.loads((ROOT/'submission.template.json').read_text())['baseline_id']=='REPLACE_WITH_MEASURED_GCP_BASELINE_ID')
# Authorization fails before environment access/GPU/lock acquisition.
spec=importlib.util.spec_from_file_location('nf4_driver',ROOT/'notebook_driver.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
try:
    mod.run({'authorized':False},{},{})
    raise AssertionError('authorization gate failed')
except RuntimeError as e:
    check('default_authorization_gate', 'authorization' in str(e))
# CPU-only process-tree timeout fixture, including detached child process group.
with tempfile.TemporaryDirectory(prefix='nf4-timeout-fixture-') as tmp:
    tmp=Path(tmp)
    mod.QUARANTINE_PATH=tmp/'quarantine.json'
    child_pid=tmp/'child.pid'
    source=("import subprocess,sys,time,pathlib; "
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True); "
            f"pathlib.Path({str(child_pid)!r}).write_text(str(p.pid)); time.sleep(60)")
    receipt=mod.command([sys.executable,'-c',source],ROOT,dict(__import__('os').environ),tmp/'run.log',1)
    check('timeout_censored',receipt['outer_timeout'] and receipt['exit_code']!=0)
    check('timeout_lane_quarantined',mod.QUARANTINE_PATH.is_file())
    pid=int(child_pid.read_text())
    check('detached_child_targeted',pid in receipt['timeout_owned_pids_signalled'])
    stat=Path('/proc')/str(pid)/'stat'
    # Killed child may remain a zombie briefly; it cannot execute GPU or CPU work.
    state=stat.read_text().rsplit(')',1)[1].split()[0] if stat.exists() else 'absent'
    check('detached_child_no_longer_running',state in ('Z','absent'))
# Real KeyboardInterrupt while a detached child is alive: receipt persists and
# both owned PID identities terminate before the lock-owning caller unwinds.
import os
import signal
import threading
with tempfile.TemporaryDirectory(prefix='nf4-interrupt-fixture-') as tmp:
    tmp=Path(tmp)
    mod.QUARANTINE_PATH=tmp/'quarantine.json'
    child_pid=tmp/'child.pid'
    source=("import subprocess,sys,time,pathlib; "
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True); "
            f"pathlib.Path({str(child_pid)!r}).write_text(str(p.pid)); time.sleep(60)")
    timer=threading.Timer(1.2,lambda:os.kill(os.getpid(),signal.SIGINT))
    timer.start()
    try:
        try:
            mod.command([sys.executable,'-c',source],ROOT,dict(os.environ),tmp/'run.log',30)
            raise AssertionError('interruption not propagated')
        except KeyboardInterrupt:
            pass
    finally:
        timer.cancel();timer.join()
    receipt=json.loads((tmp/'run.log.receipt.json').read_text())
    check('keyboard_interrupt_receipt',receipt['interrupted']=='KeyboardInterrupt')
    check('interrupted_lane_quarantined',mod.QUARANTINE_PATH.is_file())
    check('keyboard_interrupt_cleanup_confirmed',receipt['cleanup_confirmed'] is True)
    pid=int(child_pid.read_text())
    check('keyboard_interrupt_detached_child_targeted',pid in receipt['timeout_owned_pids_signalled'])
    record=mod.process_record(pid)
    check('keyboard_interrupt_child_not_running',record is None or record['state']=='Z')
# Stale identity cannot signal a new process; already-exited root is harmless.
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
identity=mod.process_record(p.pid)
mod.signal_owned(p.pid,identity['birth']+1,signal.SIGKILL)
check('pid_birth_identity_guard',p.poll() is None)
p.terminate();p.wait()
_, confirmed=mod.terminate_owned_tree(p,{p.pid:identity['birth']})
check('already_exited_root_cleanup_race',confirmed)
# Constructor-interruption uncertainty is quarantined even without a PID handle.
with tempfile.TemporaryDirectory(prefix='nf4-constructor-fixture-') as tmp:
    tmp=Path(tmp);mod.QUARANTINE_PATH=tmp/'quarantine.json'
    original_popen=mod.subprocess.Popen
    def interrupted_popen(*args,**kwargs):
        raise KeyboardInterrupt()
    mod.subprocess.Popen=interrupted_popen
    try:
        try:
            mod.command([sys.executable,'-c','pass'],ROOT,dict(os.environ),tmp/'run.log',30)
            raise AssertionError('constructor interruption not propagated')
        except KeyboardInterrupt:
            pass
    finally:
        mod.subprocess.Popen=original_popen
    receipt=json.loads((tmp/'run.log.receipt.json').read_text())
    check('constructor_interruption_unknown',receipt['exit_code'] is None and receipt['cleanup_confirmed'] is None)
    check('constructor_interruption_quarantined',mod.QUARANTINE_PATH.is_file())
# Full-run lock wrapper writes its crash marker before work and clears only a
# normal verified return. No GPU/prepared-asset function is called in this fixture.
with tempfile.TemporaryDirectory(prefix='nf4-crash-marker-fixture-') as tmp:
    tmp=Path(tmp);mod.QUARANTINE_PATH=tmp/'quarantine.json';mod.GPU_LOCK_PATH=tmp/'gpu.lock'
    original_run=mod._run_locked
    def completed(*args):
        marker=json.loads(mod.QUARANTINE_PATH.read_text())
        check('owner_marker_exists_before_work',bool(marker.get('owner')))
        return 'complete'
    mod._run_locked=completed
    check('normal_completion_result',mod.run({'authorized':True},{},{})=='complete')
    check('normal_completion_clears_owner_marker',not mod.QUARANTINE_PATH.exists())
    def failed(*args):
        raise RuntimeError('synthetic preflight failure')
    mod._run_locked=failed
    try:
        mod.run({'authorized':True},{},{})
    except RuntimeError:
        pass
    check('exception_retains_owner_marker',mod.QUARANTINE_PATH.exists())
    try:
        mod.run({'authorized':True},{},{})
    except RuntimeError as exc:
        check('abandoned_marker_blocks_new_run','reconciliation' in str(exc))
    else:
        raise AssertionError('abandoned marker was ignored')
    mod._run_locked=original_run
# Abrupt SIGKILL cannot invoke finally; the pre-created marker must survive it.
with tempfile.TemporaryDirectory(prefix='nf4-sigkill-fixture-') as tmp:
    tmp=Path(tmp);marker=tmp/'quarantine.json';ready=tmp/'ready'
    source=("import importlib.util,pathlib,time; "
            f"s=importlib.util.spec_from_file_location('d',{str(ROOT/'notebook_driver.py')!r}); "
            "d=importlib.util.module_from_spec(s);s.loader.exec_module(d); "
            f"d.QUARANTINE_PATH=pathlib.Path({str(marker)!r});d.GPU_LOCK_PATH=pathlib.Path({str(tmp/'lock')!r}); "
            f"d._run_locked=lambda *args:(pathlib.Path({str(ready)!r}).write_text('ready'),time.sleep(60)); "
            "d.run({'authorized':True},{},{})")
    guard=subprocess.Popen([sys.executable,'-c',source])
    try:
        until=time.monotonic()+5
        while not ready.exists() and time.monotonic()<until:
            time.sleep(.01)
        check('sigkill_fixture_started',ready.exists())
        guard.kill();guard.wait(timeout=10)
        check('sigkill_retains_owner_marker',marker.is_file() and bool(json.loads(marker.read_text()).get('owner')))
    finally:
        if guard.poll() is None:
            guard.kill();guard.wait()
with tempfile.TemporaryDirectory(prefix='nf4-sigterm-fixture-') as tmp:
    tmp=Path(tmp);mod.QUARANTINE_PATH=tmp/'quarantine.json';mod.GPU_LOCK_PATH=tmp/'gpu.lock'
    original_run=mod._run_locked;previous_handler=signal.getsignal(signal.SIGTERM)
    def term(*args):
        os.kill(os.getpid(),signal.SIGTERM)
    mod._run_locked=term
    try:
        try:
            mod.run({'authorized':True},{},{})
            raise AssertionError('SIGTERM not propagated')
        except SystemExit as exc:
            check('sigterm_converted_to_cleanup_exception',exc.code==143)
        check('sigterm_retains_quarantine',mod.QUARANTINE_PATH.is_file())
        check('sigterm_handler_restored',signal.getsignal(signal.SIGTERM)==previous_handler)
    finally:
        mod._run_locked=original_run
result={'status':'STATIC_AND_CPU_ORCHESTRATION_CHECKS_PASSED','gpu_tested':False,
        'checks':checks,'checks_count':len(checks),'python_files_compiled':len(python_files),
        'candidate_sha256':hashlib.sha256(raw).hexdigest(),
        'notebook_sha256':hashlib.sha256((ROOT/'candidate.ipynb').read_bytes()).hexdigest(),
        'official_source_files_checked':len(frozen),'admission':'UNKNOWN','score':None}
(ROOT/'static_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
