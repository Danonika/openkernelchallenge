"""CPU-only nested-session cleanup regression; no CUDA import or query."""
import importlib.util,json,os,subprocess,sys,tempfile,time
from pathlib import Path
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prepared',root/'run_prepared.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.enable_subreaper()
work=Path(tempfile.mkdtemp(prefix='rotary-descendant-test-'))
code="""import subprocess,sys,time,os,json
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
Path(sys.argv[1]).write_text(json.dumps({'parent':os.getpid(),'nested':child.pid}))
time.sleep(60)
"""
try:runner.command([sys.executable,'-c',code,str(work/'pids.json')],work,dict(os.environ),work/'test.log',0.5)
except subprocess.TimeoutExpired:pass
else:raise AssertionError('Expected timeout')
pids=json.loads((work/'pids.json').read_text())
assert all(runner._proc_record(pid) is None for pid in pids.values()),pids
assert not runner._owned_descendants()
# Also reject and clean an orphan after its direct parent exits normally.
code2=code.replace('time.sleep(60)\n','time.sleep(0.2)\n')
try:runner.command([sys.executable,'-c',code2,str(work/'pids2.json')],work,dict(os.environ),work/'test2.log',2)
except RuntimeError as exc:assert 'leaving descendants' in str(exc)
else:raise AssertionError('Expected orphan rejection')
pids2=json.loads((work/'pids2.json').read_text())
assert all(runner._proc_record(pid) is None for pid in pids2.values()),pids2
assert not runner._owned_descendants()
# A runner-directed SIGTERM must execute cleanup, not leave detached descendants.
script=work/'signal_runner.py'
script.write_text('''import importlib.util,signal,subprocess,sys,os
from pathlib import Path
spec=importlib.util.spec_from_file_location('r',sys.argv[1]);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.enable_subreaper()
def stop(signum,frame):raise InterruptedError('requested stop')
signal.signal(signal.SIGTERM,stop)
r.command([sys.executable,'-c',sys.argv[2],sys.argv[3]],Path(sys.argv[3]).parent,dict(os.environ),Path(sys.argv[3]).with_suffix('.log'),60)
''')
proc=subprocess.Popen([sys.executable,str(script),str(root/'run_prepared.py'),code,str(work/'signal-pids.json')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
for _ in range(200):
    if (work/'signal-pids.json').is_file():break
    time.sleep(.01)
else:raise AssertionError('Signal test did not launch')
proc.send_signal(__import__('signal').SIGTERM);assert proc.wait(timeout=10)!=0
signalpids=json.loads((work/'signal-pids.json').read_text())
assert all(runner._proc_record(pid) is None for pid in signalpids.values()),signalpids
# Isolate marker protocol fixtures from other agents' real canonical marker.
fixture=work/'quarantine';fixture.mkdir()
(fixture/'run_prepared.py').write_text((root/'run_prepared.py').read_text().replace('/tmp/soup-gpu0.lock',str(fixture/'gpu.lock')))
(fixture/'official_frozen_manifest.json').write_bytes((root/'official_frozen_manifest.json').read_bytes())
launch=fixture/'launch.py'
launch.write_text('''import run_prepared as r,subprocess,sys
def no_gpu(*args,**kwargs):raise subprocess.TimeoutExpired('CPU fixture: no GPU invoked',1)
r.command=no_gpu
r.main()
''')
args=[sys.executable,str(launch),'--prepared-repo','/workspace/shared/soup_cloud_env/source','--hub-cache','/nonexistent','--run-root',str(fixture/'runs'),'--backend','GCP','--confirm-cloud-gpu-run']
p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=10)
assert p.returncode==1,p.stderr
marker=fixture/'gpu.lock.blocked';assert marker.is_file()
assert '"operator_reconciliation_required": true' in p.stdout
# Another invocation must refuse that unresolved quarantine before any GPU call.
p2=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=10)
assert p2.returncode and 'quarantined' in p2.stderr
assert marker.is_file()
print(json.dumps(dict(status='PASS_CPU_ONLY',nested_session_timeout_cleanup=True,normal_exit_orphan_cleanup=True,sigterm_descendant_cleanup=True,pidfd_signals=True,outer_timeout_retains_quarantine=True,existing_quarantine_refused=True,gpu_executed=False),indent=2))
