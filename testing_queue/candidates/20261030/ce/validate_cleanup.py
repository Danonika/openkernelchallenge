"""CPU-only supervisor lifecycle tests, never uses the real GPU lock/marker."""
import importlib.util,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
reports=[]
spec=importlib.util.spec_from_file_location('ce_runner_fixture',ROOT/'notebook_runner.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
with tempfile.TemporaryDirectory(prefix='ce-cleanup-fixtures-') as temporary:
    base=Path(temporary)
    for mode in ('timeout','SIGINT','SIGTERM'):
        root=base/mode;root.mkdir();pids=root/'owned_pids.jsonl'
        workload=root/'workload.py'
        workload.write_text("""import subprocess,sys,time,json,os
from pathlib import Path
out=Path(sys.argv[1])
with out.open('a') as f: f.write(json.dumps({'pid':os.getpid(),'kind':'root'})+'\\n');f.flush()
children=[]
for i in range(100):
 p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
 children.append(p)
 with out.open('a') as f: f.write(json.dumps({'pid':p.pid,'kind':'nested-session'})+'\\n');f.flush()
 time.sleep(.02)
time.sleep(60)
""")
        driver=root/'driver.py'
        driver.write_text("""import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('runner',sys.argv[1]);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
root=Path(sys.argv[2]); result={}
try:
 code,wall=r.execute([sys.executable,str(root/'workload.py'),str(root/'owned_pids.jsonl')],cwd=root,env=None,logfile=root/'run.log',timeout=float(sys.argv[3]),quarantine_path=root/'blocked')
 result={'code':code,'wall':wall}
except BaseException as error: result={'interruption':type(error).__name__}
(root/'driver-result.json').write_text(json.dumps(result))
""")
        driver_proc=subprocess.Popen([sys.executable,str(driver),str(ROOT/'notebook_runner.py'),str(root),'0.5' if mode=='timeout' else '20'])
        deadline=time.monotonic()+5
        while not pids.exists() and driver_proc.poll() is None and time.monotonic()<deadline: time.sleep(.01)
        assert pids.exists(),mode
        before_signal_count=None
        if mode!='timeout':
            time.sleep(.04)
            before_signal_count=len(pids.read_text().splitlines())
            assert before_signal_count<101,'Fixture did not interrupt during active spawning'
            driver_proc.send_signal(getattr(signal,mode))
        driver_proc.wait(timeout=15)
        assert driver_proc.returncode==0,(mode,driver_proc.returncode)
        result=json.loads((root/'driver-result.json').read_text())
        cleanup=json.loads((root/'run.log.cleanup.json').read_text())
        assert cleanup['verified_known_tree_terminated'],cleanup
        assert (root/'blocked').exists()
        assert (result.get('code')==-9) if mode=='timeout' else result.get('interruption')=='KeyboardInterrupt'
        recorded=[json.loads(line)['pid'] for line in pids.read_text().splitlines()]
        live=[]
        for pid in recorded:
            path=Path(f'/proc/{pid}/stat')
            if path.exists() and path.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X'): live.append(pid)
        assert not live,(mode,live)
        reports.append({'mode':mode,'passed':True,'known_owned_count':cleanup['known_owned_count'],
                        'recorded_workload_pids':len(recorded),'before_signal_count':before_signal_count,'all_recorded_stopped':True,
                        'frozen_census_stable':cleanup['frozen_census_stable'],
                        'quarantine_persisted':True,'driver_result':result})
    # The active owner guard exists before work; wrong owners/foreign markers
    # cannot be cleared, and an abruptly killed owner cannot remove its guard.
    owner_root=base/'owner-marker';owner_root.mkdir();marker=owner_root/'blocked'
    runner.create_active_marker(marker,'owner-a')
    original=marker.read_bytes()
    assert not runner.clear_owned_marker(marker,'owner-b') and marker.read_bytes()==original
    try: runner.create_active_marker(marker,'owner-b')
    except FileExistsError: pass
    else: raise AssertionError('Foreign marker overwritten')
    assert runner.clear_owned_marker(marker,'owner-a') and not marker.exists()
    runner.create_active_marker(marker,'foreign-owner')
    foreign=marker.read_bytes()
    assert not runner.clear_owned_marker(marker,'owner-a') and marker.read_bytes()==foreign
    assert runner.clear_owned_marker(marker,'foreign-owner')
    code="import importlib.util,sys,time;from pathlib import Path;s=importlib.util.spec_from_file_location('r',sys.argv[1]);r=importlib.util.module_from_spec(s);s.loader.exec_module(r);r.create_active_marker(Path(sys.argv[2]),'abrupt-owner');Path(sys.argv[2]+'.ready').write_text('ready');time.sleep(60)"
    killed=subprocess.Popen([sys.executable,'-c',code,str(ROOT/'notebook_runner.py'),str(marker)])
    deadline=time.monotonic()+5
    while not Path(str(marker)+'.ready').exists() and killed.poll() is None and time.monotonic()<deadline: time.sleep(.01)
    assert Path(str(marker)+'.ready').exists() and marker.exists();before=marker.read_bytes();killed.kill();killed.wait(timeout=5)
    assert killed.returncode<0 and marker.read_bytes()==before
    assert not runner.clear_owned_marker(marker,'abrupt-owner')  # same token but different PID
    reports.append({'mode':'owner_and_foreign_marker_SIGKILL','passed':True,
                    'wrong_owner_rejected':True,'foreign_marker_unchanged':True,
                    'own_normal_clear_verified':True,'SIGKILL_guard_persisted':True})
    # A normally exiting command must not leave even observed CPU-only orphans.
    orphan_root=base/'normal-orphan';orphan_root.mkdir()
    orphan_code="import subprocess,sys,time;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True);time.sleep(.4)"
    try:
        runner.execute([sys.executable,'-c',orphan_code],cwd=orphan_root,env=None,
                       logfile=orphan_root/'run.log',timeout=5,quarantine_path=orphan_root/'blocked')
    except RuntimeError as error: assert 'Owned descendants survived' in str(error)
    else: raise AssertionError('Normal exit with surviving child accepted')
    cleanup=json.loads((orphan_root/'run.log.cleanup.json').read_text())
    assert cleanup['verified_known_tree_terminated'] and (orphan_root/'blocked').exists()
    reports.append({'mode':'normal_exit_with_known_orphan','passed':True,
                    'survivor_rejected_and_terminated':True,'quarantine_persisted':True})
output={'status':'PASS_CPU_PROCESS_LIFECYCLE_ONLY','fixtures':reports,'gpu_executed':False,
 'real_gpu_lock_or_marker_touched':False,'limits':'SIGKILL fixture proves persistent guard, not descendant cleanup. Unobserved preexisting orphans remain outside these fixtures; interrupted/failed owner guard requires operator reconciliation.'}
(ROOT/'validation/cleanup_checks.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
