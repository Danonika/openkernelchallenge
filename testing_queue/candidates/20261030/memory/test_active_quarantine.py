"""CPU-only active-marker protocol fixtures; no training/GPU call."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import notebook_runner as runner


def main():
    results=[]
    old=(runner.LOCK_PATH,runner.BLOCK_PATH,runner._DEDICATED_RUNNER,runner._run_prepared_locked)
    try:
        for case in ('normal_completion','scientific_failure','raised_failure','foreign_token'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);runner.LOCK_PATH=root/'lock';runner.BLOCK_PATH=root/'lock.blocked';runner._DEDICATED_RUNNER=True
                def work(**kwargs):
                    active=json.loads(runner.BLOCK_PATH.read_text())
                    assert active['phase']=='active' and active['owner_token']
                    if case=='raised_failure':raise RuntimeError('synthetic failure')
                    if case=='foreign_token':runner.BLOCK_PATH.write_text(json.dumps({'owner_token':'foreign','phase':'active'}))
                    return root,{'clean_process_completion':case!='scientific_failure'}
                runner._run_prepared_locked=work
                try:runner.run_prepared(authorize_gpu=True)
                except RuntimeError:
                    assert case=='raised_failure'
                assert runner.BLOCK_PATH.exists() is (case!='normal_completion')
                if case=='foreign_token':assert json.loads(runner.BLOCK_PATH.read_text())['owner_token']=='foreign'
                results.append({'case':case,'passed':True,'marker_present':runner.BLOCK_PATH.exists()})
    finally:
        runner.LOCK_PATH,runner.BLOCK_PATH,runner._DEDICATED_RUNNER,runner._run_prepared_locked=old
    # A separate tiny CPU process is killed without exception/finally handling.
    # Its active marker must remain. This is not an actual GPU run.
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);marker=root/'lock.blocked';lock=root/'lock'
        source=("import time,json;from pathlib import Path;import notebook_runner as r;"
                f"r.LOCK_PATH=Path({str(lock)!r});r.BLOCK_PATH=Path({str(marker)!r});r._DEDICATED_RUNNER=True;"
                "r._run_prepared_locked=lambda **kwargs: time.sleep(120);r.run_prepared(authorize_gpu=True)")
        proc=subprocess.Popen([sys.executable,'-c',source],cwd=Path(__file__).parent)
        descriptor=os.pidfd_open(proc.pid)
        try:
            until=time.monotonic()+10
            while not marker.exists() and time.monotonic()<until:time.sleep(.01)
            assert marker.is_file()
            signal.pidfd_send_signal(descriptor,signal.SIGKILL,None,0);proc.wait(timeout=5)
            evidence=json.loads(marker.read_text());assert evidence['owner_token'] and evidence['phase']=='active'
            results.append({'case':'abrupt_sigkill','passed':True,'marker_retained':True})
        finally:
            os.close(descriptor)
            if proc.poll() is None:proc.kill();proc.wait()
    report={'passed':True,'gpu_executed':False,'scope':'active owner-token quarantine lifecycle only','cases':results}
    Path(__file__).with_name('active_quarantine_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
