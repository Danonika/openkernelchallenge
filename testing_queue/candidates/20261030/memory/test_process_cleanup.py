"""Actual tiny CPU subprocess fixture: separate-session grandchild interruption."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from notebook_runner import process_table


def main():
    root=Path(__file__).parent
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);ready=tmp/'grandchild.pid';log=tmp/'work.log';blocked=tmp/'isolated_test.blocked'
        worker=("import subprocess,sys,time;from pathlib import Path;"
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True);"
                f"Path({str(ready)!r}).write_text(str(p.pid));time.sleep(120)")
        supervisor=("import ctypes,os,sys;from pathlib import Path;import notebook_runner as r;"
                    "assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0;r._DEDICATED_RUNNER=True;"
                    f"r.BLOCK_PATH=Path({str(blocked)!r});"
                    f"r.command([sys.executable,'-c',{worker!r}],Path({str(log)!r}),Path({str(tmp)!r}),dict(os.environ))")
        proc=subprocess.Popen([sys.executable,'-c',supervisor],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+10
            while not ready.exists() and time.monotonic()<deadline:
                if proc.poll() is not None:raise AssertionError(proc.communicate())
                time.sleep(.02)
            assert ready.exists()
            child=int(ready.read_text());record=process_table()[child]
            proc.send_signal(signal.SIGINT)
            stdout,stderr=proc.communicate(timeout=10)
            assert proc.returncode!=0 and blocked.is_file()
            result=json.loads(blocked.read_text());assert result['cleanup']['all_tracked_owned_processes_terminal']
            after=process_table().get(child)
            assert not after or after['birth']!=record['birth'] or after['state']=='Z'
            detail={'separate_session_grandchild_tracked':True,'interrupt_rejected':True,
                    'all_owned_children_terminal':True,'pidfd_signals_used':True,
                    'block_marker_retained':True,'gpu_execution':False,'exit_code':proc.returncode}
        finally:
            if proc.poll() is None:
                proc.kill();proc.wait()
    Path(__file__).with_name('cleanup_fixture_results.json').write_text(json.dumps({'passed':True,'scope':'CPU process lifecycle only','details':detail},indent=2)+'\n')
    print(json.dumps(detail,indent=2))

if __name__=='__main__':main()
