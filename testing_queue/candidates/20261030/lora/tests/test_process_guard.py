"""CPU process fixtures only. Never import/query GPU or touch unrelated PIDs."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import process_guard as guard


def test_normal_process_no_descendants(tmp_path):
    guard.prepare()
    with (tmp_path/'normal.log').open('w') as log:
        assert guard.run([sys.executable,'-c','print("normal")'],cwd=tmp_path,env=dict(os.environ),stdout=log,timeout=5)==0
    assert not guard.descendants()


def test_timeout_nested_session_cleanup(tmp_path):
    import pytest
    guard.prepare()
    script="""import subprocess,sys,time,json,os
from pathlib import Path
c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
Path(sys.argv[1]).write_text(json.dumps([os.getpid(),c.pid]))
time.sleep(30)
"""
    with (tmp_path/'nested.log').open('w') as log:
        with pytest.raises(subprocess.TimeoutExpired):
            guard.run([sys.executable,'-c',script,str(tmp_path/'pids')],cwd=tmp_path,env=dict(os.environ),stdout=log,timeout=.4)
    assert all(guard.identity(pid) is None for pid in json.loads((tmp_path/'pids').read_text()))
    assert not guard.descendants()


def test_orphan_normal_parent_exit_is_rejected_and_cleaned(tmp_path):
    import pytest
    guard.prepare()
    script="""import subprocess,sys,time,json,os
from pathlib import Path
c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
Path(sys.argv[1]).write_text(json.dumps([os.getpid(),c.pid]))
time.sleep(.1)
"""
    with (tmp_path/'orphan.log').open('w') as log:
        with pytest.raises(RuntimeError,match='leaving descendants'):
            guard.run([sys.executable,'-c',script,str(tmp_path/'pids')],cwd=tmp_path,env=dict(os.environ),stdout=log,timeout=5)
    assert all(guard.identity(pid) is None for pid in json.loads((tmp_path/'pids').read_text()))
    assert not guard.descendants()


def test_sigterm_supervisor_cleans_and_keeps_quarantine(tmp_path):
    import signal,time
    driver=tmp_path/'driver.py'
    nested="""import subprocess,sys,time,json,os
from pathlib import Path
c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
Path(sys.argv[1]).write_text(json.dumps([os.getpid(),c.pid]))
time.sleep(30)
"""
    driver.write_text('import os,sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(Path(guard.__file__).parent))+')\nimport process_guard as g\n'
        +'g.prepare();g.install_signal_handlers();g.begin_quarantine(sys.argv[1])\n'
        +'try:\n with open(sys.argv[2],"w") as f:g.run([sys.executable,"-c",'+repr(nested)+',sys.argv[3]],cwd=None,env=dict(os.environ),stdout=f)\n'
        +'finally:g.finish_quarantine()\n')
    p=subprocess.Popen([sys.executable,str(driver),str(tmp_path/'blocked'),str(tmp_path/'log'),str(tmp_path/'pids')],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    deadline=time.monotonic()+5
    while not (tmp_path/'pids').exists():
        assert p.poll() is None and time.monotonic()<deadline
        time.sleep(.02)
    p.send_signal(signal.SIGTERM);p.wait(timeout=10)
    assert p.returncode!=0
    assert all(guard.identity(pid) is None for pid in json.loads((tmp_path/'pids').read_text()))
    value=json.loads((tmp_path/'blocked').read_text())
    assert value['status']=='BLOCKED_OPERATOR_RECONCILIATION_REQUIRED'


def test_quarantine_clean_owner_clear_and_foreign_refusal(tmp_path):
    import pytest
    marker=tmp_path/'blocked'
    guard.prepare();guard.begin_quarantine(marker)
    with pytest.raises(RuntimeError,match='quarantined'):guard.check_quarantine(marker)
    guard.finish_quarantine();assert not marker.exists()
    guard.begin_quarantine(marker)
    marker.write_text(json.dumps({'owner_token':'foreign-owner'}))
    guard.finish_quarantine();assert marker.exists()
