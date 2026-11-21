"""Re-embed frozen local source/runner after review; no notebook cells execute."""
import json
from pathlib import Path
root=Path(__file__).parent
nb=json.loads((root/'candidate.ipynb').read_text())
sources={'solution/'+p.name:p.read_text() for p in sorted((root/'solution').glob('*.py'))}
hashes=json.loads((root/'source_manifest.json').read_text());protected=json.loads((root/'official_manifest.json').read_text());runner=(root/'notebook_runner.py').read_text()
nb['cells'][3]['source']=('import json\nSOURCES = '+repr(sources)+'\nSOURCE_HASHES = '+repr(hashes)+'\nPROTECTED_HASHES = '+repr(protected)+'\nRUNNER_SOURCE = '+repr(runner)+'\n').splitlines(keepends=True)
launch='''# The dedicated supervisor, not the notebook, owns CUDA subprocesses/cleanup.
# Default False fails before process launch or GPU use.
assert AUTHORIZE_GPU is True, "Prepared only: explicit target-run authorization required"
import os, signal, subprocess, sys, tempfile
from pathlib import Path
assert hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "Linux pidfd support required"
Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
launch_dir = Path(tempfile.mkdtemp(prefix="memory_vs1_launcher_", dir=OUTPUT_ROOT))
(launch_dir / "runner.py").write_text(RUNNER_SOURCE)
settings = dict(authorize_gpu=True, repo=OFFICIAL_REPO, output_root=OUTPUT_ROOT,
                hf_hub_cache=HF_HUB_CACHE, sources=SOURCES, hashes=SOURCE_HASHES,
                protected_hashes=PROTECTED_HASHES, timeout_s=900)
(launch_dir / "config.json").write_text(json.dumps(settings))
proc = subprocess.Popen([sys.executable, str(launch_dir / "runner.py"),
                         "--config", str(launch_dir / "config.json")], start_new_session=True)
pidfd = os.pidfd_open(proc.pid, 0)
try:
    try:
        code = proc.wait()
    except BaseException:
        Path("/tmp/soup-gpu0.lock.blocked").write_text(json.dumps({
            "reason": "notebook launcher interrupted", "supervisor_pid": proc.pid,
            "manual_owner_reconciliation_required": True}))
        try: signal.pidfd_send_signal(pidfd, signal.SIGTERM, None, 0)
        except ProcessLookupError: pass
        # Keep waiting for owned cleanup; another Ctrl-C must not release it.
        while proc.poll() is None:
            try: proc.wait(timeout=1)
            except (subprocess.TimeoutExpired, KeyboardInterrupt): pass
        raise
    if code < 0:
        Path("/tmp/soup-gpu0.lock.blocked").write_text(json.dumps({
            "reason": "supervisor terminated by signal", "exit_code": code,
            "manual_owner_reconciliation_required": True}))
    assert code == 0, "Supervisor failed; preserve all logs and quarantine; no score/admission"
finally:
    os.close(pidfd)
print("Evidence is under", OUTPUT_ROOT, "— inspect the new memory_vs1_* directory")
'''
nb['cells'][4]['source']=launch.splitlines(keepends=True)
(root/'candidate.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
