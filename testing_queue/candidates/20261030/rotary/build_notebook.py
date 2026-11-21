"""Materialize the self-contained notebook from reviewed local files (no execution)."""
import base64,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
names=['candidate.zip','source_manifest.json','official_frozen_manifest.json','run_prepared.py',
       'check_kaggle_environment.py','prepare_asset_manifest.py','requirements-kaggle.txt',
       'submission.template.json','README.md']
payload={n:base64.b64encode((root/n).read_bytes()).decode() for n in names}
hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}

def cell(kind,text,n):
    value=dict(cell_type=kind,id=f'rotary-{n}',metadata={},source=text.splitlines(keepends=True))
    if kind=='code':value.update(execution_count=None,outputs=[])
    return value

config='''from pathlib import Path
import os
# Use an EXISTING cloud checkout/cache. No install, download, deployment or queue submission.
PREPARED_REPO = Path("/kaggle/working/soup-kernel-challenge")
PUBLIC_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", "/root/.cache/huggingface/hub"))
RUN_ROOT = Path("/kaggle/working/rotary-paired-runs")
BACKEND = "KAGGLE_T4"  # Change to GCP on the authorized GCP T4; it needs its own baseline.
NATIVE_TIMEOUT_SECONDS = 900  # Declare before the cohort; no transplanted 79 s GCP budget.
RUN_GPU = False  # Explicit opt-in only after source review and cloud resource authorization.
'''
stage='''import base64, hashlib, json, tempfile, zipfile, io
PAYLOAD = '''+repr(payload)+'''\nPAYLOAD_SHA256 = '''+repr(hashes)+'''
STAGED = Path(tempfile.mkdtemp(prefix="rotary-reviewed-candidate-"))
for name, encoded in PAYLOAD.items():
    data = base64.b64decode(encoded, validate=True)
    assert hashlib.sha256(data).hexdigest() == PAYLOAD_SHA256[name], name
    (STAGED / name).write_bytes(data)
manifest = json.loads((STAGED / "source_manifest.json").read_text())
with zipfile.ZipFile(STAGED / "candidate.zip") as archive:
    assert set(archive.namelist()) == set(manifest)
    for name, wanted in manifest.items():
        data = archive.read(name)
        assert hashlib.sha256(data).hexdigest() == wanted, name
        if name.endswith(".py"):
            compile(data, name, "exec")  # syntax only; no import or GPU execution
print("Reviewed bundle:", PAYLOAD_SHA256["candidate.zip"])
print("Staged safely at:", STAGED)
print("GPU status: NOT RUN; score: null; admission: UNKNOWN")
'''
run='''import subprocess, sys, signal
if not RUN_GPU:
    raise RuntimeError("Prepared only. Review README/source, configure existing cloud paths, then explicitly opt in with RUN_GPU=True.")
command = [sys.executable, str(STAGED / "run_prepared.py"),
           "--prepared-repo", str(PREPARED_REPO), "--hub-cache", str(PUBLIC_HUB_CACHE),
           "--run-root", str(RUN_ROOT), "--backend", BACKEND,
           "--timeout", str(NATIVE_TIMEOUT_SECONDS), "--confirm-cloud-gpu-run"]
# Runner holds /tmp/soup-gpu0.lock; environment/preflight use short-lived children.
# Complete raw logs, timeout evidence and normalized result stay under RUN_ROOT.
runner_process = subprocess.Popen(command, start_new_session=True)
try:
    return_code = runner_process.wait()
except BaseException:
    if runner_process.poll() is None:
        runner_process.send_signal(signal.SIGTERM)
    # Do not force-kill the runner: it owns the lock and nested-session cleanup.
    print("Stopping the runner and waiting for its owned child cleanup...")
    runner_process.wait()
    raise
if return_code:
    raise subprocess.CalledProcessError(return_code, command)
'''
cells=[
cell('markdown','''# Original FP32 half-split RoPE — research candidate

**Prepared, GPU untested.** Only Qwen2 RoPE application changes; no QK norm, FP16 early cast, attention backend, model, task or LoRA changes. The exact source bundle below is the queue artifact.

20 CPU Triton-interpreter fixtures passed on Torch 2.15 CPU / Triton 3.9, not target CUDA/Triton 3.6. No speed, peak-memory, useful-share, quality or admission result exists. Useful original-kernel share >=5% is a particularly important unresolved gate.

Run on an explicitly authorized **cloud** single-visible-Tesla-T4 environment only. Reuse existing packages/model/data. Configuring this notebook does not submit to the queue. Final Kaggle admission is separate; GCP needs its own measured baseline.
''',1),
cell('code',config,2),
cell('markdown','''## Verify and materialize the exact packet
This cell only writes a new temporary preparation directory and compiles Python syntax. It does not import the solution, access a GPU, download, install, or alter the prepared official repository.
''',3),
cell('code',stage,4),
cell('markdown','''## Future GPU preflight and unchanged paired official task
The runner requires exact direct package pins/Python, official commit, fixed public data and retained model revision. It creates a fresh local official tree, replaces only `solution/`, and holds the canonical GPU0 lock. GPU preflight is a separate process/cache; it must pass arithmetic, signed-zero/native-adjoint, unchanged SDPA/GQA routing and PTX no-contraction/FTZ checks before training.

Then the unchanged official harness runs base/solution/solution/base (two repeats, fixed seed 1) with fresh training state/output/compiler caches, whole-command wall time, external nvidia-smi peak, public evaluation and native kernel-share telemetry after two warmups. Original regression checks run unchanged. Timeout/partial/failure logs are retained; no score or automatic official admission is assigned. `ROTARY_EVIDENCE` launch-returned lines survive timeouts but are not CUDA-completion proof.

This notebook has never been executed on a GPU. Default execution stops at the explicit opt-in guard below.
''',5),
cell('code',run,6)]
nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.13.15')),nbformat=4,nbformat_minor=5)
(root/'candidate.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
