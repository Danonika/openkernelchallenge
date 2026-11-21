#!/usr/bin/env python3
"""Assemble a deterministic cleared research notebook; does not execute it."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
payload={str(p.relative_to(root)):p.read_text() for p in [*sorted((root/'solution').glob('*.py')),root/'run_prepared.py',root/'process_guard.py']}
hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in payload.items()}
def md(s):return {'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)}
def code(s):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(keepends=True)}
cells=[md('''# Original LoRA scale/add epilogue — T4 untested research candidate

Fuses the half-rounded adapter scale and base add only. Native separate q/v dropout,
A/B GEMMs, FP16 boundaries, backward and the defensive PEFT clone remain intact.
Six local CPU-interpreter fixtures are **not** pinned-stack PEFT/CUDA/NF4 validation.
Own-kernel useful share >=5%, performance, peak and quality remain unmeasured.

This notebook runs only in an already-prepared authorized T4 environment. It does not
install/download, change the task/harness/tests, submit to a queue, or manufacture a baseline ID.
Missing source/packages/public assets fail closed. Reuse prepared public model/data only;
all measured processes, adapter state, outputs and compiler caches are fresh.
'''),code('''import os, sys, pathlib, uuid, hashlib, json, subprocess
# Explicit opt-in is required for a future authorized cloud GPU run.
AUTHORIZE_GPU_EXECUTION = False
# Set environment variables before Run All if prepared paths differ; no downloads occur.
OFFICIAL_REPO = pathlib.Path(os.environ.get('SOUP_OFFICIAL_REPO', '/kaggle/working/soup-kernel-challenge'))
HF_CACHE_TEXT = os.environ.get('SOUP_HF_HUB_CACHE') or os.environ.get('HF_HUB_CACHE')
if not HF_CACHE_TEXT:
    raise RuntimeError('Set SOUP_HF_HUB_CACHE to the already-prepared dedicated public Hugging Face hub cache')
HF_CACHE = pathlib.Path(HF_CACHE_TEXT)
RUN_ROOT = pathlib.Path(os.environ.get('SOUP_RUN_ROOT', '/kaggle/working/soup-lora-research'))
BACKEND = os.environ.get('SOUP_BACKEND', 'GCP')
assert BACKEND in ('GCP', 'KAGGLE')
RUN_ROOT.mkdir(parents=True, exist_ok=True)
PACKAGE = RUN_ROOT / ('prepared-lora-' + uuid.uuid4().hex[:12])
PACKAGE.mkdir()
print('Prepared package:', PACKAGE)
'''),code('PAYLOAD = '+repr(payload)+'\nEXPECTED = '+repr(hashes)+'''
for name, source in PAYLOAD.items():
    assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED[name], name
    destination = PACKAGE / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)
print('Embedded source hashes verified:', len(PAYLOAD))
'''),md('''## Unchanged official paired run

Uses **base, solution, solution, base**, seed 1, native timeout 900 s per run.
This records an independent baseline in this same environment: GCP is not Kaggle.
Full native command wall, external nvidia-smi peak, quality and useful own-kernel
share (steps 3 onward) are read from unchanged official evidence. A separate outer
harness duration is recorded. No historical 79 s limit is transplanted.

The runner retains logs and raw results, rejects partial/failed native runs even if
wrapper exit is zero, and only then invokes unchanged official checks/regressions.
`score` remains null and `admission` UNKNOWN. Hidden-quality/final Kaggle admission
are separate. Flushed kernel launch-attempt/submitted markers survive train-log timeouts;
submitted is **not** synchronized GPU completion.
'''),code('''cmd = [sys.executable, str(PACKAGE / 'run_prepared.py'),
       '--repo', str(OFFICIAL_REPO), '--cache', str(HF_CACHE),
       '--out-root', str(RUN_ROOT), '--backend', BACKEND]
# The native harness supplies finite per-process timeouts and cleanup. Do not run
# concurrent jobs on this GPU or interrupt and immediately start another job.
import signal
if not AUTHORIZE_GPU_EXECUTION:
    raise RuntimeError('Prepared only: configure existing cloud paths and explicitly authorize the GPU run first')
proc = subprocess.Popen(cmd, start_new_session=True)
try:
    result = proc.wait()
except KeyboardInterrupt:
    # Ask the supervisor to clean up; do not kill it or release its GPU lock.
    proc.send_signal(signal.SIGTERM)
    while True:
        try:
            proc.wait()
            break
        except KeyboardInterrupt:
            continue
    raise
if result:
    raise subprocess.CalledProcessError(result, cmd)
'''),md('''## Evidence interpretation

Read the emitted evidence directory's `assessment.json`, `harness_wall.json`,
`runs/result.json`, `runs/checks.json`, train logs, source/environment/asset hashes.
A share failure is an actual limitation of this narrow ablation; never pad work to meet it.
The worker accepts reviewed `candidate.zip` solution source, not this notebook.
Do not submit the template until a real measured backend-specific baseline ID is known.
''')]
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'soup_candidate':{'id':'lora-fwd-half-scale-add-20261030','status':'PREPARED_T4_UNTESTED','official_commit':'c7c2fcd15972ce8b898d8d3ff732ee192c817ceb'}},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells):c['id']=f'lora-{i:03d}'
(root/'candidate.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
