"""Regenerate cleared-output standalone notebook from reviewed local sources."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OFFICIAL = Path('/workspace/shared/soup_cloud_env/source')
payload = {str(p.relative_to(ROOT)): p.read_text() for folder in ('solution', 'support')
           for p in sorted((ROOT/folder).rglob('*.py'))}
frozen = {str(p.relative_to(OFFICIAL)): hashlib.sha256(p.read_bytes()).hexdigest()
          for folder in ('harness', 'task', 'tests') for p in sorted((OFFICIAL/folder).rglob('*'))
          if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
(ROOT/'official_manifest.json').write_text(json.dumps(frozen,indent=2)+'\n')
cells=[]
def md(text): cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
def code(text): cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':text.splitlines(keepends=True)})
md('''# NF4 blockpair v1 — prepared research candidate, T4 untested

Original byte-pair decode + block-scale broadcast + aligned packed FP16 stores. Both actual
bitsandbytes0.50.2 decode boundaries; native tiny-M routing, GEMM, casts, and autograd retained.
This is **not** unchanged trial002/004. Historical standalone share failures remain failures.

No GPU run, installation, download, queue submission, or external write was performed in preparation.
This notebook embeds all candidate/driver source. It uses only a **prepared local official checkout,
public data, exact installed packages and public model cache**. It never installs or downloads.

After separate run authorization, supply the three paths below and enable the gate. On Run All it
creates a new staging directory and calls the untouched official paired harness in order
base→solution→solution→base. Every measured subprocess has fresh state/output/compiler caches.
Whole-command time, nvidia-smi peak, public evaluation, exact checks and route receipts are retained.
GCP compares its own paired baseline, not Kaggle's historical79-second screen. Score stays null;
public automatic checks cannot establish hidden quality or final Kaggle/jury admission.
''')
code('''import os
CONFIG = {
    "authorized": False,  # Set True only after current explicit GPU execution authorization.
    "backend": "GCP",   # Or KAGGLE_T4. Never transplant a baseline between environments.
    "prepared_repo": os.environ.get("SOUP_PREPARED_REPO", "/REPLACE/WITH/PREPARED/OFFICIAL/CHECKOUT"),
    "hf_hub_cache": os.environ.get("HF_HUB_CACHE", "/REPLACE/WITH/PREPARED/PUBLIC/MODEL/CACHE"),
    "output_parent": os.environ.get("SOUP_OUTPUT_ROOT", "/kaggle/working/nf4-blockpair-runs"),
}
''')
md('''## Frozen source and runner
The following cells are generated, not a network bootstrap. Compare their embedded source hashes
to source_manifest.json and candidate.zip before enabling execution. The worker accepts the ZIP,
not this notebook. The submission template intentionally has no measured baseline ID.
''')
code('PAYLOAD = '+repr(payload)+'\nFROZEN_MANIFEST = '+repr(frozen)+'\n')
code((ROOT/'notebook_driver.py').read_text())
code('RUN_DIRECTORY = run(CONFIG, PAYLOAD, FROZEN_MANIFEST)\nprint("Evidence directory:", RUN_DIRECTORY)\n')
md('''## Evidence boundaries
CPU interpreter:48 fixtures passed; local Torch2.15 CPU/Triton3.9 is not target Torch2.11/Triton3.6.
Four offline SM75 compilations succeeded with zero shared bytes, separate FP32 multiply/add and
packed32-bit stores where requested. No GPU code was executed and no compiled binaries are embedded.
T4 numerical parity, actual route coverage, full task quality, speed, peak and≥5% own share are unknown.
Never sum historical shares or slow/add work to meet the share threshold. Inspect failed/partial logs;
they receive neither scores nor admission. Raw measured own-share comes only from official events
following two warmup steps. Flushed route logs are activation receipts, not timing evidence.
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
          'language_info':{'name':'python'},'candidate_id':'nf4_blockpair_v1','t4_tested':False},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells): c['id']='nf4-cell-'+str(i)
(ROOT/'candidate.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
