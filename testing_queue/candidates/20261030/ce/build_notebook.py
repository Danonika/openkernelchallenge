"""Build a cleared self-contained research notebook from this exact packet."""
import ast
import base64
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
raw=(ROOT/'candidate.zip').read_bytes()
manifest=json.loads((ROOT/'source_manifest.json').read_text())
official=json.loads((ROOT/'official_manifest.json').read_text())
runner=(ROOT/'notebook_runner.py').read_text()
config={"execute_gpu":False,"backend":"GCP","official_repo":"/prepared/soup-kernel-challenge",
        "data_dir":"/prepared/public-data","hf_hub_cache":"/prepared/hf_hub",
        "work_root":"/output/ce-v3-research"}
(ROOT/'runtime_config.template.json').write_text(json.dumps(config,indent=2)+'\n')
intro='''# Materialized CE v3 — timeout-visible routing

**Prepared research candidate; new v3 has not run on a GPU. No performance, quality, 5% useful-kernel-share, score, or admission is established.**

This notebook contains the exact reviewed `candidate.zip` and source hashes. It reuses an **existing** official checkout at commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`, prepared public data/model cache, and exact installed packages. It never installs, downloads, submits, deploys, or edits the original checkout. Only its fresh staging copy's `solution/` is replaced.

V3 retains the five original numerical kernel bodies and native SUM / supplied-denominator / GradScaler order. It removes one duplicated denominator host read, uses the captured native loss for internal fallback, and flushes compact route evidence before the first JIT and at loss/forward/backward boundaries. A backward enqueue is **not** proof of completed CUDA work. Official CUDA-event timing after two warmup steps remains the only share evidence.

## Run later in an authorized prepared cloud runtime

Edit the paths below, choose `GCP` or `KAGGLE`, and set `execute_gpu=True` only when actual GPU testing is authorized. A nonblocking canonical `/tmp/soup-gpu0.lock` covers the entire run; a busy lane fails immediately. The shared `/tmp/soup-gpu0.lock.blocked` quarantine is checked before and after lock acquisition; an owner-tagged guard is durably created before the first GPU operation and survives abrupt runner death. Only the same owner clears it after normal verified success; any failure/timeout/interruption retains it for explicit operator reconciliation. GPU inventory runs only in a short-lived subprocess, which exits before the idle check. Run All also fails closed on missing assets, version drift, non-idle GPUs, or anything other than one visible Tesla T4. GCP always measures its own baseline; it never imports Kaggle's historical 79-second cutoff.

The public model is fixed at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; `refs/main` drift is rejected. An actual-class v3 integration gate precedes the unchanged official repeat2 harness in base/solution/solution/base order. Each measured native process has fresh state/output/compiler caches. The official harness records complete command wall time, external 100ms nvidia-smi peak, public quality, all Soup outputs, and own-kernel timing. Failed/partial/timeout runs have no normalized comparison or score. All official checks run only after a verified complete pair. Final Kaggle admission remains separate.
'''
cells=[]
def markdown(text): cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
def code(text):
    ast.parse(text)
    cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[], 'source':text.splitlines(keepends=True)})
markdown(intro)
code('''# Paths are operator configuration, not assumptions about an existing session.
# No GPU work occurs unless this explicit execution flag is true.
CONFIG = '''+repr(config)+'''\nprint("Prepared candidate; execute_gpu =", CONFIG["execute_gpu"])\n''')
markdown('''## Exact embedded artifact and runner
The ZIP is checked against its SHA256 and every source-file hash before staging. No notebook-dependent optimization behavior is used: the worker receives exactly this `solution/` bundle.''')
code('''import base64, hashlib, json
BUNDLE_B64 = '''+repr(base64.b64encode(raw).decode())+'''
BUNDLE_SHA256 = '''+repr(hashlib.sha256(raw).hexdigest())+'''
SOURCE_MANIFEST = '''+repr(manifest)+'''
OFFICIAL_MANIFEST = '''+repr(official)+'''
RUNNER_SOURCE = '''+repr(runner)+'''
BUNDLE_BYTES = base64.b64decode(BUNDLE_B64, validate=True)
assert hashlib.sha256(BUNDLE_BYTES).hexdigest() == BUNDLE_SHA256
namespace = {"__name__": "prepared_ce_notebook_runner"}
exec(compile(RUNNER_SOURCE, "notebook_runner.py", "exec"), namespace)
namespace["validate_zip"](BUNDLE_BYTES, SOURCE_MANIFEST)
print("Bundle integrity verified:", BUNDLE_SHA256)
''')
code('''# Future GPU execution: no package installation, asset download, queue submission,
# source/harness modification, hidden rows, or historical baseline substitution.
REPORT = namespace["run"](CONFIG, BUNDLE_BYTES, SOURCE_MANIFEST, OFFICIAL_MANIFEST)
print(json.dumps(REPORT, indent=2, allow_nan=False))
''')
markdown('''## Interpret evidence conservatively
- `receipt.json`: actual environment/source/data/cache bindings, native-pair completion, route records, and raw result/check paths. `score` is always null and automatic admission UNKNOWN.
- `runs/result.json`: unchanged official raw timing/memory/quality evidence. A timeout's wall/peak are censored, not a completed comparison.
- `runs/solution_1/train.log`: immediately flushed `SOUP_CE_V3_EVENT` records survive native process termination; an absent record is UNKNOWN. Core-native and adapter-native counters can describe the same fallback and must not be added.
- `runs/solution_1/metrics.json`: official own-kernel launch/event timing, when available. The useful-own-kernel requirement is >=5% after two warmup steps; Python counters never satisfy it.
- `runs/checks.json`: unchanged official check results. No leaderboard score is manufactured and a GCP pair cannot establish official Kaggle admission.

The old trial008 all-native routing and trial009 timeout remain failed historical variants. Previous v2 numerical preflight results are not relabeled as a v3 integration or whole-task pass. Instrumentation overhead is included in measurements and remains unknown until measured.
''')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.13.15'},
    'candidate_id':'soup-ce-v3-routing-20261030','gpu_execution_during_preparation':False},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells): c['id']=f'ce-v3-{i:02}'
(ROOT/'candidate.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
