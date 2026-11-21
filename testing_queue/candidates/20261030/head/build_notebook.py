"""Create a cleared self-contained notebook. Packaging only, no GPU execution."""
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
            "execution_count": None, "outputs": []}


files = ["candidate.zip", "candidate-files.json", "bundle-info.json", "official-source-manifest.json",
         "run_candidate.py", "process_guard.py", "owned_process_tree.py", "README.md", "submission.template.json"]
embedded = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in files}
cells = [markdown("""# Head-memory: private FP16 state + native CE replay

**Actual prepared source; T4 UNTESTED.** 24/24 focused CPU checks passed, including
a tiny real Qwen2 fixture. This is a native-replay memory control, not full H0.
No original Triton kernels are present: official own-kernel checks 3/4 will fail.
No speed, peak-memory, quality, score or admission claim has been established.

The fixed task/harness are preserved. An authorized run measures a fresh native
baseline and candidate in the same environment, with base/solution/solution/base
order. It captures full command wall time, external nvidia-smi peaks, quality and
immediate native-route telemetry. GCP is not official Kaggle admission.
No package installation or model/data download is performed by this notebook.
"""), code("""from pathlib import Path
import os
import sys

# Set existing prepared paths. Do not point at an output from an earlier run.
PREPARED_REPO = Path(os.environ.get("SOUP_OFFICIAL_REPO", "/absolute/prepared/pinned-official-checkout"))
PUBLIC_HF_CACHE = Path(os.environ.get("SOUP_PUBLIC_HF_CACHE", "/absolute/prepared/public-hf-hub-cache"))
OUTPUT = Path(os.environ.get("SOUP_EXPERIMENT_OUT", "/absolute/new/head-experiment-01"))
BACKEND = "GCP"  # or KAGGLE; neither reuses historical baseline timings
NATIVE_TIMEOUT_S = 900  # fixed before measuring both sides, NOT the Kaggle79s reference
OUTER_TIMEOUT_S = 8100
AUTHORIZE_GPU_EXECUTION = False  # set True only after separate current run approval
"""), markdown("""## Frozen reviewed source

The next cells materialize only this packet into a fresh temporary directory;
they do not run training. ZIP SHA256 and every solution member are verified.
The driver will separately verify pinned official source, exact direct packages,
Python3.13.15, cached public data/model revision, idle physical GPU memory and one
visible Tesla T4. Missing prerequisites stop; they do not trigger downloads.
"""), code("EMBEDDED_FILES = " + repr(embedded) + "\n"), code("""import base64
import hashlib
import json
import tempfile
import zipfile

PACKET = Path(tempfile.mkdtemp(prefix="soup-head-reviewed-"))
for name, payload in EMBEDDED_FILES.items():
    (PACKET / name).write_bytes(base64.b64decode(payload))
binding = json.loads((PACKET / "bundle-info.json").read_text())
manifest = json.loads((PACKET / "candidate-files.json").read_text())
assert hashlib.sha256((PACKET / "candidate.zip").read_bytes()).hexdigest() == binding["source_sha256"]
with zipfile.ZipFile(PACKET / "candidate.zip") as z:
    assert set(z.namelist()) == set(manifest)
    for name, expected in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == expected
print("Prepared source:", binding["source_sha256"])
print("Status: T4 UNTESTED; native-replay control; own Triton share = 0; admission BLOCKED")
print("Packet:", PACKET)
"""), markdown("""## Authorized fixed-task comparison

Executing the following cell requires explicit authorization and all preflight
guards. It stages a **new** repository, leaves the prepared checkout untouched,
and invokes the unchanged official harness. Every measured run gets fresh
training state, output and compiler caches. Model/data assets are reused offline.
Compilation and native-CE recomputation count in complete time. Interrupted,
timed-out, incomplete or failed training/evaluation never becomes a score/PASS.
"""), code("""import signal
import subprocess

if not AUTHORIZE_GPU_EXECUTION:
    raise RuntimeError("Prepared only: GPU execution is disabled. Obtain run approval and configure paths first.")
command = [sys.executable, str(PACKET / "run_candidate.py"),
           "--repo", str(PREPARED_REPO), "--cache", str(PUBLIC_HF_CACHE),
           "--out", str(OUTPUT), "--backend", BACKEND,
           "--timeout", str(NATIVE_TIMEOUT_S), "--outer-timeout", str(OUTER_TIMEOUT_S),
           "--execute"]
driver = subprocess.Popen(command, start_new_session=True)
try:
    returncode = driver.wait()
except KeyboardInterrupt:
    # Ask the driver to clean up its owned nested process tree and persist a
    # partial report. Do not abandon an executing GPU child on notebook stop.
    driver.send_signal(signal.SIGTERM)
    while True:
        try:
            returncode = driver.wait()
            break
        except KeyboardInterrupt:
            print("Waiting for owned-process cleanup; GPU lock remains held.")
print("Driver exit:", returncode)
report_path = OUTPUT / "research-result.json"
if report_path.exists():
    report = json.loads(report_path.read_text())
    print(json.dumps(report, indent=2))
    assert report["score"] is None
    assert report["admission"] != "PASS"
else:
    print("No complete driver report. Inspect preflight/driver error; do not infer success.")
"""), markdown("""## Interpretation and queue handoff

Read `research-result.json` together with all raw `measured/` results, logs,
official checks and source hashes. Native-replay activation requires 64 backward
completions in each candidate run; absence is not an optimization success.
Saved-state reduction does not prove lower external peak: replay can increase
backward peak and runtime. This packet does not suppress output logits or replace
native entropy, accuracy, Rewind, LoRA, dropout or RNG.

The worker consumes the exact reviewed `candidate.zip`, **not notebook cells**.
`submission.template.json` deliberately leaves the measured GCP baseline ID
unresolved. Do not submit until the reviewer authorizes the specific research
control and supplies the real registered cohort. Full H0 summary-only finish is
blocked pending an original safe consumer bridge and target numerical evidence;
recovered foreign kernels are not copied or credited as original work.
""")]
notebook = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "candidate_status": "T4_UNTESTED_NATIVE_CONTROL_ADMISSION_BLOCKED"},
            "cells": cells}
for index, cell in enumerate(cells):
    cell["id"] = "head-" + str(index)
(ROOT / "candidate.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
print(hashlib.sha256((ROOT / "candidate.ipynb").read_bytes()).hexdigest())
