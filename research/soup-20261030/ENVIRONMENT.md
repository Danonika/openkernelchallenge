# Cloud preparation / official Kaggle execution checklist

Static inspection only by this research agent; installs/tests belong to the separate cloud environment agent. No use of the user's PC. No GPU/training result is claimed here.

## Source and dependencies
Official starter commit: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb` at [archived-source/soup-kernel-challenge](https://example.invalid/redacted-github).

Notebook/README explicitly pin:
```
soup-cli[train]==0.77.0
transformers==5.16.1
peft==0.20.0
trl==0.29.1
bitsandbytes==0.50.2
accelerate==1.14.0
pytest
```
This is a **partial manifest, not a lockfile**. Python, torch, Triton, datasets, pytest and transitive packages are not exact-pinned by notebook. User ТЗ describes Kaggle image Python3.13, torch2.11, Triton3.6.0; capture actual inventory and resolve differences before promotion. Do not install/upgrade during measured runs. Do not install banned optimized libraries as a workaround.

## Separate stages and evidence
1. **Source integrity:** pin source commit; inventory hashes of harness/, tests/, task/ and notebook. Never modify official test/evaluation/task logic.
2. **Cloud static/CPU preparation:** interpreter/version inventory, isolated install/import checks and available CPU tests. Record failures/skips honestly; CPU cannot validate CUDA kernels. No model/data download needed merely to inspect manifest.
3. **Patch-aware regression:** official tests/conftest.py calls solution.apply() only with SOUP_APPLY_SOLUTION=1; unpatched success is insufficient. Official check_rules invokes pytest on five test files (79 tests). Preserve conftest fixtures. Tests/test_trainer_init uses mocks; a patch must gracefully handle them.
4. **Authorized T4 correctness:** `python -m pytest solution -q` checks owned kernels; original starter tests need GPU/Triton. Compiled tail/empty-range, dtype, autograd, overflow and dropout cases are necessary; interpreter-only checks are not sufficient.
5. **Authorized full run:** prepare public data, unchanged run_task, unchanged check_rules, score. No full training is performed by this research agent; the parent coordinates a separately authorized free-tier Kaggle baseline check, initially private.

## Notebook flow inspected
[Official notebook](https://example.invalid/redacted-github):
- REPO_URL and COMMIT identify team repo/final exact commit; empty COMMIT follows default branch, so do not use empty for reproducible research-branch checks.
- Verifies `nvidia-smi` reports T4; choose Kaggle GPU T4 x2 + Internet; harness exposes GPU0 only.
- Installs pinned manifest, freshly clones into /kaggle/working/repo, optionally uses existing read-only GITHUB_TOKEN secret for private repo. Never embed token in URL/output/files.
- `python task/prepare_data.py`
- `python -m pytest solution -q`
- `python harness/run_task.py`
- `python harness/check_rules.py`
- `python harness/score.py runs/result.json`

Notebook default uses one repeat. Jury-style `run_task.py --repeats 2` orders base, solution, solution, base; harness summarizes median full-command seconds and maximum sampled memory. Fresh per-run Triton/Inductor caches include compile/tuning time. Environment fixes allocator to expandable_segments:True; do not override. Externally sampled nvidia-smi includes all GPUs, so the second GPU must remain idle. No background work, prebuilt cache, graph replay or setup hidden outside timing.

## Gates before any claim
- Exact step/config/adapter keys and shapes, log-metric set, token counts and rewind row associations.
- First loss <=1%; mean **per-step relative absolute** loss/entropy/accuracy/rewind errors <=3%; finite, matching-length series.
- Held-out adapter loss <=0.5% worse; same ordinary saved PEFT adapter.
- Whole-command time and peak memory <=10% worse; sampled-vs-reserved consistency and idle GPU checks.
- Useful original solution-defined @triton.jit kernels in every run, >=5% of summed step wall time from step3 onward, weakest run. Starter/near-copy/Inductor/setup-only kernels do not qualify.
- All79 patch-aware Soup regressions, own GPU correctness tests, source/runtime banned-import checks and manual rule review.

Evidence bundle: commit, source hashes, Python/package/CUDA/driver/GPU versions, pip freeze before/after, logs, runs/result.json, runs/checks.json, per-run profiles and saved-adapter eval. Cloud smoke results must be labeled non-scoring; only Kaggle T4 full runs support contest numbers. Account/login/phone verification, paid capacity, publishing notebook, contest submission and main/merge changes require separate user action/authority.

## Completed cloud CPU preparation (separate environment agent)

[Evidence snapshot](cloud-cpu-evidence/HANDOFF.md): 79 baseline +79 starter-apply regressions passed, 4 CUDA tests skipped, config/tiny CPU forward-backward and pip-check passed. Source unchanged. Python3.12.14, torch2.15.0+cpu,52-package CPU lock; **not** Kaggle GPU stack/scoring. No model/data downloads or training. Snapshot is archival; paths/commands in HANDOFF refer to original cloud environment. Initial failure superseded by recorded final runs.
