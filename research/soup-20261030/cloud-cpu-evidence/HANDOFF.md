# Soup cloud CPU testing environment

Cloud-only path: `/redacted/local/soup_cloud_env`. No user-PC execution, global installs, drivers, paid compute, model downloads, dataset downloads, training, pushes, or Kaggle submission.

## Scope

Official upstream: https://github.com/MakazhanAlpamys/soup-kernel-challenge
Pinned source commit: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb` (detached HEAD).
Source, solution, harness, task, tests and official notebook remain unchanged.

This container has Python 3.12.14 and no exposed NVIDIA GPU/device or nvidia-smi. CPU torch is deliberately used. This is a regression/import environment, **not** the official CUDA training environment and not valid for challenge timing/scoring.

## Repeat offline checks

```bash
cd /redacted/local/soup_cloud_env
./run_cpu_checks.sh
```

The custom `verify_cpu.py` loads the unchanged official config and uses a tiny randomly initialized Qwen2 MLP (16 hidden units) to check that starter `apply()` preserves CPU forward/backward fallback. It does not load any pretrained model. Official `tests/` are run twice, with and without `SOUP_APPLY_SOLUTION=1`. Official GPU kernel tests must skip on this machine.

## Rebuild CPU environment

```bash
python3.12 -m venv venv
venv/bin/python -m pip install 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
venv/bin/python -m pip install -r requirements-cpu.lock.txt
```

The lock includes torch's CPU version, already installed by the previous command. Package artifact URLs and hashes are retained in `logs/*install.json`. `logs/source-sha256.txt` contains upstream tracked-file digests. `logs/inventory.txt` records pre-install host inventory.

## GPU/Kaggle handoff

`requirements-official-gpu.txt` preserves the official six dependency pins. `source/templates/kaggle_notebook.ipynb` remains the untouched official notebook. Create a separate GPU environment; do not install this CPU lock into Kaggle. No peft/trl/bitsandbytes/accelerate training stack is installed locally because it is unnecessary for the CPU regression suite.

Only actual Kaggle T4 measurements count. Before any Kaggle launch, parent must coordinate user authentication, quota/budget, and explicit bounded run scope. No GPU smoke, 64-step training, baseline-vs-solution comparison, quality check, or scoring has been executed here. The shipped SwiGLU example is not an original competition kernel; no eligibility or performance claim is made.

## Verified results

- Official baseline regression suite: **79 passed** (`logs/regression-base-final.txt`).
- Official regression suite after starter `apply()`: **79 passed** (`logs/regression-applied.txt`).
- Official CUDA kernel suite: **4 skipped**, not passed (`logs/kernel-tests.txt`).
- Official config parsing and tiny CPU MLP forward/backward fallback: **PASS** (`logs/validation.json`).
- `pip check`: **No broken requirements found** (`logs/pip-check.txt`).
- Git tracked and untracked status: clean (`logs/source-status.txt`); every tracked source SHA-256 check passed (`logs/source-integrity.txt`).
- Isolated venv size: approximately 2.5 GB; source checkout under 1 MB.
- CPU lock: 52 resolved packages; no pretrained weights or training dataset downloaded.

The initial minimal run's missing-pandas failure is retained in `logs/regression-base.txt` for traceability; the final re-run above supersedes it. The broader training-stack dry-run was canceled to avoid unnecessary downloads, and none of that training stack was installed; its partial planning log is not a verification result.
