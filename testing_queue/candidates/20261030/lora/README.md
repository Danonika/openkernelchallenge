# LoRA forward epilogue — prepared research candidate, T4 UNTESTED

**Candidate ID:** `lora-fwd-half-scale-add-20261030`  
**Mechanism:** fuse only the native adapter-output scale and base-result add into one original Triton pass. Retain base NF4, defensive clone, native q/v dropout calls, A/B GEMMs, adapter/optimizer state, task, and all outputs.

## What is implemented

`solution/epilogue.py` contains the original `lora_half_scale_add` kernel. It reads the already-produced FP16 adapter B output, multiplies by fixed LoRA scale 2, **rounds to FP16 before adding**, then writes fresh output. Base output may be FP16 or FP32. Fusion removes the materialized scaled-adapter tensor and its read, not the native B output or defensive clone. This is a useful-work ablation, not a full LoRA/GEMM rewrite.

Autograd preserves native adjoint order: the add's incoming gradient is converted to FP16 for the delta edge **before** multiplication by 2. The base edge receives the incoming gradient converted to the original base dtype. Backward uses native differentiable tensor operations, preserving higher-order differentiation in tested fixtures. No native dropout node or GEMM backward is replaced. Native q and v calls retain separate random draws; no seed/mask substitution, reordering, replay, or extra RNG consumption occurs.

`solution/patch.py` inspects installed PEFT 0.20.0 `bnb.py` and requires its entire source SHA256 `96ed6fda63d7c67a4646faea8ba38483e8be8670059a3ac3dd740a99bdab6950`. It rewrites exactly one vanilla scale/add pair in an in-memory AST. Vendor source is not copied into this package, and the vendor forward is **not** claimed as original kernel work. All native branches stay native. Non-AMP forward takes the wholly original method; the transformed path only executes inside actual CUDA FP16 model-forward autocast, avoiding the Round 7 outer-trainer AMP-boundary error.

Guards require exact PEFT `Linear4bit`, ordinary one-adapter rank16/alpha32/scale2, q/v-compatible matrix shapes, FP32 trainable adapter weights, native non-inplace dropout p=.05, training and grad mode, no merge/disable/variant/extra arguments. The kernel additionally requires SM75, matching contiguous nonempty 2D/3D tensors, FP16 delta and FP16/FP32 base. Unsupported epilogues evaluate the original two native operations; unsupported forwards remain wholly native. Unsupported source versions and conflicting wrappers fail closed before patching. Native input/layout/cast/dropout behavior is not broadened.

### Clone is deliberately retained

Fresh output alone does not establish that PEFT's defensive NF4 clone can be removed safely. The pinned source actually uses **out-of-place addition after a defensive clone**, contrary to the generic clone-plus-in-place description in older research summaries. This candidate retains that clone exactly. No clone-elision memory benefit is claimed; actual NF4 view/autograd behavior has not been executed here.

## Evidence and limitations

Executed: thirteen tests: six focused CPU numerical/contract tests, two synthetic fail-closed runner-gate tests, and five CPU process/lock cleanup tests using available **Python 3.12.14, Torch 2.15.0+cpu and Triton 3.9.0**, not the target pins. Triton interpreter tests cover half/float bases, tail sizes, finite values, NaN/Inf/signed-zero patterns, mandatory half-overflow rounding, scaled incoming gradients (including 65536), fresh output/no input mutation, native fallback, second derivatives, AST rejection, and separate native q/v dropout draws in a synthetic two-branch fixture. Outputs/input/adapter gradients and RNG match in those fixtures. See `cpu-tests.log` and `validation.json` for exact versions and outcomes. Expected overflow/invalid warnings are caused by the explicit nonfinite fixtures.

**Not executed:** pinned PEFT integration, CUDA or NF4, SM75 compilation or launch, full Qwen/Soup training, public quality, external peak measurement, speed or official regression/rule checks. The recovered Round 7 research is a contract reference only; none of its source or CPU/T4 results is claimed as this candidate's validation. The available CPU stack differs from target. No prior failed prototype is relabeled validated.

**Own-kernel useful share >=5% is unknown and may well fail for this narrow epilogue alone.** Do not add work or slow the kernel to make that threshold. The notebook reports actual steps-3-onward event share and preserves an official share-gate failure. Source/launch count is not share evidence. No speedup, peak-memory gain, score or admission PASS is asserted.

`SOUP_LORA_EPILOGUE` first-install/first-launch-attempt/first-launch-submitted markers are printed flushed to the official train log, and exit counters are printed when possible. These survive ordinary native timeout capture; `launch_submitted` is explicitly host submission, not synchronized GPU completion. Complete runs use unchanged harness CUDA-event activation/share records. A killed run never gets a score or quality/pass result.

## Notebook and offline target run

`candidate.ipynb` is output-cleared and embeds these exact source/runner bytes. It creates a unique research directory and starts `run_prepared.py` in a fresh Python process. It does not install packages, download assets, deploy anything, or submit to a queue. A prepared authorized T4 environment must already contain:

- Official clean checkout `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb` (default `/kaggle/working/soup-kernel-challenge`, or `SOUP_OFFICIAL_REPO`).
- Exact existing target packages/Python from `environment-reference.json`: Python 3.13.15, Torch 2.11.0+cu128, Triton 3.6.0, soup-cli 0.77.0, Transformers 5.16.1, PEFT 0.20.0, TRL 0.29.1, bitsandbytes 0.50.2, Accelerate 1.14.0, datasets 4.8.5.
- Prepared public data with fixed hashes and dedicated already-downloaded public model cache (`SOUP_HF_HUB_CACHE` or `HF_HUB_CACHE`), pinned revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Missing/mismatched assets stop; no silent download or model substitution.
- One visible Tesla T4, idle physical GPUs and four available logical CPUs. Host/driver/CPU differences are recorded, not declared Kaggle-equivalent.

The notebook execution opt-in `AUTHORIZE_GPU_EXECUTION=False` is disabled by default; enable only with current run authorization. Set `SOUP_BACKEND=GCP` for the GCP environment (default) or `KAGGLE` for the genuine Kaggle environment; this label confers no admission. Optional `SOUP_RUN_ROOT` chooses the output root. From a prepared environment, an equivalent command is:

```bash
python run_prepared.py --repo /prepared/official-repo --cache /prepared/public-hf-cache \
  --out-root /fresh/research-runs --backend GCP
```

The runner holds the shared nonblocking `/tmp/soup-gpu0.lock` throughout the paired run/checks. It checks `/tmp/soup-gpu0.lock.blocked` before and after acquisition. An owner-token marker exists before GPU work so an abruptly killed supervisor cannot silently free the lane for another notebook. Only its own marker clears on clean non-interrupted exit with no descendants; outer interruption/timeout, orphaned-child detection or uncertain cleanup retains quarantine for explicit operator reconciliation. A dedicated Linux subreaper uses PID descriptors to freeze/kill/reap owned descendants across the official harness's nested process sessions, retaining the supervisor and lock until exit. Notebook interruption requests supervisor cleanup and waits, rather than killing its process group. CPU tests cover nested-session timeout, normally exiting orphan parent, SIGTERM, and marker ownership. This is crash/race mitigation for trusted research jobs, not a hostile-code sandbox. GPU inventory uses a short-lived subprocess so the runner parent holds no CUDA context that would bias external peaks. It rejects other active compute processes. This local lock does not prevent unrelated work bypassing it or coordinate with a remote queue; the operator must ensure exclusivity.

The runner stages fresh source/data/solution without previous adapters, outputs or compiler caches and calls the **unchanged official** `harness/run_task.py --repeats 2 --seed 1 --timeout 900`. Order is base, solution, solution, base. The unchanged harness starts fresh training processes and fresh Triton/Inductor caches each time; full native command timing and external `nvidia-smi` peak capture remain intact. Offline warmup only reads the existing model cache. The whole harness wall is recorded separately. GCP gets its own same-session baseline; no historical Kaggle 79-second cutoff is transplanted.

Only after all four 64-step native runs, both evaluations, outputs, external peaks and own-kernel activation/timing are complete does the runner invoke unchanged `check_rules.py` including regression tests. Result bytes are retained verbatim; nested failures remain `FAILED_OR_INCOMPLETE` even if an outer wrapper returns zero. `assessment.json` always has `score: null` and `admission: UNKNOWN`. Final official Kaggle admission and hidden-row quality are separate jury gates.

## Review/queue packet

- `candidate.zip`: deterministic reviewed-source bundle containing only `solution/` files.
- `candidate-files.json`: per-source SHA256 from the queue's prescribed bundler.
- `submission.template.json`: compatible schema; `baseline_id` remains an explicit unresolved replacement token. Do **not** submit until the actual same-environment measured baseline is registered and this exact ZIP reviewed.
- `manifest.json`: whole packet file hashes, source ZIP hash and evidence boundaries.
- `tests/`, `validation.json`, `cpu-tests.log`: local verification, outside the worker solution bundle.

No queue submission, remote mutation, GPU job, package installation, Sheet/GitHub write or user-PC action was performed.

## Provenance

Original solution and tests written for this preparation. Research contracts read:

- `/workspace/shared/soup_audit_1919/lora.json` (fresh LoRA audit).
- Recovered Round 7 `round7_l1e/README.md` and `batch2/LORA_RESEARCH_ROUND_7.md` (native separate dropout, casts, AMP boundary, negative evidence; no code carried over).
- [PEFT 0.20.0 NF4 LoRA wrapper](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py) (pinned source dispatch/order).
- Local original starter `harness/`, `task/`, `tests/` and queue bundler/runner reference. None modified.
