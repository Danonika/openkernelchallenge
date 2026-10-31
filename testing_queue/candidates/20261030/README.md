# Six experimental Soup candidate notebooks

Prepared for one visible Tesla T4, using the fixed Soup Kernel Challenge task. **No candidate in this package has been validated on the target T4 during this preparation.** CPU/interpreter/static checks are not performance or contest admission evidence.

## Contents

| Folder | Experiment | Important limit |
|---|---|---|
| ce | Materialized cross-entropy v3 routing and timeout-surviving activation evidence | New v3 full-task routing, quality, timing and own-kernel share untested |
| head | Native half-logit head/loss replay memory control | No original Triton kernel; not an admissible standalone contest solution |
| memory | Saved-activation restoration with zero-storage version guard | Distinct safety revision; memory/time benefit unmeasured; tied-weight alternative remains blocked |
| nf4 | Packed NF4 decoding, block-scale reuse and paired stores | New decode variant; historical failures do not establish its result |
| lora | Original LoRA scale/add epilogue with native dropout/GEMMs | Defensive clone retained; own-share threshold unproven |
| rotary | Original half-split RoPE application and native-style adjoint | Target compiler/GPU floating-point behavior still requires validation |

Each folder contains `candidate.ipynb`, `solution/`, matching `candidate.zip`, source hashes, a submission template, a README and validation evidence. Read the individual README before running.

## Run on an already prepared VM or Kaggle session

1. Prepare the exact Python/package environment and unchanged official source at commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb` using the queue installation guide. These research notebooks intentionally do not install dependencies or download models on every run.
2. Keep the fixed public data and model revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` cached. Fill the notebook's local checkout/cache/output settings. These are research notebooks, not the final public competition submission notebook.
3. Run **one notebook at a time**. The wrappers use `/tmp/soup-gpu0.lock`, but this does not coordinate arbitrary external jobs or a separately configured queue worker. Do not run manual notebooks alongside that worker.
4. The paired measurements use the same machine/session and unchanged official harness. GCP needs its own measured baseline; the earlier Kaggle 79-second engineering cutoff is not a universal GCP timeout.
5. Keep complete raw evidence, including failures. An interrupted run, partial peak or compilation-only check is not a completed measurement. No automatic earned contest score is asserted.

## Submit to the web queue

Upload each **solution `candidate.zip`**, not the outer six-candidate delivery ZIP. The current GPU worker executes reviewed solution bundles, not arbitrary notebook cells. Fill the submission template with the real registered GCP baseline ID. Reviewer approval and the worker's exact source allowlist are required before execution. Do not put API tokens in notebooks, source ZIPs or URLs.

No job was submitted, no GPU was started, and no VM was deployed by preparing this package. The final competition still requires the official clean Kaggle run, all admission checks, a qualifying original Triton share, README and submission artifacts.
