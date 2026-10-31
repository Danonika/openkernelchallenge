# Official Soup baseline verified on Kaggle T4

**Result:** the official public, base-only one-repeat baseline completed successfully. Downloaded run, evaluation, metrics and checkpoint artifacts agree.

| Measurement | Verified result |
|---|---:|
| Whole training command | **77.92 s** |
| 64-step training loop | 54.04 s |
| Peak GPU memory (nvidia-smi) | **14,345 MiB** |
| Public evaluation loss | **2.04965** |
| Evaluation coverage | 256 rows / 38,109 answer tokens / 0 skipped |
| Starter GPU tests | 4 passed |
| Total baseline session | 214.631 s |

## Evidence checks

- Actual Tesla T4 hardware; one visible device (`CUDA_VISIBLE_DEVICES=0`) despite two physical T4s.
- Train and evaluation exit codes both 0; 64 training steps and saved global step 64.
- Idle GPU start 0 MiB; sampled peak 14,345 MiB exceeds 97% of Torch reserved peak 14,202 MiB.
- Official allocator and package pins; no package changes during measurement or banned runtime imports.
- All 64 loss/token/entropy/accuracy/gradient-norm logs present and finite. Rewind has 64 batches of 4 and exactly public row IDs 0–255 once each; saved adapter metadata contains 112 tensors.
- Saved LoRA config is r=16, alpha=32, dropout=0.05, q_proj/v_proj.

Starter commit: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. Python 3.13.15, Torch 2.11.0+cu128, Triton 3.6.0, Soup 0.77.0. Public data only, official preparation script and pinned dataset revision.


## Scope and limits

This is a **provisional one-repeat baseline**, not a candidate result, full jury correctness/admission, hidden-evaluation result or points claim. No solution comparison, speedup ratio or own-kernel share was measured. Four starter GPU tests are not the 79 Soup regression tests. No new GPU run is needed; the temporary output-download issue was resolved by the parent with a minimal same-route retrieval.

Machine-readable evidence and artifact SHA-256 hashes: `v3_verified_handoff.json`. Parser outcome: `v3_artifact_validation.json`; cross-artifact checks: `v3_artifact_consistency.json`.

Publication sanitization: private notebook/account identifiers, notebook URL and account quota usage omitted. The scientific artifacts and measurement values are unchanged.
