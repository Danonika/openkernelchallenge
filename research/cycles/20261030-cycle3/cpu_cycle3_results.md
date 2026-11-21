# Soup CPU cycle 3 — norm/residual saved state and Qwen-style RoPE

**6 original fixture groups passed; 0 unexpected failures.** Several tested shortcuts failed equivalence as intended. “PASS” means the stated witness was established, **not** that a candidate passed challenge gates. Prior **17 groups remain immutable**. No performance-graph nodes or points were added.

## Main decisions

### Frozen FP32 RMSNorm: preserve staged backward; do not mistake saved references for storage

- **144 cases** covered widths 1/7/1536, strided inputs, zero/tiny/ordinary/large/extreme-finite input and affine values, and incoming gradient scales .25/65536. The staged derivative matched native CPU autograd's finite values and nonfinite classes exactly. Recomputing inverse RMS from the same unchanged FP32 input matched the saved inverse RMS.
- The compact algebraic derivative differed in **46/144 cases**, including **14 cases with different nonfinite classes**. A simple finite-input witness is `x=3e38, w=1, dy=1`: forward output is zero; native/staged backward is **NaN**, but the compact expression returns **0**. Mathematical equivalence does not preserve overflow/order semantics.
- Saving x in FP16 then reconstructing it changed the small-input fixture's gradients by up to **0.1800537**. Reconstructing x from normalized x/r was not exact; reconstructing via output/(w*r) with zero w or normalized x/r with zero r produced NaNs. These are not lossless saved-state substitutes.
- Native frozen-weight autograd saved **5 references but only 3 unique storages: x, inverse RMS, w**. A custom x/r/w save reduced references to 3 but retained the same 148 bytes of referenced storage in the tiny strided fixture. Recomputing r retained x/w only, 140 referenced bytes; the difference was one 8-byte inverse-RMS tensor. **These are storage inventories, not incremental allocations or measured peak-memory savings.** The model weight storage already exists independently.
- Native/save-r/recompute-r controls with no-grad input and frozen weight all saved **zero** references. The trainable-weight case used the native fallback and retained its weight gradient.

### Residual + norm: standalone exactness does not establish branch exactness

- Nine ordinary cases exercised independent inputs, identical inputs, overlapping views, norm-only/residual-only use, and two residual consumers. Both forward outputs matched, inputs were not mutated, and view gradients were handled. Some combined-branch gradients still differed by up to **4.77e-7**, within exploratory `atol=1e-6, rtol=2e-5`—**not challenge tolerances**.
- **Decisive cancellation witness at both width 1 and actual task width 1536:** `a=b=512`, frozen `w=1`, norm upstream `2^34`, bypass upstream `1`, epsilon `1e-6`. Norm's direct and variance terms are `+2^24` and `−2^24`. Native expanded-graph gradients are **all 0**; computing complete norm dx then adding bypass gives **all 1**. Native accumulation in this witness corresponds to `(bypass + direct) + variance`, not `(direct + variance) + bypass`. The ordinary tolerance passes cannot certify residual fusion.
- **Unused is not connected-zero.** With finite residual `3e38` and only the residual output used, native gradient is **1**. A custom function that materializes a zero upstream for the unused norm branch introduces **NaN**; `set_materialize_grads(False)` plus explicit None handling preserves 1. If the norm branch is actually connected with zero upstream, native and correctly implemented custom backward both produce NaN and must retain that distinction.
- Deliberate in-place modification of a saved residual triggered version-check errors in both native and custom controls. No unsafe input/output alias overwrite is proposed.

### Existing `attn-sgl-rounding-fusion`, variant `qwen2-fp32-half-split-apply-only`

- **32 FP32 paired q/k cases**: head counts 12/2, width128, batch2, T7, transposed and alternate-axis layouts, position batch1/2 broadcasting, duplicated and deliberately unequal coefficient halves, independently trainable/unused q/k outputs. Forward values, input gradients, requires-grad routing and unused-gradient None behavior matched the tested native expression exactly.
- The source-like `[2,12,7,128]` q view and reference output had stride **`[10752,128,1536,1]`**, not contiguous BHTD. k and custom output strides were also recorded and matched. Do not impose a contiguous output layout without validating downstream behavior.
- Native frozen-coefficient RoPE saved **4 references to only 2 unique storages: cos and sin—not q/k**. The custom boundary saved 2 references to the same 2 storages, 7168 referenced bytes in this fixture. Thus “no saved q/k” is **not a demonstrated storage improvement over this baseline**.
- Interleaved rotation failed against split-half rotation. An inverse-style adjoint failed for unequal coefficient halves; the general pair adjoint must use each half's actual c/s values. Actual duplicated-half coefficients are a narrower source contract, not permission to assume it for arbitrary callers.
- A bounded one-product FP32 contraction simulation with ordinary duplicated trig coefficients changed **1784/4096** outputs. One witness: separate FP32 result **−1.2591073513**, contracted result **−1.2591072321**. This is a numerical counterexample, **not execution of a GPU FMA or proof that a challenge tolerance fails**.
- FP16, mixed FP16 q/k + FP32 coefficients, and trainable coefficients were routed to the reference in explicit fallback controls. Mixed inputs produced FP32 outputs and FP16 q gradients, showing why a universal dtype assumption is unsafe. No custom mixed-dtype equivalence is claimed.

## Exactness and false-positive limits

“Exact” above means equality of tested finite numerical values plus separate finite/NaN/+Inf/−Inf classification where applicable. It does **not** establish signed-zero bit identity, NaN payload identity, untested reduction trees, or GPU behavior. RoPE nonfinite inputs/gradients were not added to this bounded cycle. Ordinary residual close-tolerance passes coexist with a large exact cancellation counterexample; neither group counts nor small average errors override that negative evidence.

No actual model snapshot, runtime dtype routing, CUDA compiler/codegen, T4 speed, allocator peak, full training quality, challenge numerical tolerance, or owned-kernel-share gate was tested. Baseline-only performance visualization remains untouched.

## Reproduce and inspect

```bash
bash /redacted/local/soup_cycle/cpu_tests/cycle3/run.sh
```

Files under `cpu_tests/cycle3/`:

- `run_cycle3.py`: independently written toy reference/custom-autograd formulas.
- `results.json`: detailed evidence; `summary.json`: concise outcomes.
- `run.log`, `console.log`: final outputs; `results_first6.json` / `run_first6.log`: earlier checkpoint.
- `prior17_before.sha256`, `verification.txt`: hashes verifying all 32 prior artifact/report files unchanged, plus official source integrity.

Cloud CPU only; Python 3.12.14, Torch 2.15.0+cpu at `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`; two intra-op/two inter-op threads. No reinstall, official-suite rerun, model/data download, user's-PC execution, GPU/Kaggle operation or remote write.

**Next question for reassignment:** Which narrow runtime dtype/layout contract and backward accumulation policy will the first actual T4 candidate guarantee? CPU evidence favors standalone frozen FP32 norm and guarded apply-only split-half RoPE investigation; it does not establish residual-fusion parity or any measured configuration result.
