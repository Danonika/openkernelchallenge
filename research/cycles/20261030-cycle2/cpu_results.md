# Soup CPU research-cycle results — 2026-10-30

**8 original fixture groups passed; 0 unexpected failures.** These are cloud-only, synthetic CPU semantic/numerical witnesses—not GPU kernels, training, timing, quality, or challenge scoring. Several *proposed shortcuts failed equivalence* as intended; their counterexamples are retained below and in JSON.

## Findings for the jury

| Candidate / question | CPU evidence and outcome | Jury implication |
|---|---|---|
| SOUP-NEW-001/002: streamed head semantics | 24 length/chunk cases; causal targets, global denominator, ignored rows, row association and first-index argmax matched; maximum FP32 metric difference **4.77e-7**. Attention entropy included prompt/final positions. All-ignored mean stayed **NaN**, while sum stayed zero. | Behavioral fixture passed only. Do not replace global normalization or entropy masks. |
| SOUP-NEW-001/002: stable reduction state | With logits `[32768,32752]`, single-LSE subtraction produced CE **0** instead of **1.192092824e-7**, and entropy **0** instead of **1.919772103e-6**. With 19 equal logits 65504, uncentered entropy was **2.953125** instead of **2.944439411**. Centered max/logsum formulations matched the reference within the stated tolerances. | **Reject single-LSE-only numerical design.** Keep row maximum and centered log-sum state separate; construct CE, probabilities and entropy without subtracting large common offsets. Vocabulary-tile merging is still untested. |
| SOUP-NEW-001/002: scaled gradient boundary | Five incoming scalar gradients tested. At scale 65536, baseline and scale-before-cast hidden gradient were **-0.0027008056640625**; unscaled FP16 intermediate then late scaling produced **0**. | **Reject late scaling after FP16 gradient construction.** Not a real CUDA GradScaler/optimizer test. |
| SOUP-TRAIN-001: supervised-row-only head backward | Compact/recompute/scatter matched dense Torch autograd **exactly in 15 cases** across mixed/all-supervised/all-ignored masks and scales .25/1/3/128/65536; final hidden gradients were zero on ignored rows. A separate causal-mixing witness had zero final prompt-row gradient but nonzero earlier prompt activation gradient. | Conditional algebraic support for compacting **head backward only**. Preserve all-token metric forward and complete transformer backprop; never prune prompt transformer gradients. |
| Cast-aware norm + residual bypass | 12 FP32 cases: widths 1/7/13/1536, magnitudes 1e-8/1/1e8, strided rows; analytic backward vs autograd max absolute difference **4.77e-7**. Residual-bypass difference **2.38e-7**. FP16 cast-aware backward was exact in its small fixture. Wrong cast-before-affine ordering changed output by **1.397e-3**; premature FP16 output changed it by **9.735e-4**. | Preserve FP32 output for FP32 input, input-dtype cast before affine, and both residual gradient routes. Formula is not claimed bitwise-equivalent: near-zero width-1 gradients had large relative error despite tiny absolute error. |
| SOUP-NEW-003/007: nested scale rounding | Synthetic scale code/absmax/offset gave separate FP32 scale **0.500732421875** versus single-rounded **0.5007323622703552**; endpoint NF4 value +1 then cast to FP16 changed **0.5009765625 → 0.50048828125**. 16,451 synthetic elements exercised every nibble, 256 synthetic scale codes, 64/256-scale boundaries and an odd tail; scalar/vector decode matched exactly. | **Reject unproven FMA contraction.** Codebook values/metadata here are synthetic, not a real bitsandbytes QuantState; this does **not** validate bnb reconstruction or dispatch. |
| SOUP-LORA-001: B dual-adjoint / scale-add casts | M17/N23/r16, scale2: reference adjoints and base gradient matched autograd exactly for normal/subnormal/overflow input gradients. Scaling before the first FP16 cast changed all **391 D16 elements** in the subnormal case. Removing final FP16 grad_B rounding changed all **368 elements**, max **1.1763e-4**. Overflow was preserved: 391 nonfinite D16 and 368 nonfinite grad_B elements. | Shared-adjoint tile may be investigated only while preserving G32→G16→scale→D16 and FP16 GEMM-output/parameter-gradient rounding. Overflow is not “fixed” or discarded. No dropout in this fixture. |
| SOUP-LORA-002: native-dropout saved-mask bitpacking | **10 paired train/eval cases**, lengths 1/7/8/9/257, two sequential calls: outputs, input gradients and RNG endstate matched exactly. Strided bool roundtrip also exact. At length257, bool payload for two masks was 514 bytes vs 66 packed bytes. | CPU lossless representation support only. Explicit `aten.native_dropout`, not proof of actual Kaggle `nn.Dropout` dispatch. Payload arithmetic is **not measured memory savings**; the fixture deliberately retains debug mask clones. |

## Reproduce

```bash
bash /workspace/shared/soup_cycle/cpu_tests/run.sh
```

- Script: `cpu_tests/run_cpu_semantics.py` — independently written toy tensor formulas; no source-kernel implementation copied.
- Complete evidence: `cpu_tests/results.json`; concise machine-readable outcomes: `cpu_tests/summary.json`.
- Text logs: `cpu_tests/run.log`, `cpu_tests/console_final.log`. First seven-group result/log retained as `results_initial7.json` / `run_initial7.log`.
- Existing isolated interpreter: `/workspace/shared/soup_cloud_env/venv/bin/python`; Torch **2.15.0+cpu**, Python **3.12.14**, CPU compute/inter-op threads **2/2**. Installed Torch git version is **`d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`**, different from research's source-tag commit `31a77fc...`; do not infer exact source/build equivalence from the version label. Script SHA-256 is in JSON.
- No installation, pretrained model, dataset, full training, CUDA kernel, GPU/Kaggle operation, authentication, remote write or user's-PC execution occurred.
- Official source remained at `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`; post-run git status was clean and tracked diff size zero. Existing 79+79 regression passes / 4 CUDA skips were read from the handoff, **not rerun or counted as this cycle's results**.

## Remaining gates and next question

No CPU result clears T4 runtime, real dtype/dispatch, CUDA GradScaler overflow-step behavior, true dropout dispatch, real NF4 state, full quality, peak memory, or >=5% owned-kernel-share gates. Actual bnb codebook/QuantState reconstruction remains explicitly unvalidated.

**Next CPU test question for reassignment:** Can a centered `(max, sumexp, weighted-centered-sum)` vocabulary-tile merge preserve CE/entropy/probability and lowest-index argmax across unequal tiles, high offsets and tails, without materializing full logits? The current script checks token chunking and untiled row reductions, not that merge. The next *GPU* question is whether supervised-row compaction changes head GEMM rounding/runtime or integration boundaries on the actual pinned run; that requires separately authorized T4 scope.
