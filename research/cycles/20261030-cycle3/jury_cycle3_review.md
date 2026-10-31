# Jury increment 3 — Qwen2 RoPE and RMSNorm/residual contracts

**Decision:** refine existing IDs only; keep **121 rows**. Advance narrow FP32 RoPE application as a P2 hypothesis, retain standalone FP32 RMSNorm at P1, and keep residual fusion at P2 after its independent gradient proof. No new graph nodes, candidate points or measured performance coordinates are created. The display remains baseline-only under the user's instruction.

Prior review preserved unchanged in `jury_snapshots/cycle2_closed_20261030T1131Z/` with a 15-file hash manifest. All prior 17 CPU fixture groups remain separate evidence, not admission. This increment adds source contracts and bounded CPU witnesses; it does not implement a Triton kernel or run T4 training.

## Decisions that matter

| Existing ID(s) | Retained transfer / verdict | Exact promotion obligation |
|---|---|---|
| `attn-sgl-rounding-fusion`; shared restrictions on `trt_qknorm_rope_fusion`, `attn-vllm-vector-prep` | **Conditional P2:** own FP32 application of Qwen2 RoPE to q/k; one implementation group, not three new kernels | Half-split rotation, unequal q/k head counts, original coefficient broadcast, separate multiply/add rounding, original output strides and both adjoints. Keep coefficient generation and SDPA unchanged. No added QK norm, interleaving, FP8 cache or early half conversion. |
| `liger_castaware_norm`, `apex_rms_reduction`, `trt_rms_shmem`; norm branch of `vllm_vector_norm` | **Conditional P1:** own frozen-weight standalone FP32 RMSNorm fwd/bwd | Use source-staged backward as initial oracle; compiled reduction/operation order and actual prepared-model dtype/epsilon still require T4 proof. Tiny CPU exactness is not whole-training parity; an atomic standalone custom norm also changes residual-connected gradient grouping. |
| `sglang_norm_boundary`, `xformers_rms_cachepolicy`; residual branch of `vllm_vector_norm` | **Conditional P2**, not a lossless drop-in assertion | Return both original rounded residual and norm output; handle aliases and independent consumers; preserve saved-tensor version checks. Expanded autograd's branch accumulation order is not guaranteed by summing completed norm dx then bypass. |
| `apex_shape_specialization` | **Conditional P2** refinement | RMSNorm remains RMSNorm: no LayerNorm mean subtraction. Tune after independent norm parity and real shape/resource evidence. |
| Saved-state reconstruction or inverse-RMS recomputation variants | **Hold P3** within existing norm IDs | Reconstructing x from normalized/output state is not generally lossless; duplicate saved references do not mean duplicate storage. Recomputing r removes only one scalar per row and adds unmeasured work. No standalone memory-win claim. |
| `vllm_packed_swiglu`, `xformers_packed_swiglu` | Existing **hold/reference** unchanged | Plain SiLU×up is already the shipped example. Renaming/rewriting it does not establish own-share credit; no new activation ID justified. |

## 1. RoPE: a narrower, source-supported target

Pinned Qwen2 generates cos/sin once per model forward, from FP32 frequency/trig calculations then a cast to embedding dtype. It passes these shared coefficients to every layer. q/k projections are viewed and transposed; q has 12 heads and k has 2 in this fixed model. Rotation uses first/second feature halves, **not even/odd interleaving**. [Pinned Qwen2 source](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2/modeling_qwen2.py#L51-L135).

For halves a,b and coefficients c1,c2,s1,s2, the backward routes cross-half sine terms: `da=g1*c1+g2*s2`, `db=g2*c2−g1*s1`. A generic “inverse RoPE” shortcut using the same-half sine is wrong unless the required coefficient symmetry is guaranteed. Even with real trig coefficients, single-contracted product/add differs from separately rounded FP32 operators. Preserve the baseline arithmetic first; a CPU difference is not by itself an actual harness failure prediction.

The expected Soup path is FP32 q/k and coefficients, but this still needs actual T4 instrumentation. Do not cast q/k early merely because SDPA later autocasts: q's FP32 base-plus-LoRA sum can contain values not representable in half, and optional key-cache update occurs before attention. [Attention/cache sequence](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2/modeling_qwen2.py#L190-L233), [SDPA routing](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/integrations/sdpa_attention.py).

**Memory claim correction:** frozen-coefficient native RoPE already saves cos/sin, not q/k activations. Four saved references versus two in a custom function still point to the same two storages. This is a launch/transient-traffic hypothesis, not demonstrated retained-activation compression. Save coefficients through `save_for_backward`; preserve version checking and return baseline for trainable coefficients, unsupported dtype/layout/device or generic API cases.

A separate tiny CPU routing probe found the installed Qwen2 file's Git blob matched the pinned source, with no `kernels` package; a temporary module-global rotary wrapper was reached once and restored. This supports the local hook location only—not Kaggle routing, automatic hub substitution, SDPA dispatch or performance. See `rope_cycle3_route_probe.json` and its original script.

## 2. Standalone norm: staged derivative, not a universal simplified identity

Pinned Qwen2 casts x to FP32, squares and averages, applies reciprocal square root, normalizes, casts normalized values to the original input dtype, then multiplies the affine weight. Expected frozen FP32 Soup input/weight must be confirmed rather than globally assumed. [Qwen2 RMSNorm](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2/modeling_qwen2.py#L236-L255).

The source-staged backward preserves separate direct, rsqrt, mean and square-path operations. The compact analytic identity is mathematically useful but not a universal floating-point oracle: finite huge x can give baseline NaN gradient while the simplified expression returns zero. Similarly, half-saving x or reconstructing it from output/weight is not generically exact, especially with zero weight, zero inverse RMS, overflow or lost low bits. Reject **claims of losslessness**, not all activation-compression research as a blanket rule category.

The native frozen norm already retains unique x/r/w storage. Duplicate x/r saves are references, not extra activation buffers. At B4,T512, one inverse-RMS scalar per row is **8 KiB per norm** by geometry, not an observed peak saving. Recomputing r adds reduction/rsqrt work and requires actual GPU parity; hold that variant behind a real storage bottleneck. Frozen bnb base matmul also does not save activation A, so do not assume a norm-output buffer is already retained there for free. [Pinned bnb saved state](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/autograd/_functions.py#L350-L390).

## 3. Residual fusion: two separate correctness obligations

1. **Branch association:** the native expanded graph may accumulate bypass, direct-norm and variance-path terms separately. Completing norm dx then adding bypass changes FP32 association. A source-matched CPU cancellation witness at the actual width1536 gives native gradient all0 and regrouped gradient all1. This refutes bitwise-equivalence claims, not all possible full-run tolerance compliance. GPU engine ordering remains unmeasured.
2. **Unused versus connected-zero outputs:** a two-output custom function must distinguish an unused norm gradient (`None`) from a connected zero gradient. Default zero materialization can execute `0×Inf` in an otherwise untraversed norm branch, creating NaNs where the residual-only baseline gradient is1. Appropriate materialization control/None handling is necessary; skipping all numerically zero gradients would also be wrong.

Keep rounded residual and normalized output separately visible; no in-place overwrite of saved inputs or outputs, including identical/overlapping inputs. Preserve autograd version checks via `save_for_backward`, not raw ctx tensor attributes used to evade saved-state handling. Standalone norm parity alone cannot clear these obligations.

## 4. Evidence and exact remaining boundary

**Six new CPU fixture groups passed as positive/negative evidence; 23 groups total remain non-admission evidence.** The final `cpu_tests/cycle3/results.json` SHA-256 is `08a828bd0926f6432809002e9988856414dc5d291414d3079549853fc3a25715`; its script SHA-256 is `285ec2e2013d6b19b7f2955220aaf0ed9778dda7b7cd4f911d3146d457c7b1c9`. The CPU lane's `cpu_cycle3_results.md` records exact cases/tolerances: 144 staged norm cases, actual-width1536 residual counterexamples, 32 finite RoPE cases, storage/fallback controls and contraction/layout negatives. Existing-ID overlay `jury_cycle3_updates.json` binds the evidence by group and preserves all earlier artifacts. `jury_cycle3_source_contracts.json` records independently reviewed source pins, Git blob SHAs, line ranges and contract findings.

What remains: actual prepared-model q/k/norm/coefficient dtypes and output strides; actual decorator and SDPA routing; compiled Triton separate-rounding behavior, backward branch/reduction order and resource/launch costs; full unchanged T4 numerical/quality/time/memory/own-share gates. No candidate receives coordinates or a measured gain from these source/CPU findings.

**Stop this increment after the bounded fixtures and source checks.** No duplicate IDs, no extra broad audit, no automatic implementation rollout. The publisher may attach this overlay to the existing table; the prior review snapshot stays intact.
