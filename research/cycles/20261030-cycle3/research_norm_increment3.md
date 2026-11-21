# Norm cycle 3 — exact graph boundaries before fusion or saved-state claims

2026-10-30 · refinements to existing norm rows only · source/CPU study, no kernel implementation, GPU, model download, installation or external write.

## Findings that change the plan

1. **A five-to-three saved-tensor count is not a memory saving.** The CPU lane confirms native frozen-weight Qwen-style norm saves five references but only three unique storages: input x, inverse RMS r, and affine weight w. The repeated references alias existing tensors. A custom save{x,r,w} function has not removed those bytes.
2. **The usual compact analytic derivative is not a universal floating-point oracle.** With finite x=3e38, native/staged backward produces NaN while one compact reassociation returns zero. Finite input alone does not establish finite internal square/derivative operations.
3. **Residual accumulation order is an independent contract.** At least one CPU cancellation witness gives native expanded-graph gradient 0 versus completed-norm-gradient-plus-bypass=1. Standalone backward equivalence is insufficient to establish fused residual backward equivalence.
4. Saving only y or normalized x and reconstructing x is not generally lossless. Recomputing r is the narrow plausible state reduction, but saves only one FP32 scalar per row and still needs matching GPU reduction/rsqrt evidence.

No new catalog ID is justified. Apply the findings to `liger_castaware_norm`, `sglang_norm_boundary`, `apex_rms_reduction`, `apex_shape_specialization`, `vllm_vector_norm`, `xformers_rms_cachepolicy` and `trt_rms_shmem` as appropriate. Cited-library provenance remains reference-only; kernels must be independently written Triton.

## 1. Source-defined Soup path

[Pinned Transformers Qwen2RMSNorm](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2/modeling_qwen2.py#L237-L252) records input dtype, casts x to FP32, computes mean(x²), applies reciprocal square root after adding instance epsilon, multiplies x by that factor, casts the normalized value back to input dtype, then applies the affine weight. Keep that order, the actual width 1536 rather than a padded block width, epsilon 1e-6 from the module, and the original tensor strides.

[PEFT k-bit preparation](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/utils/other.py#L151-L223) freezes existing parameters and upcasts non-Params4bit FP16/BF16 parameters to FP32. [Soup invokes that preparation](https://example.invalid/redacted-github) before q/v LoRA attachment. Thus the expected fixed path is FP32 residual/input, FP32 frozen norm weight, FP32 normalized output. Actual prepared module requires_grad/dtype/forward identity must still be inspected before routing. Do not emit FP16 merely because a later GEMM autocasts; LoRA dropout and bitsandbytes return-dtype behavior make that a wider semantic change.

The fixed [gradient-checkpointing default is false](https://example.invalid/redacted-github). PEFT's input-requires-grad hook in k-bit preparation is conditional on checkpointing; its [PeftModel helper](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/peft_model.py#L745-L765) also excludes already quantized models from that extra hook. Consequently the very first norm may have no input gradient requirement as well as frozen weight. Use actual autograd need, not a blanket “57 backward norms” assumption; do not force gradients or save state just to increase own-kernel share.

The Qwen class has a Hub-kernel decorator, but [the integration source describes explicit opt-in kernel mapping](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/integrations/hub_kernels.py#L133-L136). Do not activate/download a Hub kernel. Verify the actual baseline callable and patch only the intended source path.

## 2. Staged backward oracle versus algebraic shortcuts

The CPU build is Torch 2.15.0+cpu at `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`; the official notebook does not pin Kaggle Torch. Relevant source rules: [multiply](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/tools/autograd/derivatives.yaml#L1251-L1258), [rsqrt](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/tools/autograd/derivatives.yaml#L1519-L1521), [mean](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/torch/csrc/autograd/FunctionsManual.cpp#L988-L997), and [power](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/torch/csrc/autograd/FunctionsManual.cpp#L704-L727).

For the **FP32 input, FP32 frozen weight** path, derive these stages, retaining FP32 intermediates:

1. g = incoming_dy * w.
2. direct = g * r.
3. dr = sum(g*x) over the true hidden width H.
4. dv = (−0.5*dr) * r³.
5. variance_path = (dv/H) * (2*x).
6. Standalone norm dx = direct + variance_path.

This staged expression is an original decomposition of the documented autograd rules, not copied kernel code. It requires no dWeight when w is genuinely frozen. For trainable w or unsupported input dtype/layout, the initial wrapper should fall back to the unchanged reference rather than silently return no gradient. In a low-precision input fallback, the cast-before-affine has a corresponding backward cast; this FP32 specialization must not pretend to cover it automatically.

The shorter identity `r*g − x*r³*mean(g*x)` changes multiplication, division and reduction association. It can differ in cancellation, underflow or overflow. A scaled incoming gradient must participate in the same stages; do not precompute an unscaled rounded dx then scale it later.

### Triton-specific boundary, not a GPU proof

[Triton 3.8's NVIDIA backend](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/third_party/nvidia/backend/compiler.py#L113-L123) enables FP contraction by default; [backend emission](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/third_party/nvidia/backend/compiler.py#L488-L551) passes the option through LLVM and uses a ptxas no-FMA option when disabled. For an independently written norm kernel, comparing `enable_fp_fusion=False` is a precision-control experiment, not a promise of PyTorch-equivalent numerics. It does not force identical reduction trees, reciprocal/rsqrt lowering, branch accumulation or every compiler rewrite. Actual version support, emitted operations and T4 outputs need verification. Do not import/copy native CUDA source or install a different compiler.

## 3. Saved-state inventory and what can actually be removed

| State | Needed for frozen-weight first-order backward? | Memory/lifetime interpretation |
|---|---|---|
| FP32 input x | Yes in staged derivative unless exactly recomputed from independently retained state | Saving a reference does not copy x; it extends lifetime. Compressing it to FP16 is not generally lossless. |
| FP32 inverse RMS r per row | Needed, or recompute from exactly the same unmodified x and epsilon | Native graph can save multiple references to the same r storage. Recompute removes one row-scalar storage, not each reference. |
| Frozen FP32 weight w | Needed as multiplicand | Already a persistent model parameter; saving its reference adds no second weight tensor. Preserve version/lifetime checks. |
| x², variance, pre-affine normalized value, output y | Not independently required by the staged frozen-weight derivative | Do not introduce extra saved copies. Native autograd already avoids retaining several of them in the tested frozen-weight graph. |
| Residual operands a,b | Not both required once the rounded residual z=a+b is saved | Saving two full operands merely to reconstruct z can be worse; only count a saving if other graph consumers already retain them with proven lifetimes. |

**Small legitimate recomputation candidate:** for B4,T512, r contains 2048 FP32 values = 8 KiB. At most 57 such norms gives 456 KiB of raw r payload before actual requires-grad/liveness differences. This is not GPU peak-memory saving. Recomputing r duplicates square/reduction/rsqrt work and may reproduce a different reduction in backward compilation. Treat as P3 until forward/recomputed r bit patterns and complete-run memory/time are measured. It is own recomputation, not permission to enable Soup checkpointing.

**Rejected exactness claims, not a blanket ban on compression:** recovering x via `y/(w*r)` loses information when w=0, when affine/normalization rounds or underflows, and can amplify errors for tiny w. Recovering x from normalized values also cannot generally invert a rounded normalization; reconstructing r from normalized second moment is particularly ill-conditioned when that moment is near 1. Storing r or x at FP16 is a new rounding point, not lossless by definition. Other genuinely lossless compression schemes are not ruled out, but need their own equality, cost and legality evidence. No such scheme was validated here.

[Autograd's save_for_backward contract](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/torch/autograd/function.py#L34-L92) preserves saved-tensor hooks and detects in-place modification before backward. Do not bypass those checks by storing mutable tensors as arbitrary ctx attributes. Saved-output residual aliases are acceptable references only if no participant overwrites their contents. Higher-order gradients are outside this fixed training study; do not claim them from first-order tests.

## 4. Residual fusion: preserve values, branch existence and observed association

[Qwen decoder source](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2/modeling_qwen2.py#L279-L297) uses both residual additions and normalized branches. A fusion of add→norm must emit both the **rounded FP32 residual z=a+b** and normalized y; bypass consumers cannot receive a hidden higher-precision or overwritten version.

Mathematically the residual adjoint is `norm_dx + bypass_dz`, and it routes to both original add inputs. But **that grouping is not guaranteed to be the baseline floating-point grouping**. Native norm is expanded into several autograd nodes; the source residual may receive bypass, direct-norm and variance-norm contributions separately. [InputBuffer accumulation](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/torch/csrc/autograd/input_buffer.cpp#L133-L171) adds contributions as the graph supplies them. It does not specify one universal module-level association.

The CPU lane found a concrete witness: a=b=512, H=1, w=1, dy=2³⁴, bypass=1, epsilon 1e-6. Direct norm contribution is +2²⁴ and variance contribution −2²⁴. Native expanded graph returns 0, matching `(bypass+direct)+variance`; computing standalone norm dx first and then adding bypass returns 1. The same all-zero versus all-one gradient discrepancy is confirmed at the actual hidden width 1536, not only the scalar-width witness. This disproves universal bitwise equivalence, not legality of all residual fusion or a measured Soup gate failure. Even replacing norm alone with an atomic custom backward can change its interaction with a separate bypass.

**Unused versus zero-valued branch:** a two-output fused autograd function must distinguish an unused norm output from a connected norm output whose incoming gradient happens to be zero. Native autograd does not traverse the unused norm branch, while a custom backward that computes its extreme-value derivative from materialized zero gradients can create NaNs. [set_materialize_grads(False)](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/torch/autograd/function.py#L362-L407) permits explicit None handling; the implementation must handle norm-only, residual-only and both-output paths. The CPU fixture confirms this distinction: residual-only reference and explicit-None custom gradients are 1, while default materialization introduces NaN at finite residual 3e38. A connected zero norm gradient instead yields NaN in both reference and correctly routed custom paths. This is not an integration claim.

Do not mutate x, z, incoming dy or bypass gradients in place. Distinct views or identical a/b inputs can alias storage while remaining separate autograd edges; return correct contributions to both and let the engine handle their accumulation. Do not silently drop an alias edge. One-input-requires-grad and no-input-requires-grad paths must preserve their original behavior.

## 5. Promotion gates and bounded next question

- Standalone forward/backward: width 1536 plus small/tail widths, strided rows, zeros and tiny/large finite inputs, scale-sensitive dy, exact finite/nonfinite class comparison, frozen/trainable-weight routing and actual output dtype.
- Saved state: storage-aware native versus custom inventory, recomputed-r equality, rejected inversion/narrowing witnesses, version-counter/in-place protection. No GPU memory inference from Python object counts.
- Fusion: compare against the **original expanded graph**, not only an already-custom norm; independent bypass, duplicate/shared input views, norm-only/residual-only/both branches and cancellation witnesses.
- T4 still required: reference reduction/division/rsqrt behavior, compiled contraction, actual graph accumulation, no alias corruption, LoRA gradients and unchanged full harness/quality/own-share/time/memory gates.

Keep standalone cast-aware norm at P1 conditional, residual fusion at P2 after graph-boundary evidence, and recomputed inverse RMS at P3. The next useful question is whether a narrow width 1536 staged implementation preserves the installed T4 reference's realistic scaled-gradient and residual-boundary behavior—not whether fewer saved Python references look smaller.

## Separate legality, hardware and feasibility decisions

- **Legality:** independently written, semantics-preserving Triton specialization is conditional, not a ready-made-library exemption. No copied kernels, forced gradients, changed precision boundaries, checkpointing changes or unsafe aliases. Existing challenge rules and jury review remain authoritative.
- **Hardware:** FP32 elementary operations and reductions are plausible on T4/SM75; this cycle executed no GPU compiler, kernel, occupancy experiment or timing.
- **Feasibility:** CPU narrows the staged derivative and saved-state contract and rejects universal exactness claims. Even an atomic standalone custom norm can change accumulation in a branched graph. Actual T4 integration and every unchanged challenge gate remain open.

## Final CPU evidence

Final [CPU report](cpu_cycle3_results.md). Inspected [cycle3/results.json](cpu_tests/cycle3/results.json), [summary.json](cpu_tests/cycle3/summary.json), and the norm portions of the original [CPU script](cpu_tests/cycle3/run_cycle3.py). Generated 2026-10-30 11:40:46Z; Python 3.12.14, Torch 2.15.0+cpu, CUDA unavailable, two threads. Script SHA256 `285ec2e2013d6b19b7f2955220aaf0ed9778dda7b7cd4f911d3146d457c7b1c9`. Cycle 3 has six passing evidence groups, three for this norm lane and three for the separate RoPE lane; none is GPU admission. Prior 17 groups were retained separately.

- **144 standalone cases:** staged backward matches tested finite values and nonfinite classes; recomputed inverse RMS also matches the CPU reference. The compact formula differs in 46 cases, including 14 nonfinite-class mismatches. This establishes counterexamples, not expected real-model frequency.
- **Lossy shortcuts:** storing x in FP16 changes a tested gradient by up to 0.1800537109375; normalized-value division reconstruction has nonzero error; zero weight or zero inverse RMS makes the tested inversion all-NaN. No lossless output-only reconstruction was established.
- **Storage fixture, shape [2,7]:** native 5 references/3 storages/148 referenced bytes; own save{x,r,w} 3 references/3 storages/148 bytes; recompute-r 2 references/2 storages/140 bytes. The 8-byte difference is the two FP32 row scalars. These are referenced toy-storage bytes, not allocator peaks. Weight storage is already a persistent parameter. No-grad input plus frozen weight saves 0 references in all three paths; trainable-weight fallback retains its gradient.
- **Residual ordinary cases:** norm-only and residual-only paths match tested values; both-output/two-bypass-consumer cases can differ by up to 4.76837e-7 despite passing exploratory tolerances. Inputs remain unmodified across independent, identical and overlapping-view cases. Saved-residual mutation is rejected.
- **Residual negatives:** H=1536 cancellation gives native 0 versus completed-norm-then-bypass 1 across all coordinates. At finite residual 3e38, unused norm output must remain None rather than materialized zero to avoid an incorrect NaN; connected-zero norm gradients must not be skipped.

The tests do **not** establish signed-zero or NaN-payload identity, actual prepared-model routing, CUDA reduction/accumulation, allocator release, full training quality, own-kernel share or speed. They narrow the safe implementation contract and reject overbroad exactness/memory claims.
