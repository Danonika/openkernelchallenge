# Soup NF4 + rank-16 LoRA: focused source audit
2026-10-30. No benchmark, GPU execution, installation or copied implementation. This refines SOUP-NEW-003 and existing small-rank fusion ideas; only one new candidate merits a new ID.

## Highest-value findings
1. **A functional.dequantize_4bit patch misses forward.** bitsandbytes 0.50.2 MatMul4Bit.forward calls torch.ops.bitsandbytes.gemm_4bit, whose CUDA fallback invokes backend dequant functions directly. Backward calls functional.dequantize_4bit. Instrument/patch both intentionally.
2. **The baseline already has T4-native fused NF4 GEMM for some small M.** At B=4,T=512, M=2048 exceeds its 1536 hard cap and uses dequant+linear. Shorter shapes are not automatically that path. Its T4 heuristic is much tighter for many Qwen shapes.
3. **Keep PyTorch dropout intact in first LoRA experiments.** q and v use separate module invocations and random draws; sharing one mask or moving RNG around changes the task.
4. Prioritize cast-aware norms + complete head/metrics first; NF4 fused decoding is a bounded next candidate; bespoke NF4 GEMM and fused LoRA autograd remain higher-risk until T4 evidence.

## Pinned official sources
All bnb links below use [0.50.2 commit 08a9956](https://github.com/bitsandbytes-foundation/bitsandbytes/commit/08a9956b939857ae2c4fbd56753960cc370fa7cf), dated 2026-08-27:
- [functional.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/functional.py): quantization metadata, nested decode, transpose compatibility.
- [autograd functions](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/autograd/_functions.py): canonical packing, forward dispatch, input-only backward.
- [CUDA backend](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/backends/cuda/ops.py): small-M/T4 thresholds and fallback.
- [default backend](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/backends/default/ops.py): explicit unpacking semantics, useful as mathematical reference.
- [native dequant implementation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/csrc/kernels.cu): NF4 lookup and multiply/cast boundaries.
- [GEMM dispatch](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/csrc/gemm_4bit.cu): SM75 fp16 m16n8k8 path.
- [Linear4bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/nn/modules.py): input/compute/output dtype and bias handling.

PEFT [0.20.0 LoRA bnb.py](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/lora/bnb.py), commit 2026-07-28. Soup [quant configuration](https://example.invalid/redacted-github), commit 2026-10-26. [NVIDIA Turing tuning guide](https://docs.nvidia.com/cuda/pdf/Turing_Tuning_Guide.pdf) supports hardware constraints; [bnb release notes](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.50.0) confirm fused 4-bit support was introduced in the baseline family, not a novel external solution. Release inference speedups are **not Soup training measurements**.

## Exact NF4 reconstruction contract
Let i index the flattened original [N,K] weight, b=floor(i/blocksize), c=floor(b/state2.blocksize). Typical task block sizes are 64 then 256, but read actual QuantState rather than hardcode.

- Packed byte at floor(i/2): **high nibble for even i**, low nibble for odd i.
- First-stage code is one of 16 **NF4** values. It is not signed INT4, uniform INT4, FP4 or NVIDIA E2M1.
- Nested scale is first FP32 multiply of state2.code[quant_state.absmax[b]] by state2.absmax[c], then **separate offset addition**. state2.code is the 256-entry quantizer codebook; don't replace it with a hand-assumed linear scale.
- Multiply NF4 value by reconstructed scale in FP32, then cast to requested dequant output dtype. Fusion must not silently turn scale multiply+offset into FMA; compare all decoded weights with the original path before using them.
- quant_state.shape is source of truth for [N,K]. Current matmul canonicalizes storage to [(N*K+1)//2,1]; a raw functional caller with shape[0]==1 gets a historical transposed output. Square weights make orientation inference ambiguous.
- Quant metadata includes original dtype, code, blocksize, nested state and offset. Preserve tail masking and non-uint8 storage view behavior. Do not rewrite/quantize base weights.

Forward and backward are not one function:
- CUDA fallback reconstructs scales, dequantizes directly to A.dtype and uses F.linear(A,W,bias).
- MatMul4Bit backward obtains W through functional.dequantize_4bit, then casts to grad_output.dtype, and calculates grad_input=grad_output @ W. No frozen weight gradient.
- Linear4bit casts original input to compute dtype, casts bias as necessary, then restores original input dtype on result. Narrow replacement must preserve these casts, bias and autograd behavior, including no-input-grad early layers and inference fallback.
- Replacing only the Python dequant function can help backward while leaving forward unchanged; interpreting that as full coverage would be an experimental error.

## Legal implementation boundaries
**First candidate: original fused nested decode, still ordinary matmul.** Reconstruct scales and weights in a single streaming Triton pass. Use regular launches, current stream and solution-owned source. Keep bnb's existing small-M fused forward unless own replacement is proven faster/correct for that exact shape. Prefer explicit narrow wrapper routing over globally replacing all torch custom-op registrations.

**Do not cache every fully dequantized layer.** That can retain gigabytes, violate memory gate and reverse the point of NF4. A decode scratch tensor must live until the consumer matmul completes on the same stream; backward reuses packed weights or a safe bounded cache, not a dangling alias.

**Do not treat existing bnb native kernels as owned.** Their runtime is allowed as baseline dependency but does not contribute to the >=5% own-Triton share. Fusing already-fused small-M operations may have little opportunity.

## One incremental candidate worth adding
| ID | Mechanism | Status / constraints | Cost and priority | Evidence / dedup |
|---|---|---|---|---|
| SOUP-NEW-007 | Cache only reconstructed frozen FP32 **block scales**, not full weights, across forward/backward and steps. Decode packed NF4 weights from that table. | **Conditional.** Same exact scale multiply then offset, no mutable quant_state rewrite; invalidate on packed weight/scale/offset/version/device changes; construction time counts; one measured process only. Own initialization kernels alone do not satisfy steady-step kernel share. | Approx. **78.094 MiB** for all seven quantized Qwen projections over 28 layers at blocksize64 (derived geometry, verify actual coverage). Compare against fused on-the-fly scale decode; cached loads may be slower than cheap arithmetic. **P2**. | Inference from pinned bnb frozen metadata/lifetimes; narrows generic gemm-12 cache/lifetime idea to two-level NF4 scale amortization. Distinct from caching 445 MiB head or dense base weights. |

Estimated scale bytes: 28*(2*1536² + 2*256*1536 + 3*8960*1536)/64*4. This is a storage calculation, not a performance result. It excludes metadata, padding/alignment and allocator overhead.

## Rank-16 q/v LoRA path and opportunities
Per layer, base shapes: q=[1536,1536], v=[256,1536]. Adapter A=[16,1536] for both; Bq=[1536,16], Bv=[256,16]; scaling alpha/r=2. Base output is normally FP32 because norm input is FP32; LoRA linears run fp16 under autocast.

Pinned PEFT sequence: base_layer(x), **defensive clone**, dropout(x), A linear, B linear, scaling, out-of-place add. The clone has an explicit autograd/view-safety reason, so do not delete it just by observing that it looks redundant.

Potential experiment order:
1. **Scale+base-add epilogue**, preserving fp16 adapter-output rounding and multiply rounding before FP32 promotion/add. Original Triton pointwise, ordinary A/B GEMMs and ordinary dropout retained. It overlaps generic L6-10; no new ID needed.
2. **B GEMM + scale+add**, skinny K=16 fp16 with FP32 accumulation. Preserve intermediate cast exactly; T4 tile size/register pressure may outweigh launch savings. Needs grad_B, grad_A-output and base-gradient routing.
3. **Two-stage A/B on-chip fusion**, only if the rank16 intermediate and required rounding fit; large output width does not justify recomputing A for every output tile without measuring total work. This is existing L6-16/inductor_b2b_fusion, not new.
4. q/v launch grouping may share scheduling but must keep different parameters, shapes, dropout draws and output indexing. Never substitute one shared dropout result. Shared *un-dropped* source loads are not equivalent to shared dropped values.

Autograd obligations:
- grad_B uses original (or recomputed identically) dropout→A intermediate.
- grad_A uses original dropped input; grad_x has both base and dropout-mask-scaled adapter paths.
- Keep FP32 trainable adapter parameters and optimizer state; fp16 autocast inputs/outputs create real rounding points in both directions.
- Upstream gradients are scaled by GradScaler; retain those scalings before lower-precision casts. Avoid parameter-grad atomics in fp16.
- Save exact dropout mask/output or retain original dropout autograd. Recomputing dropout without preserving generator state produces a different gradient and advances RNG twice.
- Torch is not pinned by notebook. Exact custom Philox compatibility cannot be promised before actual torch version/implementation and generator advancement are identified.

## Risk-adjusted priority relative to current plan
| Work | Priority | Main reason / unknown |
|---|---|---|
| FP32 cast-aware RMSNorm fwd/bwd | P0/P1 | Narrow semantics, repeated layers; actual share unknown. |
| Complete streamed head+metrics+rewind | P0/P1 | Largest visible memory opportunity; GradScaler and recomputation costs matter. |
| Fused nested NF4 decode on confirmed fallback/backward paths | P1 | Bounded, frozen-state transform; forward routing coverage must be proven. |
| Adapter scale+add epilogue | P2 | Low semantic scope if casts preserved; likely small saved work. |
| Frozen scale table cache | P2 | Small relative memory cost, unmeasured arithmetic-versus-read tradeoff. |
| New fused NF4 GEMM / full LoRA autograd | P3 pending profile | Baseline already tuned for SM75 in some shapes, much larger correctness surface. |

## Tests needed before promotion (not executed)
- Decode all 16 NF4 values and all 256 second-level codes, scale boundaries 63/64 and 255/256, odd element tails, zero/negative/large decoded centered scale terms, exact offset rounding; compare fp16 bytes when expected.
- Real q/v/o/gate/up/down shapes, forward and backward orientations, 2D/3D activations, noncontiguous input and packed-view compatibility. Frozen bias values matter on q/v.
- Dispatch fixtures just below/above actual bnb thresholds and M=2048; prove which replacement runs without measurement-specific branching.
- LoRA ordinary dropout retained: exact RNG state progression, separate q/v calls, both A and B gradients, x gradient, training/eval distinction, scale and autocast dtype boundaries.
- Full cold harness: unchanged outputs/adapter and all numerical gates, own-kernel share, no memory regression. No speedup is inferred from operation count.

## Recent-source search conclusion
The useful recent result is a **stronger baseline**, not an unclaimed free optimization: bnb 0.50 introduced native fused nested NF4 kernels and SM75-specific dispatch. Hardware guidance reinforces fp16 m16n8k8/resource-aware tiling; Hopper TMA/WGMMA, Ampere cp.async and FP8 are not T4 transfers. No additional genuinely distinct, validated T4-training technique was found in this bounded pass; avoid padding the catalog with generic fusion entries already present.

