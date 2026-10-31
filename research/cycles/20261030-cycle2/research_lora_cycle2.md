# Soup LoRA cycle 2 — make the next T4 test decisive

**Same IDs, no new catalog rows.** This bounded follow-on refines SOUP-LORA-001 and SOUP-LORA-002. Cloud CPU/source work only; no GPU, installation, authentication, training workload, external write or performance claim.

## Decisions

1. **Separate `001-D16` from `001-composite`.** Sibling-adjoint reuse by itself has a weak logical traffic case at small tiles. Owning B→scale→FP32-add allows the backward kernel to consume G32 and remove two full-sized intermediate gradient materializations while preserving both FP16 roundings. That is a wider autograd proof, not a free promotion.
2. **Retain S16; do not recompute it by default.** At the maximum task shape, each rank-16 state is only 64 KiB. Recomputing A costs an additional read of about 6 MiB of dropped FP16 inputs and about 100.7 million FLOPs per branch.
3. **Do not fuse GradScaler unscale or optimizer updates into these adjoints.** LoRA alpha/r=2 is not the dynamic loss scale. Parameter gradients remain scaled until ordinary global nonfinite checking, unscale, clipping and optimizer-step control.
4. **For mask packing, prefer an individual SavedTensor hook on the verified native mask**, not a default saved-tensors context. This guarantees selectivity by target rather than hoping that the context saves nothing else. Unrecognized nodes and existing-hook conflicts stay baseline.

## 1. Exact graph and source-derived shapes

PEFT's pinned forward is base→defensive clone, followed by dropout→A→B→scale→out-of-place add. Keep that clone and base path unchanged in this experiment. Source: [PEFT 0.20.0, commit a5526d27](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/lora/bnb.py#L513-L565).

For each of 28 layers, H=1536, r=16; q has N=1536 and v has N=256. The [official model config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json) gives 12 query heads and 2 KV heads; [pinned Transformers projection construction](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2/modeling_qwen2.py#L180-L192) implies head dimension 128 and these output widths. Config `main` is mutable, read on 2026-10-30; a model-API request to identify a snapshot was unavailable, so no model commit is invented. M is the unchanged batch's padded token count, at most 4×512=2048. The actual loaded shapes and strides remain the dispatch authority. Our expected graph is:

```text
X32 -- native dropout q/v separately --> dropped X32 -- cast --> dropped X16
  -- A16 GEMM --> S16[M,16] -- B16 GEMM --> Z16[M,N]
  -- multiply 2, round FP16 --> U16 -- add to cloned base32 --> Y32

backward of B+scale+add:
G32 -------------------------------> base gradient unchanged
G32 -- cast FP16 --> G16 -- multiply 2, round FP16 --> D16
  D16 @ B16                    --> dS16
  D16.T @ S16                  --> dB16 --> cast FP32 master contribution
```

The CPU metadata-only probe [script](lora_cycle2_opgraph.py), [trace](lora_cycle2_opgraph.json) confirms this **explicit-cast eager toy**: a G32→G16 `_to_copy`, FP16 multiplication by 2, two FP16 `mm` calls, then dB16→FP32 `_to_copy`. It observes base gradient equal to G32. No CUDA trace, timing or autocast-equivalence inference is made.

Keep dB's canonical shape `[N,16]`; one-axis partial storage is `[ceil(M/BM),N,16]`. Earlier first-cycle prose's `[16,Ntile]` notation was corrected; the byte arithmetic never depended on orientation.

## 2. Tiling, traffic, and the occupancy trap

### Variant A: `001-D16`

The custom kernel receives already-materialized D16. One CTA owns an M tile and loops over N tiles. It retains dS accumulators and writes dB partials; a second kernel reduces only dB across token tiles. Baseline two-GEMM D reads cost ideally `4MN` bytes; fused D read plus scratch write/read costs `2MN + 8*ceil(M/BM)*N*r`.

### Variant B: `001-composite`

A custom autograd function owns B+scale+base-add, but keeps ordinary forward math initially. It receives G32 in backward, returns the same base gradient, and computes **cast-to-half first, then multiply by 2, then round-to-half** inside each gradient tile. It must take B's FP32 master parameter as input and return `float(fp16(dB_accum))`, not unrounded FP32 dB. Keep ordinary A/dropout outside the function and save the exact effective S16/B16.

Under the CPU-observed eager materialization pattern, the ideal gradient-pipeline traffic is `14MN` bytes: G32 read4, G16 write2/read2, D16 write2, and two D16 reads4, all multiplied by MN. The composite variant costs `4MN + 8*ceil(M/BM)*N*r` for G32 plus partials. These formulas intentionally omit other operands, final outputs, cache hits, layout conversion, allocation, launches, and compiler work. They are a hypothesis to test against the **actual CUDA graph**, not measured device traffic.

At M=2048, Nq=1536, r=16, BN=128:

| BM | Token CTAs | dB scratch storage | D16-only gradient bytes (baseline 12 MiB) | Composite gradient bytes (eager model 42 MiB) |
|---:|---:|---:|---:|---:|
| 32 | 64 | 6 MiB | 18 MiB | 24 MiB |
| 64 | 32 | 3 MiB | 12 MiB | 18 MiB |
| 128 | 16 | 1.5 MiB | 9 MiB | 15 MiB |

For v, these bytes are one-sixth, but CTA counts are unchanged. This exposes why the composite and D16-only experiments must not be conflated. At BM32, D16-only actually loses the simplified byte comparison; the composite can still remove gradient intermediates. Neither comparison predicts runtime.

Turing has 32 maximum resident warps/SM, 64K 32-bit registers/SM and 64 KiB shared memory/SM; FP16 Tensor Core products can accumulate in FP32. [NVIDIA tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#occupancy) The T4's 320 Tensor Cores and the guide's 8/SM imply 40 SMs; this is a hardware-count inference, not utilization evidence. [T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf)

Thus 16 token CTAs do not occupy all T4 SMs simultaneously. BM32 supplies 64 at M2048, but short batches still underfill. Splitting N creates more blocks **and dS partials**; add `8*ceil(N/CN)*M*r` scratch write/read bytes for that design. No free grid expansion exists.

The dS+dB accumulator payload lower bound is `4*r*(BM+BN)` bytes: 10/12/16 KiB at BM32/64/128 and BN128. This is not a register-allocation prediction: opposing D/D-transpose dot layouts, operand fragments, duplicate lanes, temporary casts, loop hoisting, and spills add costs. Inspect actual compiler register/shared/local-memory metadata before considering a launch shape successful. Do not assume Hopper scheduling, cp.async, BF16 or FP8.

Reproducible logical counts across M=4/28/128/512/2048 and both N values: [budget script](lora_cycle2_budget.py), [budget JSON](lora_cycle2_budget.json). These are derived payload bounds only.

## 3. Materialize, recompute, or fuse a scale?

**S16:** save it. At M2048, 56 possible q/v states total only 3.5 MiB. Recomputing S from already-saved dropped X16 adds `2*M*1536*16` FLOPs and a large activation read per branch. It must use the original dropped values and exact effective A16; replaying dropout is not an acceptable shortcut. This reinforces the existing L6-16 split-at-rank refinement, not a new idea. [LoRAFusion §5.1](https://arxiv.org/html/2510.00206v1#S5.SS1)

**D16:** transiently materializing it can still be fastest if vendor GEMMs outperform the coupled implementation. A simple cast+LoRA-scale kernel producing D16 for the two ordinary GEMMs is a necessary lower-risk ablation. It removes only G16 write/read, without changing GEMM tiling. Keep this under existing pointwise-fusion L6-10, not another row.

**Independent adjoint prologues:** each product could recompute exact D16 from G32 instead of sharing a tile. That doubles G32 reads but avoids the coupled layouts. It is a useful explanation/control if sibling fusion fails; it is not automatically worth a new implementation.

**LoRA scale versus optimizer scale:** factor2 may be fused into each gradient tile only after G32→FP16 rounding. Dynamic loss-scale unscale is different. PyTorch's unscale is performed after all assigned gradients are accumulated and jointly checks nonfinite values. Clipping uses the global norm. [Pinned GradScaler](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/torch/amp/grad_scaler.py#L260-L332), [pinned clipping](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/torch/nn/utils/clip_grad.py#L155-L175)

Do not move unscale before the FP16 boundaries, update B in backward, mark only a subset of gradients unscaled, or suppress Inf/NaN. All adapter FP32 gradient payload is about 8.3125 MiB by the known shapes, whereas the targeted activation-gradient intermediates are much larger in aggregate. A separate whole-optimizer fusion belongs to SOUP-NANO-001 and its global ordering proof, not this experiment. No optimizer traffic saving is counted here.

## 4. Select only the native mask: an individual SavedTensor hook

Instead of installing default hooks, call the original dropout first. On a recognized native-dropout output node, target only its `_raw_saved_result1` and register a pack/unpack pair there. The actual CPU build exposes `NativeDropoutBackward0` and that raw attribute. This is a version-dependent target with fallback, not a universal name guarantee. [Official individual SavedTensor-hook documentation](https://docs.pytorch.org/docs/2.15/notes/autograd.html#registering-hooks-for-a-saved-tensor)

Required guard sequence:

1. The module is an actual selected q/v adapter dropout; training mode, supported dtype/device/shape, and input requires gradients. Fake modules/no-grad/eval retain baseline.
2. Output has the expected native node and exactly the expected raw mask target. Do not read `_saved_result1` just to inspect it; that can trigger an existing unpack hook.
3. In the immediate pack callback, verify bool, original shape/device, and supported logical layout. Keep only packed payload plus scalar shape/device metadata; do not capture X, Y, grad_fn, the raw SavedTensor or original mask in a closure/packet.
4. Reconstruct the same bool contents on every unpack, including retained-graph repeated backward. Do not mutate either callback input.

Pinned [SavedVariable registration](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/torch/csrc/autograd/saved_variable.cpp#L243-L287) rejects an already-hooked saved tensor **before mutation**. For this verified conflict, retain the existing hook and decline packing; do not replace it. Track whether the pack callback was entered, and recognize the already-hooked error only when it was not entered. A callback can itself raise matching text; that is not the safe conflict. Other exceptions are not safe fallback signals: the implementation installs its hook before calling pack. A pack failure can therefore leave changed state; propagate that failure and fix the implementation rather than retrying or silently continuing.

After successful registration, that source clears the SavedVariable's original data reference. This proves removal of one owner, **not actual VRAM release**: external references, temporary masks, packed data, allocator state and backwards lifetime still require observation. The first-cycle debug-copy fixture cannot establish memory savings. No broad global hook, no interception of A/B/norm saved tensors, and no default-hook nesting override is needed in this refined route.

## 5. Decisive GPU admission experiment, when separately available

1. **Read-only baseline facts:** actual Torch/Triton/CUDA, q/v shapes/dtypes/strides, S/B saved lifetimes, graph/kernel identities for casts/multiply/two adjoints, combined durations, native mask identity/count/peak overlap. If the expected intermediates or mask lifetime do not exist, reject that rationale before implementation.
2. **Numerical gates first:** compiled exact cast chain including `G=3*2^-27` and overflow, final FP16 dS/dB boundaries, noncontiguous/tail shapes, FP32 base gradient, independent dropout RNG, finite/nonfinite parity and genuine GradScaler skip/clip/update behavior. Inspect compiled output for erased first narrowing; CPU success is not enough.
3. **Ablate only distinct causes:** unchanged baseline; fused cast+scale with ordinary GEMMs; D16-only sibling; composite sibling. Keep S materialized and NF4 unchanged in all arms. Test a small shape/launch set including BM32/64/128 and BN64/128, not an unbounded autotuner. Time complete two-output work, reductions, casts, scratch and cold compilation.
4. **Stop conditions:** fail correctness→reject that realization; no substantial baseline adjoint budget→hold; extra layout/scratch/compile cost erases combined gain→retain simpler baseline/epilogue; mask does not overlap measured peak→retire codec. Only survivors enter unchanged full harness/held-out/79-test/own-share gates. No gain, admission or priority upgrade follows from this CPU/source pass.

## 6. Completed CPU cycle 2b evidence

Independent CPU lane: [results](cpu_tests/cycle2b/results.json), [summary](cpu_tests/cycle2b/summary.json), [original script](cpu_tests/cycle2b/run_cycle2b.py), [runner](cpu_tests/cycle2b/run.sh). Three fixture groups passed on Torch 2.15.0+cpu, build `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`; no GPU or full model.

- **Composite B+scale+add ownership:** six normal/subnormal/overflow × train/eval paired q/v cases (toy M=17, H=13, r=16, qN=23, vN=11) preserve outputs, finite gradient values, nonfinite classes and final RNG state. X/A/B/base/S16 gradients were compared; independent dropout calls and the defensive clone were retained. The base is an independent FP32 toy linear, not bitsandbytes. The custom Function still executes ordinary Torch operations, so this is an autograd contract test, not fused-kernel feasibility.
- **Actual CPU autocast crosscheck:** three normal/subnormal/overflow cases show CPU autocast `F.linear(S16,B32)` agrees with the explicit half-cast reference for output and S/B/base gradients. CUDA/PEFT autocast remains unverified.
- **Selective mask hooks:** five tail lengths 1/7/8/9/257 pass two retained-graph backward calls, exact output/gradient/RNG comparisons, and neighboring sin/square saved-tensor noninterception checks. Existing-hook conflict declines packing before the callback. Intentional pack failures, including a forged error with the same conflict text, propagate because the callback-entered flag distinguishes them. No original-mask debug clone or output-graph capture is retained. This is still not a storage-release/peak-memory measurement.

The research jury accepts the two 001 variants under the existing ID and keeps their current priority. This note ends the bounded second cycle; further broad searching is not needed before runtime evidence.
