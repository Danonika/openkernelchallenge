# Soup 0.77: source-derived head/metrics and RMSNorm design
Research date: 2026-10-30. Official sources only in this pass. No execution, installations or implementation copying. This refines the existing SOUP-NEW-001/002 and cast-aware-norm ideas; it is not another list of duplicate ideas.

## Immediate decisions
- **Preserve FP32 RMSNorm output initially.** Direct FP16 output changes more than one cast: bitsandbytes returns its input dtype, and PEFT's LoRA dropout consumes the norm output.
- **Perform streamed-head differentiation in backward**, with the incoming scalar gradient applied before conversion to FP16 dlogits. Precomputing an unscaled FP16 hidden gradient in forward and scaling afterwards can destroy the protection of GradScaler.
- **Correct prior precision assumption:** Accelerate converts the model's returned FP16 logits to FP32. TRL/rewind should therefore see FP32, although the head GEMM itself emits FP16. Soup's optional chunk helper computes metrics inside that boundary and is not an exact baseline oracle.
- First implement ordinary fp16 head matmul + independent Triton reductions, not a bespoke huge GEMM. Recomputed-head backward costs another head GEMM: measure total command, not just memory reduction.

## Source/version authority
The [official notebook](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/templates/kaggle_notebook.ipynb) **explicitly pins**, not merely suggests: soup-cli 0.77.0, transformers 5.16.1, peft 0.20.0, trl 0.29.1, bitsandbytes 0.50.2, accelerate 1.14.0. Torch, Triton, datasets and pytest are not exact-version pinned there. Soup's [package requirements](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/pyproject.toml) are broader and do not override notebook pins.

Read actual run version metadata before promoting any implementation. Torch/Triton/compiler/CUDA build, model snapshot and actual autocast wrapper installation remain runtime facts. No claim of a fully locked environment.

Pinned primary sources:
- **Q** [Qwen2 implementation, Transformers 5.16.1](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2/modeling_qwen2.py); **L** [loss](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/loss/loss_utils.py); **H** [Trainer](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/trainer.py). Commit 2026-08-26.
- **A** [Accelerator](https://github.com/huggingface/accelerate/blob/beb0672aa8444ea7647aee056f624effe5996346/src/accelerate/accelerator.py), [output conversion](https://github.com/huggingface/accelerate/blob/beb0672aa8444ea7647aee056f624effe5996346/src/accelerate/utils/operations.py), [optimizer wrapper](https://github.com/huggingface/accelerate/blob/beb0672aa8444ea7647aee056f624effe5996346/src/accelerate/optimizer.py). Commit 2026-06-11.
- **P** [PEFT preparation](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/utils/other.py), [LoRA 4-bit forward](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/lora/bnb.py). Commit 2026-07-28.
- **B** [bitsandbytes Linear4bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/nn/modules.py). Commit 2026-08-27.
- **T** [TRL trainer](https://github.com/huggingface/trl/blob/617406a4e69551c115c145114497999618d24f3c/trl/trainer/sft_trainer.py), [entropy utility](https://github.com/huggingface/trl/blob/617406a4e69551c115c145114497999618d24f3c/trl/trainer/utils.py). Commit 2026-03-20.
- **S** [Soup SFT](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/trainer/sft.py), [schema defaults](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/config/schema.py), [mixed precision](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/utils/mixed_precision.py), [chunk protocol](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/utils/stream_loss_chunk.py), [rewind](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/trainer/rewind_hf.py). Commit 2026-10-26.
- **C** [official model configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json), read 2026-10-30; mutable main, record downloaded revision at runtime.

## Baseline dataflow and shapes
Source-derived expected path, to confirm with one instrumented T4 fixture:
1. Fixed batch B=4, dynamically padded length T<=512; no changed row order, packing or step count. H=1536, V=151936, MLP=8960, layers=28, query heads=12, KV heads=2, head dim=128, RMS epsilon=1e-6; tied head/embedding (C).
2. PEFT freezes base and upcasts non-quantized half/bfloat tensors to FP32 (P). Embedding/residual/norm weights are expected FP32. Frozen NF4 weights use fp16 compute. B Linear4bit saves input dtype, casts compute input, and casts result back to input dtype. PEFT q/v branch adds dropout→A→B→scale to that base result; don't erase these boundaries.
3. Final norm gives hidden [B,T,1536] FP32. Autocast head uses FP16 operands, produces z16 [B,T,V]. L converts z16 to FP32 for CE. A recursively converts returned logits to FP32 before T/rewind inspect them. Thus three different hook positions can expose different dtypes.
4. Target at position t is labels[t+1]; final target is -100. CE is FP32 sum divided by supplied num_items_in_batch, otherwise mean. H counts causal targets from labels[...,1:] when applicable. Never divide each chunk by its own label count.
5. Entropy is FP32 log_softmax→exp→product→sum over V, for *all* attention-masked positions, including prompt and final positions. T weights by attention_mask and its sum. Accuracy is first-index argmax at shifted supervised positions, divided by supervised count; zero count yields accuracy zero.
6. Rewind uses per-row shifted token CE, divides by that row's count clamped to at least one, and records original row association. Unsupervised row gives (0,0). Entire all-ignored mean-loss batch is not automatically zero: preserve baseline NaN behavior rather than silently redefining loss.
7. A scales loss before backward. FP32 CE gradients cross the z16 cast into FP16, then head backward matmul returns FP16 activation gradient which casts back to original FP32 hidden. Remaining decoder gradients flow to LoRA; base/head/norm weights stay frozen.
8. H clips gradients through A (unscale first), then A's optimizer wrapper uses scaler.step/update; overflow can skip optimizer and scheduler. Keep all of that untouched. Soup defaults: AdamW torch, decay .01, max norm 1, cosine; task LR 2e-4, accumulation 1, LoRA dropout .05, r16, alpha32, q/v only (S).

Expected adapter shapes per layer: q A=[16,1536], q B=[1536,16], v A=[16,1536], v B=[256,16], total 2,179,072 adapter elements across 28 layers. Verify actual adapter keys/shapes; do not generate parameters from this estimate.

At T=512: hidden FP32=12 MiB; full logits FP16=593.5 MiB and FP32=1187 MiB EACH; head FP16 cache=445.125 MiB. A flattened 128-token logit chunk is 37.094 MiB FP16. These are tensor-byte arithmetic, not predicted peak VRAM or measured savings.

## Independent streamed-head design (conditional, not implemented)
**Forward**
- Narrow hook on real frozen Qwen2 causal head; preserve original fallback for unlabelled forward/generation, unexpected models, trainable heads, fake regression-test modules and unsupported output modes.
- Retain original hidden tensor and unchanged labels/attention metadata. Process a fixed, small number of token-chunk variants; no data reordering. Use baseline-equivalent autocast fp16 matmul.
- An original Triton reduction computes row max, sum exp(z-max), sum exp(z-max)*z, target z and lowest-index argmax. Produce CE=LSE-target and entropy=LSE-weighted_sum/sum; use FP32 math on **fp16-rounded logits**, not pre-rounding GEMM accumulators.
- For large V, reduce vocabulary tiles into partial states, then merge stably. Never allocate a next-power-of-two 262144-element vector per row without checking register/resource cost. A two-pass exp/logsoftmax-like variant may match baseline rounding more closely than the algebraic entropy identity; compare before promotion.
- Save compact per-token LSE or recompute it; store per-token CE/entropy and integer hit/counts. Supply metrics once, with the same names/modes and rewind protocol; no giant dummy logits to placate TRL.

**Backward**
- Receive grad_loss from autograd (includes GradScaler). Recompute each fp16-rounded head-logit chunk. Construct ((softmax-onehot)/global_denominator)*grad_loss in FP32 in the baseline order as closely as possible; ignored labels exactly zero. Cast at the baseline fp16-logits-gradient boundary.
- Compute chunk dHidden16 = dLogits16 @ head16 with ordinary matmul initially; cast to FP32 and place into corresponding slice. Return only hidden gradient; no frozen head gradient.
- Do not store a pre-rounded unscaled FP16 dHidden in forward and multiply by scaler later. Powers-of-two scaling does not recover values already underflowed.
- Head cache, if used, must match baseline autocast values and version/device/dtype; account for its persistent memory. No cache imported across measured runs.
- Leave optimizer, scaler, clipping, scheduler and trainer settings untouched. Head recomputation adds compute, so this is memory-first; a forward-precompute alternative needs an explicit scaled-gradient proof and tests.

**Integration**
Soup's optional chunked-head feature requires layer streaming in its config; **do not enable that feature/switch**. Its _soup_chunked_step/ChunkedStep data contract is usable as an integration reference, but develop a narrow solution patch and preserve TRL outputs without changing checked settings. Rewind wrapper must remain outer enough to see the current step, never a stale previous step.

## Cast-aware RMSNorm design (conditional, not implemented)
Stage 1: independently written one-row Triton fwd/bwd, width 1536, epsilon from instance, FP32 reduction, FP32 output on baseline FP32 input. Preserve Q's order: cast input to FP32 → mean(square) → rsqrt(eps+mean) → normalize → cast normalized value back to input dtype → multiply FP32 weight. For non-FP32 input, the cast before affine matters.

For frozen FP32 weight and FP32 input, analytic backward uses g=dy*w and dx=r*g-x*r^3*mean(g*x), but operation/reduction order must be validated against autograd. No dWeight needed only after checking requires_grad=False. Save x and inverse RMS; do not mutate x.

Stage 2: fuse residual addition into following norm only if both the rounded FP32 residual (needed by bypass) and norm output are returned. Backward combines norm contribution with bypass gradient and routes to both original add inputs; no alias overwrite of saved tensors. Keep separate implementation until branch-gradient and lifetime tests pass.

Do not emit FP16 merely because next GEMM uses fp16: LoRA dropout executes before its A GEMM, and Linear4bit uses original input dtype for return. Also final/head and q/v consumers differ. Any later producer-consumer fusion needs a separate precision proof per call site.

## Validation fixtures and stopping gates
[Soup official chunk tests](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/tests/test_stream_loss_chunk.py) provide *behavioral* fixture ideas (shift, denominator, ignore labels, rewind, no-grad, frozen head, cleanup), but their principal unit dtype coverage is FP32/BF16, not evidence of fp16 GradScaler correctness. Write original tests in solution/; don't edit supplied tests.

| Fixture | Required comparison |
|---|---|
| Prepared-model boundary | Record hidden/head/raw output/trainer output dtypes, actual wrappers, loss denominator, scaler scale, adapter dtype. |
| Head tails | T=1/7/31/127/128/129/511/512, chunk boundary crossing, strided inputs; no token lost/duplicated. |
| Label edge cases | First/final positions, ignored rows, all ignored, varying assistant lengths; exact counts and original row mapping. |
| Metrics | Prompt/final entropy, equal-logit argmax ties, extreme finite logits, masked padding; same train/eval metric behavior. |
| Scaled backward | Upstream gradients 1, small/large powers of 2, non-power scalar; genuine GradScaler finite and overflow steps; compare LoRA gradients, clipped norm, optimizer moments/parameters and scheduler advancement. |
| Norm | FP32 baseline plus supported fp16 fallback, width tails, tiny/large magnitudes, noncontiguous rows, frozen/trainable fallback, epsilon; add+bypass backward with independent gradients. |
| End to end | Unchanged regression tests and full cold-cache harness, first/mean loss and held-out loss, entropy/accuracy/rewind, own-kernel >=5%, time/memory limits. |

Unknown until T4 profiling: exact Torch/Triton versions and reduction implementations, whether head recomputation beats memory pressure, useful chunk size, cache lifetime, GEMM shape-dependent rounding, norm fraction, own-kernel share after acceleration. Select tuning from legal shape metadata; no hidden rows, benchmark-mode branches, graphs or artificial kernel work.

