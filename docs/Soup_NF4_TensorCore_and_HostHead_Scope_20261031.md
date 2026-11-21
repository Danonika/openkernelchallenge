# Two architectural paths beyond the 58.78s / 5,005MiB T4 frontier

**Date:** 2026-10-31. Research only. The developer queue's baseline is saved, not paired. The fast 58.78s source-family *failed* an actual useful-original-Triton share gate at 1.967%, below 5%.

## Correctness-first original NF4→FP16 Tensor Core GEMM

Source-based:
- T4 SM75 has `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` (FP16 inputs, FP32 accumulation).
- NF4 is a nonlinear public 16-value codebook, **not signed integer INT4**. Two NF4 values per packed byte; separate FP32 nested scale multiply and offset before the final NF4 multiply/FP16 cast.
- Qwen2.5-1.5B fixed decoder dims have K=1536, MLP width8960, and quant block size64. The critical K is divisible by 64 and 8; do not generalize to odd-K without tests.

Original kernel research sequence:
1. Decode a 64xK tile of packed NF4 weight directly into aligned FP16 fragments in shared memory; independently check dequantization bit patterns against the pinned native bitsandbytes 0.50.2 on T4 for real `QuantState`.
2. Issue original inline PTX FP16 `mma.sync.m16n8k8` for actual useful GEMM, with the original input autocast boundaries and output FP16 rounding. Test forward and d(input) at native target dimensions; preserve native fallback on unsupported layouts.
3. Compare CUDA Forward dX and adapter LoRA gradients and over 64 exact updates. No stochastic dropout RNG or original batch/optimizer changes.
4. Measure training time, true peak, and countable `@triton.jit` share from step3 onward. No ready kernel packages or disguised starter kernels. Ensure >5% *in every repetition*, not only in a microbenchmark.

A local **CPU-only** reproducible mathematical oracle was executed on five decoder-like shapes with two independently written NF4 decode implementations. It passed bit-exact FP16 dequant and finite GEMM checks; largest CPU forward absolute difference was 6.103515625e-5. **Not CUDA parity.** The original code and JSON artifact are available in the conversation.

## Possible memory-bound host-resident tied output head

The total retained tied FP16 head is 151936×1536×2 bytes = **445.125 MiB**. If the head is on pinned host memory and loaded in fixed vocabulary tiles only when needed, the static GPU weight floor decreases by that amount. A CPU-side indexed embedding gather can return only the required input rows while preserving the original FP32 decoder embeddings, provided it has identical token order, exception behavior, and output dtype.

For gradient correctness and all-output semantics, the full vocabulary remains covered (entropy for every attention-live row, main CE for causally supervised rows, argmax first-index ties, ignored-row nonfinite behavior). The frozen head must still contribute the correct d(hidden); moving only head weights and no computations is insufficient.

Idealized communication cost for **one full output-head read in forward and one in backward each step**:

- 445.125 MiB × 2 × 64 = **55.64 GiB** over 64 training steps.
- T4 has PCIe 3.0 x16; theoretical ~15.75GB/s one-way. Minimum transfer time is **approximately 3.8s** (before launch, CPU, GPU, page pinning, recurrent casting and all other work). Actual speed could be far lower.
- With token-chunk recomputation, more head reads may be necessary; upper traffic grows accordingly. Double buffering requires its own 8–64MiB slots and truthful peak measurement. No cached device weights may escape the intended lifetime.
- This is not a tested sub-1359MiB implementation, and aggressive host streaming can slow training badly (an earlier generic activation CPU-offload candidate hit the 80s cutoff at a partial 3607MiB).

**Do not enable Soup's protected `stream_layers`, `gradient_checkpointing`, `activation_offloading` or `use_cut_ce` switches; only an explicitly allowed custom solution patch is eligible.** Do not modify model/data/train configuration or required outputs.

## Bounded admission work already in Sheets

- Row440: `chatgpt-cache1g-validnorm-maskhead-share-20261031-v1`: finite padded head + 1GiB cache + valid-row norm + original fused residual; after unchanged measured run fail closed unless useful own kernels ≥5%.
- Row480: `chatgpt-lowmem5005-ownshare79-20261031-v1`: exact 5005MiB 66.33s source with postrun 5% attribution and all 79 Soup tests through test-only offline pytest bundle.
- Both input rows are Ready=TRUE but **no VM job ID was recorded at last check**. The VM Queue status had its last poll at 2026-10-31T10:27Z, and there were no current running/queued jobs in its last snapshot. This is an intake/update blocker, not a GPU result. Do not edit those accepted-input rows or invent metrics.

## Admission math and timing target

The pinned parameter floor is 624.750MiB packed decoder NF4 + 445.125MiB tied FP16 embedding + 33.250MiB q/v LoRA weights, gradients, Adam slots = **1103.125MiB** (excludes metadata/runtime/activation). With 143MiB of *indicative smoke* context, only **112.875MiB** remains under the requested 1359MiB. This is a conditional storage calculation, not an impossibility proof.

At 58.78s full train command, the observed training-loop portion was 41.07s (non-loop17.71s). If overhead stays fixed, <45s needs <27.29s training loop (about 33.6% faster). Setup improvements and training compute optimizations can both contribute, but any cost shifted outside measurement or hidden work is prohibited.

### Primary evidence and references
- [Official competition correctness / kernel share](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/main/harness/check_rules.py).
- [Official scoring and 5% tie threshold](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/main/harness/score.py).
- [NVIDIA PTX SM75 FP16 MMA](https://docs.nvidia.com/cuda/archive/11.2.0/parallel-thread-execution/index.html).
- [NVIDIA T4 PCIe3x16 / FP16 Tensor Core specs](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf).
- [bitsandbytes public NF4 lookup representation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py).
- [Google Sheets T4 development queue](https://example.invalid/redacted).
