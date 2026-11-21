# Soup Kernel Challenge — Research Round 2: toward <45 s and <1,359 MiB

**Date:** 31 October 2026, Astana time  
**Scope:** one Tesla T4, fixed Soup 0.77.0/Qwen2.5-1.5B-Instruct NF4/QV-LoRA task, 64 training steps. **Public VM development results**, compared against a saved baseline, are not official paired hidden-row jury results. Memory figures use **MiB**, following the VM queue; the user's “1,359 MB” is interpreted here as 1,359 MiB.

## Verified results from the connected Google Sheets queue

Baseline: **71.85 s / 14,345 MiB / eval loss 2.04968**.

| Candidate / Request ID | Completed wall time | Peak VRAM | Eval loss | Development checks |
|---|---:|---:|---:|---|
| Full NF4 weight cache + MLP pad + retained FP16 — `chatgpt-fullcache-mlppad-retain-20261031-v2` | **60.24 s** | 8,827 MiB | 2.04969 | time/loss/VRAM pass |
| Full cache + valid-row RMSNorm + fused residual — `chatgpt-fullcache-validnorm-residual-20261031-v1` | 60.52 s | 8,111 MiB | 2.04965 | pass/pass/pass |
| Full cache + fused MLP residual — `chatgpt-fullcache-mlp-residual-20261031-v1` | 60.60 s | 8,071 MiB | 2.04966 | pass/pass/pass |
| **1 GiB NF4 cache** + MLP padding — `chatgpt-nf4-cache1g-mlppad-20261031-v2` | **62.77 s** | **6,721 MiB** | 2.04966 | pass/pass/pass |
| No full cache + fused MLP residual — `chatgpt-mlp-fused-residual-20261031-v3` | 64.03 s | 5,405 MiB | 2.04960 | pass/pass/pass |
| **Lowest completed VRAM**: MLP padding + fused scatter — `chatgpt-mlp-padding-fused-scatter-20261031-v2` | 64.83 s | **5,345 MiB** | 2.04965 | pass/pass/pass |
| CPU saved-activation offload — `chatgpt-cpu-saved-activation-20261031-v1` | **80-second cutoff: incomplete** | 3,607 MiB observed before termination | **not evaluated** | FAIL: TIME_LIMIT_EXCEEDED |

The CPU-offload VRAM is a **partial-window** peak in a failed job, **not** a completed valid solution. The 1 GiB budget experiment previously failed from a notebook construction error; a new, corrected notebook was submitted and completed in 62.77 s. Do not treat the failed notebook as a GPU result.

### Performance budget

The fastest completed full-command result has a **42.03-second training loop**, leaving **18.21 seconds** for measured non-loop work. With that overhead unchanged, a <45-second command requires a loop below **26.79 seconds** — a **36.3% further loop reduction**. Microsecond-level pointwise changes alone will not achieve this reliably.

### Memory budget (optimistic floor, not a proof of impossibility)

The Qwen2.5-1.5B published architecture includes 28 layers, 1.31 billion nonembedding parameters and 151,936 × 1,536 tied embeddings. Approximating frozen nonembedding weights at 4 bits per weight gives **624.66 MiB**. A byte of nested scale per 64 weights is **19.52 MiB**. The frozen tied embedding/output head in FP16 occupies **445.13 MiB**. Q/V LoRA FP32 parameters, Adam moments and gradients total about **33.25 MiB**. Illustrative subtotal: **1,122.55 MiB** *before* other quant metadata, parameters, CUDA overhead, activations, logits and workspaces. With a separate ~143 MiB bare-GPU queue smoke observation, there would be only ~93 MiB under the requested 1,359 MiB ceiling. A 256-row FP16 vocabulary score chunk alone occupies **74.19 MiB**, while its FP32 counterpart needs **148.38 MiB**. This shows why the current head/materialization strategy cannot satisfy the target under fully resident parameters.

## Actual new research and experiments

### 1. Resident decoded-weight cache budget: measured

The full decoded-NF4 cache improves speed but holds many large frozen FP16 weight copies. The fixed 1 GiB limit variant changes only `solution/nf4.py` cache budget, preserving source/algorithm and the normal 64-step harness. Its **62.77 s / 6,721 MiB** result is a point between the 60.24 s speed extreme and 5,345 MiB memory extreme. This is an end-to-end observed trade-off, not an assumed theoretical gain.

Public source notebook: project-repository/blob/master/soup-kernel-challenge/queue_20261031_nf4_cache_1g_mlp_padding_v2.ipynb

### 2. Custom saved-activation CPU offload: experimentally rejected at this setting

A narrow `torch.autograd.graph.saved_tensors_hooks` wrapper around Qwen2 decoder forward moved only non-leaf CUDA FP16/FP32 saved activations >=8 MiB to CPU, restoring their exact data on backward. A deterministic CPU toy compared outputs, gradients, dropout RNG across an autograd hook and found bit-exact agreement. In the actual GPU queue it hit the server's **80 s hard cutoff**; no eval loss was produced. The sampled **3,607 MiB** does not validate the method as a finished contest candidate. A more restricted selected-layer/threshold variant could be studied, but any time saving is unproven. It is distinct from turning on the forbidden Soup `activation_offloading` switch.

Source notebook: project-repository/blob/master/soup-kernel-challenge/queue_20261031_storage_mlp_pad_cpu_saved_offload_v1.ipynb

### 3. Output-head masked-row compaction: queued for T4

The ordinary streamed head still performs vocabulary projection on all padded positions. The prototype retains each output row where `attention_mask!=0` **or its causally shifted target is supervised**, and packs those rows (with a dummy terminal row) through the same `streamed_linear_ce`. It scatters full-shaped per-position loss/entropy/predictions back before TRL and Rewind. A conservative bound checks skipped rows for nonfinite or FP16-overflow risk and falls back to the full original head when unsafe. It preserves the original global denominator and all supervised gradients. **100 randomized finite CPU dense-versus-compact tests passed**, including gradients; this does not establish actual Qwen/T4 quality or runtime.

Current accepted queue Request ID: **`chatgpt-finite-padded-head-20261031-v2`** (a previous row was overwritten before intake; the immutable notebook was requeued at a distant row).

Source notebook: project-repository/blob/master/soup-kernel-challenge/queue_20261031_finite_padded_head_v1.ipynb

### 4. Whole-model variable-length padding removal: CPU attention proof only

For each unchanged batch, packed real tokens can be concatenated in original within-example order with a strict block-diagonal causal attention mask and original position IDs, provided no attention is permitted across examples. A stand-alone **100-seed CPU test** for `scaled_dot_product_attention` and its input gradients matched the dense padded reference. This is **not** validation of actual Qwen+PEFT: q/v LoRA dropout must still draw its RNG on the original padded geometry before selecting packed rows, and the full-model rounding/metrics/exception contracts must be checked. The existing queue already contains a prior q-projection unpadding result with hard LoRA-gradient failures, so a whole-model implementation is high risk. A shape-based guard should avoid cases where packed S² attention becomes more expensive than B·T² dense attention. For B=4, packing can be computationally favorable only for total active S < 2T, absent more efficient varlen kernels.

CPU reference scripts available in this conversation:
- `queuework/check_mask_head.py`
- `queuework/attention_pack_proof.py`

### 5. Fused NF4 + Turing Tensor Core research direction

**NF4 is a nonlinear 16-value lookup codebook, not linear signed INT4.** Therefore feeding its raw four-bit indices directly into INT4 Tensor Core dot products would *not* reproduce the QLoRA weights. A compliant custom fused kernel must first reconstruct exact scaled NF4 values (nested FP32 scale multiply and offset, codebook lookup, FP16 round), then feed FP16 fragments to Turing `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`. SM75 lacks later SM80-specific conveniences. A practical design must compare on-the-fly decode redundancy across M tiles versus resident cache reuse, and implement the input-gradient GEMM separately. This has **not** yet been implemented or tested as a full NF4 GEMM; it is a research priority, not a speed claim.

## Why the simultaneous 45 s / 1,359 MiB goal remains difficult

Caching dense decoded weights is good for speed but uses ~2–3 GiB VRAM; CPU offloading saves GPU storage but costs PCIe transfers and synchronizations; checkpoint/recomputation trades time for activation memory. A plausible memory-first architecture would keep a small amount of GPU scratch space, stream frozen NF4/tied-vocabulary weights from CPU/pinned host storage, fuse row-local kernels and use a truly tiled CE/head, while recomputing/offloading saved activations. A speed-first architecture would keep high-use decoded weights in GPU memory and fuse more of Q/K/V/O and MLP work. No evidence currently establishes a single candidate capable of both targets.

## Competition/measurement caveats

The Google Sheets VM uses a **saved public baseline**; all numbers above are single-solution development screens. Passing the queue's three badges does **not** establish the required useful original `@triton.jit` share >=5% (from step 3 onward), 79 Soup regressions, all official checks, paired baseline/solution Kaggle runs, hidden-data quality, or jury admission. The original starter SwiGLU kernels are not countable. Do not promote any candidate until source attribution and actual full checks are reviewed. Deadline: **31 October 2026, 23:59 Asia/Almaty**.

## References

- Live queue: https://example.invalid/redacted
- Official Soup rules: https://example.invalid/redacted-github
- Qwen 1.5B parameters: https://qwen.ai/blog?id=qwen2.5-llm
- Qwen configuration: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json
- NF4 codebook source: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/functional.py
- Turing FP16 MMA PTX: https://docs.nvidia.com/cuda/archive/11.2.0/parallel-thread-execution/index.html
- NVIDIA T4 throughput/PCIe: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf
- PyTorch saved tensor hooks: https://docs.pytorch.org/docs/stable/autograd.html