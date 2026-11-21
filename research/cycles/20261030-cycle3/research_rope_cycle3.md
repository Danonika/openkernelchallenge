# Soup cycle 3 — Qwen2 half-split RoPE application, not QK normalization

## Decision first

**Refine existing `attn-sgl-rounding-fusion`, variant `qwen2-fp32-half-split-apply-only`; do not add a row.** An original, out-of-place FP32 Q/K rotary-application forward/backward kernel is a plausible P2 research candidate. It replaces only the existing pointwise application, not frequency generation, projections, SDPA, cache semantics, or the architecture. This is a launch/transient-traffic hypothesis, not an established speedup or activation-retention saving.

The generic catalog sources are not drop-in Qwen2 recipes: this model has **no QK norm**, uses **half-split rather than adjacent/interleaved rotary pairs**, and the Soup preparation path is expected to present FP32 Q/K and coefficients despite FP16 quantized compute. Preserve those differences. `trt_qknorm_rope_fusion` and `attn-vllm-vector-prep` receive the same transfer-constraint correction, not separate candidate claims.

Three RoPE-specific CPU fixture groups passed, including 32 FP32 paired Q/K cases. These are semantic checks, not challenge or performance passes. No GPU, full model, installation, pretrained download, official-source modification, or performance measurement was performed.

## 1. Pinned source contract

### Source identity and shape

- Official challenge: [`c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`](https://example.invalid/redacted-github).
- Transformers 5.16.1: [`93c8b7b485963a10800c91f55304db6be211c2bd`, Qwen2 model](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2/modeling_qwen2.py). Git blob `b4fed98fafae8e4ee39be3b8cd766b1c85b079cd`, matching the locally installed source bytes in the routing probe.
- The [Qwen2.5-1.5B-Instruct model config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json) gives hidden width 1536, 12 query heads, 2 KV heads, 28 layers, and theta 1,000,000; head dimension is 128. This model-config URL is an unpinned upstream reference, not a claimed immutable snapshot. The fixed-task geometry is independently recorded in the existing Soup audit.
- Fixed task B=4, dynamic padded T<=512, M=B*T<=2048. Q is `[B,12,T,128]`, K is `[B,2,T,128]`; projection outputs are viewed as BTHD then transposed. These are generally **not contiguous BHTD**. [Projection/application/cache order, lines 180–233](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2/modeling_qwen2.py#L180-L233).

### Coefficients, broadcasting, and lifetime

[Rotary embedding lines 51–102](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2/modeling_qwen2.py#L51-L102) construct default inverse frequencies in FP32. Inside a no-grad, autocast-disabled region, FP32 frequency/position multiplication produces frequencies; concatenating the same half twice precedes cosine/sine and attention scaling. Only the final coefficients are converted to `x.dtype`. Default attention scaling is 1. Preserve this implementation; no trigonometric approximation, coefficient regeneration, position rewrite, or altered theta is proposed.

The model creates position embeddings **once per model forward**, before the decoder-layer loop, then passes the same pair to all 28 layers. Thus the RoPE application is repeated, but frequency/trigonometric generation is not repeated per layer. The default implementation does not construct a persistent cos/sin table cache here. `max_seq_len_cached` and the dynamic wrapper do not justify assuming that such a table exists. Dynamic/longrope update paths are outside this candidate; preserve the default path and fall back if configuration differs. [Model forward lines 355–397](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2/modeling_qwen2.py#L355-L397), [dynamic wrapper](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/modeling_rope_utils.py).

Coefficients initially have `[Bpos,T,D]`, normally `Bpos=1` for generated positions; explicit batched positions may give Bpos=B. `apply_rotary_pos_emb` unsqueezes dimension 1 by default, broadcasting across heads. Its generic dimension-2 route serves BTHD, but the proposed narrow Soup kernel need only support the actual dimension-1 BHTD route; unsupported axes fall back.

### Source-expected dtype, not a measured Kaggle claim

The earlier audit establishes the relevant composition: PEFT preparation upcasts nonquantized parameters; the embedding/norm/residual stream is expected FP32; bitsandbytes 4-bit linear returns the input dtype; the q adapter adds its FP16 result to the FP32 base. Therefore Q, K, and model-generated cos/sin are expected FP32. Crucially, the q sum can contain FP32 values that are **not** exactly representable in FP16. A model-config `torch_dtype` label or FP16 internal GEMM does not justify an early half cast.

Sources: [PEFT k-bit preparation](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/utils/other.py), [PEFT quantized LoRA forward](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/lora/bnb.py#L513-L565), [bitsandbytes linear implementation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/nn/modules.py). Actual Kaggle dispatch/dtypes remain a mandatory observation gate.

## 2. Exact operation graph and backward

[The source application, lines 105–135](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2/modeling_qwen2.py#L105-L135), splits a head in half. For one corresponding pair, let `x1=x[...,j]`, `x2=x[...,j+64]` and likewise split coefficients. The forward graph is:

```
p1 = float32(x1 * c1)
p2 = float32((-x2) * s1)
y1 = float32(p1 + p2)
p3 = float32(x2 * c2)
p4 = float32(x1 * s2)
y2 = float32(p3 + p4)
```

This is not adjacent-pair rotation. It is also not permission to contract multiply-plus-add or to replace the first addition with a differently ordered subtract. Keep multiplication and addition rounding separate; signed-zero and nonfinite behavior must be checked in the actual compiled path. A CPU FP64 simulation of a contraction counterexample is useful evidence that contraction can differ, not a compiler proof.

For coefficient tensors that do not require gradients, the mathematical adjoint for each independent Q or K is:

```
dx1 = g1*c1 + g2*s2
dx2 = g2*c2 - g1*s1
```

Equivalently `dx = g*cos - rotate_half(g*sin)`. The commonly guessed `g*cos - rotate_half(g)*sin` is not the general adjoint: it uses the wrong cross-half sine coefficients when halves differ. Actual Qwen2 generation duplicates halves, but the generic application accepts tensors, so either implement the general pair formula or prove/guard coefficient structure. Preserve the eager graph's product/negation/add accumulation behavior, not merely the real-number equation.

The narrow kernel requires `cos.requires_grad == sin.requires_grad == False`. Otherwise it falls back; computing coefficient gradients requires additional reductions and is a different contract. Q and K may independently require gradients, and an output may be unused. Respect these cases rather than forcing both backward paths or retaining large Q/K activations. GradScaler's incoming scale is already in the gradient: this kernel neither unscales nor clips it.

### Saved-state correction

The source derivative needs coefficient values, not Q/K values, when coefficients are no-grad. The eager graph can save multiple references/views of the same cos/sin storages. A custom function saving the pair once reduces reference count, not necessarily unique storage bytes. **No retained-Q/K or 28-fold coefficient-copy saving is claimed.** The CPU storage-inventory fixture checks this directly. Saving shared coefficients with `save_for_backward` plus small metadata is sufficient for the candidate's first-order backward; do not clone them or retain Q/K just to implement rotation.

Higher-order differentiation, unusual subclass behavior, overlapping/unsupported layouts, mixed dtype, and non-default configuration remain fallback cases until explicitly supported. The target training task only requires first-order training, but silently changing an exposed generic function's contract is avoidable.

## 3. Original T4 route and important controls

Proposed narrow dispatch guards: CUDA, FP32 Q/K/cos/sin, D=128, Hq=12/Hk=2, default unsqueeze axis, valid Bpos broadcast, recognized dense-transposed projection layout and coefficient layout, no coefficient gradients, expected out-of-place alias contract. Unsupported calls use the captured original function. Sequence length remains dynamic, including tails. No persistent cache of training activations, frequency tables, or compiled artifacts is introduced.

A feasible original mapping is a token/combined-QK-head tile, with 64 half-pairs per head and both halves loaded into registers. Each output pair is independent; backward requires no atomics or reduction when coefficients are fixed. Compare combined Q/K dispatch with separate-Q/separate-K dispatch: fewer launches may not offset extra pointer-selection and tail handling. A small multi-head tile can share coefficient loads, but actual reuse/register pressure must be inspected. This is ordinary FP32 elementwise work and does not require tensor cores, FP8, BF16, asynchronous-copy instructions, Hopper clusters, or a ready-made kernel runtime.

Do not assume contiguous output. Eager TensorIterator can preserve the projection's dense transposed stride. Preserve the observed output layout or establish that any layout change leaves the downstream SDPA path and total cost unchanged; do not count a hidden reorder as free. The CPU fixture observes actual eager strides, but CUDA must be checked separately.

Triton's inspected NVIDIA backend has `enable_fp_fusion=True` by default. An explicit no-contraction setting is a starting implementation control, **not a correctness certificate**; inspect the generated SM75 instructions and compare adversarial cases. The official notebook does not pin Triton, so verify the runtime option/version rather than assuming this source is installed. [Pinned backend source](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/third_party/nvidia/backend/compiler.py#L120-L132).

### Do not fuse through SDPA or the cache yet

The attention method updates `past_key_values` using rotated keys **before** invoking the attention interface. Retaining the original FP32 output makes the narrow application independent of cache on/off; fusing an early FP16 store into an assumed SDPA path does not. The [SDPA integration](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/integrations/sdpa_attention.py) chooses GQA/repeated KV behavior and invokes `scaled_dot_product_attention`; there is no explicit blanket FP16 conversion at that Python boundary. Keep its masks, scaling, dropout, backend selection, and cache observations untouched. No fused RoPE→SDPA cast variant is promoted.

Nor is projection/LoRA/RoPE fusion accepted merely because all are nearby: the q branch has a rounded mixed-precision sum, K has different head count and no q/v adapter symmetry, base projections include bias, and a larger custom-autograd region broadens proof obligations. This cycle chooses the smaller operation boundary.

### Routing is a separate gate

Qwen2 uses `use_kernel_forward_from_hub` / `use_kernelized_func` decorators. The inspected [hub integration](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/integrations/hub_kernels.py) has identity paths when the optional kernels package is absent and lazy opt-in hub mappings when present. Do not install/enable a ready-made hub runtime. The local route probe verifies a temporarily replaced module-global `apply_rotary_pos_emb` is reached exactly once by a tiny CPU `Qwen2Attention`, then restores the function. That confirms this environment's route, not every optional decorator state or Kaggle configuration. Future GPU work must verify the intended call is actually intercepted and unsupported routes remain original.

## 4. Activation-fusion negative result

The actual [Qwen2 MLP, lines 35–48](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2/modeling_qwen2.py#L35-L48), computes independent gate/up projections, `act_fn(gate) * up`, then down projection. In the task the activation is SiLU. Plain forward/backward SiLU×up fusion is **already the official example** and is represented by existing packed-SwiGLU ideas; rewriting or renaming the example does not earn own-kernel share. [Official solution entry](https://example.invalid/redacted-github), [explicit example exclusion](https://example.invalid/redacted-github).

Thus no new activation row, copied example, packed-gate/up weight transformation, or unmeasured ownership claim is proposed. Any later distinct MLP fusion needs a mechanism beyond that example plus preservation of frozen NF4 projection semantics and backward; existing `vllm_packed_swiglu`, `xformers_packed_swiglu`, and `liger_scalar_promotion` remain the relevant catalog entries.

## 5. Geometry is not a speed or peak-memory result

At maximum M=2048, Q+K contain `2048*(12+2)*128` FP32 elements, i.e. 14 MiB. Merely reading and writing them once transfers 28 MiB of Q/K payload, excluding coefficients, gradients, layout effects, caches, and launch overhead. An unfused source expression has intermediate negation/concatenation/products/additions; fusion can remove some launches and transient tensors. Actual eager/compiled CUDA dispatch must be measured before assigning those intermediates a measured byte or latency cost.

For ordinary Bpos=1,T=512, the cos/sin pair occupies 0.5 MiB of FP32 payload; with Bpos=4 it is 2 MiB. The pair is shared across layers. A reference-count reduction does not multiply this payload by 28 or establish a lower allocated peak. No time, memory-coordinate, percent gain, own-share percentage, or Pareto point is assigned.

## 6. Evidence and closure gate

- Tiny routing probe: `rope_cycle3_route_probe.py` / `rope_cycle3_route_probe.json`. CPU Torch `2.15.0+cpu`, git `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`; Transformers 5.16.1 installed Qwen2 blob matches pinned source. Optional kernels package absent; tiny eager attention only, not pretrained/full model. One intercepted call, restored patch, finite input gradient. No performance claim.
- Independent semantic CPU report: [`cpu_cycle3_results.md`](cpu_cycle3_results.md); full [`cpu_tests/cycle3/results.json`](cpu_tests/cycle3/results.json), [`summary.json`](cpu_tests/cycle3/summary.json), original [`run_cycle3.py`](cpu_tests/cycle3/run_cycle3.py), and [`run.sh`](cpu_tests/cycle3/run.sh). The shared cycle contains 6 groups total, of which 3 are RoPE-specific; do not add the routing probe or the 32 cases to the fixture-group count. Prior 17 groups remain immutable and are hash-verified by the CPU lane.
- **32 FP32 paired Q/K cases passed exact finite-value and input-gradient equality**, including Hq/Hk=12/2, D=128, B=2,T=7, transposed and alternate-axis layouts, Bpos=1/2, duplicated and unequal coefficient halves, independently trainable/unused inputs. This establishes the toy operation contract, not actual model/GPU routing.
- Reference and custom source-like Q outputs preserve stride `[10752,128,1536,1]` for shape `[2,12,7,128]`. Coefficient inventory is **4 native saved references / 2 unique storages versus 2 custom references / the same 2 storages**, 7168 referenced bytes in this fixture. No Q/K retained-storage improvement or peak-memory improvement is established.
- Wrong interleaved rotation and generic inverse-style adjoints fail as expected. One-product contraction simulation using ordinary duplicated trig coefficients differs in **1784/4096 outputs**. Witness: x1=-3.098583459854126, x2=-2.299267292022705, cos=0.8253356218338013, sin=0.5646424889564514; separate FP32 result=-1.2591073513031006, contracted=-1.259107232093811. No actual GPU FMA was run, and a difference is not a demonstrated challenge-tolerance failure.
- Explicit all-FP16, mixed FP16-input/FP32-coefficient, and trainable-coefficient controls route to reference. The mixed case yields FP32 output and FP16 input gradients. No custom mixed-dtype equality is claimed.
- **Untested limits:** RoPE nonfinite inputs/gradients, signed-zero or NaN-payload bit identity, actual runtime dtype/stride dispatch, CUDA compiler/reduction behavior, allocator peak, end-to-end quality/performance and own-kernel share. These remain gates; do not label the finite toy results a full exact implementation.

**Next decisive evidence, once authorized GPU access exists:** (1) observe exact baseline call route, dtype, input/output strides, coefficient aliasing and `requires_grad`, cache mode, and SDPA dispatch on the fixed task; (2) implement only the guarded original FP32 half-split application, inspect no-FMA SM75 codegen, and compare adversarial plus realistic forward/backward including signed-zero/nonfinite cases; (3) compare total forward+backward, allocation peak, layout/SDPA overhead and cold whole-command time against unchanged baseline, separate-Q/K and combined variants; (4) run official correctness/quality/regression checks and measure own-kernel share. Reject or demote if route/layout/rounding changes, fallback dominates, end-to-end cost is not lower, or this operation is too small to matter. CPU parity alone cannot pass any of these GPU/performance gates.
