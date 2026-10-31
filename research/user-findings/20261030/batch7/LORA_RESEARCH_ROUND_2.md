# LoRA kernel research — round 2

Date: 2026-10-30  
Workload: Decentrathon 6.0 Soup Kernel Challenge, Qwen2.5-1.5B-Instruct, one Tesla T4  
Status: source research and analytical design; no solution implementation or T4 measurement

## Decision

**Start with original, strided metric reductions; then test the narrow LoRA expansion. Develop the full chunked output head as a separate experiment.**

The second research pass changes our priorities because the prescribed training stack contains substantial work around logits, while some apparent LoRA memory savings are already present in bitsandbytes or concern temporary buffers. The strongest low-scope lead is the explicit contiguous copy made for token accuracy. The largest structural memory opportunity remains avoiding full logits, but that requires preserving several different metrics and the exact fp16 loss-scaling order.

Use KERNEL_HANDOFF.md to resume work. This document contains the detailed reasoning and sources. “Verified” below means verified in the specified source version, not observed on the actual Kaggle runtime. Memory sizes and FLOP counts are derived estimates, not benchmark results.

## 1. The intended dependencies are known

The earlier handoff treated library versions as unresolved. The starter notebook and README explicitly prescribe this stack. [S01], [S02]

| Package | Intended version |
|---|---:|
| soup-cli | 0.77.0 |
| transformers | 5.16.1 |
| peft | 0.20.0 |
| trl | 0.29.1 |
| bitsandbytes | 0.50.2 |
| accelerate | 1.14.0 |

That install command does not explicitly pin torch, Triton, datasets, or pytest. The challenge brief expects torch 2.11 and Triton 3.6.0. Record the installed versions and resolved source commits before interpreting any benchmark.

Inspected starter commit: c7c2fcd15972ce8b898d8d3ff732ee192c817ceb.  
Inspected Soup 0.77.0 publication commit: 846888a045b31084452fa08a684a7a38c02f5f87.

The fixed task remains q_proj and v_proj LoRA, rank 16, alpha 32, dropout 0.05, batch 4, at most 512 positions per row, 64 steps, NF4 base weights, fp16 compute, and the prescribed optimizer and data order. The omitted dropout setting inherits Soup’s default; it is not zero. [S03], [S04]

## 2. Mixed precision is part of the operator contract

### The expected dtype path

PEFT’s k-bit preparation freezes base parameters and upcasts non-Params4bit fp16/bf16 parameters to fp32. Qwen’s tied embedding/head and norm parameters therefore have a strong source-based reason to reside in fp32. The model configuration ties the embedding and head; count their storage once. [S05], [S06], [S07]

bitsandbytes Linear4bit converts its input to the configured compute dtype, performs the matmul, and converts the result back to the original input dtype. An fp32 input can therefore produce an fp32 base result even when the GEMM uses fp16. PEFT runs the adapter branch under autocast and then adds it to that base result. [S08], [S09]

There is a further setup ordering issue: TRL casts trainable parameters of a quantized model to bf16 during construction. Soup subsequently converts bf16 trainable parameters to fp32 before fp16 training so gradient unscaling works. A setup hook must retain current parameter references, not cache copies made before this conversion. [S10], [S11]

**Runtime hypothesis to verify:** residual/base outputs are often fp32, adapter GEMM outputs fp16, and trainable adapter storage fp32. No dtype trace has yet confirmed the full chain.

### Consequence for LoRA fusion

For X[M,K], A[r,K], B[N,r], and r=16:

    Z = dropout(X) @ A.T
    Y = Ybase + 2 * Z @ B.T

The corresponding mathematical gradients are:

    dYbase = G
    dZ = 2 * G @ B
    dB = 2 * G.T @ Z

An implementation must also preserve the relevant rounding and conversion boundaries: the adapter GEMM result, scaling, final addition, incoming gradient conversion, and parameter-gradient storage. Feeding an unrounded fp32 dot accumulator directly into an fp32 base addition is a different numerical path from rounding the adapter result to fp16 first.

Keep the original base computation, native dropout, A projection, and defensive clone for the first experiment. Compare against the actual autocast expression. Test with nonzero B and several updates; the usual zero initialization of B can hide errors in an initial forward comparison. [S09]

## 3. A concrete first target: metric reductions over logits

The pinned TRL accuracy implementation makes the next-token slice contiguous before taking argmax. For batch 4 and sequence length 512, that slice contains 4 × 511 × 151,936 values. [S10]

Accelerate’s native AMP wrapper recursively converts returned fp16/bf16 tensors to fp32. Trainer-facing logits are therefore expected to be fp32, even if the head GEMM produced fp16. This makes the accuracy copy potentially **1,184.68 MiB** at the maximum shape. [S12], [S13]

This is an identified allocation candidate. Eliminating it does not guarantee the same reduction in whole-run peak memory: another phase may set the peak.

### The metrics do not share one mask

| Output | Required positions | Required aggregation |
|---|---|---|
| Training CE | Shifted targets where label is not -100 | Sum divided by the supplied global valid-target count; ordinary mean if no explicit count |
| Entropy | Every attention-mask-selected input position, including prompt and final positions | Token entropy sum divided by real-token count |
| Accuracy | Logits at t predicting labels at t+1, excluding ignored targets | Correct supervised predictions / supervised predictions |
| Rewind row loss | Shifted supervised targets within each original row | Per-row CE sum / row target count; zero-count row gives zero |
| Logged token count | Attention-mask-selected input positions | Cumulative count across training batches |

These contracts come from the pinned transformers loss, TRL trainer, and Soup rewind implementation. In particular, batch loss is not the unweighted mean of four row losses. [S10], [S14], [S15]

TRL already computes entropy in chunks of 128 flattened token rows. It does not allocate a probability tensor for all batch positions simultaneously. The remaining opportunity is to fuse its log-softmax/exponential/multiply/reduction work and avoid other traversals and copies. [S16]

### Proposed original reduction

Use vocabulary tiles to compute small per-token summaries directly from the existing strided logits. The final reduction can produce entropy, argmax, and a supervised target logit. Initially, keep the model loss and its backward unchanged.

For a finite logit row z, define:

\[
m=\max_j z_j,\quad
s=\sum_j e^{z_j-m},\quad
u=\sum_j e^{z_j-m}(z_j-m).
\]

Then:

\[
\mathrm{LSE}=m+\log s,\qquad
\mathrm{entropy}=\log s-u/s,\qquad
\mathrm{CE}(y)=m+\log s-z_y.
\]

For two vocabulary tiles, let m=max(m1,m2), a=exp(m1-m), b=exp(m2-m). Their summaries merge as:

\[
s=as_1+bs_2,
\]

\[
u=a[u_1+(m_1-m)s_1]+b[u_2+(m_2-m)s_2].
\]

This is an analytical design using standard stable softmax identities, not copied kernel code. Track the smallest vocabulary index when maximum values tie. Tail entries outside the vocabulary must contribute zero without introducing a zero-times-infinity NaN.

The vocabulary has 151,936 entries. A single-program approach that pads a row to 262,144 entries is a poor default for T4. Use bounded vocabulary tiles and a small second reduction; determine the tile size on hardware. Triton’s introductory fused-softmax example assumes the row fits in on-chip memory. [S17], [S18]

### Integration scope

A fused entropy helper is a narrow first integration. Removing the accuracy copy additionally needs a guarded trainer metric replacement. The trainer imports the entropy helper into its own module namespace, so changing only the original utility module after import is insufficient. Preserve train/eval metric buckets, cumulative counts, gathering behavior, output structure, and rewind row order. [S10], [S16]

A later variant can reuse per-token CE for rewind instead of computing row softmax again. Keep the recorder’s ordinary sampler/FIFO and step bookkeeping. A summary is useful only if it carries the same real information the original code would compute. [S15]

## 4. Full chunked loss: two traps to avoid

### Trap A — repeated casts of the frozen head

PyTorch 2.11’s autocast weight-cache condition includes requires_grad. A frozen fp32 head does not meet that condition. [S19]

A naïve ordinary-autograd loop that repeatedly calls F.linear on small hidden-state chunks with the same fp32 frozen head can therefore cast that entire head repeatedly. Each chunk’s backward needs the compute-dtype weight to form its hidden gradient, potentially retaining distinct cast copies.

Derived size of one fp16 head copy:

    151936 × 1536 × 2 bytes = 445.125 MiB

For sixteen 128-token chunks, sixteen such copies would total **7,122 MiB**. This is a conditional source-derived failure mode, not an observed allocation in our runtime. A custom forward that suppresses the ordinary autograd graph may avoid retaining all those copies, but repeated casts still waste traffic.

**Design:** create one transient head copy in the same dtype baseline autocast uses and share it across chunks and backward. Preserve the original frozen fp32 tied parameter. Casting the parameter permanently changes the baseline and is not this proposal.

An independent chunked-CE implementation explicitly casts its projection weight before its token loop, providing a useful design precedent. Its code remains an idea source, not a dependency for this challenge. [S20]

### Trap B — computing fp16 gradients before loss scaling

Standard AMP scales the loss before backward computes intermediate gradients. Precomputing an unscaled fp16 hidden gradient during forward and multiplying it by grad_loss later can lose values to underflow before that multiplier arrives. [S21]

A simple IEEE fp16 arithmetic illustration, checked with Python’s half conversion:

    fp16(1e-8) × 65536 = 0
    fp16(1e-8 × 65536) ≈ 0.0006551743

This illustrates ordering, not a measured failure of a particular external library.

For an initial custom backward, recompute logits for a chunk and form:

\[
dZ=\frac{\mathrm{grad\_loss}}{D}(P-Y)
\]

in fp32, including the incoming scaled-loss multiplier, before converting dZ to the GEMM dtype. Then compute dH=dZ W. Preserve the regular scaler and optimizer path. Do not divide away GradScaler’s multiplier or apply an independent second scale.

The conservative design adds projection recomputation. An alternative that precomputes gradients needs a demonstrated precision-preserving strategy; it is not automatically wrong, but it is a separate research problem.

### Baseline logit rounding also matters

Baseline head GEMM values are rounded to fp16 before transformers computes fp32 CE. A completely fused projection/CE kernel that sends unrounded fp32 accumulators directly into softmax changes those semantics. Beginning with permitted PyTorch GEMMs and original Triton reductions preserves a clearer reference boundary. [S07], [S14]

## 5. Chunking can save memory while increasing computation

The official model dimensions are hidden size 1,536 and vocabulary 151,936. At 2,048 padded positions, the derived sizes are: [S06]

| Object | fp16 | fp32 |
|---|---:|---:|
| Full logits | 593.5 MiB | 1,187 MiB |
| 128-position logits chunk | 37.09 MiB | 74.19 MiB |
| 256-position logits chunk | 74.19 MiB | 148.38 MiB |
| Tied embedding/head | 445.125 MiB | 890.25 MiB |
| Final hidden states | 6 MiB | 12 MiB |

These buffers do not necessarily coexist, and references or aliases must not be counted as independent allocations.

One full head projection costs approximately:

\[
2(2048)(1536)(151936)=0.956\ \mathrm{TFLOPs}.
\]

The frozen-head baseline needs a forward projection and a hidden-gradient GEMM: roughly 1.912 TFLOPs. Recomputing all logits in backward brings that to about 2.868 TFLOPs, a **50% increase in head arithmetic**. This says nothing directly about the fraction of total training time.

### A useful head-only pruning condition

Let Np be padded positions processed by baseline, Nr real positions required for entropy, and Ns shifted supervised positions requiring a gradient. A chunked forward plus supervised-only recompute backward requires approximately:

\[
2HV(N_r+2N_s),
\]

against baseline:

\[
4HVN_p.
\]

Thus projection arithmetic falls when:

\[
N_r+2N_s<2N_p.
\]

With no padding, this requires fewer than half the positions to be supervised. That is a break-even calculation, not a claim about this dataset’s actual label ratio.

Gather/scatter only at the output head, within each original batch. Keep the decoder, attention layout, data order, and dropout sequence unchanged. For unusual masks, the forward uses the union of entropy-required and loss-required positions. Preserve original row identities for rewind.

The next profile should record padded, real, and supervised position counts. Those counts determine whether recomputation is plausibly competitive.

## 6. What LoRA memory accounting now tells us

### bitsandbytes already omits two large saved objects

MatMul4Bit retains packed weights and quantization state for backward, not the base input activation or the decoded dense forward weight. Its backward dequantizes again to form the input gradient. Frozen base weights still affect that gradient. [S22]

At the task dimensions, decoded fp16 q and v weights occupy 4.5 MiB and 0.75 MiB respectively. They are per-call temporaries. Caching both across 28 layers would introduce about 147 MiB of retained storage, not remove that amount.

The pinned CUDA backend already has fused NF4 GEMM logic for T4. Its heuristic chooses dequantization plus F.linear for these q/v shapes when flattened M exceeds 64; the outer dispatch also falls back above M=1536. At M=2048, a new kernel competes with a deliberately chosen library GEMM path. [S23]

This supports delaying NF4/GEMM fusion until profiles show it can beat that path. Existing fusion capability does not rule out improvement; it changes the comparison.

### Saved dropout inputs need dtype-aware accounting

At M=2048 and K=1536:

| Object | Derived storage |
|---|---:|
| One original fp32 input | 12 MiB |
| One masked input retained in fp16 | 6 MiB |
| One native bool dropout mask | 3 MiB |
| One bit-packed mask | 0.375 MiB |
| One rank-16 fp16 intermediate | 0.0625 MiB |

Native CUDA dropout produces a bool mask, and its backward uses that mask. A’s parameter gradient additionally needs the masked input, potentially its fp16 autocast copy. [S24], [S25]

Replacing two q/v saved fp16 masked inputs, 6+6 MiB, with a newly retained shared fp32 input of 12 MiB saves no activation storage by itself. It can help only when that input is already live for another reason or can be reconstructed from retained state. A saved reference can extend a lifetime even when it makes no copy.

### New candidate: pack the existing random mask

Keep the same native dropout forward decisions and pack the saved bool mask into bits. An original backward can unpack/apply the same mask, or a narrowly scoped saved-tensor strategy can reconstruct it for the existing backward.

The theoretical saving is 2.625 MiB per retained full-size mask, up to **147 MiB for 56 masks**. Actual counts may be lower, including when early-layer inputs need no gradient. Shorter batches save less. Native forward still creates its temporary mask, and packing/unpacking adds work.

Do not generate a second mask and assume a matching seed makes it equivalent. Compare both the output mask and RNG state after the operation. This candidate is attractive for exact random decisions, but it is unlikely to establish the mandatory kernel-time share on its own.

### The PEFT clone is primarily a traffic target

PEFT clones the base result defensively before adding the adapter result. Clone backward passes the gradient through and does not require all cloned values to remain saved. [S09], [S25]

If q/v outputs are fp32, the clones request 12+2 MiB per layer: 392 MiB across a full-length forward and at least 784 MiB of copy read/write traffic. That is aggregate traffic/allocation activity, not 392 MiB of peak-memory savings.

After the narrow epilogue passes correctness checks, test clone removal separately only if the new operation returns fresh storage and never modifies the base output or its aliases.

## 7. How the literature informs the design

LoRAFusion remains the most directly relevant LoRA fusion reference from round 1: selective fusion and backward planning are useful, but its reported hardware and RNG implementation do not establish performance or dropout equivalence for this T4 run. [S26]

A newer Hugging Face fused-linear-CE kernel card describes vocabulary tiling, backward recomputation, and entropy/accuracy collected during the same sweep. Its metrics cover nonignored targets, which differs from the pinned trainer’s attention-mask entropy. Its generic CUDA listing is not a T4 benchmark. Use the algorithmic ideas and cite them; do not import its ready-made kernels. [S27]

The practical direction is therefore a hybrid: established GEMMs where they are strong, original Triton reductions and adapter operations where they remove concrete traffic, and explicit preservation of this task’s metrics and precision.

## 8. Ranked experiments and stopping criteria

| Order | Experiment | Question it resolves |
|---|---|---|
| 0 | Untouched T4 harness baseline and focused trace | Actual dtypes, live storage, position counts, time shares |
| 1 | Fused entropy and strided accuracy | Can we remove the logits copy and repeated reductions while leaving loss/backward intact? |
| 2 | Narrow rank-16 expansion/scale/add with backward | Does useful LoRA fusion improve command time and kernel share? |
| 3 | Reuse per-token metric summaries for rewind | Can we eliminate redundant CE reductions with identical row records? |
| 4 | Chunked head with one cast and backward recomputation | Do memory/traffic savings outweigh recomputation for the observed masks? |
| 5 | Mask packing and clone removal, separately | Are these bounded savings worth their launch/integration cost? |
| 6 | NF4 fused training GEMM | Can it beat the pinned backend on measured training shapes? |

The starter SwiGLU example may remain enabled as allowed by the kit, but it does not count as our original contribution, even if renamed or rewritten. Do not present its memory saving as a new result. [S02]

### Baseline evidence to capture

Record actual package versions; GPU and compute capability; runtime dtypes at norm, base q/v, adapter A/B, head, and trainer output; original parameter identities; lengths and masks; time by phase; peak allocated/reserved memory; and saved tensor storage identities.

Deduplicate aliases when accounting for saved tensors. A shape census alone overcounts shared storage, while an overly broad saved-tensor hook can itself prolong lifetimes. Use profiling to resolve a specific question rather than treating a profiler run as the scored result.

### Correctness and acceptance

Validate metric masks, shifts, ties, tail tiles, zero-supervision rows, and original row order. For gradient-bearing operations, compare a complete module forward/backward and multiple AMP optimizer steps, including nonzero B, actual loss scaling, parameter-gradient dtypes, and overflow behavior.

Run the unchanged official harness and checks. The README’s commands include:

    python -m pytest solution -q
    python harness/run_task.py
    python harness/check_rules.py

The required Soup regression command must be run in the appropriate Soup test checkout with the solution hook enabled. The starter documents it as:

    SOUP_APPLY_SOLUTION=1 python -m pytest tests -q

Original useful kernels under solution must contribute at least 5% of measured training-step wall time from step 3 onward in every measured run. Normal launches are required for instrumentation. A fast small kernel may fail this eligibility condition; add meaningful covered work, never timing padding. [S02], [S28]

Each scored process uses a fresh per-run Triton cache, and scored command time includes setup, compilation, training, and saving. Broad autotuning is therefore a real cost over only 64 steps. Use a small justified configuration set. Preserve all fixed task flags and the ordinary scaler, optimizer, and logging behavior. [S28], [S29]

## 9. What is complete, and what remains open

Completed: version-specific source review, corrected memory accounting, metric contract mapping, a stable tiled-reduction design, a GradScaler-safe chunked-loss design, arithmetic checks, and a revised experiment order.

Still open: the authorized team checkout, an available T4 run, installed dependency verification, runtime dtype/lifetime tracing, an original solution implementation, and all performance/quality results. No speedup or peak-memory reduction has been demonstrated.

**Next concrete action:** run the untouched official baseline on T4, capture the narrow trace above, and implement the metric-only experiment against those observed tensors. Use its result to decide how much effort to put into full head chunking versus LoRA fusion.

## Primary sources

- [S01] — Starter notebook and dependency pins
- [S02] — Starter rules, hooks, commands, and organizer measurements
- [S03] — Fixed task
- [S04] — Soup defaults
- [S05] — PEFT k-bit preparation
- [S06] — Official model configuration
- [S07] — Qwen implementation
- [S08] — bitsandbytes Linear4bit
- [S09] — PEFT 4-bit LoRA forward
- [S10] — Pinned trainer, metric semantics, dtype conversion
- [S11] — Soup fp16 parameter alignment
- [S12] — AMP model preparation and scaled backward
- [S13] — Recursive fp32 output conversion
- [S14] — Causal loss shifts, upcast, and denominator
- [S15] — Rewind row statistics and bookkeeping
- [S16] — Existing 128-row entropy chunking
- [S17] — Fused softmax scope
- [S18] — Turing hardware constraints
- [S19] — Autocast cache condition
- [S20] — Cast-once chunking precedent
- [S21] — AMP gradient scaling and custom autograd
- [S22] — 4-bit saved state and backward
- [S23] — T4 fused-GEMM dispatch
- [S24] — Native dropout mask
- [S25] — Dropout and clone derivatives
- [S26] — LoRAFusion
- [S27] — Fused projection, loss, and metrics algorithm
- [S28] — Training instrumentation
- [S29] — Scored process and cache setup

[S01]: https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/templates/kaggle_notebook.ipynb "Starter notebook and dependency pins"
[S02]: https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md "Starter rules, hooks, commands, and organizer measurements"
[S03]: https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/task/task.yaml "Fixed task"
[S04]: https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/config/schema.py "Soup defaults"
[S05]: https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/other.py "PEFT k-bit preparation"
[S06]: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json "Official model configuration"
[S07]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py "Qwen implementation"
[S08]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py "bitsandbytes Linear4bit"
[S09]: https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py "PEFT 4-bit LoRA forward"
[S10]: https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py "Pinned trainer, metric semantics, dtype conversion"
[S11]: https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/utils/mixed_precision.py "Soup fp16 parameter alignment"
[S12]: https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py "AMP model preparation and scaled backward"
[S13]: https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py "Recursive fp32 output conversion"
[S14]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/loss/loss_utils.py "Causal loss shifts, upcast, and denominator"
[S15]: https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/trainer/rewind_hf.py "Rewind row statistics and bookkeeping"
[S16]: https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/utils.py "Existing 128-row entropy chunking"
[S17]: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html "Fused softmax scope"
[S18]: https://docs.nvidia.com/cuda/turing-tuning-guide/index.html "Turing hardware constraints"
[S19]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/autocast_mode.cpp "Autocast cache condition"
[S20]: https://github.com/mgmalek/efficient_cross_entropy/blob/main/modules.py "Cast-once chunking precedent"
[S21]: https://docs.pytorch.org/docs/2.11/notes/amp_examples.html "AMP gradient scaling and custom autograd"
[S22]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py "4-bit saved state and backward"
[S23]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py "T4 fused-GEMM dispatch"
[S24]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu "Native dropout mask"
[S25]: https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml "Dropout and clone derivatives"
[S26]: https://arxiv.org/html/2510.00206v1 "LoRAFusion"
[S27]: https://huggingface.co/kernels/trl-lib/fused-linear-ce "Fused projection, loss, and metrics algorithm"
[S28]: https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/_train.py "Training instrumentation"
[S29]: https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/run_task.py "Scored process and cache setup"

