# LoRA and kernel research — Round 5

Updated 2026-10-30. Fixed target: Decentrathon 6.0 Soup Kernel Challenge, one T4, Qwen2.5-1.5B-Instruct, rank-16 q/v LoRA. This round advances the prior source review into original kernel source, CPU numerical experiments, a real tiny Qwen/Trainer integration test, and offline target compilation.

## Decision

**Keep the native matrix multiplications in the first integration. Prioritize the private-logits head/loss boundary H0, original vocabulary reductions, and the narrower LoRA epilogue L1e.**

The decisive new finding is specific: ordinary blocked `tl.dot` in the audited **Triton 3.6.0** falls back to scalar FMA below compute capability 8.0. Our SM75 compilation reproduces this; the same source compiled as an SM80 control emits MMA. T4 hardware has Tensor Cores, but this ordinary compiler route does not use them. The challenge installation does not explicitly pin Triton, so the actual target environment must be inspected before applying this finding to its installed stack. [1–2]

L1e preserves the native LoRA input-gradient GEMM and fuses its subsequent half-to-float conversion and dropout backward. It retains two thirds of full L1's calculated intermediate-traffic removal while reducing numerical and compiler uncertainty. Full L1 remains a candidate: its small rank can make traffic and launch overhead more consequential than arithmetic throughput. No runtime winner has been measured.

**Status:** six original Triton entry points exist; 13 SM75 configurations compile; CPU and interpreter results are recorded. No CUDA kernel execution, T4 timing, full Soup training, official numerical gate, or qualifying kernel-time share has been established. These are research artifacts, not a ready submission.

## Deliverables and evidence levels

- `KERNEL_HANDOFF.md`: compact state, fixed constraints, corrections, and next actions for a new session.
- `ROUND_5_EVIDENCE.json`: complete component results, including failures and scope limitations.
- `KERNEL_ROUND_5_REPRO.zip`: original Python/Triton sources, reproducible inputs, component notes, result JSON, and a SHA256 manifest. It excludes dependency installations, compiler caches, and generated binaries.

| Evidence | Executed or inspected | What it establishes |
|---|---|---|
| Versioned source review | Triton, PyTorch, Transformers, Accelerate, TRL, Soup, challenge harness | Concrete dispatch, loss, output, and timing contracts |
| Native CPU integration | Tiny real Qwen2; actual HF Trainer loss method and Accelerate output conversion | Tested integration behavior and CPU gradient/update agreement |
| Triton CPU interpreter | Six original kernels across dedicated cases | Indexing, masking, reference comparisons, and exceptional-value diagnostics |
| Offline compilation | 13 SM75 variants; one SM80 diagnostic control | Code generation, instruction-family evidence, and compiled resource counts |
| Target GPU and full challenge | Not executed | Runtime, GPU numerics, peak memory, eligibility, and official quality remain open |

The interpreter evaluates operations using NumPy and executes programs sequentially. It does not reproduce CUDA concurrency, races, target reduction instructions, Tensor Core arithmetic, or timing. Offline compilation also does not execute the generated kernel. [3]

## 1. Compiler discovery changes the matrix-kernel assumptions

Triton 3.6.0's `AccelerateMatmul.cpp` explicitly rejects the ordinary MMA rewrite for compute capability below 80 and reports a fallback to FMA. We compiled independently written full-L1 kernels with an explicit `GPUTarget("cuda", 75, 32)`, inspected their PTX, and used the wheel's `cuobjdump` to inspect the assembled cubin. All three SM75 full-L1 variants lack MMA instructions. The same 32×64 source compiled for SM80 contains MMA. [1,4]

This is a statement about this version and lowering path. It does not establish that another Triton version, an independently written inline-MMA implementation, or a native library GEMM cannot use T4 Tensor Cores. It does mean that a speed estimate assigning native Tensor Core throughput to these ordinary SM75 `tl.dot` kernels is unjustified.

### Compiled resources

All configurations below use four warps and report zero local and stack bytes. Shared-memory figures are compiler requirements; where dynamic shared memory is used, a zero static-shared cubin field does not mean the kernel needs no shared memory.

| Kernel/configuration | Variants | Registers/thread | Shared requirement |
|---|---:|---:|---:|
| Full L1, 16×64 | 1 | 95 | 4,096 B |
| Full L1, 32×64 | 1 | 96 | 4,096 B |
| Full L1, 32×128 | 1 | 168 | 8,192 B |
| L1e pointwise epilogue, block 256 | 1 | 22 | 0 |
| A0 aligned mask, block 256 | 1 | 24 | 0 |
| Vocabulary partials, tile 1,024 | 1 | 68 | 16 B |
| Vocabulary partials, tile 4,096 | 1 | 189 | 16 B |
| Vocabulary merge, 64 partial lanes | 1 | 33 | 8 B |
| Vocabulary merge, 256 partial lanes | 1 | 43 | 16 B |
| Diagnostic dZ, tile 1,024, two formulas | 2 | 34 | 0 |
| Diagnostic dZ, tile 4,096, two formulas | 2 | 72 | 0 |

These are measured compiler outputs, not measured occupancy. In particular, 4,096-wide vocabulary partials reduce scratch and program count but use substantially more registers than 1,024-wide partials. Both deserve a target comparison. The half-rounding boundary remains visible in full-L1 PTX. The loss variants contain approximate exponential instructions, reinforcing the need to test their GPU numerics rather than transfer interpreter tolerances uncritically.

Replacing the large head backward GEMM is especially unattractive as the first experiment. At maximum shape its arithmetic is about 0.956 TFLOP, approximately 170 times the whole rank-16 L1 dot family. A changed arithmetic engine can dominate any saved intermediate traffic. These are operation counts, not runtime predictions. H0 leaves that native head computation intact.

## 2. L1e: a narrower LoRA change with an executable autograd boundary

The relevant native sequence is conceptually:

1. Native GEMM produces the half input gradient for the adapter's dropout-and-cast path.
2. Autograd converts that gradient to fp32.
3. Native dropout backward multiplies the retained mask and fp32 inverse keep probability.

L1e fuses steps 2 and 3. Its original `dropout_half_gradient_epilogue` consumes the native half gradient, upcasts, multiplies the original mask, then applies the exact fp32 scale and writes fp32 input gradients. It generates no new random decisions and does not change the A/B parameter GEMMs.

The prototype supplies a custom dropout-and-cast boundary: forward calls the native dropout primitive on the original fp32 input shape, stores its mask, and returns the half cast to the unchanged native LoRA projections. Backward uses the original pointwise Triton kernel. A `[2,9,48]` test checks that forward preserves a genuine 3D shape; flattening is confined to pointwise backward. A noncontiguous backward reshape may still copy and needs target profiling.

### Executed evidence

Four controlled CPU graph cases use actual fp16 autocast, nonzero adapter weights, two adapter branches, and incoming scales 1, 1,000, and 65,536. Each compares seven gradients, covering 4,464, 4,464, 9,840, and 4,576 elements respectively. All forward outputs and all seven gradients are exactly equal. The native-primitive CPU RNG state after forward also matches.

A separate exceptional-value test retains the reference behavior of false-mask multiplication: infinity or NaN multiplied by zero yields NaN. Finite signed zeros also match. Replacing multiplication with a conditional zero would change this behavior.

These comparisons deliberately use the same native-dropout primitive on both CPU sides. They do not prove equivalence to the separate generic CPU `nn.Dropout` path or establish CUDA RNG agreement. The pinned PyTorch source dispatches ordinary nonempty CUDA training tensors with `0 < p < 1` to `native_dropout`. A target patch must retain the original shape, strides, device, probability, q/v call order, and native dispatch. Unsupported modes require the original path. [5]

### Traffic calculation

At M=2,048 and H=1,536, one half input-gradient matrix is 6 MiB and its fp32 equivalent is 12 MiB. Full L1 can remove the half intermediate's write/read and the fp32 intermediate's write/read: **36 MiB per branch**. L1e removes only the fp32 intermediate's write/read: **24 MiB per branch**.

| Maximum-shape scenario | Full L1 | L1e |
|---|---:|---:|
| Removed logical traffic per branch | 36 MiB | 24 MiB |
| Removed logical traffic over 56 q/v branches | 2,016 MiB | 1,344 MiB |
| Fraction of full L1 removal | 100% | 66.7% |

If the first layer does not require input gradients, use 54 branches: 1,944 MiB and 1,296 MiB respectively. The target trace must resolve the actual active branches.

These are logical intermediate reads/writes. They are neither measured DRAM traffic nor retained-memory savings. Cache hits, repeated operand loads, launch costs, and the actual native kernels can change their performance value. Likewise, eliminated operator sites should not be labeled exact CUDA launch counts.

Full L1 remains viable enough to benchmark. A simple 32×64 rank-16 tile model gives about 4.92 FLOPs per logical byte, below the ideal scalar-fp32 balance implied by T4's published ceilings. That leaves room for a bandwidth-limited kernel despite FMA lowering; it does not prove one. L1e is the cleaner first comparison. [6]

## 3. H0 now has a real tiny-Qwen integration test

The earlier head/loss design now executes through a locally initialized **real Qwen2ForCausalLM**, actual **Transformers 5.16.1 `Trainer.compute_loss`**, and actual **Accelerate 1.14.0 output conversion**. The tiny configuration uses vocabulary 37, hidden width 32, MLP width 64, two decoder layers, four query heads, two KV heads, and a tied frozen head/embedding. Ordinary decoder weights remain trainable. No pretrained weights or dataset are downloaded.

Both sides retain the same native head forward and backward. The candidate stores the already-rounded full-row fp16 logits privately in autograd, computes native fp32 loss/statistics in chunks, and recomputes native log-softmax for a pinned ATen backward diagnostic. The actual incoming loss scalar and exact denominator are applied before the half-gradient store. This backward is an oracle for further work; it is not an original Triton kernel. [7–9]

### What passed

- Five main comparisons produce exact loss, genuine summaries, and all **25 trainable gradient tensors / 18,720 elements** per comparison.
- Cases cover scales 1/65,536/1,000, an external denominator of 17, and all-ignored labels with or without a positive supplied denominator.
- A retained-graph second backward matches the first and leaves the saved logits unchanged.
- Three consecutive real CPU GradScaler, gradient-clipping, and AdamW updates keep all model parameters exactly equal.
- Sixteen lifecycle checks pass, including ordinary inference/evaluation fallback, unsupported modes, exceptions, stale-summary prevention, repeated installation, and preservation of auxiliary outputs.
- Four ignored-row nonfinite cases preserve native NaN gradient and entropy patterns.

Head hooks observe exactly one native forward with `[3,9,32] → [3,9,37]` on each side and the original full fp16 gradient shape on backward. Installation leaves state-dict names, the head/embedding tie, and global CPU RNG state unchanged.

### The output-object pitfall is now reproduced

For the tested ModelOutput implementation, assigning `outputs.logits = None` leaves the prior tensor in its mapping. Actual Accelerate conversion still finds and upcasts that stale mapping value. Reconstructing a genuine `CausalLMOutputWithPast` without a logits entry fixes this. The candidate's returned object contains no vocabulary-shaped tensor even under recursive traversal. [7,9]

Saved-tensor hooks see two baseline fp32 full-vocabulary references aliasing **one 3,996-byte storage**, versus zero such candidate storages. The candidate retains a private 1,998-byte half cache in this tiny example. Counting the two baseline references as two allocations would double-count. This is structural saved-state evidence, not a peak-memory measurement.

The test exercises a genuine Qwen/HF/Accelerate boundary. It does not execute actual TRL SFTTrainer, Soup Rewind/FIFO, PEFT, NF4, full-size Qwen, CUDA, or the official harness. The summary contract still needs integration with those real consumers. Soup's existing no-logits Rewind branch can consume `_soup_chunked_step.token_loss`; that is source evidence, not a completed recorder test. [10–11]

## 4. Original vocabulary reductions: promising forward, unresolved backward

With V=151,936, padding a full row to a power of two gives 262,144 lanes. A monolithic row reduction is a poor default without examining resource use. The original implementation instead partitions vocabulary columns and merges small partial summaries.

For block k, define:

\[
m_k=\max_i z_i,\qquad s_k=\sum_i e^{z_i-m_k},\qquad q_k=\sum_i e^{z_i-m_k}(z_i-m_k).
\]

With \(m=\max_k m_k\) and \(a_k=e^{m_k-m}\):

\[
S=\sum_k a_ks_k,\qquad Q=\sum_k a_k[q_k+(m_k-m)s_k].
\]

Then, for finite logits:

\[
\mathrm{CE}(y)=(m-z_y)+\log S,\qquad H=\log S-Q/S.
\]

The normalizer merge follows the established online-softmax identity; the centered moment follows by expansion around each block maximum. Keeping the subtraction centered avoids the cancellation in an uncentered `LSE - E[z]` expression. It does not make different reduction trees numerically identical. [12]

Three original kernels implement block partials, row completion, and diagnostic scaled logits gradients. The forward emits genuine token CE, entropy, argmax, and correctness without materializing full fp32 probability or log-probability tensors. It performs the causal shift within each sequence, never across batch-row boundaries. PyTorch's first-index tie and first-NaN argmax policy is handled explicitly. [13]

At M=2,048, partial scratch is **5.8203 MiB** with tile 1,024 (149 partials/row) and **1.4844 MiB** with tile 4,096 (38 partials/row). Per-token outputs add O(M) storage; the H0 half-logits cache remains separate. Smaller scratch does not by itself select the faster configuration.

### All numerical results, including failures

The CPU arithmetic study runs 20 cases × four tile sizes × three entropy methods = **240 comparisons**. Every method passes all 80 CE comparisons and all argmax checks. Centered entropy fails four strict reference comparisons; the uncentered negative control fails 16; a native-expression second pass fails four. Centering removes the large-common-offset failures exposed by the negative control.

The original kernels in Triton's CPU interpreter produce:

| Quantity | Result |
|---|---:|
| Forward CE | 42/42 pass |
| Forward entropy | 40/42 pass |
| Argmax, correctness, ignored-token zero loss | All exact |
| Diagnostic dZ comparisons | 164/168 pass |
| Gradient nonfinite patterns | All match |

The two entropy failures are cross-block ties, with absolute errors around **2.19–2.29×10^-5** against native fp32. A separate fp64 check finds that the centered CPU result is closer to mathematical entropy than the native fp32 reference. That context does not erase a baseline mismatch or justify relaxing the gate.

The four dZ failures occur at saturated margin 16 and upstream scale 65,536, for both tile sizes and both formulas. Maximum absolute half-gradient error is **0.001953125**, about **1.47% of the reference gradient peak**. Forward CE agreement therefore does not authorize this backward.

The exploratory forward gate is `atol=rtol=2e-6`; the gradient gate allows an absolute error at most 2e-6, or both peak-relative and relative-L2 errors at most 0.002. These are research criteria, not the challenge tolerances. **Keep native backward as the first H0 control.** GPU exponentials, fused instructions, and native reductions must be compared on the actual target before choosing a custom dZ formula.

## 5. Correction to H1: ignored rows can still affect nonfinite behavior

Round 4's finite-input evidence does not establish general equivalence for dropping ignored or padded rows. A NaN or positive infinity in an ignored logits row can yield **zero token CE but NaN logits gradients** through native log-softmax backward. Entropy can remain NaN after multiplication by a zero attention mask. The new loss and H0 tests both reproduce this.

The distinction matters because suppressing those gradients can change overflow detection and the resulting training trajectory. That is an inference from the observed primitive behavior; no complete nonfinite training trajectory was run.

| Logit row | Supervised CE | Native entropy | Ignored-row gradient |
|---|---|---|---|
| All finite | Ordinary CE | Ordinary entropy | Zero |
| Finite values mixed with -Inf | Finite, or +Inf for a -Inf target | NaN from zero probability × -Inf | Zero |
| Any NaN or +Inf | NaN | NaN | NaN |
| All -Inf | NaN | NaN | NaN |

The prototypes keep explicit nonfinite row flags and preserve these patterns. A later supervised-only cache could retain small flags and reconstruct full gradients for exceptional ignored rows. Skipping padded head-forward computation is harder: it can omit the computation that reveals the exceptional state. H1 needs a compatible policy or original-path fallback; zero labels alone do not supply a proof. The native entropy expression and subsequent masking are visible in the pinned TRL sources. [10]

## 6. Participation and scoring are separate constraints

### The exact original-kernel share

The pinned harness wraps ordinary `JITFunction.run` calls, records source-qualified CUDA-event intervals, and measures synchronized callback step-wall intervals from step 3 onward. The checker intersects eligible kernel names across successful solution runs and uses the lowest per-run share. Its denominator is not whole-command time or profiler-summed GPU time. [14–15]

\[
\mathrm{share}_j=\frac{\sum \mathrm{eligible\ event\ milliseconds}_j}{1000\sum \mathrm{callback\ step\ wall\ seconds}_j},\qquad \min_j\mathrm{share}_j\ge0.05.
\]

Names absent from a measured run cannot contribute through the common eligible set. The checker warns that 5–6% is marginal under hidden rows. Source location alone also does not establish originality or usefulness: the challenge's review rules remain binding.

For a simplified serial planning model, let p be a kernel family's baseline share of this same wall-time denominator and s its acceleration factor. Its new share is:

\[
f=\frac{p}{s(1-p)+p}.
\]

To retain 5% after acceleration, \(p\ge s/(19+s)\). Thus a 2× accelerated family needs at least 9.52% baseline wall share. A hypothetical 7% family accelerated 2× falls to 3.63%. The organizer's approximately 7% softmax/CE profiler figure is a different denominator and must not be substituted for p.

This motivates several useful original pointwise/reduction operations in a coherent solution. Artificial delay, unused computation, timing manipulation, and imported ready-made kernels are not acceptable ways to pass. H0's native ATen oracle contributes no original Triton time. L1e's speed and share remain unknown until measured.

### Whole-command scoring adds a second hurdle

`score.py` maps base/solution ratios below 1.05 to a tie. Clearing 1.05 requires a **4.7619% reduction relative to the baseline time or peak**. With baseline command time T, steady training savings D, and added compilation/setup cost C:

\[
T_{\mathrm{solution}}=T-D+C,\qquad D\ge C+T/21.
\]

Using the organizer's reported 79–83 second commands only as an illustrative scenario, the required net saving is 3.762–3.952 seconds. Adding two seconds of setup/compilation raises required aggregate training savings to 5.762–5.952 seconds, approximately 90–93 ms averaged across 64 steps. Fresh compiler caches make this material. [16–17]

Similarly, at the organizer's reported 14,350 MiB peak, a 1.05 ratio needs about 683.33 MiB of actual peak reduction. A0's conditional 54 MiB retained-mask saving could contribute, but cannot independently clear that threshold. Logical traffic, tensor sizes, and structural saved-state evidence must not be added together as a measured peak benefit.

## 7. Next experiment sequence and acceptance evidence

The remaining bottleneck in the research is access to the authorized team checkout and actual T4 runtime. Neither has been identified in the available session. No public starter files or harness have been changed.

| Order | Experiment | Required evidence before promotion |
|---:|---|---|
| 1 | Untouched target baseline | Installed versions; actual dtypes and masks; SDPA path; row IDs/order; timing and peak; native kernel shapes |
| 2 | Genuine-summary control and H0 | Full Soup/TRL/Rewind integration; original native head schedule; no returned logits; exact counts and required loss/metric checks |
| 3 | Original vocabulary forward reductions | 1,024/4,096 tile comparison; CUDA entropy/CE/argmax/nonfinite checks; keep native loss backward initially |
| 4 | L1e dropout-gradient epilogue | CUDA RNG/masks unchanged; all adapter/input gradients; several AMP updates; target runtime and eligible share |
| 5 | A0 aligned additive mask | Actual efficient attention path; reuse/lifetime evidence; no backend regression; target memory/runtime |
| 6 | Full L1, then H1/L2 if justified | Measured benefit over controls; preserved rounding/nonfinites; full official acceptance checks |

Use a few structural compile configurations and runtime dimensions where practical; there are only 64 training steps. Record each real run in an experiment log with commit, environment, source hashes, correctness status, whole-command time, allocator peak, and per-run eligible share. Run the unchanged official harness/checker and required Soup tests before claiming a submission result.

The useful outcome of this round is a narrower implementation choice with runnable evidence: H0 can be exercised through real library boundaries; L1e has a tested autograd design; original vocabulary reductions have explicit numerical failures and resource tradeoffs; the actual rules are expressed as measurable quantities. Target execution now has concrete alternatives and acceptance checks.

## Primary references

1. [Triton 3.6 ordinary MMA lowering and SM75 FMA fallback](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp).
2. [Triton 3.6 supported hardware](https://github.com/triton-lang/triton/blob/v3.6.0/README.md).
3. [Triton interpreter documentation](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html).
4. [Triton explicit-target compilation](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/compiler/compiler.py).
5. [PyTorch 2.11 dropout dispatch](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Dropout.cpp) and [CUDA dropout](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu).
6. [NVIDIA T4 specifications](https://www.nvidia.com/en-sg/data-center/tesla-t4/) and [T4 product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/T4%20Product%20Brief.pdf).
7. [Qwen2 model/head/output contract](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py) and [ModelOutput implementation](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/utils/generic.py).
8. [Transformers Trainer](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py) and [causal loss/denominator](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/loss/loss_utils.py).
9. [Accelerate output conversion](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py).
10. [TRL entropy helper](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/utils.py) and [SFT metrics](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py).
11. [Soup pinned Rewind integration](https://example.invalid/redacted-github).
12. [Milakov and Gimelshein, Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867v2).
13. [PyTorch 2.11 GreaterOrNan comparator](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/SharedReduceOps.h).
14. [Pinned harness launch timing and step callback](https://example.invalid/redacted-github).
15. [Pinned eligibility checker](https://example.invalid/redacted-github).
16. [Pinned score calculation](https://example.invalid/redacted-github).
17. [Pinned run lifecycle and aggregation](https://example.invalid/redacted-github) and [organizer reference measurements](https://example.invalid/redacted-github).

All reported experiment outcomes are original observations recorded in `ROUND_5_EVIDENCE.json` and the component result files. Published hardware and organizer figures are labeled separately. External ideas are attributed; no external GPU kernel was imported or copied into the original prototypes.
