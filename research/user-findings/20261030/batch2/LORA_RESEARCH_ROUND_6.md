# LoRA and kernel research — Round 6

Updated 2026-10-30, Asia/Almaty. Target remains the fixed Decentrathon 6.0 Soup Kernel Challenge on one T4 with Qwen2.5-1.5B-Instruct and rank16 q/v LoRA. This report continues Round 5; it does not change the model, prescribed training task, or challenge rules.

## Decision and new evidence

**Keep H0 with native head/loss backward and L1e with native GEMM as the first target controls. Add the new original SM75 inline-MMA kernel as a concrete full-L1 comparison. Evaluate loss arithmetic against native CUDA before choosing a replacement formula.**

Three uncertainties have narrowed substantially:

1. The private-logits H0 design now executes through actual PEFT LoRA, real TRL SFT initialization/collation/logging, the HF training loop, and Soup's real Rewind sampler/FIFO/disk recorder. Adapter updates, genuine metrics, and row records agree on the tiny CPU fixture.
2. Round 5's four saturated-gradient mismatches are exactly explained by a one-ULP difference between NumPy and PyTorch exponentials, amplified by the real loss scale. The entropy discrepancies mainly originate in the native CPU normalizer. These are diagnoses of the observed tests; they do not turn the original failed comparisons into passes.
3. An independently written inline-PTX kernel compiles to actual **HMMA.1688.F32** instructions on SM75 under Triton3.6.0. Ordinary `tl.dot` still follows the FMA route observed in Round 5; explicit MMA is a separate, now demonstrated compilation path.

**No CUDA kernel has been executed in this research environment.** There is no measured T4 speedup, model peak reduction, original-kernel share, full challenge run, or official pass. Preflight explicitly reports no available CUDA device. The authorized team checkout and executable T4 runtime were not found in the available challenge context.

## Deliverables and verification boundaries

- `KERNEL_HANDOFF.md`: compact current state for the next session.
- `ROUND_6_EVIDENCE.json`: combined integration, numerical-diagnosis, compiler, mapping, and runner results, with limitations and hashes.
- `KERNEL_ROUND_6_REPRO.zip`: original sources, subsystem notes, dependency pins, necessary unchanged Round 5 references, results, and an integrity manifest. No generated binaries, dependencies, or compiler caches are included.

| Workstream | New executed evidence | Still unestablished |
|---|---|---|
| H0 integration | Real tiny PEFT/SFT/HF training and Soup Rewind; two accumulation settings; gradients/updates/records | Full Soup CLI construction, NF4, CUDA, full model, public/hidden data |
| Loss diagnosis | Controlled CPU substitutions isolate exponential and normalizer differences | Native CUDA agreement and production dZ selection |
| Inline MMA | Original SM75 source, assembled HMMA, compiled resource counts, CPU fragment mapping | GPU correctness, bounds/races, cuBLAS agreement, speed |
| Target runner | CPU preflight and explicit interpreter smoke; preserved expected mismatches | All CUDA validation and optional timing modes |

The reproduction package preserves these categories. A compile success is not a GPU numerical pass; CPU mapping is not an MMA emulator; interpreter simulation is not native CUDA arithmetic. Triton explicitly documents its interpreter as sequential NumPy simulation. [1]

## 1. H0 now passes through actual SFT, PEFT, and Rewind consumers

### What executes

The new fixture uses the real Qwen2 implementation with vocabulary37, hidden32, MLP64, two layers, four query heads, two KV heads, and head dimension8. The head remains tied to the frozen embedding. Actual PEFT0.20.0 applies q/v LoRA with rank16, alpha32, scale2, and dropout0.05. Only eight adapter tensors, totaling **3,584 elements**, are trainable. B matrices start nonzero so A and input-gradient paths are exercised immediately.

The data comprise 12 locally generated pretokenized rows, lengths `[9,6,4,11,8,7,10,5,12,9,6,13]`, totaling100 real tokens and64 supervised shifted targets. One row is entirely unsupervised. Actual TRL collation applies assistant masks and dynamic right padding. A local toy tokenizer avoids a weight or dataset download.

The baseline executes stock `SFTTrainer.compute_loss`. The candidate uses an original guarded SFT specialization: it calls actual HF `Trainer.compute_loss`, receives genuine H0 summaries, and aggregates them for inherited SFT logging. Stock SFT's logits-dependent metric body cannot run unchanged after logits are omitted. This specific replacement is explicit; SFT construction, collation, training hooks, and logging remain real. Fallback calls retain stock SFT behavior. [2–4]

Soup's actual Rewind factory subclass surrounds those trainers. Its wrapped sampler supplies row IDs to its FIFO; its genuine row reducer, JSONL writer, and reader execute. No synthetic recorder object or fabricated FIFO record is used in the full training comparisons. [5–6]

Four complete installed source files were compared with their pinned remote sources: TRL SFT, PEFT causal wrapper, Soup Rewind trainer, and Soup RewindLog. All match. The evidence records source URLs, lengths, and hashes without bundling those external implementation files.

### Full-loop results

Two baseline/candidate pairs train over the tiny dataset once:

| Fixture | Micro-batches per side | Optimizer updates per side | Purpose |
|---|---:|---:|---|
| Accumulation1 | 4 | 4 | Ordinary prescribed-style step structure |
| Accumulation2 | 4 | 2 | Diagnostic of accumulated normalization and recorder micro-step state |

Accumulation2 is a test of integration semantics. The challenge remains accumulation1.

For both pairs, every clipped adapter-gradient snapshot and every model parameter snapshot after an update is elementwise exactly equal. Actual entropy, token accuracy, cumulative token count, gradient norm, and the compared schedule/step fields agree. CPU RNG states after training also match.

**Scalar loss is not exactly equal.** The largest raw or logged discrepancy is **4.76837158203125e-7**. The accumulation2 raw-loss maximum is2.384185791015625e-7. These differences remain in the JSON and are below the declared1e-6 research loss gate. Exact parameter updates do not justify labeling all floating-point logs bitwise identical.

The native head executes four forwards and four backwards per run, with the same full shapes and fp16 dtype on both sides. The observed shapes are `[3,T,32] → [3,T,37]`, with T11/13/12/9. There is no row selection, extra head forward, or custom head GEMM.

### Actual Rewind evidence

The CPU fixture's observed row order is `[3,5,0,11,7,6,8,2,10,1,4,9]`. Independent first-token identifiers verify which rows reached the model, while the recorder obtains IDs from its real sampler. This is evidence for the fixture, not an estimate of the challenge's shuffled order.

| Actual rows | Supervised counts | Accumulation1 `(step,micro)` | Accumulation2 `(step,micro)` |
|---|---|---|---|
| 3,5,0 | 9,3,7 | (1,0) | (1,0) |
| 11,7,6 | 9,2,8 | (2,0) | (1,1) |
| 8,2,10 | 8,0,3 | (3,0) | (2,0) |
| 1,4,9 | 3,5,7 | (4,0) | (2,1) |

Baseline and candidate row IDs, supervised counts, row losses, and step/micro values are exact. Every row appears once, with no FIFO underflow or recorder disable. The unsupervised row correctly records loss0/count0. Header fields match except the expected wall timestamp. Real record values are retained in the evidence.

### Scaled gradients and lifecycle

Three direct real-PEFT/SFT comparisons use incoming scales1/65,536/1,000; the third supplies denominator17. All eight adapter gradients/3,584 elements match exactly, as do genuine summaries and post-forward CPU RNG state. Direct calls detach Rewind because they do not consume the dataloader; sampler evidence comes from the full-loop tests above.

Three separate actual CPU GradScaler, unscale, clipping, and AdamW updates preserve all parameters and gradient norms exactly. The scale remains65,536 in these finite cases. The normal HF training-loop test uses an explicitly supplied CPU fp16 autocast policy; it does not run a CUDA GradScaler. These are distinct coverage claims.

Eight lifecycle checks pass: idempotent installation, direct labeled PEFT output, evaluation fallback, fp32 fallback, disabled-adapter fallback, trainable-head fallback, invalid-target cleanup, and subsequent-call recovery. State-dict names and installation-time RNG state are preserved.

The tested nesting installs H0 on the inner Qwen while the trainer receives outer PEFT. The label identity survives the nested call. A genuine output is reconstructed without logits before outer Accelerate conversion; Soup receives `_soup_chunked_step.token_loss` before its Rewind wrapper consumes it. H0 still uses native log-softmax and the native private ATen backward as its oracle. It contributes no original Triton time by itself. [4–7]

## 2. The scaled-gradient discrepancy is explained to one fp32 ULP

Round 5 found four dZ failures: saturated margin16, upstream scale65,536, both vocabulary tiles, and both gradient formulas. The new diagnostic changes one arithmetic ingredient at a time while leaving the original kernels and gates untouched.

The case has three rows, vocabulary37, winning logit16, all other logits0, two supervised winner targets, and one ignored row. The effective multiplier is65,536/2=32,768.

| Quantity | Recorded value |
|---|---:|
| Native winning log-probability | -4.0531076592742465e-6 |
| Largest blocked/native logp difference | 4.547473508864641e-13 |
| PyTorch exponential of the same logp | 0.9999959468841553 |
| NumPy exponential of the same logp | 0.9999960064888 |
| Probability difference | **2^-24**, one fp32 ULP |
| Native scaled target gradient | -0.1328125 |
| Interpreter scaled target gradient | -0.130859375 |

The observed difference is exactly:

\[
\Delta g=2^{-24}\times\frac{65536}{2}=2^{-9}=0.001953125.
\]

At this gradient magnitude, one fp16 step is2^-13. The error therefore spans **16 half-precision steps**, or about1.47% of the reference peak. The half store does not erase it; the discrepancy already exists in fp32.

The cross-substitutions isolate the cause:

| Computation | Half-gradient result |
|---|---|
| Native ATen VJP with native logp | Reference |
| Native ATen VJP with blocked logp | Exactly the same reference |
| Explicit formula with PyTorch exp on blocked logp | Exactly the same reference |
| Same formula with NumPy exp on blocked logp | Exactly reproduces the original mismatch |
| FP64 exp rounded to fp32 on blocked logp | Matches this CPU case; diagnostic only |

The tiny blocked-normalizer difference does not explain this particular failure. PyTorch's CPU vector exponential and Triton's interpreter NumPy exponential are different implementations; the pinned interpreter source explicitly maps `create_exp` to `np.exp`. [8–10]

This resolves the cause of the observed CPU-interpreter result. It does **not** establish a passing CUDA gradient kernel. The original four failed comparisons remain recorded. We have not patched interpreter arithmetic, changed the denominator, moved scaling after a premature half store, or loosened a gate.

## 3. The entropy mismatch mainly inherits CPU normalization error

The V4103 tie case has three logits equal to3 and all others0. Native CPU and blocked interpreter normalizers differ:

| Quantity | Native CPU | Blocked interpreter |
|---|---:|---:|
| Centered log normalizer | 5.333334922790527 | 5.333332061767578 |
| Maximum logp error against fp64 | 2.8861565386506527e-6 | 2.5133589431902692e-8 |
| Native entropy expression using those logp values | 8.289857864379883 | 8.28987979888916 |

FP64 entropy is8.289880430345924. Substituting only the blocked log probabilities into the native entropy expression moves its result near fp64 truth. The centered-moment entropy identity is therefore not the dominant source of the baseline mismatch.

A source-motivated16-lane fp32 fold over the same PyTorch exponential values reproduces the native CPU normalizer exactly in this run. The sum of those values in fp64 is207.12697419524193; the lane fold yields207.12762451171875; PyTorch log of that lane sum is5.333334922790527. This matches the audited CPU's lane accumulation and horizontal reduction structure. [11–12]

The finding is tied to the recorded AVX512 CPU. It is not a requirement that a GPU imitate that reduction order. Mathematical accuracy and agreement with the specified native backend are separate questions; the candidate being closer to fp64 does not authorize a different challenge tolerance. Both original entropy mismatches remain visible.

### A target-only arithmetic probe

One new original diagnostic JIT kernel now offers three modes on identical supplied fp32 log probabilities:

| Mode | Exponential | VJP arithmetic |
|---:|---|---|
| 0 | `tl.exp` | Separate multiply/subtract |
| 1 | CUDA libdevice exp | Separate multiply/subtract |
| 2 | CUDA libdevice exp | Explicit FMA |

All three compile for SM75 with20 registers/thread and zero shared/local/stack bytes. Both exponential routes contain `ex2.approx`, but the libdevice route also has argument-reduction/correction instructions. The presence of `ex2.approx` alone therefore does not identify the same numerical implementation. Static total FMA counts are0/8/10; the explicit VJP FMA contributes separately from libdevice's internal corrections.

PyTorch's CUDA backward sources spell the operation using `std::exp`; the installed CUDA wheel's exact exponential and contraction behavior cannot be inferred from that spelling alone. The runner will compare native CUDA VJP on identical logp first, then the normalizer can be varied independently. Its current result is explicitly `cuda_unavailable`, not a numerical pass. [13–15]

This probe materializes native fp32 log probabilities for diagnosis. It is not the production memory-efficient loss or a claimed useful-kernel participation contribution.

## 4. An original inline-MMA path reaches SM75 Tensor Cores

### What changed from Round 5

Round 5 correctly established that the tested ordinary Triton3.6 `tl.dot` SM75 route falls back to FMA. Round 6 adds an explicit original PTX route. The new full-L1 prototype uses two rank8 operations per warp:

```text
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
```

The assembled SM75 cubin contains exactly **two HMMA.1688.F32 instructions**. This is machine-code evidence of the desired instruction family, rather than an inference from source spelling. Compilation used the ordinary installed Triton3.6 wheel and its bundled assembler/disassembler. No compiler source, prescribed precision setting, or challenge harness was altered. [16–18]

Triton3.6 documents NVIDIA support beginning at compute capability8.0. This offline older-target compilation does not establish supported runtime behavior. The candidate has not launched on T4.

### Physical-lane mapping and participation

The inline-assembly API does not promise which packed tensor elements an invocation receives. A naive assumption connecting `tl.arange` coordinates to fixed physical lanes would be unsafe. [19]

This prototype places addressing, loads, both MMAs, half rounding, mask/scale, and stores in one scalar side-effecting assembly block with a memory clobber. Inputs are CTA-uniform; physical lane and warp are computed from `%tid.x`. Four warps cover a16×32 output tile, each owning16×8.

For lane l, let g=l//4 and t=l%4. In each rank slice k∈{0,8}, U fragments cover rows g/g+8 and columns k+2t/k+2t+1; A fragments cover rows k+2t/k+2t+1 and column g. Output registers cover rows g/g+8 and columns2t/2t+1. The four warps offset columns by0/8/16/24. The source expresses the documented fragment contract independently. [16]

Invalid rows/columns predicate loads and stores; all lanes still execute both collectives. Inspection found one side-effecting LLVM asm call, no PTX branch/early exit, no SASS branch or exit before the last HMMA, and both HMMAs unpredicated. Later epilogue branches occur after the collective work. This is an inspected property of this compiler artifact, not an assumption about every compiler version.

### Compiled and mapping evidence

| Property | Observed |
|---|---:|
| Registers/thread | **35** |
| Required threads | 128 |
| Shared/local/stack bytes | 0 / 0 / 0 |
| Assembled HMMA instructions | 2 |
| Explicit fp32→fp16 rounds | 4 |
| Explicit fp32 mask/scale multiplies | 8 |

CPU mapping checks cover every fragment coordinate, every one of65,536 half bit patterns in both packed positions, and six shape/value cases, including65×1536, tails, and overflow. Across103,144 logical outputs and1,024 simulated warp tiles, valid output ownership and guards match. The modeled epilogue preserves ties-to-even, false-mask NaNs, and signed zeros.

These tests reconstruct matrices from fragments and use ordinary CPU products. They do not execute the assembly or simulate Tensor Core numerics. The PTX specification leaves relevant MMA accumulation/rounding/subnormal details unspecified; two rank8 MMAs need not match native cuBLAS arithmetic despite the final half-round boundary. [16]

### Why HMMA plus35 registers is not a speed result

The simple candidate increases tile count and operand replication:

| Maximum-shape logical ledger | Inline MMA16×32 | Earlier ordinary-dot32×64 |
|---|---:|---:|
| CTAs at M2048,N1536 | **6,144** | 1,536 |
| U loads per branch | 12 MiB | 1.5 MiB |
| A loads per branch | 6 MiB | 3 MiB |
| Mask reads + fp32 output writes | 15 MiB | 15 MiB |
| Total modeled streams | **33 MiB** | 19.5 MiB |

The inline tile repeats U across its four column warps. Cache can serve repeated loads, and the other compiler may implement source loads differently, so this is not measured DRAM traffic. It nevertheless explains why low register use and Tensor Core instructions are insufficient to choose a winner. L1e remains the first control; this source is now a concrete full-L1 alternative to compare.

## 5. Target checks are prepared with explicit evidence states

The new synthetic runner records package versions, architecture/name, observed precision flags, source hashes, numerical comparisons, and optional eager invocation timings. It defaults to CUDA SM75, rejects accidental interpreter mode in CUDA runs, and distinguishes execution attempts from confirmed custom invocation completion.

Its default loss suite reuses all21 Round 5 fixtures:42 forward comparisons and168 dZ comparisons at two tiles. The core suite compares L1e and ordinary-dot L1 with native expressions, checks aligned-mask logical values/guards, and can explicitly include the new inline-MMA source. Maximum core shapes are opt-in.

Executed here: CPU preflight reports no CUDA. An explicit interpreter smoke passes five core cases and runs three loss fixtures/12 gradients at tile1024, retaining exactly two known saturated dZ failures. The inline-MMA and optional timing paths remain unexecuted.

From the extracted bundle root in an appropriate target environment:

```bash
python round6_target/target_checks.py --mode preflight --output preflight.json
python round6_target/target_checks.py --suite core --output core_cuda.json
python round6_loss/run_cuda_probe.py --output loss_math_cuda.json
python round6_target/target_checks.py --suite loss --output loss_cuda.json
python round6_target/target_checks.py --suite core --include-mma --max-core-shape --output mma_cuda.json
```

Optional timing reports CUDA-event and synchronized host intervals for ordinary invocations. First-call time includes compilation if needed, allocation, dispatch, and execution; later shapes can reuse specializations. It is not isolated compiler cost, a full-model speedup, a model peak, or the official≥5% statistic. A separate fresh scratch cache is used; generated binaries are never packaged. [20]

## Next decisions on the actual target

1. Capture the untouched challenge baseline, actual versions/dtypes, sampler order, and attention backend.
2. Run the focused loss math probe on native CUDA logp. Resolve exponential/contraction behavior before changing normalizer or accepting dZ.
3. Run original vocabulary forward/backward comparisons, retaining strict failures and nonfinite behavior.
4. Compare native GEMM+L1e, ordinary-dot full L1, and explicit-MMA full L1. Check memory safety and complete gradients before timing or full integration.
5. Adapt the tested inner-Qwen/outer-PEFT H0 integration to actual CUDA/NF4/Soup construction. Keep genuine SFT/Rewind summaries and native loss backward initially.
6. Use the unchanged official harness for whole-command time, peak, quality, counts, and per-run qualifying share. The Round 5 participation/scoring analysis remains applicable.

The result of this round is a stronger implementation starting point: real consumers have executed, earlier numerical failures have isolated causes, and a T4-specific instruction route is now present in original source. Target execution is the remaining evidence gap for performance and submission acceptance.

## Primary references

1. [Triton interpreter semantics](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html).
2. [TRL0.29.1 SFT trainer](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py).
3. [PEFT0.20 causal wrapper](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py) and [LoRA layer](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/layer.py).
4. [Transformers5.16.1 Trainer](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py).
5. [Pinned Soup Rewind trainer/FIFO](https://example.invalid/redacted-github).
6. [Pinned Soup Rewind writer/reader](https://example.invalid/redacted-github).
7. [Accelerate1.14 output conversion](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py).
8. [PyTorch2.11 CPU softmax backward](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cpu/SoftMaxKernel.cpp).
9. [PyTorch AVX512 float vector exponential](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/cpu/vec/vec512/vec512_float.h).
10. [Triton3.6 interpreter operation mapping](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/runtime/interpreter.py).
11. [PyTorch CPU log-softmax forward](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h).
12. [PyTorch vector accumulation](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/cpu/vec/functional_base.h).
13. [PyTorch large CUDA softmax/log-softmax](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/SoftMax.cu).
14. [PyTorch persistent CUDA softmax](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/PersistentSoftmax.cuh).
15. [CUDA single-precision math and fast-math qualifications](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html).
16. [NVIDIA PTX7.2 fragment and MMA instruction contracts](https://docs.nvidia.com/cuda/archive/11.2.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688).
17. [Triton3.6 ordinary MMA lowering](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp).
18. [Triton3.6 documented hardware support](https://github.com/triton-lang/triton/blob/v3.6.0/README.md).
19. [Triton inline assembly API](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html).
20. [PyTorch CUDA event timing](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.Event.html).

Experiment outcomes are original observations in the combined evidence and subsystem result files. None of the external GPU kernels was copied or imported into the original prototypes. All compilation and CPU evidence is labeled independently of the unperformed target and official checks.
