# Nanochat and n-gram ideas for Soup Kernel Challenge

Updated: 30 October 2026, research round 6.  
Status: official starter inspected at a pinned commit; versioned loss/metric, attention, NF4, loading, RMSNorm, and dropout audits; recent-paper screening; CPU representation, replay, and derivative checks. No challenge GPU benchmark or submission was run in this session.

## Session handoff: read this first

- **Task boundary:** ideas from nanochat and modded-nanogpt are useful when credited and implemented originally. Learned n-gram / Engram tables, fixed predictors that change outputs, vocabulary restriction, optimizer changes, and different batches/steps/dropout do not preserve this challenge.
- **Round-6 frozen-weight candidate:** the T4 loading chain first creates FP16 embedding/head values and PEFT widens them. A shared FP16 stored table can reproduce the original FP32 embedding outputs and effective half head operands, conditionally removing 445.125 MiB while idle or 890.25 MiB when the baseline half head operand is live. Preserve tie identity, audit model.dtype consumers and non-autocast calls, and account for conversion/allocator peaks. This replaces the separate head-cache design. See §34.
- **Round-6 cheap replay:** save a half attention increment beside an already-retained residual, then repeat the original FP32 addition for post-attention norm backward. Separately, rebuild q/v dropout operands from saved norm input/rstd and the actual original masks. Each removes another 162 MiB of its respective conditional saved component; they require no matrix replay or new randomness. See §§35–36.
- **Round-6 fusion target:** eager RMSNorm backward includes a full pow(X,1) copy and fragmented pointwise work. Original norm/residual/replay kernels are useful candidates, but frozen-gamma weight-gradient pruning already exists. Recent output-sharing and low-rank-reconstruction proposals do not establish exact savings on this graph. See §§35 and 37.
- **Round-5 next-stage prototype:** reuse native CE's existing FP32 log-softmax result for entropy and Rewind while preserving native NLL's training scalar and native backward. At B=4,T=512, this changes 6,140 vocabulary-row normalizations in 21 calls into 2,048 rows in one call. It retains the 1,187 MiB CE state. Accuracy must still use original logits because normalization can create finite-precision ties. See §30.
- **New conditional attention finding:** the audited Transformers/PyTorch path can request unequal-head GQA on T4 when the mask is absent, rejecting the available efficient backend and selecting math attention. Test the existing explicit-repeat branch only after observing this path; a backend change is numerically nonidentical and must pass all gates. Padded-batch mask sharing and repeat/cast fusion are separate candidates. See §29.
- **New concrete NF4 target:** combine nested-scale decode, offset addition, and weight decode into one original launch, preserving their rounding and dtype stages. Keep native small-M bnb forward dispatch. A reusable 89.25 MiB current-layer slab is a later launch-batching experiment that spends up to 63 MiB more workspace than the largest single matrix. See §31.
- **Paper screening:** the April 2026 NF4 paper's tree-to-lookup improvement is already present in bnb 0.50.2, so its headline gains do not transfer. LoRAFusion supports keeping the small rank-16 intermediate as a graph boundary; its multi-job scheduling does not fit the fixed Soup batches. See §32.
- **Earlier simple fixes remain useful:** remove TRL's 1,184.68 MiB causal-shift logits copy used only for accuracy, and share one differentiable FP32 logits conversion between loss and returned metrics. The latter models 1,780.5 MiB less cast traffic at maximum shape. See §§24–25; neither number is a measured peak-memory reduction.
- **Strongest established saved-component target:** retain the frozen MLP gate/up outputs in their native FP16 representation, preserving the supplied activation's FP32 arithmetic and output. The conditional saved-state reduction is 1,960 MiB. Direct ownership avoids the packing prototype's conversions; do not add the two versions' savings together. See §§17 and 27.
- **Version boundary:** the supplied brief specifies Python 3.13, torch 2.11, and Triton 3.6.0. Starter commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb` documents the remaining dependency pins. This audit uses PyTorch 2.11.0 and Triton 3.6.0 source; the actual T4 build remains unobserved. See §§16 and 29. [D0 §5, D1]
- **Compiler distinction:** audited Triton 3.6 ordinary dot lowering does not use its old Tensor Core path on SM75. Native bnb 0.50.2 does contain SM75 MMA for eligible small shapes. Keep substantial GEMMs in the existing libraries initially; original reductions, casts, and decoding are more concrete targets. See §§10 and 31.
- **Gradient boundaries matter:** shared MLP casts must widen each separate FP16 branch dX before the FP32 join. The analogous masked-attention KV join must widen each head before summing. Ordinary cast motion changes these roundings. See §§27 and 29.
- **Metric contract:** TRL and Rewind receive FP32 logits on the audited AMP path. Entropy uses unshifted attention masks, including prompts and the final real token; accuracy, CE, and row losses use shifted supervision. Preserve row denominators, IDs, counts, order, and logging. See §§13, 24, and 30.
- **Other measured-work candidates:** frozen-head representation caching, 78.09 MiB nested-scale caching, lossless dropout-state compression, CPU-readback batching, one-branch MLP replay, and selective INT8 saved-state compression remain conditional follow-ups. See §§11–13, 18–21, and 26.
- **Rules and scoring:** the starter explicitly permits own recomputation/offloading while keeping built-in switches fixed. Its supplied fused SwiGLU, including rewrites, earns no own-kernel credit. Useful original Triton work must reach 5% from step 3 in every measured run. Compilation, loading, logging, and adapter saving all count in the timed command. See §§2, 8, and 16.
- **Evidence status:** source audits and original CPU algebra/layout probes are complete for the findings below. No PyTorch/CUDA execution, T4 timing, measured GPU peak, integrated training, or submission has occurred. The current experiment order and new probe results are in §38.

## 1. What the n-gram idea actually does

An n-gram is a short sequence of n tokens. A bigram uses two, and a trigram uses three. In the relevant learned-memory approach, those token sequences select trainable vectors that augment a Transformer.

The likely direct reference is [nanochat PR #854](https://github.com/karpathy/nanochat/pull/854). It hashes bigrams/trigrams into a host-memory table and injects retrieved features into attention values. The author reports 91.74 minutes against a 99-minute reference on 8 H100s. The PR was open when checked; these are submitted results, not an independently verified Soup result. Its approximately 72 GiB table is omitted from the saved checkpoint, which therefore cannot run alone. [S1]

This mechanism changes the learned model. It does not prove that an unchanged Qwen training step executes faster. The official Engram project also describes conditional memory as an augmentation of a model backbone. [S2]

The distinction matters even when a paper describes the lookup itself as inexpensive. Lookup cost, improved convergence, model capacity, checkpoint completeness, and equivalent execution of an existing model are different questions.

### Two related speedruns, with different rules

Nanochat measures time to exceed a GPT-2 capability target. Its leaderboard documentation defines reported training time as training iterations, excluding evaluations and logging. [S3]

Modded-nanogpt is a separate FineWeb loss speedrun. Its rules permit architecture and batch/sequence changes that Soup forbids. Its history includes both architectural changes such as bigram embeddings and systems changes such as kernel fusion. [S4]

A port of the host n-gram idea, [modded-nanogpt PR #366](https://github.com/KellerJordan/modded-nanogpt/pull/366), explicitly compares 1,140 candidate steps against 1,285 baseline steps. That is a concrete example of a result involving fewer optimization steps. Soup fixes 64 steps. [S5, D0 §5]

## 2. The rule boundary for our submission

Authority for this assessment is the supplied challenge PDF, Russian pages 1–7. The important sections are §5 (task), §7 (admission), §8 (restrictions), §9 (suggested directions), and §11 (measurement and scoring). [D0]

Round 3 retrieved the official starter and inspected commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`, including its README, task YAML, measurement instrumentation, rule checks, prohibited-package list, and example SwiGLU implementation. These resolve several uncertainties from the earlier rounds. The actual running Kaggle environment and its timings are still uninspected. [D1–D4]

### Fixed task

Qwen2.5-1.5B-Instruct; NF4 base weights; LoRA rank 16, alpha 32 on q_proj and v_proj; batch size 4; sequences up to 512; 256 training rows; one epoch and 64 steps. The final check changes training/evaluation rows and seed while retaining the remaining settings. The target is one Kaggle T4 using fp16. [D0 §5, §8]

### Allowed, forbidden, or conditional

| Candidate | Assessment from the brief | Boundary |
|---|---|---|
| Read speedrun research and derive an original kernel | Explicitly allowed | Credit sources; do not copy their kernel code. |
| Add trainable hashed n-gram / Engram memory | Prohibited | Changes the model and adds trainable parameters. |
| Add fixed n-gram logits or hidden-state corrections | Prohibited as a model change | Freezing the new table does not preserve the prescribed model. |
| Restrict vocabulary using n-gram predictions, top-k, or sampled softmax | Changes the prescribed objective | Full normalization and required metrics would differ. |
| Use n-gram speculative decoding | Irrelevant to the timed training job | SFT consumes known token sequences rather than generating each target autoregressively. |
| Fuse full-vocabulary loss and metrics | Explicitly encouraged direction | Preserve real loss, gradients, entropy, accuracy, and rewind output. |
| Fuse the existing Qwen and LoRA operations | Explicitly encouraged direction | Preserve parameterization, dropout, precision-sensitive behavior, and outputs. |
| Remove padding inside each existing batch using own code | Explicitly allowed | Preserve examples, order, causal boundaries, position IDs, accounting, and logs. |
| Sort by length or repack examples across batches | Prohibited | Changes batch composition/order. |
| Change optimizer, learning-rate schedule, steps, or LoRA dropout | Prohibited | Faster convergence is not an exception. |
| Enable Soup acceleration flags | Prohibited | Includes gradient_checkpointing, activation_offloading, fast_lora, use_cut_ce, and packing. |
| Import ready-made optimization kernel libraries | Prohibited | The brief names Unsloth, Liger, xformers, flash-attn, torchao, and similar packages. |
| Reuse invariant numerical data generated during the run | Conditional engineering approach | Must preserve the model, account for costs, and avoid hidden-data or benchmark-specific behavior. |
| Ship precompiled caches | Prohibited | Compilation belongs inside the measurement. |
| Implement recomputation or offloading in own code | Explicitly allowed by the inspected starter README | Keep built-in settings fixed; all transfer/recompute costs count; preserve training and outputs. |
| Compress retained activations, including INT8 experiments | Suggested direction in the brief | Keep the task and parameters fixed; distinguish lossless storage from approximate backward and pass every numerical gate. |

Sources: D0 §§7–9, pp.3–5, and D1–D3. The judges retain final authority over a concrete implementation.

### Admission requirements that matter to implementation

The hidden evaluation loss may be at most 0.5% higher than baseline. Runtime and GPU peak memory must each be no more than 10% worse. Token counts must match per step. Required entropy, accuracy, and rewind per-row losses must remain within 3%. The first training loss must be within 1%; mean training-curve deviation must be within 3%. The specified Soup tests must pass. [D0 §7]

Own Triton kernels must execute on every measured run and collectively account for at least 5% of training time from step 3 onward. Copied kernels, torch.compile output, unused results, and starter-example kernels even when rewritten do not count. Compare the actual starter before treating a normalization implementation as a qualifying original contribution. [D0 §7.2]

The tolerances are numerical acceptance gates. They do not allow a new optimizer or architecture merely because its loss happens to fit inside the thresholds.

## 3. Transfer one: full-vocabulary loss and metric fusion

### Source and proposed adaptation

Modded-nanogpt PR #199 combines operations around cross-entropy to avoid intermediate tensor materialization. Its discussion also suggests removing the forward loss calculation when only its derivative is used. Soup requires the real forward losses and metrics, so that suggested shortcut does not transfer. [S6]

Our adaptation is an independently implemented reduction path that produces everything Soup needs while avoiding full-size copies. It must not add the source model's softcap or multi-token losses.

The current nanochat source still contains a chunked-cross-entropy TODO. This report proposes an implementation for Soup; it does not claim nanochat master already contains that exact solution. [S7]

### Why the output head matters

The official Qwen configuration has vocabulary size 151,936 and hidden width 1,536. [S8] At the maximum specified batch shape, the illustrative logits tensor contains:

N = 4 × 512 = 2,048 token positions

N × V = 2,048 × 151,936 = 311,164,928 logits

| One logits buffer | Calculated size |
|---|---:|
| FP16, 2,048 rows | 0.5796 GiB |
| FP32, 2,048 rows | 1.1592 GiB |
| FP32, a 64-row chunk | 37.0938 MiB |

These are tensor-size calculations, not measured total memory savings. Actual batch lengths, label shifting, allocator behavior, saved backward state, and callback copies affect the observed peak. A 32-fold reduction in one temporary buffer does not imply a 32-fold reduction in total VRAM.

### Implementation stages

First, keep the existing output projection and remove unnecessary full-logits copies in metric computation. Calculate target loss, argmax, entropy, and per-example sufficient statistics in original Triton reductions. This isolates the metric contract before replacing the larger autograd path.

Second, process the head in token-row chunks and discard each chunk after extracting the necessary statistics. This preserves the full vocabulary for each token.

Third, consider vocabulary tiling if the remaining per-chunk workspace is still costly. That requires a stable reduction across vocabulary tiles and may have a different GEMM efficiency tradeoff.

A simple Python loop that constructs a loss for each chunk can still retain all chunks' backward intermediates. Actual memory reduction requires a deliberate custom-autograd design, recomputation inside backward, or another correctly derived method that avoids retaining the large tensors.

### Exact statistics, without vocabulary pruning

For one token, let z_j be its logit for vocabulary entry j and y its correct target.

Let m = max_j z_j, then define:

S = sum_j exp(z_j - m)

R = sum_j exp(z_j - m) × (z_j - m)

The required quantities follow:

- log normalizer = m + log(S)
- token cross-entropy = m + log(S) - z_y
- entropy = log(S) - R/S
- predicted token = argmax_j z_j
- loss derivative with respect to z_j = p_j - 1[j = y], before masking, reduction weights, and incoming gradient scaling

These equations are a mathematical derivation. They require every vocabulary column. The small statistics replace large stored probability arrays; they do not replace the model's probability distribution.

For stable vocabulary tiling, when a new tile changes the running maximum from m to M, rescale old accumulators by a = exp(m - M):

S_new = a S_old + sum_new exp(z - M)

R_new = a [R_old + (m - M) S_old] + sum_new exp(z - M) (z - M)

Track the target logit and argmax alongside these quantities. Initialize the first nonempty tile explicitly rather than evaluating undefined arithmetic involving an initial negative infinity.

### Masks and backward are part of the contract

Training loss, token accuracy, entropy, and per-row logging can use different masks and denominators. Inspect the actual loaded Soup code rather than assuming all metrics use the supervised-label mask. In particular, do not discard prompt-token logits just because their targets are ignored for supervised loss; a callback may still need their entropy.

Preserve the causal target shift, ignore index, valid-token weighting, per-row aggregation, and argmax tie handling. Retain the original mapping to batch rows even if computation is flattened or unpadded.

Separate the rows needed for forward metrics from the rows with nonzero output-head loss gradients. It may be possible to skip only the latter's zero work in the head backward, while still calculating required prompt metrics. This does not remove prompt tokens from attention or their contribution to later-token gradients.

The output weights may be frozen, but hidden-state gradients through those weights remain necessary. For a custom backward, preserve the incoming loss scale. Converting a small unscaled gradient to fp16 before multiplying by the scale can irreversibly erase values. Keep adequate accumulation precision and place conversions deliberately.

### Optional frozen-head representation cache

Round 4 resolves the earlier cache uncertainty: PyTorch 2.11's autocast cache requires a weight to have `requires_grad=True`, so the prescribed frozen head does not qualify. A reusable representation matching the existing effective GEMM dtype occupies 445.125 MiB for 151,936 × 1,536 FP16 values. See §26. [B5, B7]

Preserve weight tying, the original FP32 parameter, saved parameter identity, and dtype behavior. The baseline already creates this effective FP16 matrix for each head call and needs it for hidden-state backward. Caching extends its lifetime across steps; it does not necessarily add a second simultaneous copy during the head. Its setup, longer lifetime, and eventual release all belong inside normal measured execution.

## 4. Transfer two: fuse existing Qwen operations

Modded-nanogpt PR #197 demonstrates fusion of a linear operation with its activation, including backward. The general lesson is to remove avoidable launches and intermediate memory traffic. [S9]

Qwen uses a different activation path. The relevant architectural reference is the official Qwen2 implementation:

MLP(x) = down_proj[SiLU(gate_proj(x)) × up_proj(x)]

Use that existing function. Do not replace it with the source speedrun's squared ReLU, add QK normalization, or change attention structure. The referenced Transformers version is an architectural reference, not a claim about the challenge's installed version. [S10]

### A practical first fusion — already supplied by the starter

Round 3 correction: the retrieved starter already implements the following SwiGLU forward/backward fusion and saves two wide operands. It may remain in the solution, but its kernels and rewrites do not count as original work. The new saved-state proposals in §§17–19 are additional changes; this subsection is architectural background, not an original submission contribution. [D4]

Keep the existing matrix multiplication implementation initially. Fuse SiLU and elementwise multiplication, and implement the corresponding derivatives together:

If u = gate_proj(x), v = up_proj(x), and a = SiLU(u) × v, then for incoming gradient g:

d_v = g × SiLU(u)

d_u = g × v × sigmoid(u) × [1 + u × (1 - sigmoid(u))]

This is our derivation of the original function's derivative. The implementation should reuse intermediate quantities rather than launch several independent elementwise operations.

Normalization, required casts, and residual or adapter addition offer other fusion boundaries. Preserve fp32 reductions and the reference conversion order initially. Removing a tensor round-trip is useful; indiscriminately moving all computation to fp16 may change numerics enough to fail.

Only after a local measurement should we attempt to absorb these operations into a custom NF4 GEMM. A fused kernel can still lose if its matrix multiplication is slower than the existing implementation.

## 5. Transfer three: algebraic simplification for rank-16 LoRA

Modded-nanogpt PR #166 includes a change in multiplication order to apply a scale to a smaller operand. It also illustrates that apparently equivalent rearrangements deserve numerical validation. [S11]

For the prescribed LoRA path, use row-oriented notation:

Y = X W^T + s [D(X) A^T] B^T

Here W is the frozen base projection, D is the existing dropout operation, A and B are the trainable adapter matrices, and s = alpha/r = 32/16 = 2.

### Proposed original kernel

Compute the existing low-rank intermediate U = D(X) A^T. Then combine the B projection, scale, and addition to the base result in one implementation:

Y = Y_base + 2 × U B^T

The potential saving comes from avoiding separate writes and reads of a full-width LoRA result followed by separate scale/add launches.

Keep the stored A and B parameters unchanged. Permanently absorbing the scale into B, changing its initialization, or optimizing a differently scaled parameter alters optimizer dynamics. If an ephemeral scaled operand is used, gradients must still reach the original B with the correct factor.

The mathematical gradients, before dropout's own derivative and any incoming loss scale, include:

d_B = 2 × G^T U

d_U = 2 × G B

d_A = d_U^T D(X)

The frozen base path still contributes d_X = G W, in addition to the adapter contribution. Do not discard that gradient because W itself is frozen.

Q and V may receive the same normalized input, but their dropout operations must retain the baseline's randomness. Reusing a deterministic cast may be reasonable; reusing one stochastic mask for both branches is a different operation.

This experiment is especially relevant because the brief reports substantial time in elementwise operations and conversions. The actual gain and the qualifying Triton share must be measured. [D0 §2, §9]

Round 2 qualification: an implementation that replaces the B projection with an ordinary Triton dot must beat the current library operation despite the T4 lowering limitation in §10. Start by fusing only the scale/add and related pointwise work around existing matrix multiplications. The full projection fusion remains a measured experiment rather than a default choice.

## 6. The closest n-gram-inspired experiment: frozen first-layer reuse

This is an original engineering hypothesis from this research. It is not the implementation in nanochat PR #854 and has not been approved by the organizers.

Before the first attention operation, some computations are token-local and fully frozen. Subject to verification of the actual loaded model, a token ID determines:

1. Its frozen input embedding.
2. The frozen first-layer RMS normalization result.
3. The frozen base Q/K/V projection results and biases, before rotary position encoding.

The official Qwen2 structure supports this ordering. [S10] One could deduplicate repeated token IDs or use a bounded runtime cache of these exact results.

The current q/v LoRA branches must remain live for every occurrence, with original dropout. Add their current outputs to the cached base results, then apply the original positional handling. Later contextual layers remain live.

### Why it differs from a learned n-gram module

The cache introduces no new predictor and no trainable memory. It memoizes an existing deterministic function. An exact token ID can be used as the key, so there is no need to accept hash collisions.

A reusable entry must be valid for the model's weight version, device, dtype, quantization state, and any other relevant configuration. Cache creation, lookup, allocation, and eviction costs belong inside the measured execution.

Autograd integration must be checked. A fully frozen predecessor need not receive optimizer updates, but hooks, checkpoint assumptions, and the adapter paths must still receive the gradients they require.

### Why this should rank low

The official configuration gives hidden width 1,536, 12 query heads, 2 key/value heads, intermediate width 8,960, and 28 layers. The key/value width is therefore 256. [S8]

First-layer base QKV weight count:

1,536 × (1,536 + 256 + 256) = 3,145,728

All Transformer linear weights, ignoring biases:

28 × [2 × 1,536² + 2 × 1,536 × 256 + 3 × 1,536 × 8,960]
= 1,310,195,712

The first quantity is approximately 0.2401% of the second.

For the same token count, this is a rough ratio of linear forward arithmetic. It excludes the output head, backward, and non-matmul work; it is not a measured runtime bound. Still, it shows why eliminating the entire first QKV base projection is unlikely to dominate the score.

Deduplication, gathers, dynamic allocation, cache misses, and inefficient small GEMMs may consume the small saving. A whole-vocabulary cache adds substantial memory. Test briefly only if profiling reveals an unexpected first-layer cost.

### Why caching repeated phrases deeper in the model is invalid

A repeated local n-gram does not identify a contextual hidden state. That state can also depend on all earlier tokens, positions, masks, current LoRA parameters, and dropout.

Across optimizer steps, the adapter changes. A previously cached contextual result becomes stale even though the quantized base weights stay frozen.

Within one step, sharing an identical complete prefix is more subtle than matching an n-gram. Correct positions, masks, stochastic decisions, and backward accumulation would all need to agree. Independent dropout occurrences commonly defeat naive sharing.

For this hackathon, general prefix/KV reuse is not a first-line experiment.

## 7. Padding, data movement, and short-run economics

### Remove unnecessary rows, preserve all vocabulary columns

The supplied brief expressly permits own-code padding removal inside each existing batch. [D0 §8]

That can remove work across many layers, unlike the small first-layer cache. Preserve example boundaries and positions. Flattened examples must not attend across row boundaries. Restore per-example loss and metrics using a faithful mapping.

Do not sort examples across batches, combine batches, change step counts, or alter token accounting. Changing tensor shapes must also be assessed for its effect on dropout draws and numerical behavior.

Padding removal is compatible with the prescribed objective only where the corresponding computation has no required effect. Vocabulary pruning changes the softmax denominator even when a removed token is unlikely.

### Runtime representation reuse

The useful principle behind host-table engineering is to avoid repeated representation work and unnecessary transfers. For Soup, candidate invariant data include NF4 metadata, fixed rotary tables, and carefully bounded weight representations.

Dense copies of every NF4 matrix can consume the memory saved elsewhere. Consider a bounded scratch representation only when actual profiling demonstrates repeated useful reuse. Preserve nested quantization scales and dtype/rounding behavior.

Custom offload is explicitly permitted by the retrieved starter, but its transfer costs and memory lifetimes still need measurement. Do not flip the prohibited built-in offload setting. See §18 for the transfer budget.

### T4 requires its own launch choices

NVIDIA's Turing guide describes fp16 Tensor Core computation and a smaller shared-memory budget than many modern GPU optimizations assume. Hopper-specific kernels and their tuning parameters are not transferable performance evidence. [S12]

Start with modest tile sizes and a small number of candidates. Expensive autotuning and compilation count against the final result.

### Setup cost must amortize over only 64 steps

Let C be added setup time, and let delta be the saving per step. Ignoring other changes:

Net time saved = 64 × delta - C

| Added setup time | Average per-step saving needed to break even |
|---|---:|
| 1 second | 15.625 ms |
| 2 seconds | 31.25 ms |
| 4 seconds | 62.5 ms |
| 8 seconds | 125 ms |

These are arithmetic thresholds, not performance predictions. They apply to graph capture, cache construction, compilation, weight repacking, and similar setup work.

Soup measures full command time through saving the adapter, and samples peak GPU memory over the entire run. The prescribed order is baseline, solution, solution, baseline, averaging each side's two results. [D0 §11]

## 8. Ranked implementation queue

The following broad queue reflects the direction established through round 3. Round 4's more specific experiment order in §28 supersedes its first implementation steps; the remaining candidates stay available. These rankings are engineering judgments, not measured speedups.

| Rank | Experiment | Main benefit sought | First decisive check |
|---|---|---|---|
| 1 | Losslessly pack verified widened-FP16 gate/up saved operands | Large retained MLP activation state | Reconstruct original FP32 values before unchanged backward; verify native operands and gradients. |
| 2 | Fuse full-vocabulary loss/entropy/accuracy/per-row reductions; then chunk the output head | Peak memory, logits traffic, useful original kernel coverage | Match every output and gradient before full-run benchmarking. |
| 3 | Original LoRA scale/add, normalization, cast, or q/v dropout-state work | Repeated pointwise work and saved activations | Preserve precision and randomness; the provided SwiGLU kernels cannot count. |
| 4 | Own within-batch unpadding | Remove work across the full model | Preserve attention boundaries, token accounting, masks, and stochastic behavior. |
| 5 | Fuse NF4 decoding/casts and test decoded-scale caching | Repeated conversion and nested-scale decoding | Inspect actual dispatch, dequantization time, and added memory; avoid an unmeasured full Triton GEMM. |
| 6 | Own one-branch MLP replay or selective INT8 activation compression | Additional memory after lossless FP16 packing | Compare incremental saving, replay overhead, and gradient/quality risk. |
| 7 | Adapter-only state preparation; batch row-statistic and finite-check transfers | Final-save and synchronization overhead | Preserve every tensor, config, log, and failure condition. |
| 8 | Same-algorithm AdamW fusion, terminal backward compaction, or GC tuning | Smaller measured overheads | Stop if profile contribution and 64-step payback are too small. |
| 9 | Frozen first-layer memoization or selective dense-weight caches | Narrow repeated representation work | Stop if lookup, setup, or extra memory exceeds the small available saving. |

Do not spend the main implementation budget on a new n-gram architecture, new optimizer, new activation, auxiliary loss, inference speculative decoding, or fewer training steps.

### Verification needed before accepting a candidate

Start with correctness at the function boundary, then verify its integrated effect. Useful cases include ignored labels, empty supervised subsets, repeated tokens, ragged sequences, extreme logits, odd strides, and backward under loss scaling. These are tests of meaningful failure modes rather than a duplicate implementation.

Run the official rule checks, Soup tests, fixed training job, and evaluation without modifying the harness. Record total command time, sampled peak VRAM, per-step losses, token counts, required metrics, own-Triton time share, and sources.

Aggressive fusion can reduce a single kernel's measured share below the 5% requirement. Meet the requirement by covering additional useful original operations. Artificially slow kernels and unused work are not a valid contribution.

Score improvements must exceed the brief's 1.05 ratio threshold to score in the respective category. Speed has 35 points and memory 30; memory breaks ties. Optimize the measured combination without allowing a correctness or reproducibility regression. [D0 §11]

## 9. What has and has not been established

Established in this session:

- The uploaded rules explicitly bar the trainable n-gram architecture.
- Primary nanochat/modded-nanogpt sources contain relevant examples of lookup memory, fusion, and algebraic rearrangement.
- Several speedrun freedoms and timing conventions differ materially from Soup.
- Full-logits workspace and first-layer arithmetic estimates were calculated from the specified batch and official Qwen dimensions.
- A concrete implementation queue and the main semantic hazards have been identified.
- The exact Soup 0.77.0 wheel and pinned Triton 3.6.0 matmul pass were inspected; their provenance is recorded in §15.
- A new dropout-state compression proposal passed CPU bit-pattern reconstruction checks and an independent finite-difference check of its underlying LoRA gradients.
- Concrete row-logging and pre-save validation functions contain opportunities to batch CPU transfers without deleting required work.
- Round 3 resolved the official starter and exact recommended dependency versions, including explicit permission for own recomputation/offloading and the provided SwiGLU exclusion.
- Versioned source supports lossless FP16 storage of the frozen MLP's widened gate/up results. Exhaustive finite-FP16 CPU reconstruction and local derivative comparisons passed; this is not yet an integrated T4 result.
- CPU probes quantify local gradient errors for illustrative INT8/INT4 compression and show why outlier handling matters.

Not established:

- A speedup or memory score on Kaggle T4.
- Actual Kaggle loaded-state behavior, saved tensor lifetimes, compiler output, and measured adherence to the pinned environment. The starter's documented versions and patch points are known.
- Whether a concrete cache, compression, replay, or offload implementation passes the numerical and performance gates.
- A completed solution, runnable Kaggle notebook, or winning rank.

The current experiment queue is in §38; the source-derived lossless MLP storage specialization remains a leading memory candidate. The decisive missing evidence is the prescribed T4 run rather than another model architecture.

## 10. Round 2: a compiler constraint changes the GEMM plan

T4 has FP16 Tensor Cores, but the compiler must actually select their instruction path. The pinned Triton 3.6.0 source file `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`, in the ordinary `triton::DotOp` rewrite, returns failure for compute capability below 80 and emits a remark that this case falls back to the FMA path. T4's compute capability is 75. [S12, R1]

This is consistent with upstream PR #5066, which deprecated that lowering. A later restoration proposal, #9970, was closed without merging when checked. The pinned source, rather than an old tutorial or the existence of a restoration PR, determines the relevant behavior. [R2, R3]

Consequences for this challenge:

- A conventional `tl.dot` kernel copied conceptually from an H100 speedrun cannot be assumed to use Tensor Cores on this installation.
- A large NF4-plus-GEMM rewrite can save temporary traffic yet lose much more time in the matrix product.
- Existing PyTorch/cuBLAS GEMMs can remain the compute engine while original Triton kernels handle loss reductions, decoding, casts, activation derivatives, dropout storage, and additions.
- Manual inline PTX or another explicitly supported MMA route is a separate advanced experiment. This finding does not prove that no custom T4 Tensor Core kernel is possible.

Before investing in a matrix kernel, compile one representative dot at the exact pinned version and inspect its generated code and timings on T4. Do not upgrade Triton or change the fixed environment to make an external kernel work.

## 11. Round 2: compress retained LoRA dropout state without sharing randomness

Round-6 extension: §36 develops exact operand replay from already-retained norm state, removing the shared operand in eligible later layers. Its savings are incremental to this representation.

### The new proposal

The q and v adapters receive the same normalized hidden state, but their dropout realizations must remain separate. Merely replacing both branches with one dropped-out input changes the function. PEFT's four-bit LoRA path explicitly applies dropout in the adapter path. Soup's own optimization discussions also identify activation retention and shared-input handling as relevant costs. [R4, R5]

The proposed change is to the representation saved for backward. It adds no parameter and no predictor. It is an original combination to investigate, not a claim of a newly published method or organizer approval.

Let `Dq` and `Dv` be the actual effective operands consumed by the two A projections, after the existing dropout and dtype conversions. Let `Mq` and `Mv` be the corresponding original Boolean keep masks. If input, dropout probability, arithmetic, and conversion path match, a position kept by both branches has the same effective value.

Construct one shared array from the already realized operands:

```text
S = select(Mq, Dq, Dv)
save S
save bitpack(Mq), bitpack(Mv)
save Zq = Dq @ Aq.T and Zv = Dv @ Av.T
```

These arrays are the retained activation state, not the complete backward context. The implementation also needs the original A/B parameters or appropriate saved references, plus the dropout and dtype-conversion metadata.

For backward, reconstruct each branch using its own mask and the correct signed zero at dropped positions. Initially reconstruct one operand at a time and retain the library GEMMs. Later, packing, unpacking, and masked gradient additions can become useful original Triton kernels.

Generate the masks with the original dropout calls and their original shapes, dtypes, and ordering. Packing a mask after it has been drawn preserves its information; replacing the random generator, combining two RNG calls, or reusing one mask does not establish equivalent randomness.

This approach requires custom autograd integration that actually releases the original retained operands and masks. Keeping the original graph alive and adding a compressed copy would increase memory. Use normal saved-tensor mechanisms, preserve their lifetime semantics, and do not retain duplicate tensors as arbitrary Python context attributes.

### Conditional storage calculation

At maximum batch shape, one full-width operand has `2,048 × 1,536 = 3,145,728` elements. Across 28 layers:

| Saved-state component | Two-operand baseline | Proposed representation |
|---|---:|---:|
| FP16 operands | 336 MiB | 168 MiB |
| Alternatively, FP32 operands | 672 MiB | 336 MiB |
| Two byte masks versus two packed one-bit masks | 168 MiB | 21 MiB |
| Total, assuming FP16 operands | 504 MiB | 189 MiB |
| Total, assuming FP32 operands | 840 MiB | 357 MiB |

The differences are **315 MiB** and **483 MiB**. These are calculations from the stated dimensions, not GPU measurements. They assume two distinct retained operands, two byte masks, and all 28 layers' states live together. Do not count multiple references to the same storage as additional memory. Do not assume the FP32 case merely because adapter parameters are FP32: autocast can produce FP16 matrix operands.

Actual savings may be substantially smaller with short sequences, different saved tensors, or existing checkpointing. Forward can temporarily materialize both branch operands before compression, and backward needs workspace. The observed peak can therefore fall by less than the retained-state total. Keep the prescribed checkpointing setting unchanged.

The two rank-16 intermediates `Zq` and `Zv` occupy only 3.5 MiB in FP16 across these 28 layers. Saving those is a better first choice than paying extra matrix products to recompute them.

### Why arithmetic order and reconstruction matter

A shared operand must reproduce the effective value already consumed by the original matrix multiplication. Casting an FP32 input to FP16 before dropout scaling generally differs from scaling in FP32 and then casting. The CPU probe found different FP16 bit patterns in 25,270 of 100,000 sampled values after that reorder. This is a rounding identity check, not a 25% training error.

Use a selection operation for dropped positions. Multiplying a retained infinity by zero creates a NaN, even when the original branch dropped a finite input before the later FP16 conversion overflowed. Preserve signed zero where bit-pattern identity is required. Arbitrary nonfinite source inputs require explicit handling or fallback; the finite-input proof is not a universal IEEE-754 equivalence proof.

### Backward derivation

All equations in this subsection are real-arithmetic identities; they do not authorize changing the reference operation or conversion order. For either adapter branch j, use:

`Zj = Dj @ Aj.T`

`Yj = s * (Zj @ Bj.T)`, where `s = 2`.

For incoming gradient `Gj`:

`Pj = s * Gj @ Bj`

`dBj = s * Gj.T @ Zj`

`dAj = Pj.T @ Dj`

`dXj = (Mj / (1 - p)) * (Pj @ Aj)`

For an implementation, first obtain `Gscaled` through the reference scaling operation in its original dtype, then calculate `Pj = Gscaled @ Bj` and `dBj = Gscaled.T @ Zj`. Moving the scale across a GEMM can change rounding or overflow. Reproduce all reference casts and AMP behavior, and add the frozen-base input gradient. Do not apply dropout's reciprocal scale again to `dAj`, since it is already present in `Dj`.

### CPU checks completed in round 2

The experiment used NumPy with seed 3030. Its compression case had a 257 × 1,536 FP32 input, dropout probability 0.05, separate q/v masks, and an explicit reference ordering: multiply by the mask, multiply by the FP32 reciprocal keep probability, then cast to FP16. It included negative zero and finite source values that overflow during the final FP16 conversion. The probability is an illustrative test setting; it does not authorize a change to the challenge's dropout configuration.

| Check | Result |
|---|---|
| Number of finite source elements | 394,752 |
| Pack/unpack of both separate masks | Exact |
| Both-kept q/v operand values | Bitwise equal |
| Reconstruction of each full FP16 operand | Bitwise equal to its reference |
| Naive multiplication by a zero mask | Introduced 1 new NaN in the probe |
| Fixed-mask LoRA derivative, FP64 finite differences | Maximum absolute discrepancy below 5.86 × 10^-10 across X, Aq, Av, Bq, Bv |
| Incorrectly sharing one mask | Changed the toy adapter output; maximum absolute difference about 5.47 |

The derivative check used a separate small dense-base model with fixed masks, including the base q/k/v input-gradient paths. It checks the algebra independently of the compression test. No PyTorch, Triton, CUDA dropout generator, NF4 kernel, optimizer step, or end-to-end training result was tested locally.

## 12. Round 2: target NF4 representation work before a new matrix kernel

### Inspect dispatch rather than the operator's name

The current bitsandbytes autograd source calls an operator named `gemm_4bit` in forward. Its name alone does not establish that a given training shape uses a fused four-bit Tensor Core matrix kernel. Upstream discussion of the new Triton implementation describes shape dispatch and fallback to dequantization plus `F.linear`; token-row counts above 1,536 use the older path in that implementation. The maximum Soup shape has 2,048 rows. The inspected backward also dequantizes the base matrix before multiplying the incoming gradient. [R6, R7]

Round 2 used the inspected upstream implementation. Round 3 resolved the starter's documented bitsandbytes pin to 0.50.2. Round 5 now establishes its concrete T4 dispatch: native SM75 MMA exists for eligible small-M forward shapes, while maximum M=2,048 uses dequantization plus native matmul. The upstream 1,536-row threshold above is historical context, not the pinned version's T4 dispatch rule. See §31 for the actual per-projection ceilings, backward path, and precision boundaries. Verify the loaded version and trace before specializing. [D1, A1–A3, C10]

### Cache scales before caching all dense weights

The seven transformer projections contain 1,310,195,712 weight elements across 28 layers, calculated in §6. Assuming NF4 blocks of 64 elements:

| Representation retained for all 28 layers | Calculated size |
|---|---:|
| All seven projections as FP16 dense weights | 2,499 MiB, about 2.44 GiB |
| Only q/v projections as FP16 dense weights | 147 MiB |
| One decoded FP32 scale per 64 weights, all seven projections | 78.09375 MiB |

The scale-cache proposal is to decode the nested quantization of the block scales once during the measured run, retain those exact effective scales, and keep decoding packed NF4 weight values as needed. It avoids repeatedly decoding the second-level metadata while retaining the much smaller four-bit weights. Use only the immutable state belonging to the loaded frozen weights, and preserve the original quantization state for checkpoint compatibility.

This saves only the repeated nested-scale work, not the whole dequantization operation. Whether it is worth about 78 MiB depends on the trace. A one-time initialization kernel also cannot satisfy a recurring own-Triton requirement from step 3 onward.

Next, test an original kernel that combines the existing NF4 lookup, scale application, and required output cast into the dense buffer consumed by cuBLAS. Preserve the codebook, nibble order, nested offsets/scales, and reference rounding. Do not enable a prohibited built-in quantization acceleration flag to obtain this behavior.

### Do not overstate baseline memory

The inspected bitsandbytes autograd context retains the quantized weight and quantization state. It does not establish that every decoded full matrix remains saved for the whole backward pass. For one Qwen MLP projection, a dense FP16 matrix is 26.25 MiB. Removing such a temporary does not automatically remove a persistent 2.44 GiB buffer. [R7; dimensions S8]

Tiling the input-gradient product can reduce one decoded matrix workspace, but increases the number of GEMMs and repeated reads. For example, decoding 1,024 columns at a time of the down-projection matrix gives a 3 MiB FP16 weight tile instead of a 26.25 MiB full matrix, with nine tiles for its width of 8,960. That is a secondary workspace tradeoff to measure, not a guaranteed large memory win.

Selective q/v dense caching uses 147 MiB but covers only about 5.9% of these transformer weight elements. It may be useful if those projections have disproportionate launch cost; matrix-element coverage alone does not show that.

## 13. Round 2: concrete small opportunities in Soup 0.77.0

### Batch the CPU transfer for required row logging

In the exact wheel, `soup_cli/trainer/rewind_hf.py::row_losses` iterates over batch rows, calculates a real per-row supervised-token mean, converts that mean to a Python float, and converts its token count to a Python integer. Batch size four therefore causes eight scalar conversions. This is evidence of repeated readbacks, not proof of eight independent long GPU stalls. The alternative `row_losses_from_token_losses` path also converts each row's values individually. [R8]

An original replacement can aggregate all row statistics on the GPU, transfer a small result array once, and construct the same row records on the CPU. Preserve row IDs, order, timestamps or step association, causal shift, masks, denominators, and the rule that an empty supervised row returns loss 0 and count 0. Finish required logging inside the timed command.

The helper calculates cross-entropy in its input logits' dtype, then casts the small token-loss result to FP32 for its row mean. **Round 4 correction:** in the pinned T4 path, Accelerate already converts returned logits to FP32 before TRL and Rewind receive them, while `ForCausalLMLoss` also uses FP32. There is no demonstrated FP16/FP32 mismatch between these two losses in this task. A shared implementation must still preserve causal shifting, ignored labels, per-row denominators, batch normalization, and numerical reduction behavior; the batch training scalar cannot replace individual row means. See §§24–25. [R8, B1, B3, B4]

### Batch pre-save finite checks

`soup_cli/trainer/sft.py::_assert_finite_training_state` checks final logged metrics and then loops over trainable floating-point, non-meta parameters, calling an all-finite reduction and `.item()` for each. The q/v rank-16 configuration normally has four adapter tensors per layer, or 112 such tensors. [R8; count derived from D0 and S8]

Compute the same per-parameter flags, transfer them together, and preserve the first offending parameter's error and the refusal to save a nonfinite model. Keep the original treatment of metric history and excluded parameters. This can reduce an end-of-command synchronization tail. It is not a reason to delete the validity check or claim meaningful training-kernel coverage.

### Borrow nanochat's optimizer engineering selectively

Nanochat isolates its AdamW update and supplies changing scalars as runtime zero-dimensional tensors to avoid recompiling for scalar-value changes. This is a useful implementation lesson. Its optimizer mix and schedule are not transferable to a fixed-optimizer challenge. PyTorch's documented CUDA default can already select foreach when neither foreach nor fused is specified. [R9, R10]

The prescribed adapters contain a calculated 2,179,072 trainable elements, about 8.3125 MiB in FP32. A simple FP32 AdamW traffic model—read parameter, gradient, first moment, and second moment; write parameter and both moments—is about 58.19 MiB per step. This is a small-parameter launch target, not the main activation-memory target.

Only test an independently implemented equivalent update if profiling shows material cost and the allowed replacement preserves the fixed optimizer configuration. Match parameter groups, weight decay, epsilon placement, bias correction, state dtypes, per-parameter step counters, absent gradients, clipping order, and AMP overflow/skip behavior. Do not toggle to a different optimizer preset.

For our own Triton work, keep learning rate and step-dependent scalars as runtime arguments where appropriate. Avoid accidental per-step specialization over a 64-step job. Shape specialization should be deliberate and bounded; preserve numerical semantics when changing argument representation.

## 14. The next decisive experiment

Start with a short run of the actual prescribed environment, before broad implementation:

1. Record resolved Soup, PyTorch, Triton, Transformers, PEFT, and bitsandbytes versions, the challenge config, and the current starter's original kernels.
2. Capture operator timings after warmup, plus the full-command loading/compilation/saving timeline. Inspect actual four-bit dispatch and the generated instruction path for any proposed custom dot.
3. Use an alias-aware saved-tensor census for the q/v adapter paths: shape, dtype, unique storage, mask representation, and overlapping lifetimes. Do not sum repeated references to one allocation.
4. Preserve baseline dropout calls in a focused compression prototype. Compare the exact operands, masks, forward outputs, A/B gradients, and input gradients under native AMP. Keep the original checkpointing setting.
5. If those checks pass, compare peak memory and command time with and without compression. Reject it if reconstruction or temporary buffers consume the proposed benefit.
6. Integrate useful full-vocabulary reductions and pointwise/decode kernels, then run the official tests and baseline–solution–solution–baseline measurement unchanged.

Apply a concrete stop criterion to each experiment. Reject a generic Triton GEMM if it loses to cuBLAS at the actual shape. Reject a cache if initialization and added memory cost outweigh repeated work saved. Reject a logging change if it changes or omits required records. Reject compression if actual storage aliases or lifetimes invalidate its budget.

Keep all gains as hypotheses until the complete run passes correctness and own-kernel admission. A faster isolated kernel or lower `max_memory_allocated` value alone does not establish a challenge score, whose memory sampling and whole-command timing are specified separately.

## 15. Source and experiment provenance for round 2

The public PyPI wheel inspected was `soup_cli-0.77.0-py3-none-any.whl`, with SHA-256:

`f1346e79cecb451211b714df06759b8b7dc9a6fd2fc086883297c35d1fe60ebb`

PyPI's provenance identifies publishing commit:

`846888a045b31084452fa08a684a7a38c02f5f87`

The relevant source paths in that repository are under `src/soup_cli/`. This inspection read the release archive; it did not install or execute Soup locally. In particular, the release's single-projection fast-LoRA code includes an explicit opt-in dropout path, while its grouped QKV path has stricter eligibility. Do not infer that every current upstream fast-LoRA variant rejects nonzero dropout from an older issue. Enabling that built-in feature remains prohibited for this submission. [R8]

The pinned compiler file came from tag `v3.6.0`, path `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`, SHA-256:

`bc73e8363a12c3cb050dc330bfe2c10bc3f3696cd245727f46405952edbeae68`

CPU experiments used NumPy, seeded as described in §11. Their purpose was to check the compression identity, a concrete rounding hazard, and LoRA derivative algebra. They do not supply CUDA performance evidence.

## 16. Round 3: the official starter resolves important uncertainties

The official repository was retrieved and inspected at commit:

`c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`

The README, task YAML, `harness/common.py`, `harness/_train.py`, `harness/check_rules.py`, and `solution/swiglu.py` were read. No harness or task files were changed. [D1–D4]

### Newly established task details

| Item | Verified specification |
|---|---|
| Dependency pins in the starter instructions | Soup 0.77.0; Transformers 5.16.1; PEFT 0.20.0; TRL 0.29.1; bitsandbytes 0.50.2; Accelerate 1.14.0 |
| Learning rate and accumulation | 2e-4; gradient accumulation 1 |
| Actual adapter scope | q_proj and v_proj; rank 16, alpha 32, dropout 0.05 |
| Baseline checkpointing | Off |
| Attention and optimizer | SDPA; adamw_torch; cosine schedule |
| Supervision | Assistant tokens; required entropy and accuracy still use the trainer's full-logits path |
| Own recomputation/offload | Explicitly permitted, with costs inside the measurement |
| Provided example | Fused SwiGLU forward and backward; never counts as the participant's own kernel, including rewrites |
| Measured runs | Fresh process and empty compiler caches; one visible T4; same allocator settings; Hub offline after warmup download |

The installed package metadata must still be recorded on the actual T4. These are the documented versions and source references, not a claim that a live run was inspected.

The example's README reports approximately 2,000 MiB less peak memory by itself. That is the organizer's measurement of the provided example, not a result produced here. New storage calculations below use the example's existing two-operand state as their starting point. Do not add the organizer's measured peak reduction to our calculated retained-state reduction and present the sum as an observed result. [D1, D4]

### Prohibited-library list and measurement visibility

The inspected `BANNED_MODULES` list is:

`unsloth`, `unsloth_zoo`, `liger_kernel`, `xformers`, `flash_attn`, `flashinfer`, `apex`, `torchao`, `bitblas`, `gemlite`, `hqq`, `cut_cross_entropy`, `deepspeed`, `vllm`, `awq`, `auto_gptq`, `gptqmodel`, `exllamav2`, `marlin`, `quack`, `tilelang`.

The harness makes a narrow runtime exception when PEFT itself imports torchao; importing it in the solution remains prohibited. Installing packages during the measured run is also prohibited. [D3]

Own Triton launch timing wraps `JITFunction.run` at the Python launch boundary. Graph replay bypasses that boundary. The README therefore explicitly says to launch own kernels normally, not through CUDA graphs. The step probe also synchronizes at step boundaries; do not plan around removing those prescribed barriers. [D1, D3]

## 17. Best new route: lossless storage based on where the dtype came from

### The source-level observation

The saved dtype can be wider than the information actually present in a tensor. In this fixed workload:

1. The residual/norm path supplies FP32 inputs.
2. The frozen MLP gate/up `Linear4bit` modules use the configured FP16 compute dtype.
3. bitsandbytes casts their inputs to that compute dtype and produces FP16 matrix-product results.
4. `Linear4bit.forward` widens its result back to the original input dtype, FP32.
5. The provided SwiGLU autograd function saves those gate/up operands for backward. [D1, D4, A1–A4]

The inspected bitsandbytes 0.50.2 constructor marks an explicitly supplied compute dtype as already set, so the later automatic dtype selection does not override it. Its custom CUDA output uses the input operand's dtype; its fallback feeds the same effective dtype to the linear operation. Qwen's gate/up projections are bias-free and receive no LoRA adapter in this task. Those details establish the proposed specialization's scope. [A1–A4]

For a finite value `h` already in FP16:

`float32(float16(float32(h))) == float32(h)`

The proposed change is to store the saved operands in their original FP16 representation and restore FP32 before executing the existing backward. It introduces no extra matrix multiplication and no new activation quantization error for values with this proven origin.

This is an independently derived implementation proposal. It is not a claim that every FP32 activation can be compressed losslessly.

### The precise boundary matters

Apply the specialization only to verified outputs of the frozen, bias-free, FP16-compute gate/up projections before further arithmetic. Do not extend it automatically to RMSNorm outputs, residual sums, `SiLU(gate)`, the SwiGLU product, or q/v outputs after LoRA addition. Those can contain genuine FP32 information.

The first prototype should change only how the two operands are retained. Keep the original forward, original output dtype, original matrix operations, and original backward arithmetic.

PyTorch's saved-tensor hooks provide an appropriate seam: the pack result replaces the saved tensor, and unpack must reconstruct its original value, shape, dtype, and device. Scope the hooks to the intended activation-saving operation, and avoid retaining the original allocation in hook closures or packed objects. [A5]

Conceptually:

```text
pack verified FP32 gate/up:
    retain a detached FP16 representation and original metadata

unpack for the existing backward:
    reconstruct FP32 on the original device with the required layout
```

**Restore FP32 before the existing backward sees the operands.** The starter converts the incoming gradient to `gate.dtype` and allocates gradient outputs like the saved gate/up tensors. Returning FP16 directly would therefore change incoming-gradient rounding and output-gradient dtypes. A storage optimization must not silently change that arithmetic. [D4]

Round 4 adds a distinct implementation route in §27: own the two frozen projections and the activation boundary together, and call the supplied kernels with explicit pointer dtypes. Their arithmetic already widens loaded operands to FP32, so this route can retain native FP16 gate/up values without reconstructing full FP32 arrays. It must explicitly retain FP32 activation output and incoming-gradient handling. The warning above remains binding for the unchanged starter wrapper; simply passing half tensors to that wrapper is not the proposed optimization.

Independent original packing/unpacking kernels may perform this useful work, while the provided SwiGLU kernels remain excluded from own-kernel accounting. Whether packing/unpacking contributes enough time is a measurement question; combine it with useful full-vocabulary reductions rather than trying to inflate its runtime.

### Conditional maximum-shape budget

For B=4, T=512, intermediate width I=8,960, and L=28, each wide activation has 18,350,080 elements. Retaining two FP32 arrays per layer occupies:

`2 × 28 × 2048 × 8960 × 4 bytes = 3,920 MiB`

Retaining their exact FP16 representations occupies 1,960 MiB. The calculated difference is **1,960 MiB, approximately 1.91 GiB**. [Dimensions: S8; storage layout: D4; calculation: this report.]

This is retained-state arithmetic. Actual sequence lengths, other references to the same tensors, transient allocations during pack/unpack, and the location of the overall peak determine the measured VRAM reduction. If the operands are already FP16, this particular saving is zero.

### CPU evidence obtained in round 3

An original NumPy probe enumerated all 65,536 FP16 bit patterns and selected all **63,488 finite patterns**, including both signed zeros. Every finite value survived FP16 → FP32 → FP16 → FP32 with identical FP32 bits.

Three additional synthetic SwiGLU probes, each with 64 × 8,960 values in each operand, produced bitwise-identical FP32 input gradients after packing and restoring the operands. The incoming gradient stayed FP32. This checks the representation and local derivative computation under the probe's explicit arithmetic.

As a counterexample to blanket casting, 99,992 of 100,000 independently sampled generic FP32 values changed bit pattern after an FP16 round trip. This indicates loss of exact representation, not a 99.992% task error.

No CUDA execution was performed. Test actual NF4 outputs, native conversion/subnormal behavior, nonfinite handling, AMP overflow decisions, and all adapter gradients before calling the integrated implementation equivalent.

## 18. Exact extension: recompute one frozen MLP branch

The now-confirmed rules permit own recomputation. Qwen's up projection is a useful narrow target: it is frozen, bias-free, token-local, and has no dropout in this task. A backward-time replay can reproduce its saved operand while retaining the gate and the narrower input. [D1, A4]

The first replay version should save the original FP32 MLP input, replay `up_proj` under the original computation context, and reproduce the exact cast/contiguous conversion performed before the starter saves `up`.

A later specialization could save the original projection's **effective FP16 input** instead. This does not assert that the FP32 normalized input is FP16-representable: the projection already rounds it to FP16 in the baseline. Reuse that exact rounded operand and the same NF4 operation, then widen the output as before. [A1–A3]

Compare every alternative against the same retained state:

| Representation of the two MLP operands, all maximum-size layers | MiB retained | Additional reduction after both operands are losslessly packed to FP16 |
|---|---:|---:|
| Starter's two FP32 operands | 3,920 | — |
| Lossless FP16 representations of both | 1,960 | Reference |
| FP16 gate + original FP32 narrow input; replay up | 1,316 | 644 MiB |
| FP16 gate + original effective FP16 projection input; replay up | 1,148 | 812 MiB |
| INT8 both operands + FP32 scales, blocks of 128 | 1,010.625 | 949.375 MiB; approximate backward |

These are alternatives, not additive savings. The replay rows trade an extra matrix multiplication for memory. The INT8 row trades backward precision and packing work for memory.

### Replay correctness and cost

Preserve the forward autocast state, effective dtype, shape, stride/layout, quantization state, and relevant dispatch. Backward does not automatically recreate an arbitrary forward autocast context. PyTorch documents explicit handling for custom autograd functions. [A6]

`no_grad` can change which bitsandbytes wrapper path executes because dispatch checks grad mode and `requires_grad`. The inspected packed-NF4 paths ultimately reach the relevant GEMM operation, but the replayed result still needs an actual equality check. If necessary, use the original training forward path with a short-lived detached-input graph and detach its result. Do not retain the original wide tensor in the replay descriptor just to identify it.

Replaying one up projection per layer adds about 14.706% of the transformer-only forward-plus-input-backward linear arithmetic, or about 12.482% when the frozen vocabulary head is included. These are FLOP ratios, not time estimates. Replay also repeats decoding and casting; the 28 up matrices alone correspond to 735 MiB of FP16 decoded-weight writes when that fallback path executes. Preserve matrix multiplication in cuBLAS initially.

Because the basic lossless-packing route needs no replay, it ranks ahead of this extension. Retain replay only if the extra memory score is worth its measured time cost and every admission gate still passes.

### Own offload: legal, but budget the transfers

Offloading is now a confirmed permitted implementation direction. After lossless packing, moving both wide operands to CPU and back across all maximum-size layers would transfer about 3,920 MiB per step, before transfer metadata and workspace. For an offloaded payload F and measured effective link bandwidth B, the unoverlapped transfer budget is approximately `2F/B`.

Use explicit stream dependencies and lifetime management if testing overlap. Preserve tensors until consumers finish, avoid per-tensor global synchronization, and include all transfers in the command. The first experiment should be a bounded subset of layers with measured overlap; the host transfer path is unnecessary for the simpler lossless GPU-storage improvement.

## 19. Another research direction: selective activation quantization

The brief explicitly suggests activation compression and INT8. Relevant primary research includes HyC-LoRA (MLSys 2025), ActNN (ICML 2021), GACT (ICML 2022), and PRAC (ICML 2026). [D0 §9, A7–A10]

HyC-LoRA identifies nonlinear saved activations as a major LoRA memory cost and uses structured outlier handling plus inter-operator techniques. Its experiments use settings that differ from this challenge, including BF16 and adapters across more projections. Borrow the storage and outlier ideas with attribution; its reported model-quality and memory results do not establish a Soup/T4 result. Its implementation cannot be imported as a ready-made optimization library. [A7]

ActNN and GACT provide broader activation-compression methods; GACT explicitly models approximate gradients and chooses precision using gradient sensitivity. PRAC explores principal and random subspaces rather than scalar quantization. For a 64-step run, additional calibration, projection, or decomposition work must pay for itself. A small selective experiment is more actionable here than porting an entire framework. [A8–A10]

### Conservative experiment

After establishing the lossless FP16 route, test independently implemented symmetric INT8 compression of only the saved gate/up operands, with groups of 128 values and FP32 scales. Keep the complete forward calculation unchanged; reconstruct approximate values only for backward.

For each nonzero block:

`scale = max(abs(block)) / 127`

`q = round(block / scale), constrained to [-127, 127]`

`reconstructed = q * scale`

Handle zero blocks explicitly. This first deterministic quantizer does not consume the model's RNG stream. Do not label its gradients unbiased, and do not equate an unchanged current forward with an unchanged training trajectory: altered gradients affect later optimizer updates.

The storage budget is 1,010.625 MiB for both arrays across 28 maximum-size layers, including one FP32 scale per 128 values. It saves only another 949.375 MiB relative to the lossless FP16 baseline. That incremental benefit must justify quality risk and quantization overhead.

### Outlier experiment and CPU results

The original round-3 probe used NumPy seed 3031 and FP16-origin values widened to FP32. Each synthetic case used 64 × 8,960 elements per operand and a random FP32 incoming gradient. Error below is the relative L2 error of the concatenated gate/up input gradients, not an adapter-gradient, training-loss, or evaluation-loss result.

| Synthetic distribution | Lossless FP16 storage | INT8, block 128 | INT8, largest value per block retained separately | INT4, block 128 |
|---|---:|---:|---:|---:|
| Normal, standard deviation 1 | 0%, bitwise equal | 0.716% | 0.629% | 12.992% |
| Normal, standard deviation 3 | 0%, bitwise equal | 0.805% | 0.710% | 14.609% |
| One magnitude-64 outlier per block | 0%, bitwise equal | 6.555% | 0.285% | 41.818% |

The exception variant stores the largest-magnitude value independently, zeroes its position for ordinary block scaling, and restores it after dequantization. An illustrative layout with a FP32 exception value and one-byte position per block costs 1,048.90625 MiB across these maximum-size layers. This is a CPU sensitivity experiment, not a reproduced HyC-LoRA result or a proposed universal optimal quantizer.

The tests explain the conservative order: prove lossless storage first, then try INT8 with measured sensitivity and outlier behavior. Small aggregate local error does not guarantee the strict hidden-evaluation threshold, and a high gradient cosine can conceal localized errors. INT4/INT2 should not be the first submission experiment.

## 20. Backward sparsity: a valid small extension and an eliminated false lead

### Frozen-prefix pruning is mostly already handled

The actual task has checkpointing off, and Soup forwards that setting into PEFT preparation. With frozen embeddings and the first normalization, ordinary autograd should already stop before the first trainable adapter's input. Confirm hooks once, but do not assume a large redundant frozen-prefix backward exists.

This refines the earlier dropout budget. The first q/v dropout inputs can have `requires_grad=False`, so native autograd need not retain their masks for input backward. If only 27 baseline layers retain two byte masks while the shared-operand representation needs packed masks for all 28, the illustrative §11 savings become **309 MiB for FP16 operands or 477 MiB for FP32 operands**. The activation census decides the actual case.

### Compact the final tokenwise backward until the attention boundary

Assistant-only supervision produces zero terminal loss gradients for ignored target positions. The head-only observation in §3 can extend through final RMSNorm, the last MLP/residual, and the last attention output projection. Compute this backward on the selected rows, then restore the complete token axis before ordinary attention backward. Selection must come from the actual shifted labels and loss mask. [D1, A4; derivation in this research.]

Do not drop all prompt-token gradients. A prompt position can have zero terminal query gradient while still receiving key/value gradients from later supervised queries. A small FP64 CPU model confirmed this: selected-row tail backward matched dense tail backward exactly, its finite-difference discrepancy was below 1.9e-10, and an ignored prompt position still had nonzero K/V derivatives through attention.

The additional last-MLP/output-projection work available beyond head-only sparsity is at most about 1.414% of full forward-plus-input-backward linear arithmetic, multiplied by the ignored-row fraction in that cost model. Gather/scatter work and smaller GEMMs can consume the benefit. Rank this behind saved-MLP storage and the full-vocabulary implementation.

## 21. Whole-command improvements and routes to deprioritize

### Avoid preparing frozen NF4 state during adapter saving

The inspected versioned save chain is Soup `trainer.save_model` → Transformers `Trainer._save` → PEFT `save_pretrained` → `get_peft_model_state_dict`. Without a supplied state dictionary, PEFT first asks for the model's complete state and filters it afterward. bitsandbytes `Linear4bit._save_to_state_dict` serializes quantization metadata, including reading a nested offset via `.item()`. [R8, A1, A11–A14]

With 196 quantized transformer linears, this can prepare frozen quantization metadata and perform approximately 196 scalar readbacks to save only the 112 adapter tensors. The adapter payload is about 8.3125 MiB in FP32. This does not mean the entire base model is written to disk; the unnecessary work is state traversal and metadata preparation before filtering.

A narrow exact improvement is to supply the validated adapter tensors, under the original full parameter names, through the normal PEFT state-dictionary argument and retain the standard serializer, adapter configuration, model card, and required output files. Support only the verified q/v LoRA configuration; fall back normally for extra trainable parameters, changed embeddings, `modules_to_save`, or other unsupported configurations.

PEFT's automatic embedding-save decision also rechecks base configuration. The harness sets `HF_HUB_OFFLINE=1`, so do not claim this creates a live network round trip in the measured run. After proving unchanged vocabulary and embeddings, an explicit equivalent decision can avoid redundant cached configuration work. Transformers also saves the tokenizer through `processing_class`, and Soup saves it again; eliminate only a confirmed duplicate successful serialization to the same destination. [D3, R8, A11–A13]

Validation must compare tensor names, shapes, dtypes, and values; configuration; tokenizer files; and loading through the unchanged evaluator. Preserve finite checks, callbacks, logs, and successful save completion. Any timing gain remains unmeasured.

### Loader flags are not free improvements

Transformers 5.16.1 already uses meta initialization and lazy safetensors loading for this path. Its loader discards `low_cpu_mem_usage` as unused and forces synchronous materialization for on-the-fly quantization to avoid too many full-precision tensors accumulating on the GPU. Merely adding those familiar flags is not a demonstrated new optimization. [A15]

A bounded pipeline could overlap CPU materialization and transfer with the original per-tensor quantization, but it must preserve weight ties, tensor boundaries, quantization offsets, RNG state before adapter creation, and the whole-command memory peak. Pursue it only if a phase trace exposes several recoverable seconds; keep it inside the measured process.

### CUDA graphs are a poor main bet here

The actual harness cannot count own kernels replayed from a graph, and the README requires their normal launches. Capturing only native operations while leaving own kernels eager is a narrower possibility, but variable shapes, capture/warmup costs, graph-private pools, and 64-step amortization make it secondary. PyTorch also requires ordinary GradScaler step/update to remain outside typical capture. [D1, D3, A16]

Native CUDA RNG can advance correctly during graph replay; it does not inherently reuse one dropout mask. Setup must still preserve the production RNG stream, and warmup must not add optimizer updates. Do not modify the harness or its timing to make a graph-based proposal appear to qualify.

### A small nanochat loop idea: measure cyclic-GC pauses

Nanochat explicitly collects/freezes/disables cyclic garbage collection around training because its source reports costly scans in that workload. Python's cyclic collector supplements reference counting, so these are distinct mechanisms. [A17, A18]

For Soup, first measure actual collection pauses with callbacks. A bounded change must restore the original GC state and include cleanup in full-command timing. GPU tensors retained by cycles can raise peak memory. No evidence here establishes that Soup has nanochat's pauses, so this is a small profile-driven add-on.

## 22. Round 3 implementation decision and verification gates

Round 3 selected **lossless storage of the two widened-FP16 MLP operands** as its first experiment. It remains the main retained-state target, with a concrete source argument and no replay FLOPs or added activation-quantization error. Round 4 adds simpler copy-removal checks and an integrated ownership refinement; use §28 for the current order.

1. Reproduce the pinned starter and record the actually loaded versions. Keep its useful SwiGLU example, with explicit attribution and exclusion from own-kernel claims.
2. Capture actual gate/up operands and their dtype provenance on T4. Confirm frozen bias-free projections, FP16 compute, original FP32 saved dtype, and exact FP16 round-trip values. Check the output shape/layout that the starter actually saves.
3. Prototype scoped packing that restores FP32 before the existing backward. Verify that no closure, reference, or other autograd consumer keeps the original wide arrays alive.
4. Compare forward outputs, reconstructed operands, incoming-gradient handling, every q/v A/B gradient, and input gradients under the original AMP/GradScaler behavior. Include signed zero, subnormal values, short/odd shapes, and nonfinite handling where relevant.
5. Measure isolated overhead and then the complete unchanged task. Reject the storage change if temporary buffers or copies erase the score benefit.
6. Add useful original full-vocabulary reductions and other necessary kernels to meet the 5% share in every measured run. A useful storage change does not automatically meet that separate admission requirement.
7. Only after a measured lossless version, compare one-branch replay and selective INT8 compression as alternatives. Evaluate their incremental memory reduction and full-run loss/time effects.
8. Apply measured final-save improvements while preserving adapter contents and every required artifact. Run all official tests and baseline–solution–solution–baseline measurements unchanged.

The research now supports a concrete change in direction: optimize the representation and lifetime of existing activations before attempting a replacement matrix engine or a different model. All memory numbers in this round are calculated retained-state budgets; none is a reported T4 peak or score.

## 23. Round 3 experiment and source provenance

The original CPU probe used NumPy seed 3031. It tested all finite binary16 patterns, three local SwiGLU derivative distributions, symmetric block quantization, and the storage arithmetic in §§17–19. Its outputs were inspected after execution. The terminal-tail proof was a separate small NumPy FP64 derivative experiment. Neither executed PyTorch, Triton, CUDA, an NF4 model, or a training/evaluation loop.

Versioned files additionally inspected for the dtype argument:

| File | SHA-256 of fetched UTF-8 source |
|---|---|
| bitsandbytes 0.50.2 autograd functions | `50d732d6f878f1e64ff3c71403612a5bd3d1444645d5938558e9b2a1de382b4b` |
| bitsandbytes 0.50.2 CUDA operators | `b3560e07ac1992c2fe8692c794d0246aef8bd69b76df471c98683913653d255f` |
| Transformers 5.16.1 Qwen2 implementation | `5041a2996bb73eeca6b62521c118db3b4a84e43102c2e020d119a21a3fe3bdc4` |

The public source files were read, not installed or executed. Pinned starter evidence supersedes the earlier access limitations. The research does not establish organizer acceptance of an unimplemented variant or a winning result.

## 24. Round 4: eliminate a real copy in token accuracy

This round investigated work duplicated across framework boundaries. The exact pinned call chain reveals two small integration changes to try before replacing cross-entropy or a matrix multiplication. This is new source evidence for the earlier broad suggestion to remove logits copies. [A4, A11, B1, B3, B4, R8]

### The actual order and dtype

Transformers calls Soup's Rewind trainer, which delegates to TRL, which delegates to the ordinary Transformers loss path. Accelerate runs the model under FP16 autocast. Qwen computes the head, computes its loss, and returns logits. Accelerate converts returned half-precision tensors to FP32. TRL then computes entropy, token count, and accuracy; Soup then computes Rewind row losses. Only after these operations return does Transformers call backward. [A11, B1, B3, R8]

Consequently, **the logits consumed by TRL and Rewind are FP32 in this task**. The following masks are distinct:

| Required output | Positions included | Normalization |
|---|---|---|
| Training cross-entropy | Logit at t predicts label at t+1; ignored labels excluded; the last logit receives an appended ignored target | The supplied `num_items_in_batch`, or the native CE mean when absent |
| Token accuracy | Shifted labels other than -100 | Valid shifted-label count; zero when that count is zero |
| Rewind row loss | The same supervised shifted labels, separately for each example | Each row's valid-token count; empty row gives loss 0 and count 0 |
| Entropy | Unshifted attention mask, including real prompt tokens and the final real token | Sum of the unshifted attention mask |
| Logged token count | Unshifted attention mask | Count of non-padding tokens |

The pinned Trainer's causal-LM item count also inspects shifted labels. Preserve the value supplied by that trainer instead of inventing a new loss denominator. The entropy helper already processes flattened logits in chunks of 128 rows; a single FP32 chunk is 74.1875 MiB at this vocabulary size. It materializes log-probability and elementwise intermediates within each chunk, not a full-batch probability tensor. A proposed entropy kernel should be compared against that actual implementation. [B1, B2, B4, A11]

### Move the slice after the reduction

TRL 0.29.1 first slices `outputs.logits[..., :-1, :]`, explicitly makes the result contiguous, then takes argmax. At B=4, removing the final timestep leaves a gap between adjacent batch rows, so this is a real large copy. [B1]

An original narrow implementation can keep the native reduction and change the order:

```python
# Conceptual replacement for the prediction expression only.
predictions = outputs.logits.argmax(dim=-1)[..., :-1]
```

Keep the surrounding shifted labels, mask, counts, distributed gathering, zero-count behavior, and metric logging unchanged. This computes four additional argmax rows at the maximum shape, then discards four integer predictions. It changes neither the training objective nor the head backward.

An alternative original Triton kernel can read the strided logits view directly. For flattened supervised-position index r, use original token-row index `(r // (T-1)) * T + r % (T-1)`, or pass explicit strides. Do not assume `reshape(-1, V)` on the sliced view is free: flattening across batch gaps can recreate the copy.

| Calculated quantity at B=4, T=512, V=151,936 | Size |
|---|---:|
| Removed contiguous shifted-logits buffer | 1,184.681640625 MiB |
| Removed copy's logical read plus write traffic | 2,369.36328125 MiB per step |
| Full B×T int64 prediction tensor | 16 KiB |
| Additional reduction rows for full argmax then slice | 4, or approximately 0.196% |

Using native argmax preserves its native per-row tie and nonfinite handling. A custom implementation must match those semantics independently, including the smallest-index rule for finite equal maxima. Sixteen original CPU layout/mask cases passed with identical predictions and accuracy, including tied maxima and empty supervised rows. This is not a CUDA performance or NaN-semantics test.

### Why copy size is not a peak-memory promise

Native plain CE in PyTorch 2.11 computes log-softmax followed by NLL. Its reverse-mode rule retains the log-softmax result, not a separate enduring copy of the original logits cast. NLL and log-softmax reference the same result. [B6]

Define P as the retained FP32 log-softmax result, B as the returned FP32 logits, and C as the accuracy copy. P and B are each 1,187 MiB at maximum shape.

| Phase | Simplified large logits-related buffers |
|---|---|
| Accuracy before this optimization | P + B + C, approximately 3,558.68 MiB |
| Accuracy after removing the copy | P + B, plus tiny predictions/counts |
| Native CE backward | P plus large gradient buffers can still approach three full FP32 arrays, or 3,561 MiB |

This model excludes the transformer, head-weight cast, allocator, and other workspace. It shows why the removed copy guarantees less copying at this point while another phase can still determine the command's peak. All required forward metrics finish before backward, but that does not authorize overwriting P: native backward still reads it. Destructive reuse requires a custom loss owner that controls every reader and saved tensor.

## 25. Cast logits once before loss and metrics diverge

The pinned source has two conversions of the same FP16 head result:

1. Qwen's `lm_head` produces logits under FP16 autocast.
2. `ForCausalLMLoss` calls `logits.float()` internally.
3. Qwen returns the original half logits.
4. Accelerate separately converts those returned logits to FP32 for TRL and Rewind. [A4, B3, B4]

A differentiable output hook on the actual head, returning its existing result's `.float()`, moves the needed conversion before this split. The loss's `.float()` becomes an identity, and Accelerate no longer converts that tensor. An outer-model forward hook is too late because the loss has already executed.

The original head matmul, full vocabulary, effective weight/input dtypes, loss arithmetic, and gradient cast remain in place. The externally prepared model already returned FP32 logits, so the consumers receive the same values and dtype. This is a source-derived equivalence argument for the scoped path, not an executed PyTorch integration test.

Install the hook on the existing head after final PEFT setup. Preserve its module, parameter names, tied-weight identity, and output structure. Return the ordinary differentiable conversion, without detaching. Guard the real intended model, frozen bias-free head, CUDA FP16 autocast, and training with gradients enabled; make installation idempotent. The starter explicitly permits this patch boundary and warns that regression tests also use fake model objects. Unsupported paths must retain their existing behavior. [D1, B3, B7]

At maximum shape, one half logits tensor is 593.5 MiB and one FP32 tensor is 1,187 MiB. Avoiding one conversion removes **1,780.5 MiB of logical read/write traffic per step**. The two original FP32 casts occur at different times: the loss's input cast can die before the returned-output cast is made. Do not sum them as two persistent live buffers or promise 1,187 MiB less overall peak. Earlier conversion can also shorten the original half-logits lifetime; profile the actual allocation timeline. [B6]

This change composes with §24 and requires no new loss implementation. Reject it if the actual head already returns FP32, another wrapper invalidates the assumed path, gradient connectivity changes, or full-command time does not improve. A native hook and native argmax contribute no original-Triton share by themselves.

## 26. Frozen-head caching: a source-level uncertainty is resolved

Round-6 extension: §34 evaluates replacing the widened frozen table with its proven original representation. That is an alternative to this FP32-parameter-plus-half-cache design and carries additional metadata/integration requirements.

PyTorch 2.11's autocast cache checks that an FP32 tensor **requires gradients**, is a leaf, and meets other eligibility conditions. PEFT freezes the base model and leaves this nonquantized tied head in FP32. The task's frozen head therefore does not qualify for that automatic weight cache. [B5, B7, D1]

An explicit FP16 representation used only by the output projection can avoid repeated conversions while leaving the original FP32 tied embedding/head parameter intact. Do not simply convert the tied parameter to half without preserving FP32 embedding outputs: that would change the input-embedding path. Section 34 describes a narrower alternative that uses verified FP16 provenance while explicitly preserving FP32 embedding outputs.

| Derived quantity | Value |
|---|---:|
| Effective FP16 head representation | 445.125 MiB |
| One FP32-read plus FP16-write conversion | 1,335.375 MiB of logical traffic |
| 63 avoided conversions after the first of 64 training calls | Approximately 82.16 GiB of logical traffic |

The last row assumes one eligible head invocation per step and reuse after first construction; it is not a runtime prediction. All initialization remains timed. Build after final model preparation, use the same effective layout and dtype, and invalidate on source-weight replacement/version, device, or compute-dtype changes. Keep the cache outside serialized trainable state.

The baseline already creates a half matrix for the head and needs it for hidden-state backward, despite the head weight being frozen. A cache can replace that per-step allocation rather than duplicate it. However, it persists earlier and between steps, so it can raise the peak in another phase. Measure its interaction with MLP storage and loading before accepting it. For this reason it remains a secondary experiment behind the two logits fixes and lossless MLP ownership.

## 27. A different MLP route: own the casts and their backward boundary

### Preserve two separate gradient roundings

Qwen's normalized MLP input feeds only gate and up in this configuration. The residual holds the pre-normalization tensor; q/v-only LoRA adds no MLP branch. Each frozen projection separately casts the FP32 normalized input to FP16, computes its output, and widens it back to FP32. In backward, each projection produces a separately rounded FP16 input gradient, widens that result to FP32, and autograd adds the two FP32 contributions. [A1, A3, A4, D2]

For those branch results, the existing join is:

`dX32 = float32(dX_gate16) + float32(dX_up16)`

Hoisting a single ordinary `x.half()` into both branches would make their gradients meet at a half-precision tensor first. That introduces an extra FP16 rounding before widening. A custom owner must explicitly preserve the original join. An original cast/add kernel can load the two FP16 results, widen each, add in FP32, and store one fresh FP32 output while leaving both GEMMs unchanged.

The original CPU probe provides concrete counterexamples:

| Two FP16 branch values | Widen separately, add in FP32 | Add into FP16, then widen |
|---|---:|---:|
| 1 and 2^-11 | 1.00048828125 | 1.0 |
| 65,504 and 65,504 | 131,008, finite | Positive infinity |

For 100,000 sampled finite half-precision pairs, the extra half rounding changed 53,932 resulting FP32 bit patterns. This is evidence against an algebraic rewrite, not a measured training-error rate. A two-output custom cast with aliased storage is another possible implementation, but its PyTorch alias/autograd behavior has not been tested here; the explicit projection owner makes the required join clearer.

At maximum shape, one narrow FP32 input is 12 MiB and its half representation is 6 MiB. A logical traffic model gives:

| Narrow-array operation, per layer | Original traffic | Proposed traffic | Avoided |
|---|---:|---:|---:|
| Two identical input casts versus one | 36 MiB | 18 MiB | 18 MiB |
| Two gradient widenings plus FP32 addition versus fused widening/add | 72 MiB | 24 MiB | 48 MiB |
| Total across 28 layers | 3,024 MiB | 1,176 MiB | **1,848 MiB per step** |

These casts are transient. bitsandbytes' frozen backward retains packed weights and quantization state rather than saving these input cast buffers. Do not describe the traffic reduction as another persistent activation-memory saving. Keep the same branch GEMM shapes, strides, effective dtype, dequantization and quantization metadata. Do not overwrite an incoming residual gradient whose other consumer may still need it.

### Extend ownership through SwiGLU to avoid wide conversion detours

This is the strongest new MLP implementation proposal in this round. Instead of first widening gate/up and later packing them back to half, retain their original half results at production and consume them through FP32 arithmetic.

The supplied kernels already load gate/up into FP32 registers. Their stores select the dtype of the destination pointer. That makes the following integration possible without changing the intended SwiGLU arithmetic: [D4]

1. Inside a custom boundary, create the effective FP16 normalized input once and run the two original frozen NF4 projection operations separately.
2. Keep their native FP16 gate/up outputs. Invoke the supplied forward kernel with those input pointers and an explicitly **FP32 activation-output buffer**.
3. Retain gate/up directly in FP16 plus the original packed weight references and separate quantization states. The frozen projection backward does not require the original X values.
4. Keep the incoming activation gradient **FP32**. Invoke the supplied backward kernel with half gate/up inputs and half destination buffers for the two projection-output gradients. The calculation still takes place in FP32; each final half store occupies the rounding boundary that the original cast into projection backward already imposed.
5. Run the two original separate input-gradient GEMMs. Widen their separate half results and add in FP32 as above. Leave `down_proj` outside this boundary, consuming the same FP32 activation output.

This is not accomplished by passing half tensors to the existing `SwiGLU.apply` wrapper: that wrapper would infer the incoming-gradient dtype from the saved gate and round it too early. The surrounding ownership and explicitly selected output dtypes are essential. Preserve the original AMP state and loss scaling, kernel operation order, and FP32 intermediate arithmetic. The pinned bnb no-grad forward path calls the same four-bit operator arguments as its autograd forward for these ordinary shapes, but the actual dispatch and numerical outputs must still be compared on T4. [A3, D4]

The storage reduction is **the same 1,960 MiB** derived in §17, relative to the starter's FP32 saved operands. It is not an additional 1,960 MiB on top of packing. Its new attraction is avoiding conversions and memory traffic around that storage.

For one maximum-size wide array, FP32 is 70 MiB and FP16 is 35 MiB. The following model counts explicit reads/writes, excluding GEMMs and caches:

| Wide-array traffic avoided per layer | Versus starter FP32 state | Versus scoped half packing/unpacking |
|---|---:|---:|
| Widen both projection results | 210 MiB | 210 MiB |
| Reduced gate/up reads in activation forward | 70 MiB | 70 MiB |
| Pack both saved FP32 operands | — | 210 MiB |
| Restore both operands before backward | — | 210 MiB |
| Reduced gate/up reads in activation backward | 70 MiB | 70 MiB |
| Reduced projection-gradient writes | 70 MiB | 70 MiB |
| Cast both projection gradients back to half | 210 MiB | 210 MiB |
| **Total per layer** | **630 MiB** | **1,050 MiB** |
| **Total across 28 layers** | **17,640 MiB** | **29,400 MiB** |

These are logical traffic estimates, not measured DRAM transfers, time, or peak-memory gains. The narrow cast/add table covers separate buffers. Actual sequence lengths, cache reuse, compiler choices, and allocation lifetimes determine how much of either calculation becomes a speedup.

Reusing the supplied kernels or changing their pointer-dtype specialization still earns **zero original-kernel credit**. Attribute them explicitly. The independently written cast/add and full-vocabulary reductions can do qualifying useful work, but their measured share must meet the gate in every run. This ownership design has not been executed in PyTorch, Triton, or CUDA; compare its forward output, each rounded projection gradient, every adapter gradient, and AMP overflow decisions before integrated scoring.

### Why a larger grouped GEMM is lower priority

Concatenating gate/up outputs in one forward GEMM is mathematically possible, but does not reduce FLOPs. It changes the GEMM shape/algorithm, needs both original NF4 quantization states preserved, and produces strided half-output views that require deliberate handling.

A combined dense FP16 weight pair is 52.5 MiB. Ordinary `F.linear` autograd can retain that matrix for input backward, accumulating **1,470 MiB across 28 layers**. Retain the original NF4 representations in a custom owner instead. A naïve concatenation after separately decoding the two weights also costs 105 MiB of logical traffic per layer; decoding directly into separate row ranges of a temporary pair avoids that staging copy. [A3, A14; arithmetic derived]

Combining backward into one large dot product removes the original separate branch-output rounding. An explicit CPU counterexample returned 0 for the separately rounded products and 0.00048828125 for the merged reduction. A batched pair of GEMMs preserves separate output buffers, but can still select a different cuBLAS algorithm; PyTorch also documents that batched and separate computations need not be bitwise identical. Neither version is established as bitwise equivalent or faster. Prefer the unchanged-GEMM ownership experiment first. [B8]

## 28. Round-4 experiment order and evidence

This records the round-4 queue. Sections 33 and 38 contain the subsequent queues; §38 is current:

| Order | Experiment | Evidence required to keep it |
|---|---|---|
| 1 | Native full argmax followed by a small causal slice | Same predictions, masks/counts/logs; less copying and a measured full-command benefit |
| 2 | One differentiable head-output FP32 conversion | Same effective logits, loss and all adapter gradients; no broken hooks or output contracts |
| 3 | Lossless MLP packing as a reference, then direct half retention with explicit FP32 activation arithmetic | Same saved values and rounded branch gradients; no extra live wide arrays; memory/time improvement |
| 4 | Independent full-vocabulary metric/loss reductions and MLP cast/add | All required masks/reductions correct; useful own-kernel share at least 5% in every run |
| 5 | Frozen-head representation cache | Source cast actually disappears; unchanged tying/serialization; longer cache lifetime does not erase score benefit |
| 6 | Replay, selective INT8 state, grouped GEMMs, or broader norm fusion | Only after the simpler version is measured; compare incremental benefit and new numerical risk |

Run the unchanged official tests and measurement order. The model, optimizer, dropout, batches, step count, logs, final adapter, and standard evaluation path remain fixed. These are proposed implementations of the existing computation. Passing the numerical gates does not grant permission to change that computation's task definition. [D1–D3]

Two original NumPy probes were executed and their outputs inspected. Seed 4042 checked all 63,488 finite FP16 values for agreement between the two logits casts, 16 finite argmax/layout/mask cases, explicit row addressing, and byte arithmetic. Seed 4041 checked the shared-cast and merged-GEMM counterexamples, 100,000 random branch pairs, and grouped matrix algebra in FP64. The latter agreed algebraically within 7.11e-15; that does not establish CUDA algorithm equivalence.

The actual dependency source files were read without installing or executing them. No torch runtime or GPU is available in this workspace. No custom Function, kernel specialization, training trajectory, hidden evaluation, full-command timing, own-kernel share, or sampled GPU peak was measured. The new findings establish concrete opportunities and the correctness boundaries to test.

## 29. Round 5: a conditional attention-backend problem on T4

The new attention finding is a concrete dispatch mismatch to check before writing another attention kernel. It applies to the **audited Transformers 5.16.1 / PyTorch 2.11.0 source path on SM75**, when the model passes no explicit attention mask.

**Version qualification for this report:** the supplied brief specifies Python 3.13, torch 2.11, and Triton 3.6.0 as the target Kaggle image. The starter install command explicitly pins Soup, Transformers, PEFT, TRL, bitsandbytes, and Accelerate, while relying on the image for torch/Triton. This audit reads exact PyTorch 2.11.0 and Triton 3.6.0 source. The actual running package/build versions remain unobserved; record them before applying a version-specific patch. These are prescribed targets and inspected source versions, not a verified runtime. [D0 §5, D1, R1, C3]

### How the mismatch arises

Qwen has twelve query heads and two key/value heads, with head dimension 128. Transformers' `use_gqa_in_sdpa` enables native grouped-query attention when the mask is absent and the K/V head dimensions agree and are at most 256. Its CUDA decision does not test the GPU architecture. When the mask is present, the wrapper instead repeats each KV head six times before calling SDPA. [S8, C1]

The resulting source-level paths are:

| Existing batch condition | Transformers input to SDPA | PyTorch 2.11 outcome on an ordinary eligible T4 build |
|---|---|---|
| No explicit mask | Twelve Q heads, two KV heads, `enable_gqa=True` | Math fallback: Flash and cuDNN require SM80+, while the memory-efficient backend rejects unequal dense head counts |
| Explicit causal/padding mask | Twelve Q heads, twelve repeated KV heads, default GQA setting | Memory-efficient attention can be eligible; remaining dtype, stride, build, and shape constraints still apply |

The decisive checks are `check_flash_attention_hardware_support`, `check_cudnn_hardware_support`, and the memory-efficient branch's `check_batch_size_and_num_heads_dense<false>`. T4 itself is supported by the memory-efficient implementation; unequal head counts cause the relevant rejection. [C3]

The dtype chain does not rescue the absent-mask path. Q/K/V arrive at the Python wrapper in FP32, but CUDA autocast registers SDPA for its lower-precision dtype, so this task's operator sees FP16. The default math implementation then widens those operands to FP32 and repeats compact KV internally. Its reduced-precision math-reduction option defaults to false. [A4, B5, C4, C5]

The mask can legitimately be absent during training. The mask helper allows this for ordinary causal attention when there is no actual padding and the query/key lengths agree. Four rows truncated to 512 can qualify, as can equal shorter lengths. Assistant-only ignored labels are unrelated to input padding. **The number of affected challenge batches is unknown; do not change their composition or order to manufacture this condition.** [C2]

### First experiment: use the existing repeat branch and keep native SDPA

For an eligible, version-checked T4 call, test explicit KV repetition before the native SDPA call even when the mask is absent. Preserve the original causal flag, scaling, attention dropout, positions, AMP context, and `_attn_implementation="sdpa"`. Let native backend selection run normally; do not change global backend flags or install another package.

This preserves the attention operation algebraically, but can select another numerical backend and move the KV-gradient rounding boundary. **It is not a bitwise-equivalence claim.** Compare forward outputs, every adapter gradient, AMP overflow decisions, and the full training/evaluation gates. A native-dispatch patch alone earns no original-Triton credit. PyTorch documents numerical differences between its SDPA backends. [C1, C7]

There is corroborating public evidence for Qwen2.5-1.5B on T4 in Transformers issue #48633, but that is inference/decode evidence. Its reported improvement is not a Soup training prediction. Later PyTorch work also adds memory-efficient GQA support, so the negative finding must remain scoped to the audited version. [C7]

At B=4,T=512, the following are component sizes:

| Attention component | Calculated storage |
|---|---:|
| Compact FP16 K/V pair | 2 MiB |
| Repeated FP16 K/V pair | 12 MiB |
| Repeated FP32 K/V pair | 24 MiB |
| One FP32 score or probability array, shape [4,12,512,512] | 48 MiB |
| One such retained probability array in each of 28 math-attention layers | 1,344 MiB, conditional |

The last line is a saved-component budget, not a measured peak reduction. The math path's softmax output participates in backward; the alternative backend has its own saved state and workspaces. Actual masks, batch lengths, lifetimes, and backend availability decide the result. [C4, B6; sizes derived]

### Two secondary experiments for batches already using a mask

**Fuse KV repetition with the existing FP16 cast.** The current route repeats compact FP32 K/V into a 24 MiB FP32 pair, then autocast reads it to produce a 12 MiB FP16 pair. An original kernel can directly produce the same contiguous repeated FP16 buffers. It removes the repeated FP32 pair's 24 MiB write and 24 MiB reread: 48 MiB of explicit traffic per layer, or 1,344 MiB across 28 maximum-length layers. It still retains the repeated FP16 operands for SDPA. These are traffic calculations, not persistent-memory savings. [C1, C5]

Its backward must widen each query head's incoming FP16 gradient before summing the six heads associated with one KV head. Initially keep the native FP32 reshape-and-sum reduction. Moving an ordinary compact `.half()` before repetition changes that boundary: half-representable contributions [2048,1,0,0,0,0] produce 2049 when widened before the sum, but 2048 when the combined result is rounded to half first. The absent-mask math path has a different compact-half boundary; this masked-path argument does not prove the dispatch experiment bitwise equivalent.

A zero-copy twelve-head tensor is not available through one ordinary head stride: the source-head mapping is floor(head/6), which is not affine. Two grouped calls or repacking groups into the batch are later experiments with additional launch, layout, and gradient costs.

**Prepare one shared additive mask per forward.** Qwen constructs its Boolean mask before the decoder loop. Each native SDPA call converts that shared mask into a fresh effective-dtype additive bias, using 0 for allowed entries and negative infinity for blocked entries. Efficient-attention backward retains the bias. For a dense [4,1,512,512] mask, twenty-eight FP16 copies total 56 MiB; sharing one 2 MiB representation conditionally removes 54 MiB of retained mask storage. [A4, C2, C4, B6]

Match native alignment as well as values. `preprocess_mask` requires preceding strides divisible by eight and last stride one; otherwise it pads the underlying row and slices back. A logical T=511 mask therefore needs row stride 512; T=9 needs stride 16. Convert and align once, keep `None` as `None`, and share only within that forward. This does not address the absent-mask dispatch issue. [C4]

Finally, do not assume every visible `.contiguous()` is a copy. The efficient CUDA backend allocates [B,T,H,D], returns a transposed view, and Transformers transposes it back. Those output-layout operations are normally identities. The math path does require the final 6 MiB FP16 layout copy at maximum shape; only affected calls offer that additional opportunity. [C1, C6]

## 30. Round 5: reuse native loss normalization for the required metrics

This is a smaller first implementation than replacing the entire loss and backward. **Expose the FP32 log-softmax result that native cross-entropy already computes, and reuse it for entropy and Rewind.** Keep native NLL's scalar reduction and native autograd.

For the task's integer labels, no class weights, and zero label smoothing, native CE composes log-softmax and NLL. Its reverse-mode rules reference the same log-softmax storage; those references do not represent two copies. Transformers also provides an instance-level `loss_function` setter, allowing an eligible model to own this composition without globally changing `LOSS_MAPPING`. [B6, A15]

### Preserve the training scalar exactly as specified by the original path

The owner should perform the original FP32 logits conversion, append the ignored causal target and shift labels, flatten to [B*T,V], call native log-softmax, and call native NLL. When `num_items_in_batch` is supplied, retain native sum reduction followed by the original division; otherwise retain native mean reduction. Keep the incoming gradient scale and the ordinary autograd graph. [B4]

Do not replace the scalar with a sum of newly gathered token losses or an average of per-row means. Those can change reduction order or weighting. Reusing the already computed distribution does not require rewriting the training scalar's arithmetic.

From detached P = log-softmax(Z), compute small metric results:

| Consumer | Result and required mask |
|---|---|
| Entropy | −sum(exp(P) × P) over vocabulary, then the original **unshifted attention-mask** aggregation |
| Rewind | Gather −P at each valid shifted target; preserve each row's T−1 reduction domain, supervised count, and zero/count-zero empty-row behavior |
| Accuracy | Argmax of the **original logits Z**, compared with shifted valid labels |
| Token accounting | Original unshifted attention-mask count |

The task-specific FP32 metric dtype was established in round 4 through Accelerate's returned-output conversion. Prompt positions and the last real token remain part of entropy even when their loss targets are ignored. [B1–B4, R8]

### The integration must actually bypass duplicate work

Changing only the loss callable leaves TRL entropy and Soup's row CE recomputations intact. Likewise, adding a token-loss record does not activate Soup's existing fallback while logits remain present: its `compute_loss` prioritizes the logits branch. An original adapter must explicitly consume the current forward's small results in both metric/logging consumers. [B1, R8]

Keep an ephemeral record associated with that model and that forward. Preserve TRL gathering and metric updates, and Rewind's row identities, order, step association, and FIFO/sampler bookkeeping. Clear it after consumption and on exceptions. Only small statistics should escape the loss owner; retaining P, detached P, logits, or a closure containing them in a persistent record would extend the large buffers' lifetimes.

A conservative first version can keep ordinary returned logits and native backward, changing only how required metrics obtain their values. Guard unsupported labels, smoothing, reduction, evaluation, and prediction paths. This is an original integration, not permission to enable Soup's prohibited built-in acceleration switches.

### The removed normalizations can be counted

Let N=B*T. The current forward log-softmax call count is 1 + ceil(N/128) + B: one training loss, TRL's 128-row entropy chunks, and one Rewind CE per row. The vocabulary rows normalized total 3BT−B for ordinary T>1. [B2, B4, R8]

At maximum shape:

| Forward normalization work | Existing path | Shared native P |
|---|---:|---:|
| Training rows | 2,048 | 2,048 |
| Additional entropy rows | 2,048 | 0 |
| Additional Rewind rows | 2,044 | 0 |
| **Total vocabulary rows normalized** | **6,140** | **2,048** |
| **Log-softmax calls** | **21** | **1** |

This eliminates 4,092 duplicate row normalizations and twenty forward log-softmax launches. It does not eliminate two-thirds of training time: the head, native loss backward, metric scans, and the Transformer remain.

There is source-level numerical support for sharing these rows. At vocabulary width 151,936, audited CUDA log-softmax uses its large-row kernel with one block per row. Changing the number of rows changes the grid, while the row reduction depends on width, dtype, and alignment. This supports testing reuse across the baseline's chunk sizes, but actual T4 equality remains unverified. [C8]

### Two numerical traps found in the CPU checks

**Argmax must remain on Z.** In a 151,936-entry row, set z[0]=0, z[1] to the smallest positive FP16 value widened to FP32, and every other entry to zero. The first two FP32 log-softmax values round to the same value:

| Quantity | Index 0 | Index 1 |
|---|---:|---:|
| Original widened FP16 logits | 0 | 0.000000059604644775390625 |
| FP32 log-softmax in the NumPy probe | −11.931214332580566 | −11.931214332580566 |

Argmax(Z)=1, while argmax(P)=0. Mathematical monotonicity does not prevent finite-precision ties after subtracting the log normalizer. An accuracy kernel may read P for entropy and Z for predictions, but must not replace Z-based tie handling.

**Keep the row reduction shape.** Appending one extra ignored zero to a row's FP32 sum changed one CPU fixture by 4.77×10⁻⁷. Preserve the original T−1 Rewind domain and count shifted prediction targets, not raw labelled positions. Nonfinite values also need matching behavior; skipping masked rows can alter the baseline's propagation of zero times NaN.

### Useful original kernels and the remaining memory limit

An original metric kernel can fuse exp(P), multiplication by P, and entropy reduction, avoiding separate large exponential/product arrays. Target gathers and compact statistics are additional useful work. For example, 4,096-column tiles produce 38 partials per token; at N=2,048, a FP32 partial buffer is about 0.297 MiB. That is a candidate launch design, not a tuned choice.

This first stage still retains **P = 1,187 MiB** at maximum shape and preserves native CE's dense backward workspaces. It primarily removes duplicate forward computation. The full custom CE/chunked-head designs remain the stronger later memory candidates. Own metric kernels may still fall below 5% of training time after acceleration; measure combined useful original work rather than inflating their runtime.

A possible intermediate follow-up would preserve native P in forward but implement its backward without materializing the sparse-in-value, dense NLL-gradient array G. The native-style expression is G − exp(P) × sum(G). Preserve NLL's FP32 division/sign, incoming AMP scale, ignored targets, output dtype, and multiply/subtract rounding. The factored expression alpha × (p − one_hot) is not automatically identical. This follow-up is a numerical specification, not an implemented or validated kernel. [B6, C8]

## 31. Round 5: a more precise NF4 target than replacing the GEMM

The bitsandbytes 0.50.2 audit identifies a concrete three-stage decoding sequence on the normal nested-quantization fallback:

1. Decode the eight-bit nested scale codes into FP32.
2. Add the original FP32 offset in a separate ATen operation.
3. Decode NF4 values, multiply by their effective scales, and store dense weights for the native matrix operation.

These are three GPU launches before the matrix multiplication on this path. An original decoder can combine their data movement into one launch without replacing cuBLAS. This develops §12's proposal into a version-specific arithmetic and dispatch contract. [A2, A3, A14, C9]

### Preserve the actual format and separate rounding stages

For a flattened dense-weight element e, the task's first-level block index is floor(e/64), and its nested scale-block index is floor(floor(e/64)/256). The packed byte is at floor(e/2): even e uses the high nibble, odd e the low nibble. [A14, C9]

Use the original per-matrix scale IDs, nested codebook, second-level scales, offset, and NF4 centroids. Preserve contiguous [out_features,in_features] output order. The numerical sequence is:

    scale_product = FP32_multiply(code8[scale_id], second_level_scale)
    effective_scale = FP32_add(scale_product, original_offset)
    weight_product = FP32_multiply(nf4_centroid, effective_scale)
    dense_weight = reference_destination_cast(weight_product)

The multiply followed by offset addition crosses a kernel/storage boundary in the baseline. A fused implementation must preserve those separate FP32 rounding steps when replacing that reference; it must not silently contract them into one FMA.

The pinned bnb CMake build enables fast math. NVIDIA documents that this includes FP32 flush-to-zero and permits multiply-add contraction. Therefore, validate the installed decoder's arithmetic and the intervening ATen addition separately. Do not infer an identical fast-math policy for the whole fused expression, and do not flush valid FP16 subnormal outputs merely because a source multiplication uses FP32 FTZ. No installed binary or generated PTX has been inspected. [C11]

Backward also needs its original dtype sequence. Forward fallback decodes into the activation dtype, while backward decodes according to `quant_state.dtype` and then converts to the gradient dtype. If these differ, a direct-to-half shortcut can omit an intermediate rounding. Specialize only after checking the actual state. [A2, A3]

One original program could reconstruct a scale for its 64 weights, unpack adjacent nibble pairs, and write adjacent outputs. Tiles of 1,024 or 2,048 elements are measurement candidates. Compare decoded bits before integrating any change into a matrix product.

### Correction: native bnb can use T4 Tensor Cores at small M

The Triton 3.6 ordinary-dot limitation in §10 does **not** imply that bnb 0.50.2 lacks a T4 Tensor Core path. Its CUDA source contains SM75 FP16 m16n8k8 MMA. For the Qwen shapes on a 40-SM T4, the Python dispatcher permits the custom forward route up to these M=B*T ceilings, with further SIMT/MMA choices inside that route: [A2, C10]

| Projection | Custom forward ceiling |
|---|---:|
| q, k, v, o | M≤64 |
| gate, up | M≤128 |
| down | M≤64 |

At maximum M=2,048, all seven use the fallback. Ordinary input backward explicitly decodes before matmul. Preserve the existing small-M forward route in the first prototype; its nested-scale expression occurs inside a fast-math kernel and need not share the fallback's exact rounding boundaries. [A2, A3, C10]

Two tempting alternatives were ruled out by the source audit. Normal decode does not repeatedly call `.item()` for the offset; it passes a device pointer. Its canonical packed-weight `view` also does not establish a repeated orientation copy. Serialization readbacks remain the separate save-path issue already documented in §21. [A2, A14]

### Secondary extension: a bounded current-layer decoding slab

After per-projection fusion passes, batch only the currently needed matrices into one decode launch with independent metadata.

| FP16 dense-weight workspace | Calculated size |
|---|---:|
| Largest individual Qwen projection | 26.25 MiB |
| All seven projections in one layer | 89.25 MiB |
| Additional space relative to the largest individual buffer | Up to 63 MiB |
| All seven projections retained across all 28 layers | 2,499 MiB |

For a layer where seven forward projections all use fallback, 7 × 3 decode/add launches can become one decode launch, followed by the original seven matrix operations at their usual positions. This is a launch-scheduling experiment that spends additional bounded workspace; it is not a memory-saving claim.

Use a reusable same-stream workspace with explicit ownership. Retain the original packed weights and quantization states for autograd, reconstruct the required matrices for backward, and recycle the workspace after its readers finish. Ordinary differentiable `F.linear` on slab views can retain dense weights until backward and accidentally recreate the 2,499 MiB whole-model cache. A custom owner is therefore necessary. [A3; sizes derived from S8]

Backward batching must follow actual input-gradient requirements. With no checkpointing and no input-gradient hook, the first layer's frozen base Q/K/V projections ordinarily need no input backward. Do not decode unused matrices just to keep the launch shape uniform. [B7, §20]

The slab does not preserve weights across the forward/backward gap and does not eliminate the two ordinary decoding passes. If one-branch MLP replay is later enabled in own code, a backward decode can serve both replay and that branch's dX GEMM. That saves an additional replay-related decode, not a decode from the original no-replay baseline.

Reject this extension if the pool is unsafe, dense weights remain retained, additional peak memory harms the score, or fewer launches do not improve the complete command.

## 32. Recent papers: what transfers after checking the actual baseline

### Fast NF4 Dequantization Kernels for Large Language Model Inference

This April 2026 paper studies A100-80GB inference with bitsandbytes 0.47.0, CUDA 12.6, and PyTorch 2.1 on much larger models. It reports 2.0–2.2× decoder speedups and up to 1.54× end-to-end improvement. Its central technique replaces tree decoding with a compact shared lookup table. The evaluation uses a warmup and three profiled passes, with batches 2–64; those results are not T4 training measurements. [C12]

**The important rejection:** bnb 0.50.2 already has a direct sixteen-entry NF4 lookup, and its SM75 fused implementation uses warp-shuffled lookup values. Removing the older decision tree is already reflected in the audited baseline. The paper motivates testing local metadata reuse, but it does not justify claiming its headline gain here. The more concrete remaining experiments are §31's stage fusion and bounded launch batching. No paper kernel is copied or imported. [C9, C10]

### LoRAFusion, EuroSys 2026

LoRAFusion splits the computation at a small low-rank intermediate, then fuses surrounding memory-bound operations while preserving efficient GEMMs. It also includes scheduling across multiple adapter jobs. Its evaluation uses H100/L40S-class systems and larger models, rather than this T4 workload. [C13]

The useful design principle confirms the earlier LoRA plan: keep S=dropout(X)×A as a deliberate graph boundary, and target avoidable large-tensor casts, additions, and reads around it. At M=2,048 and rank 16, one FP16 S occupies only **64 KiB**; the q/v pair occupies 128 KiB per layer. These are calculations for our task. Eliminating that small boundary is a poor trade if it forces a large operand copy or inefficient matrix kernel.

Its multi-job batching does not transfer to the fixed Soup batch/order contract. Its fusion ideas still require original code, unchanged independent dropout draws, and correct T4 compilation. Treat the paper as support for the decomposition, not a newly reproduced speedup or a reason to import its implementation. [C13, D0]

## 33. Round-5 experiment order and evidence

Round 5 changes the plan from “replace a large operator immediately” toward exploiting existing results and fixing observed dispatch/layout waste first. The strongest older memory target—lossless frozen-MLP state—remains relevant.

| Priority | Concrete experiment | Condition for keeping it |
|---|---|---|
| 0 | Record actual package/build versions, per-batch shape/mask branch, and native operator trace | Establish which version-specific paths really execute; preserve existing batches |
| 1 | Round-4 accuracy-copy removal and shared head-output upcast | Same outputs, gradients, metrics/logs; measured benefit |
| 2 | Shared native P for entropy/Rewind, with original compact metric reductions | Native training scalar preserved, duplicate normalizations disappear, every logging consumer correct |
| 3 | Lossless MLP packing reference, then direct FP16 retention with FP32 activation arithmetic | Same saved values and rounded branch gradients; lower measured memory/time |
| 4 | Explicit-repeat SDPA experiment on calls that demonstrably hit the T4 GQA fallback | Native efficient path becomes eligible; all numerical gates and complete-command comparison pass |
| 5 | Exact NF4 stage fusion on eligible fallback/backward calls | Decoded bits and dtype stages match; launch savings survive full-command measurement |
| 6 | Mask sharing, repeat/cast fusion, then optional layer decode slab | Incremental benefit after the earlier changes; correct lifetime/reduction boundaries |
| Later | Full custom CE/backward or chunked head, replay, selective INT8 state, grouped layouts/GEMMs | A concrete remaining bottleneck justifies the extra numerical and ownership work |

The native-dispatch and Python ownership changes contribute no own-Triton share by themselves. Independently written metric reductions, casts, and decoders can perform useful qualifying work, but the unchanged harness must verify at least 5% collectively from step 3 in every measured run. The supplied SwiGLU kernels still do not count. [D1, D3, D4]

Keep component speedups in context. If attention or NF4 decoding really accounts for 5% of training time, doubling that component gives 1/(0.95+0.05/2) ≈ **1.0256×** training speedup. Even eliminating it entirely caps that model at about 1.0526×. Whole-command startup/compilation/saving dilute the effect further. The dispatch issue might change the actual profile, so these are illustrative bounds using the organizer's profile, not predictions. [D1; calculation derived]

### New verification actually performed

Three original NumPy probes were executed and their outputs inspected. They test independent algebra, indexing, and finite-precision counterexamples; none executes the challenge's GPU software.

| CPU probe | Coverage | Observed result |
|---|---|---|
| Shared native-P math, seed 5381 | Twelve finite metric fixtures, 24 denominator cases; prompt/label/padding masks and empty rows | Shared-P entropy and same-order per-row losses matched independently recomputed NumPy formulas in those fixtures; shifted-denominator mistakes were detected |
| Vocabulary tie and row-reduction probes | V=151,936 half-origin logits; extra trailing ignored zero | Argmax changed after FP32 log-softmax as shown in §30; one extra zero changed one FP32 row result by 4.77×10⁻⁷ |
| GQA algebra, seed 5051 | Sixteen FP64 cases, twelve Q/two KV heads, causal and left/right/full masking, T=1/2/5/9 | Output difference ≤6.67×10⁻¹⁶; dQ ≤9.44×10⁻¹⁶; dK/dV ≤1.78×10⁻¹⁵; independent finite differences ≤2.83×10⁻⁹ |
| GQA boundary cases | 288 fully masked query/head rows; FP16 gradient-join counterexample | Zero outputs/dQ for those rows; 2049 versus 2048 rounding boundary reproduced |
| NF4 layout, seed 5051 | Fourteen lengths spanning odd nibble tails, 64-element blocks, tile edges, and 16,384-weight nested-scale boundaries; independent per-matrix slab indexing | Zero bitwise differences between staged and tiled synthetic decodes across 85,255 values |
| NF4 rounding illustration | One million synthetic scale triples; separate FP32 steps versus FP64 multiply/add followed by one FP32 round | 87,585 scale bit patterns and six final FP16 weights changed; this illustrates contraction sensitivity, not a measured CUDA/model error rate |

The audit files are under `tmp/source_audit/round5/`, `round5_attention/`, and `round5_nf4/`. The latter includes a SHA256 source manifest. These are research probes and source snapshots, not a submitted solution.

**Still unmeasured:** actual T4 backend dispatch, loaded torch/Triton versions, CUDA/ATen decoder arithmetic, a working custom autograd integration, every adapter gradient in the real model, AMP trajectories, complete 64-step training, hidden evaluation, cold-command time, sampled GPU peak, and original-kernel share. No GPU speedup or memory-score claim is made.

## 34. Round 6: compact the frozen tied embedding/head without changing its effective values

The strongest new representation candidate concerns the shared embedding/output table. Section 26 proposed keeping the FP32 parameter plus a reusable FP16 head copy. The new audit supports testing **one FP16 stored table with FP32 embedding outputs**, eliminating the redundant widened parameter storage when its provenance is confirmed.

This is a change to a frozen parameter's storage dtype, not a general permission to reduce model precision. Its justification is that every stored value already came from FP16 and both uses can preserve their original arithmetic boundaries.

### Provenance from the actual loading path

The expected chain is:

1. Soup 0.77's `resolve_frozen_base_load_dtype` selects torch.float16 on pre-Ampere CUDA hardware, including T4. The checkpoint's advertised BF16 dtype does not override this request.
2. Transformers materializes tensors into the requested dtype. Qwen2 has no embedding-specific keep-in-FP32 override.
3. PEFT preparation freezes the base and widens non-Params4bit half/BF16 parameters to FP32.
4. Qwen's input embedding and bias-free output head share the same Parameter object through the model's tying mechanism. [E1, A15, B7, A4]

Let E16 be the actual loaded table and E32 its PEFT-widened representation. For finite values, E32 = float32(E16), and float16(E32) = E16 without information loss. Verify that provenance, the final vocabulary, and the actual tie on the running model. Soup's expansion helper returns immediately when no extra tokens are configured; a resized or subsequently modified embedding needs a different assessment. [E1]

### Preserve both uses

| Use | Baseline effective operation | Proposed operation |
|---|---|---|
| Input embedding | Gather rows from E32, producing FP32 | Gather the same rows from E16 and widen them to FP32 |
| Output head under existing FP16 autocast | Linear operation using float16(E32) | The same linear operation using E16 |
| Hidden-state gradient through the head | Multiply by the same effective half weight | Keep the original GEMM with identical effective weight values |
| Frozen table gradient | None | None |

Ordinary embedding forward is index selection for this configuration. Gathering half entries then widening gives the same finite FP32 values as widening the table then gathering, including signed zeros. A first reference can use native gather plus conversion; an original kernel can read half entries and directly write FP32 output. [E2; representation identity derived]

**Returning FP16 embeddings would break the argument.** Qwen RMSNorm converts its normalized result back to its input dtype before multiplying by gamma. A half embedding output therefore introduces a new rounding in the first norm. The CPU negative control changed all 2,048 tested norm outputs when that boundary was moved. Keep the residual stream FP32. [A4; probe in §38]

Install only after final PEFT/adapter setup and before training graphs exist. Changing the existing shared parameter's data storage can preserve its object identity, names, tie, shape, and frozen status. This mutation has not been executed in PyTorch here. Do not create separately stored embedding/head parameters, and do not change dtype-sensitive adapter initialization.

### Correct memory and traffic accounting

The table contains 151,936 × 1,536 = 233,373,696 values.

| Tied-weight component | Calculated size |
|---|---:|
| Original FP32 table | 890.25 MiB |
| One FP16 table | 445.125 MiB |
| Baseline FP32 table plus live half head operand | 1,335.375 MiB |
| Proposed shared half table | 445.125 MiB |
| Reduction while no baseline half head operand is live | **445.125 MiB** |
| Reduction while that half head operand is live | **890.25 MiB** |

These are live-component differences, not measured whole-model peaks. Against §26's FP32-plus-persistent-half-cache version, this removes the FP32 table. It **replaces that representation**; the head-cast traffic savings cannot be counted twice.

Each avoided full-table FP32-to-half conversion models 1,335.375 MiB of logical read/write traffic. At maximum input shape, a fused half-table gather with FP32 output models 18 MiB of traffic versus 24 MiB for the baseline FP32 gather. Native half gather followed by a separate widening models 30 MiB because of its intermediate. These figures exclude caches and do not predict DRAM transfers or time.

### Integration limits and rule boundary

Require a genuinely frozen shared table, no embedding/head adapters or trainable tokens, no relevant modules_to_save, unchanged vocabulary, and Embedding.max_norm=None. The last condition matters because max-norm embedding can modify weights. [E2, A13]

Two less obvious issues must be handled:

- **Reported model dtype changes.** Transformers derives model.dtype from the first floating parameter, which is the embedding here. Its value can change from FP32 to FP16 even though actual embedding outputs remain FP32. Qwen's inspected forward uses the actual inputs_embeds tensor, but every relevant wrapper/hook must be checked for metadata-dependent behavior. [A15, A4]
- **Head execution outside FP16 autocast needs a matching fallback.** A plain FP32 linear call against a half weight can fail. An explicit unsupported-path fallback can temporarily reconstruct E32 and use the original operation; its cost counts. Do not silently alter such calls.

Release every obsolete FP32 view, state-dict snapshot, closure, and diagnostic reference. Post-setup conversion temporarily holds the old and new storage together, reaching 1,335.375 MiB for this component. It cannot erase a peak already reached during loading. Freed allocations can remain reserved, and the harness samples nvidia-smi rather than only live tensors. A one-time timed cache clear after release is an experiment to document, not a reason to clear the cache each step. [D3, A16]

The brief explicitly suggests weight repacking and does not require every frozen tensor to retain its baseline storage dtype. The admission case rests on unchanged frozen values, FP32 embedding outputs, effective head operands, NF4 state, trainable parameters, flags, and saved adapter—not on a gap in the checks. Normal PEFT saving should still omit an untargeted, unresized embedding. Confirm that final behavior and plain evaluator loading. [D0 §§8–9, A13]

An original gather kernel does real work but is unlikely to satisfy the 5% kernel-share requirement alone. No loading speedup was established: the audited four-bit converter already quantizes on the current device. A GPU-to-CPU-to-GPU conversion seen in the eight-bit path is not evidence of such waste in this task. [A15]

## 35. Round 6: exact residual replay and a more precise RMSNorm fusion target

A different memory route is to save a small residual increment alongside an input already retained elsewhere.

For decoder layers 1–27, write the post-attention residual as:

    S32 = FP32_add(X32, widen(A16))

Here X32 is the incoming residual, already retained by that layer's input RMSNorm, and A16 is the FP16 output of the frozen attention output projection. S32 is a distinct tensor retained by the post-attention RMSNorm. The source chain is SDPA under FP16 autocast, the frozen output projection returning its input dtype, then addition to the FP32 residual. [A4, A1–A3, C5]

### Store the increment, reconstruct the original sum

An original owner of the residual-plus-norm pair can keep the existing X32 reference, A16, frozen gamma, and the actual forward reciprocal standard deviation, rstd. In backward, repeat the same FP32 add to obtain S32. No attention, projection, NF4 decoding, or norm-reduction replay is needed.

At B=4,T=512,H=1,536:

| Saved value | Size |
|---|---:|
| Existing X32 anchor | 12 MiB, already retained |
| Original unique S32 | 12 MiB |
| Replacement A16 increment | 6 MiB |
| **Reduction across 27 eligible layers** | **162 MiB** |

Exclude layer 0 initially: its embedding/input-norm prefix ordinarily has no backward, so retaining X32 there solely for replay would add a new 12 MiB anchor. This is the same frozen-prefix qualification used in §§20 and 36.

The custom boundary can return both S32 for the unchanged MLP residual connection and the normalized output. It must avoid keeping S32 in a second native norm graph, Python context, closure, or hook after its forward consumers finish. During backward, reconstruct within a row kernel or use one temporary; multiple independent unpackers could duplicate the workspace.

Preserve the gradient join. Add the direct residual gradient and norm-input gradient at the original FP32 boundary, then return gradients to X and A with the original conversion to A's FP16 dtype. A custom Function must not overwrite a shared incoming gradient.

**Forward replay is essential.** Floating-point addition is not reversible by subtraction: float32(1e−8 + 1) becomes 1; subtracting 1 yields 0, not the original residual. The proposed identity reuses the original inputs and repeats the add.

### RMSNorm itself contains avoidable eager work

The inspected task is expected to use eager Qwen RMSNorm. Its hub-kernel decorator alone does not activate a replacement: Soup supplies no use_kernels/kernel_config opt-in, and Transformers defaults use_kernels to false. Verify the actual bound forward and operator trace before relying on the following census. [A15, E5]

With FP32 residual inputs and frozen FP32 gamma, the two source dtype conversions are no-ops. Forward executes square, mean, epsilon addition, rsqrt, multiplication by rstd, and multiplication by gamma: **six device operations per norm**, or 342 across 57 calls. Under the frozen-prefix assumption, 56 norms participate in backward. [A4]

Their unique norm inputs total 672 MiB at maximum shape, and their saved FP32 rstd values total 0.4375 MiB. Multiple saved references to X or rstd share storage. Because gamma is frozen, native autograd already avoids its weight gradient and need not save the normalized intermediate for that gradient. A fused implementation must not advertise those already-absent allocations as new savings. [B6; sizes derived]

The eager backward is fragmented. The audited derivative path gives approximately eleven arithmetic/reduction stages plus one full input copy per gradient-bearing norm, including accumulation of the two input-gradient contributions. In particular, square backward calls pow(X,1), whose native implementation copies X to its output. Removing that copy models 24 MiB of logical read/write traffic per norm, or **1,344 MiB across 56 norms**. This is transient traffic, not retained state, and a device stage is not necessarily a distinct arithmetic-kernel launch. [E3, E4]

### Fuse while respecting the original numerical sequence

A mathematically compact RMS derivative can change rounding. A reference specification for the audited FP32 path is:

    h        = round32(gY * gamma)
    dot      = row_sum(round32(h * X))
    r3       = round32(round32(rstd * rstd) * rstd)
    dv       = round32(round32(-0.5 * dot) * r3)
    dpow     = round32(dv * rounded_reciprocal(H))
    indirect = round32(dpow * round32(2 * X))
    direct   = round32(h * rstd)
    dX       = round32(direct + indirect)

This records operation boundaries, not a ready kernel. Reduction order, scalar division lowering, rsqrt, and compiler contraction remain relevant. An initial version can retain native reductions and fuse surrounding pointwise work. Original RMSNorm/residual kernels can contribute qualifying work; importing another project's implementation cannot. [B6, E3, E4]

**Separate the claims:** residual reconstruction repeats the original add exactly under matching arithmetic; a newly fused norm still needs CUDA numerical validation. The CPU probe found zero differences for same-sequence replay but 15,650 FP32 differences after rearranging the derivative.

A later extension could retain half attention/MLP increments plus sparse FP32 anchors and replay multiple forward additions. All 56 half increments total 336 MiB before anchors and workspace, compared with the 672 MiB norm-input budget. That broader representation is an alternative to the simple 162 MiB proposal, not an additional saving, and its replay/lifetime complexity makes it lower priority.

## 36. Round 6: replay LoRA operands using saved norm state and the original masks

Section 11 retained one shared q/v dropout operand plus two compressed masks. A narrower replay strategy can remove that shared operand for the later 27 layers.

The attention-input RMSNorm already retains its raw FP32 residual X and actual forward FP32 rstd. Its gamma is frozen. Therefore, an original owner can reconstruct the norm output and then each dropped-out half operand from:

    u32       = round32(X32 * actual_rstd32)
    y32       = round32(gamma32 * u32)
    masked32  = round32(y32 * original_mask)
    dropped32 = round32(masked32 * native_forward_scale)
    D16       = cast_to_half(dropped32)

Keep the separate q and v masks, small low-rank intermediates, trainable parameters, and existing GEMM gradient computation. This adds no matrix multiplication, NF4 decoding, normalization reduction, CPU transfer, or random draw. It replays pointwise dependencies only. [A4, B6, E6; derivation in this research]

### Obtain the real masks without changing dropout

For this nonempty CUDA training input at p=0.05, pinned F.dropout(inplace=False) dispatches through aten::dropout to native_dropout. A scoped replacement can invoke that same native operation once at each original q/v site and receive both the output and its actual Boolean mask. Preserve input dtype, shape, strides, call order, probability, and default generator. Never add an extra stochastic call or regenerate masks during backward. [E6]

CUDA forward uses FP32 source × mask × scale here, followed by A's original autocast to half. Forward and backward derive their scale separately; at p=0.05 both round to the same FP32 value in the CPU check. Do not use that observation to change a general-probability implementation.

Backward through A's input cast widens its half input gradient. Native dropout backward then applies mask and scale in FP32. Preserve that order and the norm gradient edge. Exposing native masks is source-supported; actual CUDA RNG progression and the custom-autograd integration remain untested.

### Incremental savings, without double counting

At maximum shape, each narrow half operand is 6 MiB.

| q/v retained representation | Component storage |
|---|---:|
| Baseline: two half operands in 28 layers plus two byte masks in 27 gradient-bearing inputs | 498 MiB |
| Section 11: shared half operand in 28 layers plus packed masks | 189 MiB |
| New replay: first-layer shared operand only, plus packed masks in all 28 layers | 27 MiB |

The new route removes **162 MiB beyond §11**, or **471 MiB against this conditional baseline component**. The 309 MiB earlier reduction and the new 471 MiB total are alternatives; do not add them. Unchanged low-rank intermediates and already-retained norm state are excluded from the comparison.

Keep the first-layer representation initially because its frozen input norm ordinarily has no saved backward state. Saving a new 12 MiB residual solely to remove a 6 MiB operand would be counterproductive.

The residual replay in §35 and operand replay here replace different allocations and share the same attention-input X anchor, so their **162 MiB incremental component reductions can compose**. This is not a claim of 324 MiB less sampled peak memory.

### Ownership, numerical traps, and cost

Expose the actual forward rstd; do not recompute it or store it in half. A per-forward recipe should reference existing X, rstd, frozen gamma, and compressed masks, without retaining y32 or the original dropped tensors. Reconstruct one required GEMM operand at a time, or use a bounded joint owner. The norm's autograd dependency must remain even when its output value is no longer saved.

The fixed-mask CPU probe matched all 789,504 half operands, including signed zeros. Prematurely narrowing the norm output changed 189,664 operands; narrowing rstd changed 193,034. A finite ±100000 example at dropped positions produced signed zeros in the original order but NaNs if narrowed to infinity before multiplying by zero. These are concrete reasons to preserve the sequence.

Relative to §11, a joint reconstruction pass reads a 12 MiB X instead of a 6 MiB shared operand per eligible layer and performs the two normalization multiplies. That models about 162 MiB of additional logical reads per maximum-shape step before cache effects; separate q/v passes can repeat those reads. Memory savings must justify this added work. An original replay/bit-pack kernel is useful qualifying work, but its measured share and complete-command benefit remain unknown.

## 37. Recent research screening: choose the dependency to retain

### Selective replay supports the new direction

PyTorch's selective-checkpointing discussion distinguishes expensive operations worth retaining from cheap pointwise operations worth replaying. NVIDIA's fine-grained activation-offload documentation similarly recommends recomputation for lightweight normalization/activation work and offload for heavier modules. Sections 35–36 apply that principle to this specific graph using original owners, while preserving the challenge's built-in settings. They do not require importing those frameworks. [E9, E10]

### RMSNorm output sharing: real upstream work, different ownership here

Unsloth PR #13868, merged in October 2026, saves a norm output already retained by its fused LoRA consumer and reconstructs a normalized value through division by gamma. Its discussion explicitly acknowledges non-bitwise gradients and no T4 validation. [E7]

The required alias is missing in the prescribed eager path: q/v dropout retains masks and rounded post-dropout operands; frozen gate/up/head operations do not need the FP32 norm output for a weight gradient. Saving Y instead of X therefore ordinarily swaps one 12 MiB buffer for another. It also introduces reconstruction rounding and restrictions on zero/small gamma. Our CPU fixture failed exact normalized-value reconstruction in 4,356 elements. Keep the existing X anchor for the exact replay opportunities instead of transferring that project's memory percentage. [A3, B6, E7; ownership deduction and CPU evidence]

### CARE-LoRA: why unchanged forward values do not prove unchanged training

The authors' public repositories describe storing a low-rank activation plus a small reconstruction matrix. Their documented backward keeps the B and branch-input gradients exact at the current operands while approximating the A gradient. The main reproduction settings also include unequal-rank memory-budget comparisons and a zero-dropout diagnostic; those are not this fixed r=16, p=0.05 task. [E8]

A simple independent counterexample shows the distinction. Let X be the 2×2 identity, A=[1,0], B=[1], and incoming output gradient G=[0,1] transposed. The current low-rank activation is [1,0] transposed. Projecting X onto that activation's span preserves the current forward exactly, but:

| Gradient | Value |
|---|---|
| Exact dA | [0,1] |
| dA from reconstructed X | [0,0] |

The discarded feature is precisely a direction that A could learn. More generally, reconstruction error contributes dU-transpose × (X−Xhat) to the A-gradient error. A random FP64 fixture preserved the projected forward to about 3.3×10⁻¹⁶ relative error while changing dA by about 86.9% in relative L2. A deliberately low-rank fixture reconstructed both accurately. These are constructed examples, not model or paper reproduction results.

Activation compression is a permitted research direction, but this method belongs in the approximate-backward branch with separate quality validation. Keeping both matrices trainable does not make its gradients equivalent. It is lower priority than the exact representation/replay candidates.

### Disk offload is not automatically a good short-run trade

The 2026 EMNLP industry paper “Techniques for Peak Memory Reduction for LoRA Fine-tuning of LLMs on Edge Devices” combines quantization, checkpointing/offload, softmax approximation, and masking. Its Appendix F states that its implementation did not prefetch the next compute node, so transfers stall computation. Table 21 reports a Qwen-2.5 3B, 512-token quantized forward/backward pass at 5.78 seconds and the checkpoint/offload variant at 7.88 seconds on A100. These are different workloads, not T4 forecasts. [E11]

The paper supports studying the tradeoff, not promising that its complete pipeline fits a 64-step command-time gate. Its approximate softmax also changes the required full-vocabulary objective. Keep the existing targeted offload experiment behind exact GPU-resident storage and cheap replay. [D0, §18]

## 38. Current priorities, composition, and round-6 evidence

Round 6 adds a larger frozen-weight representation opportunity and two small replay opportunities without introducing matrix recomputation. It also makes the RMSNorm fusion target more concrete.

| Priority | Experiment | Main acceptance evidence |
|---|---|---|
| 0 | Observe the prescribed loaded build, actual ties/dtypes, masks, bound norm forwards, and unique saved storage | Source-derived assumptions match the running model |
| 1 | Existing accuracy-copy removal, shared logits upcast, and shared native loss normalization | Same loss/gradients/metrics; duplicate work disappears |
| 2 | Compact tied storage from §34, with FP32 embedding outputs | Half-origin values verified; tie and metadata consumers handled; same effective head operands and adapter/evaluator behavior |
| 3 | Direct half retention of frozen MLP gate/up state from §27 | Larger saved-state candidate works without conversion detours |
| 4 | Original RMSNorm pointwise fusion, then exact residual replay | Numerical sequence validated; no duplicate input saves or unsafe gradient mutation |
| 5 | Dropout-operand replay from §36 after mask/owner integration is correct | Same native masks/RNG progression, exact reconstructed operands, same adapter gradients |
| Conditional | Round-5 T4 attention dispatch fix and exact NF4 decode fusion | Actual trace shows the targeted path; full numerical and timing gates pass |
| Later | Longer residual replay chains, offload, INT8/low-rank compression, or larger custom matrix kernels | Measured residual bottleneck justifies additional complexity and quality risk |

The tied-weight proposal replaces the separate head-cache representation. The residual and dropout replay proposals affect distinct allocations and can compose with lossless MLP state, subject to their real lifetimes. None of these component budgets should be summed into a claimed nvidia-smi peak reduction. Whole-command conversion/compilation/loading/saving costs and the useful original-kernel share still control the result. [D1, D3]

### Verification completed this round

| Original CPU probe | Coverage | Result |
|---|---|---|
| Tied storage, seed 6061 | All 63,488 finite half bit patterns; 32,896 gathered values with repeated IDs | FP32 roundtrip and gather/widen commutation were bitwise exact |
| Tied head and dtype negative control | Small widened-half matrix operands; first-norm dtype boundary | Local head output/input-gradient differences were zero; incorrect half embeddings changed all 2,048 tested norm outputs |
| Residual/RMSNorm, seed 614203 | Twelve fixtures, 43,416 elements, widths 1/7/64/1536; signed zeros and half subnormal increments | Same-sequence residual and gradient replay had zero bit differences; reassociation and output inversion were not exact |
| RMSNorm derivative | Twenty-one FP64 finite-difference coordinates | Maximum scaled discrepancy below 9.96×10⁻¹⁰ |
| Dropout replay, seed 6062 | Two independent fixed masks, each 257×1536 | All 789,504 half operands and corresponding FP32 masked backward values matched bitwise; 19,611 negative-zero operands included |
| Dropout precision negative controls | Early half norm/rstd/backward; finite overflowing source at dropped positions | Detected changed operands/gradients and two new NaNs from incorrect early narrowing |
| Low-rank reconstruction, seed 6061 | Exact two-feature counterexample plus full-/low-rank FP64 fixtures | Forward projection agreement did not imply exact A gradients; residual-error identity held within 8.3×10⁻¹³ |

The probes and new source snapshots are in `tmp/source_audit/round6_tied/`, `round6_rms/`, `round6_replay/`, and `round6_root/`. No third-party kernel implementation was imported or executed. These tests cover representation identities, algebra, and rounding hazards; NumPy does not reproduce CUDA reduction trees, Philox advancement, native rsqrt, cuBLAS algorithms, or autograd storage ownership.

**Next decisive work remains the actual prescribed T4 run.** No GPU implementation, measured speedup, sampled peak reduction, integrated adapter-gradient comparison, 64-step training trajectory, hidden evaluation, or qualifying original-kernel share has been established.

## Sources

All web sources below are primary repositories or official documentation, checked on 30 October 2026. Repository branches and open-PR status may change.

- **D0 — Supplied challenge brief:** [Decentrathon 6.0: Soup Kernel Challenge](https://chatgpt.com/library?file_id=libfile_4b498c83e8088191b29adbaa7996dbda&account_id=personal). Uploaded filename: Decentrathon 6.0_ Soup Kernel Challenge(4).pdf. Russian pp.1–7; task §5 p.2; admission §7 p.3; restrictions §8 p.4; directions §9 pp.4–5; timing/scoring §11 pp.5–7.
- **S1 — Nanochat n-gram submission:** https://github.com/karpathy/nanochat/pull/854
- **S2 — Official Engram project:** https://github.com/deepseek-ai/Engram
- **S3 — Nanochat leaderboard measurement:** https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md
- **S4 — Modded-nanogpt rules and record history:** https://github.com/KellerJordan/modded-nanogpt
- **S5 — Port of host n-gram memory:** https://github.com/KellerJordan/modded-nanogpt/pull/366
- **S6 — Fused cross-entropy source idea:** https://github.com/KellerJordan/modded-nanogpt/pull/199
- **S7 — Current nanochat model source:** https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
- **S8 — Official Qwen2.5-1.5B-Instruct dimensions:** https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json
- **S9 — Linear and activation fusion source idea:** https://github.com/KellerJordan/modded-nanogpt/pull/197
- **S10 — Official Qwen2 architectural implementation:** https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py
- **S11 — Multiplication-order source idea:** https://github.com/KellerJordan/modded-nanogpt/pull/166
- **S12 — NVIDIA Turing tuning guide:** https://docs.nvidia.com/cuda/turing-tuning-guide/
- **R1 — Exact Triton 3.6.0 matmul lowering source:** https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp ; raw source: https://raw.githubusercontent.com/triton-lang/triton/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp
- **R2 — Triton deprecation of older MMA lowering:** https://github.com/triton-lang/triton/pull/5066
- **R3 — Closed restoration proposal:** https://github.com/triton-lang/triton/pull/9970
- **R4 — PEFT four-bit LoRA implementation, upstream main as inspected:** https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/bnb.py
- **R5 — Soup activation and shared-input discussions:** https://github.com/MakazhanAlpamys/Soup/issues/792 ; https://github.com/MakazhanAlpamys/Soup/issues/838
- **R6 — bitsandbytes fused four-bit GEMM dispatch discussion:** https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1949
- **R7 — bitsandbytes autograd implementation, upstream main as inspected:** https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/autograd/_functions.py
- **R8 — Exact Soup 0.77.0 distribution and provenance:** https://pypi.org/project/soup-cli/0.77.0/ ; source at publishing commit: https://github.com/MakazhanAlpamys/Soup/tree/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli ; functions inspected: `trainer/rewind_hf.py::row_losses`, `row_losses_from_token_losses`, and `trainer/sft.py::_assert_finite_training_state`.
- **R9 — Nanochat optimizer implementation:** https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py
- **R10 — PyTorch 2.11 AdamW behavior:** https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html
- **D1 — Pinned official starter README, rules, and dependency instructions:** https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md
- **D2 — Pinned fixed task:** https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/task/task.yaml
- **D3 — Pinned harness:** https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/common.py ; https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/_train.py ; https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py
- **D4 — Provided SwiGLU example, excluded from own-kernel credit:** https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/solution/swiglu.py
- **A1 — bitsandbytes 0.50.2 Linear4bit module:** https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py
- **A2 — bitsandbytes 0.50.2 CUDA operators:** https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py
- **A3 — bitsandbytes 0.50.2 autograd and dispatch:** https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py
- **A4 — Transformers 5.16.1 Qwen2:** https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py
- **A5 — PyTorch 2.11 saved-tensor hooks:** https://docs.pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.saved_tensors_hooks
- **A6 — PyTorch 2.11 AMP and custom autograd:** https://docs.pytorch.org/docs/2.11/notes/amp_examples.html
- **A7 — HyC-LoRA, MLSys 2025:** https://proceedings.mlsys.org/paper_files/paper/2025/hash/5431dca75a8d2abc1fb51e89e8324f10-Abstract-Conference.html ; https://proceedings.mlsys.org/paper_files/paper/2025/file/5431dca75a8d2abc1fb51e89e8324f10-Paper-Conference.pdf
- **A8 — GACT, ICML 2022:** https://proceedings.mlr.press/v162/liu22v.html
- **A9 — ActNN, ICML 2021:** https://proceedings.mlr.press/v139/chen21z.html
- **A10 — PRAC, ICML 2026:** https://proceedings.mlr.press/v306/li26je.html
- **A11 — Transformers 5.16.1 Trainer save path:** https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py
- **A12 — PEFT 0.20.0 PeftModel saving:** https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py
- **A13 — PEFT 0.20.0 state filtering and embedding-save decision:** https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/save_and_load.py
- **A14 — bitsandbytes 0.50.2 QuantState serialization:** https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/functional.py
- **A15 — Transformers 5.16.1 loading:** https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/modeling_utils.py ; https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/core_model_loading.py
- **A16 — PyTorch 2.11 CUDA graphs, AMP, and private pools:** https://docs.pytorch.org/docs/2.11/notes/cuda.html
- **A17 — Nanochat cyclic-GC policy, inspected upstream master:** https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py
- **A18 — Python 3.13 garbage collector:** https://docs.python.org/3.13/library/gc.html
- **B1 — TRL 0.29.1 exact SFT loss/metric path and collator:** https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py ; inspected `SFTTrainer.compute_loss`, including the causal-shift `.contiguous()` before accuracy and attention-mask entropy reduction.
- **B2 — TRL 0.29.1 entropy implementation:** https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/utils.py ; inspected `entropy_from_logits`, default chunk size 128.
- **B3 — Accelerate 1.14.0 native AMP and FP32 returned-output conversion:** https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py ; https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py ; inspected native-AMP initialization, model preparation, `convert_to_fp32`, and `ConvertOutputsToFp32`.
- **B4 — Transformers 5.16.1 causal loss:** https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/loss/loss_utils.py ; inspected `ForCausalLMLoss` and `fixed_cross_entropy`.
- **B5 — PyTorch 2.11.0 autocast cache eligibility:** https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/autocast_mode.cpp ; inspected `cached_cast`, including `arg.requires_grad()` in `can_try_cache`.
- **B6 — PyTorch 2.11.0 CE implementation and saved reverse-mode operands:** https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/LossNLL.cpp ; https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml ; inspected plain cross-entropy's log-softmax/NLL path and `_log_softmax`, `nll_loss_forward`, and `_to_copy` derivatives.
- **B7 — PEFT 0.20.0 base-parameter freezing and dtype preparation:** https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/other.py ; inspected `prepare_model_for_kbit_training`.
- **B8 — PyTorch 2.11 AMP behavior and numerical accuracy:** https://docs.pytorch.org/docs/2.11/amp.html ; https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html .
- **C1 — Transformers 5.16.1 SDPA integration and KV repetition:** https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/integrations/sdpa_attention.py ; inspected `repeat_kv`, `use_gqa_in_sdpa`, and `sdpa_attention_forward`.
- **C2 — Transformers 5.16.1 causal/padding mask decisions:** https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/masking_utils.py ; inspected `_ignore_causal_mask_sdpa`, `sdpa_mask`, and `create_causal_mask`.
- **C3 — PyTorch 2.11.0 CUDA SDPA eligibility:** https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp ; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/sdp_utils_cpp.h ; inspected architecture, dtype, shape, stride, and dense-head checks, including memory-efficient GQA rejection.
- **C4 — PyTorch 2.11.0 attention composition, mask conversion/alignment, and math defaults:** https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/attention.cpp ; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/Context.h ; inspected `convert_boolean_attn_mask_`, `preprocess_mask`, GQA repetition, math SDPA, and the default reduced-precision setting.
- **C5 — PyTorch 2.11.0 lower-precision SDPA autocast:** https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/autocast_mode.h ; CUDA registration in B5.
- **C6 — PyTorch 2.11.0 efficient-attention output layout:** https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/attention.cu ; inspected native wrapper transposes and [B,T,H,D] allocation.
- **C7 — Official SDPA behavior and version-sensitive upstream corroboration:** https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html ; https://github.com/huggingface/transformers/issues/48633 ; https://github.com/pytorch/pytorch/pull/191085 . The issue is inference evidence; the later PR is not proof that the challenge environment includes its changes.
- **C8 — PyTorch 2.11.0 native CUDA log-softmax and NLL:** https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/SoftMax.cu ; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Loss.cu ; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/SoftMax.cpp ; inspected large-row forward dispatch, row reductions, NLL backward, and log-softmax backward.
- **C9 — bitsandbytes 0.50.2 standalone NF4/scale decoding:** https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/csrc/kernels.cu ; https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/csrc/ops.cu ; inspected direct sixteen-entry lookup, nibble order, scale indexing, destination conversion, and launch setup.
- **C10 — bitsandbytes 0.50.2 native SM75 four-bit GEMM:** https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/csrc/gemm_4bit_sm75.cu ; https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/csrc/gemm_4bit_sm75.cuh ; https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/csrc/gemm_4bit_common.cuh ; dispatch source A2.
- **C11 — bitsandbytes 0.50.2 compiler flags and NVIDIA semantics:** https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/CMakeLists.txt ; https://docs.nvidia.com/cuda/archive/12.6.3/cuda-compiler-driver-nvcc/index.html ; inspected `--use_fast_math`, `--ftz`, and `--fmad`.
- **C12 — Fast NF4 Dequantization Kernels for Large Language Model Inference, April 2026, version 1:** https://arxiv.org/abs/2604.02556 ; https://arxiv.org/html/2604.02556v1 .
- **C13 — LoRAFusion: Efficient LoRA Fine-Tuning for LLMs, EuroSys 2026, version 1:** https://arxiv.org/abs/2510.00206 ; https://arxiv.org/html/2510.00206v1 ; official project: https://github.com/CentML/lorafusion .
- **E1 — Soup 0.77 frozen-base dtype and vocabulary expansion, publishing commit:** https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/utils/gpu.py ; https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/utils/data_pipeline.py ; SFT setup in R8. Inspected `resolve_frozen_base_load_dtype`, `resolve_base_load_dtype`, and no-extra-token handling.
- **E2 — PyTorch 2.11 embedding forward and mutation boundary:** https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Embedding.cpp ; https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/modules/sparse.py ; inspected index selection and max_norm behavior.
- **E3 — PyTorch 2.11 RMSNorm's constituent backward operations:** https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/FunctionsManual.cpp ; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Pow.cpp ; derivative definitions in B6. Inspected `pow_backward`, `mean_backward`, `mul_tensor_backward`, and exponent-one copying.
- **E4 — PyTorch 2.11 CUDA arithmetic underlying RMSNorm:** https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/PowKernel.cu ; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu ; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/ReduceMomentKernel.cu ; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/UnaryOpsKernel.cu .
- **E5 — Transformers 5.16.1 opt-in hub-kernel integration:** https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/integrations/hub_kernels.py ; `from_pretrained` and `set_use_kernels` in A15.
- **E6 — PyTorch 2.11 native dropout dispatch and arithmetic:** https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/functional.py ; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Dropout.cpp ; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu ; inspected CUDA fused-path predicate, original mask/Philox handling, source-mask-scale arithmetic, and native backward.
- **E7 — Recent RMSNorm output-sharing proposal and limitations:** https://github.com/unslothai/unsloth/pull/13868 ; merged October 2026. Read design and limitations, not imported as an implementation.
- **E8 — CARE-LoRA authors' implementation documentation:** https://github.com/fishandyu/CARE-LoRA ; https://github.com/fishandyu/CARE-LoRA-Diffusion ; inspected README method, approximate A-gradient contract, reproduction rank choices, and dropout diagnostic constraints. Paper identifier: arXiv:2607.11940; its full text was not retrievable in this round, so the technical assessment uses these primary project documents.
- **E9 — PyTorch selective activation checkpointing design:** https://pytorch.org/blog/activation-checkpointing-techniques/ ; published March 2025, page updated April 2026; inspected selective retention of expensive operations and replay of pointwise work.
- **E10 — NVIDIA fine-grained activation offloading:** https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/fine_grained_activation_offloading.html ; inspected the replay/offload tradeoff and dependency/lifetime requirements.
- **E11 — Techniques for Peak Memory Reduction for LoRA Fine-tuning of LLMs on Edge Devices, EMNLP Industry 2026:** https://aclanthology.org/2026.emnlp-industry.88/ ; https://aclanthology.org/2026.emnlp-industry.88.pdf ; inspected Appendix F and Table 21, including non-prefetched transfers and the separately changed softmax objective.
