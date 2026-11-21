# Nanochat and n-gram ideas for Soup Kernel Challenge

Updated: 30 October 2026 (UTC), research round 3.  
Status: official starter inspected at a pinned commit; versioned dependency source audit; CPU representation and derivative checks. No challenge GPU benchmark or submission was run in this session.

## Session handoff: read this first

- Borrowing ideas from nanochat and modded-nanogpt is useful and expressly compatible with the supplied Soup brief when sources are credited and the implementation is original.
- Adding a learned n-gram / Engram table is prohibited: the model and trainable parameter set are fixed. A fixed, nontrainable table that changes predictions is still a model change.
- Do not import speedrun optimizer changes, sampled softmax, auxiliary prediction losses, fewer steps, different batches, different activations, or altered dropout.
- Strongest new candidate: save the frozen MLP gate/up operands in FP16 when source inspection proves their FP32 values are only widened FP16 results. Restore FP32 before the existing backward. This can remove a conditional 1,960 MiB of retained state at maximum shape, with no extra matrix multiplication or activation-quantization error. All 63,488 finite FP16 bit patterns passed the CPU round-trip check. See §17; this is not a measured GPU saving.
- Official starter is now accessible and inspected at commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. It explicitly permits own recomputation and offloading while prohibiting changes to the built-in switches. It confirms no checkpointing, q/v-only LoRA, dropout 0.05, and the dependency versions. See §16.
- The provided example is already fused SwiGLU forward/backward. Its kernels and rewrites cannot count as our own. Keep it as an allowed baseline improvement; add independent useful packing/reduction kernels and measure their share. Do not claim the example's reported memory improvement as our new discovery.
- Main implementation pair: lossless saved-MLP packing plus original full-vocabulary loss/metric work. Keep substantial matrix products in cuBLAS. Optional next memory extensions are one-branch MLP replay or carefully validated INT8 saved-state compression; compare their incremental benefit after lossless packing. See §§18–19.
- New hardware constraint: the pinned Triton 3.6.0 source declines ordinary Tensor Core dot lowering below compute capability 8.0. T4 is 7.5. Keep substantial GEMMs in PyTorch/cuBLAS initially; focus original Triton work on reductions, decoding, casts, masks, and elementwise fusion. See §10.
- Earlier dropout-state hypothesis remains available. Its 315/483 MiB budgets assumed byte masks retained in all 28 layers; under the now-confirmed no-checkpointing configuration, a first-layer input without gradients can reduce those illustrative budgets to 309/477 MiB. Verify runtime hooks and count actual saved storage before relying on either. See §§11 and 20.
- New NF4 direction: cache only decoded nested scales before considering dense-weight caches. With block size 64, all seven projections across 28 layers need about 78.09 MiB of FP32 scales versus about 2.44 GiB for all FP16 dense weights. Verify the loaded bitsandbytes dispatch before designing a fused GEMM. See §12.
- Exact Soup 0.77.0 source exposes repeated CPU scalar readbacks in rewind row logging and pre-save finite checks. Batch those transfers while preserving every log and failure check. These are secondary full-command opportunities, not a substitute for the required own-Triton share. See §13.
- A new n-gram-inspired hypothesis is exact memoization of frozen token-local computations before the first attention layer. Rank it low: the first base Q/K/V projection accounts for about 0.24% of the transformer's linear forward work.
- Runtime numerical caches differ from prohibited precompiled caches. Any cache must preserve the computation and pay its setup and memory costs within the measured run.
- Soup measures the whole command through a saved adapter, including compilation. Nanochat's reported training time excludes evaluations and logging. Gains and timing conventions cannot be transferred directly.
- Every candidate remains unmeasured on the prescribed T4. Passing numerical tolerances does not authorize changing the task.
- Next implementation step: verify the exact gate/up FP16-to-FP32 provenance on T4, prototype scoped pack/unpack hooks that restore original FP32 before backward, and compare saved operands and all adapter gradients. Then measure complete-command time, sampled peak memory, and useful own-Triton share. Run the official harness unchanged. See §22.

## 1. What the n-gram idea actually does

An n-gram is a short sequence of n tokens. A bigram uses two, and a trigram uses three. In the relevant learned-memory approach, those token sequences select trainable vectors that augment a Transformer.

The likely direct reference is [nanochat PR #854](https://example.invalid/redacted-github). It hashes bigrams/trigrams into a host-memory table and injects retrieved features into attention values. The author reports 91.74 minutes against a 99-minute reference on 8 H100s. The PR was open when checked; these are submitted results, not an independently verified Soup result. Its approximately 72 GiB table is omitted from the saved checkpoint, which therefore cannot run alone. [S1]

This mechanism changes the learned model. It does not prove that an unchanged Qwen training step executes faster. The official Engram project also describes conditional memory as an augmentation of a model backbone. [S2]

The distinction matters even when a paper describes the lookup itself as inexpensive. Lookup cost, improved convergence, model capacity, checkpoint completeness, and equivalent execution of an existing model are different questions.

### Two related speedruns, with different rules

Nanochat measures time to exceed a GPT-2 capability target. Its leaderboard documentation defines reported training time as training iterations, excluding evaluations and logging. [S3]

Modded-nanogpt is a separate FineWeb loss speedrun. Its rules permit architecture and batch/sequence changes that Soup forbids. Its history includes both architectural changes such as bigram embeddings and systems changes such as kernel fusion. [S4]

A port of the host n-gram idea, [modded-nanogpt PR #366](https://example.invalid/redacted-github), explicitly compares 1,140 candidate steps against 1,285 baseline steps. That is a concrete example of a result involving fewer optimization steps. Soup fixes 64 steps. [S5, D0 §5]

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

If profiling shows repeated conversions of an unchanged output matrix, test a reusable representation that matches the baseline's effective GEMM dtype. A 151,936 × 1,536 fp16 copy occupies approximately 445.125 MiB.

That is a speed/memory tradeoff, not free acceleration. First verify whether autocast already reuses the conversion. Preserve weight tying, saved parameter identity, and dtype behavior. Build and free any extra representation inside normal execution, with its cost included.

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

The following order is an engineering judgment based on the rules, the supplied profile, and the derived opportunity sizes.

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

The next session should follow §22, using the source-derived lossless storage specialization first. The decisive missing evidence is the prescribed T4 run rather than another model architecture.

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

Round 2 used the inspected upstream implementation. Round 3 resolved the starter's documented bitsandbytes pin to 0.50.2 and inspected that version's module, autograd, and CUDA dispatch sources. The declared lower bound in Soup's package metadata alone had not established this. Verify the actual loaded version and operator trace; shorter batches can take another path. [D1, A1–A3]

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

The inspected routine calculates cross-entropy in the logits' dtype, then casts the small token-loss result to FP32 for its row mean. Training loss can follow a different FP32 path. Consequently, do not blindly substitute the training scalar or assume one differently rounded CE output reproduces rewind. Match the required metric behavior before combining reductions.

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

## 22. Revised implementation decision and verification gates

The best new first experiment is **lossless storage of the two widened-FP16 MLP operands**. It has a concrete source proof, a large conditional storage budget, and avoids both replay FLOPs and approximate-gradient risk.

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

## Sources

All web sources below are primary repositories or official documentation, checked on 30 October 2026. Repository branches and open-PR status may change.

- **D0 — Supplied challenge brief:** [Decentrathon 6.0: Soup Kernel Challenge](https://chatgpt.com/library?file_id=libfile_4b498c83e8088191b29adbaa7996dbda&account_id=personal). Uploaded filename: Decentrathon 6.0_ Soup Kernel Challenge(4).pdf. Russian pp.1–7; task §5 p.2; admission §7 p.3; restrictions §8 p.4; directions §9 pp.4–5; timing/scoring §11 pp.5–7.
- **S1 — Nanochat n-gram submission:** https://example.invalid/redacted-github
- **S2 — Official Engram project:** https://github.com/deepseek-ai/Engram
- **S3 — Nanochat leaderboard measurement:** https://example.invalid/redacted-github
- **S4 — Modded-nanogpt rules and record history:** https://example.invalid/redacted-github
- **S5 — Port of host n-gram memory:** https://example.invalid/redacted-github
- **S6 — Fused cross-entropy source idea:** https://example.invalid/redacted-github
- **S7 — Current nanochat model source:** https://example.invalid/redacted-github
- **S8 — Official Qwen2.5-1.5B-Instruct dimensions:** https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json
- **S9 — Linear and activation fusion source idea:** https://example.invalid/redacted-github
- **S10 — Official Qwen2 architectural implementation:** https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py
- **S11 — Multiplication-order source idea:** https://example.invalid/redacted-github
- **S12 — NVIDIA Turing tuning guide:** https://docs.nvidia.com/cuda/turing-tuning-guide/
- **R1 — Exact Triton 3.6.0 matmul lowering source:** https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp ; raw source: https://raw.githubusercontent.com/triton-lang/triton/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp
- **R2 — Triton deprecation of older MMA lowering:** https://github.com/triton-lang/triton/pull/5066
- **R3 — Closed restoration proposal:** https://github.com/triton-lang/triton/pull/9970
- **R4 — PEFT four-bit LoRA implementation, upstream main as inspected:** https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/bnb.py
- **R5 — Soup activation and shared-input discussions:** https://example.invalid/redacted-github ; https://example.invalid/redacted-github
- **R6 — bitsandbytes fused four-bit GEMM dispatch discussion:** https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1949
- **R7 — bitsandbytes autograd implementation, upstream main as inspected:** https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/autograd/_functions.py
- **R8 — Exact Soup 0.77.0 distribution and provenance:** https://pypi.org/project/soup-cli/0.77.0/ ; source at publishing commit: https://example.invalid/redacted-github ; functions inspected: `trainer/rewind_hf.py::row_losses`, `row_losses_from_token_losses`, and `trainer/sft.py::_assert_finite_training_state`.
- **R9 — Nanochat optimizer implementation:** https://example.invalid/redacted-github
- **R10 — PyTorch 2.11 AdamW behavior:** https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html
- **D1 — Pinned official starter README, rules, and dependency instructions:** https://example.invalid/redacted-github
- **D2 — Pinned fixed task:** https://example.invalid/redacted-github
- **D3 — Pinned harness:** https://example.invalid/redacted-github ; https://example.invalid/redacted-github ; https://example.invalid/redacted-github
- **D4 — Provided SwiGLU example, excluded from own-kernel credit:** https://example.invalid/redacted-github
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
- **A17 — Nanochat cyclic-GC policy, inspected upstream master:** https://example.invalid/redacted-github
- **A18 — Python 3.13 garbage collector:** https://docs.python.org/3.13/library/gc.html

