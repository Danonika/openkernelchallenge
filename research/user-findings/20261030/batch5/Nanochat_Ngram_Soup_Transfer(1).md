# Nanochat and n-gram ideas for Soup Kernel Challenge

Updated: 30 October 2026 (UTC), research round 2.  
Status: primary-source research, exact Soup 0.77.0 wheel inspection, and CPU mathematical checks. No challenge GPU benchmark or submission was run in this session.

## Session handoff: read this first

- Borrowing ideas from nanochat and modded-nanogpt is useful and expressly compatible with the supplied Soup brief when sources are credited and the implementation is original.
- Adding a learned n-gram / Engram table is prohibited: the model and trainable parameter set are fixed. A fixed, nontrainable table that changes predictions is still a model change.
- Do not import speedrun optimizer changes, sampled softmax, auxiliary prediction losses, fewer steps, different batches, different activations, or altered dropout.
- Main candidates: full-vocabulary loss plus all required metrics; lossless compression of retained q/v LoRA dropout operands; original LoRA scale/add and existing Qwen activation fusion; own removal of padding within the existing batches.
- New hardware constraint: the pinned Triton 3.6.0 source declines ordinary Tensor Core dot lowering below compute capability 8.0. T4 is 7.5. Keep substantial GEMMs in PyTorch/cuBLAS initially; focus original Triton work on reductions, decoding, casts, masks, and elementwise fusion. See §10.
- New memory hypothesis: one shared effective dropout operand plus two separately packed original masks can replace two full operands plus two byte masks. The conditional retained-state reduction is 315 MiB for FP16 operands or 483 MiB for FP32 operands at B=4, T=512, H=1,536, L=28. These are storage calculations, not measured VRAM savings. CPU reconstruction and derivative checks passed. See §11.
- New NF4 direction: cache only decoded nested scales before considering dense-weight caches. With block size 64, all seven projections across 28 layers need about 78.09 MiB of FP32 scales versus about 2.44 GiB for all FP16 dense weights. Verify the loaded bitsandbytes dispatch before designing a fused GEMM. See §12.
- Exact Soup 0.77.0 source exposes repeated CPU scalar readbacks in rewind row logging and pre-save finite checks. Batch those transfers while preserving every log and failure check. These are secondary full-command opportunities, not a substitute for the required own-Triton share. See §13.
- A new n-gram-inspired hypothesis is exact memoization of frozen token-local computations before the first attention layer. Rank it low: the first base Q/K/V projection accounts for about 0.24% of the transformer's linear forward work.
- Runtime numerical caches differ from prohibited precompiled caches. Any cache must preserve the computation and pay its setup and memory costs within the measured run.
- Soup measures the whole command through a saved adapter, including compilation. Nanochat's reported training time excludes evaluations and logging. Gains and timing conventions cannot be transferred directly.
- Every candidate remains unmeasured on the prescribed T4. Passing numerical tolerances does not authorize changing the task.
- Next implementation step: capture loaded package versions, a short T4 trace, and an alias-aware saved-tensor census. Then implement full-vocabulary reductions and test dropout-state compression against actual native masks and AMP. Keep large matrix products in cuBLAS. Run the official harness unchanged.

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

The official challenge starter repository could not be retrieved in this session. This is an access limitation, not evidence that it is private or absent. Round 2 did retrieve and inspect the exact public Soup 0.77.0 wheel; that identifies concrete package-level functions but does not establish the starter's overrides or the exact dependency versions installed on Kaggle. The full forbidden-package list still requires the current starter.

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
| Implement custom offload or general recomputation | Unresolved interpretation | The broad permission for accompanying changes is not explicit approval for every strategy. Prioritize the clearly encouraged kernel directions. |

Source: D0 §§7–9, pp.3–5. These assessments are interpretations of the supplied rules; the judges retain final authority.

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

### A practical first fusion

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

Custom offload is lower priority because its rules interpretation and PCIe costs require additional investigation. Do not flip the prohibited built-in offload setting.

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
| 1 | Fuse full-vocabulary loss/entropy/accuracy/per-row reductions; then chunk the output head | Peak memory and repeated logits traffic | Match every output and gradient before full-run benchmarking. |
| 2 | Compress retained q/v dropout operands and pack their separate masks | Repeated full-width saved activations | Count unique saved storage first; reproduce native masks, arithmetic, and both branches bit for bit. |
| 3 | Fuse existing SwiGLU, LoRA scale/add, normalization, and cast operations | Elementwise overhead across layers | Keep large GEMMs in cuBLAS; confirm originality and combined own-kernel share. |
| 4 | Own within-batch unpadding | Remove work across the full model | Preserve attention boundaries, token accounting, masks, and stochastic behavior. |
| 5 | Fuse NF4 decoding/casts and test decoded-scale caching | Repeated conversion and nested-scale decoding | Inspect actual dispatch, dequantization time, and added memory; avoid an unmeasured full Triton GEMM. |
| 6 | Batch required row-statistic and finite-check CPU transfers | Repeated synchronization and end-of-run overhead | Preserve values, order, error semantics, and completion inside the measured command. |
| 7 | Same-algorithm AdamW fusion | Small-parameter launch overhead | First confirm the existing optimizer is a meaningful cost and its implementation may be replaced without changing the fixed configuration. |
| 8 | Frozen first-layer memoization or selective dense-weight caches | Narrow repeated representation work | Stop if lookup, setup, or extra memory exceeds the small available saving. |

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

Not established:

- A speedup or memory score on Kaggle T4.
- The exact challenge-loaded patch points, saved tensor lifetimes, or resolved PEFT/Transformers/bitsandbytes versions. Public Soup 0.77.0 function locations are known, but the running environment remains uninspected.
- Current starter implementation details or the full forbidden-library list.
- Organizer approval of speculative caching/offload interpretations.
- A completed solution, runnable Kaggle notebook, or winning rank.

The next session should obtain the short trace and storage census in §14, then implement the highest-priority original kernels. The decisive missing evidence is now the prescribed T4 run rather than another model architecture.

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

These sources describe the inspected upstream implementation, not proof of the version installed by the challenge. Soup 0.77.0 declares a bitsandbytes lower bound rather than an exact version. Record the resolved version and actual operator trace first. Shorter batches can also take another path.

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

