# Soup research cycle: NF4, q/v rank-16 LoRA, and saved backward state

Research date: 2026-10-30. Cloud-only source research and CPU-oracle coordination. **No GPU run, training, installation, authentication, kernel copying, or external write.** Deduplicated against the newly assembled 118-row catalog at `soup_cloud_research/research/soup-20261030/master.json`.

## Decisions first

1. **Retain SOUP-LORA-001:** a two-output B-adjoint design that reuses an upstream-gradient tile for both activation and parameter gradients. Its two reductions must be budgeted together; a narrow, one-axis-partial version is specified below. This is different from keeping dependent A→B GEMMs on chip.
2. **Retain SOUP-LORA-002, conditionally accepted by the research jury:** losslessly bit-pack only the boolean masks saved by the existing q/v dropout operations. Keep the native dropout operation, both RNG draws, and native backward. This is a memory hypothesis, not a speed claim.
3. **Do not add another NF4-LUT row:** a 2026 dequantization paper motivates an on-chip lookup experiment, but its tree-decoder motivation is already stale for the pinned bitsandbytes baseline. The baseline directly indexes its NF4 table.
4. **Refine L6-16 / inductor_b2b_fusion rather than duplicating them:** materializing the tiny rank-16 state can be preferable to full A/B fusion. Treat the rank boundary as a deliberate split, not necessarily an intermediate to eliminate.

No proposal has established a Soup/T4 speedup, memory-peak improvement, or >=5% own-kernel share. Independent CPU tests establish the small cast and mask-representation contracts below. Jury review is a research classification, not organizer acceptance.

## Source authority and exact pins

| Key | Primary source and pin | What it establishes |
|---|---|---|
| R | [Official README](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md), commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb` | Fixed task, independent Triton/autograd/patch permission, normal launches, full timing and task gates. |
| P | [PEFT 0.20.0 4-bit LoRA forward, lines 513–565](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/lora/bnb.py#L513-L565) | Native base call, defensive clone, dropout→A→B→scale→add; no permission to erase cast boundaries or clone safety. |
| LF | [LoRAFusion §5.1, arXiv 2510.00206v1](https://arxiv.org/html/2510.00206v1#S5.SS1), submitted 2025-09-30 | Split at rank-r state; sibling dS/dB fusion reuses upstream gradients. Paper evidence is H100/L40S, not Soup/T4. |
| LF-pin | [LoRAFusion artifact README](https://github.com/CentML/lorafusion/blob/c48d7fdb1b5458e9df00963a63df00436f89a059/benchmarks_paper/README.md), commit `c48d7fdb1b5458e9df00963a63df00436f89a059`, 2025-09-17 | Source repository/version context; tuned devices listed there do not include T4. No source kernel was copied or used. |
| T-dispatch | [PyTorch Dropout.cpp, lines 48–59 and 155–166](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/Dropout.cpp#L48-L59) | CUDA nonempty floating dropout at 0<p<1 selects native dropout in this version. CPU ordinary dropout can follow a different path. |
| T-mask | [CUDA native dropout, lines 416–456](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cuda/Dropout.cu#L416-L456) | Native CUDA mask is bool; backward consumes bool. |
| T-save | [Autograd derivative entry, lines 620–625](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/tools/autograd/derivatives.yaml#L620-L625) | Backward uses saved mask `result1`. |
| T-hook | [Saved tensor hooks, lines 297–365](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/autograd/graph.py#L297-L365), [official autograd notes](https://docs.pytorch.org/docs/stable/notes/autograd#hooks-for-saved-tensors) | Storage may use pack/unpack representation; restore exact contents, shape, dtype, device; avoid mutation/reference cycles; unpack may repeat; nesting needs care. |
| B | [bitsandbytes NF4 decoder, lines 26–43 and 108](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/csrc/kernels.cu#L26-L43), [input-only 4-bit backward](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/autograd/_functions.py#L370-L400), commit `08a9956b939857ae2c4fbd56753960cc370fa7cf` | Existing direct lookup and frozen-weight backward; prior nested-NF4 contract remains authoritative. |
| NF-paper | [Fast NF4 Dequantization Kernels, arXiv 2604.02556v1](https://arxiv.org/html/2604.02556v1), 2026-04-02 | A100 inference hypothesis: stage a small NF4 table on chip. Its performance and baseline description are not our baseline evidence. |

PyTorch pin `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457` resolves v2.15.0, dated 2026-10-28. **The official Soup notebook does not pin Torch.** This pin supports a versioned semantic hypothesis; actual Kaggle version, dispatch, saved-mask dtype and autocast behavior remain admission tests. It does not authorize upgrading the runtime.

The CPU lane actually used `torch==2.15.0+cpu`, build commit `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`, not that tag commit. A subsequent official-source lookup at the actual build commit confirmed matching blob SHAs for all four inspected files: Dropout.cpp `ffad792af8c68df18d946da23eca3aef4fd5a084`, Dropout.cu `73727eab117d50c06cd3799fa7d1a5e6ec06bad1`, derivatives.yaml `d0ee6fb1e7b5261eb1e02eaf07cf08deff09a95f`, graph.py `51d5e69513969fc337ce698b3558b09e46ec4f46`. This resolves those source-file contents only, not compiler/library or Kaggle equivalence.

## Independent CPU evidence

Reproducible artifacts: [CPU report](cpu_results.md), [machine results](cpu_tests/results.json), [original tests](cpu_tests/run_cpu_semantics.py), [runner](cpu_tests/run.sh). These are toy CPU semantics, not candidate GPU implementations or performance measurements.

| Candidate | Observed evidence | Remaining limit |
|---|---|---|
| SOUP-LORA-001 | M17,N23,r16 no-dropout autograd branch exactly matches the stated adjoints for normal, subnormal and overflow cases. Moving scale before the first half cast changes all 391 D entries in the constructed subnormal case. Removing final FP16 dB rounding changes all 368 normal dB entries, max absolute difference 0.0001176297664642334. Overflow classification is retained. | The sibling tiled algorithm, partial reduction, actual CUDA autocast, dropout integration, optimizer and GradScaler skip were not exercised. |
| SOUP-LORA-002 | Ten explicit CPU native_dropout cases (train/eval × sizes 1, 7, 8, 9, 257), each with two consecutive calls, preserve outputs, input gradients and final RNG state exactly. Strided bool roundtrip is exact. Size 257 pair has 514 bytes raw payload versus 66 bytes packed. | Tests retain debug copies, so **no memory saving was measured**. Actual nn.Dropout CUDA dispatch, repeated retained-graph unpack, hook nesting, T4 codec and peak/timing remain untested. |
| SOUP-NEW-003 /007 refinement | Synthetic double-quant counterexample: separate FP32 scale multiply then offset yields half 0.5009765625; single-rounded contraction yields0.50048828125. All 16 nibbles, 256 synthetic scale codes, odd tails and block boundaries covered. | Synthetic codebook values are not actual bnb QuantState; no decoded real-weight equivalence or GPU FMA behavior claimed. |

## SOUP-LORA-001 — shared-gradient, two-output B-adjoint

### Novelty and attribution

LoRAFusion describes graph cuts at the small rank state and fusing the B-branch's two backward products around their shared large gradient. The transfer here is only that sibling-product reuse. Its full frozen-GEMM fusion, new RNG generation, multi-adapter scheduling, packing, distributed runtime and source kernels are not adopted. [LF]

This is distinct from L6-16/inductor_b2b_fusion, which concern dependent GEMMs. Here both products consume the same gradient but reduce over different axes. It also differs from generic pointwise L6-10 and grouped independent projection launches.

### Expected Soup precision contract

Use PEFT-oriented matrices: saved rank state `S16[M,16]`, B's autocast weight `B16[N,16]`, and upstream add gradient `G32[M,N]`. N=1536 for q, 256 for v. The incoming G already includes any GradScaler multiplier propagated through the graph.

The source-derived expected contract is:

- `G16 = fp16(G32)` at the FP32-base / FP16-adapter addition boundary.
- `D16 = fp16(G16 * 2)` for alpha/r=32/16.
- `dS16 = fp16(D16 @ B16)`.
- `dB16 = fp16(D16.T @ S16)`; the FP32 master parameter receives `fp32(dB16)` for this branch contribution.
- Subsequent A/dropout backward and base-gradient routing stay unchanged.

This is an expected autocast/autograd contract to instrument, not a claim that a CPU GEMM reproduces cuBLAS rounding. For adapter matrices stored in different layouts, transpose indexing rather than changing the meaning of the products. Do not substitute `fp16(2*G32)` or retain unrounded FP32 dB just because either looks more accurate. Do not unscale internally, alter clipping, or update parameters before GradScaler's global overflow decision.

### Independent design inference: only one partial-reduction axis

One program owns an M-token tile. It loops over N-output tiles, reusing the D16 tile for:

1. an accumulated `dS[Mtile,16]`, retained over the N loop;
2. a temporary `dB_partial[Ntile,16]`, immediately written for this M tile (canonical PEFT B layout).

Only dB then needs a deterministic across-M reduction. Cast complete dS and dB at their expected FP16 boundaries. Use ordinary current-stream launches; no cross-block semaphore, graph capture, FP16 parameter-gradient atomics, or copy of the paper's implementation. Retain the saved S and effective B values until both products finish. Frozen NF4 forward/backward remains untouched.

This one-axis choice avoids a naive two-axis design that would also spill dS partials. The proposed layout is our inference, not a paper implementation claim. Opposing D/D-transpose dot layouts can require extra register/shared-memory conversion. This and register pressure may dominate on SM75.

### Falsifiable traffic model, not a speed prediction

Assume both standalone products each read D16 once, ignoring cache and all other operands. Sharing D saves `2*M*N` logical bytes. One-axis FP32 partials add `8*ceil(M/BM)*N*r` bytes for write plus read. For divisible M their ratio is `4*r/BM`.

At M=2048, r=16, BM=128:

| Projection | One D16 read potentially removed | FP32 partial storage | Partial write+read added |
|---|---:|---:|---:|
| q, N=1536 | 6 MiB | 1.5 MiB | 3 MiB |
| v, N=256 | 1 MiB | 0.25 MiB | 0.5 MiB |

These numbers exclude B/S reads, final writes, transpose/layout movement, caches, allocator cost and launch/compile overhead. At BM<=64 the simplified partial traffic already equals or exceeds the removed read; tails worsen the bound. Even BM=128 supplies only 16 CTAs for M=2048 if one CTA owns a token tile, potentially underfilling T4. More CTAs by splitting N reintroduce dS partials. This parallelism-versus-scratch tradeoff is an explicit reason to keep the candidate behind profiling, not promise a win.

### CPU and GPU kill gates

- CPU: compare the explicit cast path with an ordinary small FP16 autograd branch; vary upstream scale, subnormals, overflow, noncontiguous inputs and r=16 tails. Reject any formulation that removes baseline finite/Inf/NaN behavior.
- CPU: simulate one-axis partial sums and report disagreements against an ordinary GEMM; deterministic does not mean bitwise equal to another reduction tree.
- T4: compare **both adjoints together**, including final reduction, all casts, scratch allocation, launches and cold compilation. Profile actual q/v backward share first. Reject if combined time is not improved, memory gate fails, or custom integration changes saved adapter/optimizer behavior.
- Full harness: same task/metrics/rewind, exact structure, numerical loss and held-out gates, own-share definition, 79 regressions. CPU cannot establish any of these GPU performance gates.

Priority: **P2 research; P3 implementation until profile**, consistent with independent jury review.

## SOUP-LORA-002 — exact saved-dropout-mask bitpacking

### Mechanism

Wrap only each real q/v adapter's existing dropout invocation in a saved-tensor pack/unpack context. If that invocation saves a bool mask, pack 8 mask bits into one uint8 byte using an original Triton codec. Leave all other saved tensors untouched and detached appropriately. Before native backward consumes a saved mask, reconstruct an identical bool tensor on the same device with the same shape. Both ordinary dropout calls and their order remain unchanged. [T-dispatch, T-mask, T-save, T-hook]

This does not generate a replacement mask, replay RNG, change dropout output dtype, compress activations numerically, offload, enable checkpointing, or share q/v masks. The saved representation is the sole change. The public hook contract supports representation substitution; the particular bitcodec is our transfer inference.

### Strict guards and lifetime constraints

- Patch only genuine q/v LoRA dropout instances, not every dropout or all saved tensors globally. Fake regression modules, eval, empty input and unsupported saved dtypes use baseline.
- Apply only if the runtime actually saves bool masks. Do not reinterpret a floating scaled mask as bool. Ordinary CPU F.dropout may save floating masks and is not the CUDA dispatch oracle.
- No mutation of the mask, model tensors or hook inputs; preserve tail bits, original shape/device/dtype, and repeated unpack behavior. Noncontiguous masks need a tested logical-index mapping or fallback.
- Do not retain a reference to the original uncompressed mask in the packed object; doing so defeats memory reduction. Preserve any surrounding saved-tensor-hook behavior or decline this optimization if nesting conflicts. Hook context is thread-local; do not change threading.
- If input does not require gradients, no dropout mask may be saved. Do not force a gradient or fabricate work. First-layer mask count must be observed, not assumed.
- Backward unpacks a temporary full mask. Count that transient, packed masks, Python hook overhead, and pack/unpack launches in the measured command. Avoid caching a full unpacked mask past its legitimate consumer lifetime.

### Bounded memory model

At M=2048 and H=1536, one bool mask is 3 MiB; bitpacked payload is 0.375 MiB. Across at most 56 q/v calls, the retained-mask **upper bound** changes from 168 to 21 MiB (147 MiB payload reduction). Some masks are not saved; allocator reuse, transient masks, dynamic T and actual peak location reduce or eliminate measured peak benefit. Even the upper bound is about 1% of the organizer's roughly 14-GiB peak, below the standalone 5% scoring tie threshold. It can only be a complementary memory refinement, not a primary score plan.

Extra pack/unpack work probably increases local operation time. A useful original codec may count only if training genuinely consumes its result and the full rules pass; do not add it merely to inflate own-kernel time. There is **no prediction of speedup or measured VRAM reduction**.

### Falsifiable CPU claim and promotion gate

With identical seed and two sequential native_dropout calls, hooks that restore masks exactly must preserve both outputs, all input gradients and final RNG state. Test sizes 1/7/8/9/257, multiple shapes, repeated backward with retained graph, no-grad input, eval, noncontiguous masks and hook nesting/fallback. Compare the same native operation in both arms; do not compare CPU native_dropout against ordinary CPU F.dropout and call that CUDA equivalence.

On T4, first record number, dtype, bytes and lifetime of saved masks. Reject if the mask is already compact, hooks conflict, original mask storage stays live, peak does not fall, or added total time violates the 10% gate. Keep native backward initially; fusing unpack with backward would be a separate, larger cast/gradient proof.

Priority: **P2 conditional memory experiment**, not ahead of head/norm work.

## Refinements, negatives, and deduplication

| Existing ID / source | Result | Why no new row |
|---|---|---|
| L6-16 / inductor_b2b_fusion + LoRAFusion | Add a split-at-rank alternative: at M2048, r16, S16 is only 64 KiB per branch. Test ordinary materialization against recomputation/synchronization/full fusion. | Same dependent-GEMM optimization space; stronger decision rule, not another primitive. |
| SOUP-NEW-003 + gemm-07 + Fast NF4 paper | Try lookup placement only inside the existing exact nested-decode experiment. The paper stages a 16-entry table, but pinned bnb already has direct table lookup, not its cited tree decoder. | Existing codebook/lookup mechanism. Current direct lookup does not prove shared staging is faster or slower; only removes a claimed novelty. |
| LoRAFusion full horizontal base-GEMM fusion | Defer. Soup base is nested NF4 with bias and FP32 return boundaries, not the paper's plain dense path. | Much larger implementation surface than the selected sibling-adjoint experiment. |
| LoRAFusion scheduling, cross-job grouping, packing, distributed runtime | Exclude direct transfer. | Single fixed adapter/task, one T4, exact batches/order; no new runtime dependency. |
| [CE-LoRA](https://arxiv.org/html/2502.01378v1#S3) | Reject direct transfer: deliberately approximate activation-gradient multiplication and add an SVD-derived frozen factorization. | This changes the gradient algorithm rather than implementing the fixed one faster. Convergence claims do not establish Soup equivalence. |
| HyC-LoRA search lead | Not promoted. The primary OpenReview PDF request returned browser verification; no attempt was made to bypass it. Only search-level text was available. | Incomplete evidence; no exact compatible method or performance conclusion asserted. |
| “Power-of-two scaling can move across half casts” | Rejected as a universal identity; subnormal rounding/overflow boundaries must be tested. | Correctness falsifier, not optimization. |
| “FP32 master weights imply FP32 GEMM parameter gradients” | Rejected without boundary proof. Autocast backward can produce rounded FP16 contributions before casting into FP32 master grad. | Preserve actual graph, not intended precision. |

## Next best question

**Does the actual Kaggle graph spend enough combined time in the q/v B-adjoints to justify sibling fusion, and does M-tile parallelism force two-axis partials on T4?** Required evidence: Torch/CUDA/Triton versions, q/v shapes/dtypes, separate dS/dB kernel durations, register/layout metadata for any future prototype, and full cold command. Without that evidence, more broad LoRA kernel searches are lower value than completing the head/norm plan.

For mask packing, the next decisive measurement is simpler: count actual saved bool-mask bytes and whether they overlap the measured peak. If not, retire it without implementing a GPU codec.
