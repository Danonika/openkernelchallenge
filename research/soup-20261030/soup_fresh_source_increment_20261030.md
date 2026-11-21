# Fresh-source scouting increment — 2026-10-30
Bounded search of recent primary papers, upstream issues and repositories. No installs, executions, copied kernels or remote edits. Two narrow compiler-control mechanisms retained; no newly verified Soup/T4 training speedup found.

## Retained incremental mechanisms
Both are **untested hypotheses** on this challenge. They alter compiler strategy, not model mathematics, and retain ordinary solution-defined @triton.jit launches.

| ID | New mechanism and source | Soup/T4 transfer and priority | Constraints / falsification |
|---|---|---|---|
| SOUP-COMP-001 | Suppress **pointer-alignment specialization only on scalar metadata arguments**, avoiding extra variants when views have different offsets. [Triton v3.8 runtime](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/python/triton/runtime/jit.py), [API](https://triton-lang.org/main/python-api/generated/triton.jit.html); recent motivating [upstream issue Aug15](https://github.com/vllm-project/vllm/issues/52413). | **Allowed method, conditional API availability. P1 if actual cold variants occur.** For row-index, label/mask or chunk metadata consumed scalarwise, experiment with do_not_specialize_on_alignment. Architecture-neutral; no vLLM import. | Preserve specialization where aligned wide data loads benefit. Do not lie with alignment hints. Measure number of unique compilations plus full cold command; reject if no extra variants existed or lost vectorization costs more. Distinct from prior runtime scalar-value specialization: this controls pointer-address alignment. |
| SOUP-COMP-002 | Selectively shorten loop live ranges using **disable_licm**, and independently test **disallow_acc_multi_buffer** on dot-loop accumulators. [Triton v3.8 language core](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/python/triton/language/core.py), [range API](https://triton-lang.org/main/python-api/generated/triton.language.range.html). | **Allowed method, conditional compiler support. P2 after spill evidence.** Relevant to vocabulary-streaming reductions or a future skinny-LoRA/NF4 dot loop when hoisting/buffering increases register occupancy or spills. Does not require Hopper instructions. | Test controls separately against default with identical math and launch tile. Extra recomputation or serialized accumulation can lose performance. Confirm actual register/spill metadata and T4 runtime; not a blanket optimization. Distinct from generic occupancy sweep: explicit compiler-live-range intervention. |

The official v3.8.0 source pin above is dated **2026-08-28**. It confirms API existence there, not that Kaggle supplies that version or that the mechanisms were invented in August. The notebook does not pin Triton. Do not upgrade/install during measured runs.

The motivating alignment issue concerns a different workload/hardware and includes a shared-cache failure allegation; neither its performance claim nor its multi-node cache issue is transferred to single-process Soup. Only the documented specialization-axis mechanism is retained.

## Fresh correctness alert, not a speed idea
[Upstream Triton #12525](https://github.com/triton-lang/triton/issues/12525), opened **2026-10-27**, reports loop_unroll_factor executing an epilogue for an empty range. The report includes positive/negative-step cases and says interpreter execution was correct while compiled GPU execution was wrong (tested upstream commit 1b4a204 on A800, not our T4).

Action: add compiled empty/tail-range cases before unrolling chunked reduction loops; keep no-unroll fallback. Do not infer the task's installed Triton is affected, or that interpreter-only success establishes GPU correctness. No compiler patch/install proposed.

## Newly inspected sources that should NOT inflate the idea catalog
| Source / date | Useful finding | Disposition |
|---|---|---|
| [dh8116/triton-kernels README, Oct3 commit](https://github.com/dh8116/triton-kernels/blob/c03502565dde5c1fcb664692f23f10898334b5d2/README.md) | Author reports T4 fp16 CE fwd+bwd 15.90ms vs 24.03ms for N2048,V131072, and explicitly reports chunked linear-CE as memory-saving but slower. | **Source-reported microbenchmark only**, not reproduced or Soup-compatible evidence. Reinforces our existing CE/head hypotheses and risk; no new ID. Linked method pages were unavailable to the web tool, so detailed methodology remains unverified. No implementation copied. |
| [Noeris README, Aug19 commit](https://github.com/0sec-labs/noeris/blob/2c0d64c64771624076096972c5c32118177688b8/README.md) | Reports T4 fusion experiments, QK norm+RoPE and search machinery. | Existing catalog already covers these ideas; Qwen2 must not gain absent QK normalization. New ready-made library is not authorized merely because absent from named blacklist; **reference only**. No import/install or fresh kernel row. |
| [FORGE, June22 preprint](https://arxiv.org/abs/2606.22932), method §2 | Consumes parameter-gradient tiles in an in-register optimizer update; exactness assumes coordinate-separable update and old-weight consumers completed. | **Reject direct transfer here.** Soup globally unscales/clips and can skip all updates on GradScaler overflow; eager per-tile updates cannot know that outcome. LoRA gradients are small compared with full-model gradients. Its low-precision/state changes are not allowed. Not a T4 training proof. |
| [Trident, Sep29 preprint](https://arxiv.org/abs/2609.37241), [Triton launch RFC Apr27](https://github.com/triton-lang/triton/issues/10140) | Lower recurring host dispatch through native guarded launch paths. | **Do not adopt alternate launch machinery.** Extra compiler/runtime unsupported; bypassing JITFunction.run would bypass the harness's own-kernel instrumentation. Useful host-overhead awareness, already covered by full-command profiling. |
| [Compiler-Grounded Hierarchical Diagnosis, Jul25](https://arxiv.org/abs/2607.23089) | Explains profiling→IR→compiler evidence chain. | Evaluation is Ascend NPU, not T4. Generic diagnosis overlaps existing L4 research-method IDs. No new T4 speed claim. |

## What evidence unlocks the next useful pass
Further broad searches are lower value than:
1. Actual torch/Triton/CUDA versions and baseline kernel-shape histogram from first T4 run.
2. Counts and durations of fresh compilations, specialization keys/argument metadata, and host-vs-device step time.
3. Register count, spill/local-memory traffic and shared-memory usage for own norm/head-reduction kernels.
4. Full harness correctness/share results, especially fp16 scaled backward and near-zero metrics.

Then select **one** narrow study: unwanted specialization explosion if compilation dominates; live-range controls if spills dominate; or no new compiler tuning if GEMM dominates. This pass stops after these findings rather than repeating searches without runtime evidence.

