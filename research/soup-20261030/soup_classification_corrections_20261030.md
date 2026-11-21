# Soup classification audit: corrections before publication
Audit time: 2026-10-30 10:53 UTC. One read of the target research branch returned **No commit found for ref research**; no repeated polling. Audited original104 catalog, first-pass additions and focused notes against exact official harness instead. No remote writes.

Authority: [README rules](https://example.invalid/redacted-github), [common.py](https://example.invalid/redacted-github), [check_rules.py](https://example.invalid/redacted-github), [_train.py](https://example.invalid/redacted-github), [run_task.py](https://example.invalid/redacted-github), all at official commit c7c2fcd (2026-10-28).

## Concrete changes to classifications
| Severity | Affected rows / claim | Correct replacement |
|---|---|---|
| Critical | Treating the old104 catalog's previous challenge prerequisites as Soup authority | Mark original prereq/test text **legacy context, not submission instructions**. In particular L1-CUDAPY-01's native-CUDA/load_inline route, L1-TVM-01's outside-scored tuning, vllm_chunked_prefill's SGLang serving patch and L6-03's graph replay are not permissions for Soup. |
| Critical | Banned project = banned mathematical idea, or cited project = allowed dependency | Separate **source provenance** from **implementation legality**. liger_castaware_norm, apex_rms_reduction, xformers_rms_cachepolicy and vllm_vector_norm may inspire independently written Triton. Importing their libraries or copying kernels is prohibited. A citation does not legalize copying. |
| Critical | 'Dead branch/test-only import does not run, so okay' | Source check scans every Python file under solution/ recursively using AST imports. Prohibited imports in tests or an if False branch still fail. Runtime imported prohibited libraries also fail. torchao exception is only incidental runtime loading by PEFT; importing torchao from solution/ remains prohibited. Research prose/links outside solution/ are not imports. |
| High | 'Not on banned list, so any extra package is allowed' | Do not turn absence from BANNED_MODULES into positive authorization. Rules explicitly permit torch/triton/transformers/peft/trl/bitsandbytes/soup; own submission kernels must be Triton. CUDA Python, TVM, CUTLASS, Numba, CuTe runtime etc are **reference-only / clarify before runtime dependency**, not approved Soup kernel routes. |
| High | TMA/WGMMA/FP8 marked simply 'prohibited' | Distinguish **hardware-incompatible** from **rule-prohibited**. Hopper TMA/WGMMA and Ampere cp.async cannot be directly used on T4; their abstract scheduling/reuse ideas may be legal to reimplement. Changing required fp16/NF4 compute/model semantics is separately prohibited. Apply to L1-CUTLASS-01, L1-TRITON-03, gemm-01/03 and architecture-specific attention rows. |
| Critical | Graph replay / cached compile / outside-timed setup as a speed solution | Official instructions require normal launches, not CUDA graphs. Every train run deletes its run directory and uses fresh Triton/Inductor cache locations; command wall time includes compile/autotune/setup. Offline experimental config selection is okay as research; prebuilt compiled caches or moving actual run work outside timing are not. Update L6-03/07, L1-TVM-01, L6-15 and L4-SCALINGINTELLIGENCE-KERNELBENCH-03 accordingly. |
| Critical | 'Keep same tokens' allows packing/reordering or masking changes | Preserve exact batches/order/steps, row identities, labels, positions, causal isolation, LoRA dropout and RNG behavior. Own within-batch pad removal is allowed; enabling Soup packing/padding_free/fast_lora/gradient_checkpointing/offloading switches is not. Matching cumulative token counts alone is insufficient; rewind rows and token counts are checked. |
| Critical | '5% GPU time' or 'one kernel launched' proves admission | Own share = eligible CUDA-event kernel ms / **summed wall-clock step seconds** from step3+, minimum across solution runs. Eligible names must be launched from solution/ in **every** solution run. Starter/renamed/close-copied kernels and Inductor-generated kernels do not count. Baseline NF4's approximate 5% GPU fraction cannot establish this gate. No intentional slowing or busywork. |
| High | 'Loss mean differs <3%' interpreted as relative error of means | Harness computes **mean of per-step relative absolute errors**, not relative error of averaged loss. Same for entropy/accuracy; rewind flattens all row-loss entries. Relative denominator is max(abs(base),1e-12), so near-zero metrics are strict. Require finite, same-length series. First loss has separate 1% gate; held-out loss is directional <=0.5% worse. |
| High | All same-task fields can differ within 3% | Structural fields are exact: steps, adapter config (except harness-declared volatile keys), adapter tensor names/shapes, trainer settings except class, logged metric-name set, cumulative tokens, rewind row identities/counts. Only specified numeric series have tolerance. A subclass name may differ; its behavior/settings may not. |
| High | 'Use Soup existing chunk-head feature' as approved shortcut | Its optional configuration requires layer streaming and would change task switches. Use its logits-free metric/rewind **contract** only in a narrow independent patch; do not enable the feature. Preserve entropy over attention mask, not just assistant CE tokens. |
| High | Norm/head precision inferred only from local helper | Prior notes corrected: Accelerate wraps output conversion to FP32, so trainer-visible logits differ from raw fp16 head output. Direct fp16 norm output also changes PEFT dropout and bnb return dtype. Mark precision-elision paths **conditional until boundary fixture and full gates**, not 'lossless by definition'. |
| Medium | 'Time and memory within 10%' on one favorable run | Gate uses median whole-command time and maximum memory across relevant runs. Memory is externally sampled nvidia-smi over GPUs; allocator is fixed, one T4 visible, idle start and >=97% of PyTorch reserved peak consistency required. No allocator hacks, second GPU or background process. |
| Medium | Competitor claims / paper results / generic hardware speedups described as measured Soup gains | Keep **source claim**, **derived estimate**, **hypothesis**, **our T4 measurement** separate. Competitor tables inspected were placeholder/TBD. Tensor byte calculations are not peak-memory results; bnb inference speedups are not Soup training gains. |

## Full current prohibited-module list
unsloth, unsloth_zoo, liger_kernel, xformers, flash_attn, flashinfer, apex, torchao, bitblas, gemlite, hqq, cut_cross_entropy, deepspeed, vllm, awq, auto_gptq, gptqmodel, exllamav2, marlin, quack, tilelang. Exact official common.py remains authoritative. Hyphenated package names and import names differ; document both accurately rather than checking only pip names. No installation during measured runs.

## Minimal table schema to prevent conflation
For each row retain original ID, then separate:
1. **idea_transfer**: allowed method / conditional semantic proof / prohibited task change.
2. **runtime_dependency**: allowed baseline / prohibited library / reference-only-unapproved.
3. **hardware_fit**: T4-ready concept / requires SM75 rewrite / incompatible mechanism.
4. **evidence**: official source inspected / hypothesis / test-pass artifact / measured full T4 run.
5. **own_share_eligible**: no (reference/library/Inductor/example/setup-only), or conditional (own useful solution Triton launched every run).

This prevents the false dichotomy 'allowed => implement now' or 'banned source => discard useful idea'.

## Near-term preflight, not new implementation work
- Apply the corrections above to legacy rows and keep original IDs; don't silently erase incompatible source context.
- Keep citations and research notes out of executable solution imports; all source reuse remains idea-level.
- Before claiming any row 'validated', require the exact harness result/check artifacts and versions tied to a commit. Passing static checks does not supersede manual rule review.
- Final evidence must use the unchanged jury harness and saved ordinary PEFT adapter. Hidden-seed/row validity is not established by fitting public inputs.

No further branch checks were made. This bounded audit is complete; no new idea rows were added.

