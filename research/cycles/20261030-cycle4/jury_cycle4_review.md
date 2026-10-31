# Cycle4 jury increment — five research refinements and first measured-trial selection

## Decisions

- **Conditional go to trial001:** `liger_castaware_norm`, original guarded standalone FP32/frozen-affine width1536 norm. No residual fusion or starter activation. All matching actual Qwen norms are eligible; full-width realistic scaled-gradient/expanded-bypass GPU preflight precedes one frozen-harness run. This is a useful bounded experiment, not proof of universal identity, a gain, or admission.
- **Keep conditional, same priority:** `SOUP-NEW-001`, `SOUP-NEW-002`, `SOUP-TRAIN-001`. A dense-forward/native-metric CPU oracle establishes guard, mask and lifecycle contracts, not streamed-forward, analytic-gradient or GPU equivalence.
- **Keep conditional, same priority:** `SOUP-NEW-003`, `SOUP-NEW-007`. Actual bitsandbytes CPU public-API metadata coverage closes the former absent-runtime gap, but a genuine native CPU parity failure remains. No CUDA or performance conclusion follows.
- **No new IDs:** 121 catalog rows; original104 and five categories preserved. No candidate gain or graph point is added. Previous cycle3 immutable snapshot remains authoritative for its cutoff.

## Useful findings and exact remaining proof

### Head/loss/metrics

Four CPU evidence groups passed: 25 finite native-autograd compact-backward cases, 18 full-original-geometry exceptional fallbacks, metric discriminators and a synthetic once-only packet lifecycle. Actual finite tested input/head strides are contiguous. Dense forward and the native metric helper are intentionally shared between paths, so matching these outputs is not independent optimized-metric arithmetic validation.

**Contract correction:** main causal CE evaluates all B*T rows including final ignored positions. Entropy evaluates all B*T rows and then multiplies the attention mask, preserving 0*NaN contamination and explicit exp(logp)*logp NaNs at negative infinity. Accuracy and rewind instead consume T-1 positions. Do not unify these into one selected-row mask.

**Compact backward:** finite effective FP16 weight, every actual rounded FP16 forward logit, incoming scalar, actual normalized upstream and positive finite denominator are guards, not a proof of GEMM-rounding equivalence. Retain full unmodified hidden state for a full-geometry native fallback when any guard fails. A final ignored NaN row can affect native gradients despite finite scalar and rewind losses. Decoder prompt backprop remains untouched.

**Arithmetic gate:** native summed CE followed by division normalizes incoming gradient before NLL and log-softmax. Even finite `(p-onehot)*(g/D)` is not automatically the same rounded target-subtraction order. Current tests deliberately use native autograd; they do not validate that analytic replacement. Prior single-LSE loss and approximately5.785% near-certain relative CE merge witnesses remain open.

**Next proof:** a later bounded actual-T4 full-versus-gathered head output/gradient comparison; separately original reductions on materialized logits at V151936 and real HF/TRL/Soup packet integration. Include guard synchronization, fallback allocation and cold-command costs. Head work is not part of norm trial001.

### NF4 decoder and scale cache

Parent-authorized official bitsandbytes0.50.2 installation changed only that package in the isolated CPU environment; Torch remains2.15.0+cpu. Eight inspected installed Python files match pinned source blobs; wheel/native-library hashes are separate evidence, not CUDA-build identity. Optional hub kernels were absent and not enabled.

**Three groups passed; one genuine native CPU parity group failed.** Independent scalar/vector/public-API comparisons cover all16 NF4 x all256 supplied nested indices with actual default and synthetic tables, FP16/FP32, tails and layouts. The primary exhaustive odd-K case uses generic4bit CPU fallback; a separate aligned[128,128] FP16/default-table native-eligible exhaustive control is exact. Six actual tiny quantizer-returned states also match. These are not model weights or GPU quantizer proof.

**Retained FAIL:** native CPU[3,80],block64,all0xFF,directscales[1,2,3,4] differs from the independent flat format in48/240 values,maxabs1. Same payload/scales represented as[1,240] are exact, as are aligned/other selected controls. Source row-local scale indexing explains the limitation, but no instruction-level profiler proves a particular binary branch. Do not relabel this all-pass, adapt the flat oracle to the bug, infer actual aligned-Qwen failure, or infer CUDA failure.

**Contract:** state2.code is the supplied FP32 dynamic256 table, not a ramp/NF4 table; outer code metadata does not replace the canonical NF4 selector. First-level blocks index flat row-major elements. Preserve separate FP32 multiply and offset addition, nibble/tail layout and literal BC shim. CUDA forward fallback requests A.dtype directly; backward functional dequantization first emits state.dtype, then converts to gradient dtype. Both shortcut differences have CPU witnesses.

**Next proof:** observe actual T4 forward fallback versus existing small-M fused routes and separate backward interception; compare decoded weight and scaled input/LoRA gradients with caller-specific casts. Scale caching remains P2 only after inline decode cost is observed; include metadata/lifetime invalidation, construction and allocation. No full dense-weight cache or mutable QuantState shortcut.

### Trial001 norm

The original source was reviewed at12:03UTC with no source-level blocker to conditional measurement. The formal contract is `jury_trial001_contract.md`, SHA256 `bc092444d4d6b94f17e8c857f1ec76ff0e7b7bc278da98ff8747b26b3baf9a83`.

Fixed ordinary-case engineering preflight: forward allclose rtol1e-4/atol1e-5; gradients normalized only for test diagnostics by the known upstream scale rtol1e-3/atol2e-5. Record raw errors and finite/nonfinite classes. These are not challenge thresholds or a training change. Retain actual-width cancellation native0/atomic1 and overflow witnesses; universal identity is disproved, but the contest's tolerances—not a newly invented universal-bitwise requirement—decide full-run acceptance. Ordinary-case failure stops the trial without silently widening thresholds.

No norm GPU result exists at this review cutoff. Actual patched routing, compiled reduction/rsqrt/pow3 behavior, 79 regressions, 64-step metrics/rewind/adapter/quality, time/memory and useful collective own-share remain gates. A solution-only cross-session comparison with verified baseline is diagnostic, not official paired/repeated score or admission.

## Source and legality boundaries

All concepts require independently written allowed Triton/autograd code; source attribution does not authorize copied kernels or prohibited libraries. Baseline dependency invocation as an oracle earns no own share. Rules: [unchanged task and prohibited routes](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md#L147-L170), [numerical/resource constants](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L29-L44), [structure and curves](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L151-L236).

Pinned mechanism citations and distinct legality/hardware/semantic/CPU/GPU fields are retained in all six canonical replacement rows. Source reports: `research_head_cycle4.md`, `research_nf4_cycle4.md`; CPU files: `cpu_tests/cycle4/head_results.json`, `nf4_results.json`. CPU evidence totals become31 groups: prior23 unchanged plus8new, of which7pass and1explicit native-backend parity FAIL. These are evidence groups, not official challenge checks.

This review approves local content. External disclosure is controlled separately by parent/user authorization; previous denial must not be evaded by alternate bundles. Do not publish private notebook/account/quota identifiers, credentials, weights or unapproved raw logs. Three visible tabs remain; Fusion Ideas stays blank, Soup Runs contains genuine measured runs only, and graph remains baseline-only pending actual reviewed candidate results and parent instruction.
