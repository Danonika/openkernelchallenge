# Soup cycle — jury decisions and missing proof

**Decision:** advance original FP32 RMSNorm and complete-head metric reductions first; retain fused nested NF4 decode next. Add **three** genuinely distinct conditional hypotheses, not a new generic fusion list. Keep all original 104 IDs and all 118 existing catalog rows: **121 reviewed rows total** (39 keep, 45 conditional, 32 hold, 5 reject). **Nothing here establishes a T4 speedup, memory saving, quality pass or admission.**

Review scope: the 118-row master, original104 IDs, prior additions/NF4/head/classification/fresh-source notes, incoming independent research and original CPU witnesses. Official authority is starter **c7c2fcd15972ce8b898d8d3ff732ee192c817ceb**, read directly from the clean cloud source checkout. No GPU, package installation, remote write, external submission or judge contact by this lane.

## 1. What changed this cycle

1. **Correct the head state, not just its label.** A single rounded `LSE=max+log(sum(exp(z-max)))` is not a safe generic oracle. CPU finite-FP16 counterexamples lose a positive CE and distort entropy. Store maximum and centered log-normalizer separately; evaluate log-probabilities with the centered subtraction. The centered entropy identity still needs actual GPU rounding tests. This refines `SOUP-NEW-001/002`, not a new ID.
2. **Stage the complete head.** Validate own reductions on materialized baseline logits first. Only then remove logits, hook metrics/rewind and recompute backward. This isolates the riskiest semantic changes and leaves a viable reduction-only fallback.
3. **Add `SOUP-TRAIN-001`: supervised-row-only frozen-head backward.** Full forward entropy remains. The compact/scatter CPU fixture supports the final-head gradient-mask idea, while a causal toy explicitly disproves pruning earlier prompt backpropagation.
4. **Add `SOUP-LORA-001`: share incoming B-adjoint tiles across two sibling gradients.** Distinct from the existing dependent A→B fusion. Research retained; implementation remains behind higher-value candidates until a real profile justifies it.
5. **Add `SOUP-LORA-002`: exact bit-packed native-dropout saved masks.** Native dropout/RNG/backward remain; only the saved representation changes. Conditional own autograd technique, not a task-switch or dropout replacement.
6. **Correct two table ambiguities.** Reject direct FP8-cache conversion in `attn-fi-packed-fp8-convert`; retain only its generic vector-load lesson under applicable existing candidates. Hold generic packed SwiGLU for own-share purposes unless genuinely independent work is distinguished from the shipped example. Do not add absent Muse residual gates, QK normalization or LayerNorm mean subtraction when transferring norm tiling.

The source ideas are allowed independently of whether their package is banned. Conversely, a source citation never permits importing/copying its kernel. [Rules](https://example.invalid/redacted-github), [actual dependency list](https://example.invalid/redacted-github).

## 2. Highest-value verdicts

**Keep** means retain a method, not certify an implementation. **Conditional** means worth advancing with specified proof. **Hold** means no implementation budget until its trigger, not block the whole catalog. **Reject** addresses the named direct route, preserving original ID and provenance.

| IDs / implementation group | Verdict / priority | Next authorized research or implementation target | Exact missing proof / stop condition |
|---|---|---|---|
| `liger_castaware_norm`, `apex_rms_reduction`, `vllm_vector_norm`, `xformers_rms_cachepolicy`, `sglang_norm_boundary`, `trt_rms_shmem` | Conditional **P1** | Own one-row FP32 Qwen2 RMSNorm fwd/bwd; residual fusion only later | Actual prepared-model dtype/epsilon/frozen affine/stride contract, compiled reductions and full LoRA gradients. Return original rounded residual and route independent bypass adjoints for fusion. No direct FP16 output or added gates. CPU analytic parity is not a compiled kernel. |
| `SOUP-NEW-002` | Conditional **P0/P1** | Complete CE+entropy+accuracy+rewind reduction on materialized baseline logits | FP32 values upcast from rounded FP16 logits; causal shift/global denominator, prompt/final entropy, ties, row mapping; centered state/tile merge and actual near-zero metric errors. Do not omit “unused” metric rows. |
| `SOUP-NEW-001` | Conditional **P0/P1** | Frozen-head chunks; scaled derivative recomputed in backward | Incoming scalar applied before FP16 dlogits; actual scaler overflow/unscale/clip/optimizer/scheduler behavior; correct TRL/rewind integration and ordinary fallback. Added head recompute may lose time: retain reduction-only fallback if total-command gate fails. |
| `SOUP-TRAIN-001` — new | Conditional **P1** | Head backward only on valid **shifted** labels; full-shape zero scatter then ordinary decoder backward | Actual changed-M cuBLAS rounding/efficiency, all-ignore mean versus sum, nonfinite/scaler skip behavior. Never prune prompt decoder gradients or forward entropy. `N+2U` work model is arithmetic, not measured gain. |
| `SOUP-NEW-003` | Conditional **P1** | Own nested NF4 scale+offset+decode, ordinary matmul | Actual bnb state/codebooks/cast/orientation/bias, separate FP32 multiply then add, all tails; forward CUDA-backend fallback **and** separate functional backward dispatch. CPU synthetic scale values do not validate real QuantState. Keep native small-M fused bnb path until evidence favors replacement. |
| `SOUP-NEW-004/007` | Conditional **P2** | Cache only when repeated casts/scale reconstruction are actually costly | Existing autocast-cache behavior first; exact frozen values/invalidation; real memory lifetime and cold construction cost. Head and block-scale byte estimates are not peak savings. Initialization alone earns no steady-step share. |
| `L6-10` | Keep method; conditional LoRA application **P2** | Small scale+base-add epilogue retaining ordinary dropout/A/B GEMMs | Preserve branch FP16 multiply then FP32 add and backward cast→scale order. Do not delete defensive clone without alias proof. Actual launch/read savings unknown. |
| `SOUP-LORA-001` — new | Conditional **P2 research / P3 implementation** | M-tile owns dS accumulator over N; emits dB partials for deterministic reduction | Preserve `G32→G16→scale→D16`, `dS16`, `dB16→FP32` boundaries; original q/v dropout. Compare **both adjoints plus scratch/reduction/compile**, not saved dY bytes. Hold if no profile budget. |
| `SOUP-LORA-002` — new | Conditional **P2** | Pack/unpack only exact saved native-dropout masks, narrow scoped hooks | Actual installed Torch dispatch and saved tensor type; no retained mask/cycle/nonmask interception; nested-hook/fallback safety and exact unpack device/dtype/shape. Native RNG and backward unchanged. GPU pack/unpack plus real retained-mask lifetime/peak remain unknown. |
| `SOUP-NANO-002` | Keep **P1 methodology** | Bound real cold compilation variants and keep changing scalars runtime | Count variants/compile durations, no stale constants, prebuilt caches or alternate launches. Not its own kernel-share claim. |
| `SOUP-COMP-001/002` | Conditional **P2** | Tune alignment specialization/live ranges only after observed variants/spills | Installed API support; no false alignment; one control at a time; compiled empty/tail ranges, register/spill evidence and full-command cost. No trigger means retain defaults. |
| `SOUP-NEW-005/006`, `SOUP-NANO-001`, `L6-16`/full LoRA fusion | Hold or deferred **P3** | Exact frozen storage, whole-model unpadding, optimizer fusion, large autograd rewrite | Larger proof surface than current payoff evidence. Exact representability alone does not preserve dtype dispatch; unpadding must preserve original dropout RNG; optimizer updates cannot precede global unscale/clip/finite decision. |

These are **portfolio** choices. No individual candidate must independently reach the 5% share floor; all eligible useful original kernels count together. Faster kernels can make the floor harder, so never infer it from a baseline profiler percentage or slow kernels intentionally. [Exact collective share and intersection of launches](https://example.invalid/redacted-github).

## 3. Explicit variant rejections and holds

- **Reject unsafe numerical variants:** precompute unscaled FP16 head gradients then scale later; contract nested scale multiply+offset into single-rounded FMA without exact proof; directly accumulate unrounded FP32 LoRA parameter gradients in place of the baseline FP16 boundary. CPU witnesses establish non-equivalence—not a prediction of real-model failure magnitude.
- **Reject semantic shortcuts:** dropping prompt/final entropy, per-chunk CE denominators, pruning transformer prompt backprop, shared q/v dropout masks, altered AdamW/Muon/FP8/model task, or eager per-tile optimizer updates before global scaler/clip checks.
- **Reject direct graph/prebuilt-cache and ready-made kernel routes.** `L6-03/07`, `SOUP-NANO-X01/X02` remain explicit exclusions. An HF wrapper or new package name does not turn ready-made FlashAttention-derived kernels into owned Triton. Imported-library/hardware coverage and idea legality are separate axes.
- **Hold background compilation (`SOUP-COMP-X01`).** Background processes are prohibited; same-process threading is not established permitted by a competitor README. Current cycle does not contact judges or execute that route.
- **Hold architecture-incompatible reference mechanisms**, rather than label their abstract scheduling ideas illegal. Direct TMA/WGMMA/cp.async source paths are not SM75 implementations. Native CUDA/CUTLASS/TVM runtime absence from a blacklist is not positive package authorization.
- **No own-share credit for starter or source copies.** Merely falling below an automated similarity threshold is not ownership proof. [Example rule](https://example.invalid/redacted-github), [source checks](https://example.invalid/redacted-github).

## 4. CPU evidence — narrow claims only

**8 original CPU fixture groups passed; 0 unexpected failures.** Several deliberately unsafe variants failed equivalence as intended. Final CPU evidence is referenced by test-group name and artifact hash in `jury_verdicts.json`; see `cpu_tests/results.json` and `cpu_results.md`. Tests use original synthetic tensors on cloud CPU Torch, not challenge/model data and not Triton kernels. Passes do not establish real bnb state compatibility, CUDA RNG, CUDA GradScaler, quality, T4 time/memory or own-share.

Already inspected witnesses:

| Evidence group | Established | Still missing |
|---|---|---|
| Head causal/global metric fixtures | 24 chunk/length cases, exact tested shifted counts/mapping; prompt-only entropy and local denominator errors exposed | V=151936, actual TRL integration, GPU reduction/order and runtime |
| Centered reduction | `[32768,32752]` baseline CE `1.192092824e-7` becomes `0` via single-LSE subtraction; centered form preserves value | Real model logit distribution and compiled centered/two-pass comparison; no claim that model emits adversarial inputs |
| Scaled gradient | Scale 65536 yields nonzero reference hidden gradient `-0.0027008057`; scale-after-FP16 variant yields 0 | Genuine CUDA scaler overflow/skips and complete training |
| Supervised head backward | 15 finite CPU mask/scale cases match dense gradient exactly; causal toy shows earlier prompt gradient remains nonzero | GPU shape-dependent GEMM rounding, nonfinite behavior, full decoder |
| Norm/bypass | Widths including 1536, strided rows, magnitude cases and independent residual bypass checked; wrong cast order exposed | SM75 reduction/resource behavior and full gradients |
| NF4 synthetic transform | All nibble IDs/synthetic scale-code indices, odd tail and block boundaries; separate/FMA rounding counterexample | **Actual** bnb codebook/QuantState/dispatch; no “NF4 validated” claim |
| LoRA adjoint boundary | Deterministic no-dropout forward/backward boundary fixture; subnormal and overflow witnesses | Native q/v CUDA dropout, full autograd and fused kernel |
| Native-dropout saved-mask codec | 10 paired train/eval native-dropout cases had exact tested output/gradient/RNG equality; strided mask roundtrip exact. Debug clones were retained: no memory result | Actual CUDA dispatch, kernel implementation, saved-mask coverage, measured peak/time |

## 4a. Cycle 2 changes the proof obligations, not the idea count

**Six additional CPU fixture groups passed as evidence tests**, including deliberate counterexamples; they did not pass challenge numerical gates. Separate immutable artifacts: `cpu_tests/cycle2/results.json`, `cpu_cycle2_results.md`, and `research_training_increment2.md`. Their hash and test names are bound to the existing head rows.

- **Unconditional sparse zero-scatter is disproved.** Ignored NaN/+Inf/all−Inf logits can yield finite summed CE yet NaN dlogits; a compact path returning zero hides those gradients. Nonfinite effective FP16 weights can also turn zero-row matmul into NaNs. Finite operands can overflow actual FP16 head outputs.
- **A generic derivative rewrite is also disproved:** with infinite effective upstream, baseline NLL→log-softmax target gradient can be NaN while `(p−onehot)*scale` yields −Inf. This affects analytic head backward generally, not only compaction.
- **Retain a finite-domain opportunity**, subject to finite actual rounded logits, effective FP16 weight, incoming/effectively normalized scalar and positive finite denominator, plus a reference-ordered exceptional fallback. Retain full final hidden or sufficient exceptional data until routing is settled. No clamp/nan_to_num or direct scaler manipulation. Guard synchronization and fallback work/memory count.
- **Centering alone does not clear relative-error gates.** Of 900 finite tiled cases, one sequential singleton merge gave CE `3.051711246e-5` versus baseline `2.884823152e-5`: about **5.785% relative error**, despite passing exploratory `atol=3e-6`. Reverse/balanced orders matched this witness only. Require near-zero relative CE/rewind tests and actual GPU reduction-order evidence; do not claim this is a measured whole-run failure or that balanced merge always solves it.
- Empty-tile identity must avoid `0 × −Inf`; argmax ties select lowest **global** vocabulary index, not traversal-left. Causal selection begins one position before an assistant label suffix, because the last prompt hidden row predicts its first token. Actual public U/N statistics remain unavailable; no histogram or crossover gain was fabricated.

The head candidates stay **conditional**, not globally rejected or promoted. The finite positive witness and exceptional negative witnesses answer different questions.

## 4b. Final LoRA increment: proof improved, implementation priority unchanged

**Three further CPU fixture groups passed**, bound to `SOUP-LORA-001/002` by `cpu_tests/cycle2b/results.json` hash; see `cpu_cycle2b_results.md`. Earlier 8-group and 6-group artifacts remain immutable. **17 fixture groups total is not 17 challenge passes.**

- A toy **B+scale+base-add custom-autograd boundary** preserved six q/v train/eval normal/subnormal/overflow cases, including exact tested outputs/RNG, zero finite gradient error and zero nonfinite-class mismatch. Three CPU autocast F.linear crosschecks support the tested cast contract. A/dropout and defensive base clones remain outside/unchanged. The custom function still executes ordinary Torch operations: no kernel fusion, real bnb/PEFT route or T4 benefit was established.
- Retain **D16-only versus wider composite ownership as two variants under SOUP-LORA-001**, not new IDs. The wider route could avoid observed G16/D16 materializations if the actual graph has them; logical traffic arithmetic omits caches/layout/other operands. Both sibling products plus deterministic partial reductions and compile costs still need measurement. Implementation priority remains P3 until profile.
- A **per-saved-native-mask hook** preserved five tail lengths, repeated retained-graph backward, neighboring saved tensors and outer-hook behavior. It is more selective than a broad hook context but uses a private version-specific `_raw_saved_result1` API. Actual q/v CUDA node selection must be proven or fall back. Only demonstrated pre-mutation registration conflicts may be skipped; failures inside pack callbacks must propagate, not be silently treated as an unsupported case.
- Exact CPU codec behavior and absence of debug mask clones in this increment still do not demonstrate storage release, sampled peak savings or acceptable added time. No priority promotion from payload arithmetic.

No further CPU cycle is required for this bounded review. The next decisive evidence is an **authorized actual T4 boundary/profile/harness run**, not another broad audit or unconditional candidate rollout.

## 5. What qualifies a later promotion

Use a commit-bound baseline/candidate comparison from the unchanged harness. Exact structural outputs are not covered by a broad “3% tolerance”: steps, adapter structure, trainer settings, metric names, token and rewind row/count identities stay exact. Numeric curves use **mean per-element relative absolute gap**, not relative gap of means, and near-zero references matter. [Exact comparisons](https://example.invalid/redacted-github).

The pinned harness uses **median full-command time and maximum sampled memory**, plus first successful adapter evaluation—not an invented average of memory/eval results. The attached ТЗ describes averaging repeats: with two runs median time equals mean time, but maximum memory does not equal mean memory. Keep this discrepancy visible, preserve raw runs, and label engineering comparisons by the pinned harness; no official-score or leaderboard claim is made here. Do not edit the harness or contact judges in this task. [Aggregation](https://example.invalid/redacted-github), [eval selection](https://example.invalid/redacted-github), [resource checks](https://example.invalid/redacted-github).

**Missing evidence, not a research stop:** actual Kaggle T4 versions and prepared-model boundaries; baseline shape/profile histogram; own-kernel compiled fixtures; unchanged complete cold runs with all checks and plain-PEFT saved-adapter quality. Continue the legal independent candidates in their stated stages; do not require every speculative catalog row to prove itself first.

## 6. Publisher handoff

- `jury_verdicts.json`: one verdict per preserved ID plus three new IDs; separate legality, runtime dependency, reuse, hardware, semantic risk, CPU evidence, GPU unknowns, priority, exact missing proof, next action, stop condition and pinned rule citations. All measured T4 gains are `null`.
- `jury_verdicts.csv`: same rows, arrays encoded as JSON; do not flatten CPU PASS into full-kernel “validated.”
- `jury_build.py`, `jury_incoming_build.py`, `jury_incoming.json`, `jury_cpu_merge.py`, `jury_cpu_updates.json`: reproducible local assembly inputs; no solution/harness edits.
- Incoming provenance: `research_training.md` / `research_training_rows.json`, `research_lora.md` / associated rows when finalized. Preserve citations and earlier rejected routes; map related IDs to one implementation group rather than count duplicate methods as independent implementations.

The exact-five-category requested regrouping is in `category_mapping.json` (118 existing IDs) and `category_mapping_incoming.json` (3 incoming IDs); it is administrative only and never overwrites verdicts.

Only the publisher lane may make the separately authorized table/GitHub updates. This jury lane makes no remote changes.
