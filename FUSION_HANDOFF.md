# Soup Fusion: merged kernel candidate and reproducible T4 search

**30 October 2026 — Asia/Almaty**

## Outcome and current status

The five research streams have been combined into an executable `solution.apply()` package, with a native-head loss path, original Triton reductions, normalization, NF4 decoding, native-mask LoRA gradient fusion, and compact MLP activation storage. This is a new integration of the recovered work, not only another research plan.

**The candidate has not been run on a physical T4. Its runtime and peak memory are unknown. It has not established a sub-33-second result.** The current executor exposes no CUDA GPU. The supplied 77-second baseline and 33-second record are user-reported comparison targets; their original result files were not available.

### Service preparation update — 31 October 2026, UTC+5

The user authorized the Soup T4 testing service. A dedicated offline notebook, `notebooks/Soup_Fusion_Service.ipynb`, now installs the exact fused source into `SOUP_REPO/solution`, runs the selected physical readiness checks, and preserves all three original service-template cells. An added synchronous failure hook retains small training reports if a cell fails before the original export cell. The source remains `dfe2b63ee35ff1e93dd2fa109a7eed84b0ac0f45cde50a0b4c4e9f89c8aee8e2`; no arithmetic or optimization setting changed during service preparation.

The exact template and unchanged API client were recovered from the private Duo team repository's `testing-queue` branch at `3a4953f76288969bb91956f114eda5b7e541f104`. The notebook's source installation and failure-report function passed isolated filesystem checks; its code cells compile, all 34 solution files match, all 15 protected files remain unchanged, and `apply()` has no arguments. These new checks are not physical GPU execution.

Submission is blocked by this workspace's HTTPS proxy: it presents a certificate without the GPU IP in its Subject Alternative Names. Both verified HTTP clients and the cloud browser rejected the connection. The token was not sent, no job was created, and no runtime/result was obtained. The recovered service source documents only the IP endpoint, so an alternate hostname is not established. See `docs/SERVICE_RUN.md` and `validation/service_preparation.json` for the prepared workflow and exact limitations.

The source service documentation describes a saved public development baseline of 71.85 seconds; the running service must verify the active reference. This differs from the user's 77-second planning reference and from a final paired Kaggle measurement. No speedup or admission is inferred from either reference.

Executed checks on the merged code:

| Check | Observed result | Scope |
|---|---:|---|
| Unchanged official Soup regression suite with `solution.apply()` installed | 79 passed | CPU compatibility, all five official test files |
| Candidate numerical/integration suite | 250 passed; 63 skipped | CPU/Triton interpreter; CUDA-only cases remain unexecuted |
| Independent result verifier | 32 passed | Synthetic evidence fixtures; no benchmark measurements |
| Actual Soup factory, cached head and LoRA fusion | Four optimizer updates passed | Exact tested gradients, parameter updates, RNG and counts on a CPU fixture |
| Offline SM75 compilation | Existing 34-case compiler check plus new head/dropout/mask checks passed | Compilation only; compiled caches are excluded from delivery |
| Physical full-fusion NF4/AMP test | Not run | Required by the notebook before benchmark acceptance |
| Official cold-run runtime, sampled peak, public adapter evaluation and ≥5% own-kernel share | Not run | Required before a target-met claim |

The CPU factory fixture uses a local two-layer Qwen model, real PEFT rank-16 q/v adapters, real HF/TRL training, and Soup's actual Rewind factory/writer. It explicitly substitutes CPU FP16 autocast and native CPU bool-mask dropout on both sides and overrides device admission inside the test. It is not a full Soup CLI or NF4 GPU run. All substitutions are recorded in `validation/soup_factory_fusion.json`.

## 1. Target and performance budget

The pinned challenge targets Soup 0.77.0, Qwen2.5-1.5B-Instruct, nested NF4 weights, rank-16/alpha-32 q/v LoRA with dropout 0.05, batch 4, up to 512 tokens, 256 instruction rows and 64 steps on one visible Kaggle T4 using FP16 computation. The official starter snapshot is `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. The copied `harness/`, `task/`, original `tests/`, starter SwiGLU, license and notice remain unchanged. [C1–C4]

For the requested numbers:

\[
S > 77/33 = 2.3333,\qquad \text{time reduction}>1-33/77=57.14\%.
\]

These are target calculations, not measured gains. The official timer includes command startup, solution application, loading, fresh compilation, training and final saving. Download warm-up is outside that timer. The whole-command baseline must be compared with the whole-command candidate. [C2]

If unchanged overhead is \(O\), the necessary training speedup is greater than \((77-O)/(33-O)\). For example, **hypothetical** unchanged overhead of 10 or 20 seconds would require training speedups greater than approximately 2.91× or 4.38×. No overhead value has been measured here. This is why standalone kernel improvements cannot establish the requested total runtime.

## 2. What was fused from each research stream

| Research input | Decision incorporated into the implementation |
|---|---|
| Qwen-inspired optimization report | Keep native GEMMs; reduce pointwise passes and casts; use original RMSNorm and nested NF4 fusion; account for cold compilation. |
| Qwen kernel review | Implement the conservative full-native-head/compact-loss-state direction before relying on head replay. Preserve GradScaler ordering and all training consumers. |
| Nanochat/ngram transfer | Use lossless storage changes for half-origin MLP values. Follow the audited eager RMSNorm backward sequence. Do not add learned n-gram features or change the model. |
| LoRA research, Round 7 | Keep native dropout and A/B/base GEMMs. Reuse the actual saved bool mask in a fused input-gradient epilogue. Preserve the first-layer no-input-gradient case. |
| Existing build/benchmark overlay | Retain the streamed head as an alternate; reuse original reducers, NF4, optional rotary and scale/add implementations, source checks and GPU tests. |

The earlier reports describe several mutually competing choices. They are not all enabled simultaneously. The default prioritizes one coherent arithmetic and ownership contract; the notebook measures a short sequence of fixed alternatives.

## 3. Default candidate

### Native output head with private FP16 logits

`solution/cached_head.py` and `solution/integration.py` keep Qwen's original full-shape `lm_head` forward and its original matrix-multiply backward. The implementation temporarily replaces the model's per-call loss slot, runs the original forward exactly once, then restores the slot even if the call raises.

Original Triton tile reductions read the native FP16 logits and produce genuine causal per-token loss, entropy and argmax summaries. The logits remain private to backward. A newly constructed `ModelOutput` omits the large logits tensor before Accelerate performs its outward FP32 conversion. Normal inference and unsupported calls retain the original behavior.

Backward reconstructs native PyTorch cross-entropy in bounded chunks of 128 rows by default. It applies the actual incoming loss scale before the native FP32-to-FP16 gradient conversion, writes a complete FP16 logit-gradient tensor, and lets the untouched full-shape head backward compute the hidden gradient. **The default does not replay the head projection and does not use the research's sensitive custom scaled-softmax gradient.**

The implementation retains native treatment of ignored and nonfinite rows. Saturated or nonfinite forward rows use bounded native statistics on the same retained logits; the decoder and dropout are never retried. Discovering exceptional rows introduces a real synchronization. Native per-chunk backward also has CPU dispatch and launch costs. The 128/256-row comparison measures that tradeoff.

### Fused RMSNorm with reference arithmetic boundaries

`solution/rmsnorm.py` performs the FP32 row reduction, normalization and frozen-scale multiplication with one forward kernel and one backward kernel. It preserves the reference cast between normalization and multiplication by the FP32 scale.

The previous candidate used a factored backward formula. This merge replaces it with the staged eager derivative found in the latest Nanochat report:

```text
g        = rounded weighted upstream gradient
direct   = g * rstd
dr       = row_sum(g * x)
r3       = (rstd * rstd) * rstd
dv       = (dr * -0.5) * r3
dpow     = dv * rounded_reciprocal(width)
indirect = dpow * (2 * x)
dx       = direct + indirect
```

Compiler contraction is disabled for this path. Reduction trees and device reciprocal-square-root behavior still require tolerance-based CUDA validation; this is not a claim of bitwise equality across hardware. The additional checks compare against native Qwen normalization with upstream scales 1, 1,000 and 65,536.

### Compact MLP activation ownership

`solution/activation_storage.py` keeps frozen gate/up projection outputs in the FP16 representation they already have internally. The activation computation and output remain FP32. Each projection's gradient is rounded at the original boundary.

The gate and up projections deliberately receive **separate** FP16 input-cast nodes. Sharing one node would sum the two gradient branches in FP16 before converting to FP32, changing the reference arithmetic. The numerical checks retain this counterexample.

This storage optimization uses the unchanged, credited starter SwiGLU kernels. It contributes no original-kernel time for the five-percent gate.

### Native-mask LoRA input-gradient fusion

`solution/lora_dropout.py` leaves native dropout and all native matrix multiplies in place. Inside a scoped PEFT call, it reads PyTorch's actual saved bool mask, bridges the already-existing FP32-to-FP16 input conversion, and fuses the reverse cast with dropout-mask multiplication.

Dropout runs once and consumes the same RNG sequence. Unknown saved-mask layouts retain the native graph. Inputs that do not require gradients retain their native path; the first q/v input branches therefore do not create artificial kernel work. Noncontiguous, conjugate/negative-view and higher-order gradient cases have explicit fallbacks.

This path is guarded to the reviewed Torch/PEFT versions and CUDA SM75/FP16. It is not enabled by detecting a seed, test, dataset row or benchmark process.

### Nested NF4 decoding

`solution/nf4.py` combines nested-scale decoding, offset addition and packed NF4 decoding. It covers both the public dequantization entry used in backward and the existing CUDA fallback used by some forward shapes. It preserves bitsandbytes' own decision to use its fused GEMM for other shapes.

High/low nibble ordering, nested scale indexing, transpose convention, output dtype, tail stores, alignment and the separate FP32 multiply/add rounding are retained. Only the tiny format table is cached; dense model weights are not globally retained.

## 4. Fixed profiles and selection policy

| Profile | Changes relative to the default | Purpose |
|---|---|---|
| `fused` | Cached head, CE chunk 128; NF4, RMSNorm, compact MLP and LoRA mask epilogue on | Initial integrated candidate |
| `cached256` | CE chunk 256 | Fewer native CE launches versus more temporary memory |
| `epilogues` | Chunk 256 plus original LoRA scale/add and rotary fusion | Test additional cast/pointwise savings against compilation and numerical costs |
| `streamed` | Original streamed head with native GEMMs, chunk 256 | Test lower retained-logit storage against recomputation and synchronization costs |
| `shared_mask` | Chunk 256 plus per-forward additive mask reuse | Optional attention experiment; not in the default search ladder |
| `frozen_storage` | Lossless captured tied-weight storage | Optional loading/storage experiment; not in the default ladder |

`tools/select_profile.py` writes ordinary source constants **before** a fresh process. Nothing selects a profile during training. The actual model, dataset, optimizer, batches, steps and Soup switches remain fixed.

`tools/run_search.py` creates a separate local source snapshot for each profile, selects constants, makes a local commit, and runs native GPU readiness followed by the unchanged official protocol. It excludes existing data so the pinned preparation script regenerates the canonical public rows. It excludes previous results and compiler caches.

The search tests the first four profiles in order. It stops at the first profile that both passes the automatic public gates and records a median strictly below 33 seconds. With `--continue-after-target`, it evaluates every requested profile and retains the fastest verified admissible result. If none reaches the target, it retains the fastest admissible result and explicitly reports that the target was not met. If none qualifies, it reports no admitted candidate.

## 5. What the memory numbers mean

At the maximum shape, \(N=4\times512=2048\), vocabulary \(V=151936\), intermediate width \(I=8960\), and 28 layers:

| Allocation | Calculated size | Interpretation |
|---|---:|---|
| Native retained FP16 logits, \(NV\times2\) | 593.5 MiB | Still present in the cached candidate |
| Full FP16 logit gradient | 593.5 MiB | Produced in backward; may coexist with logits |
| One full FP32 logit-sized buffer | 1,187 MiB | Large eager/output buffers are the removal target |
| One FP32 CE workspace for 128 rows | 74.1875 MiB | More than one workspace can coexist during native CE |
| One FP16 gradient chunk for 128 rows | 37.09375 MiB | Additional temporary, not included in the two full buffers |
| FP16 head compute weight | 445.125 MiB | Existing native head-backward ownership remains |
| Gate/up storage reduction across 28 full-length layers | 1,960 MiB ≈ 1.914 GiB | Difference between two FP32 and two FP16 saved activation pools |

The MLP calculation is \(28\times2\times2048\times8960\times(4-2)\) bytes. It is a logical retained-storage difference at the maximum sequence length. The real public batches vary in length.

**These sizes must not be added and presented as a measured peak-memory improvement.** Tensor lifetimes, temporary overlap, padding, native backward, allocator reservations and the external GPU sampler determine the scored peak. The package contains no measured model peak.

## 6. Verification detail

The four-update Soup factory comparison recorded four exact gradient snapshots and four exact parameter snapshots. It exercised four native head forwards, four native head backwards, four compact summary calls, 16 native dropout forwards and eight fused dropout-gradient backwards. Row order, token counts, accuracy and RNG matched exactly. Maximum observed differences were:

- Scalar/logged loss: `4.76837158203125e-7`.
- Rewind row loss: `4.76837158203125e-7`.
- Entropy: `2.384185791015625e-7`.

The consolidated suite includes vocabulary 151,936, causal shifting, padding and ignored labels, empty supervision, explicit denominators, scaled backward, saturation, NaN/Inf propagation, storage ownership, adapter gradients, fallback behavior and patch composition. CUDA-only tests are listed as skipped, not counted as passes.

The physical readiness gate additionally requires `test_physical_t4_full_fusion_training`. It constructs a two-layer local Qwen with real nested-NF4 projections, real q/v LoRA and real Soup/TRL/Accelerate training for two updates. Delegating launch observers require the selected head, NF4, RMSNorm, compact MLP and LoRA paths to execute. The fixture preserves the real TRL adapter conversion followed by Soup's existing FP32 correction. No production device guards or GEMMs are replaced in that physical test. It remains unexecuted here.

The static scanner finds 22 candidate JIT function definitions, including disabled experiments and helpers. Definition count is not active-kernel count or eligibility. The supplied two SwiGLU functions are excluded. The simple dropout epilogue is flagged at 0.91 structural similarity to the example, requiring manual originality review; its separate derivation and provenance are documented in `PROVENANCE.md`. No changes were made to evade that detector.

## 7. Run the delivered notebook

Open **Soup_Fusion_T4.ipynb** in Kaggle, enable Internet for dependencies/model/data downloads, select T4 GPU hardware, and use Run All. The notebook contains a compressed copy of the source package; no repository URL or source upload needs to be filled in. It exposes only GPU 0 and refuses to proceed without the target T4.

It installs the organizer's pinned packages, unpacks the source, runs the profile search, and produces a JSON summary and downloadable result archive. The archive includes the best verified source, its actual official outputs, and logs. Local commits are evidence snapshots; nothing is pushed or submitted to a team repository.

The official notebook pins Soup, Transformers, PEFT, TRL, bitsandbytes and Accelerate. It does not explicitly pin Torch/Triton. The implementation was audited with Torch 2.11 and Triton 3.6; readiness reports actual versions, and strict feature guards can make a different runtime fail readiness rather than silently claim the optimizations executed.

## 8. Command-line continuation

From the extracted source package, on the actual T4 environment:

```bash
python tools/run_search.py --record-seconds 33 --out runs_search
```

To inspect the plan without accessing CUDA:

```bash
python tools/run_search.py --plan
```

To compare only the two cached variants:

```bash
python tools/run_search.py --profiles fused cached256 \
  --record-seconds 33 --out runs_search_cached
```

The underlying official commands remain:

```bash
python task/prepare_data.py
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2 --out runs/final
python harness/check_rules.py --result runs/final/result.json
python harness/score.py runs/final/result.json
python tools/verify_record.py runs/final/result.json \
  --record-seconds 33 --reference-baseline-seconds 77
```

The official two-repeat order is base, solution, solution, base. Runtime is summarized by median; memory by the maximum sampled peak for each side. Preserve the final source after measurement. The verifier binds exact result bytes, the official checks and current solution source; with a readiness report it also validates the runner's before/after source observation. The unchanged harness does not provide per-run source attestation, and the verifier does not invent it. [C2–C4]

## 9. Acceptance and the remaining blocker

A useful target result must pass the original tests and gates: genuine original kernels in every measured run with at least five percent of synchronized step-wall time from step 3, unchanged task/adapter/outputs and exact counts, the specified loss/metric tolerances, public adapter evaluation within 0.5%, and no more than the permitted time/memory regressions. Hidden-row evaluation and manual originality review remain the jury's work. [C3–C4]

`verify_record.py` prints `target_met_under_official_public_protocol=true` only for a verified, admissible, strictly sub-target median. It always distinguishes that from a proven world record. Its tests use explicitly synthetic timing fixtures; those fixtures are never reported as this candidate's performance.

**The next necessary input is a physical T4 run.** The authorized service notebook and API client are prepared; a connection that verifies the service's HTTPS identity is needed to obtain its first result ZIP. Use its actual completion/error state, time, peak, evaluation, launch and output evidence to decide the next change. A final paired Kaggle run must separately establish the full automatic gates and sub-33-second comparison.

## Sources and research provenance

- **C1 — Official starter, pinned snapshot:** https://github.com/MakazhanAlpamys/soup-kernel-challenge/tree/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb
- **C2 — Complete cold-run protocol:** https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/run_task.py
- **C3 — Gates and original-kernel accounting:** https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py
- **C4 — Fixed task and score:** https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/task/task.yaml and https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/score.py
- **Qwen semantics:** https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py
- **PEFT LoRA semantics:** https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py
- **NF4 dataflow:** https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py and https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py
- **AMP semantics:** https://docs.pytorch.org/docs/2.11/notes/amp_examples.html

Recovered user research: `Qwen_Inspired_Soup_Kernel_Optimization.md` version 6; `Qwen_Kernel_Optimization_Review.md` version 6; `Nanochat_Ngram_Soup_Transfer.md` version 5; `KERNEL_HANDOFF.md` version 6 and `LORA_RESEARCH_ROUND_7.md`; `Kernel_Handoff.md` version 3, `Research_Update.md`, and both recovered code bundles. Exact original artifact identities and bundle hashes are listed in `PROVENANCE.md`. Validation JSON/XML/logs are included in the source archive. No new GPU timing is claimed by any research or validation file in this deliverable.
