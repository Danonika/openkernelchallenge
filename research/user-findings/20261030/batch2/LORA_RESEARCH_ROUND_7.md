# LoRA kernel research — Round 7

**30 October 2026 · Asia/Almaty**  
**Focus:** actual integration of native-mask LoRA backward fusion, forward loss statistics, and shared attention masks; combined training behavior; deferred Soup installation; AMP ownership.

## Decision

The best next T4 experiment is now a concrete composition: **keep native LoRA GEMMs and loss backward, add the L1e dropout-gradient epilogue, compute genuine forward loss/metrics with the original vocabulary kernels, and reuse one additive attention mask per forward.** Each component executes in a real-model CPU research fixture, and the combination passes through Soup's actual Rewind trainer construction and training loop. This gives us a more defensible correctness starting point than selecting a replacement GEMM from offline assembly alone.

The combined tiny-model run produces **exact adapter gradients and parameter updates at all four optimizer steps**, identical RNG, native head activity, row identities and counts, and small explicitly measured scalar differences. Maximum logged loss error is **4.76837158203125e-7**; maximum entropy error is **2.384185791015625e-7**. These results are in `ROUND_7_EVIDENCE.json` and the runnable `KERNEL_ROUND_7_REPRO.zip`.

**There is still no CUDA execution, NF4 execution, T4 benchmark, measured model peak, qualifying-kernel share, full Soup CLI run, or official challenge pass.** Two previously identified standalone entropy comparisons still fail their unchanged tight research gate. The combined trajectory does not erase those failures. The CPU trainer policy remains a research implementation, and this round identifies an additional AMP boundary that a target port must handle.

## 1. What this round establishes

| Question | New evidence | Remaining limit |
|---|---|---|
| Can L1e reuse the original dropout decision? | Actual PEFT native-dropout bridge preserves outputs, x/A/B gradients and RNG; original dropout backward disappears | CUDA saved-mask layout/RNG and NF4 layer remain unexecuted |
| Can custom forward statistics feed genuine consumers? | Real TRL metrics and Soup Rewind consume original CE/entropy/correct counts; native backward preserves gradients | Two standalone entropy mismatches; target exp/reduction behavior unknown |
| Does shared attention-mask conversion work in real Qwen? | One conversion per forward, reused across layers; exact model/optimizer comparisons | T4 efficient-versus-math dispatch and peak effect unknown |
| Do the components coexist? | All three execute together through actual Soup factory, Rewind and HF optimizer loop | Tiny floating-point CPU model, explicit CPU autocast |
| Where should installation occur? | Reversible deferred hook around Soup's actual trainer factory; global TRL class untouched | Full Soup setup and CUDA/NF4 eligibility policy still need validation |
| Is the existing outer autocast guard portable? | Real Trainer/Accelerate probe shows autocast inactive outside model forward and active inside | CUDA conclusion follows pinned shared source, not target execution |

The principal fixture has two Qwen2 layers, H32, V37, four query/two KV heads, and actual PEFT q/v LoRA with rank 16, alpha 32, scale 2 and dropout 0.05. It has eight trainable adapter tensors containing 3,584 elements. Its locally generated 12 rows contain 100 real tokens and 64 supervised shifted targets, including one wholly unsupervised row. Nonzero B matrices exercise adapter A and input-gradient paths. A separate attention fixture has three layers and 12 trainable tensors / 5,376 elements.

These are real model classes, actual adapters, ordinary autograd, optimizers, SFT metric consumers, and Rewind writer/reader implementations. They do not substitute a mock recorder or fabricate row IDs. Locally generated data and tiny floating-point base weights keep the scope reproducible without pretrained downloads. The full challenge remains the fixed Qwen2.5-1.5B NF4 workload described in the handoff.

## 2. L1e now participates in actual PEFT

### The bridge preserves the native forward

The new `l1e_bridge.py` wraps selected PEFT q/v instances and their existing dropout modules. It invokes PEFT's original layer forward and `nn.Dropout`'s original forward exactly once. Base computation, LoRA A/B matrix multiplications, scaling and result addition continue through native code. The insertion point is the dropout output's existing FP32-to-FP16 boundary before the autocast adapter GEMM.

The original dropout output is passed detached into a custom autograd bridge, alongside the original FP32 input and actual saved mask. The bridge returns the same rounded half activation. In backward it supplies the input VJP directly, so the old dropout backward is not also traversed. This avoids both a second random draw and an accidental doubled gradient. Native A/B parameter-gradient computation still sees the correct half activation.

For the CUDA route, the pinned ordinary dropout dispatch points to `native_dropout`, whose autograd state retains the bool keep mask and dropout probability. The bridge accepts only that recognized structure and reuses the bool mask directly. CPU `F.dropout` has a different observed form: a saved FP32 scaled mask on `MulBackward0`. The explicit CPU research mode validates that mask and extracts its bool pattern. That diagnostic conversion is not proposed GPU work. Unknown structures retain the ordinary native graph. The source contract is supported by [PEFT's LoRA layer][peft-layer], [its NF4 wrapper][peft-bnb], [PyTorch dropout dispatch][dropout] and [autograd derivative definitions][derivatives].

For the supported native bool-mask route the relevant order is:

`native half GEMM gradient → FP32 conversion → multiply by keep mask → multiply by 1/(1-p)`.

The implementation retains multiplication semantics rather than selecting literal zero for a dropped value. That matters for NaN, Inf and signed-zero behavior. It also preserves the preceding half rounding boundary; folding that rounding away would describe a different numerical operation.

### What was executed

Three actual PEFT-layer cases cover 2D/3D inputs and incoming scales 1, 65,536 and 1,000. Outputs, input gradients, both adapter gradients and final CPU RNG are exact. The profiled forward/backward keeps seven `aten::mm` and one `aten::addmm` on each side. Native dropout and Bernoulli counts remain one each. The observed CPU `MulBackward0` count falls from two to one; the remaining node is adapter scaling. A separate dispatch trace observes one dropout-output half cast in both paths, excluding the native base input cast.

A structural test uses actual CPU `aten.native_dropout` to exercise the private `_saved_result1`/`_saved_p` contract and bool-mask branch, including nonfinites and finite signed zeros. It establishes that branch's pinned autograd structure on CPU, not CUDA RNG agreement. One nonlinear fixture also compares first and second derivatives for x/A/B: higher-order backward uses native differentiable tensor arithmetic instead of Triton. General `torch.func`, `vmap`, compilation and arbitrary higher-order support are not certified.

The H0+L1e full-loop comparison preserves four clipped-gradient and parameter snapshots, genuine metrics apart from the known small H0 scalar-loss drift, actual Rewind records and RNG. Three separate CPU GradScaler/unscale/clip/AdamW updates are exact. Sixteen lifecycle/fallback records pass, including unknown-mask fallback, wrapper ownership and idempotent removal.

### An observed branch-count correction

Across four microbatches in the two-layer model there are 16 native dropout calls but only **eight L1e backwards**. Both first-layer q/v inputs come from frozen embeddings and do not require input gradients; those calls correctly remain native. Later-layer q/v inputs use the bridge.

This supports the previously distinguished 54-versus-56 branch scenarios for the 28-layer target. If the full target has the same gradient structure, 54 input-gradient epilogues are candidates. At M=2048/H=1536, the earlier logical-traffic estimate is then 1,296 MiB removed over 54 branches, rather than 1,344 MiB over 56. This remains an operation-level traffic calculation, not measured DRAM traffic, runtime or allocator peak. Runtime guards must inspect actual target tensors.

The default CUDA mode is restricted to SM75, FP16 autocast, FP32 contiguous grad-requiring input and FP32 A/B weights, one ordinary rank 16/alpha32 adapter, p=0.05 non-inplace dropout, and supported native PEFT module types. Unsupported adapters, layouts, hooks and execution modes retain native behavior. Private saved-mask access is a concrete portability risk requiring the installed target wheel to be checked.

## 3. Original forward statistics with native backward

### A useful intermediate design

Round 6 established the private-fp16-logits H0 path with native loss math. Round 7 replaces its forward statistics with our original vocabulary reductions while inheriting the native ATen backward unchanged. The saved backward inputs remain **the original rounded logits, shifted targets and exact denominator**. The new maximum/log-normalizer values are not used to reconstruct the gradient.

This separation matters because Round 6 traced saturated interpreter dZ failures to a one-ULP exp difference amplified by loss scaling. A forward-only integration can make the genuine CE, entropy and accuracy work useful while preserving the established backward oracle. It does not fix or certify the separate custom dZ kernel.

The two forward kernels produce per-token CE, entropy and correct counts. SFT aggregates actual entropy over attention-mask positions, counts real tokens cumulatively and computes supervised shifted accuracy. Soup consumes the genuine per-token CE for original-row means/counts. No constant or fabricated metric is introduced. The inherited model-output reconstruction prevents Accelerate from upcasting a returned full logits tensor; the private cache remains the actual full-row half logits.

The centered reduction still merges block maxima, normalizers and exponential first moments. For block maximum m_k and global m, it uses `a_k=exp(m_k-m)`, sums `a_k*s_k`, and merges `a_k*(q_k+(m_k-m)*s_k)`. CE and entropy then follow from the merged normalizer and moment. This is the existing algorithm from our prior rounds; the new work is its consumer integration and output specialization.

### Actual training and numerical coverage

Four paired full training experiments cover vocabulary tiles 1,024/4,096 and accumulation 1/2. Accumulation 2 is a diagnostic of microbatch/denominator behavior; the challenge still requires accumulation 1. Every clipped adapter-gradient snapshot and every model-parameter snapshot after an update is exact. RNG and four native head forwards/backwards per run match. Maximum loss and Rewind row-loss difference is 4.76837158203125e-7; entropy differs by at most 2.384185791015625e-7. Accuracy, counts and row identities are exact.

Six scaled real-PEFT comparisons, six CPU GradScaler updates and 11 lifecycle checks per tile size also pass. Separately, 42 forward fixture/tile comparisons include the real vocabulary dimension 151,936, multiple partial blocks, batch-local causal boundaries, saturation, ignored labels and nonfinites. All 126 native-backward comparisons have exact finite values and matching nonfinite patterns.

The real full-loop fixture uses V37, so it has one vocabulary partial. Multi-part merging is established by the standalone numerical sweep, not by that tiny end-to-end trajectory. Both scopes are retained explicitly.

**Two known tie-entropy failures remain under atol=rtol=2e-6.** They are the earlier CPU normalization/reduction discrepancy, where blocked math can be closer to FP64 yet differ from the native CPU reference. The result therefore reports `all_integration_checks_pass=true` and `all_forward_numerical_gates_pass=false`. No tolerance was relaxed to conceal the failures. The previous four custom-dZ mismatches remain separate historical evidence.

### Remove output work that this version does not use

The original diagnostic finish emitted maximum, log-sum, flags and argmax arrays in addition to the consumed summaries. The native-backward version does not use those four arrays. The new original `token_finish_summary` keeps the same arithmetic but stores only CE, entropy and correct counts. The old Round 5 files remain immutable.

Across 42 fixture/tile combinations, all 126 summary-output bitwise comparisons match the old wrapper, including nonfinite bits. The complete integration suite reran and all earlier evidence sections remained unchanged after the expected source/name/metadata normalization. The preceding results are preserved as `results_before_summary.json` rather than silently discarded.

At 2,048 positions the specialization removes four allocations and 32,768 bytes of output buffers/stores. This is an exact output-footprint ledger, not a measured peak or speed gain. One explicit SM75 configuration compiled offline: `BLOCK_P=256`, four warps, 43 registers/thread, 16 bytes launch shared memory, zero local/stack. Its resource footprint matches the earlier finish at that shape. Compilation used explicit `num_stages=1`; no GPU runtime artifact or launch is claimed.

The diagnostic CUDA wrapper still performs a host-synchronizing valid-label guard. The native backward still uses the correctness-oriented chunk policy inherited from H0; the fixture chunk 7 is not a target tuning recommendation. Both costs must be addressed or included in a real target measurement.

## 4. Shared attention masks: integration and a backend qualification

### Preserve construction; share the converted result

Pinned Qwen2 constructs its ordinary mask mapping before the decoder-layer loop. The same bool mask can reach each full-attention layer. The native SDPA composite converts bool values to query-dtype additive values inside each call. For efficient CUDA attention, preceding mask strides must satisfy an eight-element alignment requirement. These observations come from the pinned [Qwen2 forward][qwen], [Transformers SDPA adapter][sdpa-hf] and [PyTorch attention implementation][attention].

The new controller opens a context on the selected decoder's forward and observes that existing mask at attention prehooks. The first supported consumer runs our original conversion; later consumers reuse its result. Its logical shape remains `[B,1,T,T]`, with padded backing pitch `ceil(T/8)*8`. Bool true becomes 0 and false becomes negative infinity. `None` remains `None`, preserving the ordinary causal/GQA branch decisions.

The cache exists only within one forward, holds the source strongly, and keys it by identity and tensor version. Cleanup occurs in `finally`. This avoids stale reuse across optimizer steps or same-shape batches with different padding. It does not reconstruct the model mask, change embeddings/projections, or choose a new attention backend. Eligibility checks FP16 autocast and FP32 hidden states inside model execution, before the projections; actual NF4 Q/K/V dtype remains to be verified.

### Numerical and structural evidence

Six real three-layer Qwen2/PEFT comparisons cover right and left padding, consecutive same-length masks with different content, a wholly masked row, all-ones attention and `None`. Loss, logits, all adapter gradients, parameters and RNG are exact. Three AdamW updates also agree exactly without relying solely on operation-tracing mode. A forced CPU math-backend pair is exact.

For an active masked forward the checks require one original conversion and two reuse hits. Three native `aten.where` mask conversions become zero. SDPA input flags, Q/K/V shape/dtype/stride and the default CPU FlashAttention forward/backward operator sequence remain unchanged. Four direct conversion cases preserve values, nonfinite/zero-sign patterns and untouched physical padding, including a nonunit input stride. Eleven lifecycle checks cover eval, no-grad, inference mode, inference-created masks, FP32, existing additive/dictionary masks, eager fallback, exception cleanup/recovery and uninstall.

### Storage evidence is real but conditional

Saved-tensor hooks observe distinct backing allocations retained for backward:

| Three-layer CPU fixture, B2 | Native retained owners | Shared padded owner | Difference |
|---|---:|---:|---:|
| T9 | 3 × 324 = 972 bytes | 576 bytes | 396 bytes |
| T13 |3 × 676 =2,028 bytes | 832 bytes | 1,196 bytes |
| T8 |3 × 256 =768 bytes | 256 bytes | 512 bytes |

These are observed tensor-storage sizes, not process or allocator peaks. Padding makes the shared allocation larger than one unpadded mask, but the layer references share one owner.

The forced CPU math backend retains **zero additive-mask tensors on both sides**, even though repeated conversions disappear. This is a concrete counterexample to treating retention savings as backend-independent.

The earlier maximum-shape target estimate therefore stays conditional:

`(28-1) × B4 × T512 × T512 × 2 bytes =54 MiB`.

It applies if the target backend retains one separate half-mask allocation per layer and the candidate shares one. A `None` path has no conversion to remove; a math path need not retain those owners. Short sequences reduce the footprint quadratically. Pinned CUDA hardware guards reject SM75 for Flash and cuDNN attention, leaving efficient or math depending on the installed build and actual inputs; the CPU Flash observation does not identify the T4 backend. See [CUDA SDPA eligibility][sdp-cuda]. Python hooks/cache overhead and allocator overlap must be included in target measurements.

## 5. Composition through Soup and the AMP boundary

### A concrete deferred installation point

The challenge calls `solution.apply()` before Soup builds its model and trainer. Soup's `_build_rewind_trainer` later receives the completed PEFT model in `trainer_kwargs`. The new reversible deferred adapter selects a compatible candidate trainer at that point and delegates construction to Soup's original factory. That preserves its actual Rewind class, dataset fingerprint, writer and state attachment. The startup/factory order is sourced from the [challenge entrypoint][apply] and [pinned Soup SFT implementation][soup-sft].

In the executed fixture, two actual Soup wrapper instances construct baseline and candidate trainers. Local models/tokenizer/data are supplied directly; Soup `setup()` and full CLI loading/tokenization are not exercised. Selection is explicitly limited to the CPU research class and fixture marker. Global TRL exports and stock `SFTTrainer.compute_loss` remain unchanged; the candidate owns its genuine-summary loss method as before.

All three integrations then run together for four microbatches and four optimizer updates. This loop uses the fixture's explicit CPU FP16 autocast policy without a GradScaler; the separate CPU scaler comparisons are different experiments. The checks require four nonempty gradient/parameter/log/row traces, four native head forwards/backwards on each side, and the exact requested component set. Actual activity is:

| Component | Observed combined activity |
|---|---:|
| `token_block_partials` | 4 interpreted invocations |
| `token_finish_summary` | 4 interpreted invocations |
| `aligned_additive_mask` | 4 conversions, each reused once |
| `dropout_half_gradient_epilogue` | 8 interpreted backwards |

All four clipped-gradient and parameter snapshots are exact. Real Rewind rows/counts and first-token identity witnesses agree. Every row appears once in order `[3,5,0,11,7,6,8,2,10,1,4,9]`; this is the fixture order, not an inferred challenge shuffle. Token totals, accuracy, gradient norm, learning rate, step/epoch, RNG and head metadata agree. Loss/entropy/row-loss discrepancies remain recorded rather than described as bitwise identical.

Factory installation consumes no RNG, is idempotent for the same callbacks, refuses conflicts, and restores its original method. Already-created trainer controllers continue working after the startup hook is removed. Constructor mapping/object identities and Soup configuration values are preserved; that does not claim deep immutability of objects that Trainer legitimately initializes.

### A newly executed reason the CPU guard cannot simply become a CUDA guard

Transformers 5.16.1's `autocast_smart_context_manager()` returns a null context and relies on Accelerate. Accelerate 1.14.0 wraps **model.forward** with autocast and output conversion. Thus a trainer method can run outside autocast even when the model subsequently executes under AMP. The pinned [Trainer][trainer] and [Accelerate preparation][accelerator] source files match the installed bytes.

A separate one-step actual Trainer test uses CPU BF16 solely to activate Accelerate's real native-AMP preparation path. It does not override Trainer's context manager and has no GradScaler. The observed sequence is:

| Observation point | Active autocast | Logits dtype |
|---|---|---|
| Trainer before model call | False | Not produced |
| Original model forward | True | BF16 |
| Trainer after model return | False | FP32 |

The ordinary CUDA-FP16 path uses that shared preparation mechanism in the pinned source. Therefore a direct `cpu`→`cuda` substitution in the existing outer H0 eligibility check could reject a valid prepared training call. The target design must check configured AMP/device policy at trainer entry and actual autocast/device/logits dtype inside the model/loss callback. L1e and A0 already inspect their active model-execution boundary. CUDA itself remains unexecuted; other accelerator backends/wrappers may differ.

CPU BF16 here is a lifecycle diagnostic, not a proposal to change the challenge's FP16 training. Soup also aligns only BF16 trainable `lora_` parameters to FP32 just before training in its FP16 path; that is not evidence for the dtype of every model tensor. See [Soup dtype alignment][dtype].

## 6. Priority and target acceptance plan

The evidence supports the following order of target experiments. Each is an ablation with the fixed model/data/order/optimizer/settings preserved.

1. **Untouched baseline trace.** Record exact checkout and package versions, CUDA dtype/scaler/dropout contracts, actual SDPA backend, row order, command time and peak. Resolve the AMP boundary in the target trainer integration before interpreting a silent fallback as success.
2. **Native GEMMs plus L1e.** Verify native bool-mask reuse, one dropout draw, actual active branches, gradients and updates under CUDA scaling. Measure whole-step savings and eligible kernel intervals.
3. **Private logits plus summary-only forward kernels and native backward.** First establish CUDA CE/entropy/row and gradient agreement. Keep the known tie and saturated diagnostics. Treat the label synchronization and native-backward chunk size as measured costs, not assumed harmless overhead.
4. **Shared masks as a separate ablation.** Confirm whether the target uses efficient or math SDPA, preserve its flags/backend, inspect retained owner identities and peak. Include hook overhead. A0 is not presumed to provide 5% participation by itself.
5. **Combined candidate.** Only after each CUDA ablation passes, run the unchanged full harness across all measured seeds and required cold caches. Compare Round 6 full-L1 inline MMA later against the native-GEMM+L1e control; offline HMMA presence still does not establish a winner.

The challenge numerator is eligible original CUDA-event time under `solution/`; the denominator is synchronized callback step-wall time, steps 3 onward. Eligible names are intersected across runs and minimum run share must be at least 5%. The new combined CPU invocation counts do not measure that fraction. Whole-command scoring includes apply/load/compile/train/save; a 1.05 ratio requires at least 4.7619% baseline-time reduction. Added setup cost can erase a steady-state gain. These rules remain in the unchanged [starter/rules][rules] and detailed handoff.

The new bundle is a reproducible research package, not an installed submission checkout. The target environment and authorized team checkout are still absent. No public starter file was edited or submitted. Official tests remain:

```bash
python -m pytest solution -q
python harness/run_task.py
python harness/check_rules.py
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
```

## 7. Reproduction and continuity

`KERNEL_ROUND_7_REPRO.zip` includes all new source, component READMEs/results, required unchanged Round 5 loss and Round 6 integration dependencies, source-verification records and a file-hash manifest. Old result sections are retained where the finish specialization was refined. Logs, bytecode, installed dependencies, generated compiler binaries and downloaded third-party source are excluded.

With the recorded packages installed and sibling directory layout preserved:

```bash
TRITON_INTERPRET=1 python3 round7_l1e/integration_checks.py --output l1e.json
TRITON_INTERPRET=1 python3 round7_loss_integration/integration_checks.py --output loss.json
TRITON_INTERPRET=1 python3 round7_attention/attention_checks.py --output attention.json
TRITON_INTERPRET=1 python3 round7_composition/composition_checks.py --output combined.json
python3 round7_composition/autocast_boundary_probe.py --output autocast_boundary.json
```

The interpreter variable must be set before Triton imports; the AMP boundary probe runs in a separate process. These commands are CPU correctness experiments. Their printed training runtimes do not predict T4 speed. The component READMEs specify dependencies and optional offline compilation commands.

For a new session, start with **`KERNEL_HANDOFF.md`**, then use this report for reasoning and `ROUND_7_EVIDENCE.json` for exact results. The next research round should preserve the known failures and target limitations rather than recounting these CPU tests as new evidence.

## Primary references

[peft-layer]: https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/layer.py
[peft-bnb]: https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py
[dropout]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Dropout.cpp
[derivatives]: https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml
[qwen]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py
[sdpa-hf]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/integrations/sdpa_attention.py
[attention]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/attention.cpp
[sdp-cuda]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
[apply]: https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/solution/__init__.py
[soup-sft]: https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/trainer/sft.py
[trainer]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py
[accelerator]: https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py
[dtype]: https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/utils/mixed_precision.py
[rules]: https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md

All new implementation is original research code or a specialization/carry-forward of our own earlier kernels. Native installed libraries are references and integration components; no third-party GPU kernel implementation is copied into the deliverable.
