# Soup kernel challenge — session handoff

Updated **30 October 2026, Asia/Almaty, after Round 7**. Read this first. User wants continued LoRA/kernel research and **Markdown for fast session transfer**; PDF was rejected. Latest detail/evidence/source: `LORA_RESEARCH_ROUND_7.md`, `ROUND_7_EVIDENCE.json`, `KERNEL_ROUND_7_REPRO.zip`.

## Current decision and honest status

Seven rounds completed. The strongest correctness starting point is **native LoRA GEMMs + L1e dropout-gradient epilogue + private FP16 logits with original forward-summary kernels/native backward + shared per-forward attention mask**. Round 7 has actual PEFT integration, real SFT/Rewind consumers, observed CPU mask-storage lifetime, and a successful combined run through Soup's actual trainer factory.

**No CUDA/NF4 execution, T4 benchmark, full Soup CLI/setup, model peak, qualifying-kernel share, authorized submission checkout, or official pass.** No external code was pushed/submitted. The team repository and usable T4 runtime remain unidentified; public starter is not a team checkout. Do not infer access from unrelated user projects.

**Next:** target baseline trace; correct the H0 outer AMP eligibility boundary; verify CUDA dropout/label/metric/gradient behavior; then measure individual and combined ablations. Retain native GEMM+L1e as the control for the Round 6 inline-MMA candidate. Avoid presenting repeated CPU checks as new target evidence.

## Fixed contract

| Item | Required |
|---|---|
| Model | Qwen2.5-1.5B-Instruct; 28 layers; H1536; I8960; V151936; 12 q/2 KV heads; head dimension128; tied frozen head |
| Platform/base | One Kaggle T4 SM75; FP16 compute; NF4 double quantization |
| LoRA | q_proj1536/v_proj256; rank16; alpha32; scale2; dropout.05 |
| Training | Batch4; sequence≤512; 256 rows; 64 steps; epoch1; accumulation1; AdamW lr2e-4 |
| Evaluation | Hidden rows/different seed; preserve order/settings |
| Deadline | 2026-10-31 23:59 Asia/Almaty |

Pins: soup-cli0.77.0, transformers5.16.1, PEFT0.20.0, TRL0.29.1, bitsandbytes0.50.2, Accelerate1.14.0. Brief expects Torch2.11/Triton3.6; install command does not explicitly pin them. Actual GPU versions remain unobserved.

Starter `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`; Soup `846888a045b31084452fa08a684a7a38c02f5f87`.

- Do not enable fast_lora, cut CE, checkpointing, offloading or packing; preserve data, schedule, genuine metrics/Rewind and exact counts.
- Original useful `@triton.jit` under `solution/` must cover≥5% of synchronized callback step-wall time, steps3+, **every measured run**. Numerator is eligible CUDA-event intervals. Names intersect across runs; minimum share decides. Warmup launches lack intervals; 5–6% is marginal.
- No copied/ready-made kernels, starter SwiGLU even rewritten, compiler-generated kernels, artificial delay or unused work.
- Whole-command scoring includes apply/load/compile/train/save, fresh caches. Ratio<1.05 ties: need≥4.7619% baseline reduction; with added setup C and training savings D, require `D≥C+Tbaseline/21`.
- First loss±1%, mean difference≤3%, metrics/Rewind≤3%, exact counts, eval≤0.5% worse, time/memory≤10% worse. Official unchanged checks remain outstanding.

## Round 7: executed results

### L1e native-mask bridge

`round7_l1e/l1e_bridge.py` preserves original PEFT forward and native dropout exactly once. It reads the saved mask, passes the native output detached to a custom FP32→half bridge, and returns the direct input VJP so old dropout backward does not execute. Native base/A/B GEMMs/scaling/addition remain. CUDA accepts recognized `NativeDropoutBackward0` bool mask directly; CPU F.dropout diagnostic validates/extracts a scaled FP32 mask. Unknown structures fall back.

Three real PEFT cases (scales1/65536/1000) have exact output/x/A/B gradients/RNG. Native GEMM counts unchanged; original dropout backward disappears; one dropout/random call and one dropout-output half cast. Actual CPU native_dropout validates bool-mask private attributes; NaN/Inf/signedzero checks pass. One higher-order fixture uses native fallback and matches first/second derivatives. Sixteen lifecycle/fallback records pass, including wrapper ownership/idempotent removal.

H0+L1e real loop: four exact clipped-gradient/parameter updates, genuine metrics/Rewind/RNG retained; three exact CPU GradScaler updates. Across four microbatches:16 native dropouts, **8 L1e backwards**; first-layer q/v inputs do not require gradients. This supports conditional54 active branches, not56, in the28-layer model. CUDA/NF4/private-mask portability remains untested.

### Forward statistics + unchanged native loss backward

`round7_loss_integration/forward_stats_backend.py` installs on the existing H0 controller. Backward inherits R6 `CachedLogitsLoss.backward` unchanged, saving original half logits, shifted targets and divisor; new statistics do not feed gradient reconstruction.

Four full-loop pairs (tiles1024/4096 × accumulation1/2) have exact clipped gradients, parameters, RNG and native head calls. Max loss/row-loss difference4.76837158203125e-7; entropy2.384185791015625e-7; accuracy/counts/row IDs exact. Six scaled PEFT and six CPU AMP comparisons pass;11 lifecycle checks/tile.

Standalone42 forward cases include V151936, multi-part merge, causal boundaries and nonfinites;126 native-backward comparisons exact. **Two known tie-entropy failures remain at atol=rtol2e-6.** `all_integration_checks_pass=true`, `all_forward_numerical_gates_pass=false`. Full trainer V37 has one partial; do not conflate standalone vocabulary coverage with full-loop size.

New `summary_stats.py` reuses original `token_block_partials`; `token_finish_summary` stores only CE/entropy/correct. It removes4 unused diagnostic arrays/allocations,32KiB at2048positions. All126 summary bitwise comparisons match old wrapper. Old result preserved; specialization_equivalence verifies all earlier evidence unchanged after expected metadata edits. One explicit offline SM75 config BLOCK_P256/4warps/num_stages1:43regs,16B launch shared,0local/stack; no launch.

CUDA wrapper is diagnostic and has a synchronizing label guard. Native-backward chunk7 is a CPU correctness fixture, not target tuning. H0 remains first-order. Do not declare old custom-dZ kernel fixed.

### A0 shared attention mask

`round7_attention/shared_mask.py` wraps one real Qwen decoder forward, converts the existing bool mask once at its first attention consumer, reuses by identity/version within that forward, and clears in finally. Logical[B,1,T,T], padded physical pitch ceil(T/8)*8. None/additive/unsupported inputs pass through; no global attention registry/config edit. Active model autocast controls dtype; requires supported FP32 hidden/FP16 computation, training+grad, no cache/checkpointing.

Three-layer PEFT fixture:6 standalone exact model comparisons,3 exact AdamW updates,1 forced math pair,11 lifecycle checks,4 direct conversion/padding cases. One conversion+2reuses;3 native mask aten.where calls→0. Actual SDPA flags/QKV metadata/default CPU Flash backend unchanged.

Observed retained mask storage at B2: T9 native972/shared576B; T13 native2028/shared832B; T8 native768/shared256B. **Forced CPU math retains0 additive masks on both sides**, despite conversion removal. Thus prior54MiB target saving stays conditional on28 separate efficient-backward owners atB4T512. It is not allocator peak. Pinned CUDA Flash/cuDNN guards rejectSM75; actual efficient/math route unknown.

### Combined Soup factory + AMP boundary

`round7_composition/deferred_factory.py` demonstrates reversible deferred selection around actual `SFTTrainerWrapper._build_rewind_trainer`. Global TRL/stock SFT method untouched; original Soup factory attaches actual Rewind writer/state. Test supplies local model/tokenizer/data directly: **not full setup()/CLI**. Selection policy is explicit CPU research, not a drop-in submission.

Combined 4 microbatches/4 updates: all adapter gradients and model parameter snapshots exact; RNG/head activity/rows/counts exact; logged loss≤4.768e-7, entropy≤2.384e-7. This uses explicit CPU FP16 autocast without a GradScaler; separate CPU scaler tests are distinct. Non-vacuous gates require 4 head forwards+backwards, 4 statistics calls/no fallback, 4 A0 conversions+4 reuses, 8 L1e backwards across 4 installed q/v modules. Real order `[3,5,0,11,7,6,8,2,10,1,4,9]`, every row once. This is fixture order, not challenge order.

**New target-porting issue:** pinned HF Trainer autocast context is null; Accelerate wraps model.forward with autocast+FP32 output conversion. A separate real one-step CPU BF16 Trainer/Accelerate probe observes outerautocast=false, inside=true/nativeBF16, after=false/returnedFP32. CPU BF16 is only a lifecycle diagnostic; challenge remainsCUDAFP16. A simple `cpu`→`cuda` substitution in R6's outer active-autocast guard can silently reject valid prepared calls. Validate configured AMP/device at trainer entry, actual autocast/device/logits dtype inside model/loss. CUDA relevance is source inference, not execution.

## Prior results to preserve

- R6 actual PEFT/TRLSFT/Rewind H0 pairs:8 gradients/3584elements exact, scalar loss≤4.768e-7; two accumulation cases, real FIFO/writer/reader. No NF4/fullCLI.
- R5 six original kernels/13SM75 configurations; ordinary tl.dot lowers to FMA on SM75. R6 original inline PTX compiles to2 HMMA.1688.F32,35regs/0shared/local/stack. Mapping/participation checked; **no GPU correctness/speed**. At maxshape inline16×32 uses6144CTAs/33MiB modeled streams versus ordinary32×64 1536CTAs/19.5MiB; native GEMM+L1e remains control. Triton3.6 documentsCC8.0+ support; offlineSM75 compilation≠runtime support.
- R5 loss42forward:CE42,entropy40; dZ164/168. R6 saturated failure: one FP32 exp ULP2^-24 ×32768=0.001953125=16half steps. Native backward with blocked/native logp agrees; NumPy exp reproduces mismatch. Two entropy failures mainly CPU normalization reduction. CUDA math probe unexecuted.
- H1 nonfinite correction: ignored NaN/+Inf/all-Inf rows may have0CE but NaN dZ/entropy; mixed finite/-Inf gives valid gradients but NaN entropy. Supervised-only cache/head compaction needs flags/fallback/proof; do not silently skip such rows.
- H0 retains native full-shape head and actual full-row half logits privately, tied frozen FP32 weight. Reconstruct ModelOutput without logits; assignment alone leaves mapping stale. No denominator clamp: all-ignored/no denominator gives NaN loss and zero gradient; a positive denominator gives zero loss and zero gradient.
- Metrics: shifted supervised CE/accuracy; entropy includes all attention-mask positions; Rewind original-row mean/count, zero loss and count for zero-target rows; tokens cumulative.
- Maxshape half logits593.5MiB,FP321187MiB,shiftedaccuracycopy1184.68MiB,halfhead445.125MiB. Logical sizes, not peaks. Frozen head autocast excludes ordinary caching; avoid duplicate retained weight casts.
- Public audit Nr52890/Ns33608,16truncated,medianT140. Source-orderNp83700 and Pythonseed42Np93196 illustrative only. Rewind-wrapped sampler consumes globalCPU RNG; preserveactualroworder. Source64batches have48distinctT; preferruntime dimensions/few structural compileconfigs.
- L1e modeled max traffic removed24MiB/branch;54branches1296MiB. FullL1 36MiB/branch;54branches1944MiB. Family2× faster needs≥9.52% baseline wall share to retain5% new share; GPU-profiler percentages differ.

## Files and reproduction

New ZIP layout: `round7_l1e`, `round7_attention`, `round7_loss_integration`, `round7_composition`, plus unchanged `round6_integration` and `round5_loss` dependencies. Component READMEs contain exact commands/source links. Four original kernels invoked by combined test: `dropout_half_gradient_epilogue`, `aligned_additive_mask`, `token_block_partials`, `token_finish_summary`. H0/native backward contributes no original Triton time itself.

Recorded CPU environment: Torch2.11.0+cpu, Triton3.6.0, TF5.16.1, PEFT0.20.0, TRL0.29.1, Accel1.14.0, Soup0.77.0, datasets4.8.4, fsspec2026.2.0. Scratch cwd `/workspace/scratch/fc71c5cbd20c`; components under `tmp/`, outputs under `output/`. Packages isolated under `/tmp/kernel_r6_integration_deps:/tmp/kernel_r5_h0_deps:/tmp/kernel_r4_torch_deps:/tmp/kernel_r5_triton_deps` in that PYTHONPATH order. They may disappear; use bundle requirements in a fresh environment.

```bash
TRITON_INTERPRET=1 python3 round7_composition/composition_checks.py --output combined.json
python3 round7_composition/autocast_boundary_probe.py --output autocast_boundary.json
```

Use separate processes; interpreter environment must precede imports. CPU runtime prints are not target timings. Round6 ZIP retains target core/loss/math-probe runners, including optional inline-MMA diagnostics. Official target checkout commands:

```bash
python -m pytest solution -q
python harness/run_task.py
python harness/check_rules.py
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
```

After access, record actual checkout/commit, versions, row order, numerical outcomes, cold command time, peak and **each run's** eligible share in experiments.jsonl. Preserve fixed task/hidden evaluation. All new artifacts should be saved durably; update this same handoff identity/version rather than creating competing handoffs.
