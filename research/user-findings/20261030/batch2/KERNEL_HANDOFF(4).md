# Soup kernel challenge — session handoff

Updated 2026-10-30 after Round 6. **Read this first.** Latest detail: `LORA_RESEARCH_ROUND_6.md`; results: `ROUND_6_EVIDENCE.json`; runnable source: `KERNEL_ROUND_6_REPRO.zip`. User prefers Markdown for session transfer; PDF was rejected for that purpose.

## Current state and direction

- Goal: accelerate the fixed Decentrathon 6.0 Soup Kernel Challenge run, preserving required behavior.
- Completed: six research rounds; public audit; CPU numerical diagnosis; actual tiny PEFT/TRL/HF training with real Soup Rewind. Round5 has six original JIT functions/13SM75 configurations; Round6 adds an original inline-MMA candidate and one three-mode diagnostic probe (four new SM75 configurations).
- **No GPU execution, T4 benchmark, challenge submission integration, full Soup CLI run, or official pass.** Research kernels exist but have not been installed in an authorized submission checkout.
- Team checkout and T4 environment remain unidentified. Public starter is not the team's repository.
- Compiler finding: ordinary `tl.dot` in audited Triton3.6 takes the SM75 FMA route. **New original inline PTX compiles to two HMMA.1688.F32 instructions on SM75.** Compilation/mapping are verified; GPU correctness/speed are unknown. Preserve the distinction between ordinary-dot and explicit-MMA paths.
- **Next:** untouched T4 trace; native-CUDA loss math probe; adapt tested H0 to actual CUDA/NF4/Soup setup, retaining native head/backward. Compare native GEMM+L1e against ordinary-dot and inline-MMA full L1. Vocabulary reductions/A0 follow target correctness; H1/L2 remain later.

## Fixed workload, pins, rules

| Item | Contract |
|---|---|
| Model | Qwen2.5-1.5B-Instruct; 28 layers; H1536; I8960; V151936; 12 query/2 KV heads; head dimension128 |
| Platform/base | One Kaggle T4 (SM75); fp16 compute; NF4 double quantization; tied frozen head |
| LoRA | q_proj1536 and v_proj256; rank16; alpha32; scale2; dropout0.05 |
| Training | Batch4; sequence≤512; 256 rows; 64 steps; epoch1; accumulation1; AdamW lr2e-4 |
| Evaluation | Hidden rows, different seed; preserve optimizer/schedule/data order |
| Brief deadline | 2026-10-31 23:59 Asia/Almaty |

Pins: soup-cli0.77.0, transformers5.16.1, peft0.20.0, trl0.29.1, bitsandbytes0.50.2, accelerate1.14.0. Install command does not explicitly pin torch/Triton; brief expects2.11/3.6.0. Actual GPU versions/dtypes remain unverified.

Starter `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`; Soup `846888a045b31084452fa08a684a7a38c02f5f87`.

- Preserve dropout/masks, genuine metrics, Rewind, labels/counts, order, settings. Do not enable fast_lora, cut CE, checkpointing, offloading, or packing.
- Original useful `@triton.jit` under `solution/` must contribute≥5% of **synchronized callback step-wall time, steps3 onward, every measured run**. Numerator: eligible CUDA-event intervals. Checker intersects eligible names across runs and takes minimum share. Warmup launches lack timed intervals; 5–6% is warned as marginal.
- Copied/ready-made kernels, starter SwiGLU even rewritten, and compiler-generated kernels do not count. Attribute ideas; no artificial delay/unused work.
- Whole-command score includes apply/load/compile/train/save with fresh caches. Ratios below1.05 tie; positive category needs≥4.7619% reduction. If training savings D and added cost C, require `D≥C+Tbaseline/21`.
- First loss±1%; mean difference≤3%; metrics/Rewind≤3%; exact token/row counts; eval≤0.5% worse; time/memory≤10% worse. Soup tests and official unchanged harness/checks remain outstanding.

## Round 6 evidence and interpretation

Environment: CPU torch2.11.0, Triton3.6.0, Transformers5.16.1, Accelerate1.14.0, PEFT0.20.0, TRL0.29.1, Soup0.77. Installed SFT/PEFT/Rewind/recorder source matches pinned remote files. No pretrained weights or public data loaded. Source, dependencies, results, and reproduction commands are in the ZIP.

**Real integration:** tiny Qwen2 V37/H32/I64/layers2, tied frozen head, real q/v LoRA rank16/alpha32/dropout.05. Eight trainable adapter tensors/3,584 elements; nonzero B exercises all paths. Actual TRL collator, SFT construction/logging, HF train loop, Accelerate conversion, Soup sampler FIFO/JSONL writer/reader run. Candidate owns genuine-summary SFT compute_loss; stock SFT's logits-dependent metric body cannot run unchanged without logits.

Two full-loop pairs: accumulation1 has four updates; accumulation2 diagnostic has two updates/four micro-batches. Every clipped adapter-gradient and parameter snapshot is exact. Real entropy/accuracy/tokens and Rewind IDs/counts/row losses/step-micro state are exact, no underflow/disable. **Scalar/logged loss differs by at most4.76837158203125e-7.** Do not call all logs bitwise equal. Four native head forwards/backwards per run, identical full shapes/fp16 dtype.

Fixture:12rows,100real tokens,64supervised. Actual order `[3,5,0,11,7,6,8,2,10,1,4,9]` verified independently by first-token witnesses; not challenge order. Three scaled real-PEFT comparisons (scales1/65536/1000, externalD17) match all8 gradients exactly; three separate CPU GradScaler+clip+AdamW updates exact; eight lifecycle checks pass. Main loop uses explicit CPUfp16 autocast; separate scaler test is not a CUDA scaler trajectory. No NF4/fullSoupCLI/fullmodel.

**Loss diagnosis:** original R5 four saturated dZ failures remain. At margin16/scale65536/D2, NumPy and PyTorch exp differ by one fp32 ULP (2^-24); multiplying32768 gives exactly0.001953125, or16half steps. Block/native logp differ only4.55e-13; native VJP with either gives identical half gradients. PyTorch-exp substitution matches; NumPy-exp substitution reproduces the failures. This diagnoses interpreter arithmetic, not a GPU pass.

Two R5 entropy failures mainly inherit native CPU normalizer reduction error: native logS5.3333349228 versus blocked5.3333320618. A source-motivated16-lane fold reproduces nativeCPU logS; blocked logp is closer to fp64. Keep original gates/failures. New target-only diagnostic compares tl.exp, libdevice exp, and explicit FMA on identical native logp. Three SM75 variants compile at20registers, zero shared/local/stack. It materializes fp32 logp and is not the production loss/eligibility contribution.

**Inline MMA:** original rank16 kernel uses four warps/16×32 outputs per CTA; two rank8 MMAs/warp, explicit half round then fp32 native-mask/scale products. Compiled35registers, zero shared/local/stack, two assembled HMMA instructions. Collective participation inspected: no branch/exit before final HMMA, both unpredicated. CPU mapping covers65,536half bit patterns and six shapes/103,144outputs. It is not an MMA numerical emulator. Contiguous/nonaliasing/SM75 guards only; launch helper unexecuted.

At M2048,N1536, inline MMA uses6,144CTAs and33MiB modeled streams versus1,536CTAs/19.5MiB for ordinary-dot32×64. Operand repetition can erase arithmetic advantage;35registers/HMMA is not a speed claim. Keep L1e first control.

**Runner:** preflight says unavailable_no_cuda. Explicit interpreter smoke: five core passes, three loss fixtures/12gradients, two known saturated mismatches. Default CUDA suite prepares42forward/168gradient checks; optional inline-MMA/maxshape/timing modes remain unexecuted. Preserve sibling `round6_mma` directory. CPU/interpreter/mapping/compile evidence is separate from target execution.

**Prior R5:** L1e four controlled graphs match all7 gradients; full L1/A0 five interpreter cases each pass declared gates. Full L1 differs from native CPU half GEMM at K1536 despite matching explicit fp32-dot reference. R5 loss CE42/42, entropy40/42, dZ164/168; nonfinite patterns exact. Larger partial tile4096 uses189registers versus68 for1024. Detailed old results remain in Round5 report/evidence/bundle.

## Head/loss contract and H1 correction

H0 preserves native full-shape head forward/backward. Cache actual rounded full-row fp16 logits privately; stream genuine fp32 statistics. Native backward includes actual upstream scalar and exact denominator before half store. Preserve tied fp32 parameter. Frozen weights are excluded from normal autocast caching; avoid duplicate retained head casts per chunk.

| Quantity | Semantics |
|---|---|
| CE | Within-sequence shifted labels≠-100; exact passed denominator |
| Entropy | All attention-mask positions, including prompts/final tokens |
| Accuracy | Supervised shifted correct count/supervised count |
| Rewind | Original-row CE mean; zero-target row loss0/count0 |
| num_tokens | Cumulative attention-mask count |

No denominator clamp: finite all-ignored batch without supplied D has NaN loss/zero gradients; positive D gives zero loss/gradients. Soup already consumes `_soup_chunked_step.token_loss[B,T]` without logits. Guarded context must retain genuine summaries until Rewind consumes them, clear stale/exception state, and preserve direct/eval/unsupported outputs. Tiny real TRL/PEFT/Rewind nesting is now tested; actual CUDA/NF4/Soup construction remains work.

**H1 correction:** ignored/padded NaN/+Inf rows can have zero CE but NaN native gradients and entropy even after mask0. Finite-only evidence does not justify skipping them. Supervised-only caches may retain row flags and reconstruct exceptional gradients. Skipping head-forward rows omits the state needed to detect them; require compatible fallback/proof. Mixed finite/-Inf gives valid gradients but NaN entropy.

Centered vocabulary merge: block `(m_k,s_k,q_k)`, `q_k=sum(exp(z-m_k)*(z-m_k))`; merge `S=sum(a*s)`, `Q=sum(a*(q+(m_k-m)*s))`, `a=exp(m_k-m)`. CE=`(m-z_target)+logS`; entropy=`logS-Q/S`. Save m/logS separately; track nonfinites and first-index/first-NaN argmax. Scratch at 2,048 rows: 1.4844 MiB for 4,096 tiles versus 5.8203 MiB for 1,024 tiles. Larger tile uses more registers; benchmark both.

## LoRA, attention, economics

Preserve NF4 base, native dropout decisions, cast/round boundaries, separate branch accumulation. Frozen base W contributes dX. Expected fp32 residual/base outputs/adapter storage and fp16 adapter GEMMs; verify after Soup alignment.

- L1e keeps native GEMMs, fuses cast+dropout backward. Gate to enabled CUDA fp16-autocast/FP32-input training, p.05, ordinary non-inplace dropout. Preserve shape/strides/call order; fallback unsupported modes and higher-order use.
- Full L1 adds rank16 GEMM fusion; preserve half round before fp32 mask/scale products. No conditional zero replacing multiplication or unsafe reassociation/contraction.
- Maxshape traffic removal: L1e 24 MiB/branch, 1,344 MiB over 56 branches; full L1 36 MiB/branch, 2,016 MiB. For 54 active branches: 1,296/1,944 MiB. Logical traffic is not DRAM measurement/peak saving. Full L1 may remain bandwidth-limited despite FMA; compare with L1e.
- L2 grouping remains later; dA depends on U. No unsynchronized cross-CTA dependencies. Sharing half casts can change accumulation: separate2048+1→2049 versus half sum2048.
- A0 converts existing bool mask once to fp16{0,-Inf}, physical pitch ceil(T/8)*8, unchanged logical shape, reused across 28 layers. Conditional retained-bias saving 54 MiB; backend/peak unobserved. Preserve None/fallbacks; trace efficient-versus-math SDPA.
- A family accelerated 2× needs≥9.52% baseline share of the same wall denominator to retain 5% participation. Organizer GPU-profiler percentages differ. Several useful original kernels may be necessary; eligibility unmeasured.

## Prior audit and practical constraints

Public audit: Nr 52,890 real positions; Ns 33,608 supervised; 16 truncated rows; median length 140. Soup response masking active. Actual shuffle unobserved. Illustrative source-order Np83,700; Pythonseed42 Np93,196. Never reorder batches. Source-order selected-cache/recompute head arithmetic ratios0.5167/0.7175 are counts, not speedups; H1 now needs nonfinite qualification.

Maximum-shape logits: half 593.5 MiB / fp32 1,187 MiB; shifted accuracy copy 1,184.68 MiB; half head 445.125 MiB. Supervised cache mean 152.18 MiB, source-order max 432.95 MiB, possible public-grouping max 534.09 MiB. None is measured allocator peak.

Dataset `c641407ee50723c84a43ea2ec867b464d3893c12`; tokenizer `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Audit/repro: `PUBLIC_WORKLOAD_AUDIT.json` and ZIP. Round4 retains174+24 loss comparisons and detailed dtype/metric/L2 contracts; finite compaction conclusions are qualified above.

Rewind wraps RandomSampler before Accelerate; seedable-conversion isinstance misses wrapper. Inner generator=None consumes global CPU RNG. Preserve RNG, capture real row IDs; bare randperm(seed) is not harness order. Source-order 64 batches have 48 distinct T; runtime dimensions/few structural configs reduce fresh-cache compilation burden.

Target commands from the new extracted bundle (separate CUDA environment):

```bash
python round6_target/target_checks.py --mode preflight --output preflight.json
python round6_loss/run_cuda_probe.py --output loss_math_cuda.json
python round6_target/target_checks.py --suite loss --output loss_cuda.json
python round6_target/target_checks.py --suite core --include-mma --max-core-shape --output core_cuda.json
```

These are synthetic diagnostics; official commands remain in the appropriate checkout:

```bash
python -m pytest solution -q
python harness/run_task.py
python harness/check_rules.py
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
```

After target access, record checkout/commit, versions, row order, correctness, command time, peak, and every run's eligible share in `experiments.jsonl`. Update handoff at each stopping point.

## References

- Latest report/evidence/bundle named at top; component READMEs contain pinned sources/reproduction commands.
- Prior: Round5 report/evidence/repro; Round4 dtype/numerics detail; Round3 public workload report/audit.
- [Challenge brief](https://chatgpt.com/library?file_id=libfile_37b0abe044f88191b002d808e4d00570&account_id=personal).
- [Pinned starter/rules](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md).
- [Triton3.6 SM75 lowering](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp).
