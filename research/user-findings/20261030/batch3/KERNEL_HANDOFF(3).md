# Soup kernel challenge — session handoff

Updated 2026-10-30 after Round 5. **Read this first.** Latest detail: `LORA_RESEARCH_ROUND_5.md`; results: `ROUND_5_EVIDENCE.json`; runnable source: `KERNEL_ROUND_5_REPRO.zip`. User prefers Markdown for session transfer; PDF was rejected for that purpose.

## Current state and direction

- Goal: accelerate the fixed Decentrathon 6.0 Soup Kernel Challenge run, preserving required behavior.
- Completed: five research rounds; public-row audit; CPU numerical studies; tiny real Qwen/HF Trainer/Accelerate integration; **six original Triton entry points and 13 successful offline SM75 configurations**.
- **No GPU execution, T4 benchmark, integrated solution, full Soup run, or official pass.** Research kernels exist but have not been installed in an authorized submission checkout.
- Team checkout and T4 environment remain unidentified. Public starter is not the team's repository.
- Critical discovery: ordinary blocked `tl.dot` in audited Triton 3.6.0 falls back to scalar FMA below SM80. Our SM75 PTX confirms this; SM80 control emits MMA. Do not assign Tensor Core throughput to these kernels. T4 hardware still has Tensor Cores; other versions/inline-MMA routes are separate questions.
- **Next:** untouched T4 trace → genuine-summary control M0 → private fp16 logits-cache H0 with native head/native loss backward → original vocabulary forward reductions and narrower LoRA epilogue **L1e**. A0 shared aligned mask is a small parallel candidate. Full L1, H1 compaction, L2 grouping follow accepted controls.

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

## Round 5 evidence

CPU torch2.11.0+cpu; Triton3.6.0; integration Transformers5.16.1/Accelerate1.14.0. Interpreter is sequential NumPy simulation; offline compilation generates code. Neither proves GPU behavior. Independent staged reruns of H0, loss CPU/interpreter, and economics reproduce complete JSON results exactly.

**H0:** tiny real Qwen2 (V37/H32/I64/layers2), tied frozen head, trainable decoder; actual HF loss method, Accelerate conversion, CPU fp16 autocast. Five comparisons exactly match loss/summaries and all 25 gradient tensors / 18,720 elements each. Includes scales1/1000/65536, denominator17, all-ignored cases, retained graph/second backward. Three CPU GradScaler+clip+AdamW updates preserve exact parameters; 16 lifecycle and 4 ignored-row nonfinite checks pass. Native head shape/one forward preserved. No TRL/Soup/PEFT/NF4 execution.

Setting `outputs.logits=None` leaves its old mapping tensor; actual Accelerate still upcasts it. Reconstruct genuine CausalLMOutputWithPast. Saved-tensor hooks: baseline two fp32 vocabulary references alias one storage; candidate none, private half cache instead. Structural evidence, not peak memory.

**L1e:** pointwise half-gradient→fp32→native mask→fp32 scale. Four controlled native-dropout graph cases, including 3D input, match all 7 gradients exactly. CPU primitive RNG states equal; not CUDA RNG proof. Nonfinite/signed-zero check passes. Forward preserves original shape; backward reshape may copy noncontiguous input.

**Full L1/A0:** five interpreter cases each pass declared gates, including strides/tails/guards/nonfinites. Full L1 exactly matches explicit fp32-dot→half→dropout reference, but differs from native CPU half GEMM in 35 elements at K1536 (max6.42538e-5). External rounding boundaries do not fix internal reduction order.

**Vocabulary:** three original kernels: partials, merge, diagnostic dZ. CPU 240 comparisons: centered entropy 4/80 strict failures; uncentered negative control 16/80; CE/argmax all pass. Interpreter 42 forward: CE 42 pass, entropy 40 pass, discrete outputs exact. dZ 164/168 pass; four saturated-margin16/scale65536 failures: maxabs0.001953125, about 1.47% reference peak. Nonfinite patterns match. Keep failures visible; native backward remains oracle.

**Compilation:** 13 SM75 configurations, zero reported local/stack bytes. Full L1 registers 95/96/168, dynamic shared 4/4/8 KiB; L1e 22 registers; A0 24. Vocabulary partial tiles 1024/4096: 68/189 registers; merge 33/43; dZ 34/72. Resource counts are not measured occupancy. Do not ship generated binaries/cache.

## Head/loss contract and H1 correction

H0 preserves native full-shape head forward/backward. Cache actual rounded full-row fp16 logits privately; stream genuine fp32 statistics. Native backward includes actual upstream scalar and exact denominator before half store. Preserve tied fp32 parameter. Frozen weights are excluded from normal autocast caching; avoid duplicate retained head casts per chunk.

| Quantity | Semantics |
|---|---|
| CE | Within-sequence shifted labels≠-100; exact passed denominator |
| Entropy | All attention-mask positions, including prompts/final tokens |
| Accuracy | Supervised shifted correct count/supervised count |
| Rewind | Original-row CE mean; zero-target row loss0/count0 |
| num_tokens | Cumulative attention-mask count |

No denominator clamp: finite all-ignored batch without supplied D has NaN loss/zero gradients; positive D gives zero loss/gradients. Soup already consumes `_soup_chunked_step.token_loss[B,T]` without logits. Guarded context must retain genuine summaries until Rewind consumes them, clear stale/exception state, and preserve direct/eval/unsupported outputs. Real TRL/PEFT wrapper nesting remains work.

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

Run in the appropriate checkout:

```bash
python -m pytest solution -q
python harness/run_task.py
python harness/check_rules.py
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
```

After target access, record checkout/commit, versions, row order, correctness, command time, peak, and every run's eligible share in `experiments.jsonl`. Update handoff at each stopping point.

## References

- Latest report/evidence/bundle named at top; component READMEs contain pinned sources/reproduction commands.
- Prior: `LORA_RESEARCH_ROUND_4.md`, `CPU_NUMERICS_ROUND_4.json`, `KERNEL_ROUND_4_REPRO.zip`, Round3 workload report.
- [Challenge brief](https://chatgpt.com/library?file_id=libfile_37b0abe044f88191b002d808e4d00570&account_id=personal).
- [Pinned starter/rules](https://example.invalid/redacted-github).
- [Triton3.6 SM75 lowering](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp).
