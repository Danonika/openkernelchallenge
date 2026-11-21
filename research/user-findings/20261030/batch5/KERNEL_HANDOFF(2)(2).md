# Soup kernel challenge — session handoff

Updated: 2026-10-30 after research round 4. Read this first. Open LORA_RESEARCH_ROUND_4.md for the latest evidence and implementation contracts; round 3 retains public-workload arithmetic, and round 2 retains detailed metric/dtype contracts.

## Current state and next action

- Goal: accelerate the fixed Decentrathon 6.0 Soup Kernel Challenge run.
- Completed: four research passes, prescribed-version source review, a reproduced public-row audit, and independently rerun CPU loss/LoRA/sampler programs.
- **No solution kernels or T4 benchmark yet.** CPU counts/numerical results are measured; GPU speed/memory/quality gains are not.
- The authorized team submission checkout and available T4 environment remain unidentified. The public starter is not the team submission repository.
- User prefers Markdown for cross-session continuity.
- **Next:** untouched T4 trace -> genuine-summary control M0 -> private fp16 logits-cache H0 with unchanged native head forward/backward. Develop independent-branch LoRA fusion L1 in parallel. Shared aligned attention mask A0 is a new small candidate. H1 row compaction and L2 grouping follow accepted controls.
- H0 must start with native loss backward as numerical control. CPU success of softmax-minus-onehot does not select a CUDA/Triton formula. Full head recomputation remains a later memory tradeoff.

## Fixed workload and source pins

| Item | Fixed value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B-Instruct; 28 layers; hidden 1536; MLP 8960; vocabulary 151936 |
| Platform/base | One Kaggle T4; fp16 compute; NF4 with double quantization |
| LoRA | q_proj width1536, v_proj width256; rank16, alpha32, scale2, dropout **0.05** |
| Training | Batch4; sequence<=512; 256 rows; 64 steps; one epoch |
| Optimizer | AdamW torch; lr2e-4; preserve prescribed settings |
| Evaluation | Hidden rows and a different seed |
| Brief deadline | 2026-10-31 23:59 Asia/Almaty |

Intended pins: soup-cli0.77.0, transformers5.16.1, peft0.20.0, trl0.29.1, bitsandbytes0.50.2, accelerate1.14.0. The install command does not explicitly pin torch/Triton; brief expects torch2.11/Triton3.6.0. Installed runtime and dtypes still need verification.

Starter: c7c2fcd15972ce8b898d8d3ff732ee192c817ceb.  
Soup: 846888a045b31084452fa08a684a7a38c02f5f87.

## Rules

- Preserve model, data/batch order, steps, optimizer/schedule, LoRA including dropout, required metrics, and rewind recorder state.
- Fixed built-in modes: do not enable fast_lora, use_cut_ce, gradient_checkpointing, offloading, or packing.
- Original useful Triton JIT kernels under solution/ must cover >=5% of measured step wall time from step3 onward in every measured run. Use ordinary launches.
- Ready-made/copy kernels, starter SwiGLU even rewritten, and compiler-generated kernels do not count. Attribute external ideas.
- Scored command includes apply/loading/compilation/training/saving; each run starts with an empty Triton cache.
- First loss within1%; mean loss difference within3%; metrics/rewind within3%; token/row counts match; eval loss<=0.5% worse; time/memory<=10% worse.
- Run unchanged official harness/checks and Soup tests. No task/harness changes or precompiled caches.
- n-gram embeddings, sampled softmax, model/optimizer replacements, and changed schedules from speedruns alter this fixed task.

## CPU evidence retained from round 3

Audit used the exact pinned Soup mask function plus pinned Qwen tokenizer/template through a thin adapter. It reproduced formatting/truncation/masking, not the full Trainer or torch sampler.

- Real positions Nr=**52,890**; supervised shifted targets Ns=**33,608**; Ns/Nr=63.5432%.
- 16 rows truncated; median length140; median row supervised ratio42.6771%. Use token-weighted totals, not row medians.
- Response masking is active through Soup-provided labels/assistant_masks despite TRL flag defaults. Qwen’s no-generation-marker template uses Soup’s incremental fallback, including assistant prefix/suffix tokens.
- Dynamic right padding. Actual shuffled batch order unobserved.
- Source-order groups4: Np=83,700; Python shuffle seed42: Np=93,196. Both illustrative, not harness order.
- Recompute head condition Nr+2Ns<2Np requires Np>60,053. Sorted minimum53,580 is only an analytical bound; never reorder actual batches.
- Source-order head arithmetic ratios: selected-row cache0.5167, recompute0.7175. Python seed42: cache0.4641, recompute0.6444. **These are operation counts, not speedups.**
- Supervised-only fp16 cache: mean152.18MiB/batch; source-order max432.95MiB; maximum possible public grouping534.09MiB. Mean is not peak.

Dataset revision c641407ee50723c84a43ea2ec867b464d3893c12. Tokenizer revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306. Audit dependencies tokenizers0.23.3/Jinja2 3.1.6. Hashes, limitations, per-row counts, and reproduction script are in PUBLIC_WORKLOAD_AUDIT.json / PUBLIC_WORKLOAD_AUDIT_REPRO.zip.

## Round 4 numerical evidence

CPU torch2.11.0+cpu, two threads; original scripts independently rerun. `CPU_NUMERICS_ROUND_4.json` holds full results; `KERNEL_ROUND_4_REPRO.zip` holds runnable scripts, inputs, notes, and hashes.

- Loss: 174 comparisons (29 case/scale pairs x6 formulas), plus24 all-ignored edges. Exact rounded cache/integer metrics; zero entropy/per-row loss error; maximum scalar loss error4.77e-7.
- Per formula,27 finite cases +2 matching-overflow cases. Native ATen diagnostic and softmax-minus-onehot have0 finite failures; manual VJP6; saved uncentered LSE8; centered statistics6; deliberately wrong scale-after-half7. These are research gates, not official challenge results.
- Saturated margin18/scale65536: manual VJP12.5% peak-normalized gradient error. Offset1000 exposes uncentered-LSE cancellation. CUDA exp/reduction/fusion can differ from CPU.
- Comparator uses the same chunked fp16 head schedule on both sides; it does not prove original full-shape CUDA equivalence. True-V151936 test uses H16. First-order only.
- Empty batch: no denominator -> NaN loss/zero gradient; positive supplied denominator -> zero loss/gradient. Accuracy/per-row diagnostics0. Never clamp training denominator silently.
- LoRA: preserved graph matches7 gradients/9840 elements exactly; actual CPU fp16 autocast matches4/1848. Ten audit gates pass. External rounding boundaries do not fix different CUDA GEMM reduction trees.

## Head/loss design and metric contract

**H0:** preserve the native full-shape head and its backward; install an original loss/statistics operation on actual rounded fp16 logits, cached privately in autograd. Stream genuine fp32 loss/metrics. Initially use native backward as the oracle; include actual grad_loss and exact denominator before returning the fp16 logits gradient to the original head. Do not assume algebraically equivalent hand-written formulas match.

Share one transient compute-dtype frozen-head copy. Autocast caching excludes frozen weights; recasting once per chunk can retain large duplicate weights. Preserve the original fp32 tied parameter. Do not return the large half cache: Accelerate would upcast it. No hidden values are needed for frozen-head dH; a trainable-head caller requires fallback.

H1 retains only supervised logits, after computing entropy for every required real position, then optionally selects real head-forward rows within each unchanged batch. Keep separate dZ scratch initially; do not overwrite saved logits.

| Quantity | Contract |
|---|---|
| CE | Shifted labels!=-100; exact passed denominator, not mean of row means |
| Entropy | All attention-mask positions, including prompts and final tokens |
| Accuracy | Shifted supervised correct count / supervised count |
| Rewind | Original-row CE mean; zero-target row -> loss0/count0 |
| num_tokens | Cumulative attention-mask count |

Returning logits=None alone breaks TRL. New source finding: Rewind already consumes `trainer._soup_chunked_step.token_loss[B,T]` when logits are absent. No recorder rewrite needed. Own guarded SFT subclass delegates outer handling to HF Trainer, logs real summaries, and sets this attr. An instance-specific Qwen loss hook and forward wrapper omit logits only under that call's explicit context. Reconstruct ModelOutput so its mapping cannot retain logits. Keep summary until enclosing Rewind consumes it, then prevent stale reuse. Direct/eval/unsupported calls retain original outputs. Detailed guard contract is in the round4 bundle.

Round2 findings still apply: accuracy makes a shifted contiguous logits copy (up to1184.68MiB fp32); entropy already chunks128 rows. Full max-shape logits: fp16 593.5MiB/fp32 1187MiB; half head445.125MiB.

## LoRA work

Keep native dropout, original masks, base NF4 path, and reference casts initially. With X[M,K], A[r,K], B[N,r]:

    Z = dropout(X) @ A.T
    Y = Ybase + 2 * Z @ B.T

Mathematical derivatives alone do not reproduce dtype boundaries. Test nonzero B, complete gradients, and multiple AMP optimizer updates. Frozen base W still contributes dX.

- **L1:** start with one kernel per branch: expansion -> fp16 round -> fp32 cast -> native-mask dropout backward. Preserve incoming `R16(2*R16(G32))`, parameter-gradient fp16 output before upcast, independent masks, and separate fp32 dropout products. Keep existing autograd branch accumulation initially. Paired q/v is later.
- Mask multiplication must not become `where(...,0)`: false-mask*inf is NaN in the reference. Factoring dropout scale or contracting multiply-add can change rounding; fp32-to-fp32 cast is not a barrier.
- **L2:** dB needs T/Z; dA also needs U=T@B. Compare grouped U then4 weight gradients against grouped Uq/dBq/Uv/dBv then2 dA. No unsynchronized cross-CTA producer/consumer assumptions. A64×16 tile model gives76 output tiles; split4 partials1.1875MiB. Final half rounding still does not reproduce cuBLAS reduction order.
- Approximate maximum-shape LoRA arithmetic26.776GF vs12.645TF frozen dense-linears including head (~0.21%), with up to336 adapter GEMMs. Optimize launch/traffic overhead; this is not a runtime-share measurement.
- Shared fp16 casts can change gradient accumulation: separate upcasts of2048 and1 sum2049; shared fp16 sum rounds2048. Own backward conversions explicitly before sharing casts.
- q/v clones are temporary traffic; bnb already avoids retaining decoded forward weights/base inputs. Do not count those as new retained-memory savings.
- Mask packing preserves existing dropout decisions and could save up to147MiB mask storage at maximum shape; actual savings lower. Two saved half masked inputs equal one newly retained fp32 shared input.
- bnb already has T4 fused NF4 dispatch; these q/v shapes with M>64 generally choose dequantize+F.linear. NF4 fusion remains later.

Mixed dtype expectation: fp32 residual/base outputs, fp16 adapter GEMMs, fp32 adapter storage after Soup’s alignment. Verify; do not cache dtype-specific parameter copies before trainer setup completes.

## Attention, sampler, compilation, execution

Pinned SDPA Flash/cuDNN guards reject SM75; actual efficient-vs-math backend unobserved. New A0: convert existing bool causal/padding mask once to fp16 {0,-inf}, with physical row pitch ceil(T/8)*8 and unchanged logical shape, reuse across28 layers. Source conditions preserve dispatch for the same masked path. Max shared mask2MiB; potential54MiB retained-bias saving if28 separate efficient-backward biases become1, unmeasured. Preserve None/fallbacks. This small kernel's5% share is unproven.

Masked SDPA repeats2 KV heads to12. Direct repeat+half-store can avoid expanded fp32 buffers. Backward must upcast each expanded gradient before fp32 reduction of6 replicas. Optional RoPE remains fp32 until half store. Do not initially change the no-mask/GQA path.

Rewind wraps RandomSampler in a Sampler subclass before Accelerate. Its `isinstance(RandomSampler)` seedable conversion therefore does not match. Inner generator=None uses global CPU RNG. Minimal CPU diagnostic shows one extra draw changes order. Do not label torch.randperm(seed1) as actual harness order; preserve RNG and capture actual IDs.

Only64steps with fresh compile cache. Source-order public batches have48 distinct T; illustrative shuffles42–49, not actual trace. Prefer runtime M/T plus few structural configs; avoid autotuning every length. Added10seconds compile cost needs156.25ms average saving/step to break even (arithmetic only).

ALTO supplies a grouping idea, not T4 evidence; its LoRAFusion citation is a different paper from the earlier CentML reference. Use independently written kernels.

    python -m pytest solution -q
    python harness/run_task.py
    python harness/check_rules.py

Required Soup tests in the appropriate checkout:

    SOUP_APPLY_SOLUTION=1 python -m pytest tests -q

Guard fake-module setup paths and ordinary fallbacks. At session end update this file with checkout/commit, real runs and artifacts, rejected ideas, and next action. Once GPU experiments exist, record them in experiments.jsonl.

## Durable references

- Latest: LORA_RESEARCH_ROUND_4.md; CPU_NUMERICS_ROUND_4.json; KERNEL_ROUND_4_REPRO.zip.
- Prior workload evidence: LORA_RESEARCH_ROUND_3.md; PUBLIC_WORKLOAD_AUDIT.json; PUBLIC_WORKLOAD_AUDIT_REPRO.zip.
- [Round2 source/metric detail](https://chatgpt.com/library?file_id=libfile_87aa9af594748191be975dfe0559bdd8&account_id=personal)
- [Challenge brief](https://chatgpt.com/library?file_id=libfile_37b0abe044f88191b002d808e4d00570&account_id=personal)
- [Pinned starter and rules](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md)
- [Pinned task](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/task/task.yaml)
