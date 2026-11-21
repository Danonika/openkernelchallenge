# Soup kernel challenge — session handoff

Updated: 2026-10-30 after research round 3. Read this first. Open LORA_RESEARCH_ROUND_3.md for the latest designs and evidence; round 2 retains the detailed metric/dtype contracts.

## Current state and next action

- Goal: accelerate the fixed Decentrathon 6.0 Soup Kernel Challenge run.
- Completed: three research passes, prescribed-version source review, and a reproduced CPU audit of public rows 0–255.
- **No solution kernels or T4 benchmark yet.** CPU token counts are measured; GPU speed/memory/quality gains are not.
- The authorized team submission checkout and available T4 environment remain unidentified. The public starter is not the team submission repository.
- User prefers Markdown for cross-session continuity.
- **Next:** untouched T4 trace -> metric-only baseline M0 -> private fp16 logits-cache experiment H0. Develop adapter input-gradient/dropout fusion L1 in parallel. Add supervised-row compaction H1 after H0 passes.
- Change from round 2: full head recomputation is now a later memory tradeoff; caching already-rounded fp16 logits avoids its extra projection and still respects GradScaler.

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

## New CPU evidence

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

## Head/loss design and metric contract

**H0:** cache actual rounded fp16 logits privately in autograd; stream fp32 LSE/loss/metrics. In backward form dZ=(grad_loss/D)*(softmax-onehot) in fp32, then cast dZ to fp16 and compute dH. Include the real upstream scale before the fp16 gradient boundary.

Share one transient compute-dtype frozen-head copy. Autocast caching excludes frozen weights; recasting once per chunk can retain large duplicate weights. Preserve the original fp32 tied parameter. Do not return the large half cache: Accelerate would upcast it. No hidden values are needed for frozen-head dH; a trainable-head caller requires fallback.

H1 retains only supervised logits, after computing entropy for every required real position, then optionally selects real head-forward rows within each unchanged batch. Keep separate dZ scratch initially; do not overwrite saved logits.

| Quantity | Contract |
|---|---|
| CE | Shifted labels!=-100; exact passed denominator, not mean of row means |
| Entropy | All attention-mask positions, including prompts and final tokens |
| Accuracy | Shifted supervised correct count / supervised count |
| Rewind | Original-row CE mean; zero-target row -> loss0/count0 |
| num_tokens | Cumulative attention-mask count |

Returning logits=None alone breaks the trainer. Coordinate real summaries with trainer/rewind, preserve FIFO/step/row identity, and retain ordinary logits for callers that need them.

Round2 findings still apply: accuracy makes a shifted contiguous logits copy (up to1184.68MiB fp32); entropy already chunks128 rows. Full max-shape logits: fp16 593.5MiB/fp32 1187MiB; half head445.125MiB.

## LoRA work

Keep native dropout, original masks, base NF4 path, and reference casts initially. With X[M,K], A[r,K], B[N,r]:

    Z = dropout(X) @ A.T
    Y = Ybase + 2 * Z @ B.T

Mathematical derivatives alone do not reproduce dtype boundaries. Test nonzero B, complete gradients, and multiple AMP optimizer updates. Frozen base W still contributes dX.

- **L1:** fuse A-backward input-gradient expansion, its fp16->fp32 cast, and dropout backward. Preserve each q/v branch’s fp16 rounding and independent mask. Two single-branch fused kernels may beat one paired kernel.
- **L2:** group dAq,dAv,dBq,dBv. A64×16-oriented tile model gives76 output tiles; four token-reduction splits need1.1875MiB fp32 partials. Test limited splits and stable reductions.
- Approximate maximum-shape LoRA arithmetic26.776GF vs12.645TF frozen dense-linears including head (~0.21%), with up to336 adapter GEMMs. Optimize launch/traffic overhead; this is not a runtime-share measurement.
- Shared fp16 casts can change gradient accumulation: separate upcasts of2048 and1 sum2049; shared fp16 sum rounds2048. Own backward conversions explicitly before sharing casts.
- q/v clones are temporary traffic; bnb already avoids retaining decoded forward weights/base inputs. Do not count those as new retained-memory savings.
- Mask packing preserves existing dropout decisions and could save up to147MiB mask storage at maximum shape; actual savings lower. Two saved half masked inputs equal one newly retained fp32 shared input.
- bnb already has T4 fused NF4 dispatch; these q/v shapes with M>64 generally choose dequantize+F.linear. NF4 fusion remains later.

Mixed dtype expectation: fp32 residual/base outputs, fp16 adapter GEMMs, fp32 adapter storage after Soup’s alignment. Verify; do not cache dtype-specific parameter copies before trainer setup completes.

## Other lead and execution

Pinned SDPA Flash/cuDNN hardware guards reject SM75. Efficient-vs-math backend remains unobserved. Masked SDPA repeats2 KV heads to12; inspect repeat/cast traffic. Forcing GQA can select the math backend.

ALTO supplies a grouping idea, not T4 evidence; its LoRAFusion citation is a different paper from the earlier CentML reference. Use independently written kernels.

    python -m pytest solution -q
    python harness/run_task.py
    python harness/check_rules.py

Required Soup tests in the appropriate checkout:

    SOUP_APPLY_SOLUTION=1 python -m pytest tests -q

Guard fake-module setup paths and ordinary fallbacks. At session end update this file with checkout/commit, real runs and artifacts, rejected ideas, and next action. Once GPU experiments exist, record them in experiments.jsonl.

## Durable references

- Latest: LORA_RESEARCH_ROUND_3.md; PUBLIC_WORKLOAD_AUDIT.json; PUBLIC_WORKLOAD_AUDIT_REPRO.zip.
- [Round2 source/metric detail](https://chatgpt.com/library?file_id=libfile_87aa9af594748191be975dfe0559bdd8&account_id=personal)
- [Challenge brief](https://chatgpt.com/library?file_id=libfile_37b0abe044f88191b002d808e4d00570&account_id=personal)
- [Pinned starter and rules](https://example.invalid/redacted-github)
- [Pinned task](https://example.invalid/redacted-github)
