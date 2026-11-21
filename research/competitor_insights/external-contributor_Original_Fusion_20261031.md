# Original external contributor + external contributor algorithmic fusion — 2026-10-31

**Never copy competitor kernel code.** This is an independent source-guarded research implementation. external contributor's published paired Kaggle T4 experiment reports 85.0→37.7 s, 14,345→3,461 MiB, +0.007% eval, 12.4% countable original-kernel time and 10/10 checks (not independently rerun here). external contributor's README describes compact frozen NF4 MLP and FP16 RMSNorm/output with approximate 48 s / 2,850 MiB, but no raw result/checks files were found, so treat the timing claim as unverified.

Our measured fastest development candidate was 58.78 s / 9,325 MiB (a follow-up original-kernel gate returned **1.967% <5%, FAILED**). Our lowest-memory completed candidate was 66.33 s / 5,005 MiB; its full admission remains unproven. None is a paired hidden-jury result.

## Original implementation and executed CPU evidence

We independently implemented:
- Within-batch valid-token packing with one **native causal SDPA call per original row** (no quadratic N×N additive mask). Preserves row-major order and RoPE indices, restores [B,T,H] outputs.
- Native LoRA dropout RNG over the original [B,T,D] geometry, then gathers valid positions. Unsupported/incorrect engaged SDPA geometry fails explicitly rather than allowing cross-example attention.
- In a separate original A/B variant, Qwen2 RMSNorm retains FP32 reductions/multiplication but stores its final output in FP16 under CUDA train autocast, using our own original Triton kernel and backward formula.

CPU reference oracles executed locally:
- 9 GQA attention shapes and dropout cases, forward max abs **4.7684e−7**, input-gradient max abs **1.9073e−6**, and exact CPU RNG end state.
- 9 RMSNorm FP16-output shapes, including 1427×1536 and 2048×1536, with exact FP32 analytical vs autograd gradient in the CPU oracle.

These are not GPU/QLoRA training evidence, and the original solution is not yet eligible or known to win.

### Three immutable original notebook experiments in the GPU queue

| Row | Request ID | Notebook | Last observed status |
|---:|---|---|---|
|490|chatgpt-external-contributor-own-segpack-20261031-v2|[segmented decoder source](project-repository/blob/master/soup-kernel-challenge/queue_20261031_original_fullpack_segmented_v2.ipynb)|Ready; no VM job |
|491|chatgpt-original-fp16-rmsnorm-lowmem-20261031-v1|[FP16 RMSNorm A/B](project-repository/blob/master/soup-kernel-challenge/queue_20261031_lowmem_rmsnorm_fp16_v1.ipynb)|Ready; no VM job |
|492|chatgpt-external-contributor-fusion-20261031-v3|[full fusion](project-repository/blob/master/soup-kernel-challenge/queue_20261031_combined_fp16rms_fullpack_segmented_v3.ipynb)|Ready; no VM job |

Sheets worker's latest visible status heartbeat remains 2026-10-31 10:27 UTC. Do not treat Ready as accepted/completed or duplicate the requests blindly.

All notebooks preserve original 64-step solution-only harness and exports; physical CUDA Qwen2/PEFT/dropout checks run in separate processes **after** training, plus a postrun original-kernel-time gate. We must still establish full 79 Soup regressions, 5% original useful-kernel time, native exceptional numeric rows, paired time/memory and hidden quality. Starter SwiGLU is excluded from own-kernel credit.

Sources: [external contributor](https://github.com/external-contributor/hackathon-decentrathon-6), [external contributor](https://github.com/external-contributor/soup-kernel-challenge), [official harness](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/main/harness/check_rules.py).

Full detailed report and executable CPU oracle tests are available as a downloadable research archive in the conversation.
