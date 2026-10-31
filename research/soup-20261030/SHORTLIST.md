# Filtered shortlist — independent implementation only

**Allowed method ≠ validated kernel. Conditional ≠ permission to change the task. All entries are unmeasured.** Source attribution is in master.csv/master.json and linked reports.

| Priority | IDs | Retained approach | First proof required |
|---|---|---|---|
| P0 | L4-RIGHTNOW-AI-AUTOKERNEL-02, L4-APACHE-TVM-03, L4-GPU-MODE-KERNELBOT-03 | Profile complete cold command; budget by end-to-end benefit | Versions, exact commit, compile/step/whole-command split; unchanged harness |
| P0/P1 | liger_castaware_norm, apex_rms_reduction, vllm_vector_norm, xformers_rms_cachepolicy, sglang_norm_boundary | Own FP32 cast-aware RMSNorm forward/backward, then residual fusion | FP32 residual/output, frozen affine, cast order, bypass gradients; no banned library import or copied kernel |
| P0/P1 | SOUP-NEW-001, SOUP-NEW-002 | Stream complete frozen head + CE + entropy + accuracy + rewind | Trainer-visible FP32 logits from rounded FP16 GEMM; global denominator, prompt/final entropy, row identities; GradScaler before FP16 backward rounding |
| P1 | SOUP-NEW-003 | Own fused nested NF4 scale+offset+decode, ordinary matmul | Actual bnb fallback/backward dispatch, NF4 not E2M1, all code values and tails, separate multiply/add rounding |
| P1 | SOUP-NANO-002, SOUP-COMP-001 | Avoid unnecessary scalar-value/alignment specialization | Actual installed API and compilation explosion; no prebuilt caches, normal launches |
| P1/P2 | SOUP-NEW-004 | Frozen head FP16 autocast-equivalent cache | Tied weights/version invalidation; construction and persistent memory count |
| P2 | SOUP-NEW-007 | Frozen reconstructed FP32 block-scale cache | Exact scales and invalidation; compare reads vs recomputation, construction not steady-step own share |
| P2 | L6-10, inductor_mem_bound_mm, L6-16, inductor_b2b_fusion | Own LoRA scale/add or skinny-matmul fusion | Keep separate q/v dropout and RNG, all gradients, original cast boundaries and scaler/clipping |
| P2 | SOUP-NANO-001 | Own AdamW arithmetic fusion with runtime scalars | Same optimizer/state/decay/epsilon/bias correction/overflow skip; likely small LoRA-only payoff |
| P2 | SOUP-COMP-002 | Control LICM/accumulator live ranges | Actual register/spill problem first; API availability, empty/tail-range compiled tests |
| High-risk conditional | SOUP-NEW-005, SOUP-NEW-006 | Exact frozen-value storage or within-batch unpadding | Exhaustive equality and boundary semantics, or exact original dropout RNG/rows/labels/positions; no blanket half()/packing switch |

## Excluded routes
- SOUP-NANO-X01: Muon/new architecture/loss/attention/task changes.
- SOUP-NANO-X02: ready-made attention downloader/FP8 route.
- L6-03/L6-07: CUDA-graph route (cost-accounting concept remains useful).
- SOUP-COMP-X01: background compilation overlap is held pending organizer clarification, not assumed legal from competitor text.
- Hopper TMA/WGMMA, Ampere cp.async, native FP8/BF16 and E2M1 kernels are not direct T4/NF4 implementations. Hardware exclusions remain distinct from rule exclusions.
- Paged decode, MoE and serving-batch policy have no direct role in this fixed dense SFT workload.

Full official banned import list and exact gate corrections: [audit](soup_classification_corrections_20261030.md). Research citations do not authorize installing/importing referenced projects, copying their kernels, or treating source-reported gains as team measurements.
