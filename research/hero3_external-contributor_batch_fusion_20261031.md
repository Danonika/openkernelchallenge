# Hero3 batch fusion research — 2026-10-31

## Source evidence (verified by reading linked repositories)
external contributor: https://github.com/external-contributor/hackathon-decentrathon-6
- README and current docs/final_numbers.json report Kaggle T4 at code commit `REDACTED`: median train wall **85.0 -> 37.7s**, NVML peak **14345 -> 3461MiB**, eval-loss delta **+0.007%**, original useful Triton share **12.4%**, official checks **10/10**, **432** stated tests. The saved Kaggle notebook is linked there; we have **not independently repeated** that run. 
- Source-described ideas: vocabulary-axis chunked CE/entropy/argmax and dH without full logits; exact FP16 tied head storage; original fused RMSNorm+residual, Q/K RoPE, layer-bounded grouped frozen NF4; padded-token removal with causal row-segment SDPA, preserving q/v LoRA native dropout randomness; fused q|k|v frozen base projection; cast minimization and host sync reduction. Their full MLP recomputation option is **OFF by default**, as it was measured to add about 25% training-loop time for ~1.2 GiB savings.
external contributor: https://github.com/external-contributor/soup-kernel-challenge
- Source at commit `REDACTED` develops compact frozen MLP saved activations, shared scratch, supervised-token-only head gradients, and per-row attention unpadding. README claims ~48s /2850MiB; the visible SUBMISSION.md has placeholder measurement fields, so **not established as a verified jury result**.

## Our measured context
Public VM T4 baseline: 71.85s /14345MiB /eval 2.04968. Our fastest 58.78s/9325MiB public screen is not a qualified winning candidate: a comparable branch's source-classified audit found only **1.967% eligible original Triton**, below the hard **5%**.
Another independent original NF4 watched branch completed 68.52s/6503MiB with 8.85% own source intervals; this is NOT transferable to another source without retesting. Never sum unrelated parent benchmarks.

## Independent original implementation — source (not copied)
- https://example.invalid/redacted
  - 8192-vocabulary-slice stable online log-sum-exp, centered entropy, first-index argmax, exact shifted target CE.
  - One true original Triton `hero1_slice_stats_inplace` computes useful logits statistics while overwriting each small FP16 logits slice with *unnormalized* exp scores. Frozen FP16 weight slice + cuBLAS accumulates weighted hidden gradient for only labelled rows, and one final original Triton kernel outputs token CE/entropy/predictions. Output loss is differentiable only through the frozen hidden gradient; GradScaler's runtime scale is applied BEFORE the returned hidden gradient is cast.
  - CPU mathematical oracle on four shapes including vocab 64, 512, 256, 1536: relative CE errors 0 to ~1.01e-7; gradient cosine essentially 1; maximum FP16 dx abs error <=1.22e-4. Not a T4 Triton correctness or performance result.
- https://example.invalid/redacted
  - Frozen native q|k|v base dequantization via the team's original NF4/bnb path, concatenate *temporarily* for one cuBLAS matmul and one backwards matmul. Intercept ONLY the frozen bnb base calls within Qwen2Attention; native PEFT q/v LoRA, dropout masks and RNG remain untouched. Unsupported type/quantstate/hook always falls back to original.
  - CPU randomized pure-PyTorch model of q/v LoRA dropout + grouped QKV: 12/12 cases pass, identical generator state, forward max error 0, max dX abs error 0.00390625 FP16 or 1.43e-6 FP32. This is a surrogate, NOT actual bitsandbytes T4 parity.
- Existing independent full decoder packed half-RMS/RoPE source from our team (To Run row493): native padded-size LoRA dropout bridged to packed active tokens, row-segmented causal SDPA, original NF4/MLP Triton.

## Batched GPU notebooks — ready, unmeasured
**Hero3-019 (safer):** https://github.com/Danonika/public/blob/master/soup-kernel-challenge/queue_20261031_hero2_ee_vocab_fullpack_batch_v1.ipynb ; To Run **row488**.
Combines full-decoder within-batch packed tokens; segmented SDPA; original padded LoRA RNG; FP16 T4 RMSNorm+RoPE; own NF4+MLP scatter; new vocabulary-sliced head with supervised gradient; unchanged native optimizer and 64 steps.
**Hero3-020 (aggressive):** https://github.com/Danonika/public/blob/master/soup-kernel-challenge/queue_20261031_hero2_ee_beast_fullpack_vocab_qkv_v2.ipynb ; To Run **row489**.
Adds grouped frozen Q/K/V GEMM to all Hero3-019 components. No q-only compaction overlay or copied competitor kernel. Uses synthetic physical-T4 qkv/head numerical oracles only AFTER measured training; keeps real 64-step solution-only harness/export, original-source 5% attribution check and all 79 Soup regressions.

## Important compatibility and admission risks
- Vocab head forward dH is accumulated in FP32 from FP16 exp matmuls, with loss scaling applied in backward. It is mathematically correct but not necessarily bitwise identical to the native FP16 dLogits GEMM route. The T4 synthetic reference, held-out loss, first-step and mean curves and Rewind row losses must pass unchanged numerical thresholds.
- Grouped QKV may change cuBLAS rounding or fail to activate with unexpected bnb hooks. A prototype's synthetic CPU pass is not proof that actual PEFT/bitsandbytes Qwen engages. The physical T4 oracle and full run must decide.
- Packed full-decoder requires right-padding/causal segmentation and original **B*T*D** q/v LoRA dropout masks; parent source has shape/attention/RNG and regression gates. Attention/dropout changes must never alter batch membership, order or target labels.
- Full task has 80-second wall cutoff including cold Triton compile and adapter save; do not prebuild compiler caches, alter harness/tests/task, use a second GPU or switch forbidden Soup flags.
- Official jury qualification requires 10/10 checks, **>=5% useful original Triton (excluding starter/compile-only)**, eval loss <=+0.5% baseline, first-step loss <=1%, mean loss/entropy/accuracy/Rewind gap <=3%, all 79 pinned Soup regressions, valid adapter and task provenance, then paired Kaggle T4 and code review. A completed public GPU screen alone is NOT jury admission.

## Execution blocker
The connected Sheets queue's most recent published **VM Queue -> Last poll UTC** was `2026-10-31T10:27:00Z`. At latest check, both Hero3 rows are `Ready — awaiting intake`; no terminal T4 time/peak/eval is available. Do not claim <37.7s or 3461MiB before real GPU evidence.
