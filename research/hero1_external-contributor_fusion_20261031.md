# Hero1 × external contributor — independent fusion research (31 Oct 2026)

## Verified source facts
Public competitor: https://github.com/external-contributor/soup-kernel-challenge
Latest inspected commit: `REDACTED` (31 Oct 2026).
Repository reports ~48 s, ~2850 MiB, and ~11.5% own Triton in README. However SUBMISSION.md still has placeholders instead of `runs/result.json`, Kaggle saved-version output, a score report, or complete paired measurements. **These numbers are unverified self-reports, not directly comparable to our public 71.85 s / 14345 MiB / eval 2.04968 T4 baseline.**

Source-level idea inventory:
- `fused_ce.py`: 151936-vocabulary tiled online stable CE + entropy + argmax, active-supervised-token backward, reused scratch, chunk 32.
- `nf4_mlp.py`: nested NF4 dequantization; shared scratch lifetime; input-only frozen Qwen2 gate/up/down rematerialization with optional fused gate+up GEMM.
- `rmsnorm.py`: per-row fused RMSNorm/adjoint.
- `rope.py`: fused Q/K RoPE plus optional sequence unpadding and attention checkpointing.
- `__init__.py`: attaches those patches in apply().

We do **not** copy competitor kernels. We also reject their unguarded model-wide frozen FP32→FP16 downcast as a direct import: Soup's baseline intentionally carries FP32 residual/normalization boundaries and hidden numerical parity is unproven. Flattened attention unpadding changes q/v LoRA dropout shape and potentially RNG; our candidate keeps native full-geometry dropout.

## Our independently authored fusion
Own module: `research/hero1_original_frozen_mlp_replay.py`.
Beast v2 notebook: https://github.com/Danonika/public/blob/master/soup-kernel-challenge/queue_20261031_hero1_external-contributor_replay4_nocache_share_v2.ipynb
Queue ID `hero1-external-contributor-original-replay4-20261031-v2`, To Run row **485**, graph `SOUP-FUSION-HERO1-015`.

Architecture: original NF4 Triton decoding **without** large full decoded-weight cache (aim to retain >=5% real original-kernel share); active-row q/o projections with **native full padded LoRA dropout RNG**; own valid-row post-attention RMSNorm + Triton fused residual/MLP scatter; preserved native half-logit CE head and Soup Rewind outputs. In exactly four frozen MLP layers (20,22,24,26), forward runs the same grad-enabled native path but discards its inner graph, retaining only input; backward replays the same operations under captured autocast with unchanged upstream scale. No task switch, parameter, adapter, optimizer, data, label, step or test changes.

The new forward replay guard explicitly preserves grad-mode dependent internal MLP patches. Frozen-parameter version stamps reject intervening weight mutation. Qwen2MLP type count and frozen three-projection checks fail safely to the original path.

## Independent correctness evidence (not GPU admission)
- CPU frozen MLP: exact forward and dX for FP32/FP16/BF16 across 9 cases, plus grad-mode-dispatch regression.
- CPU toy Q/V LoRA dropout + frozen MLP: **12/12** steps exactly matched forward, CE, trainable gradients, AdamW update tensors and RNG.
- CPU saved-tensor lifetime demonstration: after one synthetic 256×64 hidden frozen MLP forward, original autograd kept **737280 bytes** of tensors, replay kept **65536 bytes**; this is not measured T4 peak VRAM and includes synthetic dimensions.
- Notebook has three-seed synthetic Tesla T4 preflight plus real 64-step official solution-only harness and post-run own-original-Triton share audit; all pending.

## Pending hard gates
1. GPU service queue currently has not ingested new Hero1 rows. Wait for a fresh terminal VM Runs/VM Details record; a ready row is not a T4 result.
2. Require full 64-step completion, wall time, NVML peak, eval loss, same CPU/GPU tokenizer/data/steps, Rewind and TRL metric trajectories.
3. Require actual **>=5% source-classified, useful original Triton intervals**, excluding starter and compile-only work. Prior faster 58.78 s branch separately failed at **1.967%**.
4. Run all 79 Soup regression tests, pinned 10 official checks and paired Kaggle T4/hide-safe review before promotion. No invented score or hidden-row access.

Next after 4-layer result: change replay set to 8, 12 or 28 layers only if the speed/memory curve and 80s command cap justify it; do not infer additive gains from separate parent runs.
