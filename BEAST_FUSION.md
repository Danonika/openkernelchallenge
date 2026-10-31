# Beast fusion — direct Soup solution in the team's repository

**Implementation branch:** `soup/t4-integrated-optimization-20261031`
**Team repository:** `BAITC-Hacks/hack-57faaa42-duo`
**Entrypoint:** `solution/__init__.py::apply()` with zero arguments.

This is a **real integrated implementation**, not a standalone research notebook.
The challenge jury only copies `solution/` from the official team repository.
This branch is an experimental candidate until full Kaggle T4 verification.

## Algorithmic foundations and implementation boundaries

The relevant primary research is documented in [`solution/TECHNICAL_REFERENCES.md`](solution/TECHNICAL_REFERENCES.md): online normalization, Cut Cross-Entropy, Liger Kernel, QLoRA/NF4, LoRA, efficient attention and variable-length packing. These are technical references, not claims that any published software package was imported.

The listed optimization methods are established algorithmic foundations. Reviewing research examples does not establish that a kernel in this branch was copied from another solution. Assess authorship and eligibility from the actual implementation, its development history, and the challenge's rules.

The team's `nf4.py`, `rmsnorm.py`, `rotary.py` and `hero3_*.py` files are included as candidate implementations. Independent authorship and competition eligibility are subject to comparison with the original sources, not established by a comment or refactoring.

## What `apply()` really enables

The modules are installed before Soup builds the model, with the same patches used for training and evaluation (no benchmark-/seed-specific branching):

1. `nf4.py`: original nested NF4 dequantization and native bitsandbytes GEMM.
2. `frozen_storage.py`: verify that a frozen tied FP32 embedding/head stores only exactly representable FP16 values, then reuse the exact FP16 representation.
3. `rmsnorm.py`: use our original FP32-accumulating Triton RMSNorm but write FP16 outputs on the guarded T4 autocast training hot path. FP32 frozen scale values remain intact.
4. `rotary.py`: use original Q/K forward/backward Triton rotation with correctly typed FP16 outputs and gradients.
5. `integration.py` + `hero3_vocab.py`: fully participate in TRL/Soup’s existing training/Rewind metric contract; stream the 151,936-token vocabulary in 8,192-row slices, using original online CE, entropy, argmax and labelled-only gradient kernels. The source streams weight casts one slice at a time if frozen storage is still FP32, without an additional whole-vocabulary FP16 copy.
6. `hero3_pack.py`: compact valid tokens **only within the current batch**, preserve each original example’s causal attention segment, reconstruct padded-shaped LoRA dropout masks using the **native** CUDA Philox operation and scatter back to the original `B×T×D` positions. The wrapper around labelled Qwen2ForCausalLM.forward is intentionally installed **after** the Soup/TRL head hook: that order is necessary to pass labels into the packed Qwen2Model.forward path.
7. `hero3_qkv.py`: independently fuse frozen quantized Q/K/V base projections into one GEMM and one input-gradient GEMM, leaving native PEFT Q/V adapters and dropout untouched. Unexpected quantization/hook/module states take the native path.

We deliberately do **not** enable full frozen-MLP recomputation by default. A prior public benchmark, identified in `PROVENANCE.md`, reported about +25% training-loop time for an additional 1.2 GiB saving; this trade-off needs independent measurement on the final candidate.

## Testing; no results invented

The current commit and direct unit tests are **source-complete but not GPU-admitted**.
The connected Soup Google Sheets T4 queue’s latest published poll was `2026-10-31T10:27:00Z`; no fresh Beast candidate T4 completion is available as of this commit.

Use the repository’s pinned task and standard commands on an eligible single Kaggle Tesla T4:
```bash
python -m pytest solution/test_beast_fusion.py -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

**Admission requirements:** all ten `check_rules.py` checks (including 79 exact Soup tests), useful source-classified original Triton kernels ≥5% of timed steps, unchanged LoRA/data/optimizer/Soup switches, first-step loss within 1%, mean loss/entropy/accuracy/Rewind drift ≤3%, eval loss no higher than baseline +0.5%, per-command time ≤80 s, and independent source/code review.

Do not merge this experimental branch as the official submission or quote <37.7 s or <3,461 MiB until a complete T4 run passes these gates. If any component fails, investigate the one integrated source/metric path, not a new standalone notebook side project.
