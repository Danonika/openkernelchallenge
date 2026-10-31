# Fast-and-small Soup fusion from measured research — R30 / R31
Date: 2026-10-31 (Astana). **Research only, no official team submission.**

## Strict objective and reference
User-reported newest target: **38.49 s and 2,261 MiB jointly**, source not yet independently bound to exact public notebook/commit. Do not silently substitute older 71.85 s/14,345 MiB baseline.

Actual sheet-bound control (To Run row49): **40.00 s / 2,263 MiB / eval loss 2.04985**, 64-step complete. This is the attributed external contributor `ce62d67` architecture plus a separately authored input-only grouped-NF4 frozen MLP autograd. Imported competitor GPU kernels remain **third-party** and cannot be claimed as our original contribution. The service saves this optimized control as the baseline for rows 46–49. Consequently rows46–48 show **memory check failed** against 2,263 MiB despite better speeds: 32.94/3,461, 35.54/3,061, 36.03/3,463.

## Ranked ideas actually relevant to this baseline
Based on connected `Soup Ideas`, `Fusion Ideas` and `To Run` tables—not speculative sums of unrelated per-component gains:
1. **Already present in the control, not to duplicate:** vocabulary-major fused head/online CE+entropy+accuracy/Rewind, model-wide valid-token packing and native row-varlen SDPA, frozen tied FP16 embedding, fused RMSNorm+RoPE, layer-scoped one-launch NF4 dequant, q|k|v GEMM fusion, original LoRA dropout/RNG and host startup/sync work. These are the main sources of the competitor's speed, but their GPU kernels are *not ours*.
2. **Best measurable missing tradeoff:** input-only MLP autograd (our original) replaces saved gate/up with native gate/up recomputation during backward, preserving FP16 qgroup buffer lifetime. It gives ~2,263 MiB but increases command to ~40 seconds.
3. **Potential speed-only enhancement at same VRAM peak:** for sufficiently short actual packed batches, preserve normal MLP forward/backward (skip expensive rematerialization); for long batches retain input-only replay. The tradeoff is measured, not guaranteed. Public shape selection is based only on runtime tensor geometry, not hidden data or measurement status.
4. **Original grouped MLP backward VJP:** one GEMM for the two frozen gate/up adjoints instead of two, with an original Triton SiLU derivative and same saved narrow input; fallback for unsupported/split layers. The source is our independently authored `own_backward_packed_mlp.py` in research notebooks, not rival code.
5. **Peak-memory protection:** selectively split decoded NF4 weight lifetime only at outer layer boundaries (fwd=2/bwd=3), retaining byte-exact packed/decoded weights. Can cost extra kernel launches, so needs A/B.

Other table research already tested and not priorities here: direct NF4 MMA was slower, broad CPU activation offload timed out, unconditional MLP replay has a time cost, and extra standalone stats/LoRA kernels without useful activity are not score wins.

## Two NEW exact A/B notebooks
Both preserve unchanged `--only solution` measured training and report-export cells, correct 64 steps, dtype/seed, TRL/Soup metrics, LoRA dropout, optimizer. Their installer embeds pinned competitor source and labels it as attributed **research control**.

- **R30**: [Notebook](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/research_external-contributor_pin/benchmark_r30_adaptive320_grouped_vjp.ipynb); [queue row523](https://example.invalid/redacted). Request `research-r30-adaptive320-vjp-20261031`. Original grouped backward GEMM, native MLP when actual tokens <=320, full input-only replay otherwise, default NF4 split.
- **R31**: [Notebook](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/research_external-contributor_pin/benchmark_r31_adaptive384_grouped_vjp_split23.ipynb); [queue row524](https://example.invalid/redacted). Request `research-r31-adaptive384-split23-20261031`. Same grouped VJP, native short batches <=384, split decoded NF4 cache fwd2/bwd3 for memory.
- Existing other agents' adjacent studies: [HERO4 rows 516–520](https://example.invalid/redacted); three-band variants rows521–522. Their untouched data/queue results are independent.

## Acceptance decisions
Poll `To Run!I523:U524`, `X523:Z524`, `BS523:BS524`; use `VM Runs`/`VM Details` after terminal. A passing benchmark requires **complete** process, no post-training notebook failure, actual T4 memory/loss/timing, preserved all 79 regressions and required metrics/Rewind, useful original-owned >=5% share in each run, and correctness on another seed. The two new variants are **not** eligible original-only final submissions because inherited external contributor kernels remain third-party; replace with independently implemented kernels in the team's platform-created final repo before claiming competition eligibility.

Never invent benchmark results, combine the fastest time and lowest memory from two different candidates, override result cells, read hidden rows, or manipulate JIT caches/measurement settings. Competition deadline 2026-10-31 23:59 Astana.


## Newly verified winning control (source-bound)
The user-reported target has now been identified exactly at [To Run row518](https://example.invalid/redacted): **38.49 s / 2,261 MiB / public eval loss 2.04986**, full command completed. Exact research notebook: [adaptive input-only replay when packed N>448](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/research_external-contributor_pin/benchmark_adaptive_recompute_le448_v1.ipynb). Threshold 256 row519: 39.24 s/2,263 MiB; grouped backward row516: 38.89s/2,263MiB but notebook FAILED after measured training; combined threshold448+grouped backward row520: 38.67s/2,263MiB but notebook FAILED after training. These do not establish better-than-target completion. No copied competitor code may be claimed as ours.

### DIRECT source-bound threshold frontier (three new whole-task tests)
After locating the exact winning 448 control, submit only the following distinct variations; all keep our existing input-only grouped QLoRA MLP autograd, same 64 steps, unmodified harness and export:
- [R32 threshold512, default NF4 lifetime, row525](https://example.invalid/redacted). Notebook: [source](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/research_external-contributor_pin/benchmark_r32_adaptive512_inputonly.ipynb); request `research-r32-inputonly-cut512-20261031`.
- [R33 threshold576, default NF4 lifetime, row526](https://example.invalid/redacted). Notebook: [source](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/research_external-contributor_pin/benchmark_r33_adaptive576_inputonly.ipynb); request `research-r33-inputonly-cut576-20261031`.
- [R34 threshold640, reduced NF4 decoded-buffer lifetime at fwd2/bwd3, row527](https://example.invalid/redacted). Notebook: [source](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/research_external-contributor_pin/benchmark_r34_adaptive640_split23.ipynb); request `research-r34-inputonly-cut640-split23-20261031`.

All A:H written and verified with Ready TRUE last. The limit is **strictly under 38.49s and under 2,261MiB on one completed verified run**; simultaneous quality, 79 regressions and original-useful >=5% must also be established before any final claim. These attributed research notebooks remain ineligible as an original-only jury submission. If a threshold changes peak, compare memory by the same external GPU sampler and do not assume numerical equivalence implies score or timing improvement. The service's green time badge is NOT a scored speedup if the ratio <1.05.
