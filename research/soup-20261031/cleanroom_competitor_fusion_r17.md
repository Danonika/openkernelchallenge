# Soup Kernel Challenge — competitor ideas & independent tests (2026-10-31)

**Clean-room boundary:** only high-level algorithm ideas were studied. No external contributor or external contributor kernel source was copied. This note does not change `solution/`, `task/`, or the final submission branch.

## Sources (self-reported, not independently re-run)

- [external contributor](https://example.invalid/redacted-github): repo README/docs claim paired Kaggle T4 **85.0 → 37.7 s**, **14,345 → 3,461 MiB**, eval loss +0.007%, 10/10 checks, 12.4% useful-owned kernel share at code commit `REDACTED`. Highest-leverage concepts: **whole-decoder unpadding with truly variable-length row-isolated attention**, vocabulary-major head/online CE+entropy+accuracy+Rewind, ephemeral whole-layer NF4 decode, exact compact frozen tied weight, host startup/metric-sync reductions. Sources: [unpad](https://example.invalid/redacted-github), [head](https://example.invalid/redacted-github), [NF4](https://example.invalid/redacted-github), [host](https://example.invalid/redacted-github).
- [external contributor](https://example.invalid/redacted-github): README has *approximate*, not independently verified, 48 s/2,850 MiB figures, and SUBMISSION.md placeholders. Conceptual additions: save only the frozen MLP input for backward and stream CE/metrics; prioritize correctness/latency before porting those ideas.

## Our team's public development control

- Saved service baseline: 71.85 s / 14,345 MiB.
- Fastest completed screen: 59.50 s / 9,125 MiB; lower-memory completed screen: 64.83 s / 5,345 MiB. **Different candidates, not a paired jury cohort.** Strict target is below 45 s **and** below 1,359 MiB; not achieved.

## New original experiments (preserve team training harness and exports)

**R17** [original whole-decoder packed Qwen2 candidate](project-repository/blob/master/soup-kernel-challenge/queue_20261031_cleanroom_decoderpack_r17.ipynb) — [Google Sheets row 83](https://example.invalid/redacted), request `chatlora-cleanroom-unpad-20261031-r17`. Independent `Qwen2Model.forward` wrapper gathers right-padded tokens to [1,N], uses an explicit Boolean block-diagonal causal attention mask, and scatters back for native loss/metrics. Four CPU test groups pass (nine padded patterns included); physical T4 and tiny Qwen2 preflight run before training. **No T4 performance claim until terminal job.**

**Critical R17 risk:** A dense block mask need not skip masked cross-row multiplications. Lengths [512,340,140,52]: padded pairs 1,048,576; packed-dense pairs 1,089,936; true variable-length pairs 400,048. May be slower. Do not confuse this with external contributor's varlen implementation.

**R18** [original pinned PyTorch native-varlen research](project-repository/blob/master/soup-kernel-challenge/queue_20261031_cleanroom_varlen_probe_r18.ipynb) — [queue row 84](https://example.invalid/redacted). The existing measured storage+MLP run/export are unchanged; a *post-training* microbenchmark tests native `aten::_efficient_attention_forward/backward` on T4 against dense SDPA, logs forward/grads and event time to `SOUP_RESULTS/native_varlen_t4_research.json`. No useful-original-kernel/admission claim from microbench.

**Next:** if native T4 varlen is supported and numerically sound, implement a fresh original causal/GQA autograd path and integrate with the team's own full decoder packing, not rival code. Separately research vocabulary-major CE+Rewind and scoped NF4 materialization. Validate GradScaler, LoRA dropout, seed drift, all 79 regressions and >=5% own-kernel share. Never use hidden data, measurement evasion or prohibited libraries. Final team code belongs in the platform-created repo before **2026-10-31 23:59 Asia/Almaty**.
