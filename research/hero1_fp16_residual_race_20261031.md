# Hero1 FP16 Residual Speed Race — 2026-10-31

## Verified reference
Our fastest completed T4 public screen is 58.78s / 9325MiB / eval 2.04966. A separate source attribution gate of a comparable fast branch gave 1.967%, below the hard 5%. Watched NF4 in another lineage completed 68.52s / 6503MiB with 8.85% source-classified original time; that result does not transfer to another branch.

external contributor publicly claims about 48s and 2850MiB. Its submission document still contains placeholders, so this is a research target, not yet an independently verified race result.

## New original experiments
- Hero1-016: training-only half-precision residual activations, with exact FP16 tied embedding and unchanged FP32 frozen RMSNorm weights. Qwen original data, LoRA, optimizer, steps and Soup metrics preserved. project-repository/blob/master/soup-kernel-challenge/queue_20261031_hero1_half_residual_nocache_v1.ipynb
- Hero1-017: same half residual with four selected frozen MLP input-only rematerialized layers. project-repository/blob/master/soup-kernel-challenge/queue_20261031_hero1_halfresidual_replay4_v1.ipynb
- Source: https://example.invalid/redacted

## Stop/go
Precision is approximate and requires a full 64-step T4 train/eval. Do not claim a faster or smaller result before terminal GPU evidence. Reject if eval loss > baseline +0.5%, first-step loss gap >1%, mean loss/entropy/rewind gap >3%, original useful Triton share <5%, or any of the 79 Soup regressions fail. Scoring only credits admitted solutions: speed 35, memory 30, admission 20, README 15. Competition settings, adapters, dropout, data and harness must not change. The sheet poll remains stale at 2026-10-31T10:27:00Z.