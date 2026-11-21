# Trial001 measured result — correct within gates, insufficient own-kernel share

The original FP32 RMSNorm completed the unchanged paired two-repeat public benchmark on one visible Tesla T4. **Nine of ten official automatic checks passed. The only failure is useful own-kernel share: 1.20525%, below the required 5%. This standalone candidate is not admitted.**

| Metric | Base | RMSNorm trial |
|---|---:|---:|
| Whole-command times, two repeats | 85.38 / 81.87 s | 81.02 / 82.03 s |
| Harness median time | 83.62 s | 81.53 s |
| Maximum sampled GPU memory | 14,345 MiB | 14,343 MiB |
| Public evaluation loss | 2.04967 | 2.04964 |

Time ratio is 1.0256x, below the challenge's 5% gain floor; the 2 MiB peak difference is not a meaningful memory gain. The training-loop median was 58.08 s base versus 60.10 s solution, reinforcing that no robust improvement is established by this short comparison.

Both solution runs completed 64 steps with matching task/adapter/token/rewind structure and metric tolerances; first-loss gap 0.02%, mean loss-curve gap about 0.01%. Public evaluation and clean-measurement checks passed, as did all 79 Soup regressions. Actual eligible kernel launches were 3,648 forward and 3,584 backward in each run, consistent with 57 / 56 per step. Timed own-kernel totals were 669.909 / 670.938 ms across the 62 timed steps; respective shares were 1.21629% / 1.20525%. Raw full-run launch counts are not the timed-share denominator.

The actual T4 preflight passed the predeclared ordinary-case gates, and known extreme diagnostics remain recorded; universal bitwise equivalence is not claimed. No kernel was slowed or padded to achieve share.

## Evidence completeness

Exact downloaded `result.json` and `checks.json` are mutually hash-bound. The solution-folder hash independently matches the pinned starter files plus the three exact submitted original source files. Base standalone metrics and both evaluation JSON files are also present. The artifact host subsequently denied access, so standalone solution metrics and environment/preflight/session JSON files are missing. Embedded solution metrics in the exact result and the recovered execution log provide the corresponding evidence; individual per-case preflight arrays and the full archive are not claimed available. No alternative download route was used.

Result SHA-256: `129cc56516073b205375d9bbd93d94b0e2732b4a4e3e6f3f57a4090b7e9334fa`.

Machine-readable handoff: `trial001_verified_handoff.json`. No official jury/leaderboard or earned-score claim. Retain this negative standalone result while testing distinct useful candidates; do not rerun the same trial merely to seek a lucky ratio.
