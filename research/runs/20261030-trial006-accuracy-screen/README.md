# Trial006 — accuracy-only T4 engineering screen

**Engineering screen PASS under79 s; not candidate admission.** Exact scope: the native accuracy-metric no-shift-copy component under existing idea **SOUP-NEW-002**. The historical entropy-only trial003 is a separate variant and its own-share failure remains unchanged.

| Observed quantity | Result |
|---|---:|
| Full training command | 77.55 s |
| Training steps | 64 |
| Sampled GPU peak | 14,345 MiB |
| Training loop | 54.19 s |
| Public evaluation loss | 2.04964 |
| Public evaluation coverage | 256 rows /38,109 tokens |
| Training /evaluation exit codes | 0 /0 |

One solution-only run on one process-visible **Tesla T4**, not a paired baseline comparison. The training command completed below the user's79-second engineering cutoff; this is uncensored command time, not total outer-job time.

The rendered training-log tail reported **64 optimized-accuracy calls and zero fallback reasons**. This activation evidence is console-observed; the raw training log was not independently downloaded.

## What this does not establish

- No paired baseline, speedup, memory improvement, repeated-run robustness or causal performance gain.
- No full trial006 regression suite, official admission comparison or hidden-quality result.
- **Zero original Triton-kernel share**: this native-only component cannot meet the own-kernel admission gate by itself. No earned competition scores.
- CPU method controls and completed public evaluation do not imply full model bitwise parity or admission. Broader CE/shared-head variants remain untested.

## Verified evidence

Exact local result bytes independently verified: SHA256
`af8861634ab3fe39edae1dace96204b962a8c74d34aca69cb7c4ccf7873740fc`.

Exact screen-summary SHA256:
`e13cd6efe71d36d1f6bb39d5a743567c97290a78dcb5355be1e900f96fd4de0b`.

This publication is aggregate scientific evidence only. Raw logs/results, execution source, account/session/cache metadata and weights are not included.
