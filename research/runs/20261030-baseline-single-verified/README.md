# Single baseline: raw-result verification upgrade

Evidence cutoff: 2026-10-30 14:24:58 UTC. This upgrades the existing **79.16 s baseline**, not a new run.

| Measurement | Observed value |
|---|---:|
| Whole training command | 79.16 s |
| Training loop | 55.89 s |
| Timed loop | 52.688 s |
| Sampled GPU peak | 14,345 MiB |
| Public evaluation loss | 2.04963 |
| Training steps | 64 |

The downloaded result bytes were independently hashed locally:
`77dbd8d77e72fedd1e5871aa0e4d611fd5bab73d81237ad342d1679fb2e66ea8`.

The reviewed evidence handoff has SHA256:
`f37a3877e13f1956682aa1c58b6b1fa9e9df5a896e0ff113c652dde81f53b258`.

**Twenty local evidence-validation checks passed. These are not 20 official points, candidate admission checks, or proof of 79 regressions.** The record establishes one baseline repeat with successful training/public evaluation, one visible Tesla T4, idle start, expected allocator, and consistent peak measurements. Starter pin: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.

## Limits

- Baseline/reference only: admission is **N/A**; all competition scores remain blank. No candidate gain, useful own-share, hidden-quality, or serial-versus-parallel conclusion follows.
- The standalone metrics file and complete archive are unavailable; there is no candidate checks artifact. Four starter tests are not the 79-test candidate regression suite.
- The earlier UI-observed total job duration of 243.9 s is a different timing domain from the 79.16 s training command.
- The separate **78.89 s interactive baseline** remains console-only; this result does not replace its measurements or verify its reported hash.
- This publication is a sanitized scientific summary only. Raw results, per-row/rewind arrays, execution logs, infrastructure metadata, private account/URL identifiers, and weights are not published here.
