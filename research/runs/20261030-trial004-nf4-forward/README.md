# Trial004 — measured forward-only NF4 decode, standalone admission FAIL

Existing catalog ID: **SOUP-NEW-003**. Variant: original nested-NF4 forward fallback decoder, retaining native GEMM and native backward. This is distinct from the historical backward-only trial002 and does not test a fused NF4/GEMM design.

Actual Kaggle Tesla T4, one paired cohort with two baseline and two candidate repeats, unchanged pinned harness, 64 steps per run. Evidence cutoff: 2026-10-30 15:04:28 UTC.

| Measurement | Paired baseline | Candidate |
|---|---:|---:|
| Full-command repeat times | 83.02, 88.19 s | 88.47, 91.04 s |
| Mean (=median for two stored repeat values) | 85.605 s | 89.755 s |
| Maximum sampled GPU peak | 14,345 MiB | 14,345 MiB |
| Public evaluation loss | 2.04965 | 2.04968 |

Base/candidate speed ratio: **0.953763** (candidate is slower); candidate/base time growth: 1.048478. Memory ratio: 1.000. The three-decimal aggregates above are computed from stored two-decimal per-run wall times, not additional timer precision; the harness display rounds these aggregates.

**Nine of ten preliminary checks pass. The only failed check is useful own-kernel share:** 2.7448338809784594% and 2.819212266503628% across the two solution runs, minimum below the required 5%. Thus the standalone variant is **not admitted**, despite passing public numerical/task/resource checks and 79 regressions. All competition score fields remain blank; no earned points or robust gain are claimed. [Derived check summary](check-summary.json).

## Provenance and limits

Exact downloaded raw result SHA256: `357ad807adfa279d122e08b878f5d54a3dd333788952354c43ed3ddfcf5065c4`.

Exact downloaded raw checks SHA256: `ea011a55223fc82ddb68787f109bbb96640ab6a2a3f82364a317ceeaaf73cd5d`.

Reviewed handoff SHA256: `2ec130870dda4be402f3b8a0d9a61289dd76f07b6b456ad500294a8dbb754cc0`. Official starter pin: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.

Result, checks, environment and preflight artifacts were recovered; separate per-run metrics files and raw training logs were not. Per-run metrics are embedded in the exact result. This report publishes only aggregate scientific evidence and a derived check summary, not source/builders, raw arrays/logs, infrastructure or account identifiers, caches or model weights. It makes no model-cache persistence/export claim.

This is one cohort; baseline repeat times drifted 83.02→88.19 s. No causal hardware-contention conclusion follows. Public evaluation is not hidden admission. Full-run launch counts include untimed steps, so dividing timed milliseconds by those counts does not establish kernel latency.

Historical trial002 remains recorded separately. This failure does not ban all NF4 designs or justify an unchanged rerun, artificial workload inflation, or adding predicted kernel shares. The user's later 79 s engineering stop policy is prospective only and does not rewrite this completed result or create an official disqualification.
