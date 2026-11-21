# Trials002–003 — verified scientific reports only

**Both paired T4 experiments completed, but neither is admitted standalone.** In each execution console, nine of ten official automatic checks passed and only useful own-kernel share failed. All 79 unchanged patched Soup regressions and public quality checks passed. Hidden quality and final organizer acceptance remain UNKNOWN; no earned score is claimed.

| Tested scope | Base → solution median reconstructed from logged repeats | Max peak | Public loss base → solution | Log-rounded own share | Printed speedup |
|---|---:|---:|---:|---:|---:|
| trial002: backward-functional nested NF4 decode |84.455 →85.235 s|14345 →14345 MiB|2.04967 →2.04968|2.5%, FAIL|0.991x|
| trial003: materialized-logit entropy helper only |86.065 →84.395 s|14345 →14345 MiB|2.04963 →2.04965|1.2%, FAIL|1.020x|

Useful own share must reach 5%; the exact fractions are unavailable. Trial002's 1.009x is solution/base time growth, not speedup. Trial003 is below the 1.05 scoring floor. No robust speed/memory gain or earned points are established. Separate standalone shares cannot simply be added to admit a combination.

## Published evidence

- [Trial002 report](trial002/trial002_verified_report.md) and [handoff](trial002/trial002_verified_handoff.json).
- [Trial003 report](trial003/trial003_verified_report.md) and [handoff](trial003/trial003_verified_handoff.json).
- [Jury-approved two-row overlay](review/canonical_rows.json) and [approval/hash bindings](review/approval.json), cutoff 2026-10-30 13:15:28 UTC.
- [Published-file SHA-256 manifest](manifest.json).

Only existing SOUP-NEW-003 / SOUP-NEW-002 change. The 121-row catalog, all 104 original IDs, conditional verdicts and P1 / P0/P1 priorities are preserved. Broader forward/GEMM NF4 and full-head/main-CE/custom-gradient variants remain untested. Historical CPU failures are retained.

## Evidence limits

The completed standard execution console established emitted source hashes, nested success, four logged repeat values and all ten checker lines. **Exact raw result/checks files were not downloaded** after an artifact-host policy denial. Their hashes in the handoffs are wrapper-reported, **not independently byte-verified**. Individual preflight arrays and the complete archive are unavailable. This package does not fabricate replacements or claim complete raw reproducibility.

Immutable reports/handoffs retain historical “publication paused” wording from their cutoff; this page records their subsequent approved report-only publication. Trial002's then-pending trial003 observation is superseded by the separate final trial003 report. The source packets were checked locally against emitted hashes, but **implementation files, builders, generated runners and execution instructions are not published in this directory**. Those packets remain local; the denied source payload was not retried or moved elsewhere.

Both sessions reported clean per-run GPU measurement. Accepted overlap does not prove zero timing interference; physical GPU identities/exact overlap and a same-candidate serial-versus-parallel control were not captured. Repeat spreads and small ratios do not establish causality.

The published manifest authenticates only the files actually present; it does not independently authenticate unavailable raw results. Full raw/control-plane logs, private account/notebook/launch/quota/site/deployment identifiers, infrastructure/native-library detail, credentials, signed URLs, model/adapter weights and caches are omitted. No new run, submission, visibility, merge or main-branch change is performed.

For interpretation use the [project admission checklist](../../../../.agents/skills/soup-jury-admission/references/CHECKLIST.md). Public checks remain preliminary, not official jury acceptance.
