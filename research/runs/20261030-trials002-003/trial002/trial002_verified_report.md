# Trial002 NF4 — numerical gates pass, standalone share fails

**Result:** the original backward-functional NF4 decoder completed the unchanged paired two-repeat benchmark. Nine of ten official automatic checks passed; only useful own-kernel share failed (**2.5%**, required5%). Standalone admission fails; no earned-score claim.

| Metric | Base | NF4 solution |
|---|---:|---:|
| Whole-command repeats |86.70 /82.21s|85.28 /85.19s|
| Median from raw log values |84.455s|85.235s|
| Maximum sampled memory |14,345MiB|14,345MiB|
| Public eval loss |2.04967|2.04968|

The time ratio is about0.991x, roughly0.9% slower; no speed or memory gain is established. Both training and evaluation sides succeeded. Source hashes match the sealed packet. The exact preflight passed16 real-state cases,4 full-table/tail cases,3 nonfinite cases,8 native gradient/cast cases and4 fallback cases. All79 unchanged patched Soup regressions passed.

## What this says about two parallel sessions

The account accepted the second separate session while this one reported RUNNING. This trial's clean-measurement check passed: actual T4, one visible GPU, idle start, unchanged allocator and external/internal peak agreement. No simultaneous competing base/solution runs were placed on two GPUs inside one notebook. However, physical GPU UUIDs were not captured, and trial003's own evidence is still pending.

Base repeat spread was4.49s, about5.32% of its median; solution spread was0.09s. This shows why a tiny time ratio cannot be treated as robust. It does **not** prove concurrency caused the variation or that concurrency has zero effect. A same-candidate serial-vs-parallel control and more repetitions would be needed to isolate that effect.

## Evidence limits

This handoff verifies the completed standard execution console, emitted source hashes, nested terminal success, preflight summary, all four run measurements and all ten checker lines. Exact raw result/checks and complete individual preflight arrays are not available; no denied artifact-host retry or alternate route was attempted. The result hash **reported by the executed wrapper** is `20b63d9abad9f62ab97a6678f10d62bd66ad884911fb8b0e91d1ef2e7f924854`; it is not claimed independently verified against downloaded bytes. No archive-completeness or final-jury claim. Publication remains paused.
