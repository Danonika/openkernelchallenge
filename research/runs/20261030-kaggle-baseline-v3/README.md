# Kaggle T4 baseline v3 — one verified reference run

Verified at **2026-10-30 11:45:01 UTC**. The unchanged official public **base-only, one-repeat** run completed with **77.92 s whole training-command time**, **14,345 MiB sampled GPU peak**, and **2.04965 public evaluation loss**. It ran 64 steps on one visible Tesla T4; 256 public evaluation rows / 38,109 answer tokens / zero skipped rows. All 17 baseline evidence checks passed, and downloaded result, metrics, evaluation and checkpoint artifacts agree.

**Internal check: Pass for baseline/reference verification only — not candidate admission.** Competition score, speed points, memory points, admission / 20, README points and total points remain unknown. No candidate, relative speedup, relative memory gain or own-kernel share was measured. One repeat is provisional, not a stable performance distribution or full competition protocol. Four starter GPU tests are not the 79 Soup regression tests.

- [Current table: Soup Runs](https://example.invalid/redacted)
- [Owner-private baseline graph](https://soup-baseline-graph.contributor.chatgpt.site) — the same baseline node, never a candidate node.

## Reproducible evidence

`v3_verified_handoff.json` records hardware, versions, task/trainer/data provenance, tests, limitations and exact original artifact SHA-256 values. Private notebook/account identifiers, notebook URL and account quota usage are omitted from the published handoff and report; scientific values and source-artifact hashes are unchanged. `v3_verified_report.md` is the readable report. `artifacts/` contains the seven exact hash-checked metadata/result/checkpoint JSON files listed in that handoff. No model weights, dataset text, credentials, tokens, signed download URLs, runtime configs or unsanitized failure logs are included.

`validate_baseline.py` is the original read-only evidence validator. Run, from this directory:

```sh
python validate_baseline.py artifacts
```

Expected result: `status = provisional_baseline_validated`, `failures = []`, `full_admission = false`, `score_points_claimed = null`. This parses existing evidence; it does not train, install packages, authenticate to Kaggle or consume GPU quota. `validator_tests.json` records the validator's eight synthetic test outcomes; these are separate from the 17 actual baseline evidence checks and four starter GPU tests. `v3_artifact_consistency.json` records the exact cross-artifact comparisons.

Original official command (recorded, not re-executed by the publisher):

```sh
python harness/run_task.py --only base --repeats 1 --timeout 900 --out /kaggle/working/runs_baseline
```

Official starter: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. Public dataset revision: `c641407ee50723c84a43ea2ec867b464d3893c12`; training rows 0–255, evaluation rows 40000–40255. No hidden rows were used. The entire wrapper session took 214.631 seconds; that is not the 77.92-second training-command metric. The 54.04-second training loop is also a distinct metric.

The prior [cycle2 snapshot](../../cycles/20261030-cycle2/) remains immutable at its 11:31 UTC cutoff, before this baseline existed. Research idea count is still 121; baseline verification does not promote any candidate. `manifest.json` binds every file in this baseline snapshot. Publication is branch **research** only; main, the harness, solution and private notebook visibility are untouched.
