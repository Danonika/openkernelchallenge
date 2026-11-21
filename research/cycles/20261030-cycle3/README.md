# Soup research cycle 3 — approved existing-ID refinement

**Jury cutoff: 2026-10-30 11:46:59 UTC.** Ten existing norm/residual/RoPE rows are refined; **zero IDs added**. The catalog remains 121 ideas with all original 104 IDs and source citations preserved. This is synthetic CPU/source research evidence, separate from the real one-repeat Kaggle baseline.

[Live Soup Ideas table](https://example.invalid/redacted). Exactly three visible tabs remain: Soup Ideas, blank Fusion Ideas, and Soup Runs for genuine measured runs. The graph stays outside the table. CPU fixtures create no graph points and establish no candidate T4 performance, admission or points.

## Decisions

- Keep guarded staged FP32 standalone RMSNorm **P1 conditional**. Isolated exact CPU tests do not establish the original residual-connected graph: width-1536 cancellation yields native 0 versus regrouped 1. Preserve unused `None` versus connected-zero gradients, aliases, version checks and actual branching.
- Residual fusion stays **P2**; recomputing the reduction statistic stays **P3**. Five saved references versus three are still the same three unique x/r/w storages, not a demonstrated activation-memory saving.
- Keep a narrow half-split FP32 RoPE route **P2 conditional**. Three source IDs map to one hypothesis, not three independently measured candidates. Preserve two multiply/add rounding boundaries, output layout, unequal query/key heads, coefficient broadcast and separate fallback paths. No generic inverse-adjoint, interleaved-layout or saved-reference-count shortcut is accepted.

**Six additional CPU fixture groups passed (three norm/residual, three RoPE), zero unexpected failures. Total: 23 evidence groups including the previous 17.** Deliberately failing unsafe variants are useful counterexamples, not challenge passes. The report states unvalidated actual prepared-model routing, CUDA kernels/reduction behavior, signed-zero/NaN-payload identity, full training quality, allocator peak and performance.

## Files and reproduction

`jury_cycle3_canonical_rows.json` contains the ten complete replacement verdicts, including every source citation and proof obligation. Apply them by exact `id` to reconstructed [cycle2 catalog](../20261030-cycle2/catalog.json); all other 111 verdicts remain unchanged. `jury_cycle3_compact_rows.json` is the jury's table-update view. `approval.json` records approval, unchanged counts and artifact hashes; only cloud-machine path fields were removed. `source_snapshot_manifest.json` binds original source files, including intentionally omitted duplicate full catalogs and intermediate logs.

`jury_cycle3_review.md`, `jury_cycle3_source_contracts.json` and the two research reports explain the reasoning. `cpu_tests/cycle3/` contains the exact original final script, results and summary. Earlier 17-group artifacts remain immutable in cycle2 rather than duplicated here.

Run `python research/cycles/20261030-cycle3/verify_snapshot.py` from a checkout for package hashes, exact reconstructed 121-row content, all 104 originals, artifact hash bindings, and six additional CPU outcomes. It performs no training or installation. To rerun CPU fixtures, copy `cpu_tests/cycle3/` to fresh cloud scratch with Python 3.12.14 / Torch 2.15.0+cpu, set thread variables to 2 and `PYTHONHASHSEED=0`, then run `python run_cycle3.py`. The script writes beside itself; do not overwrite immutable evidence.

Original approved source-manifest SHA-256: `403b8852a44198af948796169c809be117ea50ab3dd2544b8ab4ac2185ef35a0`. Full original final jury JSON SHA-256: `fd6e89f6d58d735499faf578a6a66b41a2e2cd902b972a8170f68fc66587fbb9`. Every reconstructed row is checked against a canonical hash in `manifest.json`. This publication is **research branch only**, without any changes to main, solution, harness, Fusion Ideas or graph configuration.
