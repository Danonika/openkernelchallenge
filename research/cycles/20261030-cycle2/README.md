# Soup research cycle 2 — closed evidence snapshot

**Cutoff: 2026-10-30 11:31 UTC.** The source is immutable `cycle2_closed_20261030T1131Z`, not the concurrently evolving working files. This snapshot contains **121 unique idea IDs**: the original 104, 14 earlier additions, and 3 cycle additions. Every original ID and source citation is retained. No GPU training, benchmark, quality, kernel-share or official-score result is claimed.

- [Current research table — Soup Ideas](https://example.invalid/redacted): 121 catalog rows; Fusion Ideas is blank; Soup Runs has headers only. These are the three visible tabs. Historical Muse tabs remain hidden.
- [Baseline graph, outside the table](https://soup-baseline-graph.contributor.chatgpt.site): owner-private deployment; one baseline configuration bucket / subgraph represented by one point. No candidate nodes, edges or measured runs. Admission and earned points are unknown, not 20.
- [Baseline graph source](../../../graph/baseline/) and [initial 118-row research / attached ТЗ](../../soup-20261030/) are separate from this closed cycle.

## What is preserved

`compact_table.json` is the readable 11-column table view, regenerated from the final jury snapshot rather than the earlier 11:29 table export. `catalog.json` preserves **every field of every final jury verdict**, including citations, CPU artifact hashes, exact missing proofs and stopping conditions, without repeating common values unnecessarily: reconstruct each verdict as `{**catalog.defaults, **row}`, then remove the added `bucket` field. `jury_review.md` is the full final review; `jury_snapshot_manifest.json` records original immutable-source hashes. `jury_validation.json` describes that original source snapshot; its references to `jury_verdicts.json`/CSV are source names, not additional copies in this folder.

Five administrative categories, without changing legality or applicability:

| Category | Ideas |
|---|---:|
| Memory / data movement / compilation | 68 |
| NF4 dequantization / GEMM | 25 |
| Head / loss / metrics | 8 |
| RMSNorm / activations | 13 |
| LoRA forward / backward | 7 |

Research reports retain source-linked LoRA and training-boundary reasoning. Final CPU results, summaries and their exact original scripts are included; transient logs and duplicate partial runs are omitted.

## CPU evidence, not challenge passes

**17 original synthetic CPU fixture groups passed (8 + 6 + 3), zero unexpected failures.** Some groups deliberately demonstrate failure of unsafe variants. This is numerical/semantic evidence only, on Python 3.12.14 and Torch 2.15.0+cpu (Torch source commit `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`), with CUDA unavailable.

- `cpu_tests/`: causal/global metric bookkeeping, centered reduction and scaled-gradient witnesses, compact head backward, norm/residual bypass, synthetic NF4 transform, LoRA cast boundaries and native-mask codec.
- `cpu_tests/cycle2/`: six centered-merge/nonfinite fixtures; unconditional sparse zero-scatter and naive nonfinite gradient rewrites have counterexamples. Centering alone does not guarantee near-zero relative-error gates.
- `cpu_tests/cycle2b/`: three LoRA B-tail ownership / selective saved-mask-hook / CPU-autocast fixtures. Ordinary Torch custom autograd is not GPU launch fusion, and codec payload arithmetic is not measured memory savings.

No actual bnb QuantState, PEFT/CUDA dropout route, CUDA GradScaler, SM75 kernel, full model quality, T4 time/peak memory, own-kernel share or competition gate has been validated here. Candidate priorities remain conditional; no promotion follows merely from CPU PASS.

## Reproduce and verify

Run `python research/cycles/20261030-cycle2/verify_snapshot.py` from a checkout to verify all packaged SHA-256 values, 121 IDs, preserved original 104, 17 fixture counts, CPU script hashes, artifact references, and baseline-only graph invariants. This verifier does not train, install packages or invoke a GPU.

To rerun fixtures in an appropriately provisioned **cloud CPU** environment, first copy each `cpu_tests` directory to a fresh scratch location: the original scripts write `results.json` and `summary.json` next to themselves. With the recorded Torch build and Python, set `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=0`, then run `python run_cpu_semantics.py`, `python cycle2/run_cycle2.py` and `python cycle2b/run_cycle2b.py` in that scratch copy. Do not overwrite this immutable evidence. Exact timestamps change; CPU fixtures do not establish GPU behavior.

## Scope and provenance

Pinned official harness: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. The original attached ТЗ describes averaging; pinned code aggregates median time, maximum sampled peak and first successful adapter evaluation. This discrepancy stays explicit; no official ruling is invented. The packaged graph retains the live source's initial 118-row catalog provenance, while this newer closed catalog has 121. Sanitization only replaces absolute cloud evidence paths with repository links; it does not deploy or change the live site.

Publication is restricted to branch **research**, this cycle directory and `graph/baseline/`. No solution, harness, main branch, Sheet or live-site mutation is part of this publication. No credentials, `.git`, `.openai`, runtime state, private notes or older proposal nodes are included. `manifest.json` binds the exact packaged files; the original jury JSON SHA-256 is `295e6d5f995c2110e532339f4682a75756e062464f6f0dcdb1f79df75ed2ef52`.
