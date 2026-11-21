# Resumed Soup research: exact sources and scoped results

**Local publication proposal only. A new explicit approval for this exact manifest is required before any external write.**

This is an additive research snapshot, not a final competition submission. It does not modify the previously approved 74-file snapshot at commit `2cc38f42683690a7fd3850063354a8c6d3c75f96`, under `testing_queue/candidates/20261031-held-offline`.

## Completed evidence included

| Exact candidate | Full training command | Sampled peak | Public evaluation loss | Scope |
|---|---:|---:|---:|---|
| Compact MLP + frozen storage | 73.692 s | 6,703 MiB | 2.04961 | One completed 64-step solution screen |
| Batched CE + direct final dZ on the same composition | 70.872 s | 6,503 MiB | 2.04963 | One completed 64-step solution screen |

These are separate screens, not a paired causal estimate or a stable-performance proof. Neither row is a jury-admitted result. No new baseline was run. Timed original-kernel shares are descriptive single-screen values, excluding both unchanged starter kernels. See the structured summaries for exact timings, source hashes, route counts and limitations.

Both full-task preflights retain four nominal-update numerical failures as `passed:false`, with the original values and thresholds. The existing explicit finite allowlist makes those observations warning-only; it does not turn them into passes or loosen any other check.

The third result is a completed **native-return shadow diagnostic**, not a training candidate: all 84 observed frozen-MLP adjoints met the local zero-padding eligibility test. Compact gate/up results showed small local numerical drift; all 28 down-projection observations had zero local error. The compact values were discarded, so propagated-gradient correctness, a finite-range certificate and production clearance remain unestablished. It has no full-task time, model-memory, quality-score or admission metric.

## Source artifacts

- `batched-ce-direct-dz-v1`: exact reviewed and executed CE source notebook, expanded payload, source manifest and scoped CPU/review evidence.
- `shared-cast-independent-edges-v1`: independently reviewed CE-parent derivative sharing forward half bytes while retaining independent adjoint cast edges. No target outcome is included in this snapshot.

The cast-edge variant is a distinct change on the verified CE parent, not a fusion of two untested changes. Its source and exact-byte review record defines its contract; its declared native checks and source-bound target outcome are not supplied here. The separately developing down-projection candidate is excluded from this bounded checkpoint because its final review is pending. `CANDIDATE_INDEX.json` binds the included identities.

Notebook authoring labels are preserved as historical metadata. Updated review/result receipts do not rewrite submitted notebook bytes, retroactively claim execution, or grant new execution/publication authority. Existing storage/shadow notebooks are referenced by their immutable source hashes and prior snapshot rather than duplicated here.

## Disclosure boundary

Result summaries are explicitly allowlisted selections, not raw logs. API job/request/account identifiers, private paths, raw runner/environment/host dumps, executed notebooks, weights, credentials, signed URLs and unrelated/external experiments are excluded. An internal local provenance map is not part of the archive. Exact reviewed candidate code retains its required pinned runtime/reference specifications; those are not a dump of a live account or job.

Only the new prefix in `PUBLICATION_MANIFEST.json` is proposed. Destination: private `project-repository`, branch `testing-queue`; a future publisher must fresh-read the destination, obtain approval for these exact bytes, preserve every existing file and verify the resulting remote commit. No queue launch, GPU call, GitHub write, visibility change or final submission is performed by preparation.
