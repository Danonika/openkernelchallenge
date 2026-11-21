# Shape choice and materialized CE — planning / bounded CPU proof

Jury-accepted NF4 cycle8 and materialized-CE cycle7 refinements, 2026-10-30. Existing IDs only: L1-TILELANG-03, gemm-09, SOUP-NEW-001 and SOUP-NEW-002. Priorities, actual trial outcomes and the 121-row catalog are unchanged.

- [NF4 shape/rounding summary](nf4-shape-summary.md): geometry does not identify a performance winner; moving scale after dot violates the declared exact FP16-weight contract in one finite witness, not automatically the contest rules.
- [CE contract/CPU summary](materialized-ce-summary.md): external native SUM/denominator handling and proposed thin NLL; 13 tested final-FP16 matches coexist with three FP32 discrepancies. No independent CUDA proof.

These are research/proof updates, not implementation or launch authority. There is no completed paired trial004 result to publish, no new run row or graph point, and no speed, memory, share or earned-score promotion. Environment interruption is not evidence of a kernel failure.

This package contains scientific summaries and source hashes only. It excludes code/builders, infrastructure/account/control-plane details, raw fixture or training-row arrays, credentials, private links and weights. Previously denied payloads are not retried. Prior [measured trial reports](../../runs/20261030-trials002-003/README.md) retain their evidence limits.
