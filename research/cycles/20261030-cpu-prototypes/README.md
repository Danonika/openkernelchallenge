# Bounded CPU implementation evidence

This snapshot summarizes two original CPU experiments for four **existing** catalog IDs. It adds no candidate GPU measurement, priority promotion, admission decision, score or graph point.

| Scope | Existing IDs | Result |
|---|---|---|
| [Materialized cross-entropy prototype](ce-prototype-summary.md) | SOUP-NEW-001, SOUP-NEW-002 | 28/28 frozen CPU engineering checks pass; finite rounding discrepancies remain |
| [NF4 decode + matmul reference](nf4-reference-summary.md) | L1-TILELANG-03, gemm-09 | 6/6 bounded CPU groups; 1,726 FP16 outputs match the declared scalar reduction order |

Passing these engineering checks is not official challenge acceptance. The CE prototype is **not bit-exact**. NF4 equality is confined to its stated CPU arithmetic and synthetic metadata, not native GPU behavior. Prior negative evidence and measured trial failures remain unchanged.

Only data-minimized scientific summaries and provenance hashes are published. Implementation/test source, executable commands, infrastructure/private account details, runtime logs, raw arrays and model weights are excluded. Full artifacts remain local; this is not a self-contained executable archive. The manifest records hashes of the reviewed local evidence without publishing its contents.
