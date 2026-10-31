# Cycle5 — historical baseline planning, not a measured candidate

Jury accepted these report contents at **2026-10-30 12:27 UTC** as notes on nine existing IDs, not canonical verdict/priority replacements. The 121-row catalog and 104 original IDs remain unchanged. No new CPU fixture group, GPU run, graph point, priority promotion, candidate gain or score is asserted here.

**Later evidence takes precedence:** [trial001](../../runs/20261030-trial001-rmsnorm/README.md) completed the paired norm benchmark and failed standalone useful share at 1.20525%; 9/10 automatic checks passed. Its 1.0256 time ratio is below the scoring floor. These pre-trial scenarios must not overwrite that result.

- [Owned-share planning](research_share_cycle5.md): useful norm/NF4/LoRA payload models and explicitly hypothetical bandwidth sensitivities. Actual share must use eligible event milliseconds divided by the actual solution timed loop, minimum across runs. No padding or slowdown to reach share.
- [Baseline metadata arithmetic](research_baseline_cycle5.md): attention-mask and supervised-token denominators, source-conditioned padding bounds and row-work arithmetic. No exact padded shape, kernel-time partition or candidate speed is inferred.
- [Share notes](research_share_cycle5_updates.json) and [metadata notes](research_baseline_cycle5_updates.json) are annotation-only. The live sheet retains cycle4 canonical rows plus the independently reviewed actual trial001 overlay.

The nine annotations are on `liger_castaware_norm`, `SOUP-NEW-003`, `SOUP-NEW-007`, `SOUP-LORA-001`, `SOUP-TRAIN-001`, `SOUP-NEW-001`, `SOUP-NEW-002`, `SOUP-COMP-001`, `SOUP-COMP-002`. Historical prose such as “measure trial001” is preserved with an explicit superseding-result banner.

## Reproduce without GPU or installation

From this complete research checkout, using standard-library Python only:

```bash
python baseline_metadata/analyze_v3.py
python share_cycle5/estimate.py
python verify_snapshot.py
```

Scripts read the already published [baseline-v3 artifacts](../../runs/20261030-kaggle-baseline-v3/artifacts), not model weights or training text. Public row IDs and aggregate counts are metadata, never a dispatch key. Source-conditioned longest-row/no-padding-multiple assumptions stay explicit. The derived numbers were reproduced exactly; only provenance paths/hashes changed for portability.

Private cloud paths were replaced by relative artifact paths. Private account/notebook/quota/site/deployment metadata, infrastructure/native-library fingerprints, credentials, signed URLs, model/adapter weights and full logs are omitted. `manifest.json` distinguishes original approved source hashes from published sanitized hashes. No immutable earlier snapshot is overwritten and no submission/public-notebook authorization is inferred.
