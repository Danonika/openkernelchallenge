# Next-kernel research — bounded planning summaries

Jury-approved NF4 cycle7 and training cycle6 refinements, summarized on 2026-10-30. **Research only:** no implementation, GPU launch, new measured result, priority change, catalog ID or graph point is created. The existing 121 ideas and prior negative trials are preserved.

- [NF4 forward plan](nf4-forward-summary.md): investigate bias-free forward decode while retaining vendor GEMM. **93.277% is geometric weight-volume coverage, not measured runtime coverage or useful-kernel share.**
- [Training/CE plan](training-ce-summary.md): scoped rewind-state reuse or a larger materialized CE boundary. **593.5 MiB is a conditional raw retained-state difference, not measured peak-memory savings.**

Trials001–003 remain standalone own-share failures. [Published002/003 reports](../../runs/20261030-trials002-003/README.md) retain console-only/raw-artifact limitations. Rounded percentages from separate runs cannot be added to predict a portfolio's admission. No formal parallel-control result is asserted.

Only scientific prose, public source pins and provenance hashes are published here. Source code/builders, execution/infrastructure/account details, control-plane status, row-level datasets/arrays, private links, credentials and weights are omitted. This does not retry any previously denied file or authorize additional jobs.
