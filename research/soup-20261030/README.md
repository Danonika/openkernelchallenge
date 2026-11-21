# Soup research branch — 2026-10-30

## Start here
- [Master table: all 118 IDs](MASTER.md) · [CSV](master.csv) · [JSON](master.json).
- [Allowed/conditional shortlist](SHORTLIST.md): permitted independent methods, not permission to copy implementations.
- [Full original Russian ТЗ](TZ.ru.md), reproduced unchanged: 214 lines, 32,456 bytes, SHA256 `5e6c1574af02da9c4e83db0128afa97e50e9b81050d57c25de83de268457551c`.
- [Cloud/Kaggle environment and test checklist](ENVIRONMENT.md).
- [Classification corrections](soup_classification_corrections_20261030.md) override earlier research assumptions.
- [Head/metrics/norm dataflow](soup_head_norm_dataflow_plan_20261030.md), [NF4/LoRA audit](soup_nf4_lora_audit_20261030.md), [nanochat and same-challenge findings](soup_research_additions_20261030.md), [fresh compiler sources](soup_fresh_source_increment_20261030.md).

**Status:** research only, all ideas unmeasured on our Soup T4 run. This publication contains no new kernel implementation, no trained adapter, no score, no contest submission. This commit adds research documents only, atop main `077f5b1bb36466c5452aff17af036eb9aed052fd` (the independently reverted integration). No reverted starter/coordinator files are resurrected. Existing root README and main are not changed; environment work is separate.

## Coverage and authority
Original [104-idea catalog](https://example.invalid/redacted-github), Git blob `1031de186d7b051b71f27e0bdaf28a774d45d2ec`, is retained as [original104.json](original104.json). Its Muse/BF16/Ada/Hopper requirements are **legacy provenance, not Soup instructions**. New rows: SOUP-NEW-001–007, SOUP-NANO-001–002, SOUP-NANO-X01–X02, SOUP-COMP-X01, SOUP-COMP-001–002 =14; total118. Exclusions are counted transparently, not presented as viable kernels.

Classification counts: 39 allowed methods, 48 conditional, 16 outside fixed workload, 10 hardware-incompatible direct mechanisms, 4 prohibited direct transfers, 1 hold. These are separate from runtime-library legality: a banned source can inspire an independently written permitted mathematical method. Hardware mismatch is not itself a rule prohibition.

Authoritative executable rules: [official starter c7c2fcd](https://example.invalid/redacted-github), especially README, harness/common.py, check_rules.py, _train.py and run_task.py. Full ТЗ preserved as supplied, even where prose and code differ: ТЗ describes averages; harness uses median command time and maximum sampled memory. ТЗ describes image Python3.13/torch2.11/Triton3.6; notebook leaves those versions unpinned. Record actual versions, do not silently rewrite the ТЗ.

## Non-negotiable research boundary
No copying kernels, banned imports, native-CUDA replacement route, graphs, precompiled caches, measurement-specific behavior, hidden-data probing, task/switch changes, new optimizer, changed batching, changed dtype or disabled metrics. All future kernels must be independently authored in solution/, useful in every solution run, and preserve original training semantics. Every Python file under solution/ is statically checked, including tests and dead branches. Inductor/starter/renamed/copied kernels do not count as owned.

Own-kernel share is CUDA-event milliseconds divided by summed wall-clock training-step seconds from step3 onward, at least5% in the weakest solution run; it is not5% of GPU-only profiler time. All numerical and structural gates remain mandatory; paper and competitor speedups are not our measurements. No cloud preparation test can establish Kaggle T4 qualification.

## Next evidence, not endless search
Wait for actual version inventory, baseline shape/compile histogram and first authorized T4 profile. Then choose one bounded study: norm/head correctness, NF4 path coverage, specialization explosion, or register spills. Require a falsifiable hypothesis, exact source and candidate commit, test artifacts, full cold command timing and peak memory. Stop a study when rejected or promoted by unchanged harness evidence; retain failed IDs. No automatic paid GPU provisioning, training, account creation or submission is authorized by this document.
