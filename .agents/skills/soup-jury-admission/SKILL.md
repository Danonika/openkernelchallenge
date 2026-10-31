---
name: soup-jury-admission
description: Review Soup Kernel Challenge candidates and measured runs against the pinned contest rules; distinguish legal research, numerical evidence, preliminary checks, and final admission.
---

# Soup jury and admission

Use this project-local skill for Soup candidate filtering, trial review, or submission readiness. It does not install a global skill or authorize execution, publication, visibility changes, or submission.

1. Read the single canonical [versioned rule contract](references/rules-v1.json), including its evidence schema and the PDF/code aggregation discrepancy. If official main changes, pin and compare it before changing this resource; never silently reinterpret a threshold or edit the frozen harness.
2. Review each candidate separately: abstract legality; prohibited dependencies/code reuse; hardware fit; semantic risk; CPU evidence; GPU unknowns; useful original-kernel share; priority/action. Preserve catalog IDs and negative witnesses. Do not require universal bitwise equality where the official task specifies numerical tolerance, and do not replace the official tolerance with a permissive toy allclose.
3. Keep the three easily confused quantities distinct: useful own-kernel share, evaluation-loss worsening, and speed/memory noise floor. Their authoritative units/limits are in the rule contract. “5% loss is allowed” is false.
4. Run the read-only checklist helper on a schema-conforming evidence file:
   `python scripts/checklist.py evidence.json --out checklist.json --markdown checklist.md`
   Paths are relative to the invocation directory; use explicit paths. Missing evidence is UNKNOWN, explicit contradiction is FAIL. The helper never runs the harness, GPU, network or publication. It is not a replacement for unchanged `check_rules.py` or official jury review.
5. A public evaluation cannot clear hidden-row quality. COMPLETE or wrapper exit0 cannot clear nested train/eval success. CPU fixtures, source compatibility and an original kernel in a file cannot establish actual T4 execution, speed, allocator peak, useful share or full quality. Hash-bind result, checks and exact solution; retain failed/partial runs.
6. Report the highest-value next action and exact missing proof, not an unbounded catalog audit. Keep one bounded candidate per isolated Kaggle session/GPU. Total concurrent sessions must not exceed the verified account limit or the current user cap of5, must stay within free quota, and must not duplicate runs or share benchmark resources. Five is an authorization ceiling, not evidence that five sessions are available. Useful components may fail standalone own-share; never add redundant work or slow a kernel to reach it.

The [checklist guide](references/CHECKLIST.md) is generated from the canonical contract and is the table-link destination. Human/source facts require artifact references, not bare assertions. All automatic checks are preliminary; final jury authority and final submission artifacts remain distinct gates.
