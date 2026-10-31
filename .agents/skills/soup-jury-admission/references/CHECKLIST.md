# Soup jury checklist
**Canonical version: soup-c7c2fcd-v1** · [machine-readable rules and evidence schema](rules-v1.json)

This generated view is a checklist, not official admission. Missing evidence is UNKNOWN; an explicit contradiction is FAIL. A PASS requires the referenced proof. Public evaluation cannot prove hidden quality. The five mandatory sections and four submission deliverable categories are all represented below.

## Never confuse these thresholds

| Quantity | Canonical limit | Meaning |
|---|---:|---|
| Useful own Triton share | ≥ 5% in every measured run | Event time / timed-loop wall time from step3; NOT loss |
| Evaluation loss worsening | ≤ 0.5% | Hidden-row final gate; public only self-check |
| Time / peak growth | ≤ 10% each | Whole command / sampled max memory |
| First training loss gap | ≤ 1% | Relative to baseline first step |
| Loss trajectory, entropy, accuracy, rewind | ≤ 3% | Mean per-element relative gap, not gap of means |
| Speed / memory scoring floor | ratio ≥ 1.05 | Below this is a scoring tie; separate5% concept |

## Source discrepancy — visible, not silently resolved

For exactly2 repeats median time equals mean; max memory is NOT mean memory. Show both claims explicitly, use frozen code-native numbers, do not edit code or invent organizer clarification.

Own-share uses common eligible names launched in every solution run and the minimum per-run share. Exclude starter/renamed starter, copied or ready-made kernels, Inductor kernels, unused work and manufactured slowdown. Total useful own share is collective; each individual candidate need not reach the minimum alone.

## Deterministic evidence checklist

| ID | Gate | Required proof | Source |
|---|---|---|---|
| run-all | must-have-1 | Clean free Kaggle T4 Run All, no manual edits | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md#L35-L72) |
| finished | must-have-1 | Every expected base/solution train run and evaluation succeeded | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L279-L300) |
| paired | measurement | Two runs each in one session: base, solution, solution, base | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/run_task.py#L179-L199) |
| own-source | must-have-2 | Original useful solution/ Triton; no example/copy/Inductor/unused-work credit | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md#L77-L84) |
| own-share | must-have-2 | Eligible common kernel names contribute enough time in EVERY run | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L322-L362) |
| dependencies | hard-constraints | No prohibited imports/runtime libraries or package changes during measured run | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/common.py#L17-L28) |
| public-quality | diagnostic | Public evaluation quality self-check, not hidden admission | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L388-L399) |
| hidden-quality | must-have-3 | Hidden-row adapter evaluation quality; public rows cannot clear it | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md#L147-L170) |
| time | must-have-3 | Whole-command time growth within allowed margin | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L401-L417) |
| memory | must-have-3 | Maximum sampled GPU peak growth within allowed margin | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L401-L417) |
| task | must-have-4 | Exact task/LoRA/dropout/optimizer/lr/batches/steps/output structure | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md#L147-L170) |
| tokens | must-have-4 | Exact same nonempty per-step token counts | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L186-L236) |
| first-loss | must-have-4 | First-step training loss relative gap | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L239-L251) |
| loss-curve | must-have-4 | Mean per-element relative training-loss gap | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L151-L164) |
| entropy | must-have-4 | Mean per-element relative entropy gap | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L186-L236) |
| accuracy | must-have-4 | Mean per-element relative token-accuracy gap | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L186-L236) |
| rewind | must-have-4 | Exact row IDs/counts and relative per-row loss gap | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L186-L236) |
| clean-gpu | must-have-4 | Single visible T4, idle start, correct allocator and sampled peaks | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L419-L440) |
| regressions | must-have-4 | All79 unchanged Soup tests with patch; starter tests are not these tests | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L239-L251) |
| frozen-files | hard-constraints | Unmodified harness/,tests/,task/ and their settings | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md#L115-L127) |
| no-evasion | hard-constraints | No hidden-data search, external-file access in solution, special test behavior, background GPU work, adapter swap, precompiled caches or CUDA graphs | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md#L108-L113) |
| official-checks | automatic-preliminary | All10 unchanged official checks hash-bound to this exact result and solution | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py#L263-L277) |
| readme | must-have-5 | README explains mechanism/sources, exact run, measured comparison, limitations, notebook/final commit | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/templates/SUBMISSION.md) |
| platform-repo | submission | Platform-created team repo accessible to jury; do not assume research repo qualifies | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md#L35-L48) |
| final-notebook | submission | Public saved template notebook with results, COMMIT=final commit, links in README/SUBMISSION | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/templates/SUBMISSION.md) |
| submission-form | submission | Root SUBMISSION.md completed with correct final artifacts | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/templates/SUBMISSION.md) |
| demo | submission | Demo video within duration limit | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/templates/SUBMISSION.md) |
| presentation | submission | Presentation covers operation, mechanism, measured results, limits/next steps | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/templates/SUBMISSION.md) |
| team | submission | Team-size and one-result limit respected | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md#L3-L6) |
| jury-review | final-authority | Official jury source review is final; preliminary checks are not its decision | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md#L147-L170) |
| noise-floor | scoring | Speed/memory ratio below1.05 is tied with baseline, not a loss tolerance | [pinned source](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/score.py#L1-L47) |

## How to use

Create evidence JSON following `rules-v1.json.evidence_schema`; point `result_path` and `checks_path` at immutable local artifacts and provide the exact solution hash. Reviewed `facts` contain a boolean value plus nonempty artifact references. The helper does not independently certify those human/source claims. Missing facts must be omitted or left unknown, never guessed true.

`python ../scripts/checklist.py evidence.json --out checklist.json --markdown checklist.md`

Run the unchanged official harness/checker separately under authorized clean conditions. This helper is read-only and performs no GPU, network, installation, submission or publication. Synthetic examples test the reviewer logic, not challenge compliance.

## Final package, not just kernel code

Platform-created team repository; solution/apply; README with mechanism/source/run/results/limitations and notebook/final commit; completed root SUBMISSION.md; authorized public saved template notebook whose COMMIT matches; demo within the canonical duration limit; presentation. Team size/one-result/deadline and final jury review still apply. Do not publish a private notebook/account identifier in research notes merely because the eventual final notebook must be public.

Official deadline: 2026-10-31T18:59:00Z (2026-10-31T23:59:00+05:00). Current main must be rechecked before finalization. No skill or checklist grants external mutation or disclosure authority.
