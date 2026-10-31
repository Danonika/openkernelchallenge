# Soup development attempts — wave 2

This archive freezes the 16 accepted GCP submissions through 2026-10-30 22:28 UTC, including two RoPE microbenchmark-only attempts. The prior six-candidate release remains unchanged.

## Scientific status

- Only the baseline has complete fixed-task and 256-row public-evaluation evidence: 71.29 s, 14345 MiB, loss 2.04962. This is not a contest score.
- Candidate training attempts failed preflight or the unchanged 80 s full-command deadline. Internal 64 steps never overrides a command timeout.
- Mixed-row CE activated the intended route; one individual command returned at 76.76 s / 13165 MiB, but solution_2 timed out and the cohort/evaluation/check never completed. Same-window descriptive original-kernel share was 0.911%, below 5%, not an official rule-check result.
- Corrected RoPE V3 completed only synthetic native/old/runtime fixtures. JIT-key reduction is not evidence of full-task speed, memory, quality, useful share or admission.
- No paired ratios, contest scores or Kaggle admission are claimed. No unchanged failed notebook should be automatically rerun.

## Contents and use

Each candidate directory contains the exact submitted notebook bytes, its original solution source, relevant historical validation-test sources, a source manifest, and a minimized result summary. Notebooks are self-contained; copied test sources are audit material and may refer to their original packet layout. No payload is executed while preparing this archive.

The fixed task/harness/optimizer/RNG/native outputs and the 80 s engineering cap are not modified. Original useful Triton time >=5% from step 3 remains a separate requirement; final admission is Kaggle-only.

Excluded: credentials, API identity/status response blobs, private Sheet snapshots/notes, output ZIPs, model/cache/data assets, generated PTX/binaries and unsubmitted variants. Original packet preparation statements are superseded by each RESULT_SUMMARY.json.
