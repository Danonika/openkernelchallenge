# Soup research handoff

Status: research infrastructure installed; no kernel optimization measured.

## Baseline

Official Soup starter at `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.
`solution/` contains the example fused SwiGLU unchanged. It does not satisfy the
original-kernel requirements. No GPU runs have been performed by this integration.
Published organizer timings are reference values, not team measurements.

## Available evidence

- Supplied 22-page specification is archived and extracted in docs/requirements.
- Research policies and continuation protocol adapted from the pinned MIT source.
- Offline orchestration tests live in research_tests, separate from frozen tests.

## Next experiment

Establish the unmodified example baseline on one idle Kaggle T4 when access and a
GPU budget are provided. Inspect the profile and choose one original implementation
with meaningful training participation, such as a normalization or a loss/metrics
fusion preserving entropy, accuracy and per-row rewind losses. These are ideas,
not implemented or verified improvements.

## Pending

- Actual original kernel and numerical forward/backward verification.
- T4 timing, peak memory, held-out loss and all ten rule checks.
- Public Kaggle saved-version link, final submission commit, video, presentation.

## Resources

GPU harness invocations used: 0. No paid compute or scored submission authorized.
For a new run, use that run's explicit cumulative GPU budget and record consumption.
