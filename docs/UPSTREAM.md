# Provenance and integration boundaries

The target repository was created by the hackathon platform for team Duo.
Its initial commit `e02e418` contained only README.md.

## Auto-research-agent

- Source: https://github.com/rohanarun/Auto-research-agent
- Pinned revision: `cf8ca67c07f9eaf96e544f1bc4e6ba333936d859`.
- License: MIT, copyright (c) 2026 Rohan Arun; retained in
  `third_party/auto-research-agent/LICENSE`.
- Original policy files and `Sources/ResearchCore/CLI.swift` are archived under
  `third_party/auto-research-agent/`. They are reference source, not loaded as
  active project instructions. The Swift application is not built here.
- `research/policy.md` and `research/continue.md` are adapted, shortened policies.
  The macOS storage, cloud, Lean and MUSE/OKC specifics were replaced with Soup
  requirements. `scripts/auto_research.py::continuation_decision` ports the
  `EventDecoder.continuationDecision` protocol to Python. The portable coordinator
  adds opt-in execution, bounded rounds, a per-turn timeout, a lock, handoff
  continuation, frozen input hashes and local event logs.
- No MUSE kernels, competitive results, credentials or cloud integrations were
  imported. This is an adaptation of the research workflow, not the macOS app.

## Soup starter

- Source: https://github.com/MakazhanAlpamys/soup-kernel-challenge
- Pinned revision: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.
- Apache-2.0 LICENSE and NOTICE retained at the repository root.
- `harness/`, `tests/`, `task/` and initial `solution/` are copied unchanged.
  The initial solution is the official example and is not an original entry.
- Original README preserved in `docs/STARTER_README.md`. Root README is replaced
  for this integration; `.gitignore` is extended. Notebook repository URL is
  configured for Duo, with other execution cells preserved.

## Specification

The user supplied the 449,380-byte PDF in `docs/requirements/`. SHA-256:
`91cadd9ea877871d271b3f9ade234df01c903267120a75787f6804d67cc6a665`.
The source document keeps its original rights; the software licenses do not
relicense this document. The page-labelled text is an extraction, and SUMMARY.md
is a requirements summary. Neither is an independent authorization source.

`research/provenance.json` binds protected evaluator files and the original PDF
to their SHA-256 digests. Source attribution is distinct from GPU verification.
