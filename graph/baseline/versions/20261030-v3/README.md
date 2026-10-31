# Baseline graph v3 — sanitized source mirror

Live private deployment URL and access/account identifiers are intentionally omitted from this source mirror. Source deployment commit: `e3ba9231c1e7f917e709c2d9e57e84dc9961bd8a`. This versioned repository mirror does not deploy or mutate the live site, and does not overwrite the earlier baseline-only reference snapshot.

**One B0 configuration bucket/subgraph, one point, one completed base run. Zero candidate nodes or comparisons.** Observed reference values are 77.92 s training-command wall time, 14,345 MiB sampled peak, 54.04 s training loop and public evaluation loss 2.04965; 64 steps, 256 public evaluation rows, one visible T4, one repeat. X=Y=0 is self-normalization, not a candidate gain. Z/admission and all official points remain unknown. CPU evidence never creates T4 points.

## Sanitization

Private notebook/account identifiers, notebook URL, external session identifier and account/quota metadata are omitted. The private session label is replaced with the literal `redacted-private-session`, not an equivalent identifier. The notebook UI link points instead to the local sanitized scientific evidence summary. Cloud-machine evidence paths become repository links. No credentials, auth/runtime configuration, `.git`, `.openai`, model weights or unsanitized error logs are included. Scientific measurements and the four original evidence artifact hashes remain unchanged.

The raw result, metrics, evaluation and environment JSON are included under `evidence/kaggle-v3/` after explicit user authorization for detailed research/baseline artifacts. Their summary is privacy-redacted. The separately published [baseline evidence package](../../../../research/runs/20261030-kaggle-baseline-v3/) includes the original wrapper, validator and additional metadata/checkpoint evidence.

The graph retains its initial 118-row catalog provenance, while the current research catalog has 121 ideas. [Cycle3](../../../../research/cycles/20261030-cycle3/) is an existing-ID refinement; it does not create measured candidate configurations.

## Use and verify

Open `index.html` beside the other static files, or use a normal static server. No build, backend, API credentials or network writes are required. `data.js` embeds the same graph as `graph.json`; `schema.json` is its schema. Run `python verify_snapshot.py` here for all file hashes, baseline-only invariants, exact artifact hashes, null admission/points, unchanged B0 configuration hash and redaction checks. `node --check app.js` and `node --check data.js` are optional syntax checks; JSON Schema validation was also performed before publication.

The live view was cloud-browser checked by its publishing lane. Mobile layout was not browser-tested. This source mirror differs from the live source only as described by the sanitization list; it is not a new GPU result or a new deployment.
