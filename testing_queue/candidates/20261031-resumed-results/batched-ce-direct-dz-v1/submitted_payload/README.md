# Web profile generation kit V2 — evidence parsing repair only

This NEW directory supports new candidate revisions; it does not replace or mutate any submitted notebook, source ZIP, accepted result, or shared V1 file. Do not rerun an unchanged candidate merely to repair parsing: original raw logs remain available for analyst review. A timeout stays failed/censored regardless of recovered route events.

## Interface

```sh
python /workspace/shared/soup_web_ready_20261030_v2/build_profile_notebook.py \
  --lane-dir /path/to/NEW_LANE --profile /path/to/profile.json \
  --notebook-name candidate.ipynb
```

The new lane must contain a reviewed `candidate.zip` and `source_manifest.json` (or `candidate-files.json`). The builder refuses to overwrite `web_adapter/` or the notebook. Same seven profile fields as V1: `mode`, `title`, `preflight`, `preflight_status`, `kernels`, `routes`, `requires_original_kernels` (no implicit defaults). Each route now explicitly requires `prefix`, `positive_field`, and either `event` or a nonempty exact `match` map. Optional `match` values are type-sensitive. Example:

```json
{
  "mode": "head_quiet",
  "title": "Native head quiet control",
  "preflight": "solution.web_preflight",
  "preflight_status": "PASS_NATIVE_HEAD_PREFLIGHT_ONLY",
  "kernels": [],
  "routes": [
    {"prefix":"SOUP_HEAD_NATIVE ","event":"native_sum_forward_return","positive_field":"count","match":{"count":64}},
    {"prefix":"SOUP_HEAD_NATIVE ","event":"native_sum_backward_return","positive_field":"count","match":{"count":64}}
  ],
  "requires_original_kernels": false
}
```

Native zero-owned-kernel control remains subject to unchanged official eligibility rules. If rules fail, normalized comparison is null, score null, admission UNKNOWN. No bypass.

## What changes

Only the generated profile's metadata evidence reader changes: exact line-anchored prefixes are read after stripping terminal ANSI sequences; `JSONDecoder.raw_decode` accepts ordinary JSON whitespace wrapping across lines. JSON must be an object; malformed/truncated/oversize/duplicate-key/nonfinite JSON is rejected. Schema/event/exact-match and positive finite numeric count requirements remain mandatory. Bare substring hits, wrong events, zero/negative/bool counts and incomplete runs cannot pass. `match` now also rejects mismatched JSON primitive types (e.g. integer64 versus string"64" or float64.0).

Actual source-bound FusionB logs demonstrated why this is needed: V1 single-line parsing saw only installation events, because Rich inserted cursor controls before runtime prefixes and wrapped the JSON. V2 recovers those records for analysis; it **does not** convert that80-second timed-out run into completion or a measurement.

## What stays unchanged

- No numerical kernel, model, task, precision, optimizer, data, LoRA, harness, protected source hash, runtime, cache, training duration, cancellation or export behavior changes.
- All original shared runner functions are AST-identical in the V2 seed runner; only an import of the evidence helper is added. Generated standalone `normalize` specializes metadata routes as before. Generated `main` differs only in CLI mode metadata; its execution body is unchanged.
- Same `SOUP_REPO`/`SOUP_RESULTS`, `/kaggle/working`, gitless exact-source verification, runtime/idle/preflight child gates, unchanged official paired64-step task and **80-second cap per native training run**.
- Same pidfd/subreaper guardian, inherited lock, quarantine, uncertain-cleanup handling, output completeness gates, fixed56MiB export budget, score null and admission UNKNOWN.
- V1 `00_baseline.ipynb` is stored only as `workflow_seed.ipynb` to reuse its reviewed cell workflow. It is not a new runnable baseline release and is never submitted by this kit. Payloads are fully replaced in generated notebooks.
- V2 is for NEW source/candidate variants. No preflight bypass, install/download, authenticated request or submission occurs in generation/CPU tests.

The import-only V2 seed runner retains the old Fusion A/B metadata branches solely as inert seed compatibility; the builder prohibits those modes and does not issue new00/01/02 notebooks. Only generated new profiles use the repaired reader.

See `validation/v2_results.json`, `validation/v2_checks.log`, and `validation/immutable_v1_hashes.json`. GPU performance, target execution, quality and admission are not established by these parser fixtures.
