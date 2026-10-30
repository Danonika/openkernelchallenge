# Soup Fusion candidate

A merged implementation from the user's Qwen, LoRA, Nanochat transfer and Soup kernel research.

**Status:** implementation and CPU/interpreter checks complete; physical T4 correctness and performance remain unmeasured. The requested goal is a median full-command time below **33 seconds**, against a user-reported **77-second** baseline. No record-beating result is claimed.

## Start

For the prepared **Soup GPU testing service**, upload [`notebooks/Soup_Fusion_Service.ipynb`](notebooks/Soup_Fusion_Service.ipynb). It embeds actual solution files, preserves all three original service-template cells, runs physical readiness in separate processes, and uses the preinstalled offline environment. See [`docs/SERVICE_RUN.md`](docs/SERVICE_RUN.md) for exact submission commands, template provenance, report interpretation and the current HTTPS connection blocker. **No service job or GPU timing has been obtained for this candidate.**

For Kaggle, use the accompanying **Soup_Fusion_T4.ipynb** with Internet enabled and T4 hardware, then Run All. It embeds the complete source and needs no repository URL. It tests fixed candidates through the unchanged public protocol and preserves both successful and failed evidence. This Kaggle notebook installs the official dependencies and is a separate workflow from the offline service notebook.

On an already prepared T4 environment:

```bash
python tools/run_search.py --record-seconds 33 --out runs_search
```

The search compares `fused`, `cached256`, `epilogues`, and `streamed` in isolated local source snapshots. It stops at a verified sub-33-second result. Add `--continue-after-target` to finish the requested ladder and select its fastest admitted result. This is a bounded candidate search, not proof of a global optimum.

## Implementation

| Component | Default behavior |
|---|---|
| Frozen output head | Original full native FP16 GEMM once, retained private half logits |
| Loss and metrics | Original Triton summaries; native bounded CE backward with correct incoming gradient scale |
| RMSNorm | Fused FP32 normalization and staged backward matching audited arithmetic boundaries |
| MLP storage | Retain half-origin gate/up values in FP16; preserve FP32 activation and separate gradient-cast edges |
| LoRA | Native dropout/RNG and GEMMs; fused saved-mask input-gradient epilogue |
| NF4 | Original nested-scale/packed decoding on the existing compatible bnb boundaries |
| Optional experiments | Streamed head, rotary/scale-add epilogues, shared attention mask and frozen tied storage |

All implementation lives under `solution/`, because that is the package copied by the jury. The original task, harness and 79-test suite are unchanged. Original `solution/swiglu.py`, `LICENSE` and `NOTICE` remain present and credited. The starter SwiGLU contributes no original-kernel time.

## Executed verification

- **79 passed:** unchanged official regression tests with `solution.apply()`.
- **250 passed, 63 skipped:** candidate CPU/Triton-interpreter tests. The skipped physical-CUDA checks remain pending.
- **32 passed:** independent result-verifier tests, using explicitly synthetic evidence fixtures.
- **Four real Soup-factory updates:** cached head + LoRA fusion preserves tested gradients, parameter updates, RNG and counts exactly; scalar/metric/row-loss error below `5e-7`.
- **Offline SM75 compilation:** checked original and new kernels; no compiled caches included.

See `FUSION_HANDOFF.md` for the complete decision record, limits, logical memory accounting, source links and exact validation scope. See `validation/summary.json` for the machine-readable state.

## Official acceptance commands

```bash
python task/prepare_data.py
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2 --out runs/final
python harness/check_rules.py --result runs/final/result.json
python harness/score.py runs/final/result.json
python tools/verify_record.py runs/final/result.json --record-seconds 33
```

Use one visible T4. The official two-repeat sequence is base → solution → solution → base with fresh compilation caches, median whole-command time and worst sampled peak memory. All ten checks, genuine metrics/Rewind, adapter quality and ≥5% useful original-kernel share must pass before accepting a candidate.

## Result table

| Quantity | Value | Evidence |
|---|---:|---|
| User-reported reference baseline | 77 s | User request; not reproduced here |
| User-reported record/target | 33 s | User request; original protocol/logs unavailable |
| Candidate full-command time | Unmeasured | Service access blocked by this workspace's TLS certificate-name mismatch |
| Candidate sampled peak | Unmeasured | No physical T4 run of this candidate |
| Public automatic admission | Unmeasured | No full challenge training/evaluation run |
| Hidden-jury admission/rank | Unknown | Requires organizer evaluation |

Do not publish CPU duration, offline compilation time, nominal tensor bytes, or synthetic verifier fixtures as GPU performance. Actual results will be written into the search output, without changing this source or the official harness during measurement.

The official starter README is preserved at `docs/OFFICIAL_STARTER_README.md`. This is a source candidate and benchmark workflow awaiting GPU evidence. It is not a final competition submission.
