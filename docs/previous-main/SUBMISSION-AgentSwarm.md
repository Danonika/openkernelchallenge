# Submission: Duo — DRAFT, NOT YET ELIGIBLE

## Links
- Team repository: https://example.invalid/redacted
- Final commit: **TODO (must match Kaggle COMMIT)**
- Public Kaggle notebook with saved results: **TODO**
- Demo video, <=3 minutes: **TODO**
- Presentation: **TODO**

## Team
Fill in the actual human team members (maximum 4). AI agents are tools, not additional team members.

## Current code
`solution/` currently contains the **official starter example** and does not represent the research candidate. Original qualifying Triton kernel list and timed execution share: **NOT VERIFIED**.

## Intended research implementation
`adaptive448-head-vchunk4096-N1024-v1`:
- Adaptive frozen MLP replay at threshold 448.
- Vocabulary-head vchunk=4096 when N>=1024 (V=151936, H=1536).
- Preserve original Qwen NF4/LoRA training and complete output metrics.

**Provenance:** Research baseline includes attributed third-party GPU kernel code. The original competition implementation must use independently authored eligible Triton bodies; currently absent.

## Research-only T4 evidence
| Metric | Research candidate | Saved research reference |
|---|---:|---:|
| Wall time | 38.123 s | 38.49 s |
| Peak GPU memory | 2261 MiB | 2261 MiB |
| Eval loss | 2.04979 | 2.04968 |

Queue To Run row 878; VM job n_a755cc946d544ae5ab94502096245ff4. One unpaired complete 64-step research run; not official Kaggle validation and not final jury-admitted result.

## Final submission result
| Metric | Base Soup | Submitted implementation |
|---|---:|---:|
| Wall time | TODO | TODO |
| Peak MiB | TODO | TODO |
| Eval loss | TODO | TODO |
| Own Triton kernel runtime share | n/a | TODO |

- [ ] Independently authored qualifying kernels implemented in `solution/`.
- [ ] Measured >=5% useful own Triton runtime from step 3.
- [ ] All 10 official `check_rules.py` checks pass.
- [ ] All 79 Soup regression tests pass.
- [ ] No forbidden task / harness changes.
- [ ] Public Kaggle Run All saved at final commit.
- [ ] README updated with final verified metrics.
- [ ] Presentation and <=3 minute demo uploaded.

**Do not treat this draft as a completed final submission.**
