# Soup Kernel Challenge — Team Duo

Official hackathon starter kit restored from [MakazhanAlpamys/soup-kernel-challenge](https://github.com/MakazhanAlpamys/soup-kernel-challenge).

## Submission status
**NOT READY — original competition solution and Kaggle verification are outstanding.**

The current `solution/` is the **official starter example**, not the research candidate described below. The provided `solution/swiglu.py` is an example kernel and does not count toward the mandatory original >=5% kernel-time share.

## Selected research candidate
`adaptive448-head-vchunk4096-N1024-v1`, listed in Open Kernel Research Graph, To Run row 878, request `soup-adaptive448-headlarge4096-20261031-v1`.

- Adaptive frozen-MLP input-only replay threshold: 448 tokens.
- Vocabulary-head workspace width: 4096 instead of 8192 for N>=1024 with Qwen V=151936 / H=1536.
- No vocabulary pruning; the full CE, entropy, per-row statistics, and dH are retained.
- Fixed Soup Qwen2.5-1.5B NF4 q/v LoRA 64-step task and unchanged data/optimizer/dropout RNG.

### Research-only performance (one T4 run)
| Metric | Candidate | Saved research reference |
|---|---:|---:|
| Total wall time | **38.123 s** | 38.49 s |
| GPU memory peak | **2261 MiB** | 2261 MiB |
| Public eval loss | **2.04979** | 2.04968 |

Source: research queue To Run row 878 / VM Runs row 59 / VM Details row 71. These are **not** official jury results or a comparison with clean Soup. The source notebook is recorded in the research table, but its pinned public URL was not accessible during repository preparation.

## Research provenance
The row-878 candidate is flagged as **RESEARCH ONLY** because the parent includes independently attributed third-party GPU kernels. AgentSwarm research modifications do not automatically establish independent authorship of inherited kernels. Do not submit this control as an original Triton implementation without replacing ineligible copied kernel bodies and validating new kernels.

## Official verification before submission
1. Put the independently authored solution in `solution/` and implement `apply()` without changing harness, task, or tests.
2. Verify that eligible original `@triton.jit` kernels execute for >=5% of measured training time (from step 3).
3. Run all official correctness and 79 Soup regression tests; verify loss, rewind, RNG, metrics and numerical thresholds.
4. Execute `templates/kaggle_notebook.ipynb` on free Kaggle T4 with the exact final `COMMIT` and save all results.
5. Publish the notebook and add its versioned URL to this README and `SUBMISSION.md`.
6. Include demo video <=3 min and presentation in the hackathon platform submission.

**Kaggle URL:** TODO — add when public and saved.  
**Final submission commit:** TODO — pin after eligible solution and validation.

See `SUBMISSION.md` for the submission checklist and required links.
