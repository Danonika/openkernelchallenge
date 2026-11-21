# Hero1 qualification-first research — 2026-10-31

## Live evidence and hard blocker
The fixed Soup GPU service public comparator is **71.85 s / 14,345 MiB / eval loss 2.04968**. The 58.78-s / 9,325-MiB q/o-pad candidate is a speed Pareto point, **not a known admissible submission**. Its separately run original-source CUDA-event attribution audit yielded **702.596 ms / 35.718 s = 1.967%** (top original RMSNorm fwd/bwd; decoded NF4 cached away), below the mandatory 5%.

A different independently tested watched-NF4 decoder branch from the research graph achieved **68.520 s / 6,503 MiB / loss 2.04968**, with **3,645.904 ms / 41.194 s = 8.851%** attributable original Triton, of which actual NF4 work was **2,349.806 ms = 5.704%**. This source and its compiler/runtime watch integration are not automatically identical to the Hero1 no-cache source, so its 5% result cannot be transferred without a fresh audit.

The pending Hero1 research queue:
- To Run row 481, `SOUP-FUSION-HERO1-011`: 3 GiB exact frozen NF4 cache; active q/o projections + exact head reuse + valid-row norm/fused residual.
- To Run row 482, `SOUP-FUSION-HERO1-012`: identical source with cache limit 1 GiB.
- To Run row 483, `SOUP-FUSION-HERO1-013`: original one-pass stable CE/entropy/argmax Triton kernel, component only.
- To Run row 484, `SOUP-FUSION-HERO1-014`: no decoded-weight cache and same fast q/o+validnorm fusion, postrun **fail-closed 5% own-kernel attribution gate**.

All remain `Ready — awaiting intake`; no combined result and no admission. At latest inspection `VM Queue` recorded last poll **2026-10-31T10:27:00Z** with 70 accepted records; 9 newer ready requests were not ingested. This is an infrastructure blockage, not a solution benchmark outcome. Do not report metrics until a new VM receipt arrives.

## Exact fixed task and verification
Soup 0.77.0; Qwen2.5-1.5B-Instruct with NF4; 64 steps, batch4, seq<=512, public train 256 Kazakh instruction rows; LoRA r16 alpha32 on q/v with native dropout/RNG; optimizer, LR schedule, all Soup outputs, rewind, per-step metrics and hidden rows unchanged. Original `@triton.jit` launched in every measured run, measured from step3, contributes >=5% **useful** training time excluding the starter kernels and compile-only preparation. Public eval loss must be <=baseline+0.5%, loss curves/rewind within limits, 79 tests and jury review required. Official runner compilation+save time counts.

## Stop/go ladder
1. Read **terminal** queue receipt, public eval loss, full-command wall time, peak VRAM. Ignore partial or setup-only measurements.
2. Re-run the self-contained original Triton source attribution from `harness/check_rules.source_kernels` and CUDA event timing; **reject below 5%**, regardless of speed.
3. Run pinned 79 Soup regression tests with `SOUP_APPLY_SOLUTION=1`, then full `check_rules.py` with base+solution including correct Rewind row ids, token counts, entropy, accuracy and training loss curve.
4. Only then evaluate repeat variability and official Kaggle T4. Public VM T4 data is exploratory and cannot replace jury hidden-data or paired measurement.
5. Promote one allowed and reproducible single solution to the platform-created team repository, document source credits and failures, and prepare notebook, README/SUBMISSION, demo and slides before **2026-10-31 23:59 Astana**.

## One-pass vocabulary kernel math
Independent original `research/hero1_original_online_vocab_stats.py` uses numerically stable online softmax scan over FP16 scores; per row, maintain `m, Z = Σ exp(z-m), C = Σ exp(z-m)(z-m)`. When a tile raises the maximum from `m` to `m'`, rescale `Z` by `exp(m-m')` and `C` by the same factor including the shifted-centering term. Return `H = log Z - C/Z`, `CE = -((z_target - m)-log Z)`, first-index argmax. This can replace only *real* head-loss/metric work, never act as a synthetic busy-work kernel for quota.

Independent CPU FP16-replay numerical oracle on four shapes, vocab 4,096 and 151,936 with shifted logits and forced argmax ties, matched PyTorch reference: max CE abs diff about 3.03e-6; max entropy abs diff about 2.61e-5; zero argmax mismatches. This is **NOT Triton GPU compilation or kernel correctness proof**.

## Sources and idea attribution
- Soup actual Rewind and streamed loss requirements: https://github.com/MakazhanAlpamys/Soup/blob/main/src/soup_cli/trainer/rewind_hf.py and https://github.com/MakazhanAlpamys/Soup/blob/main/src/soup_cli/utils/stream_loss_chunk.py.
- Public competitor idea reviews (NO code copied): https://github.com/ainiddin/Soup-Kernel-Challenge ; https://github.com/FRIKT333/soup-kernel-challenge.
- Own implementation: https://example.invalid/redacted
- Own GPU notebook: project-repository/blob/master/soup-kernel-challenge/queue_20261031_hero1_nocache_ownshare_v1.ipynb.
