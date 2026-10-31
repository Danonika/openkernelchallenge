# Soup CPU cycle 2 — centered vocabulary merges and nonfinite semantics

**6 fixture groups passed; 0 unexpected failures.** Passing means the requested behavior or counterexample was established—not that an optimization passes challenge gates. Original cycle-one artifacts remain immutable.

## Important negative findings

1. **SOUP-TRAIN-001 cannot unconditionally scatter zero on ignored rows.** Ignored NaN, +Inf, or all−Inf logits each produced three NaN dlogits in the dense CPU baseline, even though summed CE stayed finite (including all-ignored batches). Compact autograd scattered zeros instead. Separate NLL→log-softmax inspection confirmed NLL's ignored gradient was zero; the NaNs arose later in log-softmax backward. Mixed finite/−Inf rows were a distinct control and retained zero ignored gradients.
2. **Finite operands do not guarantee finite FP16 logits.** A tiny head with finite rounded inputs and weights overflowed one ignored row to `[+Inf,−Inf,finite]`. Dense backward produced two NaN hidden-gradient elements; compact backward produced zeros. A guard on input/weight finiteness alone is insufficient.
3. **Zero dlogits multiplied by nonfinite head weights need not produce zero.** NaN, +Inf and −Inf weights each caused two NaN output elements in a zero-row matmul; direct zero scatter suppressed them. Finite FP32 weights can themselves overflow when cast to FP16, so guard the *effective FP16 weights*.
4. **Infinite effective upstream gradient invalidates a naïve derivative rewrite.** For finite selected logits and incoming +Inf, baseline NLL→log-softmax produced `[NaN,+Inf,+Inf]`; `(softmax−onehot)*scale` produced `[−Inf,+Inf,+Inf]`. Zero denominator gave the same target-class mismatch. Compact **reference autograd** matched dense autograd in these finite-logit fixtures; the naïve derivative formula did not. Incoming NaN and infinite/NaN denominator controls are recorded separately.
5. **Empty vocabulary tiles require explicit identities.** The unguarded centered-moment merge evaluated `0 × −Inf` and produced NaNs. Masked tail lanes also must contribute zero explicitly. Processing rightmost tiles first and choosing “left on tie” returned argmax index 1 instead of baseline index 0; use the lowest **global** index, not traversal order.

These findings keep SOUP-TRAIN-001 **conditional**, not rejected wholesale: its finite-path opportunity remains supported by cycle one. A conservative design needs finite actual rounded logits, finite effective head weights, finite incoming gradient and positive finite denominator, plus an exact-reference fallback otherwise. These are design obligations, not an implemented or verified routing solution. Check the effective incoming normalization for overflow as well; individually finite scalars need not have a finite quotient.

## Positive finite-path evidence—and remaining rounding error

- **900 centered vocabulary-merge cases**: vocabulary sizes 1/2/3/7/16/19/31/33/127/257; ordinary/high-offset/equal-max-half/near-certain/tied logits; six fixed/unequal tile patterns; forward, reverse and balanced merges; strided rows.
- First-index global argmax matched **exactly** in every case.
- Maximum absolute differences versus CPU reference: CE **1.90735e-6**, entropy **1.66858e-6**, probability **1.66893e-6**. Exploratory tolerances were `atol=3e-6, rtol=2e-5`; these are **not challenge tolerances or GPU claims**.
- Centering is **not a proof of bitwise reduction equivalence**. With 257 logits `[32768,32752,…]` and singleton tiles, forward merging gave target CE **3.05171125e-5** versus baseline **2.88482315e-5** (about **5.8% relative**, despite small absolute error). Reverse and balanced merges matched that target CE in this fixture. Balanced merging reduced the overall entropy maximum error to **4.76837e-7**, but no universal superiority or GPU result is claimed.
- Empty-tile identities and padded tails matched exactly after explicit handling.

## Artifacts and reproduction

```bash
bash /workspace/shared/soup_cycle/cpu_tests/cycle2/run.sh
```

`cpu_tests/cycle2/` contains independently written `run_cycle2.py`, detailed `results.json`, concise `summary.json`, logs, original-cycle hash manifest and `verification.txt`. Nonfinite comparisons record **finite / NaN / +Inf / −Inf classes and mismatching positions**, not blanket `equal_nan` comparisons. Initial six-group results/logs are retained separately.

Cloud CPU only; two intra-op and two inter-op threads; Torch 2.15.0+cpu, actual build `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`. No installs, models, data, CUDA, real GradScaler, optimizer steps, full training, remote writes, or official-test reruns. No performance/peak-memory/quality/kernel-share claim. Original challenge checkout stays untouched.

## Next scoped work

Parent-authorized LoRA cycle 2b will separately test B+scale+add custom-autograd ownership while retaining q/v dropout and base clones, plus per-saved-mask hook selectivity. It will not modify these six-group results or the original eight-group results. No third CPU cycle is planned without reassignment.
