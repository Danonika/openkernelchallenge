# Training lane increment 2 — exceptional gradients and public-label geometry

2026-10-30 · existing `SOUP-TRAIN-001` only · cloud/source research, no model download or training.

## Decisions first

- **Do not promote unconditional zero-scatter.** An ignored row with NaN, +Inf or all −Inf logits can have zero scalar CE contribution yet NaN dlogits. Skipping that row changes backward behavior.
- **Do not use `(p−onehot)*incoming_gradient` as a universal nonfinite oracle.** With +Inf incoming gradient, baseline NLL→log-softmax backward can give NaN at the target whereas the algebraically simplified expression gives −Inf. The CPU lane has confirmed these negatives; complete positional-mask evidence is separate.
- Keep a **finite-domain compact fast path plus reference-ordered exceptional fallback**, not clamping/sanitization or direct manipulation of GradScaler state. Guard and fallback costs are unmeasured.
- **The last prompt hidden position is supervised when it predicts the first assistant token.** Select with shifted labels, never by an unshifted “prompt versus assistant hidden row” shortcut.
- Exact public U/N statistics remain unavailable: the selective dataset endpoint was inaccessible through the web tool. No full dataset, hidden rows, alternate network route or model weights were fetched. Source establishes the geometry below, not its empirical distribution.

## 1. Why ignore_index does not authorize skipping every exceptional row

At the inspected [PyTorch source pin matching the CPU build, d0f2af8](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/native/cuda/SoftMax.cu#L39-L64), log-softmax backward subtracts the exponential of the saved log-probability times the row sum from its incoming per-class gradient. [CUDA NLL backward](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/native/cuda/Loss.cu#L458-L485) does not write target gradients for ignored rows. [Cross-entropy composition](https://github.com/pytorch/pytorch/blob/e9a0032ac4a867ce57c079de379ece80e230be5d/aten/src/ATen/native/LossNLL.cpp#L645-L668) applies log-softmax before NLL. These are source observations, not evidence of the actual Kaggle Torch version or executed GPU kernel.

Let v be the NLL gradient vector and q its row sum. Baseline log-softmax backward has form `v − exp(logp)*q`.

| Case | Source-derived behavior / CPU-lane finding | Consequence |
|---|---|---|
| Ignored row, finite logits, finite frozen effective head weight | NLL v=0, q=0; dlogits and head dHidden are zero. | This is the sound finite-domain compaction argument. |
| Ignored row, NaN/+Inf/all −Inf logits | Log-softmax contains NaNs; multiplying them by q=0 still yields NaN. Summed CE can remain finite because NLL ignores the row. | Zero-scatter can hide nonfinite gradients and alter later unscale/skip behavior. |
| Ignored row, mixture of finite logits and −Inf | Saved log-probabilities may include −Inf, but exp(−Inf)=0. | Do not automatically replace every nonfinite-logit row with all-NaN gradients. CPU found zero ignored-row dlogits here. |
| Supervised finite-logit row, incoming +Inf | Target v is −Inf and q is −Inf; subtraction may produce target NaN. | Simplified `(p−onehot)*Inf` has a different target nonfinite class. |
| Finite ignored row, incoming NaN/Inf | NLL still skips that row. | Multiplying a pre-masked zero tensor by a nonfinite scalar can incorrectly create NaNs. Apply the actual ignore semantics, not `0*Inf`. |
| Zero dlogits, nonfinite effective head weight | Dense matmul can encounter 0*Inf/NaN. | Even a mathematically zero row cannot generally bypass a nonfinite weight operand. |

Do not argue that both variants eventually skip an optimizer step, therefore they are equivalent: nonfinite masks can propagate differently through decoder/LoRA operations and affect recorded grad norms and other state. [GradScaler checks gradient nonfiniteness before deciding whether to step](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/torch/amp/grad_scaler.py#L349-L373); it must remain unchanged.

### Sufficient fast-path domain, not a finished implementation

Require all of the following for the simple zero-scatter argument:

1. Effective autocast FP16 head weight is finite (validate when built; invalidate with the weight/version/device/dtype cache).
2. **Actual FP16 head outputs** for the full original forward geometry are finite. Finite FP32 input/weight tensors alone are insufficient: FP16 conversion or GEMM output can overflow.
3. Incoming scalar gradient is finite; the actual global denominator is finite and positive, and the effective upstream gradient after the baseline normalization remains finite. The fixed-task denominator is a token count, but a generic positive sub-unit divisor alone would not ensure this.
4. Unchanged ignore/shift/rounding contract, including scaled FP32 dlogits before FP16 cast and original FP16 dHidden boundary.

A finite scalar loss alone is not the guard: ignored nonfinite rows can be invisible to it. A host `.item()` guard introduces synchronization; a device-side flag needs an explicit launch/control design. Either cost is part of measured work. Guard failure should take a reference-ordered backward over the necessary original rows, retaining NLL ignore behavior and log-softmax order. Do not use `nan_to_num`, replace NaNs with zero, alter the scaler, or preemptively skip an optimizer step.

**Lifetime consequence:** if the compact implementation saves only supervised hidden rows, it cannot later recompute exceptional ignored rows. Retain full final hidden state or a proven sufficient exceptional-row representation until the fallback decision is settled. The ordinary full hidden tensor is about 12 MiB at B4,T512,H1536,FP32; this is a tensor-size calculation, not a peak-memory measurement. An eager permanent “save selected rows only” optimization is therefore deferred.

The easiest first implementation is conservative dense/chunked reference fallback. A finer union of supervised and exceptional rows is a future refinement, not proven in this cycle. Reference-ordered nonfinite behavior still needs actual installed CUDA tests; the CPU lane cannot establish GPU FMA, GEMM or NaN-propagation parity.

## 2. Actual public task geometry from source

### Public row selection and unchanged controls

[Official prepare_data.py](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/task/prepare_data.py) fixes dataset revision `c641407ee50723c84a43ea2ec867b464d3893c12`, public training rows 0…255 and public evaluation rows 40000…40255. It writes only instruction/input/output fields. These are the only data-row ranges authorized for this study; no other rows were read. [Task YAML](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/task/task.yaml) fixes alpaca format, max_length 512, batch 4, one epoch and no validation split. Those are source settings, not a claim that a run completed all expected steps.

### Label construction matters more than answer character length

- [Alpaca conversion](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/data/formats.py) creates one user turn from instruction plus optional input, followed by one assistant turn. Public prepared files have no separate system column.
- [Soup schema](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/config/schema.py#L419-L435) defaults to response-only masking; [format selection](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/data/sft_format.py#L48-L103) uses it when a chat template exists.
- The [Qwen tokenizer metadata read at main](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/tokenizer_config.json) includes an inserted default system prompt and ChatML user/assistant turns, but no generation markers. Main's [repository page](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/main) resolved to [989aa7980e4cf806f80c7fef2b1adb7bc71aa306](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/commit/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) during inspection. The exact pinned raw-file request failed, so this is explicitly **main-content plus contemporaneous revision evidence**, not a separately retrieved pinned-byte artifact. Actual run snapshot still must be recorded.
- [Soup's mask builder](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/data/loss_mask.py#L110-L151) rejects a preferred all-zero assistant mask and [falls back to cumulative render token deltas](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/data/loss_mask.py#L322-L361). This unmasking includes the assistant role-prefix and suffix tokens, not only output-text tokens. `train_on_eot=False` does not remove suffix tokens already included by this fallback.
- [Truncation and validity](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/data/loss_mask.py#L220-L247) truncate to the first 512 tokens and require at least one nonignored label at positions1 onward. [SFT row mapping](https://github.com/MakazhanAlpamys/Soup/blob/846888a045b31084452fa08a684a7a38c02f5f87/src/soup_cli/trainer/sft.py#L143-L172) applies this check before training; it raises rather than silently dropping no-target tokenized rows.
- [Pinned TRL collator](https://github.com/huggingface/trl/blob/617406a4e69551c115c145114497999618d24f3c/trl/trainer/sft_trainer.py#L164-L227) right-pads to batch maximum, keeps explicit labels, pads labels with −100 and applies assistant_masks. No cross-batch rearrangement or new padding policy is proposed.

### Source-derived equations and safe specialization boundary

For row i, let p_i be the cumulative user-only rendered token length, L_i the full rendered length, and n_i=min(L_i,512). On the inspected two-turn fallback path, labels are active at positions p_i…n_i−1, provided the row passes n_i>p_i and token rendering matches the assumed prefix layout. **Head** active positions are p_i−1…n_i−2 because the loss shifts labels. Thus U_i=n_i−p_i, not “number of assistant hidden rows after p_i”.

For an unchanged four-row batch, T=max n_i, N=4T and U=sum U_i. Successful source-validated rows imply U>=4; U<=4(T−1) and T<=512. These broad bounds do not establish the useful U/N<1/2 work crossover. Dynamic padding can substantially affect N. Original shuffled batch composition must be retained when calculating N; sorting by length to improve the number would change the task.

The contiguous-suffix geometry could reduce index-map overhead, but only after checking the actual shifted-label mask for that batch. Do not hardcode p from raw text lengths, a presumed constant header token count, or sample content. Keep a general original-row map fallback if the observed mask is not a single suffix. This is an implementation refinement of the same ID, not another catalog row.

## 3. Blocked statistic, exact cause and smallest next evidence

The documented [HF selective rows API](https://huggingface.co/docs/dataset-viewer/rows) supports explicit offset/length slices, at most 100 rows per request. The request for public row 0 returned an internal “not accessible via this tool” error. A revision-info API request also failed. The [pinned repository tree](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2/tree/c641407ee50723c84a43ea2ec867b464d3893c12) was readable, but contains one 35.6 MB JSON dataset file; it was **not downloaded**, because that would expose rows outside the authorized public split. No policy escalation, credential use, retry loop or alternative network bypass was attempted.

Consequently this cycle provides **no empirical token-length distribution, U/N histogram, truncation count or expected speedup**. No average was guessed from the visible example or overall dataset-card statistics. The next authorized data-only evidence would be a readable, revision-verified copy of the already designated public train.jsonl plus the run's tokenizer files, then the original formatting/collator path under the unchanged sampler. No model weights or training are needed. If those exact files remain unavailable, this part stays blocked while other source/CPU work continues.

## 4. CPU versus T4 proof boundary

| Can be established now | Still requires actual installed T4 stack |
|---|---|
| Positional NaN/+Inf/−Inf/finite masks under tested CPU CE/autograd; wrong-zero-scatter counterexamples; reference-order versus simplified-gradient distinction | CUDA CE dispatch/NaN behavior; gathered GEMM algorithm rounding; true scaler overflow and optimizer/scheduler advancement |
| Label-shift, right-padding and suffix-index equations from pinned Soup/TRL | Actual run tokenizer snapshot, dataset-derived batch histogram and exact shape routing |
| Centered vocabulary-state merge mathematics and CPU tail/tie tests from the separate CPU lane | Triton reduction order, compile/register behavior, guard synchronization overhead, own-share and full-command time |

## 5. Final cycle 2 CPU evidence — six checks passed, with negative findings retained

Inspected [cycle2/results.json](cpu_tests/cycle2/results.json) and [summary.json](cpu_tests/cycle2/summary.json), generated 2026-10-30 11:23:49Z, Python 3.12.14 / Torch 2.15.0+cpu / CUDA unavailable / two threads. Script SHA256: `c99335ce9f4267aaeeb2383c24f239496f5b73da727e935085b60b1b2b32d813`. The initial eight-test artifact remains separate and immutable.

- **Nonfinite row masking:** NaN, +Inf and all−Inf ignored-row logits each changed three positional gradient classes when compact zero-scatter was used. Mixed finite/−Inf and finite controls did not. Summed CE was finite in these witnesses, proving that finite loss is insufficient as a guard.
- **Nonfinite upstream:** +Inf incoming scalar or zero denominator changed the selected target from baseline NaN to naive−Inf. Reference-autograd compaction retained the baseline positional classes on the finite-logit fixtures. This isolates an analytic-backward failure from row compaction itself.
- **Effective weight:** a nonfinite entry in the effective FP16 head operand produced NaNs in a dense zero-gradient-row matmul; scatter zero did not. Finite FP32 source weights can become nonfinite on FP16 conversion.
- **Actual logits:** a head with finite rounded FP16 inputs and weights still produced overflowing ignored-row logits; dense hidden gradients contained two NaNs that compact scatter removed.
- **Centered vocabulary merge:** 900 finite CPU trials covered forward/reverse/balanced merge orders, noncontiguous rows, tails, empty tiles and global-index ties. Largest tested vocabulary257, not Qwen151936. Maximum absolute errors were CE1.90735e-6, entropy1.66858e-6, probability1.66893e-6 under explicitly exploratory tolerances, not challenge gates.
- **Important promotion caution:** the near-certain singleton-tile forward merge produced CE3.051711246e-5 versus baseline2.884823152e-5: about **5.785% relative error** despite passing the exploratory absolute tolerance. Reverse/balanced order matched that particular CE witness. This does not establish universal parity or a harness failure; it disproves any inference that centered algebra plus an absolute CPU tolerance guarantees relative-error admission. Add near-zero **relative** CE/rewind tests and compare merge order/two-pass reference-order designs before promotion.

“6 passed” means the fixtures established their stated positive/negative claims; it does not mean an optimized kernel passed competition checks. Existing IDs SOUP-NEW-001/002 and SOUP-TRAIN-001 remain conditional. No GPU, actual CUDA GradScaler or performance result exists for this cycle.
