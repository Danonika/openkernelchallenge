# Source audit and lane decision

## Tied embedding/head storage (MEM005 / NEW005): BLOCKED, not implemented on target

The pinned official checkout is `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`; protected source hashes are in `official_manifest.json`. Inspected source/evidence identities are in `source_audit.json`.

- Installed Soup `trainer/sft.py:1611–1756` locally imports PEFT preparation, loads the native model, then applies vocabulary expansion **before** `prepare_model_for_kbit_training`, and only later constructs LoRA. `utils/gpu.py:634–668` selects FP16 loading on pre-Ampere frozen-base CUDA. A proposed original-half capture must therefore verify the actual post-vocabulary-expansion/pre-PEFT object and values; merely loading a half checkpoint or assuming an older reference survives is not proof.
- The official README describes PEFT keeping the tied embedding/head and norms/biases in FP32. PEFT itself is not installed in the local CPU environment, so its preparation implementation was **not independently re-read here**. Retained NEW005 review is historical source evidence, not a new runtime observation.
- Installed Qwen2 declares the tie `lm_head.weight -> model.embed_tokens.weight`; its embedding returns the native gathered dtype and its head uses native `Linear`. Halving the table changes embedding outputs unless they are explicitly widened back to FP32. The head's effective FP16 autocast operand does not authorize replacing the FP32 source globally: non-FP16 consumers/fallbacks may require the exact FP32 values.
- Installed `transformers/modeling_utils.py:912–916` derives `model.dtype` from the first floating Parameter. A tied-half replacement changes this observable property, including nested owners. A property facade cannot intercept every direct parameter-dtype helper, state dict, save wrapper or unknown consumer.
- Same-Parameter aliases preserve a tie but do not prove storage exclusivity. The retained independent NEW005 fixture demonstrates an extra FP32 buffer keeping the old storage alive alongside a new half table. Distinct Parameter aliases, registered buffers and ordinary Tensor aliases must all be addressed. A complete alias/consumer audit cannot be established by a superficial `named_parameters()` scan.
- Soup's final save uses `trainer.save_model`; the official plain evaluator imports no candidate and reloads a standard LoRA adapter. That plain load does not prove that a changed training-side semantic dtype would preserve saving behavior.

No actual T4 table was observed, no full tensor representation scan was performed, and no target mutation was attempted. The tied-storage hypothesis remains blocked pending actual representation evidence, complete dtype/save consumers and exclusive ownership. This package does not install the prior CPU-only tied-table lease or pretend a no-op is that optimization.

## Prepared alternative: Memory VS1

Parent authorized a distinct improved MEM001 producer-edge candidate if tied storage could not be proven. VS1 retains the prior original native-Qwen/bnb codec math, but fixes a specific safety hole and its evidence loss:

1. A weak-only reference cannot check an escaped detached alias after the widened Python tensor dies. A new **zero-storage shared-version witness** retains the version counter without the wide storage or autograd graph. Only the detached witness's `.data` is replaced; model/computation tensors are not modified. Runtime behavior is probed on CPU during install, checked for each witness, and additionally required on CUDA in the prepared preflight.
2. The native saved slot is not hooked if witness creation is unsupported. Both payload and wide version are checked before unpack. Native `.data` bypass mutations remain outside the version guarantee, exactly as with ordinary PyTorch version checks. This does not legitimize arbitrary foreign hooks; their existing native fallback remains.
3. Original restore work remains unchanged, separate from the starter SwiGLU. Cumulative flushed progress records at powers of two survive process termination; they are lower-bound activation counts, never completed totals or own-kernel milliseconds.

This is a safety/evidence revision, not a new measured speed optimization. The prior trial005 timed out at 79.04s after stdout 63/64 and has no completed quality/share result. VS1 receives none of that trial's numerical, timing or memory claims. The new CPU groups establish only the stated CPU contracts. The new target preflight, paired fixed-task T4 run, public quality and useful original Triton share are unexecuted.
