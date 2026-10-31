# One-bit hypothesis: actual-weight and first-batch probes

The user's proposal is a research hypothesis, not verified information about competitors. This bundle tests one specific construction: one sign per weight, one FP16 mean-absolute scale per 64 weights, optional exact exceptions, and reusable preparation of frozen weights. It does not establish the 45-second / 1,359-MiB target.

## Real first training batch

Native Soup loading, tokenizer, sampler, collator, FP16 setup, q/v LoRA configuration and loss implementation are used. Every comparison replays the same CPU/CUDA/Python/NumPy RNG state and clones the same inputs. Batch shape is [4, 235], with 392 supervised shifted tokens. The first native loss is reproduced exactly at the end. No backward or optimizer update occurs.

| Variant | Loss | Absolute relative loss gap | 1% diagnostic band |
|---|---:|---:|---|
| native_nf4 | 2.05893254 | 0.0000% | within |
| dense_native_nf4_control | 2.05893254 | 0.0000% | within |
| one_q_projection_layer_0 | 2.09730649 | 1.8638% | outside |
| last_four_decoder_layers | 16.15348053 | 684.5561% | outside |
| all_decoder_projections | 16.36842346 | 694.9956% | outside |
| native_nf4_repeat | 2.05893254 | 0.0000% | within |

These are forward diagnostics on one public training batch, not the official first-step check, a held-out evaluation, or a completed 64-step run. The dense native NF4 control checks the replacement linear path with the original dequantized weights and biases. All binary numerical matrices are expanded FP16 references, so these runs measure neither compressed storage nor a fast kernel. The first attempt failed because the test incorrectly required bias-free projections; its source, notebook, result and hashes remain in the bundle. Version 2 preserves the native biases and adds the native-forward control. No quality tolerance was relaxed.

## Storage/error study

The CPU codec passed 35 round-trip/error-bound fixtures plus the golden NF4 nibble-layout check. The T4 probe sampled 2,048 aligned blocks from each of eight real decoder matrices, and 2,048 blocks across three embedding-row windows. Native whole-tensor NF4 reconstruction was bit-exact for all eight decoder samples.

For sampled decoder coefficients, plain binary approximation had 60.56–63.41% relative L2 error in **normalized NF4 codebook space**. This is not scale-weighted aggregate error of the actual tensor. The two separate projection probes do use actual dequantized weights, but synthetic inputs. Neither result is evaluation loss. This exception format selected native 4-bit storage for all eight decoder samples through a 20% normalized block-error budget because corrections and indexing erased its savings. That finding applies to this codec and these samples, not every binary representation.

A sign bit plus an FP16 scale per 64 weights costs 1.25 bits per weight. Adaptive byte counts additionally include mode bytes, 32-bit offsets, exception bitmaps and exact exception values. Original NF4 double-quantization scales are shared and excluded from both sides of the adaptive storage comparison. No reported diagnostic duration or host peak is a contest performance result.

## Precomputation and remaining evidence

Weight preparation occurs inside each diagnostic and no model tensors or output lookup tables are exported. A production implementation would need a real packed layout, complete metadata accounting, measured preparation and decoding costs, backward/LoRA-update checks, all 64 steps, held-out quality and the unchanged official checks. Passing a loss tolerance alone does not establish admission under the fixed NF4 task and source-review requirements. The bundle makes no jury or competitor-method claim.

## Reproduction and provenance

The public checkpoint is `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. GPU scripts require one Tesla T4 and PyTorch `2.11.0+cu128`; the scripts use the prepared public model cache and unchanged Soup task. `SOUP_REPO` and `SOUP_RESULTS` are supplied by the diagnostic runner. Notebooks contain their exact source and SHA256 checks. The build JSON records preparation state; terminal states are in the result and binding JSON.

Run `python cpu_probe.py` for the CPU codec study. The three included notebooks reproduce the weight study, the preserved failing first-batch attempt and its corrected version. They are research notebooks, not final submission notebooks. Every `*_result_binding.json`/`result_binding.json` connects a submitted notebook, verified output-archive SHA256 and exact JSON member. Submitted and executed code cells were compared byte-for-byte. `manifest.json` hashes all curated files. Credentials, private service identifiers, raw execution logs, model weights and output archives are excluded.
