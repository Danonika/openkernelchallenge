# All-research GPU experiments — 31 October 2026

Goal: one valid run below 45 seconds and 1,359 MiB. This target is not established by the archived experiments. These are development candidates under the fixed 64-step Qwen2.5-1.5B NF4 LoRA task on one Tesla T4. GPU status and observed measurements are in `evidence/`; source and notebook bytes are immutable. Notebook opening notes describe their pre-execution preparation state.

Verified fast_v2 result: **62.71 seconds / 7,955 MiB**, 64 steps, public eval loss 2.04968. All 452 candidate checks and 79 unchanged Soup regressions passed. The saved-baseline public-quality comparisons are within the official thresholds; they remain unpaired. Descriptive original-kernel share is 15.04196%, excluding the starter. The target remains unmet.

Memory v3 **failed the 80-second training limit**, with 46/64 steps logged and no evaluation. The 2,179-MiB observed peak belongs to this censored run and is not a completed benchmark. Its five CUDA offload checks passed. The composition screen retained four finite LoRA update failures under its original thresholds; no strict numerical pass is claimed.

- `fast_v2/`: recovered R11 fusions, eager exact tied-table retention, and a fixed 1-GiB decoded-NF4 cache. The cache size is not total measured VRAM.
- `memory_v2/`: original streamed CE/entropy/argmax moments, eager exact tied-table retention, compact MLP retention, and same-dtype saved-activation offload.

- `memory_v3/`: the same memory implementation with a complete native backbone source pin, including @merge_with_config_defaults from official Transformers v5.16.1. The v2 five-check CUDA copy probe passed, but its incomplete source pin stopped composition before training. No numerical code or threshold is changed.

The original failed notebooks remain in `previous_attempts/`. Fast v1 passed 450 checks but two route checks expected the retired embedding kernel. V2 checks the native embedding route, exact values, FP32 output and preserved table storage; numerical tolerances are unchanged. Memory v1 stopped before CUDA because its official source manifest was omitted. V2 restores that manifest with byte-identical candidate code.

A completed notebook is insufficient evidence of completed nested training. A single public screen is unpaired, does not certify hidden quality or jury admission, and earns no invented score. Original-kernel share excludes the unchanged starter SwiGLU and still requires manual originality review. Retained finite LoRA numerical warnings in the memory preflight are not strict numerical passes.

SHA-256 hashes in `MANIFEST.json` bind every source file and exact submitted notebook. No credentials, raw logs, account identifiers, weights, adapters, compiler caches, or final platform submission are included.
