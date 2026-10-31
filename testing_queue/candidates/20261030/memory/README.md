# Memory VS1 — prepared, unmeasured component candidate

**Delivered:** runnable notebook, actual `solution/`, deterministic queue ZIP, hash manifests, and CPU/static evidence. **No GPU or remote operation was performed.**

The requested tied embedding/head-storage idea (MEM005/NEW005) remains **BLOCKED**: complete alias ownership, FP32 semantic dtype, pre-PEFT provenance and save consumers are unproven. See `SOURCE_AUDIT.md`. Rather than install an unsafe table mutation, this lane uses the explicitly authorized distinct MEM001 safety revision below. It does not change the head lane and is not trial005 relabeled as successful.

## What is genuinely different

- Replaces the old weak-only wide-operand version check with a zero-storage shared-version witness. A detached FP32 alias mutation is detected even after the original wide Python Tensor has been collected.
- Declines before raw saved-slot replacement if the runtime cannot establish the witness contract. Every unpack checks both half payload and wide version witnesses.
- Flushes geometric cumulative activation checkpoints into the native train log so a timeout preserves evidence. Counts are lower bounds until normal exit, not measured kernel share.

The native Qwen2 MLP, all native bnb GEMMs, SiLU, backward graph, separate gradient edges, LoRA, optimizer, RNG and task stay unchanged. Only known gate/up saved operands retain their original FP16 producer storage; original FP32 values are restored for native backward. The supplied SwiGLU is not installed. The original restoration Triton kernel is useful decompression, not busy work or a claimed new arithmetic primitive.

**Hypothesis:** make the existing memory-component route robust to surviving versioned aliases and auditable after timeout. No speedup forecast. Its retained-state representation is one FP32 activation plus two FP16 operands rather than three FP32 tensors; logical component accounting is not measured external peak. Escaped aliases may retain the old wide allocation, so version correctness does not prove a saving.

## Evidence from this preparation

- Six tiny actual Qwen2/bitsandbytes CPU integration groups pass: output/gradient finite bits, nonfinite classes, graph/RNG/state, native fallbacks, saved hooks and ownership.
- Six new CPU groups pass: zero-storage witness/RNG, collected-wide alias mutation, half-payload mutation, storage-wrapper release, unsupported-contract decline, flushed timeout records.
- All 79 unchanged Soup regression tests pass in a separate staged CPU copy. The production route declines on the non-target CPU Torch build; this is regression compatibility, not GPU-path proof.
- Five CPU-only marker lifecycle fixtures pass: normal owned removal, scientific failure, raised failure, foreign-token protection, and abrupt SIGKILL retention.
- One real tiny CPU process-lifecycle fixture confirms interrupted separate-session grandchild cleanup and quarantine (no GPU).
- Five synthetic report fixtures enforce no score/admission for timeout, failed/missing eval, absent final routes and missing wrapper results.
- Notebook cell syntax, Python compilation, deterministic ZIP and every source hash are checked in `static_checks.json`.

The CPU environment uses existing Torch `2.15.0+cpu`, not target `2.11.0+cu128`. No CUDA compilation, target witness behavior, full training, quality, speed, GPU peak or useful share is validated. Prior trial005's 79.04-second timeout is preserved as history, not inherited evidence. This component's own Triton share may be zero or below 5%; it is not a standalone admission claim.

## Future explicitly authorized target use

Open `candidate.ipynb` on the prepared T4 environment, set only paths and `AUTHORIZE_GPU=True` after authorization. It never installs or downloads. It requires exact direct pins/Python, one visible Tesla T4, four assigned logical CPUs, idle GPU inventory, exact official commit/protected files, the two fixed public-data hashes and pinned cached model revision. Missing assets fail closed.

A dedicated Linux subreaper supervisor holds nonblocking `/tmp/soup-gpu0.lock` throughout the workflow; preflight and inventory run in short-lived children, with an idle recheck after they exit. The notebook never creates a CUDA context. PID birth identities and pidfd signalling protect interruption cleanup, including grandchildren in separate sessions. An owner-token active marker at `/tmp/soup-gpu0.lock.blocked` is exclusively created and fsynced **before any GPU work**, so abrupt supervisor death is covered too. Every launch checks it before and after locking. Only normal verified owned completion with no descendants removes its own unchanged token; failures, timeouts, interruptions, altered tokens and sudden kills retain the marker for explicit operator reconciliation. This is process hygiene for reviewed trusted code, not a hostile-code sandbox.

The notebook stages a new official copy and its embedded reviewed candidate, runs the extended target preflight and unchanged regressions, then the unchanged official harness with **base, solution, solution, base**, seed 1, 64 steps, equal 900-second native budgets. Fresh processes/state/output/compiler caches per measured run are supplied by the harness. Preflight caches are separate. Complete native command wall time, external nvidia-smi peak, plain public quality and post-two-warmup own-kernel timing remain separate. All failures keep score null and admission UNKNOWN. Final official Kaggle admission is separate; GCP needs its own baseline and does not inherit a Kaggle 79-second limit.

For local CPU checks only (no install):
```sh
SOUP_CPU_PYTHON=/path/to/existing/cpu/python sh run_cpu.sh
```

## Queue handoff

`candidate.zip` contains only the exact manifest-listed `solution/` files. The notebook is a reproducibility/paired-measurement artifact, **not** a queue executable. `submission.template.json` leaves its GCP baseline unresolved. No job is submitted here. A reviewer must inspect source, target validation and current run authority before approving the ZIP. `source_manifest.json` and `bundle_receipt.json` bind exact bytes.

## Files

- `solution/version_sentinel.py`: new safety mechanism.
- `solution/adapter.py`: guarded native producer observation and saved slots, with new progress records.
- `solution/restore*.py`: prior original useful widening kernel, unchanged.
- `solution/verify_memory.py`: future target preflight, now including CUDA witness behavior.
- `notebook_runner.py`: transparent notebook orchestration, outside queue ZIP.
- `cpu_results.json`, `version_results.json`, `runner_fixture_results.json`, `regressions.log`: exact new evidence and limits.
- `source_audit.json`, `SOURCE_AUDIT.md`, `provenance.json`: audit and original-code lineage.
