# NF4 blockpair v1 — prepared, not GPU validated

**Implemented now:** independently authored block-local nested NF4 decode, byte-pair mapping,
aligned packed FP16 stores, plus both actual native decode entrypoints. This is a research candidate,
not an admitted submission. No GPU/model training, queue submission, remote write or installation ran.

## Hypothesis and originality

The old trial002 kernel assigned one lane per output weight, repeated packed-byte/scale accesses,
and used scalar stores. Trial004 reused that kernel on only the forward fallback. This candidate
instead assigns a lane to each byte, reconstructs one FP32 scale per64-weight block in a16×32
scale/byte tile, broadcasts that scale, independently decodes the two nibbles with a balanced table
selection, and writes an aligned FP16 pair as one32-bit word. It also covers both existing forward
fallback and functional/backward boundaries in one coherent patch; it does not force native fused
GEMMs onto a new route. FP32 output and unaligned/tail diagnostics use masked scalar stores.

The source was written for this packet, not copied from a supplied implementation or library kernel.
NF4 constants are public format data. The mechanism follows the audited packed-store research
proposal, not a claim of discovering NF4 itself. The wrapper contracts were checked against the actual
locally installed bitsandbytes0.50.2 source and prior native-route prototypes.

This should reduce redundant decode work and avoid vector-interleave shared-layout conversions.
It may still be too small or fail the≥5% useful-original-share rule, especially if it gets faster.
There is **no estimated additive share, performance prediction or demonstrated memory saving**.

## Exact contract and bounded dispatch

- Fixed Qwen shapes `(1536,1536)`, `(256,1536)`, `(8960,1536)`, `(1536,8960)` only.
- Tesla T4 SM75, packed contiguous uint8 canonical `[ceil(N*K/2),1]`, frozen weight metadata.
- Outer NF4 block64; nested scale block256; supplied256-entry FP32 dynamic code table.
- Global flat indices do not reset at matrix rows. High nibble produces the even weight.
- Scale `round32(code2[q]*abs2[j//256])`, then separate FP32 offset add; FP fusion disabled.
- Forward decodes directly to `A.dtype` (FP16). Functional/backward decodes to `state.dtype`
  (FP16 or FP32); unchanged native backward performs its own later gradient-dtype cast.
- Native F.linear, bias, GEMM dispatch, MatMul4Bit autograd, LoRA/dropout/RNG and optimizer unchanged.
- Unsupported metadata, dtype, output buffers, transposed BC-shim inputs, versions, or existing
  wrappers delegate unchanged. No broad monkeypatch of PyTorch operators or trainable gradients.
- Fresh output allocation with native lifetime. No persistent dequantized weights/scales or workspace
  aliasing, no compiler cache shipping, no autotuning, no changed task parameters.
- Flushed logarithmic stdout route receipts survive timeout; these are not own-share measurements.

## What is actually validated

`tests/cpu_results.json`:48 actual CPU Triton-interpreter fixtures and5 fallback checks passed.
These cover all256 packed bytes, both output dtypes, 64-weight and256-scale boundaries, odd tails,
misaligned FP16 output views, sentinels, signed-zero/finite bits and nonfinite classes. NaN payload
identity is not claimed. These use local Torch2.15.0+cpu/Triton3.9.0, **not** target pins.

`tests/offline_compile_results.json`:4 explicit SM75 compilations without GPU loading/execution passed
on local Triton3.9.0. All report0 shared bytes, separate FP32 multiply/add, no FP32 FMA, and32-bit
stores for aligned packed variants. This is **not** target Triton3.6 compilation or T4 execution.
No compiler binaries/caches are shipped. An initial diagnostic substring scan missed `.rn` in PTX;
its report is retained, and the corrected opcode regex produced the final report. Kernel unchanged.

`static_validation.json`: notebook JSON/cell syntax, all Python syntax, deterministic bundler bytes,
manifest/ZIP/source agreement, official-source hashes, declared native/autograd boundary scan.
The notebook has cleared outputs. GPU preflight/full training were **not run**. The CPU interruption
fixture exposed asynchronous SIGKILL completion during a repeat; its failure log is retained as
`tests/static_validation_sigkill_race.log`. Cleanup now waits boundedly for tracked identities to
become absent/zombie; final94 checks pass, and interruption still always quarantines the lane.

## Historical negative evidence stays negative

Trial002 backward-only: own-share2.5% (<5%), about0.991× command ratio; standalone failed.
Trial004 forward-only: minimum own-share2.7448%,0.9538× (slower),14345MiB unchanged; standalone failed.
These are earlier candidates, not evidence this implementation is faster or admitted. Their shares
cannot be added. Current T4 speed, memory, full quality, own share and final admission are unknown.

## Files and future authorized use

- `candidate.ipynb`: standalone embedded-source research notebook. Default authorization gate is off.
- `solution/`: actual reviewed implementation; `candidate.zip`: worker artifact, built with queue bundler.
- `source_manifest.json`: exact file hashes. `submission.template.json`: unresolved GCP baseline.
- `gpu_preflight.py`: prepared exact native output/input-gradient/bias-gradient checks on full task
  shapes and actual fused/fallback M boundaries, both state dtypes; not executed.
- `notebook_driver.py`, `support/`: transparent source embedded in notebook; no installation/download.

For a separately authorized run, supply the already-prepared official checkout at commit
`c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`, existing public model cache and new output parent.
The notebook verifies exact direct package pins/Python3.13.15, public data hashes and model revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. It copies only pinned official source into a fresh isolated
stage, replaces `solution/`, and leaves the prepared checkout untouched. Exact T4 preflight must pass
before unmodified `harness/run_task.py --repeats 2 --seed 1 --timeout 900` starts. The native timeout
is900s per command, outer budget10800s; neither is the historical Kaggle79s screening threshold.
A fresh complete baseline is measured in the same environment and paired order. A nonblocking
`/tmp/soup-gpu0.lock` flock protects the whole preflight/paired run. GPU inventory and preflight
run only in exiting subprocesses; the notebook parent never initializes CUDA, and GPU idle state
is rechecked after preflight before measurement. Timeout and KeyboardInterrupt cleanup track
owned PID birth identities and use Linux pidfds to signal only those processes; CPU fixtures verify
detached-child cleanup, SIGINT handling and stale/exited PID guards. Any outer timeout/interruption
retains `/tmp/soup-gpu0.lock.blocked` pending explicit operator reconciliation, even if observed
children were killed: a process census cannot prove absence of an unobserved orphan. Do not
delete that marker or retry automatically. An owner-token marker is fsynced before any attempt
child starts and cleared only on normal verified completion. Therefore SIGKILL/kernel death
leaves the lane blocked even if the OS releases flock. SIGTERM is converted to the same Python
cleanup path; its previous handler is restored afterward.

The harness captures complete native command wall time, external nvidia-smi memory sampled every100ms,
public quality, and own-kernel events after2warmup steps. It creates fresh processes/model/optimizer/
outputs/compiler caches per measured run. The driver retains route logs, all nested results, checker
output and hashes. Failed/partial/timeout attempts never get a score or admission PASS. Scores remain
null even on complete research pairs; all10 checks are preliminary and hidden quality/final official
Kaggle admission remain separate. GCP must register its own measured baseline before queue submission.
The queue executes ZIPs, not notebooks; nothing is submitted by this packet.

## Source evidence inspected

- Local clean official checkout: `/workspace/shared/soup_cloud_env/source` at pinned commit above.
- bitsandbytes0.50.2: `functional.dequantize_4bit`, CUDA `_dequant_linear_fallback`, `MatMul4Bit.backward`.
- `soup_gpu_trials/trial002/solution/nf4_decode.py` and verified report.
- `soup_gpu_trials/trial004/solution/forward_nf4.py` and verified report.
- `soup_audit_1919/handoff.json`, `HANDOFF-NF4-PACKED-STORES` / canonical `SOUP-NEW-003`.
- `soup_user_findings_20261030/batch3/Research_Update(2).md`, lines70–88.
- `soup_cycle/research_nf4_shape_cycle8.md`: block-local reuse versus unsafe post-dot scaling.

Preparation stops at a source-checked research packet. The missing proof is authorized target
Triton3.6/T4 native preflight, unchanged Soup regressions, paired64-step training/evaluation,
useful original share and official admission. No claim of readiness to submit without that proof.
