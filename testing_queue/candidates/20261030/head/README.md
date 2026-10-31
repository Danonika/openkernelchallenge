# Head-memory candidate: private FP16 state, native CE replay

**PREPARED RESEARCH CONTROL — T4 UNTESTED — NOT A CONTEST-ADMISSIBLE STANDALONE SOLUTION.**

This is actual runnable optimization source, not a placeholder. It changes the
loss graph's retained state while deliberately keeping the native head and every
returned logit. It has **zero original Triton kernels**, so official checks 3/4
cannot pass. No speed, peak-memory, quality, score, or admission claim is made.
The full H0/output-suppression/summary-only variant is **BLOCKED**, not silently
represented by this narrower implementation.

## Hypothesis and exact mechanism

For supported training calls, retain the *actual already-rounded native FP16
head logits* and causal targets instead of the native CE graph's FP32
log-probabilities. Compute the scalar with native full-geometry summed CE in
forward; leave the original supplied division outside the custom node. During
backward, recompute the same native summed CE from the saved half logits and
feed its actual incoming (already normalized) upstream scalar into native
autograd before FP16 gradient rounding. There is no custom softmax, analytic dZ,
chunked CE sum, selected-row replay, changed GEMM, or model/head replay.

Expected benefit is **less retained loss-graph state before backward**. Native
replay adds real work and can increase the backward peak. Returned FP32 logits
and native SFT entropy/accuracy/Rewind remain present. This is not the large
memory-saving H0 claim and not a promised complete-run gain.

## Semantic boundaries

- Runtime `apply()` checks Torch 2.11.0+cu128 and Transformers 5.16.1, installs
  only on the exact native Qwen2 constructor/property, and attaches a reversible
  per-instance loss wrapper. No global loss map, forward, Trainer, optimizer,
  task, sampler, LoRA, dropout, scaler, output object, or RNG is patched.
- Supported path: actual contiguous FP16 `[B,T,151936]` logits, matching same-
  device int64 labels, a frozen unbiased head, CUDA SM75, grad-enabled training,
  hard causal labels with ignore index -100, and an integer or scalar int64
  denominator. A CPU scalar denominator follows the native `.to(loss.device)`.
- Ordinary `output_attentions`/`output_hidden_states` bool/None kwargs are
  accepted because native Qwen already consumed them and the pinned loss ignores
  them. All other extra arguments, alternate targets/layouts/precision, eval,
  mean reduction, trainable head and unsupported routes call the captured native
  loss **on the same logits**, never by retrying forward after RNG consumption.
- Full B*T geometry remains, including the final ignored row of each example.
  Ignored NaN/Inf rows, zero/negative denominators, nonfinite upstream gradients,
  and connected-zero gradients are not sanitized. Native FP32 CE defines them.
- Saved logits must remain immutable until backward. PyTorch version checks are
  retained; mutation parity outside that contract is not claimed. Higher-order
  replay exists but is unvalidated and not part of the fixed first-order task.
- `SOUP_HEAD_NATIVE` metadata is flushed before/after every forward/backward,
  including fallback reasons. It stores no activations. These lines survive a
  process timeout; their overhead counts. The unchanged official harness still
  owns all Triton launch/share measurement. Native replay is never called an
  own-kernel launch.

## Why full H0 and summary-only finish are blocked

Reviewed local recovered Round 7 sources, especially `round6_integration/h0_core.py`,
`round7_loss_integration/{README.md,forward_stats_backend.py,summary_stats.py}`,
and `round7_composition/autocast_boundary_results.json`, plus the fresh
`soup_audit_1919/{handoff,lora,qwen}.json` audits. These are provenance/evidence
inputs, **not kernels copied into this solution or credited as ours**.

1. Omitting output logits requires an owned SFT/Rewind summary-consumption
   protocol, single-forward ticket, unchanged row identities/counts and lifecycle
   through the outer Rewind recorder. A safe actual CUDA/PEFT/NF4 adapter is not
   established by this preparation. Assigning `outputs.logits = None` alone can
   leave a dense tensor in the ModelOutput mapping; a real reconstruction and
   complete consumers would be required. This candidate does neither omission.
2. Accelerate can enter autocast *inside* prepared model forward while trainer
   entry is outside it. The recovered CPU BF16 probe demonstrates this boundary;
   merely changing an outer CPU autocast guard to CUDA is unsafe.
3. Supplied summary-only finish removes four unused per-token outputs (32 KiB
   at 2048 positions), not the half-logits cache or native replay. Its two strict
   tie-entropy failures remain: absolute errors 2.288818359375e-5 (tile 1024) and
   2.193450927734375e-5 (tile 4096). Supplied V37 loop agreement, interpreter and
   SM75 offline compilation do not establish actual full-task CUDA arithmetic,
   share, speed or peak. No supplied claims are relabeled as our new validation.

This loss-boundary control is therefore distinct from both the full H0 hybrid
and the separate materialized/analytic-CE lane. A future original summary bridge
would need its own implementation, frozen numerical gates and target evidence.
The algorithm also follows our earlier local
`soup_cycle/private_half_native_control/control.py` research control; this packet
adds a task constructor adapter, eager telemetry and a newly executed 24-case
suite. The earlier independent 22-case control results remain separate and were
not overwritten or relabeled as this packet's results.

## Files and running later

| File | Purpose |
|---|---|
| `candidate.ipynb` | Cleared, self-contained notebook with exact ZIP + driver bytes |
| `solution/` | Actual reviewed source; only this directory is judged/executed |
| `candidate.zip`, `candidate-files.json`, `bundle-info.json` | Deterministic worker bundle and exact hashes |
| `run_candidate.py` | Offline fresh-stage paired official-harness driver |
| `official-source-manifest.json` | Pinned unchanged harness/task/tests + other non-solution source |
| `submission.template.json` | Queue-compatible template; baseline ID intentionally unresolved |
| `validation/` | Source-bound focused CPU/unit/static results, not GPU measurements |

The notebook's first configuration cell requires an already prepared official
checkout, existing dedicated public model cache, a new output path, and explicit
GPU-run authorization. Default GPU execution is disabled. No installation,
download, remote edit, queue submission, or GPU run happened during preparation.

After separate execution approval, configure the notebook and Run All, or use:

```bash
python run_candidate.py \
  --repo /absolute/prepared/pinned-official-checkout \
  --cache /absolute/prepared/public-hf-hub-cache \
  --out /absolute/new/head-experiment-01 --backend GCP --execute
```

Requires Python 3.13.15 and the exact direct package pins recorded in the queue's
`requirements-kaggle.txt`: soup-cli 0.77.0, Torch 2.11.0+cu128, Triton 3.6.0,
Transformers 5.16.1, PEFT 0.20.0, TRL 0.29.1, bitsandbytes 0.50.2,
Accelerate 1.14.0 and datasets 4.8.5. This is not a full transitive freeze.
The driver verifies one visible Tesla T4, SM75/CUDA12.8, summed physical idle
memory <1000 MiB, public data hashes, and model revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` before training.

It creates a new staging repository from hash-checked official commit
`c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`, replacing only `solution/`.
It reuses public model/data assets offline and invokes unchanged
`harness/run_task.py --repeats 2 --seed 1` in base/solution/solution/base order.
Each measured process has fresh model/training state, outputs, Soup DB and
compiler caches. The native harness captures complete command wall time,
external 100ms nvidia-smi peak, public quality, outputs and steps3+ kernel share.
The driver additionally records whole-orchestrator wall time separately.
The canonical nonblocking `/tmp/soup-gpu0.lock` covers the entire paired run
and checks, preventing overlap with the other prepared notebooks. GPU preflight
is a short-lived child; it exits before the idle check. The driver/notebook
keeps no live CUDA context during measurement.
`/tmp/soup-gpu0.lock.blocked` is checked before and after acquiring the lock.
An isolated Linux subreaper guardian owns nested-session children and signals
only PIDFD-bound descendants; it inherits the lock and watches a parent-liveness
pipe. Timeout, interrupt, parent death or unexpected descendants creates the
quarantine marker and cleans/reaps owned children before releasing the lock.
Repeated notebook interrupts wait for cleanup. Markers are never auto-cleared;
an operator must reconcile before another GPU run. This is bounded trusted-task
orchestration, not a general hostile-code sandbox.

It runs unchanged official checks after complete paired train/eval evidence and
preserves their expected own-kernel failures. Failed/partial/timeouts get no
quality comparison, ratios, scores or admission PASS in `research-result.json`.
The untouched raw official result remains evidence, not a safety-filtered result.
No score is ever assigned by this driver, even after a completed control.

**GCP uses its own freshly measured baseline, never the historical Kaggle ~79 s
limit.** The default native budget is 900 s on both sides, fixed before running;
the whole paired-harness watchdog is 8100 s. GCP evidence is not official Kaggle
admission. The worker accepts the ZIP, not execution of notebook cells; leave
the queue template untouched until a reviewer provides the actual registered
same-environment baseline ID and separately authorizes a research run.

## Validation completed here

`validation/cpu_results.json`: **24/24 CPU checks passed**, Torch 2.15.0+cpu,
Python 3.12.14, Transformers 5.16.1. Finite loss/FP16-gradient values and signed
zeros matched exactly in tested cases, with matching NaN/+Inf/-Inf positions.
Cases include full V151936 at B4/T3; T512 at V19; T1/all-ignored; large/nonfinite
incoming scale; zero/negative/tensor denominator; native fallback/error identity;
saved-state inventory and denominator version checks.

A **tiny real Qwen2 CPU** fixture additionally keeps native output mapping,
head output, loss, input gradient, dropout RNG and state-dict structure equal,
uses exactly one native head forward per side and verifies eval fallback plus
restore. It is not the actual prepared PEFT/Accelerate/NF4/TRL/Rewind stack.

Saved-storage inventory is strictly a local retained-autograd-state observation,
not process peak. Missing target proofs: actual CUDA native-loss dispatch,
T4 numerical/AMP/GradScaler behavior, real PEFT/TRL/Rewind integration, complete
64-step paired runs, quality, external peak, whole-command time and final jury
review. Own Triton participation is structurally zero, not merely unmeasured.

No GPU or remote service was touched. No older failed prototype was overwritten.

`validation/process_guard_results.json`: **7/7 CPU process fixtures passed**:
nested-session timeout, SIGINT, SIGTERM, arbitrary BaseException, abrupt parent
SIGKILL, orphan reparenting, and normal completion. Failure fixtures verify that
the nested CPU child is gone, quarantine remains, and the inherited lock is
released only after cleanup. Fixtures use temporary locks/markers, not GPU work
or the canonical quarantine file. The shared PIDFD/subreaper helper is attributed
to the contemporaneous LoRA lane; it is not solution code or kernel credit.
