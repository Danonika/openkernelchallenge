# Soup CPU cycle 4 — guarded head integration and actual NF4 CPU oracle

**7 fixture groups passed; 1 actual NF4 CPU parity group failed.** The failure is retained, not converted into a passing counterexample test. Prior **23 fixture groups remain unchanged**. No performance-graph points or nodes were added.

## Actionable actual-backend failure

Installed bitsandbytes 0.50.2 CPU decoding of shape **[3,80]**, blocksize64, packed NF4 endpoint +1 everywhere and flat scales **[1,2,3,4]** disagreed with both independent flat-format decoders at **48/240 elements**, maximum absolute error **1**. The **same packed bytes and scales** with metadata shape **[1,240]** decoded correctly; reshape-only comparison reproduces the discrepancy. Controls **[3,64], [3,65], [3,66]** also matched.

This is consistent with the researched CPU row-scale alignment hazard. It is a **CPU oracle limitation**, not a CUDA result or evidence about Qwen's block-aligned dimensions. The oracle was not changed to reproduce the bug, no library was patched, and no additional kernel package was installed. The combined reproduction command intentionally returns **exit 1** while this parity failure remains.

## Head/metrics integration — four groups passed

- **25 finite integration cases** matched exact native scalar CE, detached metrics, hidden gradients and the FP16 dHidden boundary. They cover mixed causal masks, an ignored sequence, odd vocabulary sizes19/31, ties, global denominator distinct from local supervised count, native mean, scales .25/65536, and one immutable scalar int64 tensor denominator.
- Gathered recomputation produced exactly matching selected FP16 logits in these CPU cases. Actual leaf/head-input strides are recorded; this is **not** a claim that the attempted slice remained noncontiguous or that gathered/dense GEMM geometry will match on a GPU.
- **18 exceptional cases** deliberately route to full-original-geometry native CE/autograd: ignored/padded/final NaN or Inf, finite-operand head overflow, effective FP16 weight overflow, nonfinite weights, incoming Inf/NaN, zero/nonfinite denominator, normalized-upstream overflow, all-ignored batches, T=1, and connected-zero upstream with nonfinite logits. Finite/NaN/+Inf/−Inf **positions** match the dense reference. Finite loss alone is not a valid fast-path guard.
- T=1 preserves empty accuracy/rewind domains (zero counts/losses), full-token entropy, and native all-ignored scalar-loss behavior. A final ignored NaN can leave main loss and rewind finite while contaminating main-loss backward; trimming the final position changed two hidden-gradient classifications.
- Explicit entropy masking preserves `0 × NaN` contamination. Mixed finite/−Inf logits also retain NaN entropy from the specified `exp(logp) * logp`; no mathematical zero convention is silently substituted.
- A synthetic metric-packet FIFO rejects duplicate, stale and out-of-order training packets, preserves row IDs, and leaves training FIFO/cumulative token count unchanged during eval. Two train packets count **30 attention tokens**, while each has only **6 supervised causal targets**. Averaging row means does not reproduce global CE.

**Scope:** forward intentionally materializes dense logits and shares the exact native metric helper with its reference. This is a guarded integration oracle, **not independently optimized metric reductions, streamed-forward implementation or demonstrated memory saving**. Selected forward logits are retained only as diagnostics. Full hidden state and tensor references are retained for backward-only guard failures; tensor denominator metadata is saved through autograd. No actual Soup/TRL hook/FIFO integration is claimed.

## NF4 metadata and actual QuantState — three groups passed, one parity group failed

- Independently written scalar decoding uses explicit binary32/binary16 rounding; the vector decoder uses independent bit extraction and flattened block gathers. Canonical 16-value NF4 data is taken from the [pinned format table](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/backends/utils.py#L15-L34), not a copied kernel.
- All **16 NF4 × 256 nested-scale indices** were covered with both the actual public `create_dynamic_map()` table and a separately labeled synthetic nonlinear table, FP16/FP32 outputs, flattened 64/256-scale boundaries and a 16,449-element odd tail. These manually constructed QuantStates matched actual CPU decode in the tested odd-last-dimension path.
- An additional **aligned [128,128] FP16 actual-table control**, 16,384 elements, matched both independent decoders and actual CPU output exactly. This complements the odd-tail generic-path fixtures without establishing CUDA behavior.
- Six tiny tensors at shapes **[2,64], [3,65], [3,66]**, each FP16/FP32, were actually quantized on CPU with nested statistics; their returned QuantStates decoded exactly against both independent formulations. This closes the earlier absence of a genuine runtime-produced state while remaining CPU-only.
- Canonical packed columns, packed-row compatibility transpose, byte-reinterpreted float storage, and the one-byte ambiguity were checked. A packed **[1,1]** buffer with state shape **[1,2]** actually returns **[2,1]** due to the API's transpose rule; it was not silently “fixed.” Outer `QuantState.code` is not the NF4 selector; `quant_type` chooses the canonical table.
- Backward decoding to state FP16 **then** converting to FP32 changed the resulting input gradient versus illegally decoding directly to FP32. Forward requested output dtype and backward state-dtype→gradient-dtype therefore remain distinct. Generic inner-block64 decoding was checked separately from the unexecuted MatMul4Bit-forward requirement for inner-block256.
- The actual 256-entry dynamic table is saved as uint32 bit patterns plus SHA-256 in `actual_dynamic_codebook_bits.json`. No default table was independently reimplemented or guessed. The optional actual-table FMA search and frozen-scale cache invalidation were not added.

## Authorized environment change and reproducibility

Tests used bitsandbytes **0.50.2** and Torch **2.15.0+cpu**. Only the authorized bitsandbytes add-on changed; private installer/wheel/native-library provenance is retained locally and intentionally omitted from this publication.

The installed CPU library loads. Public source pins remain in `nf4_cycle4/source_manifest.json`; private installation/native-library provenance is omitted. `TORCHDYNAMO_DISABLE=1` keeps this bounded CPU oracle eager.

Import warns that optional CPU `gemm_4bit_forward` requires missing **kernels**. That package was **not installed**, the optional path was not enabled, and no broader environment changes were made. The cloud environment HANDOFF now documents this add-on and the CPU shape limitation. Original regression results were not rerun or revalidated after installation.

```bash
python cpu_tests/cycle4/head_oracle.py
python cpu_tests/cycle4/nf4_oracle.py
# Expected exit: 1, preserving the actual [3,80] CPU parity failure.
```

Published artifacts under `cpu_tests/cycle4/`: original head/NF4 scripts, complete scientific results, summary and dynamic-codebook bit patterns. Private installation/native-library provenance, wheel/cache files and absolute cloud infrastructure paths are omitted; originals remain local.

All work used the existing cloud venv with two CPU threads; no model download, full training, official-suite rerun, CUDA/GPU call, system install or remote write. CPU numerical equality is not signed-zero/NaN-payload identity or validation of the actual Kaggle Torch/CUDA stack. No timing, peak-memory, full-quality or owned-kernel-share claim is made.

**Handoff:** the next authorized GPU trial is owned by the separate GPU lane. The CPU lane has provided its standalone FP32 RMSNorm oracle and is reviewing only the trial's source/CPU contracts; this report introduces no measured configuration result.
