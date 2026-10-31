# Soup cycle 4 — an independent NF4 double-quant oracle, with backend limits

## Decision first

Refine **SOUP-NEW-003** (fused nested-scale/NF4 decode) and **SOUP-NEW-007** (frozen reconstructed-scale cache); **no new idea, candidate implementation, speed claim, or performance point**. The missing evidence is a correct byte-to-weight contract, not another fusion proposal.

A useful oracle needs the actual 16-value NF4 lookup, **the supplied 256-value nested codebook**, separate FP32 multiply/offset rounding, flat block indexing, tail masking, original shape, output dtype, and API-specific orientation. “All 256 nested codes” must mean all 256 **indices into the supplied table**, not a substituted linear ramp. A decoder test is not a test of the quantization algorithm or model quality.

Initial metadata checks found no installed bitsandbytes. Parent then authorized the CPU lane alone to install official PyPI bitsandbytes 0.50.2 with `--no-deps` in an isolated cloud environment, without changing Torch/CUDA/system packages. Installation succeeded with bitsandbytes as the only package delta; Torch remains 2.15.0+cpu. The loaded CPU native library and wheel are hashed, and all eight inspected installed Python files match the pinned source Git blobs. Public-API exhaustive and tiny quantizer-returned-state comparisons are now available. Three NF4 fixture groups pass; one native CPU parity group fails on a specific misaligned shape, retained as a negative result below. No source decoder function is copied or executed from fetched source; only public format constants and a separately written scalar/vector oracle are used.

The parent/jury-validated **77.92 s / 14,345 MiB** remains the existing baseline, not a result of these fixtures or either proposed candidate.

## 1. Pinned contracts

All bitsandbytes citations below refer to **0.50.2, commit `08a9956b939857ae2c4fbd56753960cc370fa7cf`**. [`nf4_cycle4/source_manifest.json`](nf4_cycle4/source_manifest.json) records the fetched Git blob identities. Installed Python source checks were performed locally; private environment and native-library provenance are intentionally omitted here. Public package version and pinned source references are retained. No native CUDA behavior is inferred.

### Actual QuantState fields

[QuantState and deserialization](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/functional.py#L418-L543), [nested construction](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/functional.py#L927-L960):

| Field | Meaning for task-like nested NF4 |
|---|---|
| `shape` | Original dense weight shape `[N,K]`, not inferred from packed storage length. |
| `quant_type` | `nf4`; not INT4, FP4 or NVIDIA E2M1. |
| `blocksize` | Normally 64 scalar weight elements per first-level block. Public 4-bit API permits 32/64/128/256/512/1024/2048/4096; a narrow task kernel may guard 64. |
| `dtype` | Original floating weight dtype; functional dequantization returns this dtype before any caller cast. |
| `absmax` | In nested mode, **uint8 code indices**, one per first-level block—not FP32 scales. |
| `code` | Outer 16-value quantization metadata. Current dequantization selects the canonical table from `quant_type`; it does not use an arbitrary replacement outer `code`. |
| `offset` | Scalar center originally computed as the mean of first-level FP32 absmax values. Decode adds the saved offset; do not recompute it or recenter. |
| `state2.code` | Supplied 256-entry FP32 codebook for centered scales; default signed dynamic map, not NF4 and not uniform quantization. |
| `state2.absmax` | FP32 second-level block magnitudes. |
| `state2.blocksize` | Nested construction uses 256 first-level scale entries per block. Generic functional blockwise decode honors its own positive blocksize; the MatMul4Bit nested forward path specifically requires 256. |
| `state2.dtype` | FP32 for states produced from the first-level FP32 absmax path. A narrow decoder must guard this rather than silently reinterpret arbitrary loaded metadata. |

The proposed narrow task route additionally requires same-device uint8 packed data/outer indices, FP32 `state2.code`/`state2.absmax`/scalar offset, nonnested state2, and supported 2-D shape. Other valid public metadata remains on the original path.

For n=N*K, first-level scale count is ceil(n/64), packed byte count ceil(n/2), second-level magnitude count ceil(ceil(n/64)/256). These are geometry requirements for valid fixtures, not a promise that every public entry point validates malformed lengths before native access.

The [default dynamic-map constructor](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/functional.py#L296-L348) builds a nonlinear signed 8-bit table. Its exact FP32 output should be **read as metadata from the official public API/state**, not regenerated with a mathematically similar expression that may round differently. Synthetic tables are useful adversarial fixtures but must be labeled synthetic.

### Independent scalar decoder specification

Let i be a scalar weight's **flat row-major** index, j=floor(i/blocksize), and h=floor(j/state2.blocksize). For task-like nested FP32 state:

```
byte = packed_bytes[floor(i/2)]
index4 = high_nibble(byte) if i is even else low_nibble(byte)
centered_j = round_fp32(state2.code[uint8_absmax[j]] * state2.absmax[h])
scale_j = round_fp32(centered_j + offset)
weight_i = cast_output_dtype(round_fp32(NF4[index4] * scale_j))
```

Only i<n is decoded. An odd final packed byte's unused low nibble must not become an extra output or affect any valid output. First-level block indexing is flat, **not restarted at each row**. A shape such as `[3,65]` is a decisive cross-row/tail test.

The [canonical 16 NF4 values](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/backends/utils.py#L15-L34) are format data shared with the independent test, not copied decoder implementation. The [default backend](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/backends/default/ops.py#L199-L300) and [CUDA source](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/csrc/kernels.cu#L467-L527) establish high-nibble-even ordering and separate FP32 LUT multiplication before output conversion. The outer functional wrapper performs nested scale decode and then offset addition separately. [Functional decode](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/functional.py#L992-L1077).

Do not clamp a reconstructed scale, substitute a lookup tree or linear mapping, move the offset after NF4 multiplication, cast scales/codebook to half, or contract the centered-scale multiply/add without proving that the required result is preserved. Prior synthetic FMA counterexamples establish only that a difference can exist; they do not establish a failure on actual default-table or model states.

Non-uint8 packed storage in the public backend is a **byte reinterpretation**, not numerical conversion. A narrow uint8-only implementation should fall back for other storage types until views, byte length, and alignment are supported. Empty/malformed/unsupported metadata also belongs on a safe original path, not an unchecked custom load.

## 2. Forward, backward, and BC layout are different contracts

[MatMul4Bit and entry routing](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/autograd/_functions.py#L294-L498) normalize packed B to `[(N*K+1)//2,1]`. The original dense `quant_state.shape=[N,K]` determines orientation; square weights do not permit guessing orientation from dimensions.

- **CUDA forward fallback** reconstructs nested scales, decodes weight directly to **A.dtype**, and calls ordinary `F.linear(A,W,bias)`. It does not select the decode output dtype from `QuantState.dtype`. [Fallback lines 904–916](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/backends/cuda/ops.py#L904-L916).
- **Backward** obtains W through `functional.dequantize_4bit`, which first returns **QuantState.dtype**, then casts W to `grad_output.dtype`, then computes `grad_output @ W[N,K]`. The frozen packed weight receives no weight gradient here. Do not erase the intermediate state-dtype rounding merely because the later matmul is FP32.
- **Linear4bit wrapper** casts input to compute dtype, handles bias conversion, and returns the original input dtype. Leave these casts and LoRA/base-gradient composition unchanged. [Lines 609–637](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/nn/modules.py#L609-L637).
- **Raw functional BC shim:** if packed `A.shape[0]==1`, `dequantize_4bit` returns `out.t()`, a transposed view. Canonical `[L,1]` generally returns contiguous `[N,K]`; raw `[1,L]` returns `[K,N]`. A one-byte canonical `[1,1]` is an actual ambiguous corner of this literal condition; a compatibility test should record it, not silently repair the reference. This is not a relevant full-size Qwen projection shape.
- Non-square legacy `[K,N]` quantized input can enter a separate deprecated compatibility path. Guard/fallback rather than implicitly changing it.

The candidate's functional backward interception does **not** automatically intercept forward's direct backend calls. Moreover bnb already has a fused small/medium-M forward and architecture-specific dispatch; M>1536 forces the ordinary fallback, but other shapes require the actual route. Do not count every projection call as an available unfused decode. [CUDA dispatch](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/backends/cuda/ops.py#L934-L981).

## 3. Can official CPU fallback be an oracle without copying it?

**Yes, as an independently invoked public-API cross-check if the pinned compatible package is installed; no, as a blanket CUDA equivalence certificate.** Construct tiny official `QuantState` objects, supply exact codebook tensors, invoke `F.dequantize_4bit`, and compare against separately written scalar/vector decoders. A tiny public `F.quantize_4bit(..., compress_statistics=True)` result additionally verifies state-construction assumptions without a model download. Calling the allowed baseline dependency is not copying its implementation and earns no owned-kernel share.

But “CPU fallback” is not a single implementation:

1. [CPU registrations](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/backends/cpu/ops.py#L94-L240) invoke native CPU functions when the library is available. Odd last dimensions and AVX512 blocksize>=2048 select a generic 4-bit fallback; other cases can use native code.
2. **Confirmed native CPU limitation:** the [x86 AVX512 4-bit branch](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/csrc/cpu_ops.cpp#L355-L395) admits last dimensions divisible by 16, then indexes scales using a row-local floor division without an explicit last-dimension/blocksize alignment guard. The actual installed native CPU decode for `[3,80]`, blocksize 64, packed bytes all `0xFF` and direct scales `[1,2,3,4] disagrees with the independent flat oracle in **48/240 elements, maximum absolute error 1**. Controls `[3,65]`, `[3,66]`, and aligned `[3,64]` match. Do not treat a native CPU result as an unquestionable oracle if this shape disagrees. A public-API `[1,240]` metadata control with the same packed bytes/scales matches exactly, isolating row indexing without calling or copying private implementations; its value result is reshaped only for comparison. Actual Qwen projection dimensions are blocksize 64-aligned; a synthetic misalignment limitation would not establish a task or CUDA failure.
3. The [generic helpers are torch.compile-wrapped](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/backends/default/ops.py#L12-L35). An intentionally eager oracle process should state that policy; it is not the compiled helper or Kaggle execution path. Installed Torch supports `TORCHDYNAMO_DISABLE=1` in its Dynamo entry path; the CPU lane may use it to avoid unbounded compilation during semantic tests.
4. CPU eval `Linear4bit` can repack weights for AVX512BF16 inference. Direct functional dequantization of ordinary canonical bytes avoids this separate format; do not accidentally use repacked inference weights as CUDA layout evidence.
5. Record wheel version/hash, installed Python-source identities, native library identity and actual CPU dispatch. Version text alone is not a source/build proof. The CPU package's codebook/quantizer and the GPU quantizer may differ in implementation; exact GPU quantization parity is outside this task.

No monkeypatch of official decoder arithmetic, no extracting executable source functions, no source-kernel transplant, and no enabling optional ready-made hub kernels is needed. If installation/import/native support fails, preserve the failure and retain the independent metadata-contract tests without claiming official runtime parity.

## 4. Bounded CPU evidence

The independent scalar path uses explicit FP32 rounding at each specified operation; a separately structured Torch vector path is the second reference. The bounded tests cover:

- All 16 canonical NF4 indices crossed with all 256 nested indices; separate table identity from decoder coverage.
- First-level boundaries 63/64 and second-level scale-index boundaries 255/256 (weight indices 16383/16384 at blocksize 64), partial blocks and odd final nibbles.
- Distinct centered terms, offsets, and second-level magnitudes; padding-nibble mutation must not change valid outputs.
- Canonical and BC layouts, flattened cross-row indexing, non-square shape, and explicit one-byte corner behavior.
- FP16/FP32 outputs; forward direct output cast versus backward state-dtype→grad-dtype conversion; no claim of CPU/GPU matmul accumulation equality.
- Outer code ignored versus nested code used, and guarded unsupported metadata. Synthetic nonlinear tables must remain distinguishable from the actual public default dynamic table.

**Results: 3 passing NF4 groups and 1 native CPU parity-failure group**, recorded in [`cpu_tests/cycle4/nf4_results.json`](cpu_tests/cycle4/nf4_results.json), original [`nf4_oracle.py`](cpu_tests/cycle4/nf4_oracle.py), and [`nf4_run.log`](cpu_tests/cycle4/nf4_run.log). The failure is not a candidate implementation failure or a measured performance result; it is a confirmed limitation of using this CPU backend as universal oracle.

- Exhaustive coverage crosses all 16 NF4 indices with all 256 nested indices in four primary cases: actual default dynamic versus explicitly synthetic cubic table, each FP16/FP32. Largest shape `[3,5483]` has 16,449 values and crosses both scale levels plus an odd tail. Independent scalar/vector decoders and the installed public CPU API agree exactly in these cases. These odd-last-dimension cases select generic 4-bit CPU fallback. A separate **aligned `[128,128]`, FP16, actual-default-table control** covers all 16×256 pairs over 16,384 values and also matches independent scalar/vector decoders and public CPU output exactly; its geometry is eligible for native 4-bit decode. No instruction-level profiling or CUDA inference follows.
- Actual public default table bits are retained in [`actual_dynamic_codebook_bits.json`](cpu_tests/cycle4/actual_dynamic_codebook_bits.json); SHA256 of its FP32 tensor bytes is `e732639a65f497b4ad684bb166a4467708255edd5207757de8b8f0c7e1fda89c`. The independent oracle consumes this table as input metadata, rather than reproducing its constructor. Synthetic cubic data remain separate.
- Six **actual CPU quantizer-returned nested states** pass independent decode comparison: shapes `[2,64]`, `[3,65]`, `[3,66]`, each FP16/FP32. These are genuine tiny `QuantState` objects, not model weights and not evidence of CUDA quantizer equivalence. Their nested blocksize/dtype are 256/FP32.
- Canonical packed `[8,1]` produces `[3,5]`, stride `[5,1]`; raw packed `[1,8]` produces `[5,3]`, stride `[1,5]`. Byte-reinterpreted FP32 packed storage matches. The one-byte `[1,1]` corner with original shape `[1,2]` returns `[2,1]`, confirming the literal BC shim. Mutating the odd spare nibble does not change valid values; changing the unused outer code metadata does not replace the NF4 selector.
- Erasing backward's intermediate state-FP16 rounding changes **10/10** tested gradient values (max absolute difference `7.721781730651855e-05`). Incorrectly forcing the forward requested-FP32 path through state-FP16 changes **6/6** tested outputs (max absolute difference `9.548664093017578e-05`). These are toy dtype-contract witnesses, not actual bnb/CUDA GEMM parity tests.
- Generic inner blocksize 64 decode agrees on its fixture; the separate MatMul4Bit forward requirement for inner 256 is source-derived and **not exercised through the optional CPU GEMM path**. That optional package is absent and was not installed.
- Native `[3,80]` parity failure and exact controls are detailed in section 3. It remains `FAIL` in machine-readable results; no test adjustment hides it. CPU operator dispatch is recorded, but no instruction-level profiling establishes the exact native microkernel branch.

Untested: real model QuantState distribution, CUDA decoder/quantizer/GEMM behavior, actual GPU FMA, complete nonfinite/signed-zero semantics, cache invalidation/streams/allocator peak, training gates and performance. The prior synthetic FMA witness is still not a proof about actual model states. Prior cycles and negative results remain unchanged.

## 5. Consequences for the two existing candidates

**SOUP-NEW-003:** the independent decoder becomes a conformance fixture for a later original Triton implementation. Require exact source-defined metadata, integer index/tail safety, separate scale rounding, and correct caller-specific dtype/orientation. Preserve ordinary matmul and baseline bias/autograd. No priority promotion from CPU decoder equality alone; actual T4 compiled code, route coverage, weight outputs, gradients and full challenge gates remain open.

**SOUP-NEW-007:** the cache stores only `scale_j` after the exact second FP32 rounding above. Cached and on-the-fly decoders must use identical supplied tables/offsets and output casts. Do not mutate QuantState to hold the cache, recompute offset from quantized values, or cache dense base weights. Build after final weight/state load/device placement; invalidate or fall back on any weight/state/table/offset/blocksize/shape/dtype/device/lifetime change. Pointer or ordinary version checks alone should not be advertised as protection against every possible untracked mutation. The fixed frozen-weight lifetime must be audited before relying on it.

[QuantState device moves](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/functional.py#L580-L587), serialization and Params4bit replacement can change relevant metadata. Numeric decoder fixtures do not validate cache invalidation, allocator lifetime, streams, or peak memory. Existing **78.094 MiB** all-projection scale-table figure remains derived geometry, not measured allocation or savings. Construction and cold initialization count; compare against on-the-fly decoding because a scale-table read can cost more than the arithmetic it replaces.

## Next decisive evidence

The actual CPU public-API check is complete with the explicit native alignment limitation; stop this research cycle after final artifact integrity/review. NF4 implementation is not selected for trial001; a separate FP32 norm trial proceeds under the parent’s authorization. This lane adds no candidate kernel. The next authorized T4 experiment should compare original on-the-fly and cached-scale decode against the **actual selected baseline path**, including full weight-byte/dtype/layout checks, representative backward gradients, allocation lifetime, cold end-to-end time, and official numerical/quality/ownership gates. No candidate coordinate is justified by the CPU oracle.
