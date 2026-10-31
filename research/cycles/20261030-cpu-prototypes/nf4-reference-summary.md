# Original NF4 decode + matmul CPU reference

Existing IDs: **L1-TILELANG-03 / gemm-09**. Six bounded standard-library CPU test groups passed. Eight finite matrix cases matched **all 1,726 FP16 output bit patterns**, with exact FP16 decoded-weight comparisons before dot.

## What equality establishes

The independent materialized and tiled paths reconstruct nested FP32 scale, apply the NF4 code, and form each FP16 weight before multiplication. The tiled path may reuse the reconstructed scale within its original global 64-weight block; it does not move scale after dot or reset global quantization indices at row boundaries.

Both dot paths deliberately use the **same increasing-k scalar FP32 multiply/add order**, then one final FP16 output cast. Their equality therefore does **not** establish cuBLAS, tensor-core, CUDA decoder or GPU reduction parity. The second-level table and input arrays are synthetic; actual default-table/QuantState equivalence was not established.

Bounded cases cover aligned gate/up/down analogues, aspect-ratio analogues, odd K/tails and secondary-scale boundary crossings. Shape analogues preserve an intended geometric property, not the original model's quantization-block map or native dispatch. Actual large model shapes were processed only by integer geometry calculations, with no corresponding full-sized numerical arrays or timing.

## Negative evidence retained

Six constructed finite counterexamples reject these shortcuts under the declared staged contract:

1. Move scale after dot: the retained witness differs by one FP16 ULP.
2. Round the FP32 scale to FP16 before applying the NF4 code.
3. Contract nested scale multiplication and offset across the specified FP32 rounding.
4. Reuse one scale across a whole K128 row.
5. Reset secondary-superblock indexing at an output-row boundary.
6. Pad quantization blocks independently at an odd-K65 row boundary.

These are contract counterexamples, **not automatic contest illegality, official tolerance failures or observed GPU failures**.

The aligned analogue pair has equal useful scalar work. Counts of decoded weights and scale reconstructions describe this transparent CPU algorithm, not measured native/compiler instruction counts. More CTAs or shorter K are not evidence of a speed winner; logical temporary payload is not allocator peak or GPU resource usage.

## Limits and public contract

No CUDA execution, cuBLAS parity, arbitrary nonfinite/overflow or stride coverage, native dispatch validation, autograd/optimizer proof, performance benchmark, useful-share claim or full-task admission follows. No new GPU candidate or run is selected by this summary.

Public format/contract references: [canonical NF4 constants](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/backends/utils.py#L15-L34), [pinned forward fallback](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/backends/cuda/ops.py#L904-L916).
