# Soup Kernel Challenge — T4 Research Round: MLP Padding & Fused Scatter

**Date:** 2026-10-31 (Asia/Almaty)  
**Task:** same 64-step Soup 0.77.0 / Qwen2.5-1.5B NF4 + Q/V LoRA training on Tesla T4.  
**Scope:** public-development runs using the VM's saved compatible baseline. These are **not** paired hidden-row jury measurements.

## Current confirmed measurements

| Candidate / queue request | Whole training (s) | Peak GPU (MiB) | Eval loss | Development result |
| --- | ---: | ---: | ---: | --- |
| Saved baseline | 71.85 | 14,345 | 2.04968 | Reference only |
| `chat6-storage-f749d20-queue-20261031-v1` | 72.51 | 5,963 | 2.04963 | Time/loss/memory checks passed |
| `chatgpt-storage-full-nf4-cache-20261031-v1` | 69.34 | 8,529 | 2.04965 | Time/loss/memory checks passed; larger memory footprint |
| `chatgpt-storage-mlp-padding-20261031-v1-r40` | **65.43** | **5,667** | **2.04967** | Time/loss/memory checks passed; strongest completed development result in this round |
| `chat6-storage-rope-20261031-r9` | 73.54 | 5,981 | 2.04965 | No speed improvement |
| `chatgpt-storage-retain-fp16-20261031-v1` | 71.97 | 6,563 | 2.04965 | No speed improvement |

The MLP-padding candidate yields a **1.098× time ratio** and **60.50% lower peak** than the saved baseline. Neither result has been validated in the final paired hidden-row protocol, and the VM's three development checks do **not** certify the separate ≥5% original-kernel share requirement.

## Existing best: within-batch MLP padding compaction

The observed 65.43-second notebook inserts a Qwen2 model/MLP wrapper. It derives valid active token rows from the **existing two-dimensional attention mask** once per decoder forward. Each MLP uses only those rows for its frozen 4-bit gate/up/down operations, then reconstructs the dense output with zeros on ignored padding positions. Q/V LoRA and attention remain dense and unchanged; the batch and row order are preserved. This can eliminate expensive MLP work without changing the supervised tokens.

Source notebook: project-repository/blob/master/soup-kernel-challenge/queue_20261031_storage_mlp_padding_v1.ipynb

**Caveat:** `torch.nonzero` on CUDA can synchronize to discover the active-row tensor length. Performance depends on the valid-token ratio and extra indexing/scattering overhead, so we must test different seeds and hidden-row length patterns.

## New A/B experiment 1 — fused padding scatter

**Request ID:** `chatgpt-mlp-padding-fused-scatter-20261031-v2`  
**Notebook:** project-repository/blob/master/soup-kernel-challenge/queue_20261031_storage_mlp_scatter_v2.ipynb

The original Triton `soup_mlp_scatter_dense` kernel maps every padded-row output back to its compact MLP row using a prefix index computed once per model forward. Unlike `torch.zeros_like + index_copy_`, it writes the correct dense output in a single pass. Its custom autograd backward selects precisely the active rows from the upstream gradient. A CPU reference check covered exact forward and gradient equality at widths 7 and 1536 and up to 2,048 rows. **Physical T4 and end-to-end quality tests are pending.**

## New A/B experiment 2 — fused MLP residual

**Request ID:** `chatgpt-mlp-fused-residual-20261031-v3`  
**Notebook:** project-repository/blob/master/soup-kernel-challenge/queue_20261031_storage_mlp_residual_v3.ipynb

The original Triton `soup_mlp_residual_scatter` kernel writes the residual plus compact MLP contribution directly. The layer wrapper retains the pinned Qwen2 attention call, Q/V LoRA dropout, both layer norms, the first residual, frozen MLP projection semantics, and all position/attention arguments. Autograd returns the unchanged upstream residual gradient and gathers only valid rows for the MLP gradient.

Compared with zero-fill + sparse index-copy + dense residual addition, this can remove two full dense FP32 tensor passes. At the maximum 2,048-token, 1,536-wide, 28-layer shape:

`2 × 2048 × 1536 × 4 bytes × 28 = 704,643,072 bytes ≈ 672 MiB of logical traffic per maximum-size training step`.

This is **only an operation/traffic calculation**, not a measured reduction in peak residency or wall time. The CPU algebra reference passed exact forward and input-gradient comparisons at multiple shapes; the real T4 test is pending.

The decoder layer logic was checked against pinned Transformers v5.16.1 source: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py

## Other active queue experiments

- `chatgpt-cached128-frozen-direct-20261031-v2` — clean retry after optional readiness preflight failed before training.
- `chatgpt-cached128-metricpacket-20261031-v3` — previously validated FP32/int64 packet transfer for TRL and Rewind metrics adapted to a native cached head.
- `chatgpt-cached256-frozen-20261031-v1` — optional readiness gate failed before training; **not** evidence of a GPU training failure.

## How to select a winner

1. Any new candidate must finish the complete 64-step training command under the VM's 80-second cutoff.
2. Preserve held-out evaluation loss within +0.5% of baseline and match log loss, entropy, accuracy, counts and Rewind records at their specified tolerances.
3. Explicitly validate the **≥5% original useful Triton kernel time, steps 3 onward**; time/loss/memory development badges alone do not test the full admission criteria.
4. Prioritize at least a 5% wall-time gain versus the saved baseline. Our 65.43-second result clears it; small sub-5% fluctuations do not earn speed points under the rubric.
5. Prefer the strongest time/memory trade-off and run the untouched official paired baseline → solution → solution → baseline protocol on Kaggle T4 before submission. No official admission or win is currently claimed.

## Queue and submission context

Queue: https://example.invalid/redacted

The VM queue is a development service and accepts the notebooks through new A:H rows with `Ready=TRUE` set last. No secret token is embedded in the notebooks. For the final hackathon submission, the platform-created team repository, public Kaggle notebook and required README/SUBMISSION.md, video and slides still matter. All test and harness artifacts are kept untouched.