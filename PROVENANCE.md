# Source and kernel provenance

## Recovered user artifacts

The implementation combines code and conclusions produced during this user's earlier kernel-research sessions. Reuse of these earlier original kernels is stated explicitly; this package does not claim that every kernel was first authored during the merge.

| Input | Authoritative artifact | Version / SHA-256 |
|---|---|---|
| Build candidate | [Soup_Kernel_Candidate_Overlay.zip](https://chatgpt.com/library?file_id=libfile_66ef098816d48191a60a87b7fe1380a5&account_id=personal) | `127b7929e47dea7c89d39cfb0944119c88a173a75f6d5c74751f7f6831940f3c` |
| LoRA reproduction bundle | [KERNEL_ROUND_7_REPRO.zip](https://chatgpt.com/library?file_id=libfile_1f38d5239b1c819189777bc9fc2cb65f&account_id=personal) | `5cb7a584dd9a41b6c8e0be1cab3cc4a23a5b6f6ab7aee96fb3da0e30ee15b934` |
| Qwen-inspired synthesis | [Qwen_Inspired_Soup_Kernel_Optimization.md](https://chatgpt.com/library?file_id=libfile_cd1cc7b4682c8191b9c79cc7cb430fb6&account_id=personal) | Version 6 |
| Qwen kernel review | [Qwen_Kernel_Optimization_Review.md](https://chatgpt.com/library?file_id=libfile_2c52928edc088191a6b360df80acb599&account_id=personal) | Version 6 |
| Nanochat transfer | [Nanochat_Ngram_Soup_Transfer.md](https://chatgpt.com/library?file_id=libfile_b51564b02cf08191ad7c727e3e3ac94b&account_id=personal) | Version 5 |
| LoRA handoff | [KERNEL_HANDOFF.md](https://chatgpt.com/library?file_id=libfile_4dbc8350608c8191872cdd9f919e0811&account_id=personal) | Version 6 |
| Build handoff | [Kernel_Handoff.md](https://chatgpt.com/library?file_id=libfile_4d2f66ae61848191b1790c797c28fa52&account_id=personal) | Version 3 |

## Changes made in this merge

- Added `cached_head.py`: native full-shape head forward/backward with private FP16 logits, original compact-statistics merge, bounded native CE VJP, and exception-aware forward summaries.
- Extended `integration.py`: scoped loss-property interception, original Qwen forward, compact output before Accelerate conversion, and fixed cached/streamed selection.
- Ported `lora_dropout.py` from the user's Round 7 native-mask bridge and original Round 5 epilogue; converted it into an idempotent before-construction CUDA production patch with narrow fallbacks.
- Ported `shared_attention_mask.py` as an optional original per-forward conversion/reuse path.
- Corrected `rmsnorm.py` to follow the staged derivative from the Nanochat audit; reduction behavior remains tolerance-based.
- Connected optional LoRA scale/add fusion explicitly to the native-mask context so both can execute together.
- Enabled the reviewed compact MLP storage path in the default source profile.
- Added full Soup factory composition tests, real physical-T4 NF4 integration tests, fixed-profile benchmark orchestration and independent result verification.

Existing original NF4, streamed-head, rotary, scale/add, activation-storage and optional frozen-storage code was recovered from the build bundle. Their method-specific references remain in the source headers. The inline-MMA experiments remain disabled.

## Credited organizer source

The official starter is pinned at `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`:

https://github.com/MakazhanAlpamys/soup-kernel-challenge/tree/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb

The original `harness/`, `task/`, `tests/`, `solution/swiglu.py`, `solution/test_kernels.py`, `LICENSE` and `NOTICE` are retained. The official README is preserved separately under `docs/`. Hashes and the official rule scanner confirm the protected inputs are unchanged.

The two `example_swiglu_*` kernels and every reuse/specialization of them are **excluded** from original-kernel credit. Compact MLP storage reuses them and introduces no newly countable kernel.

## Dropout epilogue originality review

The official AST scanner flags `soup_dropout_half_gradient_epilogue` at approximately 0.91 similarity to the starter, above the 0.8 manual-review threshold and below the 0.95 automatic-exclusion threshold. This is disclosed rather than hidden by code reshaping.

Its derivation is the user's earlier dropout-gradient work: the incoming gradient is FP16 from the native adapter GEMM, the retained bool tensor is PyTorch's genuine dropout mask, and the kernel writes FP32 input gradients using native multiplication and scale order. It performs no SiLU, does not read gate/up activations, and replaces cast-plus-dropout backward rather than an MLP activation. Its output is the real gradient returned to the model. The numerical tests include dropped Inf/NaN behavior, signed zeros, native mask provenance, loss scaling and actual optimizer updates.

RMSNorm, NF4 and the head statistics have independent functions and purposes. Actual eligibility requires observed useful execution and timing; the jury decides originality. Definitions, offline compilation and synthetic launch counts do not prove the five-percent share.

## External semantic references

Only primary-source equations, format semantics, APIs and the pinned reference behavior are used as external references. No ready-made third-party optimization library is imported by `solution/`.

- Transformers Qwen2: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py
- PEFT LoRA: https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py
- bitsandbytes NF4 and dispatch: https://github.com/bitsandbytes-foundation/bitsandbytes/tree/0.50.2/bitsandbytes
- PyTorch native derivatives: https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml
- PyTorch AMP: https://docs.pytorch.org/docs/2.11/notes/amp_examples.html

The memory values in the handoff are algebraic tensor-size calculations. The only 77- and 33-second values used as reference targets come from the user. Synthetic verifier fixtures are labeled tests and have no evidentiary value for the candidate's speed.
# Service notebook and client provenance

The offline service template and API client are exact, credited utility sources from `BAITC-Hacks/hack-57faaa42-duo@3a4953f76288969bb91956f114eda5b7e541f104`. They are not optimization contributions and provide no original-kernel credit. The original three template cells are preserved in the generated service notebook.

- `templates/notebook_service_template.ipynb`: source `testing_queue/examples/notebook-training-template.ipynb`, SHA-256 `bea1521d4c9d68b27591a483b19463b86ef49a0d7ea1fd8ec5e73b61001b81f8`.
- `tools/notebook_client.py`: source `testing_queue/notebook_client.py`, SHA-256 `7d7b5222b040c58af5d2f7a3d4846a7d04945809a3f6fd52d5ce73ca1312f4ff`.
- `tools/build_service_notebook.py`: new packaging code. Its embedded payload contains ordinary Python source and no weights, credentials or compiler caches.

Source provenance is not evidence of the currently deployed service release. The live runtime endpoint remains unread because the current workspace cannot verify HTTPS to the GPU IP.

