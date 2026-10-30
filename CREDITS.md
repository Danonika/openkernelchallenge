# Provenance and contribution boundaries

## Rules and task

The candidate targets [the official Soup Kernel Challenge at commit c7c2fcd15972ce8b898d8d3ff732ee192c817ceb](https://github.com/MakazhanAlpamys/soup-kernel-challenge/tree/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb). The [pinned README](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md) and [forbidden import list](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/common.py) govern the development checks. The user's service instructions additionally set the 80-second training cutoff and notebook contract.

## Supplied example: excluded from original-kernel credit

`solution/swiglu.py` is copied unchanged from the [official starter example](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/solution/swiglu.py). Its SHA-256 is `e665910faeca6091c958ed7a191604484a44216f51d6fa069f5f48220e7472e6`.

The `example_swiglu_fwd_kernel` and `example_swiglu_bwd_kernel` names are retained. Neither those kernels, copies or derivatives of them, nor the half-storage wrapper are represented as original kernel contributions. Keeping their code unchanged makes their numerical boundary and provenance reviewable.

## Original implementation

The four JIT definitions in `solution/rmsnorm.py` and `solution/metrics.py` were written for this candidate. No third-party optimization-kernel implementation was copied. The mathematical and integration references are:

- [Transformers 5.16.1 Qwen2 implementation](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py): RMSNorm formula, MLP structure, and task dimensions.
- [Transformers 5.16.1 activation mapping](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/activations.py): the actual `SiLUActivation` type used by the guarded path.
- [PyTorch 2.11 autograd derivative definitions](https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml): primitive derivative composition for RMSNorm backward.
- [TRL 0.29.1 metric utilities](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/utils.py): the full-vocabulary entropy definition and fallback API.
- [TRL 0.29.1 SFTTrainer](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py): imported entropy binding, no-grad metric context, masks, and optional accuracy-copy investigation.
- [bitsandbytes 0.50.2 Linear4bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py): FP16 computation and output widening used by the MLP storage argument.
- [bitsandbytes 0.50.2 CUDA backend](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py) and [autograd functions](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py): projection dispatch and backward ownership.

Source agreement is not a substitute for GPU numerical comparisons or end-to-end task measurements. Runtime contribution, correctness, speed, memory, and quality remain unmeasured until the service is accessible.
