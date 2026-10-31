# Technical foundations, provenance and research trace

This file accompanies the Soup T4 experimental `solution/` package. It documents **related published methods** and the team's own research trace; it does not assert that cited software is imported or that every listed idea is enabled.

## Scope of the references

The official Soup starter and framework components are documented in the repository's `NOTICE` and `PROVENANCE.md`. The references below explain published mathematical and systems techniques relevant to the implementation. **Implementing the same public idea is not, by itself, copying another codebase.**

The team's kernels must be evaluated on their actual source, development history and behavior. The research graph contains several separate implementation experiments and reference controls; results from those entries should not be conflated with the final submitted branch. These citations do not determine original-kernel eligibility.

## Published foundations by mechanism

| Mechanism / area | Research or primary technical reference | Relationship |
|---|---|---|
| Soup loss and metrics protocol | Soup 0.77.0, `soup_cli/utils/stream_loss_chunk.py`, `ChunkedStep` / `_soup_chunked_step` | Framework behavior to preserve; a team patch does not authorize changing fixed Soup config |
| Fused linear–CE, RMSNorm and RoPE | Hsu et al. (2024), [Liger Kernel, arXiv:2410.10989](https://arxiv.org/abs/2410.10989) | Algorithmic comparison; ready-made Liger kernels are **not** eligible as team-original kernels |
| Vocabulary-streamed cross-entropy | Wijmans et al. (2024/ICLR 2025), [Cut Cross-Entropy, arXiv:2411.09009](https://arxiv.org/abs/2411.09009) | Loss without materializing the full logit matrix |
| Online log-sum-exp | Milakov and Gimelshein (2018), [Online normalizer calculation for softmax, arXiv:1805.02867](https://arxiv.org/abs/1805.02867) | Streaming maximum and normalized sum |
| Numerically stable attention | Dao et al. (2022), [FlashAttention, arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | Tiled online-softmax accumulation; related to packed/variable-length attention |
| Memory-efficient attention | Rabe and Staats (2021), [Self-attention Does Not Need O(n²) Memory, arXiv:2112.05682](https://arxiv.org/abs/2112.05682); [PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | Framework-selected attention kernels are not team-created kernels |
| NF4 / double quantization | Dettmers et al. (2023), [QLoRA, arXiv:2305.14314](https://arxiv.org/abs/2305.14314); [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | NF4 codebook and block/nested-scale semantics |
| Trainable low-rank adapters | Hu et al. (2021), [LoRA, arXiv:2106.09685](https://arxiv.org/abs/2106.09685); [PEFT](https://github.com/huggingface/peft) | Rank, scaling, dropout and adapter-gradient parity |
| Dropout RNG replay and rematerialization | Chen et al. (2016), [Training Deep Nets with Sublinear Memory Cost, arXiv:1604.06174](https://arxiv.org/abs/1604.06174); [PyTorch checkpoint](https://pytorch.org/docs/stable/checkpoint.html) | Saved RNG-state/recompute technique; preserve actual seeded dropout behavior |
| Unpadding and packed sequences | [Hugging Face packing with FA2 (2024)](https://huggingface.co/blog/packing-with-FA2); [Unsloth](https://github.com/unslothai/unsloth); FlashAttention varlen and Transformers `DataCollatorWithFlattening` | Related packing ideas; protect causal row boundaries, position IDs and RNG |
| RMSNorm | Zhang and Sennrich (2019), [arXiv:1910.07467](https://arxiv.org/abs/1910.07467); [Triton layer-norm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) | Row-local reductions and exact required cast boundaries |
| Rotary embeddings | Su et al. (2021), [RoFormer, arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | RoPE mathematics and gradient behavior |
| Startup / synchronization | [PyTorch tuning guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html); [CPython gc.freeze](https://docs.python.org/3/library/gc.html#gc.freeze) | General profiling and host-side optimizations; local measurement evidence required |
| Saved-activation CPU offload | [PyTorch save_on_cpu](https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu), [saved tensor hooks](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html); [torchtune](https://github.com/pytorch/torchtune), `torchtune/training/_activation_offloading.py`; [Unsloth long context](https://unsloth.ai/blog/long-context) | Concept and APIs; does not imply TRL activation_offloading is enabled |
| CPU-resident weights | Ren et al. (2021), [ZeRO-Offload, arXiv:2101.06840](https://arxiv.org/abs/2101.06840); [Accelerate](https://github.com/huggingface/accelerate) | Related offload/prefetch designs |
| Pinned host memory and asynchronous copies | [NVIDIA CUDA C++ Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | `cudaHostRegister`, streams and transfer overlap |
| Memory allocator behavior | [PyTorch CUDA memory management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management) | Interpretation of allocated/reserved memory and caching pools |

**Important:** A citation to a library is not an assertion that the library's prebuilt kernels are installed, imported, or counted as the team's original work. Verify imports, patch behavior and the competition's originality rules against the *final submitted source*.

## Team research graph and evidence

The attached **Open Kernel Research Graph.xlsx** records distinct experiments, hypotheses, measured results and warnings; it is not a provenance certificate. Relevant trace IDs include:

- `SOUP-NEW-002`: streaming sufficient statistics for loss, entropy and accuracy.
- `SOUP-NEW-001`, `SOUP-TRAIN-001`: bounded/recomputed output-head and supervision-path research.
- `SOUP-MEM-001`, `SOUP-MEM-003`, `SOUP-MEM-004`: activation ownership, selective replay and offload tradeoffs.
- `SOUP-NEW-003`: nested NF4 dequantization and codebook handling.
- `SOUP-NEW-006`: variable-length packed attention and strict original dropout RNG geometry.
- `SOUP-NANO-002`: compile-key specialization and cold-compile overhead.
- `SOUP-FUSION-HERO12-046`: recorded 38.123 s / 2261 MiB / evaluation loss 2.04979 for a separately tested notebook, **not** a measurement of this GitHub branch. The graph records its specific parent/control experiment; that fact is not an authorship determination for the code in this branch.
- `SOUP-FUSION-HERO10-042` and `SOUP-FUSION-OWN-048`: separate attempts at original independent implementation. The graph does **not** record an equivalent successful complete result for the latter.

For an originality claim, preserve the specific submitted commit, code-generation history, relevant diffs, licenses, and an official full-task GPU report. Experimental evidence on another source snapshot does not automatically transfer to this branch.

## Submission note

The entry point is `solution.apply()`; the task and checks must remain unmodified. The jury may review the submitted kernel source, semantic parity, active original Triton kernel share and measured hardware performance. When any code is actually adapted from a third party, the corresponding attribution and license obligations still apply.
