# Review caveats — prepared artifacts, not GPU-ready approval

This directory preserves the six original prepared candidate packets and their exact delivery hashes. No target T4 validation or contest admission is asserted.

## CE and LoRA tensor-metadata guard gap

A subsequent independent CPU/interpreter review found that contiguous lazy-negative tensor views can pass the CE and LoRA fast-path guards while raw-pointer Triton loads read the underlying, un-negated storage. The LoRA witness uses the native expression `base + delta * 2.0`; flagged base or delta inputs can therefore disagree with native execution. Conjugate-view metadata also requires a conservative guard.

This is a concrete unsupported-input guard defect, not an observed failure of the fixed Qwen/Soup training task: ordinary task-produced clone/GEMM tensors have not been observed to carry these flags. That distinction does not establish readiness. The frozen sources remain unchanged for reproducibility. Minimal native-fallback guards are prepared for a separately reviewed next revision; they are not applied to these historical ZIPs or counted as validation here.

Before any target run, review this limitation and use a verified guard-fix revision or establish a sufficiently narrow supported-input contract. A new source revision requires matching notebook/ZIP/source hashes and fresh tests. No speed, peak-memory, quality, useful-kernel-share, score, or admission result changes because of this publication.

The `head` packet is also explicitly a native memory-control experiment with no original Triton kernel and cannot independently satisfy the contest's original-kernel participation requirement.
