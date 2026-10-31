# Team-submission experimental fusion. No hidden-data/measurement branching.
# Verify ten official checks, T4, 79 Soup tests before promotion.
"""Version-controlled candidate choices, identical in every measured run.

Change a choice only as an explicit experiment, then use the official harness.
There is no runtime tuning or benchmark-specific behavior.
"""

HEAD_MODE = "streamed"
# Keep the native full-shape lm_head GEMM and reconstruct native CE backward
# in bounded row chunks. The streamed alternative trades recomputation for
# less live half-logit storage and must be compared with the official harness.
CACHED_CE_CHUNK_TOKENS = 128
CHUNK_TOKENS = 128
SPARSE_BACKWARD = True
# Keep the vendor GEMM until a T4 run validates the explicit MMA alternative.
# Triton 3.6's ordinary tl.dot lowers to scalar FMA on SM75. It is available
# only as an experiment, not as the default competition candidate.
# "t4_fused" selects the integrated shared-memory statistics/gradient path;
# "t4_mma" selects the older projection-only experiment. Both require T4
# numerical and whole-run validation before selecting a final candidate.
HEAD_PROJECTION = "torch"

# Fused nested-scale + NF4 decoding; preserves the pinned backend's GEMM dispatch.
FUSED_NF4 = True

# Optional lossless storage experiments. Enable only after target-GPU checks.
FROZEN_STORAGE = True
COMPACT_MLP_ACTIVATIONS = True

# Optional original pointwise fusions; keep off until physical T4 parity and
# full cold-run timing establish whether their launch/traffic savings help.
FUSED_ROTARY = False
FUSED_LORA_EPILOGUE = False

# Combined candidate: native dropout RNG and GEMMs with an original fused
# dropout-gradient cast/mask epilogue. Optional scale/add fusion is separate.
FUSED_LORA_DROPOUT = False
SHARED_ATTENTION_MASK = False
FUSED_RMSNORM = True
