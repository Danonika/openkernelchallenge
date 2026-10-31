"""Fixed implementation profile: direct_varlen_cached_eager_storage.
Selected before a fresh run. Task settings are unchanged.
No on-device or benchmark-dependent profile selection.
"""

HEAD_MODE = 'sparse_cached'
HEAD_NUMERICS = 'guarded'
CACHED_CE_CHUNK_TOKENS = 128
CACHED_CE_BACKWARD = 'native_fused'
PACKED_METRICS = True
CHUNK_TOKENS = 256
SPARSE_BACKWARD = True
HEAD_PROJECTION = 'torch'
FUSED_NF4 = True
NF4_FORWARD_CACHE_BYTES = 1073741824
NF4_CACHE_BACKWARD = True
NF4_DIRECT_LINEAR = True
FROZEN_STORAGE = True
COMPACT_MLP_ACTIVATIONS = True
MLP_PADDING_COMPACTION = True
VARLEN_ATTENTION = True
FUSED_ROTARY = True
FUSED_LORA_EPILOGUE = True
FUSED_LORA_DROPOUT = True
SHARED_ATTENTION_MASK = False
FUSED_RMSNORM = True
