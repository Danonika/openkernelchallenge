"""Single fixed coherent fusion candidate, no runtime score-dependent switches.

Original NF4 decoding, FP16 tied embedding, output head, compact frozen MLP,
full decoder packing and direct FP16 output RMSNorm are applied together.
"""
CHUNK_TOKENS = 128
SPARSE_BACKWARD = True
HEAD_PROJECTION = "torch"
HEAD_NUMERICS = "native"
FUSED_NF4 = True
FROZEN_STORAGE = True
COMPACT_MLP_ACTIVATIONS = True
FUSED_ROTARY = False
FUSED_LORA_EPILOGUE = False
