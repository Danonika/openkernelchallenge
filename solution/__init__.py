"""One integrated original Soup 0.77.0 T4 QLoRA solution entry point.

Competitor research informs the architecture, but every non-starter module in
this package is our own code. The original supplied SwiGLU example is retained
and excluded from our original Triton kernel participation.

apply() takes zero arguments, patches only fixed task capabilities, and never
installs a secondary benchmark or hidden-data path.
"""


def apply():
    from .nf4 import patch_bitsandbytes_nf4
    from .frozen_storage import patch_frozen_storage
    from .swiglu import patch_qwen2_mlp
    from .activation_storage import patch_compact_mlp_activations
    from .frozen_mlp import patch_frozen_mlp
    from .rmsnorm import patch_qwen2_rmsnorm
    from .integration import patch_streamed_training
    from .mlp_padding import patch_mlp_padding
    from .packed_decoder import install_decoder_packing

    # Correct order matters: source capability guards, then original starter,
    # then the frozen-MVP optimization, then streamed train and packed decoder.
    if not patch_bitsandbytes_nf4():
        raise RuntimeError('Required original NF4 patch unavailable')
    if not patch_frozen_storage():
        raise RuntimeError('Lossless tied FP16 storage requires pinned PEFT')
    patch_qwen2_mlp()
    if not patch_compact_mlp_activations():
        raise RuntimeError('Compact frozen MLP requires pinned bitsandbytes')
    if not patch_frozen_mlp():
        raise RuntimeError('Expected independent frozen MLP compact path')
    if not patch_mlp_padding():
        raise RuntimeError('Expected original fallback padding scatter unavailable')
    if not patch_qwen2_rmsnorm():
        raise RuntimeError('Original RMSNorm unavailable')
    if not patch_streamed_training():
        raise RuntimeError('Bounded head and Soup metrics patch unavailable')
    if not install_decoder_packing():
        raise RuntimeError('Original packed decoder unavailable')
