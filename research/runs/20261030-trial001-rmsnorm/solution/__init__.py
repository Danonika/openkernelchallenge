"""Trial001: original frozen-FP32 RMSNorm; starter SwiGLU is not activated.

The shipped example files are retained by the runner but never receive ownership
credit. This first comparison isolates the original norm implementation.
"""
def apply():
    # A CPU-only regression environment need not have the GPU compiler installed.
    try:
        from .rmsnorm import patch_qwen2_norm
    except ModuleNotFoundError as exc:
        if exc.name == 'triton':
            return
        raise
    patch_qwen2_norm()
