"""Only original RoPE application kernels; official starter kernels are not claimed."""
def apply():
    from .rotary import patch_qwen2_rotary
    if not patch_qwen2_rotary():
        raise RuntimeError("Pinned Qwen2 rotary source guard refused installation")


def queue_smoke():
    from .cpu_contract import run
    return run()
