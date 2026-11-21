"""Memory VS1: native MLP producer codec with durable zero-storage versions.

Component candidate only. No tied-table mutation or head math replacement.
No measured runtime, memory, quality, share or admission is claimed.
"""
def apply():
    from .adapter import install
    result = install(mode="production", first_order_only=True)
    # Native fallback is allowed; its lack of activation must remain visible.
    import json
    print("SOUP_MEMORY_VS1_INSTALL " + json.dumps(result, sort_keys=True), flush=True)
    return result
