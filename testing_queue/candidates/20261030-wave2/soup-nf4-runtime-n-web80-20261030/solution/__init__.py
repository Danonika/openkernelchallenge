"""NF4 blockpair v1: original packed-store decoder; unchanged native gradients."""


def apply():
    try:
        from .routes import install
    except ModuleNotFoundError as exc:
        if exc.name == "triton":
            return {"patched": False, "reason": "triton_unavailable"}
        raise
    result = install()
    print("SOUP_NF4_INSTALL " + str(result), flush=True)
    return result
