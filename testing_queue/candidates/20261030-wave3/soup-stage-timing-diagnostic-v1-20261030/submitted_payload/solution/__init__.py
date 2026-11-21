"""Native fixed-task stage timing only: no arithmetic kernels or admission claim."""
def apply():
    from .stage_probe import install
    return install()
