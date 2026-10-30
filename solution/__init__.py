"""Soup challenge candidate: install patches before model construction."""


def apply():
    from .my_kernels import install_patches

    install_patches()
