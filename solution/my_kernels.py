"""Candidate patch integration. This module never starts training.

Installers are idempotent and retain native fallbacks for unsupported inputs.
Each one runs on every call so a missing/mock dependency during a regression
test cannot permanently prevent a later real installation in that process.
"""


def install_patches():
    from . import metrics, mlp_storage, rmsnorm

    rmsnorm.install()
    mlp_storage.install()
    # Keep the optional trainer-source rewrite out of this initial candidate.
    metrics.install(accuracy_no_copy=False)
