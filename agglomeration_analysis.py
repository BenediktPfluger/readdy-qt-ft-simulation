"""Deprecated compatibility shim. Use ``qtft.analysis`` instead.

This module re-exports everything from :mod:`qtft.analysis` so existing imports
(``import agglomeration_analysis``) keep working during the package migration.
It will be removed once notebooks and scripts are updated.
"""
from qtft.analysis import *  # noqa: F401,F403
from qtft import analysis as _analysis

# Re-export private helpers too (``*`` skips underscore names) for legacy callers.
globals().update({k: getattr(_analysis, k) for k in dir(_analysis) if not k.startswith("__")})
