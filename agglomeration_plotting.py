"""Deprecated compatibility shim. Use ``qtft.plotting`` instead.

Re-exports everything from :mod:`qtft.plotting` so existing imports
(``import agglomeration_plotting``) keep working during the package migration.
Will be removed once notebooks and scripts are updated.
"""
from qtft.plotting import *  # noqa: F401,F403
from qtft import plotting as _plotting

globals().update({k: getattr(_plotting, k) for k in dir(_plotting) if not k.startswith("__")})
