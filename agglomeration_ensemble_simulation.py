"""Deprecated compatibility shim. Use ``qtft.ensemble`` instead.

Re-exports everything from :mod:`qtft.ensemble` so existing imports
(``from agglomeration_ensemble_simulation import EnsembleSimulation``) keep
working during the package migration. Will be removed once callers are updated.
"""
from qtft.ensemble import *  # noqa: F401,F403
from qtft import ensemble as _ensemble

globals().update({k: getattr(_ensemble, k) for k in dir(_ensemble) if not k.startswith("__")})
