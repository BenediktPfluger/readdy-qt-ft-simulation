"""Deprecated compatibility shim.

The comparison library functions moved to :mod:`qtft.comparison`; the CLI moved to
``scripts/analyze_ensemble.py``. This shim re-exports the comparison helpers so
notebooks doing ``import analyze_ensemble as ae`` keep working. Run the CLI via
``python scripts/analyze_ensemble.py`` instead. Will be removed once callers update.
"""
from qtft.comparison import *  # noqa: F401,F403
from qtft import comparison as _comparison

globals().update({k: getattr(_comparison, k) for k in dir(_comparison) if not k.startswith("__")})
