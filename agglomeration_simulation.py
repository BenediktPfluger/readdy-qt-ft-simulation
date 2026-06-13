"""Deprecated compatibility shim. Use the ``qtft`` package instead.

The old monolithic module was split into :mod:`qtft.config` (dataclasses + naming),
:mod:`qtft.system` (ReaDDy builders) and :mod:`qtft.engine` (build + run). This shim
re-exports all three so existing imports (``import agglomeration_simulation as sim``)
keep working during the package migration. Will be removed once callers are updated.
"""
from qtft.config import *  # noqa: F401,F403
from qtft.system import *  # noqa: F401,F403
from qtft.engine import *  # noqa: F401,F403
from qtft import config as _config, system as _system, engine as _engine

# Re-export private helpers too (``*`` skips underscore names) for legacy callers.
for _m in (_config, _system, _engine):
    globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith("__")})
