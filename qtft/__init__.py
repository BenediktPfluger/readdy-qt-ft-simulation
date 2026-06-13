"""qtft -- Qt-Ft encapsulin / ferritin agglomeration simulation (ReaDDy2).

Public API (see CODE_REVIEW.md / README.md for the architecture):

    from qtft import SimulationConfig, run_one, EnsembleSimulation
    import qtft.analysis as analysis
    import qtft.plotting as plotting   # requires matplotlib

Layering: config (single source of truth) -> system / engine (build + run) ->
analysis (matplotlib-free metrics) -> ensemble (multi-replica) ; plotting and
comparison are separate. analysis and plotting are intentionally NOT imported
here so that headless/ReaDDy-only contexts stay light.
"""
from .config import (
    NS_TO_US,
    _steps_to_us,
    ParticleConfig,
    TopologyConfig,
    LennardJonesConfig,
    SimulationConfig,
    format_param_string,
)
from .system import create_system
from .engine import (
    create_simulation,
    place_particles,
    run_simulation,
    equilibrate_system,
    run_one,
)
from .ensemble import EnsembleSimulation

__all__ = [
    "NS_TO_US",
    "_steps_to_us",
    "ParticleConfig",
    "TopologyConfig",
    "LennardJonesConfig",
    "SimulationConfig",
    "format_param_string",
    "create_system",
    "create_simulation",
    "place_particles",
    "run_simulation",
    "equilibrate_system",
    "run_one",
    "EnsembleSimulation",
]
