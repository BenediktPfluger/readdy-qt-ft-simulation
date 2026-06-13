"""
Qt-Ft Agglomeration Simulation Module

This module provides configuration and execution components for ReaDDy2-based 
Qt-Ft nanoparticle agglomeration simulations.

Contents:
    - Configuration dataclasses (ParticleConfig, TopologyConfig, etc.)
    - System setup functions (create_system, create_simulation, etc.)
    - Simulation execution functions (run_simulation, equilibrate_system)

Related modules:
    - agglomeration_analysis.py: Core analysis functions (no matplotlib dependency)
    - agglomeration_plotting.py: All plotting and visualization
    - agglomeration_ensemble_simulation.py: EnsembleSimulation class for multi-replica runs

Usage:
    import agglomeration_simulation as sim
    
    config = sim.SimulationConfig(...)
    system = sim.create_system(config)
    simulation = sim.create_simulation(system, config)
    sim.place_particles(simulation, config)
    trajectory = sim.run_simulation(simulation, config)
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import readdy


# =============================================================================
# CONSTANTS
# =============================================================================

# Time unit conversion
# NOTE: ReaDDy observables return STEP NUMBERS, not time in ns!
# To get actual time: step_number * timestep (ns) * NS_TO_US (ns->µs)
NS_TO_US = 1e-3  # nanoseconds to microseconds


def _steps_to_us(steps: np.ndarray, timestep: float) -> np.ndarray:
    """
    Convert ReaDDy step numbers to microseconds.
    
    ReaDDy observables return step numbers, not time in ns.
    Actual time = step_number * timestep (ns) * NS_TO_US (ns->µs)
    
    Parameters
    ----------
    steps : array-like
        Step numbers from ReaDDy observables
    timestep : float
        Integration timestep in nanoseconds (must be > 0)
    
    Returns
    -------
    ndarray
        Time in microseconds
    
    Raises
    ------
    ValueError
        If timestep is not positive
    """
    if timestep <= 0:
        raise ValueError(f"timestep must be positive, got {timestep}")
    return np.asarray(steps) * timestep * NS_TO_US


def _make_particle_rngs(seed: int) -> Tuple:
    """
    Create two statistically independent RNGs for Qt and Ft particle placement.
    
    Uses numpy's SeedSequence.spawn() to guarantee independent streams,
    rather than the fragile seed/seed+1 pattern.
    
    Parameters
    ----------
    seed : int
        Base random seed
    
    Returns
    -------
    rng_qt, rng_ft : tuple of numpy.random.Generator
        Independent RNGs for Qt and Ft particles
    """
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(2)
    rng_qt = np.random.default_rng(child_seeds[0])
    rng_ft = np.random.default_rng(child_seeds[1])
    return rng_qt, rng_ft


def _random_positions(config: "SimulationConfig") -> Tuple[np.ndarray, np.ndarray]:
    """Uniform random Qt and Ft positions inside the box.

    Uses independent per-species RNGs seeded from ``config.rng_seed`` (via
    ``_make_particle_rngs``), so equilibration placement and production placement
    are reproducible and consistent. Shared by ``place_particles`` and
    ``equilibrate_system`` to avoid duplicated placement logic.

    Returns
    -------
    pos_qt, pos_ft : tuple of ndarray
        Arrays of shape (n_qt, 3) and (n_ft, 3).
    """
    rng_qt, rng_ft = _make_particle_rngs(config.rng_seed)
    L = np.array(config.box_size, dtype=float)
    pos_qt = rng_qt.uniform(-L / 2.0, L / 2.0, size=(config.n_qt, 3))
    pos_ft = rng_ft.uniform(-L / 2.0, L / 2.0, size=(config.n_ft, 3))
    return pos_qt, pos_ft


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class ParticleConfig:
    """
    Configuration for a single particle species.
    
    Parameters
    ----------
    name : str
        Species name (e.g., "Qt", "Ft")
    radius : float
        Particle radius in nm
    diffusion : float
        Diffusion coefficient in nm²/ns
    cluster_diffusion : float, optional
        Diffusion coefficient when in a cluster. Defaults to same as diffusion.
    """
    name: str
    radius: float
    diffusion: float
    cluster_diffusion: Optional[float] = None
    
    def __post_init__(self):
        if self.cluster_diffusion is None:
            self.cluster_diffusion = self.diffusion
        self._validate()
    
    def _validate(self):
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive: {self.radius}")
        if self.diffusion <= 0:
            raise ValueError(f"Diffusion must be positive: {self.diffusion}")
        if self.cluster_diffusion <= 0:
            raise ValueError(f"Cluster diffusion must be positive: {self.cluster_diffusion}")
    
    @property
    def cluster_name(self) -> str:
        """Name of the clustered particle type (e.g., 'Qt' -> 'QtC')."""
        return f"{self.name}C"


@dataclass
class TopologyConfig:
    """
    Configuration for topology-based binding.
    
    Parameters
    ----------
    name : str
        Topology type name
    binding_radius : float
        Distance within which binding can occur (nm)
    kon : float
        Binding rate constant (nm³/(ns·particles))
    k_bond : float
        Harmonic bond force constant (kJ/(mol·nm²))
    """
    name: str = "QtFt_Cluster"
    binding_radius: float = 1.5
    kon: float = 10.0
    k_bond: float = 20.0
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.binding_radius <= 0:
            raise ValueError(f"Binding radius must be positive: {self.binding_radius}")
        if self.kon <= 0:
            raise ValueError(f"Binding rate must be positive: {self.kon}")
        if self.k_bond <= 0:
            raise ValueError(f"Bond constant must be positive: {self.k_bond}")


@dataclass
class LennardJonesConfig:
    """
    Configuration for Lennard-Jones potentials with per-pair epsilon values.
    
    Epsilon values follow a cascade defaulting hierarchy:
    
    1. Free-free pairs (always explicit):
       - epsilon_QtQt, epsilon_FtFt, epsilon_QtFt
    
    2. Cluster pairs (default to their free-particle counterparts):
       - epsilon_QtCQtC  defaults to epsilon_QtQt
       - epsilon_FtCFtC  defaults to epsilon_FtFt
       - epsilon_QtCFtC  defaults to epsilon_QtFt
    
    3. Mixed-state pairs (default to cluster value for that species pair):
       - epsilon_QtQtC   defaults to epsilon_QtCQtC
       - epsilon_FtFtC   defaults to epsilon_FtCFtC
       - epsilon_QtCFt   defaults to epsilon_QtCFtC
       - epsilon_QtFtC   defaults to epsilon_QtCFtC
    
    Setting any epsilon to 0 disables that interaction entirely (the potential
    is not registered in ReaDDy).
    
    Parameters
    ----------
    epsilon_QtQt : float
        LJ well depth for Qt-Qt interaction (kJ/mol)
    epsilon_FtFt : float
        LJ well depth for Ft-Ft interaction (kJ/mol)
    epsilon_QtFt : float
        LJ well depth for Qt-Ft interaction (kJ/mol)
    epsilon_QtCQtC : float, optional
        LJ well depth for QtC-QtC. Defaults to epsilon_QtQt.
    epsilon_FtCFtC : float, optional
        LJ well depth for FtC-FtC. Defaults to epsilon_FtFt.
    epsilon_QtCFtC : float, optional
        LJ well depth for QtC-FtC. Defaults to epsilon_QtFt.
    epsilon_QtQtC : float, optional
        LJ well depth for Qt-QtC. Defaults to epsilon_QtCQtC.
    epsilon_FtFtC : float, optional
        LJ well depth for Ft-FtC. Defaults to epsilon_FtCFtC.
    epsilon_QtCFt : float, optional
        LJ well depth for QtC-Ft. Defaults to epsilon_QtCFtC.
    epsilon_QtFtC : float, optional
        LJ well depth for Qt-FtC. Defaults to epsilon_QtCFtC.
    potential_type : str
        Type of potential: "WCA" (purely repulsive) or "LJ" (full Lennard-Jones)
        - WCA: cutoff at 2^(1/6)*sigma ≈ 1.122*sigma (no attractive part)
        - LJ: cutoff at 2.5*sigma (includes attractive well)
    cutoff_factor : float, optional
        Cutoff distance as multiple of sigma. If None, auto-calculated from potential_type.
    """
    # Primary free-free epsilon values
    epsilon_QtQt: float = 10.0
    epsilon_FtFt: float = 10.0
    epsilon_QtFt: float = 10.0
    
    # Cluster epsilon values (cascade from free-free)
    epsilon_QtCQtC: Optional[float] = None
    epsilon_FtCFtC: Optional[float] = None
    epsilon_QtCFtC: Optional[float] = None
    
    # Mixed-state epsilon values (cascade from cluster)
    epsilon_QtQtC: Optional[float] = None
    epsilon_FtFtC: Optional[float] = None
    epsilon_QtCFt: Optional[float] = None
    epsilon_QtFtC: Optional[float] = None
    
    # Potential type and cutoff
    potential_type: str = "WCA"
    cutoff_factor: Optional[float] = None
    
    # WCA cutoff factor: 2^(1/6) ≈ 1.122462
    WCA_CUTOFF_FACTOR: float = field(default=1.122462, repr=False)
    LJ_CUTOFF_FACTOR: float = field(default=2.5, repr=False)
    
    def __post_init__(self):
        # Cascade defaults: cluster values from free-free values
        if self.epsilon_QtCQtC is None:
            self.epsilon_QtCQtC = self.epsilon_QtQt
        if self.epsilon_FtCFtC is None:
            self.epsilon_FtCFtC = self.epsilon_FtFt
        if self.epsilon_QtCFtC is None:
            self.epsilon_QtCFtC = self.epsilon_QtFt
        
        # Cascade defaults: mixed-state values from cluster values
        if self.epsilon_QtQtC is None:
            self.epsilon_QtQtC = self.epsilon_QtCQtC
        if self.epsilon_FtFtC is None:
            self.epsilon_FtFtC = self.epsilon_FtCFtC
        if self.epsilon_QtCFt is None:
            self.epsilon_QtCFt = self.epsilon_QtCFtC
        if self.epsilon_QtFtC is None:
            self.epsilon_QtFtC = self.epsilon_QtCFtC
        
        # Auto-calculate cutoff_factor based on potential_type if not specified
        if self.cutoff_factor is None:
            if self.potential_type == "WCA":
                self.cutoff_factor = self.WCA_CUTOFF_FACTOR
            elif self.potential_type == "LJ":
                self.cutoff_factor = self.LJ_CUTOFF_FACTOR
            else:
                raise ValueError(f"Unknown potential_type: {self.potential_type}. Use 'WCA' or 'LJ'.")
        
        self._validate()
    
    def _validate(self):
        # All epsilon values must be non-negative (0 = disabled)
        eps_pairs = {
            "epsilon_QtQt": self.epsilon_QtQt,
            "epsilon_FtFt": self.epsilon_FtFt,
            "epsilon_QtFt": self.epsilon_QtFt,
            "epsilon_QtCQtC": self.epsilon_QtCQtC,
            "epsilon_FtCFtC": self.epsilon_FtCFtC,
            "epsilon_QtCFtC": self.epsilon_QtCFtC,
            "epsilon_QtQtC": self.epsilon_QtQtC,
            "epsilon_FtFtC": self.epsilon_FtFtC,
            "epsilon_QtCFt": self.epsilon_QtCFt,
            "epsilon_QtFtC": self.epsilon_QtFtC,
        }
        for name, val in eps_pairs.items():
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")
        
        if self.cutoff_factor <= 0:
            raise ValueError(f"Cutoff factor must be positive: {self.cutoff_factor}")
        if self.potential_type not in ("WCA", "LJ"):
            raise ValueError(f"potential_type must be 'WCA' or 'LJ', got: {self.potential_type}")
    
    def get_epsilon(self, type1: str, type2: str) -> float:
        """
        Get the epsilon value for a specific pair of particle types.
        
        Parameters
        ----------
        type1, type2 : str
            Particle type names (e.g., "Qt", "Ft", "QtC", "FtC")
        
        Returns
        -------
        float
            Epsilon value for this pair
        """
        # Normalize pair order (alphabetical)
        pair = tuple(sorted([type1, type2]))
        
        pair_map = {
            ("Ft", "Ft"): self.epsilon_FtFt,
            ("Ft", "FtC"): self.epsilon_FtFtC,
            ("Ft", "Qt"): self.epsilon_QtFt,
            ("Ft", "QtC"): self.epsilon_QtCFt,
            ("FtC", "FtC"): self.epsilon_FtCFtC,
            ("FtC", "Qt"): self.epsilon_QtFtC,
            ("FtC", "QtC"): self.epsilon_QtCFtC,
            ("Qt", "Qt"): self.epsilon_QtQt,
            ("Qt", "QtC"): self.epsilon_QtQtC,
            ("QtC", "QtC"): self.epsilon_QtCQtC,
        }
        
        if pair not in pair_map:
            raise ValueError(f"Unknown particle pair: {type1}-{type2}")
        
        return pair_map[pair]


@dataclass
class SimulationConfig:
    """
    Complete simulation configuration.
    
    This is the main configuration object that combines all settings.
    Can be created directly or via SimulationConfig.from_dict().
    
    Parameters
    ----------
    qt : ParticleConfig
        Qt particle configuration
    ft : ParticleConfig
        Ft particle configuration
    topology : TopologyConfig
        Topology/binding configuration
    lj : LennardJonesConfig
        Lennard-Jones potential configuration
    box_size : tuple of float
        Simulation box dimensions (Lx, Ly, Lz) in nm
    periodic_boundary : bool
        Use periodic boundary conditions
    temperature : float
        Temperature in Kelvin
    timestep : float
        Integration timestep in ns
    n_steps : int
        Total number of simulation steps
    record_stride : int
        Save trajectory every N steps
    observable_stride : int
        Record observables every N steps
    n_qt : int
        Number of Qt particles
    n_ft : int
        Number of Ft particles
    kernel : str
        ReaDDy kernel ("CPU" or "SingleCPU")
    n_threads : int
        Number of threads for CPU kernel
    rng_seed : int
        Random number generator seed
    output_file : str
        Output trajectory filename (.h5)
    """
    # Particle configurations
    qt: ParticleConfig = field(default_factory=lambda: ParticleConfig("Qt", 1.0, 5.0))
    ft: ParticleConfig = field(default_factory=lambda: ParticleConfig("Ft", 0.25, 15.0))
    
    # Topology configuration
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    
    # Lennard-Jones configuration
    lj: LennardJonesConfig = field(default_factory=LennardJonesConfig)
    
    # Simulation box
    box_size: Tuple[float, float, float] = (50.0, 50.0, 50.0)
    periodic_boundary: bool = True
    
    # Physical parameters
    temperature: float = 300.0

    # Potential used during equilibration (reactions are always off then). Defaults to
    # "WCA" (purely repulsive) so equilibration relaxes overlaps without attraction,
    # regardless of the production potential in lj.potential_type. Set to "LJ" to
    # equilibrate under the full attractive potential instead.
    equilibration_potential: str = "WCA"

    # Integration parameters
    timestep: float = 1e-4
    n_steps: int = 200000
    
    # Recording parameters
    record_stride: int = 10
    observable_stride: int = 10
    particles_observable_stride: Optional[int] = None  # None = disabled, saves disk space
    # Cadence for heavy, currently-unread observables (forces, virial). These are recorded
    # far less often than the rest to keep trajectory files small. None = 100 x observable_stride.
    heavy_observable_stride: Optional[int] = None
    
    # Particle counts
    n_qt: int = 200
    n_ft: int = 200
    
    # Kernel settings
    kernel: str = "CPU"
    n_threads: int = 4
    
    # Random seed
    rng_seed: int = 42
    
    # Output
    output_file: Optional[str] = None  # None = auto-generate from parameters
    
    def __post_init__(self):
        # Auto-generate output filename if not specified
        if self.output_file is None:
            self.output_file = self._generate_output_filename()
        
        warnings_list = self._validate()
        if warnings_list:
            for w in warnings_list:
                warnings.warn(w)
    
    def _generate_output_filename(self) -> str:
        """
        Generate descriptive output filename from simulation parameters.

        Uses the shared `format_param_string()` convention (same string used for
        ensemble folder names) plus a `.h5` suffix:

            {n_qt}Qt_{n_ft}Ft_{potential_type}_eQQ{e}_eFF{e}_eQF{e}_kon{kon}_dt{dt}ps_{total_time}us.h5

        Examples:
            200Qt_200Ft_WCA_eQQ10_eFF10_eQF10_kon10_dt0.10ps_20us.h5
            200Qt_400Ft_LJ_eQQ10_eFF10_eQF5_kon5.5_dt10ps_30us.h5
        """
        return f"{format_param_string(self)}.h5"
    
    def _validate(self) -> List[str]:
        """Validate configuration and return list of warnings."""
        warnings_list = []
        
        # Check box dimensions
        if any(d <= 0 for d in self.box_size):
            raise ValueError(f"Box dimensions must be positive: {self.box_size}")
        
        # Check temperature
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive: {self.temperature}")
        
        # Check timestep
        if self.timestep <= 0:
            raise ValueError(f"Timestep must be positive: {self.timestep}")

        # Check equilibration potential
        if self.equilibration_potential not in ("WCA", "LJ"):
            raise ValueError(
                f"equilibration_potential must be 'WCA' or 'LJ', got: {self.equilibration_potential}"
            )
        
        # Check counts
        if self.n_qt < 0 or self.n_ft < 0:
            raise ValueError(f"Particle counts must be non-negative: n_qt={self.n_qt}, n_ft={self.n_ft}")
        
        # Physics warnings
        r0_bond = self.qt.radius + self.ft.radius
        if self.topology.binding_radius < r0_bond:
            warnings_list.append(
                f"Warning: Binding radius ({self.topology.binding_radius} nm) < sum of radii "
                f"({r0_bond} nm). Particles may overlap before binding."
            )
        
        # Packing fraction warning
        box_volume = self.box_size[0] * self.box_size[1] * self.box_size[2]
        qt_volume = self.n_qt * (4/3) * np.pi * self.qt.radius**3
        ft_volume = self.n_ft * (4/3) * np.pi * self.ft.radius**3
        packing = (qt_volume + ft_volume) / box_volume
        
        if packing > 0.3:
            warnings_list.append(
                f"Warning: High packing fraction ({packing:.1%}). May cause initialization issues."
            )
        
        # Stride consistency
        if self.record_stride != self.observable_stride:
            warnings_list.append(
                f"Note: record_stride ({self.record_stride}) != observable_stride "
                f"({self.observable_stride}). This may complicate analysis."
            )
        
        return warnings_list
    
    @property
    def equilibrium_bond_length(self) -> float:
        """Equilibrium bond length for Qt-Ft bonds (nm)."""
        return self.qt.radius + self.ft.radius
    
    @property
    def total_simulation_time(self) -> float:
        """Total simulation time in ns."""
        return self.n_steps * self.timestep
    
    @property
    def total_simulation_time_us(self) -> float:
        """Total simulation time in µs."""
        return self.total_simulation_time / 1000.0

    @property
    def effective_heavy_observable_stride(self) -> int:
        """Resolved stride for heavy/unread observables (forces, virial).

        Defaults to 100 x observable_stride when heavy_observable_stride is None.
        """
        if self.heavy_observable_stride is not None:
            return int(self.heavy_observable_stride)
        return 100 * int(self.observable_stride)
    
    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "SimulationConfig":
        """
        Create SimulationConfig from a dictionary.
        
        Supports both nested format (from to_dict/save_json) and flat format
        (for backward compatibility and quick experimentation).
        
        Parameters
        ----------
        params : dict
            Dictionary with configuration parameters. Can be:
            
            Nested format (preferred, from to_dict()):
                {"qt": {"name": "Qt", "radius": 1.0, ...}, "ft": {...}, ...}
            
            Flat format (backward compatible):
                {"qt_name": "Qt", "qt_radius": 1.0, ...}
        
        Returns
        -------
        SimulationConfig
            Configured instance
        """
        # Detect format: nested if 'qt' key is a dict, flat otherwise
        is_nested = isinstance(params.get("qt"), dict)
        
        if is_nested:
            # Nested format (from to_dict / save_json)
            qt_params = params.get("qt", {})
            ft_params = params.get("ft", {})
            topo_params = params.get("topology", {})
            lj_params = params.get("lj", {})
            
            qt = ParticleConfig(
                name=qt_params.get("name", "Qt"),
                radius=qt_params.get("radius", 1.0),
                diffusion=qt_params.get("diffusion", 5.0),
                cluster_diffusion=qt_params.get("cluster_diffusion", None),
            )
            
            ft = ParticleConfig(
                name=ft_params.get("name", "Ft"),
                radius=ft_params.get("radius", 0.25),
                diffusion=ft_params.get("diffusion", 15.0),
                cluster_diffusion=ft_params.get("cluster_diffusion", None),
            )
            
            topology = TopologyConfig(
                name=topo_params.get("name", "QtFt_Cluster"),
                binding_radius=topo_params.get("binding_radius", 1.5),
                kon=topo_params.get("kon", 10.0),
                k_bond=topo_params.get("k_bond", 20.0),
            )
            
            lj = LennardJonesConfig(
                epsilon_QtQt=lj_params.get("epsilon_QtQt", 10.0),
                epsilon_FtFt=lj_params.get("epsilon_FtFt", 10.0),
                epsilon_QtFt=lj_params.get("epsilon_QtFt", 10.0),
                epsilon_QtCQtC=lj_params.get("epsilon_QtCQtC", None),
                epsilon_FtCFtC=lj_params.get("epsilon_FtCFtC", None),
                epsilon_QtCFtC=lj_params.get("epsilon_QtCFtC", None),
                epsilon_QtQtC=lj_params.get("epsilon_QtQtC", None),
                epsilon_FtFtC=lj_params.get("epsilon_FtFtC", None),
                epsilon_QtCFt=lj_params.get("epsilon_QtCFt", None),
                epsilon_QtFtC=lj_params.get("epsilon_QtFtC", None),
                potential_type=lj_params.get("potential_type", "WCA"),
                cutoff_factor=lj_params.get("cutoff_factor", None),
            )
        else:
            # Flat format (backward compatible)
            qt = ParticleConfig(
                name=params.get("qt_name", "Qt"),
                radius=params.get("qt_radius", 1.0),
                diffusion=params.get("qt_diffusion", 5.0),
                cluster_diffusion=params.get("qt_cluster_diffusion", None),
            )
            
            ft = ParticleConfig(
                name=params.get("ft_name", "Ft"),
                radius=params.get("ft_radius", 0.25),
                diffusion=params.get("ft_diffusion", 15.0),
                cluster_diffusion=params.get("ft_cluster_diffusion", None),
            )
            
            topology = TopologyConfig(
                name=params.get("topology_name", "QtFt_Cluster"),
                binding_radius=params.get("binding_radius", 1.5),
                kon=params.get("kon", 10.0),
                k_bond=params.get("k_bond", 20.0),
            )
            
            lj = LennardJonesConfig(
                epsilon_QtQt=params.get("epsilon_QtQt", 10.0),
                epsilon_FtFt=params.get("epsilon_FtFt", 10.0),
                epsilon_QtFt=params.get("epsilon_QtFt", 10.0),
                epsilon_QtCQtC=params.get("epsilon_QtCQtC", None),
                epsilon_FtCFtC=params.get("epsilon_FtCFtC", None),
                epsilon_QtCFtC=params.get("epsilon_QtCFtC", None),
                epsilon_QtQtC=params.get("epsilon_QtQtC", None),
                epsilon_FtFtC=params.get("epsilon_FtFtC", None),
                epsilon_QtCFt=params.get("epsilon_QtCFt", None),
                epsilon_QtFtC=params.get("epsilon_QtFtC", None),
                potential_type=params.get("potential_type", "WCA"),
                cutoff_factor=params.get("cutoff_factor", None),
            )
        
        # Handle box_size - convert list to tuple if needed
        box_size = params.get("box_size", (50.0, 50.0, 50.0))
        if isinstance(box_size, list):
            box_size = tuple(box_size)
        
        # Build SimulationConfig
        return cls(
            qt=qt,
            ft=ft,
            topology=topology,
            lj=lj,
            box_size=box_size,
            periodic_boundary=params.get("periodic_boundary", True),
            temperature=params.get("temperature", 300.0),
            equilibration_potential=params.get("equilibration_potential", "WCA"),
            timestep=params.get("timestep", 1e-4),
            n_steps=params.get("n_steps", 200000),
            record_stride=params.get("record_stride", 10),
            observable_stride=params.get("observable_stride", 10),
            particles_observable_stride=params.get("particles_observable_stride", None),
            heavy_observable_stride=params.get("heavy_observable_stride", None),
            n_qt=params.get("n_qt", 200),
            n_ft=params.get("n_ft", 200),
            kernel=params.get("kernel", "CPU"),
            n_threads=params.get("n_threads", 4),
            rng_seed=params.get("rng_seed", 42),
            output_file=params.get("output_file", None),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a nested dictionary suitable for JSON serialization.
        
        This format can be loaded back with from_dict() or load_json().
        """
        return {
            # Nested particle configs
            "qt": {
                "name": self.qt.name,
                "radius": self.qt.radius,
                "diffusion": self.qt.diffusion,
                "cluster_diffusion": self.qt.cluster_diffusion,
            },
            "ft": {
                "name": self.ft.name,
                "radius": self.ft.radius,
                "diffusion": self.ft.diffusion,
                "cluster_diffusion": self.ft.cluster_diffusion,
            },
            # Nested topology config
            "topology": {
                "name": self.topology.name,
                "binding_radius": self.topology.binding_radius,
                "kon": self.topology.kon,
                "k_bond": self.topology.k_bond,
            },
            # Nested LJ config (resolved values, not None)
            "lj": {
                "epsilon_QtQt": self.lj.epsilon_QtQt,
                "epsilon_FtFt": self.lj.epsilon_FtFt,
                "epsilon_QtFt": self.lj.epsilon_QtFt,
                "epsilon_QtCQtC": self.lj.epsilon_QtCQtC,
                "epsilon_FtCFtC": self.lj.epsilon_FtCFtC,
                "epsilon_QtCFtC": self.lj.epsilon_QtCFtC,
                "epsilon_QtQtC": self.lj.epsilon_QtQtC,
                "epsilon_FtFtC": self.lj.epsilon_FtFtC,
                "epsilon_QtCFt": self.lj.epsilon_QtCFt,
                "epsilon_QtFtC": self.lj.epsilon_QtFtC,
                "potential_type": self.lj.potential_type,
                "cutoff_factor": self.lj.cutoff_factor,
            },
            # Simulation parameters
            "box_size": list(self.box_size),  # Convert tuple to list for JSON
            "periodic_boundary": self.periodic_boundary,
            "temperature": self.temperature,
            "equilibration_potential": self.equilibration_potential,
            "timestep": self.timestep,
            "n_steps": self.n_steps,
            "record_stride": self.record_stride,
            "observable_stride": self.observable_stride,
            "particles_observable_stride": self.particles_observable_stride,
            "heavy_observable_stride": self.heavy_observable_stride,
            "n_qt": self.n_qt,
            "n_ft": self.n_ft,
            "kernel": self.kernel,
            "n_threads": self.n_threads,
            "rng_seed": self.rng_seed,
            "output_file": self.output_file,
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a flat dictionary.
        
        Useful for logging, display, or parameter sweeps.
        Note: Use to_dict() for serialization/deserialization.
        """
        return {
            # Qt parameters
            "qt_name": self.qt.name,
            "qt_radius": self.qt.radius,
            "qt_diffusion": self.qt.diffusion,
            "qt_cluster_diffusion": self.qt.cluster_diffusion,
            "qt_cluster_name": self.qt.cluster_name,
            # Ft parameters
            "ft_name": self.ft.name,
            "ft_radius": self.ft.radius,
            "ft_diffusion": self.ft.diffusion,
            "ft_cluster_diffusion": self.ft.cluster_diffusion,
            "ft_cluster_name": self.ft.cluster_name,
            # Topology parameters
            "topology_name": self.topology.name,
            "binding_radius": self.topology.binding_radius,
            "kon": self.topology.kon,
            "k_bond": self.topology.k_bond,
            "equilibrium_bond_length": self.equilibrium_bond_length,
            # LJ parameters
            "epsilon_QtQt": self.lj.epsilon_QtQt,
            "epsilon_FtFt": self.lj.epsilon_FtFt,
            "epsilon_QtFt": self.lj.epsilon_QtFt,
            "epsilon_QtCQtC": self.lj.epsilon_QtCQtC,
            "epsilon_FtCFtC": self.lj.epsilon_FtCFtC,
            "epsilon_QtCFtC": self.lj.epsilon_QtCFtC,
            "epsilon_QtQtC": self.lj.epsilon_QtQtC,
            "epsilon_FtFtC": self.lj.epsilon_FtFtC,
            "epsilon_QtCFt": self.lj.epsilon_QtCFt,
            "epsilon_QtFtC": self.lj.epsilon_QtFtC,
            "potential_type": self.lj.potential_type,
            "cutoff_factor": self.lj.cutoff_factor,
            # Simulation parameters
            "box_size": self.box_size,
            "periodic_boundary": self.periodic_boundary,
            "temperature": self.temperature,
            "equilibration_potential": self.equilibration_potential,
            "timestep": self.timestep,
            "n_steps": self.n_steps,
            "total_time_ns": self.total_simulation_time,
            "total_time_us": self.total_simulation_time_us,
            "record_stride": self.record_stride,
            "observable_stride": self.observable_stride,
            "particles_observable_stride": self.particles_observable_stride,
            "heavy_observable_stride": self.heavy_observable_stride,
            "effective_heavy_observable_stride": self.effective_heavy_observable_stride,
            "n_qt": self.n_qt,
            "n_ft": self.n_ft,
            "kernel": self.kernel,
            "n_threads": self.n_threads,
            "rng_seed": self.rng_seed,
            "output_file": self.output_file,
        }

    def print_summary(self):
        """Print a formatted summary of the configuration."""
        print("=" * 60)
        print("SIMULATION CONFIGURATION")
        print("=" * 60)
        print(f"\nParticles:")
        print(f"  Qt: r={self.qt.radius} nm, D={self.qt.diffusion} nm²/ns "
              f"(cluster: D={self.qt.cluster_diffusion})")
        print(f"  Ft: r={self.ft.radius} nm, D={self.ft.diffusion} nm²/ns "
              f"(cluster: D={self.ft.cluster_diffusion})")
        print(f"  Counts: {self.n_qt} Qt + {self.n_ft} Ft = {self.n_qt + self.n_ft} total")
        print(f"\nTopology:")
        print(f"  Binding radius: {self.topology.binding_radius} nm")
        print(f"  Binding rate (kon): {self.topology.kon} nm³/(ns·part)")
        print(f"  Bond stiffness: {self.topology.k_bond} kJ/(mol·nm²)")
        print(f"  Equilibrium bond length: {self.equilibrium_bond_length} nm")
        print(f"\nLennard-Jones:")
        print(f"  Potential type: {self.lj.potential_type}")
        print(f"  Cutoff factor: {self.lj.cutoff_factor:.3f}")
        lj = self.lj
        print(f"  ε Qt-Qt: {lj.epsilon_QtQt} kJ/mol", end="")
        if lj.epsilon_QtCQtC == 0:
            print(f"  (QtC-QtC: disabled)", end="")
        elif lj.epsilon_QtCQtC != lj.epsilon_QtQt:
            print(f"  (QtC-QtC: {lj.epsilon_QtCQtC})", end="")
        print()
        print(f"  ε Ft-Ft: {lj.epsilon_FtFt} kJ/mol", end="")
        if lj.epsilon_FtCFtC == 0:
            print(f"  (FtC-FtC: disabled)", end="")
        elif lj.epsilon_FtCFtC != lj.epsilon_FtFt:
            print(f"  (FtC-FtC: {lj.epsilon_FtCFtC})", end="")
        print()
        print(f"  ε Qt-Ft: {lj.epsilon_QtFt} kJ/mol", end="")
        if lj.epsilon_QtCFtC == 0:
            print(f"  (QtC-FtC: disabled)", end="")
        elif lj.epsilon_QtCFtC != lj.epsilon_QtFt:
            print(f"  (QtC-FtC: {lj.epsilon_QtCFtC})", end="")
        print()
        # Show cluster/mixed-state summary if all defaults
        all_cluster_default = (
            lj.epsilon_QtCQtC == lj.epsilon_QtQt and
            lj.epsilon_FtCFtC == lj.epsilon_FtFt and
            lj.epsilon_QtCFtC == lj.epsilon_QtFt
        )
        all_mixed_default = (
            lj.epsilon_QtQtC == lj.epsilon_QtCQtC and
            lj.epsilon_FtFtC == lj.epsilon_FtCFtC and
            lj.epsilon_QtCFt == lj.epsilon_QtCFtC and
            lj.epsilon_QtFtC == lj.epsilon_QtCFtC
        )
        if all_cluster_default and all_mixed_default:
            print(f"  Cluster/mixed ε: same as free (default)")
        elif all_mixed_default:
            print(f"  Mixed-state ε: same as cluster (default)")
        else:
            print(f"  Mixed-state ε: Qt-QtC={lj.epsilon_QtQtC}, Ft-FtC={lj.epsilon_FtFtC}, "
                  f"QtC-Ft={lj.epsilon_QtCFt}, Qt-FtC={lj.epsilon_QtFtC}")
        print(f"\nSimulation:")
        print(f"  Box: {self.box_size[0]} × {self.box_size[1]} × {self.box_size[2]} nm")
        print(f"  Temperature: {self.temperature} K")
        print(f"  Equilibration potential: {self.equilibration_potential}")
        print(f"  Timestep: {self.timestep} ns ({self.timestep * 1e3:.2f} ps)")
        print(f"  Steps: {self.n_steps:,} ({self.total_simulation_time_us:.1f} µs total)")
        print(f"  Output: {self.output_file}")
        print("=" * 60)
    
    def save_json(self, filepath: str):
        """
        Save configuration to a JSON file.
        
        The saved configuration can be loaded later with load_json().
        
        Parameters
        ----------
        filepath : str
            Path to save the JSON file
        """
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ Configuration saved to {filepath}")
    
    @classmethod
    def load_json(cls, filepath: str) -> "SimulationConfig":
        """
        Load configuration from a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to the JSON file (saved with save_json())
        
        Returns
        -------
        SimulationConfig
            Reconstructed configuration object
        """
        # Check file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                params = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in configuration file '{filepath}': {e.msg}",
                e.doc, e.pos
            )
        
        # Detect format and validate required fields
        is_nested = isinstance(params.get("qt"), dict)
        
        if is_nested:
            # Nested format - check for nested structure
            required_fields = ['qt', 'ft', 'topology', 'lj', 'box_size', 'n_steps']
            missing_fields = [f for f in required_fields if f not in params]
            if missing_fields:
                raise ValueError(
                    f"Configuration file '{filepath}' is missing required fields: {missing_fields}"
                )
        else:
            # Flat format - check for essential fields (more lenient)
            required_fields = ['box_size', 'n_steps']
            missing_fields = [f for f in required_fields if f not in params]
            if missing_fields:
                raise ValueError(
                    f"Configuration file '{filepath}' is missing required fields: {missing_fields}"
                )
        
        # Use from_dict to reconstruct (it handles both formats)
        try:
            config = cls.from_dict(params)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Error reconstructing configuration from '{filepath}': {e}"
            )
        
        print(f"✓ Configuration loaded from {filepath}")
        return config


# =============================================================================
# PARAMETER-STRING NAMING
# =============================================================================

def format_param_string(config: "SimulationConfig") -> str:
    """Canonical parameter string for auto filenames and ensemble folder names.

    Single source of truth for the naming convention so the single-run trajectory
    filename and the ensemble folder name can never drift apart.

    Format::

        {n_qt}Qt_{n_ft}Ft_{potential_type}_eQQ{e}_eFF{e}_eQF{e}_kon{kon}_dt{dt}ps_{total_time}us

    Numbers are formatted without trailing zeros (e.g. kon10, eQQ2.5, dt20ps, 100us).
    """
    lj = config.lj

    def fmt_eps(val):
        return f"{int(val)}" if val == int(val) else f"{val}"

    eqq = f"eQQ{fmt_eps(lj.epsilon_QtQt)}"
    eff = f"eFF{fmt_eps(lj.epsilon_FtFt)}"
    eqf = f"eQF{fmt_eps(lj.epsilon_QtFt)}"

    kon_val = config.topology.kon
    kon_str = f"kon{int(kon_val)}" if kon_val == int(kon_val) else f"kon{kon_val}"

    dt_ps = config.timestep * 1000  # ns -> ps
    dt_str = f"dt{dt_ps:.0f}ps" if dt_ps >= 1 else f"dt{dt_ps:.2f}ps"

    total_us = config.total_simulation_time_us
    time_str = f"{total_us:.0f}us" if total_us >= 1 else f"{total_us:.2f}us"

    return (
        f"{config.n_qt}Qt_{config.n_ft}Ft_"
        f"{lj.potential_type}_{eqq}_{eff}_{eqf}_{kon_str}_{dt_str}_{time_str}"
    )


# =============================================================================
# SYSTEM SETUP FUNCTIONS
# =============================================================================

def create_system(config: SimulationConfig, equilibration_mode: bool = False) -> readdy.ReactionDiffusionSystem:
    """
    Create and configure a ReaDDy reaction-diffusion system.
    
    This sets up:
    - Particle species (free and clustered types)
    - Lennard-Jones potentials for excluded volume
    - Topology type with harmonic bonds
    - Spatial reactions for binding (unless equilibration_mode=True)
    
    Parameters
    ----------
    config : SimulationConfig
        Complete simulation configuration
    equilibration_mode : bool
        If True, spatial reactions are NOT registered (no binding occurs).
        Use this for equilibration runs to relax initial particle positions.
    
    Returns
    -------
    readdy.ReactionDiffusionSystem
        Configured system ready for simulation
    """
    # Create system with correct box size from the start
    system = readdy.ReactionDiffusionSystem(box_size=list(config.box_size))
    
    if not config.periodic_boundary:
        system.periodic_boundary_conditions = [False, False, False]
    
    # Add species
    _add_species(system, config)

    # Add potentials. During equilibration, use config.equilibration_potential (WCA by
    # default) so overlaps relax under a purely repulsive potential regardless of the
    # production potential in config.lj.potential_type.
    potential_override = config.equilibration_potential if equilibration_mode else None
    _add_potentials(system, config, potential_type=potential_override)

    # Add topologies and reactions (skip reactions in equilibration mode)
    _add_topologies(system, config, equilibration_mode=equilibration_mode)
    
    mode_str = " (equilibration mode - no reactions)" if equilibration_mode else ""
    print(f"✓ System created: {config.box_size[0]}×{config.box_size[1]}×{config.box_size[2]} nm box{mode_str}")
    
    return system


def _add_species(system: readdy.ReactionDiffusionSystem, config: SimulationConfig):
    """Add particle species to the system."""
    # Free particle types
    system.add_topology_species(config.qt.name, float(config.qt.diffusion))
    system.add_topology_species(config.ft.name, float(config.ft.diffusion))
    
    # Clustered particle types
    system.add_topology_species(config.qt.cluster_name, float(config.qt.cluster_diffusion))
    system.add_topology_species(config.ft.cluster_name, float(config.ft.cluster_diffusion))
    
    print(f"✓ Species: {config.qt.name}, {config.ft.name}, "
          f"{config.qt.cluster_name}, {config.ft.cluster_name}")


def _add_potentials(
    system: readdy.ReactionDiffusionSystem,
    config: SimulationConfig,
    potential_type: Optional[str] = None,
):
    """Add Lennard-Jones potentials to the system.

    Uses per-pair epsilon values from config.lj. Pairs with epsilon=0
    are skipped (interaction disabled).

    Parameters
    ----------
    potential_type : str, optional
        Override for the potential type ("WCA" or "LJ"). If None, uses
        config.lj.potential_type / config.lj.cutoff_factor (production). When set
        (e.g. "WCA" for equilibration), the cutoff factor is derived from that type
        while the same per-pair epsilon values are kept.
    """
    lj = config.lj

    # Resolve potential type and cutoff factor (production vs equilibration override)
    if potential_type is None:
        potential_type = lj.potential_type
        cf = lj.cutoff_factor
    elif potential_type == "WCA":
        cf = lj.WCA_CUTOFF_FACTOR
    elif potential_type == "LJ":
        cf = lj.LJ_CUTOFF_FACTOR
    else:
        raise ValueError(f"potential_type must be 'WCA' or 'LJ', got: {potential_type}")

    # Calculate sigma values
    sigma_qq = 2.0 * config.qt.radius
    sigma_ff = 2.0 * config.ft.radius
    sigma_qf = config.qt.radius + config.ft.radius

    # Calculate cutoffs
    cutoff_qq = cf * sigma_qq
    cutoff_ff = cf * sigma_ff
    cutoff_qf = cf * sigma_qf
    
    qt = config.qt.name
    ft = config.ft.name
    qtc = config.qt.cluster_name
    ftc = config.ft.cluster_name

    n_registered = 0
    n_skipped = 0
    
    def add_lj(t1, t2, epsilon, sigma, cutoff):
        nonlocal n_registered, n_skipped
        if epsilon == 0:
            n_skipped += 1
            return
        system.potentials.add_lennard_jones(
            t1, t2, m=12, n=6,
            epsilon=float(epsilon), sigma=float(sigma), cutoff=float(cutoff)
        )
        n_registered += 1
    
    # All 10 possible pairwise interactions with per-pair epsilon values
    # Free-free pairs
    add_lj(qt,  qt,  lj.epsilon_QtQt,   sigma_qq, cutoff_qq)
    add_lj(ft,  ft,  lj.epsilon_FtFt,   sigma_ff, cutoff_ff)
    add_lj(qt,  ft,  lj.epsilon_QtFt,   sigma_qf, cutoff_qf)
    
    # Cluster-cluster pairs (same species)
    add_lj(qtc, qtc, lj.epsilon_QtCQtC,  sigma_qq, cutoff_qq)
    add_lj(ftc, ftc, lj.epsilon_FtCFtC,  sigma_ff, cutoff_ff)
    
    # Cluster-cluster (cross-species)
    add_lj(qtc, ftc, lj.epsilon_QtCFtC,  sigma_qf, cutoff_qf)
    
    # Mixed-state pairs (same species: one free, one clustered)
    add_lj(qt,  qtc, lj.epsilon_QtQtC,   sigma_qq, cutoff_qq)
    add_lj(ft,  ftc, lj.epsilon_FtFtC,   sigma_ff, cutoff_ff)
    
    # Mixed-state pairs (cross-species: one clustered, one free)
    add_lj(qtc, ft,  lj.epsilon_QtCFt,   sigma_qf, cutoff_qf)
    add_lj(qt,  ftc, lj.epsilon_QtFtC,   sigma_qf, cutoff_qf)
    
    skip_str = f", {n_skipped} disabled" if n_skipped > 0 else ""
    print(f"✓ {potential_type} potentials ({n_registered} registered{skip_str}): "
          f"ε_QQ={lj.epsilon_QtQt}, ε_FF={lj.epsilon_FtFt}, ε_QF={lj.epsilon_QtFt}")


def _add_topologies(system: readdy.ReactionDiffusionSystem, config: SimulationConfig, equilibration_mode: bool = False):
    """Add topology type, bonds, and spatial reactions.
    
    Parameters
    ----------
    system : readdy.ReactionDiffusionSystem
        The system to configure
    config : SimulationConfig
        Simulation configuration
    equilibration_mode : bool
        If True, skip registering spatial reactions (no binding occurs)
    """
    topo = config.topology
    qt = config.qt.name
    ft = config.ft.name
    qtc = config.qt.cluster_name
    ftc = config.ft.cluster_name
    r0 = config.equilibrium_bond_length
    
    # Register topology type
    system.topologies.add_type(topo.name)
    
    # Configure harmonic bonds for all possible connections
    bond_pairs = [(qt, ft), (qtc, ftc), (qtc, ft), (qt, ftc)]
    for t1, t2 in bond_pairs:
        system.topologies.configure_harmonic_bond(
            t1, t2, force_constant=float(topo.k_bond), length=float(r0)
        )
    
    # Skip spatial reactions in equilibration mode
    if equilibration_mode:
        print(f"✓ Topology '{topo.name}': bonds configured, reactions DISABLED (equilibration mode)")
        return
    
    # Spatial reactions for binding
    reactions = [
        # Seeding: Qt + Ft -> QtC--FtC
        (f"seed_{topo.name}",
         f"{topo.name}({qt}) + {topo.name}({ft}) -> {topo.name}({qtc}--{ftc})"),
        # Growth: QtC + Ft -> QtC--FtC
        (f"grow_QtC_Ft_{topo.name}",
         f"{topo.name}({qtc}) + {topo.name}({ft}) -> {topo.name}({qtc}--{ftc})"),
        # Growth: FtC + Qt -> FtC--QtC
        (f"grow_FtC_Qt_{topo.name}",
         f"{topo.name}({ftc}) + {topo.name}({qt}) -> {topo.name}({ftc}--{qtc})"),
        # Merging: QtC + FtC -> QtC--FtC
        (f"merge_QtC_FtC_{topo.name}",
         f"{topo.name}({qtc}) + {topo.name}({ftc}) -> {topo.name}({qtc}--{ftc})"),
    ]
    
    for name, descriptor in reactions:
        system.topologies.add_spatial_reaction(
            f"{name}: {descriptor}",
            rate=float(topo.kon),
            radius=float(topo.binding_radius)
        )
    
    print(f"✓ Topology '{topo.name}': binding_radius={topo.binding_radius} nm, "
          f"kon={topo.kon}, k_bond={topo.k_bond}")


def create_simulation(
    system: readdy.ReactionDiffusionSystem,
    config: SimulationConfig,
    overwrite: bool = True,
) -> readdy.Simulation:
    """
    Create and configure a ReaDDy simulation.
    
    Parameters
    ----------
    system : readdy.ReactionDiffusionSystem
        System created by create_system()
    config : SimulationConfig
        Simulation configuration
    overwrite : bool
        If True, overwrite existing output file
    
    Returns
    -------
    readdy.Simulation
        Configured simulation ready to run
    """
    # Handle existing output file
    if os.path.exists(config.output_file):
        if overwrite:
            os.remove(config.output_file)
            print(f"✓ Removed existing file: {config.output_file}")
        else:
            raise FileExistsError(
                f"Output file exists: {config.output_file}. "
                "Set overwrite=True or choose a different filename."
            )
    
    # Create simulation
    simulation = system.simulation(
        kernel=config.kernel,
        output_file=config.output_file,
        integrator="EulerBDIntegrator",
        reaction_handler="Gillespie",
        evaluate_topology_reactions=True,
        evaluate_forces=True,
        evaluate_observables=True,
        skin=0.0,
    )
    
    # Set temperature (timestep is passed to simulation.run() instead)
    simulation.temperature = float(config.temperature)
    
    # Configure threading
    if config.kernel == "CPU" and config.n_threads is not None:
        simulation.kernel_configuration.n_threads = int(config.n_threads)
    
    # Enable trajectory recording
    simulation.record_trajectory(stride=int(config.record_stride))
    
    # Register observables
    _register_observables(simulation, config)
    
    print(f"✓ Simulation created: {config.kernel} kernel, {config.n_threads} threads")
    
    return simulation


def _register_observables(simulation: readdy.Simulation, config: SimulationConfig):
    """Register observables to record during simulation."""
    stride = config.observable_stride
    
    # Particle counts by type
    types = [config.qt.name, config.ft.name, config.qt.cluster_name, config.ft.cluster_name]
    simulation.observe.number_of_particles(stride=stride, types=types)
    
    # Particle positions and types (optional - for faster structural analysis)
    # When enabled, allows using read_observable_particles() instead of trajectory.read()
    # Trade-off: faster analysis but larger HDF5 files (position data stored twice)
    if config.particles_observable_stride is not None:
        simulation.observe.particles(stride=config.particles_observable_stride)
        particles_obs_str = f", particles observable stride={config.particles_observable_stride}"
    else:
        particles_obs_str = ""
    
    # Topologies
    simulation.observe.topologies(stride=stride)
    
    # Thermodynamic observables
    simulation.observe.energy(stride=stride)
    simulation.observe.pressure(stride=stride, physical_particles=None)

    # Heavy observables that are currently not read by any analysis (forces, virial).
    # Recorded on a much coarser stride to keep trajectory files small. Forces in
    # particular dominates trajectory.h5 size when recorded every step.
    heavy_stride = config.effective_heavy_observable_stride
    simulation.observe.forces(stride=heavy_stride, types=None)
    simulation.observe.virial(stride=heavy_stride)
    
    # Reaction counts
    simulation.observe.reaction_counts(stride=stride)
    
    # Radial distribution function
    # Configure RDF range based on particle sizes for better resolution
    box = config.box_size
    contact_distance = config.qt.radius + config.ft.radius
    rdf_max = min(3.0 * contact_distance, 0.5 * min(box))  # 3× contact or half box
    n_bins = 200  # Better resolution than default 100
    simulation.observe.rdf(
        stride=stride,
        bin_borders=(0.0, rdf_max, n_bins),
        types_count_from=[config.qt.name, config.qt.cluster_name],
        types_count_to=[config.ft.name, config.ft.cluster_name],
        particle_to_density=1.0 / (box[0] * box[1] * box[2])
    )
    
    print(f"✓ Observables registered (stride={stride}, "
          f"forces/virial stride={heavy_stride}{particles_obs_str})")


def place_particles(
    simulation: readdy.Simulation,
    config: SimulationConfig,
    positions_qt: Optional[np.ndarray] = None,
    positions_ft: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Place particles in the simulation box.
    
    Parameters
    ----------
    simulation : readdy.Simulation
        Simulation to add particles to
    config : SimulationConfig
        Configuration with particle counts and box size
    positions_qt : ndarray, optional
        Pre-defined positions for Qt particles. If None, random positions are generated.
    positions_ft : ndarray, optional
        Pre-defined positions for Ft particles. If None, random positions are generated.
    
    Returns
    -------
    pos_qt, pos_ft : tuple of ndarray
        Arrays of positions used for Qt and Ft particles
    """
    # Random positions for any species not explicitly provided (deterministic from seed)
    rand_qt, rand_ft = _random_positions(config)

    # Generate or use provided positions for Qt
    if positions_qt is not None:
        pos_qt = np.asarray(positions_qt)
        if pos_qt.shape[0] != config.n_qt:
            raise ValueError(f"positions_qt has {pos_qt.shape[0]} particles, expected {config.n_qt}")
        placement_qt = "provided"
    else:
        pos_qt = rand_qt
        placement_qt = "random"

    # Generate or use provided positions for Ft
    if positions_ft is not None:
        pos_ft = np.asarray(positions_ft)
        if pos_ft.shape[0] != config.n_ft:
            raise ValueError(f"positions_ft has {pos_ft.shape[0]} particles, expected {config.n_ft}")
        placement_ft = "provided"
    else:
        pos_ft = rand_ft
        placement_ft = "random"
    
    # Add as single-particle topologies
    for p in pos_qt:
        simulation.add_topology(config.topology.name, [config.qt.name], p.reshape(1, 3))
    
    for p in pos_ft:
        simulation.add_topology(config.topology.name, [config.ft.name], p.reshape(1, 3))
    
    print(f"✓ Placed {config.n_qt} Qt ({placement_qt}) + {config.n_ft} Ft ({placement_ft}) particles")
    
    return pos_qt, pos_ft


def run_simulation(
    simulation: readdy.Simulation,
    config: SimulationConfig,
    show_progress: bool = True,
) -> readdy.Trajectory:
    """
    Execute the simulation and return the trajectory.
    
    Parameters
    ----------
    simulation : readdy.Simulation
        Configured simulation with particles placed
    config : SimulationConfig
        Simulation configuration
    show_progress : bool
        Print progress messages
    
    Returns
    -------
    readdy.Trajectory
        Trajectory object for analysis
    """
    if show_progress:
        print(f"\n{'=' * 60}")
        print(f"RUNNING SIMULATION")
        print(f"  Particles: {config.n_qt} Qt + {config.n_ft} Ft")
        print(f"  Duration: {config.total_simulation_time_us:.1f} µs ({config.n_steps:,} steps)")
        print(f"{'=' * 60}\n")
    
    simulation.run(int(config.n_steps), float(config.timestep))
    
    if show_progress:
        print(f"\n{'=' * 60}")
        print("SIMULATION COMPLETE")
        print(f"{'=' * 60}\n")
    
    # Load trajectory
    if not os.path.exists(config.output_file):
        raise RuntimeError(f"Output file not created: {config.output_file}")
    
    return readdy.Trajectory(config.output_file)


def equilibrate_system(
    config: SimulationConfig,
    n_steps: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run an equilibration simulation to relax initial particle positions.
    
    This runs a short simulation WITHOUT binding reactions, allowing particles
    to move apart if they were initially overlapping. The final positions are
    returned and can be used as initial positions for the production run.
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
    n_steps : int
        Number of equilibration steps (default: 10000)
    
    Returns
    -------
    positions_qt, positions_ft : tuple of ndarray
        Equilibrated positions for Qt and Ft particles
    
    Example
    -------
    >>> # Equilibrate first
    >>> pos_qt, pos_ft = equilibrate_system(config, n_steps=10000)
    >>> # Then run production with equilibrated positions
    >>> system = create_system(config)
    >>> simulation = create_simulation(system, config)
    >>> place_particles(simulation, config, positions_qt=pos_qt, positions_ft=pos_ft)
    >>> trajectory = run_simulation(simulation, config)
    """
    print(f"\n{'=' * 60}")
    print("EQUILIBRATION")
    print(f"  Running {n_steps:,} steps without reactions "
          f"({config.equilibration_potential} potential)")
    print(f"{'=' * 60}\n")
    
    # Create system in equilibration mode (no reactions)
    system = create_system(config, equilibration_mode=True)
    
    # Create simulation without output file
    eq_simulation = system.simulation(
        kernel=config.kernel,
        output_file="",  # No output file
        integrator="EulerBDIntegrator",
        reaction_handler="Gillespie",
        evaluate_topology_reactions=True,
        evaluate_forces=True,
        evaluate_observables=False,  # No need to record observables
        skin=0.0,
    )
    
    # Set temperature
    eq_simulation.temperature = float(config.temperature)
    
    # Configure threading
    if config.kernel == "CPU" and config.n_threads is not None:
        eq_simulation.kernel_configuration.n_threads = int(config.n_threads)
    
    # Place particles randomly (same deterministic placement as place_particles)
    pos_qt, pos_ft = _random_positions(config)

    for p in pos_qt:
        eq_simulation.add_topology(config.topology.name, [config.qt.name], p.reshape(1, 3))
    
    for p in pos_ft:
        eq_simulation.add_topology(config.topology.name, [config.ft.name], p.reshape(1, 3))
    
    print(f"✓ Placed {config.n_qt} Qt + {config.n_ft} Ft particles (random)")
    
    # Run equilibration
    print(f"  Running equilibration...")
    eq_simulation.run(int(n_steps), float(config.timestep))
    
    # Extract final positions from the simulation context
    # We need to get current particle positions from the simulation
    context = eq_simulation.current_particles
    
    # Separate Qt and Ft particles by type
    qt_positions = []
    ft_positions = []
    
    for particle in context:
        particle_type = particle.type
        position = np.array(particle.pos)
        
        if particle_type == config.qt.name:
            qt_positions.append(position)
        elif particle_type == config.ft.name:
            ft_positions.append(position)
        # Ignore QtC and FtC (shouldn't exist in equilibration, but just in case)
    
    positions_qt = np.array(qt_positions)
    positions_ft = np.array(ft_positions)
    
    print(f"\n{'=' * 60}")
    print("EQUILIBRATION COMPLETE")
    print(f"  Retrieved {len(positions_qt)} Qt + {len(positions_ft)} Ft positions")
    print(f"{'=' * 60}\n")

    return positions_qt, positions_ft


def run_one(
    config: SimulationConfig,
    equilibration_steps: int = 10000,
    skip_equilibration: bool = False,
    overwrite: bool = True,
    show_progress: bool = True,
) -> readdy.Trajectory:
    """Run the full single-replica pipeline and return the trajectory.

    Pipeline: optional equilibration (WCA, no reactions) -> build system ->
    build simulation -> place particles -> production run. This is the single
    implementation shared by the ensemble runners and the run_replica.py CLI
    (previously duplicated in three places).

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration (its output_file determines where the trajectory goes).
    equilibration_steps : int
        Number of equilibration steps (ignored if skip_equilibration=True).
    skip_equilibration : bool
        If True, skip equilibration and start from random positions.
    overwrite : bool
        Overwrite an existing output file.
    show_progress : bool
        Print progress banners for the production run.

    Returns
    -------
    readdy.Trajectory
        Trajectory object for the completed production run.
    """
    # Ensure the output directory exists (create_simulation does not create it).
    output_dir = os.path.dirname(config.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if skip_equilibration:
        pos_qt, pos_ft = None, None
    else:
        pos_qt, pos_ft = equilibrate_system(config, n_steps=equilibration_steps)

    system = create_system(config)
    simulation = create_simulation(system, config, overwrite=overwrite)
    place_particles(simulation, config, positions_qt=pos_qt, positions_ft=pos_ft)
    return run_simulation(simulation, config, show_progress=show_progress)
