"""
Qt-Ft Ensemble Simulation Module

This module provides the EnsembleSimulation class for managing multiple replica
simulations with different random seeds for statistical analysis.

Contents:
    - EnsembleSimulation class (run, collect, compute, SLURM script generation)
    - _run_replica_worker() helper for multiprocessing

Related modules:
    - agglomeration_simulation.py: Configuration and simulation execution
    - agglomeration_analysis.py: Core analysis functions (no matplotlib dependency)
    - agglomeration_plotting.py: All plotting and visualization

This module has NO matplotlib dependency. For plotting ensemble results,
use agglomeration_plotting.plot_ensemble_observables() and related functions.

Usage:
    import agglomeration_simulation as sim
    import agglomeration_plotting as plotting
    from agglomeration_ensemble_simulation import EnsembleSimulation

    config = sim.SimulationConfig(...)
    ensemble = EnsembleSimulation(config, n_replicas=10)

    # Run locally
    ensemble.run_local()

    # Or generate SLURM scripts for cluster
    ensemble.generate_slurm_scripts(partition="cpu")

    # After completion, plot results
    stats, structural, config_dict = ensemble.to_plotting_format()
    plotting.plot_ensemble_observables(stats, config_dict, structural)
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import readdy

# Import from simulation module
from .config import (
    SimulationConfig,
    NS_TO_US,
    format_param_string,
)
from .engine import run_one

# Import analysis functions needed for collecting and computing results
from .analysis import (
    get_cluster_statistics,
    get_bond_counts,
    get_binding_kinetics,
    get_cluster_morphology,
    get_spatial_distribution,
    get_contact_analysis,
    get_cluster_composition,
    get_size_fractions,
    _extract_frame_data,
)

# Try to import scipy for interpolation (used in compute_statistics)
try:
    from scipy import interpolate as scipy_interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_interpolate = None




# =============================================================================
# ENSEMBLE SIMULATION CLASS
# =============================================================================

def _run_replica_worker(output_dir: str, config_path: str, equilibration_steps: int, replica_idx: int) -> Dict:
    """
    Standalone worker function for running a replica in parallel.
    
    This function is defined outside the class to avoid pickling issues
    with multiprocessing.
    
    Parameters
    ----------
    output_dir : str
        Directory for this replica's output
    config_path : str
        Path to the config JSON file
    equilibration_steps : int
        Number of equilibration steps
    replica_idx : int
        Index of this replica
    
    Returns
    -------
    dict
        Result with 'idx', 'success', and 'error' keys
    """
    try:
        # Load config from file
        config = SimulationConfig.load_json(config_path)

        # Run the full single-replica pipeline (equilibrate -> build -> place -> run)
        run_one(config, equilibration_steps=equilibration_steps)

        return {'idx': replica_idx, 'success': True, 'error': None}
    except Exception as e:
        return {'idx': replica_idx, 'success': False, 'error': str(e)}


def _compute_replica_structural(h5_file: str, config: SimulationConfig, stride: int) -> Dict:
    """Compute morphology/spatial/contacts/composition for one replica.

    Shared by the sequential and parallel structural-analysis paths so both produce
    identical per-replica results. Extracts frame data once and reuses it.

    Returns
    -------
    dict with keys 'morphology', 'spatial', 'contacts', 'composition' (each a result
    dict or None) and 'errors' (list of strings).
    """
    result = {'morphology': None, 'spatial': None, 'contacts': None,
              'composition': None, 'errors': []}
    try:
        frame_data = _extract_frame_data(h5_file, config, stride=stride, verbose=False)
    except Exception as e:
        result['errors'].append(f"frame_data: {e}")
        return result

    for key, fn in (
        ('morphology', get_cluster_morphology),
        ('spatial', get_spatial_distribution),
        ('contacts', get_contact_analysis),
        ('composition', get_cluster_composition),
    ):
        try:
            result[key] = fn(h5_file, config, stride=stride, frame_data=frame_data)
        except Exception as e:
            result['errors'].append(f"{key}: {e}")
    return result


def _analyze_replica_structural_worker(args) -> Dict:
    """Picklable wrapper around _compute_replica_structural for multiprocessing.

    Parameters
    ----------
    args : tuple
        (replica_idx, h5_file, config_dict, stride)
    """
    replica_idx, h5_file, config_dict, stride = args
    config = SimulationConfig.from_dict(config_dict)
    result = _compute_replica_structural(h5_file, config, stride)
    result['replica_idx'] = replica_idx
    return result



class EnsembleSimulation:
    """
    Manage multiple replica simulations for statistical analysis.
    
    This class handles:
    - Running N replicas with different random seeds
    - Collecting results from all replicas
    - Computing mean ± std statistics across replicas
    - Plotting with error bands
    - Generating SLURM scripts for cluster execution
    - Auto-saving analysis results to JSON/NPZ for later plotting
    
    Parameters
    ----------
    base_config : SimulationConfig
        Template configuration (seed and output_file will be modified per replica)
    n_replicas : int
        Number of replica simulations to run
    seeds : list of int, optional
        Random seeds for each replica. If None, auto-generated from base_config.rng_seed
    name : str, optional
        Descriptive name for this ensemble (e.g., "high_concentration", "test_run").
        If None, only auto-generated parameters are used for the folder name.
    base_dir : str
        Base directory for ensemble outputs. The actual output folder will be created
        as a subdirectory with an auto-generated name based on simulation parameters.
        Default: "." (current directory)
    
    Folder Naming
    -------------
    The output folder is automatically named based on simulation parameters:
        {name}_{n_qt}Qt_{n_ft}Ft_dt{timestep}ps_{total_time}us/
    
    Example folder names:
        - 200Qt_400Ft_dt10ps_30us/  (no name provided)
        - highFt_200Qt_800Ft_dt10ps_30us/  (name="highFt")
    
    Example
    -------
    >>> # Create ensemble with auto-generated folder name
    >>> ensemble = EnsembleSimulation(
    ...     base_config=config,
    ...     n_replicas=10,
    ...     name="experiment1",
    ...     base_dir="~/Readdy_Simulations"
    ... )
    >>> # Creates folder: ~/Readdy_Simulations/experiment1_200Qt_400Ft_dt10ps_30us/
    >>> 
    >>> # Run locally (auto-saves analysis results)
    >>> ensemble.run_local(parallel=True, n_workers=4)
    >>> 
    >>> # Or generate SLURM scripts for cluster
    >>> ensemble.generate_slurm_scripts(partition="cpu")
    >>> 
    >>> # After runs complete, analyze
    >>> ensemble.collect_results()
    >>> ensemble.compute_statistics()
    >>> ensemble.plot_observables(show_individual=True)
    >>> 
    >>> # Structural analysis is computed automatically by run_local()
    >>> # To recompute manually with different stride:
    >>> ensemble.compute_structural_statistics(stride=10)
    >>> ensemble.plot_structural(show_individual=True)
    >>> 
    >>> ensemble.print_summary()
    """
    
    def __init__(
        self,
        base_config: SimulationConfig,
        n_replicas: int = 10,
        seeds: Optional[List[int]] = None,
        name: Optional[str] = None,
        base_dir: str = ".",
    ):
        self.base_config = base_config
        self.n_replicas = n_replicas
        self.name = name
        self.base_dir = base_dir.rstrip("/")
        
        # Generate folder name from simulation parameters
        self.folder_name = self._generate_folder_name()
        self.output_dir = f"{self.base_dir}/{self.folder_name}/"
        
        # Auto-generate seeds if not provided
        # Uses SeedSequence.spawn() to guarantee statistically independent,
        # collision-free seeds — unlike the fragile base_seed + i*1000 pattern.
        if seeds is None:
            ss = np.random.SeedSequence(base_config.rng_seed)
            child_sequences = ss.spawn(n_replicas)
            self.seeds = [int(child.generate_state(1)[0]) for child in child_sequences]
        else:
            if len(seeds) != n_replicas:
                raise ValueError(f"Number of seeds ({len(seeds)}) must match n_replicas ({n_replicas})")
            self.seeds = seeds
        
        # Generate per-replica configs
        self.replica_configs = []
        for i, seed in enumerate(self.seeds):
            replica_config = self._create_replica_config(i, seed)
            self.replica_configs.append(replica_config)
        
        # Storage for results (populated by collect_results)
        self.results_collected = False
        self.replica_data = {}
        self.statistics = {}
        self.structural_statistics = {}  # For morphology, spatial, contacts, composition
        self.summary_metrics = {}
        
        print(f"✓ Ensemble created: {self.folder_name}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Replicas: {self.n_replicas}")
    
    def _generate_folder_name(self) -> str:
        """Generate folder name from simulation parameters.

        Uses the shared `format_param_string()` convention (identical to the single-run
        trajectory filename, minus the .h5 suffix), optionally prefixed with `self.name`.

        Examples:
            200Qt_400Ft_WCA_eQQ10_eFF10_eQF10_kon10_dt10ps_30us
            highFt_200Qt_800Ft_LJ_eQQ10_eFF10_eQF5_kon5.5_dt10ps_30us
        """
        params_part = format_param_string(self.base_config)
        if self.name:
            return f"{self.name}_{params_part}"
        return params_part
    
    def _create_replica_config(self, replica_idx: int, seed: int) -> SimulationConfig:
        """Create a config for a single replica with modified seed and output path."""
        # Create config dict from base config
        config_dict = self.base_config.to_dict()
        
        # Modify seed and output file
        config_dict['rng_seed'] = seed
        replica_dir = f"{self.output_dir}replica_{replica_idx:03d}/"
        config_dict['output_file'] = f"{replica_dir}trajectory.h5"
        
        return SimulationConfig.from_dict(config_dict)
    
    def _ensure_directories(self, overwrite: bool = False):
        """Create necessary directories."""
        # Check if output directory exists
        if os.path.exists(self.output_dir):
            if not overwrite:
                raise FileExistsError(
                    f"Output directory '{self.output_dir}' already exists. "
                    "Use overwrite=True to overwrite existing data."
                )
            else:
                print(f"Warning: Overwriting existing directory '{self.output_dir}'")
        
        # Create directories
        os.makedirs(f"{self.output_dir}configs/", exist_ok=True)
        os.makedirs(f"{self.output_dir}logs/", exist_ok=True)
        for i in range(self.n_replicas):
            os.makedirs(f"{self.output_dir}replica_{i:03d}/", exist_ok=True)
    
    def _save_replica_configs(self):
        """Save all replica configs as JSON files."""
        for i, config in enumerate(self.replica_configs):
            config_path = f"{self.output_dir}configs/config_{i:03d}.json"
            config.save_json(config_path)
    
    def run_local(
        self,
        parallel: bool = False,
        n_workers: Optional[int] = None,
        overwrite: bool = False,
        equilibration_steps: int = 10000,
        stride: int = 10,
    ):
        """
        Run all replicas locally.
        
        After completion, automatically collects results, computes statistics
        (including structural analysis), and saves analysis files (JSON/NPZ)
        for later plotting.
        
        Parameters
        ----------
        parallel : bool
            If True, run replicas in parallel using multiprocessing
        n_workers : int, optional
            Number of parallel workers. If None, uses CPU count - 1
        overwrite : bool
            If True, overwrite existing output directory
        equilibration_steps : int
            Number of equilibration steps per replica
        stride : int
            Stride for structural analysis (analyze every Nth frame).
            Default: 10
        """
        self._ensure_directories(overwrite=overwrite)
        self._save_replica_configs()
        
        print("\n" + "=" * 60)
        print(f"ENSEMBLE SIMULATION: {self.n_replicas} REPLICAS")
        print("=" * 60)
        
        if parallel:
            self._run_parallel(n_workers, equilibration_steps)
        else:
            self._run_sequential(equilibration_steps)
        
        print("\n" + "=" * 60)
        print("ALL REPLICAS COMPLETED")
        print("=" * 60 + "\n")
        
        # Auto-collect results and compute statistics
        print("=" * 60)
        print("POST-PROCESSING")
        print("=" * 60)
        
        self.collect_results(require_all=False)
        self.compute_statistics()
        self.compute_summary_metrics()
        
        # Compute structural statistics
        print("\nComputing structural statistics...")
        self.compute_structural_statistics(stride=stride)
        
        # Auto-save analysis files
        self.save_for_plotting()
        
        print("\n" + "=" * 60)
        print("ENSEMBLE COMPLETE")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Analysis files saved for plotting")
        print("=" * 60 + "\n")
    
    def _run_sequential(self, equilibration_steps: int):
        """Run replicas sequentially."""
        for i in range(self.n_replicas):
            print(f"\n{'─' * 60}")
            print(f"REPLICA {i + 1}/{self.n_replicas} (seed={self.seeds[i]})")
            print(f"{'─' * 60}")
            self._run_single_replica(i, equilibration_steps)
    
    def _run_parallel(self, n_workers: Optional[int], equilibration_steps: int):
        """Run replicas in parallel using multiprocessing."""
        from multiprocessing import Pool, cpu_count
        
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)
        
        n_workers = min(n_workers, self.n_replicas)
        
        print(f"\nRunning {self.n_replicas} replicas on {n_workers} workers...")
        
        # Create arguments for each replica - pass config paths instead of objects
        # This avoids pickling issues with instance methods
        args = [
            (self.replica_configs[i].output_file.replace('trajectory.h5', ''),
             f"{self.output_dir}configs/config_{i:03d}.json",
             equilibration_steps,
             i)
            for i in range(self.n_replicas)
        ]
        
        completed = 0
        with Pool(n_workers) as pool:
            for result in pool.starmap(_run_replica_worker, args):
                completed += 1
                if result['success']:
                    print(f"  Completed: {completed}/{self.n_replicas} (replica {result['idx']})")
                else:
                    print(f"  FAILED: {completed}/{self.n_replicas} (replica {result['idx']}): {result['error']}")
    
    def _run_single_replica(self, replica_idx: int, equilibration_steps: int):
        """Run a single replica simulation."""
        config = self.replica_configs[replica_idx]
        run_one(config, equilibration_steps=equilibration_steps)
    
    def generate_slurm_scripts(
        self,
        partition: str,
        time: str = "08:00:00",
        cpus_per_task: int = 12,
        memory: str = "32G",
        conda_base: str = "/dss/dsshome1/03/ge35wef2/miniconda3",
        conda_env: str = "readdy",
        scripts_dir: str = "~/Readdy_Simulations",
        cluster: Optional[str] = None,
        qos: Optional[str] = None,
        mail_user: Optional[str] = None,
        mail_type: str = "ALL",
    ):
        """
        Generate SLURM job array script for cluster execution.
        
        Parameters
        ----------
        partition : str
            SLURM partition name (required, e.g., "lrz-cpu", "serial")
        time : str
            Wall time limit (HH:MM:SS)
        cpus_per_task : int
            Number of CPUs per replica
        memory : str
            Memory per replica
        conda_base : str
            Full path to miniconda/anaconda installation directory
        conda_env : str
            Name of conda environment with ReaDDy
        scripts_dir : str
            Directory on cluster where Python scripts are located
            (agglomeration_simulation.py, run_replica.py, etc.)
        cluster : str, optional
            SLURM cluster name. If None, --clusters line is omitted.
        qos : str, optional
            Quality of service. If None, --qos line is omitted.
        mail_user : str, optional
            Email address for job notifications. If None, no email notifications.
        mail_type : str
            When to send email notifications (default: "ALL").
            Options: NONE, BEGIN, END, FAIL, REQUEUE, ALL
        """
        # Create directories and save configs
        os.makedirs(f"{self.output_dir}configs/", exist_ok=True)
        os.makedirs(f"{self.output_dir}logs/", exist_ok=True)
        self._save_replica_configs()
        
        # Build optional SLURM lines
        cluster_line = f"#SBATCH --clusters={cluster}\n" if cluster else ""
        qos_line = f"#SBATCH --qos={qos}\n" if qos else ""
        
        # Build email lines if mail_user is provided
        if mail_user:
            mail_lines = f"#SBATCH --mail-type={mail_type}\n#SBATCH --mail-user={mail_user}\n"
        else:
            mail_lines = ""
        
        # Build conda activation command (robust method)
        conda_activate = f"source {conda_base}/bin/activate {conda_base}/envs/{conda_env}"
        
        # Construct the ensemble output path relative to scripts_dir
        # e.g., scripts_dir = ~/Readdy_Simulations, folder_name = 200Qt_400Ft_dt10ps_30us
        ensemble_path = f"{scripts_dir}/{self.folder_name}"
        
        # Generate SLURM script
        slurm_script = f'''#!/bin/bash
#SBATCH --job-name=agglo_{self.folder_name[:20]}
#SBATCH -p {partition}
{cluster_line}{qos_line}#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --array=0-{self.n_replicas - 1}
#SBATCH --output={ensemble_path}/logs/replica_%a.out
#SBATCH --error={ensemble_path}/logs/replica_%a.err
#SBATCH --get-user-env
{mail_lines}
# ============================================================
# ENSEMBLE SIMULATION - REPLICA $SLURM_ARRAY_TASK_ID
# Ensemble: {self.folder_name}
# ============================================================

# Activate conda environment
{conda_activate}

# Navigate to scripts directory (where Python files are)
cd {scripts_dir}

# Format replica index with leading zeros
REPLICA_IDX=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

# Run the replica
echo "Starting replica $SLURM_ARRAY_TASK_ID (config: config_${{REPLICA_IDX}}.json)"
echo "Ensemble: {self.folder_name}"
echo "Time: $(date)"
echo ""

python run_replica.py --config {ensemble_path}/configs/config_${{REPLICA_IDX}}.json

echo ""
echo "Replica $SLURM_ARRAY_TASK_ID completed at $(date)"
'''
        
        slurm_path = f"{self.output_dir}submit_ensemble.slurm"
        with open(slurm_path, 'w') as f:
            f.write(slurm_script)
        
        print(f"✓ Generated SLURM script: {slurm_path}")
        print(f"✓ Saved {self.n_replicas} config files to {self.output_dir}configs/")
        print(f"\nFolder structure on cluster:")
        print(f"  {scripts_dir}/")
        print(f"  ├── agglomeration_simulation.py")
        print(f"  ├── agglomeration_analysis.py")
        print(f"  ├── run_replica.py")
        print(f"  ├── analyze_ensemble.py")
        print(f"  └── {self.folder_name}/")
        print(f"      ├── configs/")
        print(f"      ├── logs/")
        print(f"      ├── replica_000/")
        print(f"      └── ...")
        print(f"\nTo run on cluster:")
        print(f"  1. Upload scripts (once):     scp agglomeration_simulation.py agglomeration_analysis.py run_replica.py analyze_ensemble.py user@cluster:{scripts_dir}/")
        print(f"  2. Upload ensemble folder:    scp -r {self.output_dir} user@cluster:{scripts_dir}/")
        print(f"  3. Submit job:                sbatch {ensemble_path}/submit_ensemble.slurm")
    
    def generate_analysis_slurm_script(
        self,
        partition: str,
        time: str = "04:00:00",
        cpus_per_task: int = 4,
        memory: str = "32G",
        conda_base: str = "/dss/dsshome1/03/ge35wef2/miniconda3",
        conda_env: str = "readdy",
        scripts_dir: str = "~/Readdy_Simulations",
        stride: int = 10,
        cluster: Optional[str] = None,
        qos: Optional[str] = None,
        mail_user: Optional[str] = None,
        mail_type: str = "ALL",
    ):
        """
        Generate SLURM script for running ensemble analysis on cluster.
        
        This runs after all replicas complete and produces JSON/NPZ files
        that can be downloaded for local plotting.
        
        Parameters
        ----------
        partition : str
            SLURM partition name (required, e.g., "lrz-cpu", "serial")
        time : str
            Wall time limit (HH:MM:SS)
        cpus_per_task : int
            Number of CPUs
        memory : str
            Memory allocation
        conda_base : str
            Full path to miniconda/anaconda installation directory
        conda_env : str
            Name of conda environment with ReaDDy
        scripts_dir : str
            Directory on cluster where Python scripts are located
            (agglomeration_simulation.py, analyze_ensemble.py, etc.)
        stride : int
            Stride for structural analysis (analyze every Nth frame)
        cluster : str, optional
            SLURM cluster name. If None, --clusters line is omitted.
        qos : str, optional
            Quality of service. If None, --qos line is omitted.
        mail_user : str, optional
            Email address for job notifications. If None, no email notifications.
        mail_type : str
            When to send email notifications (default: "ALL").
            Options: NONE, BEGIN, END, FAIL, REQUEUE, ALL
        """
        os.makedirs(f"{self.output_dir}logs/", exist_ok=True)
        
        # Determine if parallel processing should be used
        parallel_flag = "--parallel" if cpus_per_task > 1 else ""
        n_workers_flag = f"--n-workers {cpus_per_task}" if cpus_per_task > 1 else ""
        
        # Build optional SLURM lines
        cluster_line = f"#SBATCH --clusters={cluster}\n" if cluster else ""
        qos_line = f"#SBATCH --qos={qos}\n" if qos else ""
        
        # Build email lines if mail_user is provided
        if mail_user:
            mail_lines = f"#SBATCH --mail-type={mail_type}\n#SBATCH --mail-user={mail_user}\n"
        else:
            mail_lines = ""
        
        # Build conda activation command (robust method)
        conda_activate = f"source {conda_base}/bin/activate {conda_base}/envs/{conda_env}"
        
        # Construct the ensemble output path relative to scripts_dir
        ensemble_path = f"{scripts_dir}/{self.folder_name}"
        
        # Generate SLURM script
        slurm_script = f'''#!/bin/bash
#SBATCH --job-name=analysis_{self.folder_name[:15]}
#SBATCH -p {partition}
{cluster_line}{qos_line}#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --output={ensemble_path}/logs/analysis.out
#SBATCH --error={ensemble_path}/logs/analysis.err
#SBATCH --get-user-env
{mail_lines}
# ============================================================
# ENSEMBLE ANALYSIS
# Ensemble: {self.folder_name}
# ============================================================

# Activate conda environment
{conda_activate}

# Navigate to scripts directory (where Python files are)
cd {scripts_dir}

echo "Starting ensemble analysis"
echo "Ensemble: {self.folder_name}"
echo "Time: $(date)"
echo "CPUs available: {cpus_per_task}"
echo ""

python analyze_ensemble.py --ensemble-dir {ensemble_path}/ --stride {stride} {parallel_flag} {n_workers_flag}

echo ""
echo "Analysis completed at $(date)"
'''
        
        slurm_path = f"{self.output_dir}submit_analysis.slurm"
        with open(slurm_path, 'w') as f:
            f.write(slurm_script)
        
        print(f"✓ Generated analysis SLURM script: {slurm_path}")
        print(f"\nTo run analysis on cluster (after simulations complete):")
        print(f"  sbatch {ensemble_path}/submit_analysis.slurm")
        print(f"\nOutput files will be saved to {ensemble_path}/:")
        print(f"  - ensemble_statistics.json (basic statistics)")
        print(f"  - ensemble_structural.npz (structural analysis arrays)")
        print(f"  - ensemble_config.json (configuration reference)")
    
    def collect_results(self, require_all: bool = False):
        """
        Collect results from all replica trajectory files.
        
        Parameters
        ----------
        require_all : bool
            If True, raise error if any replica is missing.
            If False, warn and continue with available data.
        """
        print("\n" + "=" * 60)
        print("COLLECTING ENSEMBLE RESULTS")
        print("=" * 60)
        
        self.replica_data = {
            'bonds': [],
            'energy': [],
            'pressure': [],
            'cluster_stats': [],
            'kinetics': [],
            'particle_counts': [],
            'reaction_counts': [],  # NEW: cumulative reactions
            'times': [],
            'available_replicas': [],
        }
        
        missing_replicas = []
        
        for i, config in enumerate(self.replica_configs):
            h5_file = config.output_file
            
            if not os.path.exists(h5_file):
                missing_replicas.append(i)
                print(f"  Replica {i}: MISSING ({h5_file})")
                continue
            
            print(f"  Replica {i}: Loading...")
            
            try:
                # Open the trajectory once and reuse the handle for every observable
                # read below, instead of reopening the HDF5 file in each get_* call.
                traj = readdy.Trajectory(h5_file)

                # Load basic data (silent=True to avoid per-replica bond method output)
                bonds_data = get_bond_counts(h5_file, trajectory=traj, silent=True)
                self.replica_data['bonds'].append(bonds_data)
                self.replica_data['times'].append(bonds_data['times'])

                # Energy
                try:
                    times_e, energy = traj.read_observable_energy()
                    self.replica_data['energy'].append({
                        'times': np.array(times_e),
                        'energy': np.array(energy)
                    })
                except (KeyError, ValueError, IndexError):
                    self.replica_data['energy'].append(None)
                
                # Pressure
                try:
                    times_p, pressure = traj.read_observable_pressure()
                    self.replica_data['pressure'].append({
                        'times': np.array(times_p),
                        'pressure': np.array(pressure)
                    })
                except (KeyError, ValueError, IndexError):
                    self.replica_data['pressure'].append(None)
                
                # Particle counts
                try:
                    times_pc, counts = traj.read_observable_number_of_particles()
                    self.replica_data['particle_counts'].append({
                        'times': np.array(times_pc),
                        'counts': np.array(counts),  # Shape: (n_frames, n_types)
                    })
                except (KeyError, ValueError, IndexError):
                    self.replica_data['particle_counts'].append(None)
                
                # Cluster statistics
                try:
                    cluster_stats = get_cluster_statistics(h5_file, trajectory=traj)
                    self.replica_data['cluster_stats'].append(cluster_stats)
                except (KeyError, ValueError, IndexError):
                    self.replica_data['cluster_stats'].append(None)
                
                # Binding kinetics
                try:
                    kinetics = get_binding_kinetics(h5_file, config, trajectory=traj)
                    self.replica_data['kinetics'].append(kinetics)
                except (KeyError, ValueError, IndexError):
                    self.replica_data['kinetics'].append(None)
                
                # Reaction counts (cumulative)
                try:
                    times_r, counts_dict = traj.read_observable_reaction_counts()
                    times_r = np.array(times_r)
                    # Sum all reaction types to get total cumulative
                    total_cumulative = np.zeros(len(times_r))
                    
                    def extract_series(obj):
                        if isinstance(obj, dict):
                            for v in obj.values():
                                yield from extract_series(v)
                        else:
                            yield np.asarray(obj).flatten()
                    
                    for series in extract_series(counts_dict):
                        if len(series) == len(times_r):
                            total_cumulative += np.cumsum(series)
                    
                    self.replica_data['reaction_counts'].append({
                        'times': times_r,
                        'cumulative': total_cumulative
                    })
                except (KeyError, ValueError, IndexError):
                    self.replica_data['reaction_counts'].append(None)
                
                self.replica_data['available_replicas'].append(i)
                
            except Exception as e:
                print(f"    Error loading replica {i}: {e}")
                missing_replicas.append(i)
        
        n_available = len(self.replica_data['available_replicas'])
        
        if missing_replicas:
            msg = f"Missing {len(missing_replicas)}/{self.n_replicas} replicas: {missing_replicas}"
            if require_all:
                raise RuntimeError(msg)
            else:
                print(f"\nWarning: {msg}")
                print(f"Proceeding with {n_available} available replicas.")
        
        self.results_collected = True
        print(f"\n✓ Collected data from {n_available}/{self.n_replicas} replicas")
    
    def compute_statistics(self):
        """
        Compute mean and standard deviation across all replicas.
        
        Must call collect_results() first.
        """
        if not self.results_collected:
            raise RuntimeError("Must call collect_results() before compute_statistics()")
        
        print("\nComputing ensemble statistics...")
        
        n_replicas = len(self.replica_data['available_replicas'])
        if n_replicas == 0:
            raise RuntimeError("No replica data available")
        
        # Determine common time grid
        all_times = [d['times'] for d in self.replica_data['bonds']]
        
        # Check if all times are identical
        reference_times = all_times[0]
        times_identical = all(
            len(t) == len(reference_times) and np.allclose(t, reference_times) 
            for t in all_times
        )
        
        if times_identical:
            common_times = reference_times
            print(f"  Time grids identical across replicas ({len(common_times)} points)")
        else:
            # Interpolate to common grid
            if not SCIPY_AVAILABLE:
                raise RuntimeError(
                    "scipy is required for interpolation when time grids differ. "
                    "Install with: pip install scipy"
                )
            t_min = max(t[0] for t in all_times)
            t_max = min(t[-1] for t in all_times)
            n_points = min(len(t) for t in all_times)
            common_times = np.linspace(t_min, t_max, n_points)
            print(f"  Interpolating to common time grid ({n_points} points)")
        
        self.statistics['times'] = common_times
        self.statistics['n_replicas'] = n_replicas
        
        # Helper function for interpolation
        def interpolate_to_common(times, values, common_times):
            if times_identical:
                return values
            f = scipy_interpolate.interp1d(times, values, kind='linear', 
                                           bounds_error=False, fill_value='extrapolate')
            return f(common_times)
        
        # Compute statistics for bonds
        bonds_matrix = np.array([
            interpolate_to_common(d['times'], d['n_bonds'], common_times)
            for d in self.replica_data['bonds']
        ])
        self.statistics['bonds_mean'] = np.mean(bonds_matrix, axis=0)
        self.statistics['bonds_std'] = np.std(bonds_matrix, axis=0)
        self.statistics['bonds_all'] = bonds_matrix
        
        # Compute statistics for energy
        valid_energy = [d for d in self.replica_data['energy'] if d is not None]
        if valid_energy:
            energy_matrix = np.array([
                interpolate_to_common(d['times'], d['energy'], common_times)
                for d in valid_energy
            ])
            self.statistics['energy_mean'] = np.mean(energy_matrix, axis=0)
            self.statistics['energy_std'] = np.std(energy_matrix, axis=0)
            self.statistics['energy_all'] = energy_matrix
        
        # Compute statistics for pressure
        valid_pressure = [d for d in self.replica_data['pressure'] if d is not None]
        if valid_pressure:
            pressure_matrix = np.array([
                interpolate_to_common(d['times'], d['pressure'], common_times)
                for d in valid_pressure
            ])
            self.statistics['pressure_mean'] = np.mean(pressure_matrix, axis=0)
            self.statistics['pressure_std'] = np.std(pressure_matrix, axis=0)
            self.statistics['pressure_all'] = pressure_matrix
        
        # Compute statistics for particle counts
        valid_counts = [d for d in self.replica_data['particle_counts'] if d is not None]
        if valid_counts:
            # Particle types: Qt, Ft, QtC, FtC (indices 0, 1, 2, 3)
            for idx, name in enumerate(['qt', 'ft', 'qtc', 'ftc']):
                count_matrix = np.array([
                    interpolate_to_common(d['times'], d['counts'][:, idx], common_times)
                    for d in valid_counts
                ])
                self.statistics[f'{name}_count_mean'] = np.mean(count_matrix, axis=0)
                self.statistics[f'{name}_count_std'] = np.std(count_matrix, axis=0)
                self.statistics[f'{name}_count_all'] = count_matrix
            
            # Also compute total
            total_matrix = np.array([
                interpolate_to_common(d['times'], d['counts'].sum(axis=1), common_times)
                for d in valid_counts
            ])
            self.statistics['total_count_mean'] = np.mean(total_matrix, axis=0)
            self.statistics['total_count_std'] = np.std(total_matrix, axis=0)
            self.statistics['total_count_all'] = total_matrix
        
        # Compute statistics for cluster counts
        valid_cluster = [d for d in self.replica_data['cluster_stats'] if d is not None]
        if valid_cluster:
            # Number of clusters
            n_clusters_matrix = np.array([
                interpolate_to_common(d['times'], d['n_clusters'], common_times)
                for d in valid_cluster
            ])
            self.statistics['n_clusters_mean'] = np.mean(n_clusters_matrix, axis=0)
            self.statistics['n_clusters_std'] = np.std(n_clusters_matrix, axis=0)
            self.statistics['n_clusters_all'] = n_clusters_matrix
            
            # Largest cluster (note: get_cluster_statistics returns 'max_sizes')
            largest_matrix = np.array([
                interpolate_to_common(d['times'], d['max_sizes'], common_times)
                for d in valid_cluster
            ])
            self.statistics['largest_cluster_mean'] = np.mean(largest_matrix, axis=0)
            self.statistics['largest_cluster_std'] = np.std(largest_matrix, axis=0)
            self.statistics['largest_cluster_all'] = largest_matrix
            
            # Average cluster size (NEW)
            avg_matrix = np.array([
                interpolate_to_common(d['times'], d['avg_sizes'], common_times)
                for d in valid_cluster
            ])
            self.statistics['avg_cluster_mean'] = np.mean(avg_matrix, axis=0)
            self.statistics['avg_cluster_std'] = np.std(avg_matrix, axis=0)
            self.statistics['avg_cluster_all'] = avg_matrix
        
        # Compute statistics for cumulative reactions (NEW)
        valid_reactions = [d for d in self.replica_data['reaction_counts'] if d is not None]
        if valid_reactions:
            reactions_matrix = np.array([
                interpolate_to_common(d['times'], d['cumulative'], common_times)
                for d in valid_reactions
            ])
            self.statistics['cumulative_reactions_mean'] = np.mean(reactions_matrix, axis=0)
            self.statistics['cumulative_reactions_std'] = np.std(reactions_matrix, axis=0)
            self.statistics['cumulative_reactions_all'] = reactions_matrix
        
        # Compute statistics for kinetics (fraction bound)
        # Note: get_binding_kinetics returns 'fraction_bound_qt' and 'fraction_bound_ft'
        # We compute an overall fraction bound as the average
        valid_kinetics = [d for d in self.replica_data['kinetics'] if d is not None]
        if valid_kinetics:
            # Use average of Qt and Ft fraction bound
            fraction_matrix = np.array([
                interpolate_to_common(
                    d['times'], 
                    (d['fraction_bound_qt'] + d['fraction_bound_ft']) / 2, 
                    common_times
                )
                for d in valid_kinetics
            ])
            self.statistics['fraction_bound_mean'] = np.mean(fraction_matrix, axis=0)
            self.statistics['fraction_bound_std'] = np.std(fraction_matrix, axis=0)
            self.statistics['fraction_bound_all'] = fraction_matrix
        
        print("✓ Statistics computed")
    
    def compute_summary_metrics(self, percolation_threshold: float = 0.5):
        """
        Compute summary metrics across the ensemble.
        
        Parameters
        ----------
        percolation_threshold : float
            Fraction of particles for percolation (default: 0.5)
        
        Returns
        -------
        dict with keys:
            n_replicas : int
            percolation_threshold : float
            final_bonds_mean/std : float
            final_clusters_mean/std : float
            final_largest_mean/std : float
            final_largest_fraction_mean/std : float
            final_fraction_bound_mean/std : float
            half_time_mean/std : float (in nanoseconds)
            percolation_count : int
            percolation_fraction : float
            percolation_time_mean/std : float (in nanoseconds)
        """
        if not self.statistics:
            raise RuntimeError("Must call compute_statistics() before compute_summary_metrics()")
        
        total_particles = self.base_config.n_qt + self.base_config.n_ft
        percolation_size = percolation_threshold * total_particles
        
        metrics = {
            'n_replicas': self.statistics['n_replicas'],
            'percolation_threshold': percolation_threshold,
        }
        
        # Final bond count
        if 'bonds_all' in self.statistics:
            final_bonds = self.statistics['bonds_all'][:, -1]
            metrics['final_bonds_mean'] = np.mean(final_bonds)
            metrics['final_bonds_std'] = np.std(final_bonds)
        
        # Final cluster count
        if 'n_clusters_all' in self.statistics:
            final_clusters = self.statistics['n_clusters_all'][:, -1]
            metrics['final_clusters_mean'] = np.mean(final_clusters)
            metrics['final_clusters_std'] = np.std(final_clusters)
        
        # Largest cluster
        if 'largest_cluster_all' in self.statistics:
            final_largest = self.statistics['largest_cluster_all'][:, -1]
            metrics['final_largest_mean'] = np.mean(final_largest)
            metrics['final_largest_std'] = np.std(final_largest)
            metrics['final_largest_fraction_mean'] = np.mean(final_largest) / total_particles
            metrics['final_largest_fraction_std'] = np.std(final_largest) / total_particles
        
        # Fraction bound
        if 'fraction_bound_all' in self.statistics:
            final_fraction = self.statistics['fraction_bound_all'][:, -1]
            metrics['final_fraction_bound_mean'] = np.mean(final_fraction)
            metrics['final_fraction_bound_std'] = np.std(final_fraction)
        
        # Half-time (time to 50% of final bonds) - stored in nanoseconds
        if 'bonds_all' in self.statistics:
            half_times_steps = []
            times = self.statistics['times']
            for bonds in self.statistics['bonds_all']:
                half_target = 0.5 * bonds[-1]
                # First step at/above half the final bond count. First-crossing instead of
                # searchsorted, since bonds(t) is noisy and not guaranteed monotonic.
                crossings = np.flatnonzero(bonds >= half_target)
                if len(crossings) > 0 and crossings[0] < len(times):
                    half_times_steps.append(times[crossings[0]])
            if half_times_steps:
                # Convert step numbers to nanoseconds
                timestep = self.base_config.timestep
                half_times_ns = np.array(half_times_steps) * timestep
                metrics['half_time_mean'] = np.mean(half_times_ns)  # in ns
                metrics['half_time_std'] = np.std(half_times_ns)    # in ns
                metrics['half_time_all'] = half_times_ns.tolist()   # per-replica values in ns
        
        # Percolation analysis - times stored in nanoseconds
        if 'largest_cluster_all' in self.statistics:
            times = self.statistics['times']
            percolation_times_steps = []
            percolated_count = 0
            
            for largest in self.statistics['largest_cluster_all']:
                # Find first time largest cluster exceeds threshold
                percolated_idx = np.where(largest >= percolation_size)[0]
                if len(percolated_idx) > 0:
                    percolated_count += 1
                    percolation_times_steps.append(times[percolated_idx[0]])
            
            metrics['percolation_count'] = percolated_count
            metrics['percolation_fraction'] = percolated_count / self.statistics['n_replicas']
            if percolation_times_steps:
                # Convert step numbers to nanoseconds
                timestep = self.base_config.timestep
                percolation_times_ns = np.array(percolation_times_steps) * timestep
                metrics['percolation_time_mean'] = np.mean(percolation_times_ns)  # in ns
                metrics['percolation_time_std'] = np.std(percolation_times_ns)    # in ns
        
        self.summary_metrics = metrics
        return metrics
    
    def print_summary(self):
        """Print summary statistics for the ensemble."""
        if not self.statistics:
            raise RuntimeError("Must call compute_statistics() before print_summary()")
        if not self.summary_metrics:
            self.compute_summary_metrics()
        
        m = self.summary_metrics
        
        print("\n" + "=" * 60)
        print(f"ENSEMBLE SUMMARY (N={m['n_replicas']} replicas)")
        print("=" * 60)
        
        print("\nTime Series Metrics (final values):")
        if 'final_bonds_mean' in m:
            print(f"  Bonds:              {m['final_bonds_mean']:.1f} ± {m['final_bonds_std']:.1f}")
        if 'final_clusters_mean' in m:
            print(f"  Clusters:           {m['final_clusters_mean']:.1f} ± {m['final_clusters_std']:.1f}")
        if 'final_largest_mean' in m:
            print(f"  Largest Cluster:    {m['final_largest_mean']:.1f} ± {m['final_largest_std']:.1f} "
                  f"({m['final_largest_fraction_mean']*100:.1f} ± {m['final_largest_fraction_std']*100:.1f}% of particles)")
        if 'final_fraction_bound_mean' in m:
            print(f"  Fraction Bound:     {m['final_fraction_bound_mean']:.3f} ± {m['final_fraction_bound_std']:.3f}")
        
        print("\nKinetic Metrics:")
        if 'half_time_mean' in m:
            # half_time is stored in nanoseconds, convert to microseconds
            ht_mean_us = m['half_time_mean'] * NS_TO_US
            ht_std_us = m['half_time_std'] * NS_TO_US
            print(f"  Half-time (t₅₀):    {ht_mean_us:.2f} ± {ht_std_us:.2f} µs")
        
        print(f"\nPercolation (threshold = {m['percolation_threshold']*100:.0f}%):")
        if 'percolation_count' in m:
            print(f"  Replicas percolated: {m['percolation_count']}/{m['n_replicas']} "
                  f"({m['percolation_fraction']*100:.0f}%)")
            if 'percolation_time_mean' in m:
                # percolation_time is stored in nanoseconds, convert to microseconds
                pt_mean_us = m['percolation_time_mean'] * NS_TO_US
                pt_std_us = m['percolation_time_std'] * NS_TO_US
                print(f"  Mean percolation time: {pt_mean_us:.2f} ± {pt_std_us:.2f} µs")
        
        print("\n" + "=" * 60)
    
    def to_plotting_format(self) -> Tuple[Dict, Dict, Dict]:
        """
        Convert internal statistics to the format expected by plotting functions.
        
        This allows the same plotting functions to work with both local
        (EnsembleSimulation) and cluster (JSON/NPZ files) data.
        
        Returns
        -------
        stats : dict
            Basic statistics (times, mean/std for observables)
        structural : dict
            Structural analysis data (if computed)
        config : dict
            Base simulation configuration
        """
        if not self.statistics:
            raise RuntimeError("Must call compute_statistics() before to_plotting_format()")
        
        # Basic stats - already in the right format
        stats = dict(self.statistics)
        
        # Structural stats - start with computed structural_statistics if available
        structural = dict(getattr(self, 'structural_statistics', {}))
        
        # Add final distribution values from basic statistics if not already present
        # These are used by plot_ensemble_observables for histograms
        if 'final_largest_values' not in structural and 'largest_cluster_all' in stats:
            structural['final_largest_values'] = stats['largest_cluster_all'][:, -1]
        if 'final_fraction_bound_values' not in structural and 'fraction_bound_all' in stats:
            structural['final_fraction_bound_values'] = stats['fraction_bound_all'][:, -1]
        if 'final_avg_cluster_values' not in structural and 'avg_cluster_all' in stats:
            structural['final_avg_cluster_values'] = stats['avg_cluster_all'][:, -1]
        
        # Config
        config = self.base_config.to_dict()
        
        return stats, structural, config
    
    def compute_structural_statistics(
        self,
        stride: int = 10,
        parallel: bool = False,
        n_workers: Optional[int] = None,
    ):
        """
        Compute structural statistics (morphology, spatial, contacts, composition).

        This is computationally expensive as it processes trajectory frames.
        Must be called before plot_structural().

        Parameters
        ----------
        stride : int
            Analyze every Nth frame (default: 10)
        parallel : bool
            If True, analyze replicas concurrently with a process pool. Produces
            identical results to the sequential path (same per-replica worker).
        n_workers : int, optional
            Number of worker processes when parallel=True. Defaults to
            min(n_available, cpu_count()).
        """
        if not self.results_collected:
            raise RuntimeError("Must call collect_results() before compute_structural_statistics()")

        print("\n" + "=" * 60)
        print("COMPUTING ADVANCED STATISTICS")
        print("=" * 60)
        print(f"Using stride={stride} (analyzing every {stride}th frame)")

        available = self.replica_data['available_replicas']
        n_available = len(available)

        # Per-replica results, kept in the same order as `available` so downstream
        # processing is identical regardless of sequential vs parallel execution.
        morphology_data = [None] * n_available
        spatial_data = [None] * n_available
        contacts_data = [None] * n_available
        composition_data = [None] * n_available

        def _store(pos: int, res: Dict):
            morphology_data[pos] = res['morphology']
            spatial_data[pos] = res['spatial']
            contacts_data[pos] = res['contacts']
            composition_data[pos] = res['composition']

        if parallel and n_available > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from multiprocessing import cpu_count
            if n_workers is None:
                n_workers = min(n_available, cpu_count())
            n_workers = min(n_workers, n_available)
            print(f"Running structural analysis on {n_workers} workers...")

            task_args = [
                (i, self.replica_configs[i].output_file,
                 self.replica_configs[i].to_dict(), stride)
                for i in available
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_analyze_replica_structural_worker, arg): pos
                           for pos, arg in enumerate(task_args)}
                for future in as_completed(futures):
                    pos = futures[future]
                    res = future.result()
                    _store(pos, res)
                    tag = "with errors: " + "; ".join(res['errors']) if res['errors'] else "✓"
                    print(f"  Replica {available[pos]}: {tag}")
        else:
            for pos, i in enumerate(available):
                config = self.replica_configs[i]
                print(f"\n  Replica {i} ({pos + 1}/{n_available}):")
                res = _compute_replica_structural(config.output_file, config, stride)
                _store(pos, res)
                for err in res['errors']:
                    print(f"    ✗ {err}")
                if not res['errors']:
                    print(f"    ✓ morphology / spatial / contacts / composition")

        print("\nProcessing structural data...")
        
        # Process into ensemble statistics
        self.structural_statistics = {}
        
        # Helper to process data lists into mean/std arrays
        def process_data(data_list, time_key, keys):
            """Process data list into mean/std/all arrays with explicit time key."""
            valid = [d for d in data_list if d is not None]
            if not valid:
                return
            
            # Get common time grid (shortest)
            all_times = [d['times'] for d in valid]
            min_len = min(len(t) for t in all_times)
            self.structural_statistics[time_key] = valid[0]['times'][:min_len]
            
            for key in keys:
                if key not in valid[0]:
                    continue
                
                values = []
                for d in valid:
                    v = d[key][:min_len] if len(d[key]) >= min_len else d[key]
                    if len(v) == min_len:
                        values.append(v)
                
                if values:
                    matrix = np.array(values)
                    self.structural_statistics[f'{key}_mean'] = np.mean(matrix, axis=0)
                    self.structural_statistics[f'{key}_std'] = np.std(matrix, axis=0)
                    self.structural_statistics[f'{key}_all'] = matrix
        
        # Process each category with explicit time keys
        # Note: get_cluster_morphology() returns mean_rg, std_rg, mean_rg_normalized (not max_rg or mean_compactness)
        process_data(morphology_data, 'morphology_times', ['mean_rg', 'std_rg', 'mean_rg_normalized'])
        process_data(spatial_data, 'spatial_times', ['mean_nn_dist', 'std_nn_dist', 'mean_intra_nn_dist', 'std_intra_nn_dist'])
        process_data(contacts_data, 'contacts_times', ['mean_coord_qt', 'mean_coord_ft'])
        process_data(composition_data, 'composition_times', ['mean_qt_fraction'])
        
        # Rename composition keys for consistency with plotting functions
        if 'mean_qt_fraction_mean' in self.structural_statistics:
            self.structural_statistics['mean_composition_mean'] = self.structural_statistics.pop('mean_qt_fraction_mean')
            self.structural_statistics['mean_composition_std'] = self.structural_statistics.pop('mean_qt_fraction_std')
            self.structural_statistics['mean_composition_all'] = self.structural_statistics.pop('mean_qt_fraction_all')
        
        # Compute size fractions per replica
        print("  Computing size fractions...")
        size_fraction_data = []
        for idx, i in enumerate(available):
            config = self.replica_configs[i]
            h5_file = config.output_file
            try:
                sf = get_size_fractions(h5_file, config)
                size_fraction_data.append(sf)
            except Exception as e:
                print(f"    ✗ Size fractions failed for replica {i}: {e}")
                size_fraction_data.append(None)
        
        valid_sf = [d for d in size_fraction_data if d is not None]
        if valid_sf:
            # Get common time grid
            all_times = [d['times'] for d in valid_sf]
            min_len = min(len(t) for t in all_times)
            self.structural_statistics['size_fractions_times'] = valid_sf[0]['times'][:min_len]
            
            # Store category names as numpy string array (NPZ-compatible)
            category_names = valid_sf[0]['category_names']
            self.structural_statistics['size_fractions_category_names'] = np.array(category_names)
            
            # Store boundaries as separate numeric arrays (NPZ-compatible)
            boundaries = valid_sf[0]['boundaries']
            self.structural_statistics['size_fractions_boundary_min'] = np.array([b[1] for b in boundaries])
            self.structural_statistics['size_fractions_boundary_max'] = np.array([b[2] if b[2] is not None else -1 for b in boundaries])
            
            # Compute mean per category across replicas
            category_names = valid_sf[0]['category_names']
            for cat_name in category_names:
                values = []
                for d in valid_sf:
                    v = d['category_fractions'][cat_name][:min_len]
                    if len(v) == min_len:
                        values.append(v)
                if values:
                    matrix = np.array(values)
                    safe_key = cat_name.replace(' ', '_').replace('(', '').replace(')', '').replace('>', 'gt').replace('-', '_')
                    self.structural_statistics[f'size_frac_{safe_key}_mean'] = np.mean(matrix, axis=0)
                    self.structural_statistics[f'size_frac_{safe_key}_std'] = np.std(matrix, axis=0)
            
            print(f"    ✓ Size fractions ({len(category_names)} categories)")
        
        # Add final frame values for histograms
        valid_morph = [d for d in morphology_data if d is not None]
        if valid_morph:
            final_rg = [d['mean_rg'][-1] for d in valid_morph if len(d['mean_rg']) > 0]
            if final_rg:
                self.structural_statistics['final_rg_values'] = np.array(final_rg)
        
        valid_contacts = [d for d in contacts_data if d is not None]
        if valid_contacts:
            final_qt = [d['mean_coord_qt'][-1] for d in valid_contacts if len(d['mean_coord_qt']) > 0]
            final_ft = [d['mean_coord_ft'][-1] for d in valid_contacts if len(d['mean_coord_ft']) > 0]
            if final_qt:
                self.structural_statistics['final_coord_qt_values'] = np.array(final_qt)
            if final_ft:
                self.structural_statistics['final_coord_ft_values'] = np.array(final_ft)
        
        valid_comp = [d for d in composition_data if d is not None]
        if valid_comp:
            final_comp = [d['mean_qt_fraction'][-1] for d in valid_comp if len(d['mean_qt_fraction']) > 0]
            if final_comp:
                self.structural_statistics['final_composition_values'] = np.array(final_comp)
            
            # Composition vs size scatter data (aggregated from all replicas, final frame)
            all_fractions = []
            all_sizes = []
            for d in valid_comp:
                if d['qt_fraction_per_cluster'] and d['size_per_cluster']:
                    final_fracs = d['qt_fraction_per_cluster'][-1]
                    final_sizes = d['size_per_cluster'][-1]
                    all_fractions.extend(final_fracs)
                    all_sizes.extend(final_sizes)
            if all_fractions:
                self.structural_statistics['composition_vs_size_fractions'] = np.array(all_fractions)
                self.structural_statistics['composition_vs_size_sizes'] = np.array(all_sizes)
        
        print("✓ Structural statistics computed")
    
    def save_statistics(self, filepath: str):
        """
        Save computed statistics to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to output JSON file
        """
        if not self.statistics:
            raise RuntimeError("No statistics to save. Call compute_statistics() first.")
        
        # Convert numpy arrays to lists for JSON serialization
        stats_json = {}
        for key, value in self.statistics.items():
            if isinstance(value, np.ndarray):
                stats_json[key] = value.tolist()
            else:
                stats_json[key] = value
        
        # Add summary metrics
        if self.summary_metrics:
            stats_json['summary'] = self.summary_metrics
        
        with open(filepath, 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        print(f"✓ Saved statistics to {filepath}")
    
    def save_state(self, filepath: str):
        """
        Save full ensemble state for later reconstruction.
        
        Parameters
        ----------
        filepath : str
            Path to output JSON file
        """
        state = {
            'base_config': self.base_config.to_dict(),
            'n_replicas': self.n_replicas,
            'seeds': self.seeds,
            'name': self.name,
            'base_dir': self.base_dir,
            'folder_name': self.folder_name,
            'output_dir': self.output_dir,
            'available_replicas': self.replica_data.get('available_replicas', []),
        }
        
        # Add statistics if computed
        if self.statistics:
            state['statistics'] = {}
            for key, value in self.statistics.items():
                if isinstance(value, np.ndarray):
                    state['statistics'][key] = value.tolist()
                else:
                    state['statistics'][key] = value
        
        # Add summary metrics if computed
        if self.summary_metrics:
            state['summary_metrics'] = self.summary_metrics
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"✓ Saved ensemble state to {filepath}")
    
    def save_for_plotting(self, output_dir: Optional[str] = None):
        """
        Save analysis results to JSON/NPZ files for later plotting.
        
        This creates the same files as analyze_ensemble.py, allowing
        Ensemble_Plotting.ipynb to load and plot results identically
        for both local and cluster runs.
        
        Files created:
        - ensemble_statistics.json: Basic statistics (means, stds, summary)
        - ensemble_structural.npz: Structural analysis arrays (if computed)
        - ensemble_config.json: Configuration reference
        - ensemble_state.json: Ensemble state for reloading via EnsembleSimulation.load()
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save files. If None, uses self.output_dir
        """
        if not self.statistics:
            raise RuntimeError("No statistics to save. Call compute_statistics() first.")
        
        save_dir = output_dir if output_dir else self.output_dir
        save_dir = save_dir.rstrip("/") + "/"
        os.makedirs(save_dir, exist_ok=True)
        
        # Helper to convert numpy types for JSON
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # === Save ensemble_statistics.json ===
        stats_json = {
            'n_replicas': self.statistics.get('n_replicas', self.n_replicas),
            'available_replicas': self.replica_data.get('available_replicas', list(range(self.n_replicas))),
            'times': self.statistics['times'].tolist() if 'times' in self.statistics else [],
        }
        
        # Add time series statistics
        time_series_keys = [
            'bonds', 'energy', 'pressure', 'n_clusters', 'largest_cluster', 'fraction_bound',
            'avg_cluster', 'cumulative_reactions',
            'qt_count', 'ft_count', 'qtc_count', 'ftc_count', 'total_count'
        ]
        
        for key in time_series_keys:
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            if mean_key in self.statistics:
                stats_json[mean_key] = self.statistics[mean_key].tolist()
                if std_key in self.statistics:
                    stats_json[std_key] = self.statistics[std_key].tolist()
        
        # Add summary metrics
        if self.summary_metrics:
            stats_json['summary'] = convert_numpy(self.summary_metrics)
        
        stats_path = f"{save_dir}ensemble_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_json, f, indent=2)
        print(f"✓ Saved statistics to {stats_path}")
        
        # === Save ensemble_config.json ===
        config_path = f"{save_dir}ensemble_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.base_config.to_dict(), f, indent=2)
        print(f"✓ Saved configuration to {config_path}")
        
        # === Save ensemble_structural.npz (if structural statistics computed) ===
        npz_data = {
            'times': self.statistics.get('times', np.array([])),
            'n_replicas': np.array([self.statistics.get('n_replicas', self.n_replicas)]),
            'available_replicas': np.array(self.replica_data.get('available_replicas', list(range(self.n_replicas)))),
        }
        
        # Add all time series data with individual traces
        for key in time_series_keys:
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            all_key = f'{key}_all'
            if mean_key in self.statistics:
                npz_data[mean_key] = self.statistics[mean_key]
            if std_key in self.statistics:
                npz_data[std_key] = self.statistics[std_key]
            if all_key in self.statistics:
                npz_data[all_key] = self.statistics[all_key]
        
        # Add structural statistics if computed
        if hasattr(self, 'structural_statistics') and self.structural_statistics:
            for key, value in self.structural_statistics.items():
                if isinstance(value, np.ndarray):
                    npz_data[key] = value
                elif isinstance(value, list):
                    # Handle lists (e.g., category_names, boundaries)
                    # Convert to numpy array for NPZ storage
                    try:
                        npz_data[key] = np.array(value)
                    except (ValueError, TypeError):
                        pass  # Skip non-convertible types
        
        # Add final values for histograms
        if 'largest_cluster_all' in self.statistics:
            npz_data['final_largest_values'] = self.statistics['largest_cluster_all'][:, -1]
        if 'fraction_bound_all' in self.statistics:
            npz_data['final_fraction_bound_values'] = self.statistics['fraction_bound_all'][:, -1]
        if 'avg_cluster_all' in self.statistics:
            npz_data['final_avg_cluster_values'] = self.statistics['avg_cluster_all'][:, -1]
        
        structural_path = f"{save_dir}ensemble_structural.npz"
        np.savez_compressed(structural_path, **npz_data)
        print(f"✓ Saved structural data to {structural_path}")
        
        # === Save ensemble_state.json (for EnsembleSimulation.load()) ===
        state_path = f"{save_dir}ensemble_state.json"
        self.save_state(state_path)
        print(f"✓ Saved ensemble state to {state_path}")
    
    @classmethod
    def load(cls, dirpath: str) -> "EnsembleSimulation":
        """
        Load an ensemble from a directory.
        
        Parameters
        ----------
        dirpath : str
            Path to ensemble directory (containing configs/, replica_*/, etc.)
        
        Returns
        -------
        EnsembleSimulation
            Reconstructed ensemble object
        """
        dirpath = dirpath.rstrip("/") + "/"
        
        # Try to load state file first
        state_file = f"{dirpath}ensemble_state.json"
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            base_config = SimulationConfig.from_dict(state['base_config'])
            
            # Create ensemble with _from_load flag to skip auto-generation
            ensemble = cls.__new__(cls)
            ensemble.base_config = base_config
            ensemble.n_replicas = state['n_replicas']
            ensemble.seeds = state['seeds']
            ensemble.output_dir = state['output_dir']
            ensemble.name = state.get('name', None)
            ensemble.base_dir = state.get('base_dir', '.')
            ensemble.folder_name = os.path.basename(ensemble.output_dir.rstrip('/'))
            
            # Regenerate replica configs
            ensemble.replica_configs = []
            for i, seed in enumerate(ensemble.seeds):
                ensemble.replica_configs.append(ensemble._create_replica_config(i, seed))
            
            # Initialize storage
            ensemble.results_collected = False
            ensemble.replica_data = {}
            ensemble.statistics = {}
            ensemble.structural_statistics = {}
            ensemble.summary_metrics = {}
            
            # Restore statistics if present
            if 'statistics' in state:
                ensemble.statistics = {}
                for key, value in state['statistics'].items():
                    if isinstance(value, list):
                        ensemble.statistics[key] = np.array(value)
                    else:
                        ensemble.statistics[key] = value
                # Mark results as collected since we have statistics
                ensemble.results_collected = True
            
            # Restore available_replicas to replica_data
            if 'available_replicas' in state:
                ensemble.replica_data = {'available_replicas': state['available_replicas']}
            
            if 'summary_metrics' in state:
                ensemble.summary_metrics = state['summary_metrics']
            
            print(f"✓ Loaded ensemble from {state_file}")
            return ensemble
        
        # Otherwise, reconstruct from config files
        config_dir = f"{dirpath}configs/"
        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"No configs found in {dirpath}")
        
        # Find all config files
        config_files = sorted([f for f in os.listdir(config_dir) if f.endswith('.json')])
        if not config_files:
            raise FileNotFoundError(f"No config JSON files found in {config_dir}")
        
        # Load first config as base
        base_config = SimulationConfig.load_json(f"{config_dir}{config_files[0]}")
        
        # Extract seeds from all configs
        seeds = []
        for cf in config_files:
            c = SimulationConfig.load_json(f"{config_dir}{cf}")
            seeds.append(c.rng_seed)
        
        # Create ensemble with direct output_dir (skip auto-generation)
        ensemble = cls.__new__(cls)
        ensemble.base_config = base_config
        ensemble.n_replicas = len(config_files)
        ensemble.seeds = seeds
        ensemble.output_dir = dirpath
        ensemble.name = None
        ensemble.base_dir = os.path.dirname(dirpath.rstrip('/'))
        ensemble.folder_name = os.path.basename(dirpath.rstrip('/'))
        
        # Regenerate replica configs
        ensemble.replica_configs = []
        for i, seed in enumerate(ensemble.seeds):
            ensemble.replica_configs.append(ensemble._create_replica_config(i, seed))
        
        # Initialize storage
        ensemble.results_collected = False
        ensemble.replica_data = {}
        ensemble.statistics = {}
        ensemble.structural_statistics = {}
        ensemble.summary_metrics = {}
        
        print(f"✓ Reconstructed ensemble from {len(config_files)} config files")
        return ensemble

