#!/usr/bin/env python3
"""
Ensemble Analysis Script

Run ensemble analysis on cluster and save results to JSON/NPZ files
for local plotting. Supports parallel processing for faster analysis.

Also provides ensemble comparison functionality (data loading/saving).

This module has NO matplotlib dependency. All plotting is done via
agglomeration_plotting.py (used in notebooks).

Usage:
    # Analyze single ensemble
    python analyze_ensemble.py --ensemble-dir ensemble_results/ --stride 10
    python analyze_ensemble.py --ensemble-dir ensemble_results/ --parallel --n-workers 8
    
    # Compare multiple ensembles (saves data only, plot in notebook)
    python analyze_ensemble.py compare --ensemble "Label1=path1/" --ensemble "Label2=path2/" --output comparison/
"""

import argparse
import json
import os
import sys

import numpy as np

from agglomeration_ensemble_simulation import EnsembleSimulation
from agglomeration_simulation import NS_TO_US


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ensemble simulation results and save to JSON/NPZ"
    )
    parser.add_argument("--ensemble-dir", required=True,
                        help="Path to ensemble output directory")
    parser.add_argument("--stride", type=int, default=10,
                        help="Stride for structural analysis (default: 10)")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel processing for structural analysis")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Number of parallel workers (default: min(n_replicas, cpu_count))")
    args = parser.parse_args()

    ensemble_dir = args.ensemble_dir.rstrip("/") + "/"

    print("=" * 60)
    print("ENSEMBLE ANALYSIS")
    print("=" * 60)
    print(f"Ensemble directory: {ensemble_dir}")
    print(f"Structural analysis stride: {args.stride}")
    if args.parallel:
        print("Parallel processing: ENABLED")
    print()

    # Drive the shared EnsembleSimulation pipeline so this CLI and a local
    # run_local() produce byte-identical ensemble_statistics.json /
    # ensemble_structural.npz. save_for_plotting() is the single source of truth
    # for the on-disk schema (no duplicated aggregation/writer logic here).
    ensemble = EnsembleSimulation.load(ensemble_dir)
    ensemble.collect_results(require_all=False)
    ensemble.compute_statistics()
    ensemble.compute_summary_metrics()
    ensemble.compute_structural_statistics(
        stride=args.stride, parallel=args.parallel, n_workers=args.n_workers
    )
    ensemble.save_for_plotting()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output files in {ensemble_dir}:")
    print("  ensemble_statistics.json")
    print("  ensemble_structural.npz")
    print("  ensemble_config.json")
    print("  ensemble_state.json")
    print("\nDownload these for local plotting with Plot_Ensemble_Results.ipynb")

    if ensemble.summary_metrics:
        m = ensemble.summary_metrics
        print("\n" + "-" * 60)
        print("SUMMARY METRICS")
        print("-" * 60)
        print(f"  Replicas: {m['n_replicas']}")
        if 'final_bonds_mean' in m:
            print(f"  Final bonds: {m['final_bonds_mean']:.1f} \u00b1 {m['final_bonds_std']:.1f}")
        if 'final_largest_fraction_mean' in m:
            print(f"  Final largest cluster: {m['final_largest_fraction_mean']*100:.1f}% "
                  f"\u00b1 {m['final_largest_fraction_std']*100:.1f}% of particles")
        if 'half_time_mean' in m:
            ht = m['half_time_mean'] * NS_TO_US
            ht_std = m['half_time_std'] * NS_TO_US
            print(f"  Half-time: {ht:.2f} \u00b1 {ht_std:.2f} \u00b5s")


# =============================================================================
# ENSEMBLE COMPARISON FUNCTIONS
# =============================================================================

def load_ensemble_for_comparison(ensemble_dir: str, label: str) -> dict:
    """
    Load a single ensemble's pre-computed statistics for comparison.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory containing JSON/NPZ files
    label : str
        Label for this ensemble (used in plots/legends)
    
    Returns
    -------
    dict
        Ensemble data with config, statistics, structural data, and label
    
    Notes
    -----
    Per-replica time-series arrays (*_all keys like bonds_all, fraction_bound_all)
    are merged into stats for overlay of individual replica points on bar charts.
    Structural analysis data (morphology, spatial, contacts, composition) is
    loaded into the 'structural' dict.
    """
    ensemble_dir = ensemble_dir.rstrip("/") + "/"
    
    # Load statistics JSON
    stats_file = f"{ensemble_dir}ensemble_statistics.json"
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Statistics file not found: {stats_file}")
    
    with open(stats_file, 'r') as f:
        stats_json = json.load(f)
    
    # Load config JSON
    config_file = f"{ensemble_dir}ensemble_config.json"
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Convert lists to numpy arrays
    stats = {}
    for key, value in stats_json.items():
        if isinstance(value, list):
            stats[key] = np.array(value)
        else:
            stats[key] = value
    
    # Load NPZ file: merge per-replica time-series into stats,
    # and structural analysis keys into structural dict
    npz_file = f"{ensemble_dir}ensemble_structural.npz"
    structural = {}
    structural_path = npz_file if os.path.exists(npz_file) else None
    
    # Per-replica time-series keys that go into stats
    time_series_all_keys = {
        'bonds_all', 'energy_all', 'pressure_all', 'n_clusters_all',
        'largest_cluster_all', 'fraction_bound_all', 'avg_cluster_all',
        'cumulative_reactions_all',
        'qt_count_all', 'ft_count_all', 'qtc_count_all', 'ftc_count_all',
        'total_count_all',
    }
    
    if structural_path:
        with np.load(npz_file, allow_pickle=True) as data:
            for key in data.files:
                if key.endswith('_all') and key in time_series_all_keys:
                    # Merge per-replica time-series into stats
                    stats[key] = data[key]
                else:
                    # All other keys go into structural
                    structural[key] = data[key]
    
    # Get timestep for time conversion
    timestep = config.get('timestep', 0.001)
    
    # Convert times to microseconds
    if 'times' in stats:
        times_us = stats['times'] * timestep * 1e-3
    else:
        times_us = np.array([])
    
    return {
        'label': label,
        'dir': ensemble_dir,
        'config': config,
        'stats': stats,
        'structural': structural,
        'structural_path': structural_path,
        'times_us': times_us,
        'timestep': timestep,
        'n_replicas': stats.get('n_replicas', len(stats.get('available_replicas', []))),
    }


def compare_ensembles(ensembles_dict: dict) -> dict:
    """
    Build comparison data structure from multiple ensembles.
    
    Parameters
    ----------
    ensembles_dict : dict
        Dictionary mapping labels to ensemble directories
        Example: {"200 Ft": "ensemble_200ft/", "400 Ft": "ensemble_400ft/"}
    
    Returns
    -------
    dict
        Comparison data structure with all ensembles' data
    """
    print("=" * 60)
    print("LOADING ENSEMBLES FOR COMPARISON")
    print("=" * 60)
    
    ensembles = {}
    
    for label, ensemble_dir in ensembles_dict.items():
        print(f"\nLoading: {label}")
        print(f"  Directory: {ensemble_dir}")
        try:
            data = load_ensemble_for_comparison(ensemble_dir, label)
            ensembles[label] = data
            struct_status = "with structural data" if data["structural"] else "basic stats only"
            print(f"  ✓ Loaded {data['n_replicas']} replicas, {len(data['times_us'])} time points ({struct_status})")
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue
    
    if len(ensembles) < 2:
        raise ValueError(f"Need at least 2 ensembles for comparison, got {len(ensembles)}")
    
    # Detect parameter differences
    param_diffs = _detect_parameter_differences(ensembles)
    
    # Build comparison structure
    comparison = {
        'ensembles': ensembles,
        'labels': list(ensembles.keys()),
        'n_ensembles': len(ensembles),
        'parameter_differences': param_diffs,
        'metadata': {
            'created': str(np.datetime64('now')),
        }
    }
    
    return comparison


def _detect_parameter_differences(ensembles: dict) -> dict:
    """Detect which parameters differ between ensembles."""
    if len(ensembles) < 2:
        return {}
    
    # Parameters to check
    params_to_check = [
        ('n_qt', lambda c: c.get('n_qt')),
        ('n_ft', lambda c: c.get('n_ft')),
        ('qt_radius', lambda c: c.get('qt', {}).get('radius')),
        ('ft_radius', lambda c: c.get('ft', {}).get('radius')),
        ('qt_diffusion', lambda c: c.get('qt', {}).get('diffusion')),
        ('ft_diffusion', lambda c: c.get('ft', {}).get('diffusion')),
        ('binding_radius', lambda c: c.get('topology', {}).get('binding_radius')),
        ('kon', lambda c: c.get('topology', {}).get('kon')),
        ('k_bond', lambda c: c.get('topology', {}).get('k_bond')),
        ('epsilon_QtQt', lambda c: c.get('lj', {}).get('epsilon_QtQt')),
        ('epsilon_FtFt', lambda c: c.get('lj', {}).get('epsilon_FtFt')),
        ('epsilon_QtFt', lambda c: c.get('lj', {}).get('epsilon_QtFt')),
        ('epsilon_QtCFtC', lambda c: c.get('lj', {}).get('epsilon_QtCFtC')),
        ('potential_type', lambda c: c.get('lj', {}).get('potential_type')),
        ('box_size', lambda c: tuple(c.get('box_size', []))),
        ('temperature', lambda c: c.get('temperature')),
        ('timestep', lambda c: c.get('timestep')),
        ('n_steps', lambda c: c.get('n_steps')),
    ]
    
    diffs = {}
    labels = list(ensembles.keys())
    
    for param_name, getter in params_to_check:
        values = {}
        for label in labels:
            config = ensembles[label]['config']
            values[label] = getter(config)
        
        # Check if values differ
        unique_values = set(str(v) for v in values.values())
        if len(unique_values) > 1:
            diffs[param_name] = values
    
    return diffs


def print_parameter_differences(comparison: dict):
    """Print parameter differences between ensembles."""
    print("\n" + "=" * 60)
    print("PARAMETER DIFFERENCES")
    print("=" * 60)
    
    diffs = comparison.get('parameter_differences', {})
    
    if not diffs:
        print("No parameter differences detected (ensembles have identical configs)")
        return
    
    for param, values in diffs.items():
        print(f"\n{param}:")
        for label, value in values.items():
            print(f"  {label}: {value}")


def save_comparison_data(comparison: dict, output_dir: str):
    """
    Save comparison data to JSON and NPZ files.
    
    Parameters
    ----------
    comparison : dict
        Comparison data from compare_ensembles()
    output_dir : str
        Output directory
    """
    output_dir = output_dir.rstrip("/") + "/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare JSON-serializable data
    json_data = {
        'labels': comparison['labels'],
        'n_ensembles': comparison['n_ensembles'],
        'parameter_differences': comparison['parameter_differences'],
        'metadata': comparison['metadata'],
        'ensembles': {}
    }
    
    # Prepare NPZ data
    npz_data = {}
    
    for label, ens in comparison['ensembles'].items():
        # JSON: config and scalar values
        json_data['ensembles'][label] = {
            'config': ens['config'],
            'n_replicas': ens['n_replicas'],
            'timestep': ens['timestep'],
        }
        
        # Add summary metrics if available
        if 'summary' in ens['stats']:
            json_data['ensembles'][label]['summary'] = ens['stats']['summary']
        
        # NPZ: time series arrays
        safe_label = label.replace(" ", "_").replace("/", "_")
        npz_data[f'{safe_label}_times_us'] = ens['times_us']
        
        # Add all mean/std time series
        for key in ['bonds', 'energy', 'pressure', 'n_clusters', 'largest_cluster', 
                    'fraction_bound', 'avg_cluster', 'cumulative_reactions',
                    'qt_count', 'ft_count', 'qtc_count', 'ftc_count']:
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            if mean_key in ens['stats']:
                npz_data[f'{safe_label}_{mean_key}'] = ens['stats'][mean_key]
            if std_key in ens['stats']:
                npz_data[f'{safe_label}_{std_key}'] = ens['stats'][std_key]
    
    # Save JSON
    json_path = f"{output_dir}comparison_data.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Saved comparison metadata to {json_path}")
    
    # Save NPZ
    npz_path = f"{output_dir}comparison_timeseries.npz"
    np.savez_compressed(npz_path, **npz_data)
    print(f"✓ Saved comparison time series to {npz_path}")


def load_comparison_data(output_dir: str) -> dict:
    """
    Load comparison data from JSON and NPZ files.
    
    Parameters
    ----------
    output_dir : str
        Directory containing comparison_data.json and comparison_timeseries.npz
    
    Returns
    -------
    dict
        Comparison data structure
    """
    output_dir = output_dir.rstrip("/") + "/"
    
    # Load JSON
    json_path = f"{output_dir}comparison_data.json"
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Load NPZ
    npz_path = f"{output_dir}comparison_timeseries.npz"
    npz_data = {}
    with np.load(npz_path, allow_pickle=True) as data:
        for key in data.files:
            npz_data[key] = data[key]
    
    # Reconstruct comparison structure
    comparison = {
        'labels': json_data['labels'],
        'n_ensembles': json_data['n_ensembles'],
        'parameter_differences': json_data['parameter_differences'],
        'metadata': json_data['metadata'],
        'ensembles': {}
    }
    
    for label in json_data['labels']:
        safe_label = label.replace(" ", "_").replace("/", "_")
        ens_json = json_data['ensembles'][label]
        
        # Reconstruct ensemble data
        ens = {
            'label': label,
            'config': ens_json['config'],
            'n_replicas': ens_json['n_replicas'],
            'timestep': ens_json['timestep'],
            'times_us': npz_data.get(f'{safe_label}_times_us', np.array([])),
            'stats': {},
        }
        
        # Add summary if available
        if 'summary' in ens_json:
            ens['stats']['summary'] = ens_json['summary']
        
        # Reconstruct stats from NPZ
        for key in ['bonds', 'energy', 'pressure', 'n_clusters', 'largest_cluster', 
                    'fraction_bound', 'avg_cluster', 'cumulative_reactions',
                    'qt_count', 'ft_count', 'qtc_count', 'ftc_count']:
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            npz_mean_key = f'{safe_label}_{mean_key}'
            npz_std_key = f'{safe_label}_{std_key}'
            if npz_mean_key in npz_data:
                ens['stats'][mean_key] = npz_data[npz_mean_key]
            if npz_std_key in npz_data:
                ens['stats'][std_key] = npz_data[npz_std_key]
        
        comparison['ensembles'][label] = ens
    
    return comparison


def run_comparison(ensemble_specs: list, output_dir: str):
    """
    Run full ensemble comparison workflow (data only, no plotting).
    
    Loads ensembles, detects parameter differences, prints summaries,
    and saves comparison data to JSON/NPZ files for later plotting
    in notebooks.
    
    Parameters
    ----------
    ensemble_specs : list
        List of "Label=path" strings
    output_dir : str
        Output directory for comparison results
    
    Returns
    -------
    dict
        Comparison data structure
    """
    # Parse ensemble specifications
    ensembles_dict = {}
    for spec in ensemble_specs:
        if '=' not in spec:
            raise ValueError(f"Invalid ensemble spec '{spec}'. Use format: 'Label=path/to/ensemble/'")
        label, path = spec.split('=', 1)
        ensembles_dict[label] = path
    
    # Load and compare ensembles
    comparison = compare_ensembles(ensembles_dict)
    
    # Print parameter differences
    print_parameter_differences(comparison)
    
    # Print detailed summary for each ensemble
    print("\n" + "=" * 60)
    print("ENSEMBLE SUMMARIES")
    print("=" * 60)
    
    for label in comparison['labels']:
        ens = comparison['ensembles'][label]
        print(f"\n{label}:")
        print(f"  Directory: {ens.get('dir', 'N/A')}")
        print(f"  Replicas: {ens['n_replicas']}")
        print(f"  Time points: {len(ens['times_us'])}")
        if len(ens['times_us']) > 0:
            print(f"  Duration: {ens['times_us'][-1]:.2f} µs")
        
        # Print key config parameters
        config = ens['config']
        print(f"  Particles: {config.get('n_qt', '?')} Qt + {config.get('n_ft', '?')} Ft")
        
        # Print final values
        stats = ens['stats']
        if 'bonds_mean' in stats and len(stats['bonds_mean']) > 0:
            print(f"  Final bonds: {stats['bonds_mean'][-1]:.1f} ± {stats.get('bonds_std', [0])[-1]:.1f}")
        if 'fraction_bound_mean' in stats and len(stats['fraction_bound_mean']) > 0:
            fb = stats['fraction_bound_mean'][-1]
            fb_std = stats.get('fraction_bound_std', [0])[-1]
            print(f"  Final fraction bound: {fb*100:.1f}% ± {fb_std*100:.1f}%")
    
    # Save comparison data
    print("\n" + "=" * 60)
    print("SAVING COMPARISON DATA")
    print("=" * 60)
    save_comparison_data(comparison, output_dir)
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nTo plot in notebook, use:")
    print(f"  import analyze_ensemble as ae")
    print(f"  import agglomeration_plotting as plotting")
    print(f"  comparison = ae.load_comparison_data('{output_dir}')")
    print(f"  plotting.plot_comparison_summary(comparison)")
    
    return comparison


def compare_main():
    """Entry point for compare subcommand."""
    parser = argparse.ArgumentParser(
        description="Compare multiple ensemble simulation results and save data for plotting"
    )
    parser.add_argument(
        "--ensemble", "-e",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Ensemble to include in comparison (format: 'Label=path/to/ensemble/'). "
             "Can be specified multiple times."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for comparison results"
    )
    
    args = parser.parse_args()
    
    run_comparison(
        ensemble_specs=args.ensemble,
        output_dir=args.output,
    )


if __name__ == "__main__":
    # Check if 'compare' subcommand is used
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        # Remove 'compare' from argv and run comparison
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        compare_main()
    else:
        main()
