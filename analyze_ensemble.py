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
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np

# Import the analysis module
import agglomeration_analysis as analysis
from agglomeration_ensemble_simulation import EnsembleSimulation
from agglomeration_simulation import NS_TO_US


def analyze_single_replica(args):
    """
    Analyze a single replica (for parallel processing).
    
    Parameters
    ----------
    args : tuple
        (replica_index, h5_file, config_dict, stride)
    
    Returns
    -------
    dict
        Results containing morphology, spatial, contacts, and composition data
    """
    replica_idx, h5_file, config_dict, stride = args
    
    # Reconstruct config from dict
    config = analysis.SimulationConfig.from_dict(config_dict)
    
    result = {
        'replica_idx': replica_idx,
        'morphology': None,
        'spatial': None,
        'contacts': None,
        'composition': None,
        'errors': []
    }
    
    # Extract frame data ONCE and share between all analysis functions
    try:
        frame_data = analysis._extract_frame_data(h5_file, config, stride=stride, verbose=False)
    except Exception as e:
        result['errors'].append(f"Frame data extraction: {e}")
        return result
    
    # Morphology
    try:
        morph = analysis.get_cluster_morphology(h5_file, config, stride=stride, frame_data=frame_data)
        result['morphology'] = morph
    except Exception as e:
        result['errors'].append(f"Morphology: {e}")
    
    # Spatial
    try:
        spatial = analysis.get_spatial_distribution(h5_file, config, stride=stride, frame_data=frame_data)
        result['spatial'] = spatial
    except Exception as e:
        result['errors'].append(f"Spatial: {e}")
    
    # Contacts
    try:
        contacts = analysis.get_contact_analysis(h5_file, config, stride=stride, frame_data=frame_data)
        result['contacts'] = contacts
    except Exception as e:
        result['errors'].append(f"Contacts: {e}")
    
    # Composition
    try:
        composition = analysis.get_cluster_composition(h5_file, config, stride=stride, frame_data=frame_data)
        result['composition'] = composition
    except Exception as e:
        result['errors'].append(f"Composition: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ensemble simulation results and save to JSON/NPZ"
    )
    parser.add_argument(
        "--ensemble-dir",
        required=True,
        help="Path to ensemble output directory"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Stride for structural analysis (default: 10)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing for structural analysis"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(n_replicas, cpu_count))"
    )
    
    args = parser.parse_args()
    
    ensemble_dir = args.ensemble_dir.rstrip("/") + "/"
    
    print("=" * 60)
    print("ENSEMBLE ANALYSIS")
    print("=" * 60)
    print(f"Ensemble directory: {ensemble_dir}")
    print(f"Structural analysis stride: {args.stride}")
    if args.parallel:
        print(f"Parallel processing: ENABLED")
    print()
    
    # Load ensemble from config files
    print("Loading ensemble configuration...")
    ensemble = EnsembleSimulation.load(ensemble_dir)
    
    # Collect basic results
    print("\n" + "-" * 60)
    print("COLLECTING BASIC RESULTS")
    print("-" * 60)
    ensemble.collect_results(require_all=False)
    
    # Compute statistics
    print("\n" + "-" * 60)
    print("COMPUTING STATISTICS")
    print("-" * 60)
    ensemble.compute_statistics()
    
    # Compute summary metrics
    ensemble.compute_summary_metrics()
    
    # Prepare basic statistics for JSON
    print("\n" + "-" * 60)
    print("SAVING BASIC STATISTICS")
    print("-" * 60)
    
    stats_json = {
        'n_replicas': ensemble.statistics['n_replicas'],
        'available_replicas': ensemble.replica_data['available_replicas'],
        'times': ensemble.statistics['times'].tolist(),
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
        if mean_key in ensemble.statistics:
            stats_json[mean_key] = ensemble.statistics[mean_key].tolist()
            stats_json[std_key] = ensemble.statistics[std_key].tolist()
    
    # Add summary metrics
    if ensemble.summary_metrics:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        stats_json['summary'] = convert_numpy(ensemble.summary_metrics)
    
    # Save basic statistics JSON
    stats_path = f"{ensemble_dir}ensemble_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_json, f, indent=2)
    print(f"✓ Saved basic statistics to {stats_path}")
    
    # Save config JSON
    config_path = f"{ensemble_dir}ensemble_config.json"
    with open(config_path, 'w') as f:
        json.dump(ensemble.base_config.to_dict(), f, indent=2)
    print(f"✓ Saved configuration to {config_path}")
    
    # Prepare arrays for NPZ (including individual traces)
    npz_data = {
        'times': ensemble.statistics['times'],
        'n_replicas': np.array([ensemble.statistics['n_replicas']]),
        'available_replicas': np.array(ensemble.replica_data['available_replicas']),
    }
    
    # Add all time series data with individual traces
    for key in time_series_keys:
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'
        all_key = f'{key}_all'
        if mean_key in ensemble.statistics:
            npz_data[mean_key] = ensemble.statistics[mean_key]
            npz_data[std_key] = ensemble.statistics[std_key]
        if all_key in ensemble.statistics:
            npz_data[all_key] = ensemble.statistics[all_key]
    
    # Structural analysis
    print("\n" + "-" * 60)
    print("COMPUTING STRUCTURAL ANALYSIS")
    print("-" * 60)
    
    available_replicas = ensemble.replica_data['available_replicas']
    n_available = len(available_replicas)
    
    if n_available == 0:
        print("No replicas available for structural analysis")
    elif args.parallel and n_available > 1:
        # Parallel processing
        n_workers = args.n_workers
        if n_workers is None:
            n_workers = min(n_available, cpu_count())
        n_workers = min(n_workers, n_available)
        
        print(f"Running parallel analysis with {n_workers} workers...")
        
        # Prepare arguments
        task_args = []
        for i in available_replicas:
            config = ensemble.replica_configs[i]
            h5_file = config.output_file
            task_args.append((i, h5_file, config.to_dict(), args.stride))
        
        # Run in parallel
        morphology_data = [None] * n_available
        spatial_data = [None] * n_available
        contacts_data = [None] * n_available
        composition_data = [None] * n_available
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Map futures back to their position in results arrays.
            # task_idx = position in task_args (0..n_available-1), used for array indexing
            # result['replica_idx'] = original replica number, used for logging
            futures = {executor.submit(analyze_single_replica, arg): task_idx 
                      for task_idx, arg in enumerate(task_args)}
            
            for future in as_completed(futures):
                task_idx = futures[future]
                try:
                    result = future.result()
                    replica_idx = result['replica_idx']
                    morphology_data[task_idx] = result['morphology']
                    spatial_data[task_idx] = result['spatial']
                    contacts_data[task_idx] = result['contacts']
                    composition_data[task_idx] = result['composition']
                    
                    if result['errors']:
                        print(f"  Replica {replica_idx}: Completed with errors: {result['errors']}")
                    else:
                        print(f"  Replica {replica_idx}: ✓ Complete")
                except Exception as e:
                    print(f"  Task {task_idx}: FAILED - {e}")
    else:
        # Sequential processing
        if args.parallel:
            print("Only 1 replica available, using sequential processing")
        
        morphology_data = []
        spatial_data = []
        contacts_data = []
        composition_data = []
        
        for idx, i in enumerate(available_replicas):
            config = ensemble.replica_configs[i]
            h5_file = config.output_file
            
            print(f"  Replica {i} ({idx+1}/{n_available}): Analyzing...")
            
            # Extract frame data ONCE and share between all analysis functions
            try:
                frame_data = analysis._extract_frame_data(h5_file, config, stride=args.stride, verbose=False)
                print(f"    ✓ Frame data: {frame_data['n_frames']} frames extracted")
            except Exception as e:
                print(f"    ✗ Frame data extraction failed: {e}")
                morphology_data.append(None)
                spatial_data.append(None)
                contacts_data.append(None)
                composition_data.append(None)
                continue
            
            # Morphology
            try:
                morph = analysis.get_cluster_morphology(h5_file, config, stride=args.stride, frame_data=frame_data)
                morphology_data.append(morph)
                print(f"    ✓ Morphology")
            except Exception as e:
                print(f"    ✗ Morphology failed: {e}")
                morphology_data.append(None)
            
            # Spatial
            try:
                spatial = analysis.get_spatial_distribution(h5_file, config, stride=args.stride, frame_data=frame_data)
                spatial_data.append(spatial)
                print(f"    ✓ Spatial")
            except Exception as e:
                print(f"    ✗ Spatial failed: {e}")
                spatial_data.append(None)
            
            # Contacts
            try:
                contacts = analysis.get_contact_analysis(h5_file, config, stride=args.stride, frame_data=frame_data)
                contacts_data.append(contacts)
                print(f"    ✓ Contacts")
            except Exception as e:
                print(f"    ✗ Contacts failed: {e}")
                contacts_data.append(None)
            
            # Composition
            try:
                composition = analysis.get_cluster_composition(h5_file, config, stride=args.stride, frame_data=frame_data)
                composition_data.append(composition)
                print(f"    ✓ Composition")
            except Exception as e:
                print(f"    ✗ Composition failed: {e}")
                composition_data.append(None)
    
    # Process structural data into arrays
    print("\nProcessing structural data...")
    
    def process_structural_data(data_list, keys):
        """Process list of dicts into mean/std/all arrays."""
        valid_data = [d for d in data_list if d is not None]
        if not valid_data:
            return {}
        
        result = {}
        
        # Get common time grid (shortest)
        all_times = [d['times'] for d in valid_data]
        min_len = min(len(t) for t in all_times)
        result['times'] = valid_data[0]['times'][:min_len]
        
        for key in keys:
            if key not in valid_data[0]:
                continue
                
            all_values = []
            for d in valid_data:
                if key in d:
                    vals = d[key][:min_len] if len(d[key]) >= min_len else d[key]
                    if len(vals) == min_len:
                        all_values.append(vals)
            
            if all_values:
                values_matrix = np.array(all_values)
                result[f'{key}_mean'] = np.mean(values_matrix, axis=0)
                result[f'{key}_std'] = np.std(values_matrix, axis=0)
                result[f'{key}_all'] = values_matrix
        
        return result
    
    # Process morphology
    # Note: get_cluster_morphology() returns mean_rg, std_rg, mean_rg_normalized (not max_rg or mean_compactness)
    morph_result = process_structural_data(morphology_data, ['mean_rg', 'std_rg', 'mean_rg_normalized'])
    if morph_result:
        npz_data['morphology_times'] = morph_result['times']
        for key in ['mean_rg_mean', 'mean_rg_std', 'mean_rg_all', 
                   'std_rg_mean', 'std_rg_std', 'std_rg_all',
                   'mean_rg_normalized_mean', 'mean_rg_normalized_std', 'mean_rg_normalized_all']:
            if key in morph_result:
                npz_data[key] = morph_result[key]
    
    # Process spatial
    spatial_result = process_structural_data(spatial_data, ['mean_nn_dist', 'std_nn_dist', 'mean_intra_nn_dist', 'std_intra_nn_dist'])
    if spatial_result:
        npz_data['spatial_times'] = spatial_result['times']
        for key in ['mean_nn_dist_mean', 'mean_nn_dist_std', 'mean_nn_dist_all',
                   'mean_intra_nn_dist_mean', 'mean_intra_nn_dist_std', 'mean_intra_nn_dist_all']:
            if key in spatial_result:
                npz_data[key] = spatial_result[key]
    
    # Process contacts
    contacts_result = process_structural_data(contacts_data, ['mean_coord_qt', 'mean_coord_ft'])
    if contacts_result:
        npz_data['contacts_times'] = contacts_result['times']
        for key in ['mean_coord_qt_mean', 'mean_coord_qt_std', 'mean_coord_qt_all',
                   'mean_coord_ft_mean', 'mean_coord_ft_std', 'mean_coord_ft_all']:
            if key in contacts_result:
                npz_data[key] = contacts_result[key]
    
    # Process composition
    composition_result = process_structural_data(composition_data, ['mean_qt_fraction'])
    if composition_result:
        npz_data['composition_times'] = composition_result['times']
        # Rename keys for consistency with plotting functions
        if 'mean_qt_fraction_mean' in composition_result:
            npz_data['mean_composition_mean'] = composition_result['mean_qt_fraction_mean']
            npz_data['mean_composition_std'] = composition_result['mean_qt_fraction_std']
            npz_data['mean_composition_all'] = composition_result['mean_qt_fraction_all']
    
    # Add final frame values for histograms
    valid_morph = [d for d in morphology_data if d is not None]
    if valid_morph:
        final_rg = [d['mean_rg'][-1] for d in valid_morph if len(d['mean_rg']) > 0]
        if final_rg:
            npz_data['final_rg_values'] = np.array(final_rg)
    
    valid_contacts = [d for d in contacts_data if d is not None]
    if valid_contacts:
        final_coord_qt = [d['mean_coord_qt'][-1] for d in valid_contacts if len(d['mean_coord_qt']) > 0]
        final_coord_ft = [d['mean_coord_ft'][-1] for d in valid_contacts if len(d['mean_coord_ft']) > 0]
        if final_coord_qt:
            npz_data['final_coord_qt_values'] = np.array(final_coord_qt)
        if final_coord_ft:
            npz_data['final_coord_ft_values'] = np.array(final_coord_ft)
    
    # Composition final values and scatter data
    valid_comp = [d for d in composition_data if d is not None]
    if valid_comp:
        final_comp = [d['mean_qt_fraction'][-1] for d in valid_comp if len(d['mean_qt_fraction']) > 0]
        if final_comp:
            npz_data['final_composition_values'] = np.array(final_comp)
        
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
            npz_data['composition_vs_size_fractions'] = np.array(all_fractions)
            npz_data['composition_vs_size_sizes'] = np.array(all_sizes)
    
    if 'largest_cluster_all' in ensemble.statistics:
        npz_data['final_largest_values'] = ensemble.statistics['largest_cluster_all'][:, -1]
    
    if 'fraction_bound_all' in ensemble.statistics:
        npz_data['final_fraction_bound_values'] = ensemble.statistics['fraction_bound_all'][:, -1]
    
    if 'avg_cluster_all' in ensemble.statistics:
        npz_data['final_avg_cluster_values'] = ensemble.statistics['avg_cluster_all'][:, -1]
    
    # Compute size fractions per replica
    print("\nComputing size fractions...")
    size_fraction_data = []
    for idx, i in enumerate(available_replicas):
        config = ensemble.replica_configs[i]
        h5_file = config.output_file
        try:
            sf = analysis.get_size_fractions(h5_file, config)
            size_fraction_data.append(sf)
        except Exception as e:
            print(f"  Replica {i}: Size fractions failed: {e}")
            size_fraction_data.append(None)
    
    valid_sf = [d for d in size_fraction_data if d is not None]
    if valid_sf:
        all_times_sf = [d['times'] for d in valid_sf]
        min_len_sf = min(len(t) for t in all_times_sf)
        npz_data['size_fractions_times'] = valid_sf[0]['times'][:min_len_sf]
        
        # Store category names and boundaries as string arrays
        category_names = valid_sf[0]['category_names']
        npz_data['size_fractions_category_names'] = np.array(category_names)
        
        # Store boundaries as structured array
        boundaries = valid_sf[0]['boundaries']
        boundary_min = np.array([b[1] for b in boundaries])
        boundary_max = np.array([b[2] if b[2] is not None else -1 for b in boundaries])
        npz_data['size_fractions_boundary_min'] = boundary_min
        npz_data['size_fractions_boundary_max'] = boundary_max
        
        for cat_name in category_names:
            values = []
            for d in valid_sf:
                v = d['category_fractions'][cat_name][:min_len_sf]
                if len(v) == min_len_sf:
                    values.append(v)
            if values:
                matrix = np.array(values)
                safe_key = cat_name.replace(' ', '_').replace('(', '').replace(')', '').replace('>', 'gt').replace('-', '_')
                npz_data[f'size_frac_{safe_key}_mean'] = np.mean(matrix, axis=0)
                npz_data[f'size_frac_{safe_key}_std'] = np.std(matrix, axis=0)
        
        print(f"  ✓ Size fractions computed ({len(category_names)} categories)")
    
    # Save NPZ
    print("\n" + "-" * 60)
    print("SAVING NPZ DATA")
    print("-" * 60)
    
    npz_path = f"{ensemble_dir}ensemble_structural.npz"
    np.savez_compressed(npz_path, **npz_data)
    print(f"✓ Saved structural data to {npz_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  {stats_path}")
    print(f"  {config_path}")
    print(f"  {npz_path}")
    print(f"\nDownload these files for local plotting with Ensemble_Plotting.ipynb")
    
    # Print summary metrics
    if ensemble.summary_metrics:
        print("\n" + "-" * 60)
        print("SUMMARY METRICS")
        print("-" * 60)
        m = ensemble.summary_metrics
        print(f"  Replicas: {m['n_replicas']}")
        if 'final_bonds_mean' in m:
            print(f"  Final bonds: {m['final_bonds_mean']:.1f} ± {m['final_bonds_std']:.1f}")
        if 'final_largest_fraction_mean' in m:
            print(f"  Final largest cluster: {m['final_largest_fraction_mean']*100:.1f}% ± {m['final_largest_fraction_std']*100:.1f}% of particles")
        if 'half_time_mean' in m:
            half_time_us = m['half_time_mean'] * NS_TO_US
            half_time_std_us = m['half_time_std'] * NS_TO_US
            print(f"  Half-time: {half_time_us:.2f} ± {half_time_std_us:.2f} µs")


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
