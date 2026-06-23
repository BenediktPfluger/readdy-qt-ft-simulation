"""
Qt-Ft thesis figure panel.

Composes a single 4x3 figure that combines the most relevant ensemble plots for a
thesis, reusing the low-level drawing primitives from ``qtft.plotting`` (so the
existing plotting functions stay untouched).

Layout (4x3, last cell blank):
    Row 1: Energy                | Pressure                       | Number of Bonds
    Row 2: Particle Counts       | Number of Individual Topologies | Average Cluster Size
    Row 3: Largest Cluster Size  | Particles by Size Category      | Mean Radius of Gyration
    Row 4: Coordination Number   | Mean Cluster Composition        | (empty)

Usage:
    import qtft.analysis as analysis
    import qtft.plot_ensemble_panel as panel
    stats, structural, config = analysis.load_ensemble_data(ensemble_dir)
    panel.plot_ensemble_panel(stats, structural, config,
                              save_path_base="path/to/ensemble_panel")
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .config import _steps_to_us
from .plotting import (
    _ensemble_plot_with_band,
    _ensemble_show_no_data,
    FONTSIZE_TITLE,
    FONTSIZE_LABEL,
    FONTSIZE_LEGEND,
)


def plot_ensemble_panel(
    stats: Dict,
    structural: Dict,
    config: Dict,
    *,
    show_individual: bool = False,
    individual_alpha: float = 0.3,
    figsize: Tuple[float, float] = (18, 18),
    save_path_base: Optional[str] = None,
) -> plt.Figure:
    """
    Build the thesis ensemble panel (4x3 grid) and optionally save SVG + PNG.

    Parameters
    ----------
    stats : dict
        Basic ensemble statistics (means/stds + per-replica traces).
    structural : dict
        Structural ensemble data (morphology, contacts, composition, size fractions).
    config : dict
        Configuration dictionary (for timestep / time-axis conversion).
    show_individual : bool
        If True, overlay faint per-replica traces on the band plots.
    individual_alpha : float
        Transparency for individual traces.
    figsize : tuple
        Figure size (default 18x18 for the 4x3 grid).
    save_path_base : str, optional
        If given, save ``{base}.svg`` and ``{base}.png``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    print("\nGenerating ensemble thesis panel...")

    fig, axes = plt.subplots(4, 3, figsize=figsize)

    timestep = config.get('timestep', 1e-4)
    times_us = _steps_to_us(np.asarray(stats['times']), timestep)
    n_replicas = stats.get('n_replicas', 1)
    time_label = "Time (µs)"

    # --- local helpers (mirror the inline helpers in qtft.plotting) ---
    def all_trace(key):
        all_key = f'{key}_all'
        if structural is not None and all_key in structural:
            return structural[all_key]
        if all_key in stats:
            return stats[all_key]
        return None

    def struct_ts(time_key, mean_key, std_key, all_key=None):
        times = structural.get(time_key) if structural else None
        mean = structural.get(mean_key) if structural else None
        std = structural.get(std_key) if structural else None
        all_data = structural.get(all_key) if (structural and all_key) else None
        if times is not None:
            times = _steps_to_us(np.asarray(times), timestep)
        if mean is not None:
            mean = np.asarray(mean)
        if std is not None:
            std = np.asarray(std)
        return times, mean, std, all_data

    def simple_band(ax, mean_key, std_key, color, title, ylabel,
                    legend_loc='best'):
        if mean_key in stats:
            if _ensemble_plot_with_band(ax, times_us, stats[mean_key], stats[std_key],
                                        color, n_replicas, all_trace(mean_key[:-5]),
                                        show_individual, individual_alpha):
                ax.legend(loc=legend_loc, fontsize=FONTSIZE_LEGEND)
        else:
            _ensemble_show_no_data(ax)
        ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # ======================================================================
    # Row 1: Energy, Pressure, Number of Bonds
    # ======================================================================
    simple_band(axes[0, 0], 'energy_mean', 'energy_std', 'tab:red',
                "Total Energy", "Energy (kJ/mol)", legend_loc='upper right')
    simple_band(axes[0, 1], 'pressure_mean', 'pressure_std', 'tab:green',
                "Pressure", "Pressure (kJ/mol/nm³)", legend_loc='upper right')
    simple_band(axes[0, 2], 'bonds_mean', 'bonds_std', 'tab:blue',
                "Number of Bonds", "Number of Bonds", legend_loc='lower right')

    # ======================================================================
    # Row 2: Particle Counts, Number of Individual Topologies, Average Cluster Size
    # ======================================================================
    # Particle counts (multi-line)
    ax = axes[1, 0]
    particle_colors = {'qt': 'blue', 'ft': 'red', 'qtc': 'darkblue', 'ftc': 'darkred'}
    particle_labels = {'qt': 'Qt (free)', 'ft': 'Ft (free)', 'qtc': 'QtC', 'ftc': 'FtC'}
    has_particle_data = False
    for ptype in ['qt', 'ft', 'qtc', 'ftc']:
        key_mean = f'{ptype}_count_mean'
        key_std = f'{ptype}_count_std'
        if key_mean in stats:
            mean = np.asarray(stats[key_mean])
            std = np.asarray(stats[key_std])
            ax.plot(times_us, mean, color=particle_colors[ptype],
                    linewidth=2, label=particle_labels[ptype])
            ax.fill_between(times_us, mean - std, mean + std,
                            color=particle_colors[ptype], alpha=0.2)
            has_particle_data = True
    if has_particle_data:
        ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
    ax.set_title("Particle Counts", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)

    simple_band(axes[1, 1], 'n_clusters_mean', 'n_clusters_std', 'tab:purple',
                "Number of Individual Topologies", "Number of Individual Topologies",
                legend_loc='upper right')
    simple_band(axes[1, 2], 'avg_cluster_mean', 'avg_cluster_std', 'tab:olive',
                "Average Cluster Size", "Average Size (particles)", legend_loc='lower right')

    # ======================================================================
    # Row 3: Largest Cluster Size, Particles by Size Category, Mean Radius of Gyration
    # ======================================================================
    simple_band(axes[2, 0], 'largest_cluster_mean', 'largest_cluster_std', 'tab:orange',
                "Largest Cluster Size", "Cluster Size (particles)", legend_loc='lower right')

    # Particles by Size Category (stacked area; no std bands; thesis title without descriptor)
    ax = axes[2, 1]
    if structural and 'size_fractions_times' in structural and \
            'size_fractions_category_names' in structural:
        sc_times = _steps_to_us(np.asarray(structural['size_fractions_times']), timestep)
        category_names = list(structural['size_fractions_category_names'])
        mean_fractions = []
        for cat_name in category_names:
            safe_key = (cat_name.replace(' ', '_').replace('(', '').replace(')', '')
                        .replace('>', 'gt').replace('-', '_'))
            mean_key = f'size_frac_{safe_key}_mean'
            if mean_key in structural:
                mean_fractions.append(np.asarray(structural[mean_key]))
            else:
                mean_fractions.append(np.zeros(len(sc_times)))
        colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
        ax.stackplot(sc_times, *mean_fractions, labels=category_names,
                     colors=colors[:len(category_names)], alpha=0.8)
        ax.set_ylim([0, 1])
        ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Fraction", fontsize=FONTSIZE_LABEL)
    ax.set_title("Particles by Size Category", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Mean Radius of Gyration
    ax = axes[2, 2]
    times, mean, std, all_data = struct_ts(
        'morphology_times', 'mean_rg_mean', 'mean_rg_std', 'mean_rg_all')
    if times is not None and mean is not None:
        if _ensemble_plot_with_band(ax, times, mean, std, 'tab:blue', n_replicas,
                                    all_data, show_individual, individual_alpha):
            ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Mean Rg (nm)", fontsize=FONTSIZE_LABEL)
    ax.set_title("Mean Radius of Gyration", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # ======================================================================
    # Row 4: Coordination Number, Mean Cluster Composition, (empty)
    # ======================================================================
    # Coordination Number (Qt + Ft fused)
    ax = axes[3, 0]
    t_qt, m_qt, s_qt, all_qt = struct_ts(
        'contacts_times', 'mean_coord_qt_mean', 'mean_coord_qt_std', 'mean_coord_qt_all')
    t_ft, m_ft, s_ft, all_ft = struct_ts(
        'contacts_times', 'mean_coord_ft_mean', 'mean_coord_ft_std', 'mean_coord_ft_all')
    coord_plotted = False
    if t_qt is not None and m_qt is not None:
        coord_plotted |= _ensemble_plot_with_band(
            ax, t_qt, m_qt, s_qt, 'tab:blue', n_replicas,
            all_qt, show_individual, individual_alpha, label='Qt')
    if t_ft is not None and m_ft is not None:
        coord_plotted |= _ensemble_plot_with_band(
            ax, t_ft, m_ft, s_ft, 'tab:red', n_replicas,
            all_ft, show_individual, individual_alpha, label='Ft')
    if coord_plotted:
        ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Mean Coordination", fontsize=FONTSIZE_LABEL)
    ax.set_title("Coordination Number", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Mean Cluster Composition (Equal-mix line kept but not in the legend)
    ax = axes[3, 1]
    times, mean, std, all_data = struct_ts(
        'composition_times', 'mean_composition_mean', 'mean_composition_std',
        'mean_composition_all')
    if times is not None and mean is not None:
        if _ensemble_plot_with_band(ax, times, mean, std, 'tab:blue', n_replicas,
                                    all_data, show_individual, individual_alpha):
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
            ax.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Mean Qt Fraction", fontsize=FONTSIZE_LABEL)
    ax.set_ylim([0, 1])
    ax.set_title("Mean Cluster Composition", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Empty cell
    axes[3, 2].axis('off')

    plt.tight_layout()

    if save_path_base:
        svg_path = f"{save_path_base}.svg"
        png_path = f"{save_path_base}.png"
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"✓ Saved panel to {svg_path}")
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
        print(f"✓ Saved panel to {png_path}")

    return fig
