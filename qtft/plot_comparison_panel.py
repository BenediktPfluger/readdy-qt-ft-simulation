"""
Qt-Ft cross-ensemble comparison thesis figure panel.

Composes a single 3x3 figure that overlays multiple ensembles (one colored line per
ensemble, optional ±1 SD bands) for the most relevant comparison plots. This is the
cross-ensemble counterpart to ``qtft.plot_ensemble_panel`` (which curates a single
ensemble). It reuses the comparison drawing conventions/constants from ``qtft.plotting``
(so the existing public plotting functions stay untouched).

Layout (3x3):
    Row 1: Potential Energy      | Pressure                        | Number of Bonds
    Row 2: Individual Topologies | Average Cluster Size            | Largest Cluster Size
    Row 3: Coordination Number   | Mean Cluster Composition         | Mean Radius of Gyration

Rows 1-2 read per-ensemble basic statistics from ``ens['stats']`` (keys ``{stat}_mean`` /
``{stat}_std``, time axis ``ens['times_us']`` in µs). Row 3 reads structural data from
``ens['structural']`` (step-indexed, converted to µs via ``_steps_to_us``). The structural
data is only present when the comparison is built with
``qtft.comparison.compare_ensembles`` (the live load path), not with ``load_comparison_data``.

Usage:
    import qtft.comparison as ae
    import qtft.plot_comparison_panel as cpanel
    comparison = ae.compare_ensembles({"Label A": dir_a, "Label B": dir_b})
    cpanel.plot_comparison_panel(comparison,
                                 save_path_base="path/to/comparison_panel")
"""
from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .config import _steps_to_us
from .plotting import (
    COMPARISON_COLORS,
    _get_show_bands_default,
    FONTSIZE_TITLE,
    FONTSIZE_LABEL,
    FONTSIZE_LEGEND,
    FONTSIZE_TICK,
)


def plot_comparison_panel(
    comparison: dict,
    *,
    show_bands: Optional[bool] = None,
    figsize: Tuple[float, float] = (18, 17),
    save_path_base: Optional[str] = None,
) -> plt.Figure:
    """
    Build the cross-ensemble comparison thesis panel (3x3 grid) and optionally save
    SVG + PNG.

    Parameters
    ----------
    comparison : dict
        Comparison data structure from ``qtft.comparison.compare_ensembles`` (must include
        per-ensemble ``structural`` data for Row 3).
    show_bands : bool, optional
        Show ±1 SD bands. Default (None): True for ≤3 ensembles, False otherwise.
    figsize : tuple
        Figure size (default 18x17 for the 3x3 grid).
    save_path_base : str, optional
        If given, save ``{base}.svg`` and ``{base}.png``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    print("\nGenerating ensemble comparison thesis panel...")

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    n_ensembles = comparison['n_ensembles']
    show_bands = _get_show_bands_default(n_ensembles, show_bands)
    labels = comparison['labels']

    # --- local axis-level helpers (mirror the closures in qtft.plotting) ---
    def plot_stat(ax, stat_key, ylabel, title, legend_loc='best'):
        """Overlay a basic-stats time series (ens['stats']) for each ensemble."""
        has_data = False
        for i, label in enumerate(labels):
            ens = comparison['ensembles'][label]
            times_us = ens['times_us']
            mean_key = f'{stat_key}_mean'
            std_key = f'{stat_key}_std'

            if mean_key not in ens['stats']:
                continue

            mean_vals = ens['stats'][mean_key]
            std_vals = ens['stats'].get(std_key, np.zeros_like(mean_vals))
            color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]

            ax.plot(times_us, mean_vals, color=color, linewidth=2, label=label)
            if show_bands and len(std_vals) == len(mean_vals):
                ax.fill_between(times_us, mean_vals - std_vals, mean_vals + std_vals,
                                color=color, alpha=0.2)
            has_data = True

        if not has_data:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color='gray')

        ax.set_xlabel("Time (µs)", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        if has_data:
            ax.legend(loc=legend_loc, fontsize=FONTSIZE_LEGEND)

    def plot_struct(ax, time_key, mean_key, std_key, ylabel, title, legend_loc='best'):
        """Overlay a structural time series (ens['structural']) for each ensemble."""
        has_data = False
        for i, label in enumerate(labels):
            ens = comparison['ensembles'][label]
            structural = ens.get('structural', {})

            if time_key not in structural or mean_key not in structural:
                continue

            times_steps = np.asarray(structural[time_key])
            timestep = ens.get('timestep', 0.001)
            times_us = _steps_to_us(times_steps, timestep)

            mean_vals = np.asarray(structural[mean_key])
            std_vals = np.asarray(structural.get(std_key, np.zeros_like(mean_vals)))

            # Guard against mismatched lengths (take the common prefix)
            min_len = min(len(times_us), len(mean_vals))
            times_us = times_us[:min_len]
            mean_vals = mean_vals[:min_len]
            std_vals = std_vals[:min_len]

            color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
            ax.plot(times_us, mean_vals, color=color, linewidth=2, label=label)
            if show_bands:
                ax.fill_between(times_us, mean_vals - std_vals, mean_vals + std_vals,
                                color=color, alpha=0.2)
            has_data = True

        if not has_data:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color='gray')

        ax.set_xlabel("Time (µs)", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        if has_data:
            ax.legend(loc=legend_loc, fontsize=FONTSIZE_LEGEND)

    def plot_coord_fused(ax, legend_loc='lower right'):
        """Overlay Qt (solid) and Ft (dashed) coordination per ensemble in one axes."""
        has_data = False
        for i, label in enumerate(labels):
            ens = comparison['ensembles'][label]
            structural = ens.get('structural', {})
            if 'contacts_times' not in structural:
                continue

            times_steps = np.asarray(structural['contacts_times'])
            timestep = ens.get('timestep', 0.001)
            times_us = _steps_to_us(times_steps, timestep)
            color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]

            for mean_key, std_key, ls in (
                ('mean_coord_qt_mean', 'mean_coord_qt_std', '-'),
                ('mean_coord_ft_mean', 'mean_coord_ft_std', '--'),
            ):
                if mean_key not in structural:
                    continue
                mean_vals = np.asarray(structural[mean_key])
                std_vals = np.asarray(structural.get(std_key, np.zeros_like(mean_vals)))
                min_len = min(len(times_us), len(mean_vals))
                t = times_us[:min_len]
                m = mean_vals[:min_len]
                s = std_vals[:min_len]
                # Only the solid (Qt) line carries the ensemble label
                lbl = label if ls == '-' else '_nolegend_'
                ax.plot(t, m, color=color, linewidth=2, linestyle=ls, label=lbl)
                if show_bands:
                    ax.fill_between(t, m - s, m + s, color=color, alpha=0.2)
                has_data = True

        if not has_data:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color='gray')

        ax.set_xlabel("Time (µs)", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Mean Coordination", fontsize=FONTSIZE_LABEL)
        ax.set_title("Coordination Number", fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        if has_data:
            # Ensemble colours (solid lines) + a linestyle key (Qt solid / Ft dashed)
            ens_handles, ens_labels = ax.get_legend_handles_labels()
            style_handles = [
                Line2D([0], [0], color='black', linestyle='-', linewidth=2),
                Line2D([0], [0], color='black', linestyle='--', linewidth=2),
            ]
            ax.legend(ens_handles + style_handles, ens_labels + ['Qt', 'Ft'],
                      loc=legend_loc, fontsize=FONTSIZE_LEGEND)

    # ======================================================================
    # Row 1: Energy, Pressure, Number of Bonds
    # ======================================================================
    plot_stat(fig.add_subplot(gs[0, 0]), 'energy', "Energy (kJ/mol)", "Potential Energy",
              legend_loc='upper right')
    plot_stat(fig.add_subplot(gs[0, 1]), 'pressure', "Pressure (kJ/(mol·nm³))", "Pressure",
              legend_loc='upper right')
    plot_stat(fig.add_subplot(gs[0, 2]), 'bonds', "Number of Bonds", "Number of Bonds",
              legend_loc='lower right')

    # ======================================================================
    # Row 2: Individual Topologies, Average Cluster Size, Largest Cluster Size
    # ======================================================================
    plot_stat(fig.add_subplot(gs[1, 0]), 'n_clusters', "Number of Individual Topologies",
              "Number of Individual Topologies", legend_loc='upper right')
    plot_stat(fig.add_subplot(gs[1, 1]), 'avg_cluster', "Average Size (particles)",
              "Average Cluster Size", legend_loc='lower right')
    plot_stat(fig.add_subplot(gs[1, 2]), 'largest_cluster', "Cluster Size (particles)",
              "Largest Cluster Size", legend_loc='lower right')

    # ======================================================================
    # Row 3: Coordination Number, Mean Cluster Composition, Mean Radius of Gyration
    # ======================================================================
    plot_coord_fused(fig.add_subplot(gs[2, 0]), legend_loc='lower right')

    ax_comp = fig.add_subplot(gs[2, 1])
    plot_struct(ax_comp, 'composition_times', 'mean_composition_mean',
                'mean_composition_std', "Mean Qt Fraction", "Mean Cluster Composition",
                legend_loc='lower right')
    ax_comp.set_ylim([0, 1])

    plot_struct(fig.add_subplot(gs[2, 2]), 'morphology_times', 'mean_rg_mean',
                'mean_rg_std', "Mean Rg (nm)", "Mean Radius of Gyration",
                legend_loc='upper left')

    if save_path_base:
        svg_path = f"{save_path_base}.svg"
        png_path = f"{save_path_base}.png"
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"✓ Saved panel to {svg_path}")
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
        print(f"✓ Saved panel to {png_path}")

    return fig
