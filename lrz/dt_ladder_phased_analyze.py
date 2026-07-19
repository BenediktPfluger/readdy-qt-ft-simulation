#!/usr/bin/env python
"""Analyze a phased agglomeration<->deagglomeration production run.

Stitches the per-phase observables onto one continuous time axis and shows the reversible
cycling: bonds / clustered-particle counts / average + largest cluster size / number of
topologies vs simulated time, with agglomeration and deagglomeration phases shaded. Writes
a summary JSON + a figure.

Usage:
    python lrz/dt_ladder_phased_analyze.py --config <phased.json> [--out <prefix>]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import readdy
import qtft
from qtft.analysis import get_bond_counts, get_cluster_statistics, get_binding_kinetics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = qtft.SimulationConfig.load_json(args.config)
    if not cfg.phases:
        sys.exit("not a phased config")
    dt = float(cfg.timestep)
    files = cfg.phase_output_files
    offsets = cfg.phase_step_offsets
    base = cfg.phase_base_dir
    out = args.out or os.path.join(base, "aggdeagg_analysis")

    # stitch observables across phases onto a continuous step axis
    T, BONDS, AVG, MAX, NCL, FB = [], [], [], [], [], []
    phase_spans = []   # (t_start_us, t_end_us, name)
    for i, (f, off) in enumerate(zip(files, offsets)):
        if not os.path.exists(f):
            print(f"[warn] missing phase file {f} — stopping at {i} phases")
            break
        traj = readdy.Trajectory(f)
        bc = get_bond_counts(f, trajectory=traj, silent=True)
        t = (np.asarray(bc["times"]) + off) * dt / 1000.0   # -> us
        T.append(t); BONDS.append(np.asarray(bc["n_bonds"]))
        try:
            cs = get_cluster_statistics(f, trajectory=traj)
            AVG.append(np.asarray(cs["avg_sizes"])); MAX.append(np.asarray(cs["max_sizes"]))
            NCL.append(np.asarray(cs["n_clusters"]))
        except Exception:
            AVG.append(np.full_like(t, np.nan)); MAX.append(np.full_like(t, np.nan)); NCL.append(np.full_like(t, np.nan))
        try:
            kin = get_binding_kinetics(f, cfg, trajectory=traj)
            fb = (np.asarray(kin["fraction_bound_qt"]) + np.asarray(kin["fraction_bound_ft"])) / 2
            FB.append(fb)
        except Exception:
            FB.append(np.full_like(t, np.nan))
        ph = cfg.phases[i]
        phase_spans.append((t[0], t[-1], ph.name))

    if not T:
        sys.exit("no phase data yet")
    t = np.concatenate(T); bonds = np.concatenate(BONDS)
    avg = np.concatenate(AVG); mx = np.concatenate(MAX); ncl = np.concatenate(NCL); fb = np.concatenate(FB)

    # per-phase bond trend (agg should RISE, deagg should FALL)
    trends = []
    for i, (f, off) in enumerate(zip(files, offsets)):
        if i >= len(BONDS): break
        b = BONDS[i]
        s, e = float(b[0]), float(b[-1])
        trends.append({"phase": i, "name": cfg.phases[i].name, "bonds_start": s,
                       "bonds_end": e, "bonds_peak": float(np.max(b)),
                       "trend": "rise" if e > s else ("fall" if e < s else "flat")})
    summary = {
        "config": args.config, "dt_ns": dt, "n_phases": len(BONDS),
        "sim_time_us": float(t[-1]),
        "bonds_max": float(np.nanmax(bonds)), "avg_cluster_max": float(np.nanmax(avg)),
        "largest_cluster_max": float(np.nanmax(mx)),
        "phase_trends": trends,
    }
    json.dump(summary, open(out + "_summary.json", "w"), indent=2)
    np.savez(out + "_series.npz", t_us=t, bonds=bonds, avg_cluster=avg, largest_cluster=mx,
             n_clusters=ncl, fraction_bound=fb)
    print(json.dumps(summary, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
        # shade agg (green) / deagg (red) phase bands
        for (t0, t1, name) in phase_spans:
            ax[0].axvspan(t0, t1, color=("#2e7d32" if "agg" == name[:3] and "deagg" not in name else "#c62828"),
                          alpha=0.06, lw=0)
            ax[1].axvspan(t0, t1, color=("#2e7d32" if name == "agglomerate" else "#c62828"), alpha=0.06, lw=0)
        ax[0].plot(t, bonds, color="#2c3e8c", lw=1.4, label="bonds")
        ax[0].set_ylabel("number of bonds"); ax[0].legend(loc="upper right", fontsize=9)
        ax[0].set_title("Agglomeration ↔ deagglomeration cycling (green = agg, red = deagg)")
        ax[1].plot(t, avg, color="#b8860b", lw=1.4, label="avg cluster size")
        ax[1].plot(t, mx, color="#8e44ad", lw=1.2, alpha=0.8, label="largest cluster")
        ax[1].set_ylabel("cluster size (particles)"); ax[1].set_xlabel("simulated time (µs)")
        ax[1].legend(loc="upper right", fontsize=9)
        fig.tight_layout(); fig.savefig(out + "_fig.png", dpi=135)
        print("FIGURE:", out + "_fig.png")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
