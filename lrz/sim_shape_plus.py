#!/usr/bin/env python
"""Per-cluster Qt-centroid SHAPE + TOPOLOGY descriptors for the DLA/sphere analysis.

The open-boundary twin (vem_descriptors.py) runs on FIB-SEM Qt centroids with the SAME
definitions, so sim and EM are directly comparable. This script is the sim side: it reads an
ensemble of ReaDDy trajectories, extracts Qt centroids per frame, builds the d_cut contact
graph, and for every connected cluster records — in addition to the canonical gyration-tensor
shape (Rg, kappa2, aspect) — two DLA falsifiers the prior study lacked:

  * cyclomatic number  mu = E - N + 1   (edges - nodes + 1 for a connected component):
      mu = 0  => strictly tree-like (what DLA / any hit-and-stick aggregate must be);
      mu > 0  => the cluster contains LOOPS, which a diffusion-limited aggregate cannot form.
  * mean internal coordination (edges per node), for context.

kappa2/aspect are scale-invariant, so the comparison to matched-N nulls is meaningful. Loops
are computed on the SAME d_cut graph used for the nulls, so the comparison is recipe-identical
and works on centroids alone (FIB-SEM-ready).

Output <out>_data.npz carries: cluster_N, cluster_kappa2, cluster_aspect, cluster_Rg,
cluster_mu (loops), cluster_edges, cluster_coord_mean, plus gr_r/gr and the pooled d_cut.

Usage:
  python lrz/sim_shape_plus.py --ensemble-dir <dir> --out <prefix> [--dcut <nm>] [--nmin 4]
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import struct_descriptors as SD   # load_frames, rdf, pick_dcut, shape_metrics, wrap01

STATE = 0.08   # load_frames frac_start default; we use ALL sampled frames for the size range


def cluster_descriptors(qt, L, d_cut, nmin, periodic=True):
    """Per connected component of the d_cut graph on Qt centroids qt (M,3).

    Returns list of dicts with N, Rg, kappa2, aspect, extent, edges, mu(loops), coord_mean.
    PBC (boxsize=L) when periodic (sim); open (no wrap) when periodic=False (EM/nulls).
    Cluster coordinates are PBC-UNWRAPPED before shape metrics so wrapped clusters are intact.
    """
    n = len(qt)
    if n < 2:
        return []
    if periodic:
        w = np.mod(qt, L); w[w >= L] = 0.0
        tree = cKDTree(w, boxsize=L)
    else:
        w = qt
        tree = cKDTree(w)
    pairs = tree.query_pairs(d_cut, output_type="ndarray")
    adj = [[] for _ in range(n)]
    Eset = [set() for _ in range(n)]
    for i, j in pairs:
        adj[i].append(j); adj[j].append(i)
    coord = np.array([len(a) for a in adj])

    seen = np.zeros(n, bool)
    out = []
    for s in range(n):
        if seen[s]:
            continue
        # BFS collecting nodes + PBC-unwrapped coordinates (like struct_descriptors.analyze_frame)
        unw = {s: qt[s].copy()}
        seen[s] = True
        stack = [s]
        nodes = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    dd = qt[v] - qt[u]
                    if periodic:
                        dd -= L * np.round(dd / L)
                    unw[v] = unw[u] + dd
                    seen[v] = True
                    stack.append(v)
                    nodes.append(v)
        N = len(nodes)
        c = np.array([unw[k] for k in nodes])
        extent = float((c.max(0) - c.min(0)).max())
        # edges WITHIN this component (count once): sum of degrees / 2 over its nodes
        nodeset = set(nodes)
        edges = 0
        for u in nodes:
            for v in adj[u]:
                if v in nodeset and v > u:
                    edges += 1
        mu = edges - N + 1   # cyclomatic number (connected component) -> loops
        if N >= 3:
            rg, k2, asp = SD.shape_metrics(c)
        else:
            rg = float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean())); k2, asp = 0.0, 1.0
        out.append({"N": N, "Rg": rg, "kappa2": k2, "aspect": asp, "extent": extent,
                    "edges": edges, "mu": mu,
                    "coord_mean": float(np.mean([coord[k] for k in nodes]))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dcut", type=float, default=None, help="fixed d_cut (nm); else from pooled g(r)")
    ap.add_argument("--nmin", type=int, default=4)
    ap.add_argument("--L", type=float, default=None, help="box edge (nm); else from ensemble_config or 500")
    ap.add_argument("--tfrac-min", type=float, default=0.0, help="only frames with tfrac >= this (time window)")
    ap.add_argument("--tfrac-max", type=float, default=1.0, help="only frames with tfrac <= this (time window)")
    args = ap.parse_args()

    L = args.L
    cfgp = os.path.join(args.ensemble_dir, "ensemble_config.json")
    if L is None and os.path.exists(cfgp):
        try:
            L = float(json.load(open(cfgp))["box_size"][0])
        except Exception:
            L = 500.0
    if L is None:
        L = 500.0

    h5s = sorted(glob.glob(os.path.join(args.ensemble_dir, "replica_*", "trajectory.h5")))
    if not h5s:
        sys.exit(f"no trajectories under {args.ensemble_dir}")

    # pool frames, derive d_cut + g(r) peak from pooled Qt (unless fixed dcut given)
    frames = []
    for h5 in h5s:
        frames += SD.load_frames(h5)
    qt_all = [f[0] for f in frames]
    r, g = SD.rdf(qt_all, L)
    peak_r = float(r[int(np.argmax(np.where((r >= 40) & (r <= 70), g, 0)))])
    d_cut = float(args.dcut) if args.dcut else SD.pick_dcut(r, g)

    N, K2, ASP, RG, MU, EDG, CO, EXT = [], [], [], [], [], [], [], []
    n_frames_used = 0
    n_excluded_percolating = 0   # clusters dropped by the extent>=0.5L cut (audit the selection)
    n_excl_by_bin = {}           # {N-decade: count} excluded, to check selection vs N
    for qt, ft, tf in frames:
        if tf < args.tfrac_min or tf > args.tfrac_max:
            continue
        n_frames_used += 1
        for cl in cluster_descriptors(qt, L, d_cut, args.nmin, periodic=True):
            if cl["extent"] >= 0.5 * L:   # PBC-spanning: shape undefined (unwrap ambiguous) -> drop
                if cl["N"] >= args.nmin:
                    n_excluded_percolating += 1
                    b = min(cl["N"], 70)
                    n_excl_by_bin[b] = n_excl_by_bin.get(b, 0) + 1
                continue
            N.append(cl["N"]); K2.append(cl["kappa2"]); ASP.append(cl["aspect"]); RG.append(cl["Rg"])
            MU.append(cl["mu"]); EDG.append(cl["edges"]); CO.append(cl["coord_mean"]); EXT.append(cl["extent"])

    N = np.array(N); K2 = np.array(K2); ASP = np.array(ASP); RG = np.array(RG)
    MU = np.array(MU); EDG = np.array(EDG); CO = np.array(CO); EXT = np.array(EXT)
    fit = N >= args.nmin
    summary = {
        "ensemble_dir": args.ensemble_dir, "L_nm": L, "n_frames": len(frames),
        "n_frames_used": n_frames_used, "tfrac_window": [args.tfrac_min, args.tfrac_max],
        "n_replicas": len(h5s), "gr_peak_nm": peak_r, "d_cut_nm": d_cut,
        "n_clusters_total": int(len(N)), "n_clusters_Nge4": int(fit.sum()),
        "n_excluded_percolating_Nge4": int(n_excluded_percolating),
        "excluded_fraction_Nge4": float(n_excluded_percolating / max(1, fit.sum() + n_excluded_percolating)),
        "N_range": [int(N[fit].min()), int(N[fit].max())] if fit.any() else None,
        "kappa2_mean": float(np.nanmean(K2[fit])) if fit.any() else None,
        "kappa2_median": float(np.nanmedian(K2[fit])) if fit.any() else None,
        "aspect_median": float(np.nanmedian(ASP[fit])) if fit.any() else None,
        "loop_fraction_Nge4": float(np.mean(MU[fit] > 0)) if fit.any() else None,
        "mean_mu_per_node_Nge4": float(np.mean(MU[fit] / np.maximum(1, N[fit]))) if fit.any() else None,
        "mean_edges_per_node_Nge4": float(np.mean(EDG[fit] / np.maximum(1, N[fit]))) if fit.any() else None,
    }
    json.dump(summary, open(args.out + "_summary.json", "w"), indent=2)
    np.savez(args.out + "_data.npz", cluster_N=N, cluster_kappa2=K2, cluster_aspect=ASP,
             cluster_Rg=RG, cluster_mu=MU, cluster_edges=EDG, cluster_coord_mean=CO,
             cluster_extent=EXT, gr_r=r, gr=g, d_cut=np.array([d_cut]), bond_nm=np.array([peak_r]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
