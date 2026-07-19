#!/usr/bin/env python
"""Structural descriptors for 3D vEM (FIB-SEM) Qt segmentation — the OPEN-BOUNDARY twin of
the simulation's struct_descriptors.py. Feed it segmented Qt particle CENTROIDS and it
computes the SAME descriptors with the SAME definitions, so sim and vEM numbers are directly
comparable.

INPUT
  --qt        Qt centroids: .npy (N,3) or .csv with columns x,y,z (nm). Optional 4th column =
              ROI/cell id (pool clusters across ROIs; density kept per-ROI).
  --ferritin  (optional) ferritin centroids -> enables the rough-Qt fraction.
  --roi       (optional) "xmin,xmax,ymin,ymax,zmin,zmax" (nm) for density + edge handling;
              else inferred from data extent (warns; give the true imaged volume if you can).

DESCRIPTORS (identical definitions to the sim; only difference: no periodic box)
  1. g(r)  Qt-Qt radial distribution -> ~50-60 nm bridging peak; d_cut = first min after peak.
  2. P(k)  coordination: # Qt neighbours within d_cut.  k=0 smooth, k>=1 rough Qt.
  3. df    fractal dimension from Rg(N):  df~3 compact, df<~2.1 open/branched.
  4. shape anisotropy (gyration tensor): kappa^2 (0 sphere, 1 rod) + aspect ratio
       sqrt(lambda1/lambda3) -> the quantitative form of "ELONGATED".
  5. largest-cluster fraction (percolation order parameter): max cluster / N_Qt.

MATCH THE SIM: Qt centroids only for the graph; let each dataset's own g(r) set d_cut; df and
shape need TRUE 3D (FIB-SEM), not 2D sections; report stimulated AND unstimulated (the
DIFFERENCE is the signal); compare shape/topology, never absolute counts/sizes/timescale.
"""
import os, sys, json, argparse
import numpy as np
from scipy.spatial import cKDTree


def load_xyz(path):
    a = np.load(path) if path.endswith(".npy") else np.loadtxt(path, delimiter=",", ndmin=2)
    a = np.asarray(a, float)
    roi = a[:, 3].astype(int) if a.shape[1] >= 4 else np.zeros(len(a), int)
    return a[:, :3], roi


def shape_metrics(c):
    com = c.mean(0); d = c - com
    w = np.sort(np.linalg.eigvalsh((d.T @ d) / len(c)))[::-1]
    rg2 = float(w.sum())
    if rg2 <= 0:
        return 0.0, 0.0, 1.0
    b = w[0] - 0.5 * (w[1] + w[2]); cc = w[1] - w[2]
    asp = float(np.sqrt(w[0] / w[2])) if w[2] > 1e-9 else float("nan")
    return float(np.sqrt(rg2)), float((b * b + 0.75 * cc * cc) / (rg2 * rg2)), asp


def rdf_open(xyz, roi_vol, rmax=200.0, dr=2.0):
    edges = np.arange(0.0, rmax + dr, dr); r = 0.5 * (edges[:-1] + edges[1:])
    n = len(xyz); pr = cKDTree(xyz).query_pairs(rmax, output_type="ndarray")
    if len(pr) == 0:
        return r, np.zeros_like(r)
    d = np.linalg.norm(xyz[pr[:, 0]] - xyz[pr[:, 1]], axis=1)
    h, _ = np.histogram(d, bins=edges)
    return r, (2.0 * h) / (n * (n / roi_vol) * 4 * np.pi * r**2 * dr)


def pick_dcut(r, g):
    band = (r >= 40) & (r <= 70)
    if not band.any():
        return 78.0
    pk = int(np.argmax(np.where(band, g, 0)))
    for i in range(pk + 1, len(g) - 1):
        if g[i] <= g[i + 1] and r[i] > r[pk]:
            return float(np.clip(r[i], 62, 95))
    return 78.0


def components(xyz, d_cut):
    n = len(xyz)
    pairs = cKDTree(xyz).query_pairs(d_cut, output_type="ndarray")
    adj = [[] for _ in range(n)]
    for i, j in pairs:
        adj[i].append(j); adj[j].append(i)
    coord = np.array([len(a) for a in adj])
    seen = np.zeros(n, bool); comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]; seen[s] = True; nodes = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True; stack.append(v); nodes.append(v)
        comps.append(nodes)
    return coord, comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qt", required=True)
    ap.add_argument("--ferritin", default=None)
    ap.add_argument("--roi", default=None, help="xmin,xmax,ymin,ymax,zmin,zmax (nm)")
    ap.add_argument("--out", default="vem_descriptors")
    ap.add_argument("--nmin", type=int, default=4)
    ap.add_argument("--rqt", type=float, default=20.0)
    ap.add_argument("--rft", type=float, default=6.0)
    args = ap.parse_args()

    xyz, roi = load_xyz(args.qt)
    if args.roi:
        b = np.array([float(x) for x in args.roi.split(",")]).reshape(3, 2); lo, hi = b[:, 0], b[:, 1]
    else:
        lo, hi = xyz.min(0), xyz.max(0)
        print("WARNING: ROI inferred from data extent; pass --roi for correct density/edges.")
    vol = float(np.prod(hi - lo))
    r, g = rdf_open(xyz, vol); d_cut = pick_dcut(r, g)
    peak_r = float(r[int(np.argmax(np.where((r >= 40) & (r <= 70), g, 0)))])
    print(f"N_Qt={len(xyz)} vol={vol:.3e} nm^3 g(r)peak~{peak_r:.1f} d_cut={d_cut:.1f}")

    all_coord, N_list, Rg_list, K2_list, Asp_list, largest = [], [], [], [], [], []
    for rid in np.unique(roi):
        sub = xyz[roi == rid]
        if len(sub) < 2:
            continue
        coord, comps = components(sub, d_cut)
        all_coord.append(coord)
        largest.append(max((len(n) for n in comps), default=1) / max(1, len(sub)))
        slo, shi = (lo, hi) if args.roi else (sub.min(0), sub.max(0))
        for nodes in comps:
            c = sub[nodes]
            if np.any(c.min(0) - slo < d_cut) or np.any(shi - c.max(0) < d_cut):
                if len(nodes) >= args.nmin:
                    continue  # possibly field-of-view-truncated -> drop from fit set
            if len(nodes) >= 3:
                rg, k2, asp = shape_metrics(c)
            else:
                rg = float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean())); k2, asp = 0.0, 1.0
            N_list.append(len(nodes)); Rg_list.append(rg); K2_list.append(k2); Asp_list.append(asp)

    coord = np.concatenate(all_coord) if all_coord else np.array([0])
    N = np.array(N_list); Rg = np.array(Rg_list); K2 = np.array(K2_list); Asp = np.array(Asp_list)
    fit = (N >= args.nmin) & (Rg > 0)
    df = df_err = r2 = a = float("nan")
    if fit.sum() >= 5:
        x, y = np.log(N[fit]), np.log(Rg[fit]); A = np.vstack([x, np.ones_like(x)]).T
        (slope, a), *_ = np.linalg.lstsq(A, y, rcond=None)
        df = float(1 / slope) if slope > 0 else float("nan")
        yhat = A @ np.array([slope, a])
        r2 = float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))
        se = float(np.sqrt(np.sum((y - yhat) ** 2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2)))
        df_err = float(se / slope**2)

    rough = None
    if args.ferritin:
        ft, _ = load_xyz(args.ferritin)
        pairs = cKDTree(xyz).query_ball_tree(cKDTree(ft), args.rqt + args.rft + 8.0)
        rough = float(np.mean([len(p) > 0 for p in pairs]))

    summary = {
        "N_Qt": int(len(xyz)), "n_ROI": int(len(np.unique(roi))), "ROI_volume_nm3": vol,
        "gr_bridging_peak_nm": peak_r, "d_cut_nm": d_cut,
        "mean_Qt_coordination": float(coord.mean()), "rough_Qt_fraction_k>=1": float(np.mean(coord >= 1)),
        "coordination_hist_k0..": np.bincount(coord).tolist(),
        "fractal_dimension_df": df, "df_stderr": df_err, "df_fit_R2": r2, "df_n_clusters_fit": int(fit.sum()),
        "N_range_fit": [int(N[fit].min()), int(N[fit].max())] if fit.any() else None,
        "mean_shape_anisotropy_kappa2": float(np.nanmean(K2[fit])) if fit.any() else None,
        "mean_aspect_ratio": float(np.nanmean(Asp[fit])) if fit.any() else None,
        "largest_cluster_fraction": float(np.mean(largest)) if largest else None,
        "rough_Qt_fraction_by_ferritin": rough,
    }
    json.dump(summary, open(args.out + "_summary.json", "w"), indent=2)
    np.savez(args.out + "_data.npz", gr_r=r, gr=g, coord=coord, cluster_N=N, cluster_Rg=Rg,
             cluster_kappa2=K2, cluster_aspect=Asp)
    print(json.dumps(summary, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(11, 8.5)); ax = ax.ravel()
        ax[0].plot(r, g); ax[0].axvline(peak_r, ls=":", c="gray"); ax[0].axvline(d_cut, ls="--", c="r")
        ax[0].set_xlim(0, 160); ax[0].set_xlabel("Qt-Qt distance (nm)"); ax[0].set_ylabel("g(r)"); ax[0].set_title(f"RDF peak {peak_r:.0f} nm")
        ch = np.bincount(coord); ax[1].bar(np.arange(len(ch)), ch / ch.sum())
        ax[1].set_xlabel("Qt coordination k"); ax[1].set_ylabel("P(k)"); ax[1].set_title(f"mean k={coord.mean():.2f}, rough={np.mean(coord>=1):.2f}")
        if fit.sum() >= 5:
            ax[2].loglog(N[fit], Rg[fit], ".", ms=4, alpha=.4)
            xs = np.array([N[fit].min(), N[fit].max()]); ax[2].loglog(xs, np.exp(a) * xs ** (1 / df), "r-", lw=2, label=f"df={df:.2f}"); ax[2].legend()
        ax[2].set_xlabel("cluster size N"); ax[2].set_ylabel("Rg (nm)"); ax[2].set_title("Rg vs N")
        if fit.any():
            ax[3].hist(K2[fit], bins=20, range=(0, 1)); ax[3].axvline(float(np.nanmean(K2[fit])), c="r", ls="--", label=f"kappa2={np.nanmean(K2[fit]):.2f}, asp={np.nanmean(Asp[fit]):.1f}"); ax[3].legend()
        ax[3].set_xlabel("shape anisotropy kappa^2"); ax[3].set_ylabel("count"); ax[3].set_title("cluster elongation")
        fig.tight_layout(); fig.savefig(args.out + "_fig.png", dpi=135); print("FIGURE:", args.out + "_fig.png")
    except Exception as e:
        print("(no figure:", e, ")")


if __name__ == "__main__":
    main()
