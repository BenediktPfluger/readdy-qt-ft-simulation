#!/usr/bin/env python
"""Structural descriptors of Qt clusters from Qt CENTROIDS only (+ one contact cutoff),
so the identical recipe applies to 3D vEM segmentation. PBC-aware (periodic sim box).

Descriptors (all from 3D points + d_cut):
  1. g(r)  Qt-Qt radial distribution -> ~54 nm bridging peak; d_cut = first min after peak.
  2. P(k)  coordination: # Qt neighbours within d_cut.
  3. df    fractal dimension from Rg(N):  log Rg = a + (1/df) log N.  ~3 compact, <~2.1 open.
  4. shape anisotropy  (gyration tensor): relative shape anisotropy kappa^2 (0 sphere, 1 rod)
       and aspect ratio sqrt(lambda1/lambda3)  -> the quantitative form of "ELONGATED".
  5. largest-cluster fraction (percolation order parameter): max cluster / N_Qt.
  6. rough-Qt fraction: Qt with >=1 ferritin within (rQt+rFt+tol).
"""
import os, sys, json, glob, argparse
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.expanduser("~/readdy-qt-ft-simulation"))
import readdy

QT_TYPES = {"Qt", "QtC"}
FT_TYPES = {"Ft", "FtC"}


def load_frames(h5, frac_start=0.08, n_frames=30):
    """Return (qt_pos, ft_pos, tfrac) across the trajectory. Wide pool (from frac_start on)
    so the coarsening size-range is captured for the df/shape fit; tfrac lets the caller
    restrict state descriptors (coordination, percolation) to the final agglomerated frames."""
    traj = readdy.Trajectory(h5)
    id2name = {v: k for k, v in traj.particle_types.items()}
    times, types, ids, pos = traj.read_observable_particles()
    nT = len(times)
    if nT == 0:
        return []
    lo = int(frac_start * nT)
    idx = np.unique(np.linspace(lo, nT - 1, min(n_frames, nT - lo)).astype(int))
    out = []
    for i in idx:
        tn = np.array([id2name.get(t, "?") for t in np.asarray(types[i])])
        p = np.asarray(pos[i], dtype=float)
        tfrac = i / max(1, nT - 1)
        out.append((p[np.isin(tn, list(QT_TYPES))], p[np.isin(tn, list(FT_TYPES))], tfrac))
    return out


def wrap01(p, L):
    return np.mod(p, L)


def shape_metrics(c):
    """Gyration-tensor shape of a point cloud c (N,3): (Rg, kappa2, aspect_ratio)."""
    com = c.mean(0); d = c - com
    S = (d.T @ d) / len(c)
    w = np.sort(np.linalg.eigvalsh(S))[::-1]  # lambda1>=lambda2>=lambda3
    rg2 = float(w.sum())
    if rg2 <= 0:
        return 0.0, 0.0, 1.0
    b = w[0] - 0.5 * (w[1] + w[2]); cc = w[1] - w[2]
    kappa2 = float((b * b + 0.75 * cc * cc) / (rg2 * rg2))
    aspect = float(np.sqrt(w[0] / w[2])) if w[2] > 1e-9 else float("nan")
    return float(np.sqrt(rg2)), kappa2, aspect


def rdf(qt_frames, L, rmax=200.0, dr=2.0):
    edges = np.arange(0.0, rmax + dr, dr); r = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(len(r)); nf = 0
    for qt in qt_frames:
        n = len(qt)
        if n < 2:
            continue
        w = wrap01(qt, L); t = cKDTree(w, boxsize=L)
        pr = t.query_pairs(rmax, output_type="ndarray")
        if len(pr) == 0:
            continue
        d = w[pr[:, 0]] - w[pr[:, 1]]; d -= L * np.round(d / L)
        h, _ = np.histogram(np.linalg.norm(d, axis=1), bins=edges)
        rho = n / L**3
        hist += (2.0 * h) / (n * rho * 4 * np.pi * r**2 * dr); nf += 1
    if nf:
        hist /= nf
    return r, hist


def pick_dcut(r, g):
    band = (r >= 40) & (r <= 70)
    if not band.any():
        return 78.0
    pk = int(np.argmax(np.where(band, g, 0)))
    for i in range(pk + 1, len(g) - 1):
        if g[i] <= g[i + 1] and r[i] > r[pk]:
            return float(np.clip(r[i], 62, 95))
    return 78.0


def analyze_frame(qt, L, d_cut, nmin):
    """coord array, list of (N,Rg,extent,kappa2,aspect), largest_cluster_fraction."""
    n = len(qt)
    w = wrap01(qt, L)
    pairs = cKDTree(w, boxsize=L).query_pairs(d_cut, output_type="ndarray")
    adj = [[] for _ in range(n)]
    for i, j in pairs:
        adj[i].append(j); adj[j].append(i)
    coord = np.array([len(a) for a in adj])
    seen = np.zeros(n, bool); clusters = []; largest = 1
    for s in range(n):
        if seen[s]:
            continue
        unw = {s: qt[s].copy()}; seen[s] = True; stack = [s]; nodes = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    dd = qt[v] - qt[u]; dd -= L * np.round(dd / L)
                    unw[v] = unw[u] + dd; seen[v] = True; stack.append(v); nodes.append(v)
        c = np.array([unw[k] for k in nodes])
        extent = float((c.max(0) - c.min(0)).max())
        rg, k2, asp = shape_metrics(c) if len(nodes) >= 3 else (float(np.sqrt(((c - c.mean(0))**2).sum(1).mean())), 0.0, 1.0)
        clusters.append((len(nodes), rg, extent, k2, asp))
        largest = max(largest, len(nodes))
    return coord, clusters, largest / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-dir", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/readdy_runs/descriptors"))
    ap.add_argument("--rqt", type=float, default=21.0)
    ap.add_argument("--rft", type=float, default=6.0)
    ap.add_argument("--nmin", type=int, default=4)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    L = 500.0
    cfgp = os.path.join(args.ensemble_dir, "ensemble_config.json")
    if os.path.exists(cfgp):
        try:
            L = float(json.load(open(cfgp))["box_size"][0])
        except Exception:
            pass
    h5s = sorted(glob.glob(os.path.join(args.ensemble_dir, "replica_*", "trajectory.h5")))
    if not h5s:
        print("NO TRAJECTORIES", args.ensemble_dir); sys.exit(1)

    frames = []
    for h5 in h5s:
        frames += load_frames(h5)
    qt_all = [f[0] for f in frames]
    r, g = rdf(qt_all, L)
    d_cut = pick_dcut(r, g)
    peak_r = float(r[int(np.argmax(np.where((r >= 40) & (r <= 70), g, 0)))])

    STATE = 0.70  # tfrac >= STATE => final agglomerated 'state' descriptors
    all_coord, N_list, Rg_list, K2_list, Asp_list, largest = [], [], [], [], [], []
    rough = []
    for qt, ft, tf in frames:
        if len(qt) < 2:
            continue
        coord, clusters, lf = analyze_frame(qt, L, d_cut, args.nmin)
        for (N, rg, ext, k2, asp) in clusters:            # df/shape: ALL frames (size range)
            if ext < 0.5 * L:
                N_list.append(N); Rg_list.append(rg); K2_list.append(k2); Asp_list.append(asp)
        if tf >= STATE:                                    # state descriptors: final frames only
            all_coord.append(coord); largest.append(lf)
            if len(ft):
                tqt = cKDTree(wrap01(qt, L), boxsize=L); tft = cKDTree(wrap01(ft, L), boxsize=L)
                rough.append(float(np.mean([len(p) > 0 for p in tqt.query_ball_tree(tft, args.rqt + args.rft + 8.0)])))
    if not all_coord:                                      # fallback: last quarter of frames
        for qt, ft, tf in frames[-max(1, len(frames) // 4):]:
            if len(qt) >= 2:
                coord, _, lf = analyze_frame(qt, L, d_cut, args.nmin)
                all_coord.append(coord); largest.append(lf)

    coord = np.concatenate(all_coord)
    N = np.array(N_list); Rg = np.array(Rg_list); K2 = np.array(K2_list); Asp = np.array(Asp_list)
    fit = (N >= args.nmin) & (Rg > 0)
    df = df_err = r2 = a = float("nan")
    if fit.sum() >= 5:
        x, y = np.log(N[fit]), np.log(Rg[fit]); A = np.vstack([x, np.ones_like(x)]).T
        (slope, a), *_ = np.linalg.lstsq(A, y, rcond=None)
        df = float(1 / slope) if slope > 0 else float("nan")
        yhat = A @ np.array([slope, a])
        r2 = float(1 - np.sum((y - yhat)**2) / np.sum((y - y.mean())**2))
        se = float(np.sqrt(np.sum((y - yhat)**2) / (len(x) - 2) / np.sum((x - x.mean())**2)))
        df_err = float(se / slope**2)
    shp = fit  # shape metrics for N>=nmin
    summary = {
        "label": args.label, "L_nm": L, "n_replicas": len(h5s), "n_frames": len(frames),
        "d_cut_nm": d_cut, "gr_bridging_peak_nm": peak_r,
        "fractal_dimension_df": df, "df_stderr": df_err, "df_fit_R2": r2,
        "df_n_clusters_fit": int(fit.sum()),
        "N_range_fit": [int(N[fit].min()), int(N[fit].max())] if fit.any() else None,
        "mean_Qt_coordination": float(coord.mean()) if len(coord) else None,
        "coordination_hist_k0..": np.bincount(coord).tolist() if len(coord) else [],
        "mean_shape_anisotropy_kappa2": float(np.nanmean(K2[shp])) if shp.any() else None,
        "mean_aspect_ratio": float(np.nanmean(Asp[shp])) if shp.any() else None,
        "largest_cluster_fraction": float(np.mean(largest)) if largest else None,
        "rough_Qt_fraction_mean": float(np.nanmean(rough)) if rough else None,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summary, open(args.out + "_summary.json", "w"), indent=2)
    np.savez(args.out + "_data.npz", gr_r=r, gr=g, coord=coord, cluster_N=N, cluster_Rg=Rg,
             cluster_kappa2=K2, cluster_aspect=Asp)
    print(json.dumps(summary, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(11, 8.5)); ax = ax.ravel()
        ax[0].plot(r, g); ax[0].axvline(peak_r, ls=":", c="gray"); ax[0].axvline(d_cut, ls="--", c="r")
        ax[0].set_xlim(0, 160); ax[0].set_xlabel("Qt-Qt r (nm)"); ax[0].set_ylabel("g(r)")
        ax[0].set_title(f"{args.label}  RDF peak {peak_r:.0f} nm, d_cut {d_cut:.0f} nm")
        ch = summary["coordination_hist_k0.."]; ax[1].bar(range(len(ch)), np.array(ch)/max(1,sum(ch)))
        ax[1].set_xlabel("Qt coordination k"); ax[1].set_ylabel("P(k)")
        ax[1].set_title(f"mean k={summary['mean_Qt_coordination']:.2f}")
        if fit.sum() >= 5:
            ax[2].loglog(N[fit], Rg[fit], ".", ms=4, alpha=.35)
            xs = np.array([N[fit].min(), N[fit].max()]); ax[2].loglog(xs, np.exp(a)*xs**(1/df), "r-", lw=2, label=f"df={df:.2f}"); ax[2].legend()
        ax[2].set_xlabel("cluster size N"); ax[2].set_ylabel("Rg (nm)"); ax[2].set_title("Rg vs N -> fractal dim")
        if shp.any():
            ax[3].hist(K2[shp], bins=20, range=(0, 1), alpha=.8)
            ax[3].axvline(float(np.nanmean(K2[shp])), c="r", ls="--", label=f"mean kappa2={np.nanmean(K2[shp]):.2f}\nasp={np.nanmean(Asp[shp]):.1f}")
            ax[3].legend()
        ax[3].set_xlabel("shape anisotropy kappa^2 (0 sphere, 1 rod)"); ax[3].set_ylabel("count"); ax[3].set_title("cluster elongation")
        fig.tight_layout(); fig.savefig(args.out + "_fig.png", dpi=135)
        print("FIGURE:", args.out + "_fig.png")
    except Exception as e:
        print("(no figure:", e, ")")
    print("DESCRIPTORS_DONE")


if __name__ == "__main__":
    main()
