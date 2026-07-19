#!/usr/bin/env python
"""Matched-N comparison of a sim Qt-cluster ensemble vs 4 geometric nulls, on shape AND loops.

Answers two questions with honest, matched-N statistics:
  (1) Degree of deviation from SPHERICAL: sim kappa2 vs the compact/spherical null.
  (2) Any evidence NOT consistent with DLA: sim kappa2 / loop content / df vs the DLA null
      (and the reaction-limited RLCA null, the "not diffusion-limited" alternative).

For each metric and each N-bin: Mann-Whitney U (two-sided) + Cliff's delta effect size, plus a
matched-N Wilcoxon signed-rank headline (subtract the null's mean-at-that-N from each sim cluster).
Fractal dimension df = 1/slope of log Rg vs log N (context). Everything uses the SAME shape recipe
and the SAME d_cut, so sim and nulls are recipe-identical; the sim side is Qt centroids only, so
FIB-SEM Qt centroids drop into the identical pipeline.

Usage:
  python lrz/analyze_shape_dla.py --sim-npz <sim>_data.npz --null-npz <nulls>_data.npz --out <prefix>
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

NULLS = ["compact", "random", "dla", "rlca"]
LABEL = {"compact": "compact/spherical", "random": "random", "dla": "DLA", "rlca": "RLCA"}
BINS = [(4, 5), (6, 7), (8, 10), (11, 15), (16, 22), (23, 35), (36, 70)]


def cliffs_delta_p(a, b):
    if len(a) < 3 or len(b) < 3:
        return float("nan"), float("nan")
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    return 2.0 * U / (len(a) * len(b)) - 1.0, float(p)   # delta>0 => sim (a) larger


def per_N_mean(N, val):
    out = {}
    for u in np.unique(N):
        m = N == u
        out[int(u)] = float(np.mean(val[m]))
    return out


def matched_headline(sN, sV, nN, nV):
    """Wilcoxon signed-rank on per-sim-cluster residual (sim - null_mean_at_that_N)."""
    nmean = per_N_mean(nN, nV)
    resid = np.array([sV[i] - nmean.get(int(sN[i]), np.nan) for i in range(len(sN))])
    resid = resid[np.isfinite(resid)]
    if len(resid) < 5 or np.allclose(resid, 0):
        return {"median_residual": float(np.median(resid)) if len(resid) else None,
                "frac_above": float(np.mean(resid > 0)) if len(resid) else None,
                "wilcoxon_greater_p": None, "n": int(len(resid))}
    try:
        W, p = wilcoxon(resid, alternative="greater")
    except Exception:
        p = None
    return {"median_residual": float(np.median(resid)), "frac_above": float(np.mean(resid > 0)),
            "wilcoxon_greater_p": (float(p) if p is not None else None), "n": int(len(resid))}


def df_fit(N, Rg, nmin=4):
    m = (N >= nmin) & (Rg > 0) & np.isfinite(Rg)
    if m.sum() < 5:
        return float("nan"), float("nan")
    x, y = np.log(N[m]), np.log(Rg[m])
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, a), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, a])
    r2 = float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))
    return (float(1 / slope) if slope > 0 else float("nan")), r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-npz", required=True)
    ap.add_argument("--null-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nmin", type=int, default=4)
    ap.add_argument("--label", default="")
    ap.add_argument("--extent-cap", type=float, default=250.0,
                    help="apply the SAME extent<cap (=0.5L) selection to the nulls as the sim, for a fair "
                         "matched comparison; 0 disables")
    args = ap.parse_args()

    s = np.load(args.sim_npz)
    sN = s["cluster_N"].astype(int); sK = s["cluster_kappa2"]; sR = s["cluster_Rg"]
    # cluster_mu (loops) is absent in the vem_descriptors EM npz -> tolerate (skip loop test)
    sMu = s["cluster_mu"].astype(float) if "cluster_mu" in s.files else np.zeros(len(sN))
    has_loops = "cluster_mu" in s.files
    keep = sN >= args.nmin
    sN, sK, sR, sMu = sN[keep], sK[keep], sR[keep], sMu[keep]
    sMuN = sMu / np.maximum(1, sN)            # loops per node (density-normalised)

    nz = np.load(args.null_npz)
    nulls = {k: {"N": nz[f"{k}_N"].astype(int), "kappa2": nz[f"{k}_kappa2"],
                 "Rg": nz[f"{k}_Rg"], "mu": nz[f"{k}_mu"].astype(float)} for k in NULLS}
    # SYMMETRIC extent cut: the sim drops percolating clusters (extent>=0.5L); apply the SAME
    # absolute extent cap to the nulls so the matched-N comparison is not biased by one-sided selection.
    null_excl = {}
    for k in NULLS:
        ext = nz[f"{k}_extent"] if f"{k}_extent" in nz.files else None
        if args.extent_cap and ext is not None:
            keepn = ext < args.extent_cap
            null_excl[k] = float(np.mean(~keepn))
            for key in ("N", "kappa2", "Rg", "mu"):
                nulls[k][key] = nulls[k][key][keepn]
        else:
            null_excl[k] = 0.0
        nulls[k]["muN"] = nulls[k]["mu"] / np.maximum(1, nulls[k]["N"])

    # --- pooled + df ---
    df_sim, r2_sim = df_fit(sN, sR)
    report = {
        "label": args.label, "n_sim_clusters": int(len(sN)),
        "extent_cap_nm": args.extent_cap, "null_excluded_fraction_by_extent": null_excl,
        "sim_has_loops": bool(has_loops),
        "sim_N_range": [int(sN.min()), int(sN.max())],
        "sim_kappa2_mean": float(np.nanmean(sK)), "sim_kappa2_median": float(np.nanmedian(sK)),
        "sim_loop_fraction": float(np.mean(sMu > 0)), "sim_mean_loops_per_node": float(np.mean(sMuN)),
        "df": {"sim": {"df": df_sim, "r2": r2_sim}},
        "per_bin": [], "matched_N_headline_kappa2": {}, "matched_N_headline_loops": {},
    }
    for k in NULLS:
        d, r = df_fit(nulls[k]["N"], nulls[k]["Rg"])
        report["df"][k] = {"df": d, "r2": r,
                           "loop_fraction": float(np.mean(nulls[k]["mu"] > 0)),
                           "kappa2_median": float(np.nanmedian(nulls[k]["kappa2"]))}

    # --- matched-N headlines (kappa2 and loops-per-node) ---
    for k in NULLS:
        report["matched_N_headline_kappa2"][k] = matched_headline(sN, sK, nulls[k]["N"], nulls[k]["kappa2"])
        report["matched_N_headline_loops"][k] = matched_headline(sN, sMuN, nulls[k]["N"], nulls[k]["muN"])

    # --- per-bin effect sizes (kappa2 + loops) ---
    for lo, hi in BINS:
        sm = (sN >= lo) & (sN <= hi)
        if sm.sum() < 3:
            continue
        row = {"bin": f"{lo}-{hi}", "n_sim": int(sm.sum()),
               "sim_kappa2_median": float(np.nanmedian(sK[sm])),
               "sim_loops_per_node_median": float(np.nanmedian(sMuN[sm])), "nulls": {}}
        for k in NULLS:
            nm = (nulls[k]["N"] >= lo) & (nulls[k]["N"] <= hi)
            dk, pk = cliffs_delta_p(sK[sm], nulls[k]["kappa2"][nm])
            dl, pl = cliffs_delta_p(sMuN[sm], nulls[k]["muN"][nm])
            row["nulls"][k] = {"kappa2_cliffs_delta": dk, "kappa2_p": pk,
                               "kappa2_null_median": float(np.nanmedian(nulls[k]["kappa2"][nm])),
                               "loops_cliffs_delta": dl, "loops_p": pl,
                               "loops_null_median": float(np.nanmedian(nulls[k]["muN"][nm]))}
        report["per_bin"].append(row)

    json.dump(report, open(args.out + "_summary.json", "w"), indent=2)

    # --- concise text digest ---
    print(f"\n=== {args.label} : {len(sN)} sim clusters, N in {report['sim_N_range']} ===")
    print(f"df: sim {df_sim:.2f} | " + " ".join(f"{k} {report['df'][k]['df']:.2f}" for k in NULLS))
    print(f"kappa2 median: sim {report['sim_kappa2_median']:.3f} | " +
          " ".join(f"{k} {report['df'][k]['kappa2_median']:.3f}" for k in NULLS))
    print(f"loop fraction: sim {report['sim_loop_fraction']:.3f} | " +
          " ".join(f"{k} {report['df'][k]['loop_fraction']:.3f}" for k in NULLS))
    print("matched-N kappa2 (sim vs null): " +
          " ".join(f"{k} d~{report['matched_N_headline_kappa2'][k]['median_residual']:+.3f}"
                   f"(p={report['matched_N_headline_kappa2'][k]['wilcoxon_greater_p']})" for k in NULLS))
    print("per-bin kappa2 Cliff delta (sim-null):")
    for row in report["per_bin"]:
        print(f"  N {row['bin']:>6} (n={row['n_sim']:>4}): " +
              " ".join(f"{k} {row['nulls'][k]['kappa2_cliffs_delta']:+.2f}" for k in NULLS))
    print(f"report -> {args.out}_summary.json")


if __name__ == "__main__":
    main()
