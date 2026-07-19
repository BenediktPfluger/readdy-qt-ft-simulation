#!/usr/bin/env python
"""SPECIFIC (ferritin-bond) network metrics from ReaDDy topology graphs, to decouple genuine
CaM-RS20 bridging from nonspecific LJ Qt-Qt cohesion (which inflates the spatial-contact
clustering at high Qt density). Bonds only ever form Qt-Ft, so any Qt in a topology of size>=2
is specifically ferritin-engaged; a topology IS a bond-connected cluster.

Per condition (final-state frames, pooled over replicas):
  specific_rough_Qt   = fraction of Qt in a >=2-particle topology (bonded to >=1 ferritin)
  largest_spec_frac   = largest topology's Qt-count / total Qt (specific percolation order param)
  mean_FtperQt_in_clusters = mean (Ft/Qt) within multi-particle topologies (bridges per encapsulin)
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.expanduser("~/readdy-qt-ft-simulation"))
import readdy

QT = {"Qt", "QtC"}; FT = {"Ft", "FtC"}


def analyze(edir, n_state=8):
    h5s = sorted(glob.glob(os.path.join(edir, "replica_*", "trajectory.h5")))
    rough, largest, ftperqt = [], [], []
    for h5 in h5s:
        tr = readdy.Trajectory(h5)
        id2name = {v: k for k, v in tr.particle_types.items()}
        pt, ptypes, pids, ppos = tr.read_observable_particles()
        tt, topos = tr.read_observable_topologies()
        nT = min(len(pt), len(tt))
        for fi in range(max(0, nT - n_state), nT):
            types = np.array([id2name.get(t, "?") for t in np.asarray(ptypes[fi])])
            is_qt = np.isin(types, list(QT)); nqt = int(is_qt.sum())
            if nqt == 0:
                continue
            biggest = 0; bonded_qt = 0
            for top in topos[fi]:
                gidx = np.asarray(top.particles)
                tt_t = types[gidx]
                nq = int(np.isin(tt_t, list(QT)).sum())
                nf = int(np.isin(tt_t, list(FT)).sum())
                if len(gidx) >= 2:                 # bonded cluster (only Qt-Ft bonds exist)
                    bonded_qt += nq
                    if nq > 0:
                        ftperqt.append(nf / nq)
                biggest = max(biggest, nq)
            rough.append(bonded_qt / nqt)
            largest.append(biggest / nqt)
    return {
        "specific_rough_Qt": float(np.mean(rough)) if rough else None,
        "largest_spec_frac": float(np.mean(largest)) if largest else None,
        "mean_FtperQt_in_clusters": float(np.mean(ftperqt)) if ftperqt else None,
        "n_replicas": len(h5s),
    }


def main():
    runs = os.path.expanduser("~/readdy_campaign/runs")
    out = os.path.expanduser("~/readdy_campaign/results/specific_network.json")
    manifest = json.load(open(os.path.expanduser("~/readdy_campaign/results/manifest.json")))
    res = {}
    for lab, m in manifest.items():
        edir = m.get("params") and None
        # find ensemble dir from the per-condition summary (has _ensemble_dir)
        sp = os.path.expanduser(f"~/readdy_campaign/results/{lab}_summary.json")
        edir = json.load(open(sp)).get("_ensemble_dir") if os.path.exists(sp) else None
        if not edir or not os.path.isdir(edir):
            print(f"skip {lab} (no ensemble dir)"); continue
        try:
            r = analyze(edir)
            res[lab] = r
            print(f"{lab}: specific_rough={r['specific_rough_Qt']:.3f} "
                  f"largest_spec={r['largest_spec_frac']:.3f} FtperQt={r['mean_FtperQt_in_clusters']}")
        except Exception as e:
            print(f"ERROR {lab}: {type(e).__name__}: {e}")
            res[lab] = {"error": str(e)}
    json.dump(res, open(out, "w"), indent=2)
    print("WROTE", out)


if __name__ == "__main__":
    main()
