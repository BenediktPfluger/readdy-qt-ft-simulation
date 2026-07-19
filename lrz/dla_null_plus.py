#!/usr/bin/env python
"""Matched-N geometric nulls for the Qt-cluster shape/DLA analysis — extends the prior
dla_null_campaign.py with an RLCA null and loop (cyclomatic) counting.

Four nulls, each a point cloud generated at the sim's observed per-cluster N, on the sim's
length scale (Qt-Qt contact = g(r) bridging peak), scored with the VERBATIM shape_metrics()
recipe so every comparison is recipe-identical and at matched N:

  compact  -- N hard spheres random-close-packed into a blob        (spherical floor, kappa2->0)
  random   -- N uniform Poisson points                              (isotropic, no aggregation)
  dla      -- off-lattice Witten-Sander particle-cluster aggregate  (diffusion-limited, df~1.9)
  rlca     -- same walker but sticks with probability s<1           (reaction-limited, compacts)

For each null cluster we also compute the cyclomatic number mu = E - N + 1 on the graph of
neighbours within d_cut (the SAME d_cut applied to the sim), so loops are compared fairly:
a strictly tree-like (DLA) aggregate at contact has few d_cut loops; a compact arrangement has
many. This makes "loop content" a fair, FIB-SEM-comparable discriminator, not just a density proxy.

Usage:
  python lrz/dla_null_plus.py --sim-npz <sim>_data.npz --out <prefix> \
      --repl 300 --bond-nm 41 --dcut 62 --stick 0.05 --seed 20260716
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import struct_descriptors as SD   # shape_metrics VERBATIM


def compact_ball(N, rng, contact=1.0, iters=80, phi0=0.55):
    if N == 1:
        return np.zeros((1, 3))
    R0 = (N / (8 * phi0)) ** (1.0 / 3.0) * contact
    u = rng.normal(size=(N, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    p = u * (R0 * rng.random(N)[:, None] ** (1.0 / 3.0))
    for _ in range(iters):
        d = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(d, axis=2); np.fill_diagonal(dist, np.inf)
        overlap = contact - dist
        m = overlap > 0
        if m.any():
            with np.errstate(invalid="ignore"):
                dirv = d / dist[:, :, None]
            push = np.where(m[:, :, None], 0.5 * overlap[:, :, None] * dirv, 0.0)
            p = p + np.nan_to_num(np.nansum(push, axis=1))
        p -= p.mean(0); p *= 0.997
    return np.nan_to_num(p)


def random_membership(N, rng, box=500.0, **kw):
    return rng.random((N, 3)) * box


def dla_cluster(N, rng, contact=1.0, step_min=1e-3, stick=1.0):
    """Off-lattice particle-cluster aggregation. stick=1 -> DLA; stick<1 -> RLCA-like (compacts)."""
    pts = np.zeros((N, 3)); Rc = 0.0; m = 1
    while m < N:
        Rb = Rc + 2.0 * contact
        Rk = 3.0 * Rb + 20.0 * contact
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        w = Rb * u
        while True:
            d = np.linalg.norm(pts[:m] - w, axis=1)
            j = int(np.argmin(d)); dmin = float(d[j])
            if dmin <= contact + 1e-6:
                if stick >= 1.0 or rng.random() < stick:
                    dir_ = (w - pts[j]); nn = np.linalg.norm(dir_)
                    dir_ = dir_ / nn if nn > 1e-12 else rng.normal(size=3) / np.sqrt(3)
                    pts[m] = pts[j] + contact * dir_
                    Rc = max(Rc, float(np.linalg.norm(pts[m]))); m += 1
                    break
                else:
                    # did not stick: take a step away and keep walking (reaction-limited)
                    v = rng.normal(size=3); v /= np.linalg.norm(v)
                    w = w + max(step_min, contact) * v
                    if np.linalg.norm(w) > Rk:
                        u = rng.normal(size=3); u /= np.linalg.norm(u); w = Rb * u
                    continue
            rho = max(step_min, dmin - contact)
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            w = w + rho * v
            if np.linalg.norm(w) > Rk:
                u = rng.normal(size=3); u /= np.linalg.norm(u); w = Rb * u
    return pts


def cyclomatic_mu(pts, d_cut):
    """Loops on the d_cut neighbour graph of a single connected cluster: mu = E - N + 1."""
    N = len(pts)
    if N < 3:
        return 0
    pr = cKDTree(pts).query_pairs(d_cut, output_type="ndarray")
    return int(len(pr) - N + 1)


NULLS = {
    "compact": lambda N, rng, contact, stick: compact_ball(N, rng, contact=contact),
    "random":  lambda N, rng, contact, stick: random_membership(N, rng),
    "dla":     lambda N, rng, contact, stick: dla_cluster(N, rng, contact=contact, stick=1.0),
    "rlca":    lambda N, rng, contact, stick: dla_cluster(N, rng, contact=contact, stick=stick),
}


_GLOBAL = {}


def _init(bond, dcut, stick):
    _GLOBAL.update(bond=bond, dcut=dcut, stick=stick)


def _gen_one(task):
    """Worker: generate one null cluster and return (null, N, kappa2, aspect, Rg, mu)."""
    k, N, seed = task
    bond = _GLOBAL["bond"]; dcut = _GLOBAL["dcut"]; stick = _GLOBAL["stick"]
    rng = np.random.default_rng(seed)
    c = NULLS[k](int(N), rng, bond, stick)
    rg, k2, asp = SD.shape_metrics(c)
    extent = float((c.max(0) - c.min(0)).max()) if len(c) else 0.0
    return (k, int(N), k2, asp, rg, cyclomatic_mu(c, dcut), extent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repl", type=int, default=300)
    ap.add_argument("--nmin", type=int, default=4)
    ap.add_argument("--bond-nm", type=float, default=None, help="Qt-Qt contact; default = sim bond_nm")
    ap.add_argument("--dcut", type=float, default=None, help="loop graph cutoff; default = sim d_cut")
    ap.add_argument("--stick", type=float, default=0.05, help="RLCA sticking probability (<1)")
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count-1")
    args = ap.parse_args()

    d = np.load(args.sim_npz)
    simN = d["cluster_N"].astype(int)
    uniqN = np.unique(simN[simN >= args.nmin])
    bond = float(args.bond_nm) if args.bond_nm else float(d["bond_nm"][0])
    dcut = float(args.dcut) if args.dcut else float(d["d_cut"][0])

    # build independent per-task seeds via SeedSequence.spawn
    tasks = [(k, int(N), None) for N in uniqN for k in NULLS for _ in range(args.repl)]
    children = np.random.SeedSequence(args.seed).spawn(len(tasks))
    tasks = [(t[0], t[1], int(children[i].generate_state(1)[0])) for i, t in enumerate(tasks)]

    from multiprocessing import Pool, cpu_count
    nw = args.workers or max(1, cpu_count() - 1)
    store = {k: {"N": [], "kappa2": [], "aspect": [], "Rg": [], "mu": [], "extent": []} for k in NULLS}
    with Pool(nw, initializer=_init, initargs=(bond, dcut, args.stick)) as pool:
        for k, N, k2, asp, rg, mu, ext in pool.imap_unordered(_gen_one, tasks, chunksize=8):
            store[k]["N"].append(N); store[k]["kappa2"].append(k2)
            store[k]["aspect"].append(asp); store[k]["Rg"].append(rg)
            store[k]["mu"].append(mu); store[k]["extent"].append(ext)

    out = {"uniqN": uniqN.astype(float), "bond_nm": np.array([bond]),
           "d_cut": np.array([dcut]), "repl": np.array([args.repl]), "stick": np.array([args.stick])}
    for k in NULLS:
        for key in ("N", "kappa2", "aspect", "Rg", "mu", "extent"):
            out[f"{k}_{key}"] = np.array(store[k][key], dtype=float)
    np.savez(args.out + "_data.npz", **out)
    print(f"nulls -> {args.out}_data.npz  | {len(uniqN)} uniqN x {args.repl} repl x {len(NULLS)} nulls")
    for k in NULLS:
        mu = np.array(store[k]["mu"]); k2 = np.array(store[k]["kappa2"])
        print(f"  {k:8s}: kappa2 med {np.median(k2):.3f}  loop_frac {np.mean(mu>0):.3f}  "
              f"mean_mu/N {np.mean(mu/np.maximum(1,np.array(store[k]['N']))):.3f}")


if __name__ == "__main__":
    main()
