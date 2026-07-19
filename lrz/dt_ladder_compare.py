#!/usr/bin/env python
"""Cross-condition comparer for the k_bond recalibration sweep (Phase 3) + real-kon invariance.

The Phase-1 analyzer keys ensembles by dt (one dir per dt). Phase 3 varies BOTH k_bond and dt,
so several ensembles share a dt; this tool instead globs every ensemble folder under a phase dir,
reads (k_bond, dt, kon) from each ensemble_config.json, and compares every (k_bond, dt) condition
against a single MASTER reference — by default (k_bond=10, dt=0.05), the current production point.

For each condition it reports: stability verdict; the four handover observables (kappa2, mean Qt
coordination, largest-cluster fraction, g(r) bridging peak) + binding (n_bonds, fraction-bound) +
mean cluster size, each mean+/-std across replicas; and whether it drifts from the master beyond
ensemble error. Coordination/clustering use the MASTER's d_cut throughout. It then prints, per
k_bond, the largest dt that is stable AND matches the master (the recalibrated dt_max), and the
overall best (dt, k_bond) speedup.

Usage:
    python lrz/dt_ladder_compare.py --phase-dir dt_ladder/phase23
    python lrz/dt_ladder_compare.py --phase-dir dt_ladder/phase23 --master-kbond 10 --master-dt 0.05
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import struct_descriptors as SD
from qtft import SimulationConfig
import dt_ladder_analyze as DLA   # reuse replica_observables, binding_observables, stability


def discover(phase_dir):
    """Return list of dicts {dir, k_bond, dt, kon} for every ensemble folder found."""
    conds = []
    for d in sorted(glob.glob(os.path.join(phase_dir, "*"))):
        if not glob.glob(os.path.join(d, "replica_*")):
            continue
        cfgp = os.path.join(d, "ensemble_config.json")
        if not os.path.exists(cfgp):
            continue
        c = json.load(open(cfgp))
        conds.append({
            "dir": d,
            "k_bond": float(c["topology"]["k_bond"]),
            "dt": float(c["timestep"]),
            "kon": float(c["topology"]["kon"]),
            "box": float(c["box_size"][0]),
        })
    return conds


def condition_metrics(cond, ref_dcut, nmin, ref_energy_scale=None):
    """Stability + observables for one condition (mean+/-std across replicas)."""
    d = cond["dir"]
    stabs = DLA.load_stability(d)
    sv = DLA.stability_verdict(stabs, ref_energy_scale)
    cfgp = os.path.join(d, "ensemble_config.json")
    cfg = SimulationConfig.from_dict(json.load(open(cfgp)))
    h5s = sorted(glob.glob(os.path.join(d, "replica_*", "trajectory.h5")))
    per, binds = [], []
    for h5 in h5s:
        try:
            ro = DLA.replica_observables(h5, cond["box"], ref_dcut, nmin)
            if ro:
                per.append(ro)
        except Exception as e:
            print(f"  [warn] obs {h5}: {e}")
        try:
            bo = DLA.binding_observables(h5, cfg)
            if bo:
                binds.append(bo)
        except Exception as e:
            print(f"  [warn] bind {h5}: {e}")
    obs = None
    if per:
        k2 = DLA._agg([p["mean_kappa2"] for p in per])
        co = DLA._agg([p["mean_coord"] for p in per])
        lf = DLA._agg([p["largest_fraction"] for p in per])
        pk = DLA._agg([p["gr_peak_nm"] for p in per])
        nb = DLA._agg([b["n_bonds"] for b in binds])
        fq = DLA._agg([b["frac_bound_qt"] for b in binds])
        ff = DLA._agg([b["frac_bound_ft"] for b in binds])
        sizes = [n for p in per for n in p["cluster_sizes"]]
        obs = {
            "n_rep": len(per),
            "kappa2": k2, "coord": co, "largest_fraction": lf, "gr_peak": pk,
            "n_bonds": nb, "frac_bound_qt": fq, "frac_bound_ft": ff,
            "mean_cluster_size": float(np.mean(sizes)) if sizes else None,
        }
    return {"stability": sv, "observables": obs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-dir", required=True)
    ap.add_argument("--master-kbond", type=float, default=10.0)
    ap.add_argument("--master-dt", type=float, default=0.05)
    ap.add_argument("--nmin", type=int, default=4)
    ap.add_argument("--drift-frac", type=float, default=0.15)
    args = ap.parse_args()

    phase_dir = os.path.abspath(args.phase_dir)
    conds = discover(phase_dir)
    if not conds:
        sys.exit(f"no ensembles under {phase_dir}")

    # find master + its d_cut / energy scale
    master = min(conds, key=lambda c: (abs(c["k_bond"] - args.master_kbond) * 1e6
                                       + abs(c["dt"] - args.master_dt)))
    print(f"master = k_bond={master['k_bond']} dt={master['dt']} ({os.path.basename(master['dir'])})")
    pooled = []
    for h5 in sorted(glob.glob(os.path.join(master["dir"], "replica_*", "trajectory.h5"))):
        pooled += [f[0] for f in SD.load_frames(h5)]
    ref_dcut = SD.pick_dcut(*SD.rdf(pooled, master["box"])) if pooled else None
    # master energy scale (|early_median|) so the energy-runaway detector stays active
    m_stabs = DLA.load_stability(master["dir"])
    m_meds = [abs(s["energy"]["early_median"]) for s in m_stabs
              if s.get("energy") and s["energy"].get("early_median") is not None]
    ref_energy_scale = float(np.mean(m_meds)) if m_meds else None
    print(f"reference d_cut = {ref_dcut} nm; energy_scale = {ref_energy_scale}\n")

    # metrics for every condition
    for c in conds:
        c["metrics"] = condition_metrics(c, ref_dcut, args.nmin, ref_energy_scale)
    m_obs = next((c["metrics"]["observables"] for c in conds
                  if c is master or (c["k_bond"] == master["k_bond"] and c["dt"] == master["dt"])), None)

    # drift of each condition vs master
    def drift(obs):
        if not obs or not m_obs:
            return {}
        out = {}
        for key in ("kappa2", "coord", "largest_fraction", "gr_peak", "n_bonds",
                    "frac_bound_qt", "frac_bound_ft", "mean_cluster_size"):
            v = obs.get(key); r = m_obs.get(key)
            vm = v[0] if isinstance(v, (list, tuple)) else v
            vs = v[1] if isinstance(v, (list, tuple)) else 0.0
            rm = r[0] if isinstance(r, (list, tuple)) else r
            rs = r[1] if isinstance(r, (list, tuple)) else 0.0
            if vm is None or rm is None:
                out[key] = None; continue
            comb = np.sqrt((vs or 0) ** 2 + (rs or 0) ** 2) + 1e-9
            rel = abs(vm - rm) / (abs(rm) + 1e-9)
            out[key] = {"val": vm, "ref": rm, "rel": rel,
                        "flag": bool(rel > args.drift_frac and abs(vm - rm) / comb > 2.0)}
        return out

    for c in conds:
        c["drift"] = drift(c["metrics"]["observables"])

    # analytic per-condition: mu*k*dt, dt_crit
    kT = 8.314462618e-3 * 300.0
    # (D_QtC + D_FtC)/kT from master cfg
    mcfg = SimulationConfig.from_dict(json.load(open(os.path.join(master["dir"], "ensemble_config.json"))))
    mu_rel = (mcfg.qt.cluster_diffusion + mcfg.ft.cluster_diffusion) / kT

    # ---- report ----
    conds.sort(key=lambda c: (-c["k_bond"], c["dt"]))
    lines = [f"# dt-ladder Phase-2/3 (k_bond recalibration) — {os.path.basename(phase_dir)}", ""]
    lines.append(f"- master (production point): k_bond={master['k_bond']}, dt={master['dt']} ns; "
                 f"d_cut={ref_dcut:.1f} nm" if ref_dcut else "")
    lines.append("- speedup = dt / 0.05 ns. A condition is a valid dt_max candidate only if "
                 "STABLE **and** no observable drifts from the master.")
    lines.append("")
    lines.append("| k_bond | dt (ns) | speedup | mu·k·dt | verdict | kappa2 | coord | largest_frac | g(r)peak | n_bonds | fbound(Q/F) | mean_clust | drift |")
    lines.append("|---:|---:|---:|--:|:--|--:|--:|--:|--:|--:|--:|--:|:--|")

    def fmt(v):
        if isinstance(v, (list, tuple)) and v[0] is not None:
            return f"{v[0]:.3f}±{v[1]:.3f}"
        return "—"

    best = {}   # k_bond -> max valid dt
    for c in conds:
        obs = c["metrics"]["observables"]; sv = c["metrics"]["stability"]; dr = c["drift"]
        mkdt = mu_rel * c["k_bond"] * c["dt"]
        flags = ",".join(k for k, v in dr.items() if v and v.get("flag")) or "ok"
        valid = (sv["verdict"] == "STABLE") and (flags == "ok")
        if valid:
            best[c["k_bond"]] = max(best.get(c["k_bond"], 0), c["dt"])
        fb = "—"
        if obs and isinstance(obs.get("frac_bound_qt"), (list, tuple)) and obs["frac_bound_qt"][0] is not None:
            fb = f"{obs['frac_bound_qt'][0]:.2f}/{obs['frac_bound_ft'][0]:.2f}"
        mc = f"{obs['mean_cluster_size']:.1f}" if obs and obs.get("mean_cluster_size") else "—"
        lines.append(
            f"| {c['k_bond']:g} | {c['dt']:g} | {c['dt']/0.05:.0f}x | {mkdt:.2f} | {sv['verdict']} | "
            f"{fmt(obs['kappa2']) if obs else '—'} | {fmt(obs['coord']) if obs else '—'} | "
            f"{fmt(obs['largest_fraction']) if obs else '—'} | {fmt(obs['gr_peak']) if obs else '—'} | "
            f"{fmt(obs['n_bonds']) if obs else '—'} | {fb} | {mc} | {flags} |")

    lines.append("")
    lines.append("## Recalibrated dt_max per k_bond (stable + invariant vs master)")
    overall = (0.0, None)
    for kb in sorted(best, reverse=True):
        dtm = best[kb]
        if dtm > overall[0]:
            overall = (dtm, kb)
        lines.append(f"- k_bond={kb:g}: dt_max = {dtm:g} ns ({dtm/0.05:.0f}x)")
    if overall[1] is not None:
        lines.append("")
        lines.append(f"**Best: dt_max = {overall[0]:g} ns at k_bond={overall[1]:g} "
                     f"→ {overall[0]/0.05:.0f}x speedup vs 0.05 ns.**")

    md = "\n".join(lines)
    open(os.path.join(phase_dir, "dt_ladder_compare.md"), "w").write(md)
    json.dump({"master": {"k_bond": master["k_bond"], "dt": master["dt"]},
               "conditions": [{k: c[k] for k in ("k_bond", "dt", "dir")} |
                              {"verdict": c["metrics"]["stability"]["verdict"],
                               "drift_flags": [k for k, v in c["drift"].items() if v and v.get("flag")]}
                              for c in conds],
               "best": {"dt_max": overall[0], "k_bond": overall[1],
                        "speedup": overall[0] / 0.05 if overall[1] else None},
               "recalibrated_dt_max_per_kbond": best},
              open(os.path.join(phase_dir, "dt_ladder_compare.json"), "w"), indent=2)
    print("\n" + md + "\n")
    print(f"report -> {os.path.join(phase_dir, 'dt_ladder_compare.md')}")


if __name__ == "__main__":
    main()
