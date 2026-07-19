#!/usr/bin/env python
"""Analyze a dt-stability ladder: Phase-1 stability + Phase-2 statistical invariance.

Reads a phase directory produced by dt_ladder.py (one ensemble folder per dt, each with
replica_*/trajectory.h5 and replica_*/stability.json). Produces:

  STABILITY (Phase 1): per-dt NaN / energy-blowup / core-overlap verdict, using the
    dt=0.05 run as the energy-scale baseline. Energy(t) is expected to stay bounded; a
    runaway or NaN => dt too large.
  INVARIANCE (Phase 2): for each *stable* dt, mean +/- std across replicas of the four
    handover observables — relative shape anisotropy kappa^2, mean Qt coordination,
    largest-cluster fraction, and the g(r) bridging-peak position — plus the full g(r)
    curve, coordination P(k), and cluster-size distribution. Each dt is compared against
    the dt=0.05 reference; a systematic shift beyond the combined ensemble error is flagged.

Coordination/clustering use a SINGLE d_cut derived from the dt=0.05 reference g(r), so the
comparison across dt is not confounded by a per-dt contact cutoff.

Outputs <phase_dir>/dt_ladder_report.{json,md} and (if matplotlib present) figures.

Usage:
    python lrz/dt_ladder_analyze.py --phase-dir dt_ladder/phase1
    python lrz/dt_ladder_analyze.py --phase-dir dt_ladder/phase2 --ref-dt 0.05
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
import struct_descriptors as SD  # shape_metrics, rdf, pick_dcut, analyze_frame, load_frames
from qtft import SimulationConfig
from qtft.analysis import get_bond_counts, get_binding_kinetics

STATE = 0.70   # tfrac >= STATE => 'final agglomerated' frames for state descriptors


def binding_observables(h5: str, config, tail_frac: float = 0.30):
    """Binding-fidelity metrics from ONE trajectory, averaged over the final `tail_frac`.

    Catches reaction-fidelity loss at large dt (encounter under-sampling / one-reaction-per-
    step discretization suppressing binding). Returns final-window mean n_bonds and the
    Qt/Ft fraction-bound; None on read failure.
    """
    import readdy
    try:
        traj = readdy.Trajectory(h5)
        bc = get_bond_counts(h5, trajectory=traj, silent=True)
        kin = get_binding_kinetics(h5, config, trajectory=traj)
    except Exception:
        return None

    def tail_mean(arr):
        a = np.asarray(arr, float)
        if a.size == 0:
            return None
        k = max(1, int(round(tail_frac * len(a))))
        return float(np.mean(a[-k:]))

    return {
        "n_bonds": tail_mean(bc.get("n_bonds")),
        "frac_bound_qt": tail_mean(kin.get("fraction_bound_qt")),
        "frac_bound_ft": tail_mean(kin.get("fraction_bound_ft")),
        "total_reactions": None,   # cumulative reactions read separately if needed
    }


# --------------------------------------------------------------------------- #
# per-replica structural observables (reuse struct_descriptors internals)
# --------------------------------------------------------------------------- #
def replica_observables(h5: str, L: float, d_cut, nmin: int = 4):
    """Return per-replica scalars + raw distributions from ONE trajectory.

    If d_cut is None it is derived from this replica's own g(r); otherwise the supplied
    (reference) d_cut is used for coordination/clustering.
    """
    frames = SD.load_frames(h5)
    if not frames:
        return None
    qt_all = [f[0] for f in frames]
    r, g = SD.rdf(qt_all, L)
    peak_r = float(r[int(np.argmax(np.where((r >= 40) & (r <= 70), g, 0)))])
    my_dcut = SD.pick_dcut(r, g)
    use_dcut = float(d_cut) if d_cut is not None else my_dcut

    coords, largest, kappa2s, sizes = [], [], [], []
    for qt, ft, tf in frames:
        if len(qt) < 2:
            continue
        coord, clusters, lf = SD.analyze_frame(qt, L, use_dcut, nmin)
        if tf >= STATE:
            coords.append(coord)
            largest.append(lf)
        for (N, rg, ext, k2, asp) in clusters:
            if ext < 0.5 * L and N >= nmin:
                kappa2s.append(k2)
            if ext < 0.5 * L:
                sizes.append(int(N))
    if not coords:                      # fallback: last quarter of frames
        for qt, ft, tf in frames[-max(1, len(frames) // 4):]:
            if len(qt) >= 2:
                coord, clusters, lf = SD.analyze_frame(qt, L, use_dcut, nmin)
                coords.append(coord)
                largest.append(lf)
    coord_all = np.concatenate(coords) if coords else np.array([])
    return {
        "mean_coord": float(coord_all.mean()) if len(coord_all) else None,
        "mean_kappa2": float(np.nanmean(kappa2s)) if kappa2s else None,
        "largest_fraction": float(np.mean(largest)) if largest else None,
        "gr_peak_nm": peak_r,
        "my_dcut_nm": my_dcut,
        "gr_r": r, "gr_g": g,
        "coord_hist": np.bincount(coord_all).tolist() if len(coord_all) else [],
        "cluster_sizes": sizes,
        "n_frames": len(frames),
    }


def _agg(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return None, None, 0
    return float(np.mean(v)), float(np.std(v)), len(v)


# --------------------------------------------------------------------------- #
# stability from per-replica stability.json
# --------------------------------------------------------------------------- #
def load_stability(ens_dir: str):
    """Per-replica stability records. A replica that was KILLed for hanging (exploded ->
    ReaDDy neighbour-list spin) has no stability.json but a stability_timeout.json sentinel;
    surface it as an explicit UNSTABLE_HANG record so hung dt are not silently dropped."""
    out = []
    for rep in sorted(glob.glob(os.path.join(ens_dir, "replica_*"))):
        sj = os.path.join(rep, "stability.json")
        tj = os.path.join(rep, "stability_timeout.json")
        if os.path.exists(sj):
            out.append(json.load(open(sj)))
        elif os.path.exists(tj):
            rec = json.load(open(tj))
            rec.setdefault("energy", None)
            rec.setdefault("final_frame", None)
            out.append(rec)
    return out


def stability_verdict(stabs, ref_energy_scale):
    """Aggregate per-replica stability into one dt-level verdict + evidence.

    ref_energy_scale: |early_median energy| of the dt=0.05 reference (energy magnitude scale).
    A dt is UNSTABLE if ANY replica NaNs / fails / overlaps, or if the ensemble energy runs
    away far beyond the reference scale.
    """
    if not stabs:
        return {"verdict": "NO_DATA", "n_replicas": 0}
    n = len(stabs)
    n_completed = sum(1 for s in stabs if s.get("production_completed"))
    n_hang = sum(1 for s in stabs if s.get("stage_failed") == "production_timeout"
                 or s.get("heuristic_verdict") == "UNSTABLE_HANG")
    n_nan = 0
    n_overlap = 0
    n_stage_fail = 0
    max_energy = None
    early_meds = []
    overlaps = []
    wall = []
    steps_per_s = []
    bl_means = []
    for s in stabs:
        if s.get("stage_failed"):
            n_stage_fail += 1
        en = s.get("energy")
        ff = s.get("final_frame")
        if en:
            if en.get("n_nan") or en.get("n_inf"):
                n_nan += 1
            if en.get("max") is not None:
                max_energy = en["max"] if max_energy is None else max(max_energy, en["max"])
            if en.get("early_median") is not None:
                early_meds.append(abs(en["early_median"]))
        if ff:
            # non-finite in ANY frame (worker scans all frames now)
            if ff.get("n_nonfinite_coords_anyframe") or ff.get("n_nonfinite_coords_final"):
                n_nan += 1
            # running (all-frame) overlap catches silent noise-tunneling through cores
            mor = ff.get("running_min_overlap_ratio")
            if mor is None:
                mor = ff.get("min_overlap_ratio")   # backward-compat
            if mor is not None:
                overlaps.append(mor)
                if mor < 0.5:
                    n_overlap += 1
            if ff.get("bond_length_proxy_mean") is not None:
                bl_means.append(ff["bond_length_proxy_mean"])
        if s.get("wall_seconds"):
            wall.append(s["wall_seconds"])
        if s.get("steps_per_second"):
            steps_per_s.append(s["steps_per_second"])

    verdict = "STABLE"
    reasons = []
    if n_hang:
        verdict = "UNSTABLE"; reasons.append(f"{n_hang}/{n} hung (exploded->timeout)")
    if n_completed < n - n_hang:
        verdict = "UNSTABLE"; reasons.append(f"{n - n_hang - n_completed}/{n} did not complete")
    if n_nan:
        verdict = "UNSTABLE"; reasons.append(f"{n_nan}/{n} NaN/inf")
    if n_overlap:
        verdict = "UNSTABLE"; reasons.append(f"{n_overlap}/{n} core tunneling (<0.5 contact, any frame)")
    # Energy runaway vs the reference scale (only if we have a scale and finite energy).
    if ref_energy_scale and max_energy is not None and np.isfinite(max_energy):
        if abs(max_energy) > 100.0 * ref_energy_scale and abs(max_energy) > 1e4:
            verdict = "UNSTABLE"; reasons.append(
                f"energy max {max_energy:.3g} >> 100x ref scale {ref_energy_scale:.3g}")
    return {
        "verdict": verdict,
        "reasons": reasons,
        "n_replicas": n,
        "n_completed": n_completed,
        "n_hang": n_hang,
        "n_nan": n_nan,
        "n_overlap": n_overlap,
        "n_stage_fail": n_stage_fail,
        "energy_max": max_energy,
        "energy_early_median_abs_mean": float(np.mean(early_meds)) if early_meds else None,
        "running_min_overlap_ratio": float(np.min(overlaps)) if overlaps else None,
        "bond_length_proxy_mean": float(np.mean(bl_means)) if bl_means else None,
        "wall_seconds_mean": float(np.mean(wall)) if wall else None,
        "steps_per_second_mean": float(np.mean(steps_per_s)) if steps_per_s else None,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-dir", required=True)
    ap.add_argument("--ref-dt", type=float, default=0.05)
    ap.add_argument("--nmin", type=int, default=4)
    ap.add_argument("--drift-frac", type=float, default=0.15,
                    help="relative-shift threshold to flag observable drift vs the reference")
    args = ap.parse_args()

    phase_dir = os.path.abspath(args.phase_dir)
    idx_path = os.path.join(phase_dir, "dt_ladder_index.json")
    if os.path.exists(idx_path):
        index = json.load(open(idx_path))
        dt_dirs = {float(k): v["dir"] for k, v in index.items()}
    else:
        # fall back to globbing ensemble folders that contain replica_*/
        dt_dirs = {}
        for d in sorted(glob.glob(os.path.join(phase_dir, "*"))):
            if glob.glob(os.path.join(d, "replica_*")):
                cfg = os.path.join(d, "ensemble_config.json")
                if os.path.exists(cfg):
                    dt = float(json.load(open(cfg)).get("timestep"))
                    dt_dirs[dt] = d
    if not dt_dirs:
        sys.exit(f"no dt ensembles found under {phase_dir}")

    dts = sorted(dt_dirs)
    L = 500.0
    # box from any ensemble_config
    for d in dt_dirs.values():
        cfg = os.path.join(d, "ensemble_config.json")
        if os.path.exists(cfg):
            try:
                L = float(json.load(open(cfg))["box_size"][0]); break
            except Exception:
                pass

    print(f"phase_dir={phase_dir}\ndts={dts}\nL={L}\n")

    # ---- 1. reference d_cut + energy scale from ref-dt --------------------- #
    ref_dcut = None
    ref_energy_scale = None
    if args.ref_dt in dt_dirs:
        ref_stabs = load_stability(dt_dirs[args.ref_dt])
        meds = [abs(s["energy"]["early_median"]) for s in ref_stabs
                if s.get("energy") and s["energy"].get("early_median") is not None]
        ref_energy_scale = float(np.mean(meds)) if meds else None
        # reference d_cut from pooled ref g(r)
        ref_h5 = sorted(glob.glob(os.path.join(dt_dirs[args.ref_dt], "replica_*", "trajectory.h5")))
        pooled = []
        for h5 in ref_h5:
            fr = SD.load_frames(h5)
            pooled += [f[0] for f in fr]
        if pooled:
            r, g = SD.rdf(pooled, L)
            ref_dcut = SD.pick_dcut(r, g)
    print(f"reference dt={args.ref_dt}: d_cut={ref_dcut}, energy_scale={ref_energy_scale}\n")

    # ---- 2. per-dt stability + observables --------------------------------- #
    report = {"phase_dir": phase_dir, "ref_dt": args.ref_dt, "L": L,
              "ref_dcut_nm": ref_dcut, "ref_energy_scale": ref_energy_scale, "dts": {}}
    for dt in dts:
        d = dt_dirs[dt]
        stabs = load_stability(d)
        sv = stability_verdict(stabs, ref_energy_scale)
        entry = {"dir": d, "stability": sv, "observables": None}

        # config for this dt (binding kinetics needs it)
        cfg_path = os.path.join(d, "ensemble_config.json")
        cfg = SimulationConfig.from_dict(json.load(open(cfg_path))) if os.path.exists(cfg_path) else None

        # analytic annotations so a KMC-granularity or noise-tunneling limit is never
        # mis-attributed to integrator instability (and vice versa).
        if cfg is not None:
            kT = 8.314462618e-3 * float(cfg.temperature)           # kJ/mol
            mu_rel = (cfg.qt.cluster_diffusion + cfg.ft.cluster_diffusion) / kT
            k_bond = float(cfg.topology.k_bond)
            dt_crit = 2.0 / (mu_rel * k_bond) if mu_rel * k_bond > 0 else None
            step_rms_ft = float(np.sqrt(6.0 * cfg.ft.diffusion * dt))     # 3D rms displacement
            step_rms_qt = float(np.sqrt(6.0 * cfg.qt.diffusion * dt))
            contact_ff = 2.0 * cfg.ft.radius
            entry["annotations"] = {
                "kon_dt": float(cfg.topology.kon) * dt,               # KMC per-step prob ~ 1-exp(-kon*dt)
                "mu_k_dt": mu_rel * k_bond * dt,                      # bond Euler factor; >2 => divergence
                "dt_crit_bond_ns": dt_crit,
                "bond_var_inflation": (2.0 / (2.0 - mu_rel * k_bond * dt)) if (mu_rel * k_bond * dt) < 2 else None,
                "step_rms_ft_nm": step_rms_ft,
                "step_rms_qt_nm": step_rms_qt,
                "step_rms_ft_over_contact": step_rms_ft / contact_ff,  # >0.3 => single-step tunneling risk
            }

        # observables (only meaningful if it ran; compute regardless for context)
        h5s = sorted(glob.glob(os.path.join(d, "replica_*", "trajectory.h5")))
        per_rep = []
        binds = []
        for h5 in h5s:
            try:
                ro = replica_observables(h5, L, ref_dcut, args.nmin)
                if ro:
                    per_rep.append(ro)
            except Exception as e:
                print(f"  [warn] observables failed for {h5}: {e}")
            if cfg is not None:
                try:
                    bo = binding_observables(h5, cfg)
                    if bo:
                        binds.append(bo)
                except Exception as e:
                    print(f"  [warn] binding obs failed for {h5}: {e}")
        if per_rep:
            k2_m, k2_s, k2_n = _agg([p["mean_kappa2"] for p in per_rep])
            co_m, co_s, co_n = _agg([p["mean_coord"] for p in per_rep])
            lf_m, lf_s, lf_n = _agg([p["largest_fraction"] for p in per_rep])
            pk_m, pk_s, pk_n = _agg([p["gr_peak_nm"] for p in per_rep])
            all_sizes = [n for p in per_rep for n in p["cluster_sizes"]]
            nb_m, nb_s, _ = _agg([b["n_bonds"] for b in binds])
            fq_m, fq_s, _ = _agg([b["frac_bound_qt"] for b in binds])
            ff_m, ff_s, _ = _agg([b["frac_bound_ft"] for b in binds])
            entry["observables"] = {
                "n_replicas": len(per_rep),
                "kappa2_mean": k2_m, "kappa2_std": k2_s,
                "coord_mean": co_m, "coord_std": co_s,
                "largest_fraction_mean": lf_m, "largest_fraction_std": lf_s,
                "gr_peak_mean": pk_m, "gr_peak_std": pk_s,
                "n_bonds_mean": nb_m, "n_bonds_std": nb_s,
                "frac_bound_qt_mean": fq_m, "frac_bound_qt_std": fq_s,
                "frac_bound_ft_mean": ff_m, "frac_bound_ft_std": ff_s,
                "mean_cluster_size": float(np.mean(all_sizes)) if all_sizes else None,
                "max_cluster_size": int(np.max(all_sizes)) if all_sizes else None,
                # keep pooled curves for overlays
                "_gr_r": per_rep[0]["gr_r"].tolist(),
                "_gr_g_mean": np.mean([p["gr_g"] for p in per_rep], axis=0).tolist()
                              if len({len(p["gr_g"]) for p in per_rep}) == 1 else per_rep[0]["gr_g"].tolist(),
                "_cluster_sizes": all_sizes,
            }
        report["dts"][f"{dt:g}"] = entry

    # ---- 3. invariance comparison vs reference ----------------------------- #
    ref_obs = report["dts"].get(f"{args.ref_dt:g}", {}).get("observables")
    for dt in dts:
        obs = report["dts"][f"{dt:g}"]["observables"]
        if not obs or not ref_obs:
            continue
        drift = {}
        for key in ("kappa2", "coord", "largest_fraction", "gr_peak",
                    "n_bonds", "frac_bound_qt", "frac_bound_ft"):
            m = obs.get(f"{key}_mean"); s = obs.get(f"{key}_std")
            m0 = ref_obs.get(f"{key}_mean"); s0 = ref_obs.get(f"{key}_std")
            if m is None or m0 is None:
                drift[key] = None; continue
            comb = np.sqrt((s or 0) ** 2 + (s0 or 0) ** 2) + 1e-9
            rel = abs(m - m0) / (abs(m0) + 1e-9)
            drift[key] = {
                "value": m, "ref": m0, "abs_shift": m - m0,
                "rel_shift": rel, "n_sigma": abs(m - m0) / comb,
                # flagged if BOTH a big relative shift AND > ~2 combined-sigma
                "flag": bool(rel > args.drift_frac and abs(m - m0) / comb > 2.0),
            }
        report["dts"][f"{dt:g}"]["drift"] = drift

    # ---- 4. pick dt_max ---------------------------------------------------- #
    stable_dts = [dt for dt in dts
                  if report["dts"][f"{dt:g}"]["stability"]["verdict"] == "STABLE"]
    invariant_dts = []
    for dt in stable_dts:
        dr = report["dts"][f"{dt:g}"].get("drift")
        if dr is None:
            invariant_dts.append(dt); continue
        if not any(v and v.get("flag") for v in dr.values()):
            invariant_dts.append(dt)
    report["stable_dts"] = stable_dts
    report["invariant_dts"] = invariant_dts
    report["dt_max_stable"] = max(stable_dts) if stable_dts else None
    report["dt_max"] = max(invariant_dts) if invariant_dts else (max(stable_dts) if stable_dts else None)
    if report["dt_max"]:
        report["speedup_vs_0.05"] = report["dt_max"] / 0.05

    json.dump(report, open(os.path.join(phase_dir, "dt_ladder_report.json"), "w"), indent=2)

    # ---- 5. markdown table ------------------------------------------------- #
    lines = [f"# dt-ladder report — {os.path.basename(phase_dir)}", ""]
    lines.append(f"- reference dt = {args.ref_dt} ns; d_cut = "
                 f"{ref_dcut:.1f} nm" if ref_dcut else f"- reference dt = {args.ref_dt} ns")
    lines.append(f"- dt_max (stable): **{report['dt_max_stable']} ns**; "
                 f"dt_max (stable+invariant): **{report['dt_max']} ns** "
                 f"(speedup {report.get('speedup_vs_0.05', float('nan')):.0f}x vs 0.05 ns)")
    lines.append("")
    lines.append("| dt (ns) | verdict | done | NaN | overlap | steps/s | kappa2 | coord | largest_frac | g(r) peak | n_bonds | frac_bound(Qt/Ft) | drift |")
    lines.append("|---:|:--|:--:|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|:--|")
    for dt in dts:
        e = report["dts"][f"{dt:g}"]
        sv = e["stability"]; obs = e["observables"]; dr = e.get("drift") or {}
        def fmt(m, s):
            return f"{m:.3f}±{s:.3f}" if (m is not None and s is not None) else "—"
        k2 = fmt(obs["kappa2_mean"], obs["kappa2_std"]) if obs else "—"
        co = fmt(obs["coord_mean"], obs["coord_std"]) if obs else "—"
        lf = fmt(obs["largest_fraction_mean"], obs["largest_fraction_std"]) if obs else "—"
        pk = fmt(obs["gr_peak_mean"], obs["gr_peak_std"]) if obs else "—"
        nb = fmt(obs.get("n_bonds_mean"), obs.get("n_bonds_std")) if obs else "—"
        fb = (f"{obs['frac_bound_qt_mean']:.2f}/{obs['frac_bound_ft_mean']:.2f}"
              if obs and obs.get("frac_bound_qt_mean") is not None
              and obs.get("frac_bound_ft_mean") is not None else "—")
        flags = ",".join(k for k, v in dr.items() if v and v.get("flag")) or "ok"
        sps = f"{sv['steps_per_second_mean']:.0f}" if sv.get("steps_per_second_mean") else "—"
        lines.append(f"| {dt:g} | {sv['verdict']} | {sv['n_completed']}/{sv['n_replicas']} | "
                     f"{sv['n_nan']} | {sv['n_overlap']} | {sps} | {k2} | {co} | {lf} | {pk} | {nb} | {fb} | {flags} |")
        if sv.get("reasons"):
            lines.append(f"|  | _{'; '.join(sv['reasons'])}_ |||||||||||||")

    # Analytic diagnostics: which limit (bond divergence / KMC granularity / tunneling)
    # each dt is up against, so an empirical verdict is never mis-attributed.
    any_ann = next((report["dts"][f"{dt:g}"].get("annotations") for dt in dts
                    if report["dts"][f"{dt:g}"].get("annotations")), None)
    if any_ann:
        dtc = any_ann.get("dt_crit_bond_ns")
        lines.append("")
        lines.append(f"### Analytic diagnostics (bond dt_crit ≈ {dtc:.3f} ns)" if dtc else "### Analytic diagnostics")
        lines.append("| dt (ns) | mu·k·dt (bond, >2 diverges) | bond_var_inflation | kon·dt | step_rms_Ft (nm) | rms_Ft/contact (>0.3 tunnels) |")
        lines.append("|---:|--:|--:|--:|--:|--:|")
        for dt in dts:
            a = report["dts"][f"{dt:g}"].get("annotations")
            if not a:
                continue
            bvi = f"{a['bond_var_inflation']:.2f}x" if a.get("bond_var_inflation") else "DIVERGES"
            lines.append(f"| {dt:g} | {a['mu_k_dt']:.2f} | {bvi} | {a['kon_dt']:.3g} | "
                         f"{a['step_rms_ft_nm']:.1f} | {a['step_rms_ft_over_contact']:.2f} |")
    md = "\n".join(lines)
    open(os.path.join(phase_dir, "dt_ladder_report.md"), "w").write(md)
    print("\n" + md + "\n")
    print(f"report -> {os.path.join(phase_dir, 'dt_ladder_report.json')}")
    print(f"report -> {os.path.join(phase_dir, 'dt_ladder_report.md')}")

    # ---- 6. figures (best-effort) ----------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # (a) energy(t) overlay
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for dt in dts:
            d = dt_dirs[dt]
            for p in sorted(glob.glob(os.path.join(d, "replica_*", "stability.json"))):
                s = json.load(open(p))
                en = s.get("energy")
                if en and en.get("series_times"):
                    t = np.asarray(en["series_times"], float) * dt / 1000.0  # -> us
                    y = np.asarray([np.nan if v is None else v for v in en["series_energy"]], float)
                    ax[0].plot(t, y, lw=1, alpha=0.8, label=f"{dt:g} ns")
                    break
        ax[0].set_xlabel("simulated time (us)"); ax[0].set_ylabel("potential energy")
        ax[0].set_title("energy(t) per dt (rep 0)"); ax[0].legend(fontsize=7)
        # (b) observables vs dt with reference band
        okdt = [dt for dt in dts if report["dts"][f"{dt:g}"]["observables"]]
        for key, mk in (("kappa2", "o"), ("coord", "s"), ("largest_fraction", "^")):
            xs, ys, es = [], [], []
            for dt in okdt:
                o = report["dts"][f"{dt:g}"]["observables"]
                if o.get(f"{key}_mean") is not None:
                    xs.append(dt); ys.append(o[f"{key}_mean"]); es.append(o.get(f"{key}_std") or 0)
            if xs:
                ax[1].errorbar(xs, ys, yerr=es, marker=mk, capsize=3, label=key)
        ax[1].set_xscale("log"); ax[1].set_xlabel("dt (ns)")
        ax[1].set_ylabel("observable (mean±std)"); ax[1].set_title("invariance vs dt")
        ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(phase_dir, "dt_ladder_energy_observables.png"), dpi=130)
        print("figure -> dt_ladder_energy_observables.png")

        # (c) g(r) overlay for stable dts
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        for dt in okdt:
            o = report["dts"][f"{dt:g}"]["observables"]
            if o.get("_gr_r"):
                ax2.plot(o["_gr_r"], o["_gr_g_mean"], lw=1.4, label=f"{dt:g} ns")
        ax2.set_xlim(0, 160); ax2.set_xlabel("Qt-Qt r (nm)"); ax2.set_ylabel("g(r)")
        ax2.set_title("g(r) invariance across dt"); ax2.legend(fontsize=8)
        fig2.tight_layout(); fig2.savefig(os.path.join(phase_dir, "dt_ladder_gr.png"), dpi=130)
        print("figure -> dt_ladder_gr.png")
    except Exception as e:
        print(f"(figures skipped: {e})")


if __name__ == "__main__":
    main()
