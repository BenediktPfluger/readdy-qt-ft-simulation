#!/usr/bin/env python
"""Per-replica worker for the dt-stability ladder (Phase 1/2 of HANDOVER_timestep_scaling).

Runs ONE replica of the reference Qt-Ft system at a *test* production timestep, but always
equilibrates first at a fixed *safe* timestep (default 0.05 ns) so every probe starts from
the SAME well-relaxed configuration. This isolates the one variable we are measuring — the
production integrator's step size — from artificial initial-overlap blow-ups, and mirrors a
realistic deployment (equilibrate once at a safe dt, then run long at the target dt).

Pipeline:
  1. load config (its ``timestep`` is the *production* dt; ``n_steps`` the production steps).
  2. equilibrate at ``--equil-dt`` (WCA, no reactions) -> relaxed positions  [unless skipped].
  3. build the production system (LJ + binding), place the relaxed particles, run at prod dt.
  4. read back energy(t) + final-frame positions, compute stability metrics, write
     ``<replica_dir>/stability.json`` and print a one-line STABILITY verdict to stdout.

The authoritative stable/unstable call is made later by dt_ladder_analyze.py (which sees the
whole ladder incl. the dt=0.05 baseline); this worker records raw signals + a quick heuristic.

Usage (invoked by the SLURM array that dt_ladder.py generates):
    python lrz/dt_ladder_run.py --config <cfg>.json [--equil-dt 0.05] [--equil-steps 10000]
                                [--skip-equilibration]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import qtft
from qtft.engine import (
    equilibrate_system,
    create_system,
    create_simulation,
    place_particles,
    run_simulation,
)


def _read_energy(traj):
    try:
        t, e = traj.read_observable_energy()
        return np.asarray(t, dtype=float), np.asarray(e, dtype=float)
    except Exception:
        return None, None


def _frame_scan(traj, qt_names, ft_names, ft_cluster_name, contact, box):
    """Scan ALL recorded frames for stability signals the final frame alone would miss.

    Returns a dict with:
      * final-frame positions/counts + non-finite count + max|coord|
      * running_min_overlap_ratio: min over ALL frames of (min pair distance / contact) —
        catches SILENT NOISE-TUNNELING (a large-dt step passes a particle through a core
        between frames, leaving no NaN, no energy runaway, no final-frame overlap).
      * bond_length_proxy_mean/std: over the final 30% of frames, each FtC's distance to the
        nearest Qt-group particle (~ the harmonic bond length) — inflates when dt broadens
        the bond-length distribution (integrator faithfulness signal), per 2/(2-mu*k*dt).
    Returns None if the particles observable is unavailable.
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None
    try:
        times, types, ids, pos = traj.read_observable_particles()
    except Exception:
        return None
    if len(times) == 0:
        return None
    id2name = {v: k for k, v in traj.particle_types.items()}
    nfr = len(times)

    def _pbc_min(a, b, same):
        a = a[np.isfinite(a).all(axis=1)] if len(a) else a
        b = b[np.isfinite(b).all(axis=1)] if len(b) else b
        if len(a) == 0 or len(b) == 0 or (same and len(a) < 2):
            return None
        aw, bw = np.mod(a, box), np.mod(b, box)
        # numeric guard: np.mod can return exactly box for tiny negative inputs
        aw[aw >= box] = 0.0; bw[bw >= box] = 0.0
        if cKDTree is not None:
            ta = cKDTree(aw, boxsize=box)
            if same:
                d, _ = ta.query(aw, k=2); return float(np.min(d[:, 1]))
            d, _ = ta.query(bw, k=1); return float(np.min(d))
        m = np.inf
        for i in range(len(aw)):
            dd = aw[i] - bw; dd -= box * np.round(dd / box)
            r = np.linalg.norm(dd, axis=1)
            if same: r[i] = np.inf
            m = min(m, float(r.min()))
        return m

    running_min_ratio = np.inf
    n_nonfinite_any = 0
    bl_proxy = []
    tail_start = int(0.70 * (nfr - 1))
    for fi in range(nfr):
        tn = np.array([id2name.get(t, "?") for t in np.asarray(types[fi])])
        p = np.asarray(pos[fi], dtype=float)
        n_nonfinite_any += int(np.isnan(p).sum() + np.isinf(p).sum())
        qt = p[np.isin(tn, qt_names)]
        ft = p[np.isin(tn, ft_names)]
        for (a, b, key, same) in ((qt, qt, ("Qt", "Qt"), True),
                                  (ft, ft, ("Ft", "Ft"), True),
                                  (qt, ft, ("Qt", "Ft"), False)):
            md = _pbc_min(a, b, same)
            c = contact.get(key)
            if md is not None and c:
                running_min_ratio = min(running_min_ratio, md / c)
        # bond-length proxy over the tail frames: FtC -> nearest Qt-group distance
        if fi >= tail_start and cKDTree is not None:
            ftc = p[tn == ft_cluster_name]
            ftc = ftc[np.isfinite(ftc).all(axis=1)] if len(ftc) else ftc
            qtg = qt[np.isfinite(qt).all(axis=1)] if len(qt) else qt
            if len(ftc) and len(qtg):
                qw = np.mod(qtg, box); qw[qw >= box] = 0.0
                fw = np.mod(ftc, box); fw[fw >= box] = 0.0
                d, _ = cKDTree(qw, boxsize=box).query(fw, k=1)
                bl_proxy.extend([float(x) for x in d])

    # final frame details
    tn = np.array([id2name.get(t, "?") for t in np.asarray(types[-1])])
    p = np.asarray(pos[-1], dtype=float)
    qt = p[np.isin(tn, qt_names)]; ft = p[np.isin(tn, ft_names)]
    fin = p[np.isfinite(p).all(axis=1)]
    return {
        "n_qt": int(len(qt)), "n_ft": int(len(ft)),
        "n_nonfinite_coords_final": int(np.isnan(p).sum() + np.isinf(p).sum()),
        "n_nonfinite_coords_anyframe": int(n_nonfinite_any),
        "max_abs_coord_nm": float(np.abs(fin).max()) if len(fin) else None,
        "box_half_nm": box / 2.0,
        "running_min_overlap_ratio": None if not np.isfinite(running_min_ratio) else float(running_min_ratio),
        "bond_length_proxy_mean": float(np.mean(bl_proxy)) if bl_proxy else None,
        "bond_length_proxy_std": float(np.std(bl_proxy)) if bl_proxy else None,
        "n_frames": int(nfr),
        "_qt_final": qt, "_ft_final": ft,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--equil-dt", type=float, default=0.05,
                    help="Timestep (ns) used ONLY for the WCA equilibration pre-run.")
    ap.add_argument("--equil-steps", type=int, default=10000)
    ap.add_argument("--skip-equilibration", action="store_true",
                    help="Start production from random positions (no equilibration).")
    args = ap.parse_args()

    t0 = time.time()
    config = qtft.SimulationConfig.load_json(args.config)
    prod_dt = float(config.timestep)
    prod_steps = int(config.n_steps)
    replica_dir = os.path.dirname(config.output_file) or "."
    os.makedirs(replica_dir, exist_ok=True)

    box = float(config.box_size[0])
    rqt, rft = float(config.qt.radius), float(config.ft.radius)
    contact = {("Qt", "Qt"): 2 * rqt, ("Ft", "Ft"): 2 * rft, ("Qt", "Ft"): rqt + rft}
    qt_names = [config.qt.name, config.qt.cluster_name]
    ft_names = [config.ft.name, config.ft.cluster_name]

    print("=" * 64)
    print(f"DT-LADDER PROBE  prod_dt={prod_dt} ns  n_steps={prod_steps:,}  "
          f"sim_time={prod_steps * prod_dt / 1000:.3f} us")
    print(f"  equil: {'SKIP' if args.skip_equilibration else f'{args.equil_steps} steps @ dt={args.equil_dt} ns (WCA)'}")
    print(f"  out: {config.output_file}")
    print("=" * 64, flush=True)

    stab = {
        "prod_dt_ns": prod_dt, "n_steps": prod_steps,
        "sim_time_us": prod_steps * prod_dt / 1000.0,
        "equil_dt_ns": None if args.skip_equilibration else args.equil_dt,
        "equil_steps": 0 if args.skip_equilibration else args.equil_steps,
        "seed": config.rng_seed,
        "stage_failed": None, "error": None,
        "production_completed": False,
    }

    # ---- equilibration at the SAFE dt (positions only) ----------------------
    pos_qt = pos_ft = None
    if not args.skip_equilibration:
        try:
            config.timestep = float(args.equil_dt)          # temporary: equil uses this dt
            pos_qt, pos_ft = equilibrate_system(config, n_steps=int(args.equil_steps))
        except Exception as e:
            stab["stage_failed"] = "equilibration"
            stab["error"] = f"{type(e).__name__}: {e}"
            print("PROBE_FAIL_EQUILIBRATION:", stab["error"])
            traceback.print_exc()
            json.dump(stab, open(os.path.join(replica_dir, "stability.json"), "w"), indent=2)
            print("STABILITY " + json.dumps(stab))
            sys.exit(1)
        finally:
            config.timestep = prod_dt                        # restore production dt

    # ---- production run at the TEST dt --------------------------------------
    t_prod = time.time()
    try:
        system = create_system(config)                       # LJ + binding (production)
        simulation = create_simulation(system, config, overwrite=True)
        place_particles(simulation, config, positions_qt=pos_qt, positions_ft=pos_ft)
        run_simulation(simulation, config, show_progress=True)
        stab["production_completed"] = True
    except Exception as e:
        stab["stage_failed"] = "production"
        stab["error"] = f"{type(e).__name__}: {e}"
        print("PROBE_FAIL_PRODUCTION:", stab["error"])
        traceback.print_exc()
        # keep going: a partial trajectory may still exist and be worth inspecting

    # Throughput is timed on PRODUCTION ONLY (excludes the fixed equilibration pre-run,
    # which would otherwise dominate/under-report steps/s by ~1000x at large dt).
    prod_wall = time.time() - t_prod
    stab["wall_seconds"] = round(time.time() - t0, 2)
    stab["prod_wall_seconds"] = round(prod_wall, 3)
    if prod_steps and prod_wall > 0:
        stab["steps_per_second"] = round(prod_steps / prod_wall, 1)
        stab["prod_wall_per_sim_us"] = round(prod_wall / max(1e-12, prod_steps * prod_dt / 1000.0), 3)

    # ---- read back energy + final config for stability metrics --------------
    import readdy
    try:
        traj = readdy.Trajectory(config.output_file)
    except Exception as e:
        stab["error"] = (stab["error"] or "") + f" | trajectory unreadable: {e}"
        json.dump(stab, open(os.path.join(replica_dir, "stability.json"), "w"), indent=2)
        print("STABILITY " + json.dumps(stab))
        sys.exit(0 if stab["production_completed"] else 1)

    t_e, e = _read_energy(traj)
    if e is not None and len(e):
        finite = e[np.isfinite(e)]
        stab["energy"] = {
            "n_samples": int(len(e)),
            "n_nan": int(np.isnan(e).sum()),
            "n_inf": int(np.isinf(e).sum()),
            "first": float(finite[0]) if len(finite) else None,
            "last": float(finite[-1]) if len(finite) else None,
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "median": float(np.median(finite)) if len(finite) else None,
            # early baseline = median over first 20% of finite samples
            "early_median": float(np.median(finite[:max(1, len(finite) // 5)])) if len(finite) else None,
            "series_times": np.asarray(t_e).tolist(),
            "series_energy": [None if not np.isfinite(x) else float(x) for x in e],
        }
    else:
        stab["energy"] = None

    fs = _frame_scan(traj, qt_names, ft_names, config.ft.cluster_name, contact, box)
    if fs is not None:
        fs.pop("_qt_final", None)
        fs.pop("_ft_final", None)
    stab["final_frame"] = fs

    # ---- quick heuristic verdict (authoritative call is the analyzer) -------
    verdict = "STABLE"
    en = stab["energy"]
    ff = stab["final_frame"]
    if not stab["production_completed"]:
        verdict = "FAILED"
    elif en is None:
        verdict = "NO_ENERGY"
    elif en["n_nan"] or en["n_inf"]:
        verdict = "UNSTABLE_NAN"
    elif ff and ff.get("n_nonfinite_coords_anyframe"):
        verdict = "UNSTABLE_NAN"
    elif en["early_median"] and en["max"] is not None and en["early_median"] != 0 and \
            abs(en["max"]) > 50.0 * abs(en["early_median"]) and abs(en["max"]) > abs(en["early_median"]) + 1e3:
        verdict = "UNSTABLE_ENERGY"        # runaway growth vs the early baseline
    elif ff and ff.get("running_min_overlap_ratio") is not None and ff["running_min_overlap_ratio"] < 0.5:
        verdict = "UNSTABLE_TUNNEL"        # cores interpenetrated to <50% contact in ANY frame
    stab["heuristic_verdict"] = verdict

    json.dump(stab, open(os.path.join(replica_dir, "stability.json"), "w"), indent=2)
    # Compact stdout line (no bulky series) for at-a-glance monitoring in the .out log.
    compact = {k: v for k, v in stab.items() if k != "energy"}
    if en is not None:
        compact["energy"] = {k: en[k] for k in
                             ("n_nan", "n_inf", "first", "last", "max", "early_median")}
    print("STABILITY " + json.dumps(compact))
    print(f"VERDICT {verdict}  (dt={prod_dt} ns, seed={config.rng_seed})")
    sys.exit(0 if stab["production_completed"] else 1)


if __name__ == "__main__":
    main()
