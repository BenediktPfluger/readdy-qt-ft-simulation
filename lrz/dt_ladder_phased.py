#!/usr/bin/env python
"""Phase-4b — long agglomeration <-> deagglomeration cycled production (resumable).

Reproduces Benedikt Pfluger's thesis Figure-6.4 protocol (one agg<->deagg cycle) but extended
to MANY cycles, so the reversible cluster dynamics the sensor relies on are captured over a long
trajectory. Each phase is a separate ReaDDy segment; state (positions + topology bonds) carries
over between phases via checkpoints. The run is phase-level ``--resume``-capable: if the job is
preempted or a node fails, resubmit and it continues from the last completed phase.

Protocol (Fig 6.4; parameters in the config):
  * WCA potential throughout (cohesion is from harmonic bonds; WCA lets freed particles disperse).
  * agglomeration phase: binding on (kon), breaking off, koff irrelevant.
  * deagglomeration phase: binding off, breaking on (koff), kon irrelevant.
  * dt = 0.05 ns (validated safe; Benedikt ran this regime at 0.05 ns).

This is the resume-wrapped twin of qtft.engine.run_phased (same per-phase construction), kept
separate so it never perturbs the library and can be checkpoint-chained across SLURM jobs.

Usage:
  python lrz/dt_ladder_phased.py --config phased.json            # fresh
  python lrz/dt_ladder_phased.py --config phased.json --resume   # continue
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qtft
from qtft.engine import equilibrate_system, create_system, create_simulation, place_particles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--equil-steps", type=int, default=10000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = qtft.SimulationConfig.load_json(args.config)
    if not cfg.phases:
        sys.exit("config has no phases; use make_agg_deagg_phases to build the schedule")
    dt = float(cfg.timestep)
    base_dir = cfg.phase_base_dir
    os.makedirs(base_dir, exist_ok=True)
    phase_dirs = cfg.phase_dirs
    phase_files = cfg.phase_output_files
    n_phases = len(cfg.phases)
    prog_path = os.path.join(base_dir, "progress.json")

    phases_done = 0
    if args.resume and os.path.exists(prog_path):
        phases_done = int(json.load(open(prog_path)).get("phases_done", 0))
        print(f"RESUME: {phases_done}/{n_phases} phases already done", flush=True)

    total_steps = cfg.effective_n_steps
    print("=" * 66)
    print(f"AGG<->DEAGG PRODUCTION  dt={dt} ns  {n_phases} phases  "
          f"{total_steps:,} steps -> {total_steps*dt/1000:.1f} us")
    for i, p in enumerate(cfg.phases):
        print(f"  phase {i}: {p.name:14s} {p.n_steps:,} steps  binding={p.binding} "
              f"breaking={p.breaking} pot={p.potential_type}")
    print(f"  kon={cfg.topology.kon} koff={cfg.topology.koff} k_bond={cfg.topology.k_bond}  out={base_dir}")
    print("=" * 66, flush=True)

    t0 = time.time()
    pos_qt = pos_ft = None
    if phases_done == 0:
        # equilibrate once (WCA, no reactions) before phase 0
        pos_qt, pos_ft = equilibrate_system(cfg, n_steps=int(args.equil_steps))

    prev_ckpt = os.path.join(phase_dirs[phases_done - 1], "checkpoints") if phases_done > 0 else None
    steps_done = sum(int(p.n_steps) for p in cfg.phases[:phases_done])

    for i in range(phases_done, n_phases):
        phase = cfg.phases[i]
        pdir = phase_dirs[i]
        os.makedirs(pdir, exist_ok=True)
        ckpt = os.path.join(pdir, "checkpoints")
        out = phase_files[i]

        system = create_system(cfg, phase=phase)
        sim = create_simulation(system, cfg, overwrite=True, output_file=out)
        sim.make_checkpoints(stride=int(phase.n_steps), output_directory=ckpt, max_n_saves=2)

        if i == 0:
            place_particles(sim, cfg, positions_qt=pos_qt, positions_ft=pos_ft)
        else:
            sim.load_particles_from_latest_checkpoint(prev_ckpt)
            print(f"  phase {i}: loaded state (positions + bonds) from {prev_ckpt}", flush=True)

        t_ph = time.time()
        sim.run(int(phase.n_steps), dt)
        steps_done += int(phase.n_steps)
        ph_wall = time.time() - t_ph
        sps = phase.n_steps / ph_wall if ph_wall > 0 else 0.0

        json.dump({
            "phases_done": i + 1, "n_phases": n_phases,
            "steps_done": steps_done, "n_steps_total": total_steps,
            "sim_time_us_done": steps_done * dt / 1000.0,
            "last_phase": phase.name, "last_phase_steps_per_s": round(sps, 1),
            "dt_ns": dt, "elapsed_s": round(time.time() - t0, 1),
        }, open(prog_path, "w"), indent=2)
        print(f"  phase {i+1}/{n_phases} [{phase.name}] done: {steps_done:,}/{total_steps:,} steps "
              f"({steps_done*dt/1000:.1f} us), {sps:.0f} steps/s, elapsed {(time.time()-t0)/3600:.2f} h",
              flush=True)
        prev_ckpt = ckpt

    print(f"\nAGG<->DEAGG COMPLETE: {steps_done:,} steps = {steps_done*dt/1000:.1f} us "
          f"in {(time.time()-t0)/3600:.2f} h", flush=True)

    # stitch per-phase trajectories onto one continuous step axis (best-effort)
    try:
        from qtft.analysis import combine_phase_trajectories
        combined = os.path.join(base_dir, "trajectory_combined.h5")
        combine_phase_trajectories(phase_files, combined, step_offsets=cfg.phase_step_offsets)
        print(f"combined trajectory -> {combined}", flush=True)
    except Exception as e:
        print(f"(combine skipped: {e})", flush=True)


if __name__ == "__main__":
    main()
