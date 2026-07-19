#!/usr/bin/env python
"""Phase-4 long production run at the validated safe timestep (dt = 0.05 ns).

The dt-scaling study concluded dt cannot be raised for this system (dt_max = 0.05 ns), so
"more simulated time" comes from WALL-CLOCK: a long serial_long (7-day) job. This worker runs
the reference condition in SEGMENTS chained by ReaDDy checkpoints (positions + topology bonds),
so the run is resilient to node failure and can be RESUMED / chained across jobs — resubmit the
same command with --resume and it continues from the last completed segment.

Layout under <out_dir> (the config's output directory):
  checkpoints/            ReaDDy checkpoints (latest 2 kept)
  seg_000/trajectory.h5   per-segment trajectory (step index restarts each segment)
  progress.json          {segments_done, n_segments, seg_steps, steps_done, n_steps_total}

Usage:
  python lrz/dt_ladder_production.py --config prod.json --segments 20            # fresh start
  python lrz/dt_ladder_production.py --config prod.json --segments 20 --resume   # continue
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qtft
from qtft.engine import (
    equilibrate_system, create_system, create_simulation, place_particles,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--segments", type=int, default=20, help="checkpoint-chained segments")
    ap.add_argument("--equil-steps", type=int, default=10000)
    ap.add_argument("--resume", action="store_true", help="continue from progress.json + latest checkpoint")
    args = ap.parse_args()

    cfg = qtft.SimulationConfig.load_json(args.config)
    dt = float(cfg.timestep)
    n_total = int(cfg.n_steps)
    out_dir = os.path.dirname(cfg.output_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    prog_path = os.path.join(out_dir, "progress.json")

    n_seg = int(args.segments)
    seg_steps = max(1, n_total // n_seg)
    # last segment absorbs the remainder so total == n_total
    seg_sizes = [seg_steps] * (n_seg - 1) + [n_total - seg_steps * (n_seg - 1)]

    # resume state
    seg_start = 0
    if args.resume and os.path.exists(prog_path):
        prog = json.load(open(prog_path))
        seg_start = int(prog.get("segments_done", 0))
        print(f"RESUME: {seg_start}/{n_seg} segments already done "
              f"({prog.get('steps_done', 0):,}/{n_total:,} steps)")

    print("=" * 64)
    print(f"PRODUCTION dt={dt} ns  n_steps={n_total:,}  ->  {n_total * dt / 1e6:.3f} ms simulated")
    print(f"  segments={n_seg} x ~{seg_steps:,} steps; out={out_dir}")
    print("=" * 64, flush=True)

    t0 = time.time()
    pos_qt = pos_ft = None
    if seg_start == 0 and not (args.resume and os.path.exists(ckpt_dir) and os.listdir(ckpt_dir)):
        # fresh: equilibrate at the (production == safe) dt, then place particles in seg 0
        pos_qt, pos_ft = equilibrate_system(cfg, n_steps=int(args.equil_steps))

    steps_done = sum(seg_sizes[:seg_start])
    for i in range(seg_start, n_seg):
        seg_dir = os.path.join(out_dir, f"seg_{i:03d}")
        os.makedirs(seg_dir, exist_ok=True)
        out_file = os.path.join(seg_dir, "trajectory.h5")
        ss = seg_sizes[i]

        system = create_system(cfg)                      # LJ + binding (production)
        sim = create_simulation(system, cfg, overwrite=True, output_file=out_file)
        # checkpoint at the END of the segment so the next segment (this or a future job) resumes
        sim.make_checkpoints(stride=int(ss), output_directory=ckpt_dir, max_n_saves=2)

        if i == 0 and pos_qt is not None:
            place_particles(sim, cfg, positions_qt=pos_qt, positions_ft=pos_ft)
        else:
            sim.load_particles_from_latest_checkpoint(ckpt_dir)
            print(f"  seg {i}: loaded state (positions + bonds) from {ckpt_dir}", flush=True)

        t_seg = time.time()
        sim.run(int(ss), dt)
        steps_done += ss
        seg_wall = time.time() - t_seg
        sps = ss / seg_wall if seg_wall > 0 else 0.0

        json.dump({
            "segments_done": i + 1, "n_segments": n_seg, "seg_steps": seg_steps,
            "steps_done": steps_done, "n_steps_total": n_total,
            "sim_time_us_done": steps_done * dt / 1000.0,
            "dt_ns": dt, "last_seg_steps_per_s": round(sps, 1),
            "elapsed_s": round(time.time() - t0, 1),
        }, open(prog_path, "w"), indent=2)
        print(f"  seg {i+1}/{n_seg} done: {steps_done:,}/{n_total:,} steps "
              f"({steps_done*dt/1000:.1f} us), {sps:.0f} steps/s, "
              f"elapsed {(time.time()-t0)/3600:.2f} h", flush=True)

    print(f"\nPRODUCTION COMPLETE: {steps_done:,} steps = {steps_done*dt/1e6:.3f} ms "
          f"in {(time.time()-t0)/3600:.2f} h", flush=True)

    # stitch per-segment trajectories into one continuous trajectory (best-effort)
    try:
        from qtft.analysis import combine_phase_trajectories
        seg_files = [os.path.join(out_dir, f"seg_{i:03d}", "trajectory.h5") for i in range(n_seg)]
        seg_files = [f for f in seg_files if os.path.exists(f)]
        offsets, cum = [], 0
        for i in range(len(seg_files)):
            offsets.append(cum); cum += seg_sizes[i]
        combined = os.path.join(out_dir, "trajectory_combined.h5")
        combine_phase_trajectories(seg_files, combined, step_offsets=offsets)
        print(f"combined trajectory -> {combined}", flush=True)
    except Exception as e:
        print(f"(combine skipped: {e})", flush=True)


if __name__ == "__main__":
    main()
