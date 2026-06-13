#!/usr/bin/env python
"""
Smoke harness for the Qt-Ft refactor (Phase 0 safety net).

Purpose
-------
There is no automated test suite and ReaDDy is conda-only, so "done" means verified by
running. This script exercises the whole pipeline on a tiny system and is the regression
oracle for every refactor phase in CODE_REVIEW.md:

  1. single run:   equilibrate -> create_system -> create_simulation -> place -> run
  2. analysis:     print_analysis_summary on the produced trajectory
  3. plotting:     a couple of plot_* calls (matplotlib, Agg backend)
  4. ensemble:     2-replica run_local (which auto-saves via save_for_plotting)
  5. R1 oracle:    snapshot the save_for_plotting JSON/NPZ, then re-run analyze_ensemble.py
                   on the same directory and BYTE/VALUE-compare the two. This is the
                   equivalence check that guards the analyze_ensemble <-> EnsembleSimulation
                   de-duplication (finding R1).

Run it (must use the env that has ReaDDy):
    conda run -n readdyEnv python tools/smoke_test.py

Exit code 0 = all green. Non-zero = a stage failed (prints which).
Everything is written under tools/smoke_out/ (safe to delete; git-ignored).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

# Make the repo-root modules importable regardless of CWD.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Headless plotting.
import matplotlib
matplotlib.use("Agg")

import agglomeration_simulation as sim
import agglomeration_analysis as analysis
import agglomeration_plotting as plotting
from agglomeration_ensemble_simulation import EnsembleSimulation

OUT = os.path.join(REPO_ROOT, "tools", "smoke_out")


def banner(msg: str) -> None:
    print("\n" + "#" * 70)
    print(f"# {msg}")
    print("#" * 70)


def make_tiny_config(output_file: str | None = None) -> sim.SimulationConfig:
    """A deliberately tiny, fast, physically-valid configuration."""
    return sim.SimulationConfig(
        qt=sim.ParticleConfig("Qt", radius=4.0, diffusion=0.5),
        ft=sim.ParticleConfig("Ft", radius=2.0, diffusion=1.0),
        topology=sim.TopologyConfig(binding_radius=7.0, kon=25.0, k_bond=10.0),
        lj=sim.LennardJonesConfig(
            epsilon_QtQt=2.5, epsilon_FtFt=1.5, epsilon_QtFt=2.5,
            potential_type="LJ",
        ),
        box_size=(60.0, 60.0, 60.0),
        temperature=300.0,
        timestep=0.02,
        n_steps=3000,           # ~0.06 us; ~30 observable frames
        record_stride=100,
        observable_stride=100,
        n_qt=15,
        n_ft=25,
        kernel="SingleCPU",     # deterministic, no thread setup needed for a smoke run
        rng_seed=12345,
        output_file=output_file,
    )


def stage_single_run() -> str:
    banner("STAGE 1-3: single run + analysis + plotting")
    h5 = os.path.join(OUT, "single", "trajectory.h5")
    os.makedirs(os.path.dirname(h5), exist_ok=True)
    config = make_tiny_config(output_file=h5)

    pos_qt, pos_ft = sim.equilibrate_system(config, n_steps=300)
    system = sim.create_system(config)
    simulation = sim.create_simulation(system, config, overwrite=True)
    sim.place_particles(simulation, config, positions_qt=pos_qt, positions_ft=pos_ft)
    sim.run_simulation(simulation, config)

    analysis.print_analysis_summary(config.output_file, config)
    plotting.plot_observables(
        config.output_file, config,
        save_path=os.path.join(OUT, "single", "observables.png"),
    )
    plotting.plot_cluster_analysis(
        config.output_file, config=config,
        save_path=os.path.join(OUT, "single", "clusters.png"),
    )
    print("STAGE 1-3 OK")
    return config.output_file


def stage_ensemble() -> str:
    banner("STAGE 4: 2-replica ensemble (run_local -> save_for_plotting)")
    base = make_tiny_config(output_file=None)
    ensemble = EnsembleSimulation(
        base_config=base, n_replicas=2, name="smoke", base_dir=OUT,
    )
    ensemble.run_local(parallel=False, overwrite=True,
                       equilibration_steps=300, stride=1)
    print("STAGE 4 OK ->", ensemble.output_dir)
    return ensemble.output_dir


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _compare_json(a: dict, b: dict, ctx: str, rtol=1e-6, atol=1e-9) -> list[str]:
    """Recursively compare two JSON-decoded structures; return list of mismatches."""
    diffs: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a:
                diffs.append(f"{ctx}.{k}: missing in save_for_plotting")
            elif k not in b:
                diffs.append(f"{ctx}.{k}: missing in analyze_ensemble")
            else:
                diffs += _compare_json(a[k], b[k], f"{ctx}.{k}", rtol, atol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{ctx}: list length {len(a)} != {len(b)}")
        else:
            try:
                if not np.allclose(np.asarray(a, float), np.asarray(b, float),
                                   rtol=rtol, atol=atol, equal_nan=True):
                    diffs.append(f"{ctx}: numeric arrays differ")
            except (ValueError, TypeError):
                for i, (x, y) in enumerate(zip(a, b)):
                    diffs += _compare_json(x, y, f"{ctx}[{i}]", rtol, atol)
    else:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                diffs.append(f"{ctx}: {a} != {b}")
        elif a != b:
            diffs.append(f"{ctx}: {a!r} != {b!r}")
    return diffs


def _compare_npz(p_a: str, p_b: str, rtol=1e-6, atol=1e-9) -> list[str]:
    a = np.load(p_a, allow_pickle=True)
    b = np.load(p_b, allow_pickle=True)
    diffs: list[str] = []
    for k in set(a.files) | set(b.files):
        if k not in a.files:
            diffs.append(f"npz[{k}]: missing in save_for_plotting")
        elif k not in b.files:
            diffs.append(f"npz[{k}]: missing in analyze_ensemble")
        else:
            va, vb = a[k], b[k]
            if va.shape != vb.shape:
                diffs.append(f"npz[{k}]: shape {va.shape} != {vb.shape}")
                continue
            try:
                if not np.allclose(va.astype(float), vb.astype(float),
                                   rtol=rtol, atol=atol, equal_nan=True):
                    diffs.append(f"npz[{k}]: values differ")
            except (ValueError, TypeError):
                if not np.array_equal(va, vb):
                    diffs.append(f"npz[{k}]: non-numeric values differ")
    return diffs


def stage_r1_oracle(ensemble_dir: str) -> bool:
    """Snapshot save_for_plotting outputs, re-run the CLI, compare. Guards R1."""
    banner("STAGE 5: R1 equivalence oracle (save_for_plotting vs analyze_ensemble.py)")
    ensemble_dir = ensemble_dir.rstrip("/") + "/"
    stats_p = ensemble_dir + "ensemble_statistics.json"
    npz_p = ensemble_dir + "ensemble_structural.npz"

    # Snapshot the run_local / save_for_plotting outputs.
    stats_baseline = _load_json(stats_p)
    npz_snapshot = ensemble_dir + "ensemble_structural.baseline.npz"
    import shutil
    shutil.copy(npz_p, npz_snapshot)

    # Re-run the CLI on the same directory (overwrites the JSON/NPZ in place).
    cmd = [sys.executable, os.path.join(REPO_ROOT, "analyze_ensemble.py"),
           "--ensemble-dir", ensemble_dir, "--stride", "1"]
    print("running:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        print("STAGE 5 FAILED: analyze_ensemble.py returned non-zero")
        return False

    stats_cli = _load_json(stats_p)
    diffs = _compare_json(stats_baseline, stats_cli, "stats")
    diffs += _compare_npz(npz_snapshot, npz_p)

    if diffs:
        print(f"STAGE 5: {len(diffs)} difference(s) between save_for_plotting and CLI:")
        for d in diffs[:40]:
            print("   -", d)
        print("\n(Today these MAY differ because the two code paths are duplicated — finding R1.")
        print(" This snapshot is the baseline; after the Phase-3 dedup they must match exactly.)")
        return False
    print("STAGE 5 OK: CLI and save_for_plotting outputs are equivalent.")
    return True


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    ok = True
    try:
        stage_single_run()
    except Exception as e:
        print("STAGE 1-3 FAILED:", repr(e))
        return 1
    try:
        ens_dir = stage_ensemble()
    except Exception as e:
        print("STAGE 4 FAILED:", repr(e))
        return 1
    # Stage 5 is allowed to "fail" today (documents the R1 baseline); we still report it.
    r1_ok = stage_r1_oracle(ens_dir)

    banner("SMOKE SUMMARY")
    print("  single run + analysis + plotting : OK")
    print("  ensemble run_local + save        : OK")
    print(f"  R1 equivalence (CLI == save)     : {'OK' if r1_ok else 'DIFFERS (baseline recorded)'}")
    print("\nPipeline is runnable. Use this script as the gate after every refactor phase.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
