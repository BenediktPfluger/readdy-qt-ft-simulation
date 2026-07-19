#!/usr/bin/env python
"""Generate (and optionally submit) the dt-stability ladder for the Qt-Ft reference system.

Implements Phase 1 (stability ladder) and Phase 2 (statistical-invariance replicas) of
HANDOVER_timestep_scaling.md. For each requested production timestep dt it builds the
reference condition (200 Qt / 400 Ft, LJ, box 500^3, kon=0.001, k_bond=10, eQF=3.0) at a
*fixed simulated-time budget* (so every dt covers equal simulated time, only the step COUNT
changes), writes per-replica configs, and emits a SLURM job-array that runs each replica via
``lrz/dt_ladder_run.py`` (which equilibrates at a safe dt, then produces at the test dt).

This is intentionally SEPARATE from lrz_campaign.py so it never perturbs a running campaign.
It reuses the same LRZ SLURM knobs (serial / serial_std / cm4_serial, <=32 cores/job).

Usage (on a cm4 login node inside the `readdy` env, or locally for a smoke test):
    python lrz/dt_ladder.py --phase1                       # ladder, generate only
    python lrz/dt_ladder.py --phase1 --submit              # ladder, generate + sbatch
    python lrz/dt_ladder.py --phase2 --dts 0.05,0.2,1 --submit   # invariance replicas
    python lrz/dt_ladder.py --phase1 --dts 0.05,0.2,1,5,20,100,500 --reps 1 --sim-us 5
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import qtft
from qtft import EnsembleSimulation

# ---- LRZ SLURM knobs (identical policy to lrz_campaign.py) -------------------
HOME = os.path.expanduser("~")
CONDA_BASE = os.path.join(HOME, "miniforge3")
CONDA_ENV = "readdy"
CLUSTER = "serial"
PARTITION = "serial_std"
QOS = "cm4_serial"
CPUS = 4
THREADS = 4
MEM = "6G"

# Reference condition (matches lrz_campaign.build_base ref_200Qt_400Ft, LJ).
REF = dict(nqt=200, nft=400, eqq=1.5, eff=1.5, eqf=3.0, pot="LJ", mono=False)

# Physical binding rate of the real system. Phase 1 (stability) overrides this UP so bonds
# populate within the first few production steps and the stiff-bond instability (the true
# dt limiter) is actually exercised even in short large-dt probes (see --kon).
REF_KON = 0.001

# Default Phase-1 ladder (ns), densely bracketing the predicted bond-divergence cliff at
# dt_crit = 2/((D_QtC+D_FtC)/kT * k_bond) ~= 0.5 ns (and the faithful ~0.1-0.2 ns region).
DEFAULT_LADDER = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 20.0, 100.0]


def build_base(dt: float, n_steps: int, seed: int,
               obs_stride: int, rec_stride: int, part_stride: int,
               kon: float = REF_KON, k_bond: float = 10.0,
               eqq: float = None, eff: float = None, eqf: float = None) -> "qtft.SimulationConfig":
    """Reference-condition config at a given production dt / step count / strides.

    kon and k_bond are exposed so Phase 1 can raise kon (populate bonds fast) and Phase 3
    can sweep k_bond (force recalibration). eqq/eff/eqf override the LJ well depths so the
    correct Phase-3 lever — softening the repulsive CORE (the agglomerated-state limiter) —
    can be tested; None keeps the reference value.
    """
    return qtft.SimulationConfig(
        qt=qtft.ParticleConfig("Qt", radius=21.0, diffusion=0.5, cluster_diffusion=0.3),
        ft=qtft.ParticleConfig("Ft", radius=6.0, diffusion=1.0, cluster_diffusion=0.7),
        topology=qtft.TopologyConfig(binding_radius=27.25, kon=kon, k_bond=k_bond,
                                     ft_monovalent=REF["mono"]),
        lj=qtft.LennardJonesConfig(epsilon_QtQt=eqq if eqq is not None else REF["eqq"],
                                   epsilon_FtFt=eff if eff is not None else REF["eff"],
                                   epsilon_QtFt=eqf if eqf is not None else REF["eqf"],
                                   potential_type=REF["pot"]),
        box_size=(500., 500., 500.), temperature=300.0,
        timestep=dt, n_steps=n_steps,
        record_stride=rec_stride, observable_stride=obs_stride,
        particles_observable_stride=part_stride,
        heavy_observable_stride=max(1, n_steps),      # forces/virial: record ~once (unused, keep files tiny)
        n_qt=REF["nqt"], n_ft=REF["nft"], kernel="CPU", n_threads=THREADS, rng_seed=seed,
    )


def strides_for(n_steps: int):
    """Per-dt strides: ~250 energy samples, ~40 position frames (min 1, max n_steps)."""
    obs = min(n_steps, max(1, n_steps // 250))
    part = min(n_steps, max(1, n_steps // 40))
    rec = part
    return obs, rec, part


def write_slurm(ens: "EnsembleSimulation", equil_dt: float, equil_steps: int,
                walltime: str, label: str, run_timeout: int) -> str:
    """Write a job-array SLURM that runs each replica via dt_ladder_run.py (absolute paths).

    The worker is wrapped in ``timeout`` so a run that EXPLODES and hangs ReaDDy's neighbour
    list (large-dt behaviour: positions blow up -> C++ spins, never NaNs or returns) is killed
    fast (exit 124) instead of squatting on a node until walltime. A killed run leaves a
    partial trajectory but no stability.json; the analyzer treats that as UNSTABLE (hang).
    """
    out_dir = os.path.abspath(ens.output_dir).rstrip("/")
    logs = os.path.join(out_dir, "logs")
    cfg = os.path.join(out_dir, "configs")
    os.makedirs(logs, exist_ok=True)
    conda_activate = f"source {CONDA_BASE}/bin/activate {CONDA_BASE}/envs/{CONDA_ENV}"
    equil_flag = f"--equil-dt {equil_dt} --equil-steps {equil_steps}"
    script = f"""#!/bin/bash
#SBATCH --job-name=dt_{label[:18]}
#SBATCH --clusters={CLUSTER}
#SBATCH -p {PARTITION}
#SBATCH --qos={QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPUS}
#SBATCH --mem={MEM}
#SBATCH --time={walltime}
#SBATCH --array=0-{ens.n_replicas - 1}
#SBATCH --output={logs}/replica_%a.out
#SBATCH --error={logs}/replica_%a.err
#SBATCH --get-user-env

{conda_activate}
cd {REPO}
REPLICA_IDX=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
echo "dt-ladder replica $SLURM_ARRAY_TASK_ID  ({label})  $(date)"
timeout --signal=KILL {run_timeout} python lrz/dt_ladder_run.py --config {cfg}/config_${{REPLICA_IDX}}.json {equil_flag}
rc=$?
if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
  echo "TIMEOUT_KILLED after {run_timeout}s (rc=$rc) -> treat as UNSTABLE_HANG"
  mkdir -p {out_dir}/replica_${{REPLICA_IDX}}
  echo '{{"heuristic_verdict": "UNSTABLE_HANG", "production_completed": false, "stage_failed": "production_timeout"}}' \\
    > {out_dir}/replica_${{REPLICA_IDX}}/stability_timeout.json
fi
echo "done replica $SLURM_ARRAY_TASK_ID rc=$rc  $(date)"
"""
    path = os.path.join(out_dir, "submit_ladder.slurm")
    with open(path, "w") as f:
        f.write(script)
    return path


def make_ensemble(dt: float, sim_us: float, reps: int, base_seed: int, phase_dir: str,
                  kon: float = REF_KON, k_bond: float = 10.0, tag: str = None,
                  eqq: float = None, eff: float = None, eqf: float = None):
    """Create the EnsembleSimulation for one dt (folder + per-replica configs + ensemble_config).

    ``tag`` prefixes the folder name (e.g. "kb1"); required for Phase 3 because the canonical
    param string does NOT encode k_bond, so different k_bond at the same dt would collide.
    """
    n_steps = max(1, round(sim_us * 1000.0 / dt))
    obs, rec, part = strides_for(n_steps)
    base = build_base(dt, n_steps, base_seed, obs, rec, part, kon=kon, k_bond=k_bond,
                      eqq=eqq, eff=eff, eqf=eqf)
    ens = EnsembleSimulation(base_config=base, n_replicas=reps, base_dir=phase_dir, name=tag)
    os.makedirs(os.path.join(ens.output_dir, "configs"), exist_ok=True)
    os.makedirs(os.path.join(ens.output_dir, "logs"), exist_ok=True)
    ens._save_replica_configs()
    # ensemble_config.json lets struct_descriptors/analysis pick up box size etc.
    json.dump(base.to_dict(), open(os.path.join(ens.output_dir, "ensemble_config.json"), "w"), indent=2)
    return ens, n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", action="store_true", help="stability ladder (reps=1, sim-us=5 defaults)")
    ap.add_argument("--phase2", action="store_true", help="invariance replicas (reps=4, sim-us=20 defaults)")
    ap.add_argument("--dts", default=None, help="comma list of dt (ns); default = full ladder")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--sim-us", type=float, default=None, help="fixed simulated-time budget per probe (us)")
    ap.add_argument("--equil-dt", type=float, default=0.05)
    ap.add_argument("--equil-steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345, help="base RNG seed (distinct from campaign seed=1)")
    ap.add_argument("--walltime", default=None)
    ap.add_argument("--timeout", type=int, default=None,
                    help="per-replica wall timeout (s); exploded/hung runs are KILLed and marked UNSTABLE_HANG")
    ap.add_argument("--kon", type=float, default=None,
                    help="binding rate override; Phase1 defaults HIGH (0.5) to populate bonds fast, Phase2 uses 0.001")
    ap.add_argument("--kbond", type=float, default=10.0, help="harmonic bond force constant (Phase 3 sweeps this)")
    ap.add_argument("--tag", default=None, help="folder-name prefix (Phase 3: e.g. kb1) so different k_bond don't collide")
    ap.add_argument("--eqq", type=float, default=None, help="override epsilon_QtQt (LJ core softening test)")
    ap.add_argument("--eff", type=float, default=None, help="override epsilon_FtFt (LJ core softening test)")
    ap.add_argument("--eqf", type=float, default=None, help="override epsilon_QtFt")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--array", default=None, help="override --array (e.g. '0' for a 1-replica canary)")
    ap.add_argument("--outdir", default=None, help="override phase output dir (default REPO/dt_ladder/<phase>)")
    args = ap.parse_args()

    if not (args.phase1 or args.phase2):
        sys.exit("specify --phase1 or --phase2")
    phase = "phase1" if args.phase1 else "phase2"
    reps = args.reps if args.reps is not None else (3 if args.phase1 else 4)
    sim_us = args.sim_us if args.sim_us is not None else (5.0 if args.phase1 else 20.0)
    walltime = args.walltime or ("00:30:00" if args.phase1 else "12:00:00")
    run_timeout = args.timeout if args.timeout is not None else (420 if args.phase1 else 3600)
    # Phase 1 raises kon 100x so stiff bonds populate within the first few production steps
    # (the bond mode is the true limiter), without instantly percolating everything; Phase 2/3
    # must use the REAL kon for faithful statistics.
    kon = args.kon if args.kon is not None else (0.1 if args.phase1 else REF_KON)
    dts = [float(x) for x in args.dts.split(",")] if args.dts else list(DEFAULT_LADDER)
    phase_dir = args.outdir or os.path.join(REPO, "dt_ladder", phase)
    os.makedirs(phase_dir, exist_ok=True)

    print(f"REPO={REPO}")
    print(f"phase={phase} dts={dts} reps={reps} sim_us={sim_us} kon={kon} k_bond={args.kbond}")
    print(f"equil_dt={args.equil_dt} walltime={walltime} run_timeout={run_timeout}s")
    print(f"cluster={CLUSTER}/{PARTITION}/{QOS} cpus={CPUS} mem={MEM}\n")

    index_path = os.path.join(phase_dir, "dt_ladder_index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    submits = []
    for dt in dts:
        ens, n_steps = make_ensemble(dt, sim_us, reps, args.seed, phase_dir, kon=kon,
                                     k_bond=args.kbond, tag=args.tag,
                                     eqq=args.eqq, eff=args.eff, eqf=args.eqf)
        label = f"{args.tag + '_' if args.tag else ''}dt{dt:g}ns"
        slurm = write_slurm(ens, args.equil_dt, args.equil_steps, walltime, label, run_timeout)
        submits.append((label, slurm))
        index[label] = {
            "dt_ns": dt, "n_steps": n_steps, "sim_us": sim_us, "reps": reps,
            "kon": kon, "k_bond": args.kbond, "tag": args.tag,
            "dir": os.path.abspath(ens.output_dir).rstrip("/"), "slurm": slurm,
            "equil_dt": args.equil_dt, "equil_steps": args.equil_steps,
        }
        print(f"  dt={dt:<7g} n_steps={n_steps:<10,} -> {os.path.relpath(ens.output_dir, REPO)}")
    json.dump(index, open(index_path, "w"), indent=2)
    print(f"\nindex -> {index_path}")

    sh = os.path.join(phase_dir, "submit_ladder_all.sh")
    with open(sh, "w") as f:
        f.write("#!/bin/bash\nset -e\n")
        for lab, s in submits:
            f.write(f'echo "== {lab} =="; sbatch "{s}"\n')
    os.chmod(sh, 0o755)
    print(f"submit_all -> {sh}")

    if args.submit:
        print("\nSUBMITTING:")
        for lab, s in submits:
            cmd = ["sbatch"]
            if args.array is not None:
                cmd += [f"--array={args.array}"]
            cmd += [s]
            r = subprocess.run(cmd, capture_output=True, text=True)
            print(f"  {lab}: {r.stdout.strip() or r.stderr.strip()}")


if __name__ == "__main__":
    main()
