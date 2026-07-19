#!/usr/bin/env python
"""Deploy a LONG ReaDDy run on LRZ at an enlarged, stability-validated timestep.

Reference condition (200 Qt / 400 Ft, LJ) at dt = 2 ns (40x the campaign's 0.05 ns),
reaching 4 ms of simulated time in 2e6 steps -- i.e. the SAME step budget / wall-clock as
the 100 us campaign run, but 40x longer simulated time. dt = 2 ns is well inside the
stable regime found by lrz_dt_ladder.py (dt<=5 ns equilibrates healthily; dt>=125 ns is
unstable). kon is a rate (1/ns) so kinetics auto-rescale with dt -- no manual change.

  python lrz/lrz_longrun.py            # generate SLURM (no submit)
  python lrz/lrz_longrun.py --submit   # + sbatch
"""
import os, sys, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import qtft
from qtft import EnsembleSimulation

HOME = os.path.expanduser("~")
CONDA_BASE = os.path.join(HOME, "miniforge3"); CONDA_ENV = "readdy"
DT_NS = 2.0
N_STEPS = 2_000_000            # 2e6 * 2 ns = 4 ms simulated time
REPS = 3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--partition", default="serial_std")   # serial_long for >1 day
    ap.add_argument("--qos", default="cm4_serial")
    ap.add_argument("--time", default="24:00:00")
    args = ap.parse_args()

    base = qtft.SimulationConfig(
        qt=qtft.ParticleConfig("Qt", radius=21.0, diffusion=0.5, cluster_diffusion=0.3),
        ft=qtft.ParticleConfig("Ft", radius=6.0, diffusion=1.0, cluster_diffusion=0.7),
        topology=qtft.TopologyConfig(binding_radius=27.25, kon=0.001, k_bond=10.0),
        lj=qtft.LennardJonesConfig(epsilon_QtQt=1.5, epsilon_FtFt=1.5,
                                   epsilon_QtFt=3.0, potential_type="LJ"),
        box_size=(500., 500., 500.), temperature=300.0, timestep=DT_NS,
        n_steps=N_STEPS, record_stride=5000, observable_stride=1000,
        particles_observable_stride=1000,
        n_qt=200, n_ft=400, kernel="CPU", n_threads=4, rng_seed=1,
    )
    sim_us = N_STEPS * DT_NS / 1000.0
    print(f"long run: dt={DT_NS} ns x {N_STEPS} steps = {sim_us:.0f} us = {sim_us/1000:.1f} ms, "
          f"{REPS} replicas, partition={args.partition}")
    ens = EnsembleSimulation(base_config=base, n_replicas=REPS, base_dir=REPO)
    ens.generate_slurm_scripts(
        partition=args.partition, cluster="serial", qos=args.qos, time=args.time,
        cpus_per_task=4, memory="6G",
        conda_base=CONDA_BASE, conda_env=CONDA_ENV, scripts_dir=REPO,
    )
    slurm = os.path.join(ens.output_dir, "submit_ensemble.slurm")
    print(f"-> {slurm}")
    if args.submit:
        import subprocess
        r = subprocess.run(["sbatch", slurm], capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())

if __name__ == "__main__":
    main()
