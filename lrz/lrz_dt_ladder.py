#!/usr/bin/env python
"""dt-stability ladder for the ReaDDy Qt-Ft model on LRZ.

Runs the reference condition (200 Qt / 400 Ft, LJ) at a range of integration timesteps,
each to a MATCHED simulated time, to find the largest dt that is (a) numerically stable
and (b) reproduces the equilibrium structure of the dt=0.05 ns baseline. See
lrz/README/HANDOVER. kon is a rate (1/ns) so kinetics auto-rescale with dt (no manual
rescale). Equilibration steps scaled to a matched simulated time too.

  python lrz/lrz_dt_ladder.py            # generate configs + submit.slurm + ladder.json
  python lrz/lrz_dt_ladder.py --submit   # + sbatch the array
  python lrz/lrz_dt_ladder.py --analyze  # struct_descriptors per dt + tabulate vs baseline
"""
import os, sys, json, argparse, subprocess, glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import qtft

HOME = os.path.expanduser("~")
CONDA_BASE = os.path.join(HOME, "miniforge3"); CONDA_ENV = "readdy"
STRUCT = os.path.join(REPO, "lrz", "struct_descriptors.py")
LAD = os.path.join(REPO, "dt_ladder")
T_NS = 10000.0        # 10 us matched simulated time per probe
EQUIL_NS = 500.0      # 0.5 us matched equilibration per probe
DTS = [0.05, 0.2, 1.0, 5.0, 25.0, 125.0]   # ns


def build(dt):
    n_steps = max(20, round(T_NS / dt))
    obs = max(1, n_steps // 100)
    cfg = qtft.SimulationConfig(
        qt=qtft.ParticleConfig("Qt", radius=21.0, diffusion=0.5, cluster_diffusion=0.3),
        ft=qtft.ParticleConfig("Ft", radius=6.0, diffusion=1.0, cluster_diffusion=0.7),
        topology=qtft.TopologyConfig(binding_radius=27.25, kon=0.001, k_bond=10.0),
        lj=qtft.LennardJonesConfig(epsilon_QtQt=1.5, epsilon_FtFt=1.5,
                                   epsilon_QtFt=3.0, potential_type="LJ"),
        box_size=(500., 500., 500.), temperature=300.0, timestep=float(dt),
        n_steps=int(n_steps), record_stride=obs, observable_stride=obs,
        particles_observable_stride=obs,
        n_qt=200, n_ft=400, kernel="CPU", n_threads=4, rng_seed=1,
    )
    return cfg, n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()

    cfgdir = os.path.join(LAD, "configs"); os.makedirs(cfgdir, exist_ok=True)
    ladder = []
    for i, dt in enumerate(DTS):
        cfg, n_steps = build(dt)
        rundir = os.path.join(LAD, f"run_{i}_dt{dt}ns", "replica_000")
        os.makedirs(rundir, exist_ok=True)
        cfg.output_file = os.path.join(rundir, "trajectory.h5")
        cfgpath = os.path.join(cfgdir, f"config_{i}.json")
        cfg.save_json(cfgpath)
        eq = max(4, round(EQUIL_NS / dt))
        ladder.append({"i": i, "dt": dt, "n_steps": int(n_steps), "equil": int(eq),
                       "config": cfgpath, "ensemble_dir": os.path.dirname(rundir)})

    if args.analyze:
        os.makedirs(os.path.join(LAD, "results"), exist_ok=True)
        rows = []
        for e in ladder:
            d = e["ensemble_dir"]; label = f"dt_{e['dt']}ns"
            out = os.path.join(LAD, "results", label)
            traj = glob.glob(os.path.join(d, "replica_*/trajectory.h5"))
            if not traj:
                rows.append({**e, "status": "NO_TRAJECTORY"}); continue
            try:
                subprocess.run([sys.executable, STRUCT, "--ensemble-dir", d,
                                "--out", out, "--label", label], check=True,
                               stdout=open(out + "_desc.log", "w"), stderr=subprocess.STDOUT)
                s = json.load(open(out + "_summary.json"))
                rows.append({**e, "status": "OK",
                             "kappa2": s.get("mean_shape_anisotropy_kappa2"),
                             "coord": s.get("mean_Qt_coordination"),
                             "largest_frac": s.get("largest_cluster_fraction"),
                             "df": s.get("fractal_dimension_df")})
            except Exception as ex:
                rows.append({**e, "status": f"FAIL:{type(ex).__name__}"})
        json.dump(rows, open(os.path.join(LAD, "ladder_results.json"), "w"), indent=2)
        base = next((r for r in rows if r["dt"] == 0.05 and r.get("status") == "OK"), None)
        print(f"{'dt(ns)':>8} {'steps':>8} {'status':>12} {'kappa2':>8} {'coord':>7} {'lgf':>6} {'Δκ²%':>7}")
        for r in rows:
            k = r.get("kappa2"); dk = (100*(k-base["kappa2"])/base["kappa2"]) if (base and k) else float('nan')
            print(f"{r['dt']:>8} {r['n_steps']:>8} {r['status']:>12} "
                  f"{(k if k else float('nan')):>8.3f} {(r.get('coord') or float('nan')):>7.2f} "
                  f"{(r.get('largest_frac') or float('nan')):>6.3f} {dk:>7.1f}")
        print(f"\nBaseline dt=0.05 ns: kappa2={base['kappa2'] if base else 'NA'}")
        return

    json.dump(ladder, open(os.path.join(LAD, "ladder.json"), "w"), indent=2)
    # SLURM array: one task per dt (bash arrays carry per-task config + equil steps)
    cfgs = " ".join(f'"{e["config"]}"' for e in ladder)
    eqs = " ".join(str(e["equil"]) for e in ladder)
    slurm = f"""#!/bin/bash
#SBATCH --job-name=dtladder
#SBATCH -p serial_std
#SBATCH --clusters=serial
#SBATCH --qos=cm4_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --array=0-{len(ladder)-1}
#SBATCH --output={LAD}/logs/dt_%a.out
#SBATCH --error={LAD}/logs/dt_%a.err
#SBATCH --get-user-env

source {CONDA_BASE}/bin/activate {CONDA_BASE}/envs/{CONDA_ENV}
cd {REPO}
CFGS=({cfgs})
EQS=({eqs})
echo "dt-ladder task $SLURM_ARRAY_TASK_ID  cfg=${{CFGS[$SLURM_ARRAY_TASK_ID]}}  equil=${{EQS[$SLURM_ARRAY_TASK_ID]}}  $(date)"
python scripts/run_replica.py --config "${{CFGS[$SLURM_ARRAY_TASK_ID]}}" --equilibration-steps "${{EQS[$SLURM_ARRAY_TASK_ID]}}"
echo "done $(date)"
"""
    os.makedirs(os.path.join(LAD, "logs"), exist_ok=True)
    sp = os.path.join(LAD, "submit.slurm")
    open(sp, "w").write(slurm)
    print(f"configs + ladder.json + {sp} written ({len(ladder)} dt values: {DTS})")
    if args.submit:
        r = subprocess.run(["sbatch", sp], capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())


if __name__ == "__main__":
    main()
