#!/usr/bin/env python
"""Generate (and optionally submit) SLURM job-arrays for the 11-condition CLEM-GECI
ReaDDy Qt-Ft campaign on the LRZ CoolMUC-4 (cm4) cluster.

This is the cluster twin of ~/overnight_campaign.py (which ran the same 11 conditions
locally via run_local on the ambigram stopgap). Same Table-3.6 parameters, same
particle counts, same replica counts -> a native-linux reproduction directly
comparable to the ambigram osx-64/Rosetta results.

Per condition it builds the base config, creates the ensemble (writing per-replica
configs) and emits <ensemble>/submit_ensemble.slurm via
EnsembleSimulation.generate_slurm_scripts(), overriding the generator's foreign-home
defaults with THIS user's cm4/miniforge paths.

Usage (run on a cm4 login node inside the `readdy` env):
    python lrz/lrz_campaign.py                 # generate scripts + submit_all.sh (no submit)
    python lrz/lrz_campaign.py --only ratio_200Qt_200Ft --submit   # canary one condition
    python lrz/lrz_campaign.py --submit        # generate AND sbatch all 11
"""
import os, sys, argparse, subprocess, json, glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import qtft
from qtft import EnsembleSimulation

STRUCT = os.path.join(REPO, "lrz", "struct_descriptors.py")
RESULTS = os.path.join(REPO, "results")
INDEX = os.path.join(REPO, "campaign_index.json")

HOME = os.path.expanduser("~")
CONDA_BASE = os.path.join(HOME, "miniforge3")
CONDA_ENV = "readdy"
# LRZ SLURM knobs. Small single-process ReaDDy replicas (4 threads, <2 GB) belong on
# the `serial` cluster, NOT cm4_std/cm4_tiny: those QOS enforce a large-core floor
# (cm4_tiny MinTRES cpu=17, cm4_std cpu=112). serial_std/QOS cm4_serial allows up to
# 32 cores/job, MaxSubmit=200, 1-day walltime -- ideal for many independent replicas.
CLUSTER = "serial"
PARTITION = "serial_std"
QOS = "cm4_serial"
CPUS = 4            # == n_threads below, matches the ambigram runs (n_threads=4)
THREADS = 4
MEM = "6G"
WALLTIME = "08:00:00"   # well within serial_std's 1-day limit; native replicas finish faster


def C(label, nqt, nft, steps, reps, eqq=1.5, eff=1.5, eqf=3.0, pot="LJ", mono=False):
    return dict(label=label, nqt=nqt, nft=nft, steps=steps, reps=reps,
                eqq=eqq, eff=eff, eqf=eqf, pot=pot, mono=mono)

# Identical to overnight_campaign.CONDITIONS (the ambigram driver).
CONDITIONS = [
    C("ref_200Qt_400Ft", 200, 400, 2_000_000, 4),
    C("ratio_200Qt_200Ft", 200, 200, 1_000_000, 3),
    C("ratio_400Qt_200Ft", 400, 200, 1_000_000, 3),
    C("ratio_600Qt_50Ft",  600,  50, 1_000_000, 3),
    C("ratio_600Qt_200Ft", 600, 200, 1_000_000, 3),
    C("ratio_200Qt_1000Ft", 200, 1000, 1_000_000, 3),
    C("valency_monoFt_200Qt_400Ft", 200, 400, 1_000_000, 3, mono=True),
    C("spacer_eQQ0.5_200Qt_400Ft", 200, 400, 1_000_000, 3, eqq=0.5),
    C("spacer_eQQ0.1_200Qt_400Ft", 200, 400, 1_000_000, 3, eqq=0.1),
    C("mech_WCA_multi_200Qt_400Ft", 200, 400, 1_000_000, 2, pot="WCA"),
    C("mech_WCA_mono_200Qt_400Ft",  200, 400, 1_000_000, 2, pot="WCA", mono=True),
]


def build_base(c, threads=THREADS):
    return qtft.SimulationConfig(
        qt=qtft.ParticleConfig("Qt", radius=21.0, diffusion=0.5, cluster_diffusion=0.3),
        ft=qtft.ParticleConfig("Ft", radius=6.0, diffusion=1.0, cluster_diffusion=0.7),
        topology=qtft.TopologyConfig(binding_radius=27.25, kon=0.001, k_bond=10.0,
                                     ft_monovalent=c["mono"]),
        lj=qtft.LennardJonesConfig(epsilon_QtQt=c["eqq"], epsilon_FtFt=c["eff"],
                                   epsilon_QtFt=c["eqf"], potential_type=c["pot"]),
        box_size=(500., 500., 500.), temperature=300.0, timestep=0.05,
        n_steps=c["steps"], record_stride=5000, observable_stride=1000,
        particles_observable_stride=1000,
        n_qt=c["nqt"], n_ft=c["nft"], kernel="CPU", n_threads=threads, rng_seed=1,
    )


def postprocess():
    """Compute structural descriptors for every finished ensemble (uses the label->dir
    index written at generation time; never reconstructs the ensemble, so it cannot
    clobber trajectory data). Robust: a failing condition is logged, not fatal. Builds
    manifest_lrz.json in the same schema as the ambigram results/manifest.json."""
    os.makedirs(RESULTS, exist_ok=True)
    idx = json.load(open(INDEX)) if os.path.exists(INDEX) else {}
    mpath = os.path.join(RESULTS, "manifest_lrz.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for label, info in idx.items():
        d = info["dir"]
        got = len(glob.glob(os.path.join(d, "replica_*/trajectory.h5")))
        out = os.path.join(RESULTS, label)
        print(f"== {label}: {got}/{info['reps']} replicas present -> {d}", flush=True)
        if got == 0:
            manifest[label] = {"error": "no replicas produced", "params": info}
            continue
        try:
            subprocess.run([sys.executable, STRUCT, "--ensemble-dir", d,
                            "--out", out, "--label", label], check=True,
                           stdout=open(out + "_desc.log", "w"), stderr=subprocess.STDOUT)
            summ = json.load(open(out + "_summary.json"))
            manifest[label] = {k: summ.get(k) for k in (
                "fractal_dimension_df", "df_stderr", "df_fit_R2", "df_n_clusters_fit",
                "mean_Qt_coordination", "mean_shape_anisotropy_kappa2", "mean_aspect_ratio",
                "largest_cluster_fraction", "gr_bridging_peak_nm", "d_cut_nm",
                "rough_Qt_fraction_mean")}
            manifest[label]["n_replicas_done"] = got
            print(f"   df={summ.get('fractal_dimension_df')} "
                  f"kappa2={summ.get('mean_shape_anisotropy_kappa2')} "
                  f"lgf={summ.get('largest_cluster_fraction')}", flush=True)
        except Exception as e:
            manifest[label] = {"error": f"{type(e).__name__}: {e}"}
            print(f"   ERROR {label}: {e}", flush=True)
        json.dump(manifest, open(mpath, "w"), indent=2)
    print(f"\nPOSTPROCESS_COMPLETE -> {mpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of labels to include")
    ap.add_argument("--submit", action="store_true", help="sbatch each generated array")
    ap.add_argument("--array", default=None, help="override --array (e.g. '0' for a 1-replica canary)")
    ap.add_argument("--postprocess", action="store_true",
                    help="compute descriptors + manifest_lrz.json for finished ensembles (no gen/submit)")
    args = ap.parse_args()

    if args.postprocess:
        postprocess()
        return

    conds = CONDITIONS
    if args.only:
        want = set(args.only.split(","))
        conds = [c for c in CONDITIONS if c["label"] in want]
        if not conds:
            sys.exit(f"no conditions matched {want}")

    print(f"REPO={REPO}\nCONDA_BASE={CONDA_BASE} ENV={CONDA_ENV}")
    print(f"cluster={CLUSTER} partition={PARTITION} cpus={CPUS} mem={MEM} time={WALLTIME}")
    print(f"generating {len(conds)} ensemble(s)\n")

    index = json.load(open(INDEX)) if os.path.exists(INDEX) else {}
    submits = []
    for c in conds:
        base = build_base(c)
        ens = EnsembleSimulation(base_config=base, n_replicas=c["reps"], base_dir=REPO)
        ens.generate_slurm_scripts(
            partition=PARTITION, cluster=CLUSTER, qos=QOS, time=WALLTIME,
            cpus_per_task=CPUS, memory=MEM,
            conda_base=CONDA_BASE, conda_env=CONDA_ENV, scripts_dir=REPO,
        )
        slurm = os.path.join(ens.output_dir, "submit_ensemble.slurm")
        submits.append((c["label"], c["reps"], slurm))
        index[c["label"]] = {"dir": ens.output_dir, "reps": c["reps"], **c}
        print(f"  {c['label']:32s} reps={c['reps']}  -> {slurm}")
    json.dump(index, open(INDEX, "w"), indent=2)
    print(f"campaign_index.json -> {INDEX}")

    # write a submit_all.sh for the record
    sh = os.path.join(REPO, "submit_all.sh")
    with open(sh, "w") as f:
        f.write("#!/bin/bash\nset -e\n")
        for lab, reps, s in submits:
            f.write(f'echo "== {lab} =="; sbatch "{s}"\n')
    os.chmod(sh, 0o755)
    print(f"\nsubmit_all.sh -> {sh}")

    if args.submit:
        print("\nSUBMITTING:")
        for lab, reps, s in submits:
            cmd = ["sbatch"]
            if args.array is not None:
                cmd += [f"--array={args.array}"]
            cmd += [s]
            r = subprocess.run(cmd, capture_output=True, text=True)
            print(f"  {lab}: {r.stdout.strip() or r.stderr.strip()}")


if __name__ == "__main__":
    main()
