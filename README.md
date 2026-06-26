# Qt–Ft Agglomeration Simulation (ReaDDy2)

Coarse-grained Brownian-dynamics simulation of **Qt encapsulins** and **ferritin (Ft)**
nanoparticles agglomerating in solution, built on [ReaDDy2](https://readdy.github.io/).
Two diffusing species bind into growing clusters ("topologies") through stochastic spatial
reactions; the code measures the resulting **agglomeration kinetics** and **cluster
morphology**, across single runs and multi-replica ensembles, locally or on a SLURM cluster.

The overall pipeline is:

```
configure ─▶ equilibrate (WCA, no reactions) ─▶ production (LJ + reactions)
          ─▶ analyze trajectory ─▶ plot ─▶ (ensemble averaging / cross-ensemble comparison)
```

---

## 1. Physical model

**Species.** Two free particle types plus their auto-derived "clustered" counterparts, all
managed as ReaDDy *topology species* inside a single topology type `QtFt_Cluster`:

| Symbol | Meaning              | State            |
|--------|----------------------|------------------|
| `Qt`   | Qt encapsulin        | free / monomer   |
| `Ft`   | Ferritin             | free / monomer   |
| `QtC`  | Qt in a cluster      | bound            |
| `FtC`  | Ft in a cluster      | bound            |

**Reactions** (spatial topology reactions, Gillespie handler). All fire when two eligible
particles come within `binding_radius`, at rate `kon`:

| Name                       | Reaction                          | Role                       |
|----------------------------|-----------------------------------|----------------------------|
| `seed_QtFt_Cluster`        | `Qt + Ft → QtC–FtC`               | nucleate a new cluster     |
| `grow_QtC_Ft_QtFt_Cluster` | `QtC + Ft → QtC–FtC`              | cluster captures a free Ft |
| `grow_FtC_Qt_QtFt_Cluster` | `FtC + Qt → FtC–QtC`              | cluster captures a free Qt |
| `merge_QtC_FtC_QtFt_Cluster`| `QtC + FtC → QtC–FtC`            | two clusters merge         |

**Monovalent Ft (`topology.ft_monovalent`, default `False`).** ReaDDy has no built-in bond cap;
valence is governed purely by which particle types appear as reactants. In both `seed` and
`grow_QtC_Ft` the particle that becomes `FtC` is a *free* Ft gaining its first bond, whereas
`grow_FtC_Qt` and `merge_QtC_FtC` are the only reactions that give an already-bonded `FtC` a
*second* bond. Setting `ft_monovalent=True` skips those two reactions, so `FtC` is terminal and
every Ft forms **at most one bond**. Clusters then become **single-Qt stars** (one multivalent Qt
hub + N monovalent Ft leaves): two clusters never merge, and a free Qt joins only by seeding with
a free Ft. Qt stays multivalent. Default `False` reproduces the original multivalent model.

**Potentials.**
- **Pairwise Lennard-Jones** for excluded volume, registered for all 10 type pairs.
  `potential_type="WCA"` → purely repulsive (cutoff `2^(1/6)·σ`); `"LJ"` → full attractive
  well (cutoff `2.5·σ`). σ is set so the LJ minimum / WCA exclusion fall at the contact
  distance: `σ = (r_i + r_j) / 2^(1/6) ≈ 0.8909·(r_i + r_j)`, which puts the well minimum at
  `r_i + r_j` = the harmonic bond length (see §11a, P2). ε is set per pair through a cascade of
  defaults (see §6).
- **Harmonic bonds** (`k_bond`) hold bonded particles inside a cluster at equilibrium length
  `r_Qt + r_Ft`.

**Equilibration vs production.** Equilibration runs with **reactions disabled** and a
purely-repulsive **WCA** potential (the `equilibration_potential` config field, default
`"WCA"`) to relax initial random positions without attraction; production then switches on
attractions (LJ via `lj.potential_type`) and the binding reactions. This split is handled by
`equilibrate_system()` + `run_simulation()`. Set `equilibration_potential="LJ"` to equilibrate
under the full attractive potential instead.

**Deagglomeration & cycling (`config.phases`).** A run can be split into a sequence of *phases*
to model an **agglomeration ↔ deagglomeration cycle** (e.g. bind for 50 µs, then dissolve for
50 µs, repeatably). Each `PhaseConfig` specifies `n_steps` plus the physics for that phase:
`binding` (spatial binding reactions on/off), `breaking` (bond breaking on/off), and
`potential_type` (`"LJ"` attractive or `"WCA"` repulsive). Bond breaking uses **structural
topology dissociation**: a bonded cluster loses one uniformly-random bond at total rate
`n_edges × topology.koff`, splitting into sub-clusters; a freed monomer is automatically re-typed
back to its free species (`QtC→Qt`, `FtC→Ft`). A typical cycle is agglomeration (binding on,
breaking off, `LJ`) then deagglomeration (binding off, breaking on, `WCA` so freed particles
disperse); build it with `make_agg_deagg_phases(agg_steps, deagg_steps, n_cycles=...)`. Because
ReaDDy cannot change reactions mid-run, `run_phased()` runs each phase as a separate segment and
carries state (positions **and** bonds) across phases via ReaDDy checkpoints. Each phase writes its
own `phase_NNN/trajectory.h5`; analysis stitches them onto one continuous time axis
(`analysis.load_phased_observables`, `plotting.plot_phased_kinetics`). When `phases` is unset
(default), behavior and on-disk filenames are exactly as before. Works for single runs and
ensembles alike (see §6). Single dispatch point: `run_one()` calls `run_phased()` automatically
when `config.phases` is set.

**Integrator / environment.** EulerBD Brownian-dynamics integrator, Gillespie reactions,
cubic periodic box, `T = 300 K`.

---

## 2. Repository layout

All code lives in the **`qtft`** package; `scripts/` holds thin CLI wrappers.

| Module | Purpose |
|--------|---------|
| `qtft.config` | Config dataclasses (`SimulationConfig` etc.), the `format_param_string` naming convention, and the `NS_TO_US`/`_steps_to_us` units helpers. Single source of truth. |
| `qtft.system` | ReaDDy system builders: `create_system`, species/potentials/topologies. |
| `qtft.engine` | Build + run: `create_simulation`, `place_particles`, `run_simulation`, `equilibrate_system`, and the one-shot `run_one`. |
| `qtft.ensemble` | `EnsembleSimulation` class — multi-replica orchestration, local/parallel runs, SLURM script generation, result collection, statistics, save/load. |
| `qtft.analysis` | Matplotlib-free trajectory analysis: cluster stats, bond counts, binding kinetics, morphology (Rg), spatial distribution, contacts, composition, size fractions. Also `convert_h5_to_xyz` (OVITO), `load_ensemble_data`, and numeric results tables (`build_final_state_table`, `save_table_files`). |
| `qtft.plotting` | All matplotlib plots: single-run, ensemble, and cross-ensemble comparison figures. |
| `qtft.comparison` | Cross-ensemble comparison helpers (`compare_ensembles`, `save/load_comparison_data`, `build_comparison_table`, …). |
| `scripts/analyze_ensemble.py` | CLI to (re)analyze an ensemble directory in parallel; `compare` subcommand. |
| `scripts/run_replica.py` | CLI to run **one** replica from a config JSON (used locally and by SLURM job arrays). |
| `Run_Simulation.ipynb` | Main user notebook: configure → single run + plots → build & run ensemble → XYZ export. |
| `Plot_Ensemble_Results.ipynb` | Load a finished ensemble, plot results, compare ensembles, export CSV. |
| `Different_Particle_Ratios/` | Stored ensemble datasets (see §13). |

---

## 3. Requirements

- Python 3.x with **ReaDDy2** (install via conda; the SLURM scripts assume a conda env named
  `readdy`):
  ```bash
  conda create -n readdy -c readdy -c conda-forge readdy
  conda activate readdy
  ```
- `numpy`, `matplotlib`, `pandas`, `h5py` (pulled in by ReaDDy / standard scientific stack).
- For visualization of `.xyz` exports: [OVITO](https://www.ovito.org/) (external, optional).

---

## 4. Quick start — single run

Mirrors the first half of `Run_Simulation.ipynb`:

```python
import qtft as sim
import qtft.analysis as analysis
import qtft.plotting as plotting

# 1. Configure (current standard values; see §6)
config = sim.SimulationConfig(
    qt=sim.ParticleConfig("Qt", radius=21.0, diffusion=0.5, cluster_diffusion=0.3),
    ft=sim.ParticleConfig("Ft", radius=6.0, diffusion=1.0, cluster_diffusion=0.7),
    topology=sim.TopologyConfig(binding_radius=27.25, kon=0.001, k_bond=10.0),
    lj=sim.LennardJonesConfig(
        epsilon_QtQt=1.5, epsilon_FtFt=1.5, epsilon_QtFt=3.0,
        potential_type="LJ",
    ),
    box_size=(500.0, 500.0, 500.0),
    temperature=300.0,
    timestep=0.05,        # ns  (=50 ps)
    n_steps=2_000_000,    # → 100 µs total
    record_stride=100,
    observable_stride=100,
    particles_observable_stride=1000,
    n_qt=600,
    n_ft=50,
)

# 2. Equilibrate (WCA, no reactions) → 3. build → 4. place → 5. run (LJ + reactions)
pos_qt, pos_ft = sim.equilibrate_system(config, n_steps=10000)
system     = sim.create_system(config)
simulation = sim.create_simulation(system, config, overwrite=True)
sim.place_particles(simulation, config, positions_qt=pos_qt, positions_ft=pos_ft)
trajectory = sim.run_simulation(simulation, config)

# 6. Analyze + plot
analysis.print_analysis_summary(config.output_file, config)
plotting.plot_observables(config.output_file, config, save_path="plots_observables.svg")
plotting.plot_cluster_analysis(config.output_file, config=config, save_path="plots_clusters.svg")

# Optional: export for OVITO, and save the config
analysis.convert_h5_to_xyz(config.output_file, config.output_file.replace(".h5", ".xyz"), config, overwrite=True)
config.save_json("simulation_config.json")
```

`config.output_file` is auto-generated from the parameters if left `None` (see §11).

---

## 5. Configuration reference

`SimulationConfig` (in `qtft.config`) is the single source of truth and is
fully JSON-serializable (`config.save_json(...)` / `SimulationConfig.load_json(...)`,
`from_dict` / `to_dict` / `to_flat_dict`).

The table below leads with the **current standard values used in the real runs** (the notebook /
`Simulation_Files_Single_Runs/` datasets). The code dataclass defaults are small smoke-test
values — see the footnote.

| Parameter | Meaning | Units | Standard value |
|-----------|---------|-------|----------------|
| `qt.radius`, `qt.diffusion` | Qt encapsulin size & diffusion | nm, nm²/ns | 21.0, 0.5 |
| `qt.cluster_diffusion` | Qt diffusion once bound in a cluster | nm²/ns | 0.3 |
| `ft.radius`, `ft.diffusion` | Ft ferritin size & diffusion | nm, nm²/ns | 6.0, 1.0 |
| `ft.cluster_diffusion` | Ft diffusion once bound in a cluster | nm²/ns | 0.7 |
| `n_qt`, `n_ft` | particle counts | – | swept: 200–600 / 50–2000 (notebook: 600 / 50) |
| `topology.binding_radius` | reaction capture distance | nm | 27.25 (≈ r_Qt+r_Ft+buffer) |
| `topology.kon` | binding rate | nm³/(ns·part) | 0.001 |
| `topology.k_bond` | harmonic bond stiffness | kJ/(mol·nm²) | 10.0 |
| `topology.ft_monovalent` | cap Ft at one bond → single-Qt-star clusters (see §1) | – | `False` |
| `topology.koff` | bond-breaking rate per edge (deagglomeration phases only, see §1) | 1/ns | 0.0 |
| `phases` | optional list of `PhaseConfig` for agglomeration↔deagglomeration cycling (see §1); `None` = single run | – | `None` |
| `lj.epsilon_QtQt/FtFt/QtFt` | well depths for the three free pairs | kJ/mol | 1.5 / 1.5 / 3.0 |
| `lj.potential_type` | `"WCA"` (repulsive) or `"LJ"` (attractive) | – | `LJ` for production |
| `box_size` | cubic box edge | nm | (500, 500, 500) |
| `temperature` | – | K | 300 |
| `equilibration_potential` | potential during equilibration (`"WCA"` or `"LJ"`); reactions always off | – | `WCA` |
| `timestep` | integration step | ns | 0.05 (50 ps) |
| `n_steps` | total steps (→ 100 µs) | – | 2,000,000 |
| `record_stride`, `observable_stride` | save cadence | steps | 100 |
| `particles_observable_stride` | per-particle position cadence (`None`=off, saves disk) | steps | 1000 |
| `heavy_observable_stride` | cadence for unread heavy observables (forces, virial); `None`=100×`observable_stride` | steps | optional |
| `kernel`, `n_threads` | `"CPU"`/`"SingleCPU"`, threads | – | CPU, 4+ |
| `rng_seed` | RNG seed (per-replica in ensembles) | – | varies |
| `output_file` | trajectory path (`None` = auto, §11) | – | auto |

> **Code dataclass defaults** (smoke-test only, *not* the real runs): Qt r=1.0 D=5.0,
> Ft r=0.25 D=15.0, `binding_radius=1.5`, `kon=10.0`, `k_bond=20.0`, all ε=10.0,
> `potential_type="WCA"`, `box_size=(50,50,50)`, `timestep=1e-4`, `n_steps=200000`,
> `n_qt=n_ft=200`, `rng_seed=42`.

**Epsilon cascade.** Only the three free–free well depths are normally set; the seven
cluster/mixed pairs inherit them unless overridden:

```
epsilon_QtQt, epsilon_FtFt, epsilon_QtFt        (set these)
  └▶ epsilon_QtCQtC = epsilon_QtQt   (cluster pairs default to free–free)
  └▶ epsilon_FtCFtC = epsilon_FtFt
  └▶ epsilon_QtCFtC = epsilon_QtFt
       └▶ epsilon_QtQtC, epsilon_FtFtC, epsilon_QtCFt, epsilon_QtFtC  (mixed default to cluster)
```

Setting an ε to `0` disables that interaction entirely.

---

## 6. Running ensembles

`EnsembleSimulation` (in `qtft.ensemble`) replicates a base config with
independent RNG seeds, runs the replicas, and aggregates the results.

```python
from qtft import EnsembleSimulation

ensemble = EnsembleSimulation(
    base_config=config,
    n_replicas=10,
    base_dir="Different_Particle_Ratios",   # output root
)

# Run all replicas locally (parallel), then auto-collect + compute statistics
ensemble.run_local(parallel=True, n_workers=10, overwrite=True, equilibration_steps=5000)

# Plot
stats, structural, cfg = ensemble.to_plotting_format()
plotting.plot_ensemble_observables(stats, cfg, structural, show_individual=True,
                                   save_path="ensemble_observables.svg")
plotting.plot_ensemble_structural(stats, structural, cfg, show_individual=True,
                                  save_path="ensemble_structural.svg")
```

`run_local` produces an output directory named from the parameter string (§11) containing:

```
<ensemble_dir>/
├── configs/config_000.json … config_009.json   # per-replica configs (differ only by seed)
├── replica_000/ … replica_009/                  # each has trajectory.h5 (+ optional trajectory.xyz)
├── logs/                                         # stdout/stderr (SLURM runs)
├── ensemble_config.json                          # base configuration
├── ensemble_statistics.json                      # time-series means ± std (+ per-replica traces)
├── ensemble_structural.npz                       # morphology / spatial / contacts / composition / size fractions
├── ensemble_state.json                           # full state for EnsembleSimulation.load()
└── submit_ensemble.slurm / submit_analysis.slurm # if SLURM scripts were generated
```

Reload a finished ensemble with `EnsembleSimulation.load("<ensemble_dir>")`.

**Phased (agglomeration↔deagglomeration) ensembles.** Give `base_config` a `phases` schedule
(see §1) and run the ensemble exactly as above — replicas inherit cycling because they all run
through `run_one()`. Each replica then contains `replica_NNN/phase_000/trajectory.h5 …` instead of a
single `trajectory.h5`, and result collection automatically stitches the phases per replica onto one
continuous time axis before averaging, so `ensemble_statistics.json` / `ensemble_structural.npz` keep
the same format (the phase boundaries are identical across replicas).

---

## 7. Cluster (SLURM) execution

For HPC, generate job-array scripts instead of running locally:

```python
ensemble.generate_slurm_scripts(
    partition="cm4_tiny", cluster="cm4", time="08:00:00",
    cpus_per_task=12, memory="32G",
    conda_base="<CONDA_PATH>", conda_env="readdy",
)
ensemble.generate_analysis_slurm_script(
    partition="cm4_tiny", time="04:00:00", cpus_per_task=4, stride=10,
)
```

This writes `submit_ensemble.slurm` (a job array; each task runs one replica via
`scripts/run_replica.py --config configs/config_NNN.json`) and `submit_analysis.slurm` (runs
`scripts/analyze_ensemble.py` once all replicas finish). Submit them with `sbatch`. The SLURM
scripts ship the `qtft/` package and `scripts/` to the cluster (`scp -r qtft scripts ...`).

`scripts/run_replica.py` can also be invoked directly:

```bash
python scripts/run_replica.py --config configs/config_000.json
python scripts/run_replica.py --config configs/config_000.json --equilibration-steps 20000
python scripts/run_replica.py --config configs/config_000.json --skip-equilibration
```

---

## 8. Analysis & outputs

Re-analyze (or analyze for the first time) an ensemble directory in parallel:

```bash
python scripts/analyze_ensemble.py --ensemble-dir <ensemble_dir> --parallel --n-workers 4 --stride 10
```

This (re)writes `ensemble_statistics.json` and `ensemble_structural.npz`.

**Metrics computed** (`qtft.analysis`):
- **Kinetics:** bond counts over time, binding rate, free vs clustered Qt/Ft, fraction bound, half-times.
- **Cluster stats:** number of clusters, size distribution, average & largest cluster size, adaptive size-category fractions (monomers / small / medium / large / very large).
- **Morphology:** radius of gyration Rg per cluster and normalized compactness (Rg/Rg_ideal).
- **Spatial:** cluster centers (PBC-aware), inter- and intra-cluster nearest-neighbor distances.
- **Contacts:** coordination numbers per particle type, bonds per cluster.
- **Composition:** Qt-fraction per cluster and vs cluster size.
- **RDF:** Qt/QtC–Ft/FtC radial distribution (registered as a ReaDDy observable).

**Output-file inventory:**

| File | Format | Contents |
|------|--------|----------|
| `replica_NNN/trajectory.h5` | HDF5 (ReaDDy) | frames + observables; read with `readdy.Trajectory(path)` |
| `replica_NNN/trajectory.xyz` | extended XYZ | OVITO-friendly export (large; optional) |
| `ensemble_statistics.json` | JSON | time-series means/stds + per-replica traces + scalar `summary` |
| `ensemble_structural.npz` | NumPy npz | structural arrays (Rg, NN, coordination, composition, size fractions) |
| `ensemble_config.json` | JSON | base configuration |
| `ensemble_state.json` | JSON | full reconstruction state (can be ~130 MB) |

Load aggregated results for plotting with
`stats, structural, config = analysis.load_ensemble_data("<ensemble_dir>")` and summarize with
`analysis.print_ensemble_summary(stats, config)`. For a numeric final-state table (mean ± SD,
exportable to CSV/LaTeX) use `analysis.build_final_state_table(stats, config, structural)` for one
ensemble (pass `structural` to include the radius-of-gyration and Qt-fraction composition rows)
or `comparison.build_comparison_table(comparison)` across ensembles, then
`analysis.save_table_files(df, "<path_base>", caption=..., label=...)`.

---

## 9. Plotting

Driven from `Plot_Ensemble_Results.ipynb`; all functions live in `qtft.plotting`.

**Single run:** `plot_observables`, `plot_cluster_analysis`, `plot_structural_cluster_analysis`,
`plot_cluster_composition`.

**Ensemble:** `plot_ensemble_observables`, `plot_ensemble_structural`,
`plot_ensemble_size_categories` (all support `show_individual=True` to overlay replica traces).

**Cross-ensemble comparison:** build a comparison with
`ae.compare_ensembles({label: dir, ...})` (from `qtft.comparison`, imported as `ae`), then:
`plot_comparison_summary`, `plot_comparison_final_state`, `plot_comparison_structural`,
`plot_comparison_size_categories`. Inspect differing parameters with
`ae.print_parameter_differences(comparison)` and persist with `ae.save_comparison_data(...)`.

---

## 10. Output-file naming convention

Auto-generated trajectory / ensemble names encode the run parameters:

```
{n_qt}Qt_{n_ft}Ft_{POT}_eQQ{εQtQt}_eFF{εFtFt}_eQF{εQtFt}_kon{kon}_dt{timestep}ps_{total_time}us
```

Example — `600Qt_50Ft_LJ_eQQ1.5_eFF1.5_eQF3_kon0.001_dt50ps_100us`:
600 Qt + 50 Ft, full LJ potential, ε(QtQt)=1.5 / ε(FtFt)=1.5 / ε(QtFt)=3.0 kJ/mol,
binding rate kon=0.001, 50 ps timestep, 100 µs total.

When `topology.ft_monovalent=True`, a `_FtMono` suffix is appended (e.g.
`…_dt50ps_100us_FtMono`) so monovalent and multivalent runs at otherwise-identical parameters
don't collide on disk. The suffix is absent by default, so existing names are unchanged.

When `config.phases` is set (agglomeration↔deagglomeration cycling), a `_phased{N}_koff{koff}`
suffix is appended (`N` = number of phases) and the `{total_time}us` field reflects the **sum**
of all phase durations. A phased run's outputs live under a directory derived from the trajectory
name, with one `phase_NNN/trajectory.h5` per phase (+ a `phase_NNN/checkpoints/` used to hand off
state to the next phase). The suffix is absent for ordinary single runs, so existing names are
unchanged.

---

## 11. Gotchas

- **Time axis = step numbers, not nanoseconds.** ReaDDy observables return *step counts*.
  Convert with `time_µs = steps × timestep_ns × 1e-3` (helper `_steps_to_us`, constant
  `NS_TO_US = 1e-3`). The analysis/plotting code already does this where relevant.
- **WCA is for equilibration only**; production must use `potential_type="LJ"` to get attraction.
- **Large files.** Each `trajectory.h5` / `trajectory.xyz` can be ~1.6 GB; `ensemble_state.json`
  can reach ~130 MB. Use `particles_observable_stride=None` (default) to avoid storing per-particle
  positions unless you need them, and a coarse `record_stride`/`observable_stride`. The unread
  `forces`/`virial` observables are recorded on a coarse `heavy_observable_stride`
  (default 100× `observable_stride`) so they don't dominate file size.
- **Determinism:** results depend on `rng_seed`; ensembles assign one seed per replica.

---

## 11a. Known modelling caveats

These are deliberate simplifications / open questions in the current physical model, documented
here rather than silently fixed (see `CODE_REVIEW.md` for IDs and history):

- **(P2) RESOLVED — LJ minimum now sits at the bond length.** σ is set to
  `σ = (r_i + r_j) / 2^(1/6) ≈ 0.8909·(r_i + r_j)`, so the 12-6 LJ minimum at `2^(1/6)·σ` and the
  WCA exclusion edge both land at the contact distance `r_i + r_j`, which equals the harmonic
  bond equilibrium length `r0 = r_i + r_j`. Bonded pairs are no longer squeezed by a mismatched
  LJ minimum. (ReaDDy still does not exclude intra-topology pairs from pair potentials, but the
  bond and LJ minima now coincide.) **Note:** datasets in `Different_Particle_Ratios/` predate
  this fix and were run under the old `σ = r_i + r_j` convention, so they are not physically
  comparable to runs made after this change.
- **(P3) Cluster diffusion is a single fixed value, not size-dependent.** `ParticleConfig.cluster_diffusion`
  defaults to the monomer `diffusion`; clusters still do not slow down as `D ∝ 1/R`. The current
  standard config sets it explicitly (Qt 0.3, Ft 0.7 nm²/ns), so bound particles diffuse slower than
  free monomers, but the value is constant regardless of cluster size.
- **(P4) `kon` is a microscopic rate.** It is passed straight to ReaDDy's spatial-reaction `rate`
  (a per-pair `1/time` rate), not the macroscopic `nm³/(ns·particle)` constant the older label
  implied. Treat the swept `kon` values as microscopic rates.
- **(P5) Diffusion ratio is not Stokes–Einstein consistent.** The Qt/Ft `D` values are a
  coarse-graining choice and do not follow `D ∝ 1/r` from the radii; this is intentional, noted
  here to avoid confusion.
- **Cluster bond graphs are spanning trees.** Every reaction adds exactly one bond and never closes
  a ring, so clusters are acyclic (`n_bonds = n_particles − 1`); coordination numbers from the bond
  graph reflect that tree, not true spatial contact coordination.
- **(P6) Monovalent Ft is a leaf-only model.** With `topology.ft_monovalent=True`, an Ft can hold
  exactly one bond, so it can never bridge two Qt and two clusters can never merge. Every cluster is
  therefore a single-Qt star (one Qt + N Ft leaves), and a free Qt can only enter a cluster by
  seeding a new one with a free Ft — it cannot attach to an existing cluster. This is the intended
  physical model for monovalent ferritin, not a bug. Default `False` keeps Ft fully multivalent.
- **(P7) Bond breaking (`koff`) is a mean-field per-edge rate.** Each existing bond breaks at the
  same rate `koff` regardless of its location in the cluster (interior vs leaf) or local geometry;
  the broken edge is chosen uniformly at random, not by force or strain. It is a microscopic
  dissociation rate (1/time), the deagglomeration counterpart of the microscopic `kon` (P4), not a
  macroscopic off-rate. A freed monomer is re-typed back to its free species essentially instantly
  (a fast internal cleanup reaction), so it is indistinguishable from an originally-free particle.
  Note: ReaDDy 2.0.13's built-in `add_topology_dissociation` is bypassed (it is broken in that
  build); `qtft` registers an equivalent custom structural reaction instead.

---

## 12. Existing datasets — `Different_Particle_Ratios/`

Each subdirectory is one 10-replica ensemble (layout in §6). All current sets use **200 Qt /
400 Ft, full LJ**, and mainly vary the binding rate `kon` (and some epsilons / timestep):

| Ensemble directory | kon | dt | total |
|--------------------|-----|----|-------|
| `200Qt_400Ft_LJ_eQQ2.5_eFF1.5_eQF2.5_kon1e-05_dt30ps_150us` | 1e-5 | 30 ps | 150 µs |
| `200Qt_400Ft_LJ_eQQ2.5_eFF1.5_eQF2.5_kon0.001_dt30ps_150us` | 0.001 | 30 ps | 150 µs |
| `200Qt_400Ft_LJ_eQQ2.5_eFF1.5_eQF2.5_kon0.1_dt30ps_150us` | 0.1 | 30 ps | 150 µs |
| `200Qt_400Ft_LJ_eQQ2.5_eFF1.5_eQF2.5_kon25_dt20ps_100us` | 25 | 20 ps | 100 µs |
| `200Qt_400Ft_LJ_eQQ1_eFF1.5_eQF2_kon75_dt10ps_50us` | 75 | 10 ps | 50 µs |

Plus `Plots_Comparison_*/` directories holding cross-ensemble comparison figures and data.

> **Note:** these `Different_Particle_Ratios/` ensembles use the **old** model parameters
> (42/12 nm radii, ε≈2.5/1.5/2.5) and predate the P2 σ-at-contact fix (§11a), so they are not
> physically comparable to runs made with the current standard values.

### Single-run parameter exploration — `Simulation_Files_Single_Runs/`

The current single runs sweep the **Qt/Ft particle ratio** at the new standard parameters
(full `LJ`, `eQQ1.5 / eFF1.5 / eQF3`, `kon0.001`, `dt50ps`, 100 µs). Each is one single run
(not a 10-replica ensemble) and is stored locally only — this tree is gitignored and not in the
repo:

| Run directory | n_qt | n_ft |
|---------------|------|------|
| `200Qt_200Ft_LJ_eQQ1.5_eFF1.5_eQF3_kon0.001_dt50ps_100us` | 200 | 200 |
| `200Qt_400Ft_LJ_eQQ1.5_eFF1.5_eQF3_kon0.001_dt50ps_100us` | 200 | 400 |
| `200Qt_1000Ft_LJ_eQQ1.5_eFF1.5_eQF3_kon0.001_dt50ps_100us` | 200 | 1000 |
| `200Qt_2000Ft_LJ_eQQ1.5_eFF1.5_eQF3_kon0.001_dt50ps_100us` | 200 | 2000 |
| `400Qt_200Ft_LJ_eQQ1.5_eFF1.5_eQF3_kon0.001_dt50ps_100us` | 400 | 200 |
| `600Qt_200Ft_LJ_eQQ1.5_eFF1.5_eQF3_kon0.001_dt50ps_100us` | 600 | 200 |
| `600Qt_50Ft_LJ_eQQ1.5_eFF1.5_eQF3_kon0.001_dt50ps_100us` | 600 | 50 |
