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

**Potentials.**
- **Pairwise Lennard-Jones** for excluded volume, registered for all 10 type pairs.
  `potential_type="WCA"` → purely repulsive (cutoff `2^(1/6)·σ`); `"LJ"` → full attractive
  well (cutoff `2.5·σ`). σ is taken from the sum of the two radii; ε is set per pair through a
  cascade of defaults (see §6).
- **Harmonic bonds** (`k_bond`) hold bonded particles inside a cluster at equilibrium length
  `r_Qt + r_Ft`.

**Equilibration vs production.** Equilibration runs with **WCA only and reactions disabled**
to relax initial random positions; production then switches on attractions (LJ) and the
binding reactions. This split is handled by `equilibrate_system()` + `run_simulation()`.

**Integrator / environment.** EulerBD Brownian-dynamics integrator, Gillespie reactions,
cubic periodic box, `T = 300 K`.

---

## 2. Repository layout

| File | Purpose |
|------|---------|
| `agglomeration_simulation.py` | Config dataclasses (`SimulationConfig` etc.) + ReaDDy system/simulation builders. Entry points: `create_system`, `create_simulation`, `place_particles`, `run_simulation`, `equilibrate_system`. |
| `agglomeration_ensemble_simulation.py` | `EnsembleSimulation` class — multi-replica orchestration, local/parallel runs, SLURM script generation, result collection, statistics, save/load. |
| `agglomeration_analysis.py` | Matplotlib-free trajectory analysis: cluster stats, bond counts, binding kinetics, morphology (Rg), spatial distribution, contacts, composition, size fractions. Also `convert_h5_to_xyz` (OVITO) and `load_ensemble_data`. |
| `agglomeration_plotting.py` | All matplotlib plots: single-run, ensemble, and cross-ensemble comparison figures. |
| `analyze_ensemble.py` | CLI to (re)analyze an ensemble directory in parallel; `compare` helpers across ensembles. |
| `run_replica.py` | CLI to run **one** replica from a config JSON (used locally and by SLURM job arrays). |
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
import agglomeration_simulation as sim
import agglomeration_analysis as analysis
import agglomeration_plotting as plotting

# 1. Configure (biological values; see §6)
config = sim.SimulationConfig(
    qt=sim.ParticleConfig("Qt", radius=42.0, diffusion=0.5),
    ft=sim.ParticleConfig("Ft", radius=12.0, diffusion=1.0),
    topology=sim.TopologyConfig(binding_radius=55.0, kon=25.0, k_bond=10.0),
    lj=sim.LennardJonesConfig(
        epsilon_QtQt=2.5, epsilon_FtFt=1.5, epsilon_QtFt=2.5,
        potential_type="LJ",
    ),
    box_size=(1000.0, 1000.0, 1000.0),
    temperature=300.0,
    timestep=0.02,        # ns  (=20 ps)
    n_steps=5_000_000,    # → 100 µs total
    record_stride=100,
    observable_stride=100,
    n_qt=200,
    n_ft=400,
)

# 2. Equilibrate (WCA, no reactions) → 3. build → 4. place → 5. run (LJ + reactions)
pos_qt, pos_ft = sim.equilibrate_system(config, n_steps=5000)
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

`SimulationConfig` (in `agglomeration_simulation.py`) is the single source of truth and is
fully JSON-serializable (`config.save_json(...)` / `SimulationConfig.load_json(...)`,
`from_dict` / `to_dict` / `to_flat_dict`).

The table below leads with the **biological values used in the real runs** (the notebook /
`Different_Particle_Ratios/` datasets). The code dataclass defaults are small smoke-test
values — see the footnote.

| Parameter | Meaning | Units | Biological value |
|-----------|---------|-------|------------------|
| `qt.radius`, `qt.diffusion` | Qt encapsulin size & diffusion | nm, nm²/ns | 42.0, 0.5 |
| `ft.radius`, `ft.diffusion` | Ft ferritin size & diffusion | nm, nm²/ns | 12.0, 1.0 |
| `n_qt`, `n_ft` | particle counts | – | 200, 400 |
| `topology.binding_radius` | reaction capture distance | nm | 55.0 (≈ r_Qt+r_Ft+buffer) |
| `topology.kon` | binding rate | nm³/(ns·part) | **swept: 1e-5 … 75** |
| `topology.k_bond` | harmonic bond stiffness | kJ/(mol·nm²) | 10.0 |
| `lj.epsilon_QtQt/FtFt/QtFt` | well depths for the three free pairs | kJ/mol | ≈ 2.5 / 1.5 / 2.5 (swept) |
| `lj.potential_type` | `"WCA"` (repulsive) or `"LJ"` (attractive) | – | `LJ` for production |
| `box_size` | cubic box edge | nm | (1000, 1000, 1000) |
| `temperature` | – | K | 300 |
| `timestep` | integration step | ns | 0.01–0.03 (10–30 ps) |
| `n_steps` | total steps (→ 50–150 µs) | – | 5,000,000 |
| `record_stride`, `observable_stride` | save cadence | steps | 100 |
| `particles_observable_stride` | per-particle position cadence (`None`=off, saves disk) | steps | optional |
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

`EnsembleSimulation` (in `agglomeration_ensemble_simulation.py`) replicates a base config with
independent RNG seeds, runs the replicas, and aggregates the results.

```python
from agglomeration_ensemble_simulation import EnsembleSimulation

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
`run_replica.py --config configs/config_NNN.json`) and `submit_analysis.slurm` (runs
`analyze_ensemble.py` once all replicas finish). Submit them with `sbatch`.

`run_replica.py` can also be invoked directly:

```bash
python run_replica.py --config configs/config_000.json
python run_replica.py --config configs/config_000.json --equilibration-steps 20000
python run_replica.py --config configs/config_000.json --skip-equilibration
```

---

## 8. Analysis & outputs

Re-analyze (or analyze for the first time) an ensemble directory in parallel:

```bash
python analyze_ensemble.py --ensemble-dir <ensemble_dir> --parallel --n-workers 4 --stride 10
```

This (re)writes `ensemble_statistics.json` and `ensemble_structural.npz`.

**Metrics computed** (`agglomeration_analysis.py`):
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
`analysis.print_ensemble_summary(stats, config)`.

---

## 9. Plotting

Driven from `Plot_Ensemble_Results.ipynb`; all functions live in `agglomeration_plotting.py`.

**Single run:** `plot_observables`, `plot_cluster_analysis`, `plot_structural_cluster_analysis`,
`plot_cluster_composition`.

**Ensemble:** `plot_ensemble_observables`, `plot_ensemble_structural`,
`plot_ensemble_size_categories` (all support `show_individual=True` to overlay replica traces).

**Cross-ensemble comparison:** build a comparison with
`ae.compare_ensembles({label: dir, ...})` (from `analyze_ensemble.py`, imported as `ae`), then:
`plot_comparison_summary`, `plot_comparison_final_state`, `plot_comparison_structural`,
`plot_comparison_size_categories`. Inspect differing parameters with
`ae.print_parameter_differences(comparison)` and persist with `ae.save_comparison_data(...)`.

---

## 10. Output-file naming convention

Auto-generated trajectory / ensemble names encode the run parameters:

```
{n_qt}Qt_{n_ft}Ft_{POT}_eQQ{εQtQt}_eFF{εFtFt}_eQF{εQtFt}_kon{kon}_dt{timestep}ps_{total_time}us
```

Example — `200Qt_400Ft_LJ_eQQ2.5_eFF1.5_eQF2.5_kon25_dt20ps_100us`:
200 Qt + 400 Ft, full LJ potential, ε(QtQt)=2.5 / ε(FtFt)=1.5 / ε(QtFt)=2.5 kJ/mol,
binding rate kon=25, 20 ps timestep, 100 µs total.

---

## 11. Gotchas

- **Time axis = step numbers, not nanoseconds.** ReaDDy observables return *step counts*.
  Convert with `time_µs = steps × timestep_ns × 1e-3` (helper `_steps_to_us`, constant
  `NS_TO_US = 1e-3`). The analysis/plotting code already does this where relevant.
- **WCA is for equilibration only**; production must use `potential_type="LJ"` to get attraction.
- **Large files.** Each `trajectory.h5` / `trajectory.xyz` can be ~1.6 GB; `ensemble_state.json`
  can reach ~130 MB. Use `particles_observable_stride=None` (default) to avoid storing per-particle
  positions unless you need them, and a coarse `record_stride`/`observable_stride`.
- **Determinism:** results depend on `rng_seed`; ensembles assign one seed per replica.

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
