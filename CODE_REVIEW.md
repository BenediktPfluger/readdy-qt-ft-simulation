# Code Review — Qt–Ft Agglomeration Simulation (ReaDDy2)

Status: **read-only review complete; refactor not yet started.**
This document merges two independent reviews of the codebase (a local source read and a
Claude "ultraplan" web session) into one tracked backlog, plus the phased refactor plan.
It is the anchor document for the `refactor-cleanup` work — update the status column as
phases land.

> Ground rules (from `CLAUDE.md`): `SimulationConfig` is the single source of truth; keep the
> analysis/plotting layer split; time axis is step numbers (convert to µs); WCA = equilibration,
> LJ = production; **never silently change an on-disk format** (`ensemble_statistics.json`,
> `ensemble_structural.npz`, config JSON, the auto filename convention) — existing datasets in
> `Different_Particle_Ratios/` depend on them. Verify by execution, not by CI (there is none).

---

## 1. Current structure (as-built)

```
agglomeration_simulation.py            config dataclasses + ReaDDy system/sim builders + run/equilibrate
agglomeration_analysis.py              matplotlib-free metrics (kinetics, morphology, spatial, contacts, …)
agglomeration_ensemble_simulation.py   EnsembleSimulation: replicate → run → collect → aggregate → save → SLURM
analyze_ensemble.py                    CLI: re-aggregate an ensemble dir + cross-ensemble comparison
run_replica.py                         CLI: run one replica from a config JSON
agglomeration_plotting.py              all matplotlib (single / ensemble / comparison)
```

The layer separation (analysis is matplotlib-free, plotting owns visualization, ensemble
orchestrates) and config-as-single-source-of-truth are genuinely respected. `SimulationConfig`
is the cleanest part of the codebase.

Both reviews independently confirmed the same overall picture; the findings below are the union,
with severity reconciled. IDs prefixed `P` = physics/modelling, `B` = bug/fragility,
`R`/`C` = redundancy, `E` = structure.

---

## 2. Physics / modelling issues

### P1 (High) — Equilibration does NOT use WCA
`README` §1 and `CLAUDE.md` state equilibration is WCA-only (purely repulsive, to relax
overlaps). The code does not enforce this. `equilibrate_system()`
(`agglomeration_simulation.py:1342`) calls `create_system(config, equilibration_mode=True)`, and
`equilibration_mode` **only** skips spatial-reaction registration (`:1098`). `_add_potentials`
still uses the production `config.lj.potential_type`. With the documented single-LJ-config
workflow (README quick-start uses `potential_type="LJ"`), equilibration runs under the **full
attractive LJ well with no reactions** — particles attract and pre-cluster geometrically, the
opposite of the stated purpose. Either the docs are wrong or the code is. **Affects the initial
condition of every dataset.**
→ **Decision: config flag (Phase 2).** Add `equilibration_potential` to `SimulationConfig`
(default `"WCA"`); equilibration uses it, production keeps `lj.potential_type`. Default WCA
restores the documented intent; existing datasets were equilibrated under LJ.

### P2 (High, needs experiment) — LJ minimum / WCA exclusion sits ~12% beyond the bond length
`_add_potentials` sets `σ = contact distance` (`σ_qq = 2·r`, `σ_qf = r_Qt + r_Ft`). For a 12-6
LJ the minimum is at `2^(1/6)·σ ≈ 1.122·σ`, and the WCA repulsive cutoff is at the same `1.122·σ`.
But the harmonic bond equilibrium length is `r0 = r_Qt + r_Ft = σ_qf`, where the LJ is still
repulsive. So bonded pairs are pulled to `σ` by the bond and to `1.122·σ` by the LJ — a frustrated
equilibrium, and (because ReaDDy does not auto-exclude intra-topology pairs from external pair
potentials) a double interaction on every bonded pair. Contact/Rg metrics inherit the offset.
Options (changes the meaning of every swept ε): set `σ = (r1+r2)/2^(1/6)` (LJ min at contact),
set bond length `r0 = 2^(1/6)·σ`, or deliberately exclude bonded pairs.
→ **Decision: DEFERRED.** Documented as a known caveat this round; no code change. Revisit as a
separate physics pass, with a small two-particle ReaDDy probe to confirm σ/exclusion semantics first.

### P3 (Med) — Cluster diffusion defaults to monomer diffusion
`ParticleConfig.cluster_diffusion` defaults to the monomer `diffusion`. Large clusters do not slow
down realistically (`D ∝ 1/R`). ReaDDy's bonds recover only part of the COM slowdown.
→ Likely a config-default + documentation change; confirm intended physics.

### P4 (Med) — `kon` unit label is wrong
`TopologyConfig.kon` is documented as `nm³/(ns·particles)` but is passed directly as ReaDDy's
spatial-reaction `rate` (`agglomeration_simulation.py:1121`), which is a microscopic `1/time`
rate, not a macroscopic rate constant. → Likely a documentation/label fix; verify ReaDDy semantics.

### P5 (Low) — Diffusion ratio not Stokes–Einstein consistent
Qt/Ft `D` of 0.5/1.0 vs `r` of 42/12 implies a ratio far from the `D ∝ 1/r` expectation (~0.29).
Confirm this is intentional (coarse-graining) rather than an oversight.

### A5 (Low, perf) — `skin=0.0` everywhere
Forces a neighbor-list rebuild every step (`create_simulation` and `equilibrate_system`). Correct
but slow; a positive skin would speed up production materially.

---

## 3. Bugs / fragility

### B-forces (High) — `forces` and `virial` observables recorded but never read — **VERIFIED**
`_register_observables` records `simulation.observe.forces(stride, types=None)` and
`simulation.observe.virial(stride)` every `observable_stride` (`agglomeration_simulation.py:1214-1215`).
A repo-wide grep confirms **nothing** reads `read_observable_forces` or `read_observable_virial`;
only `pressure` is read (plotting `:818`, ensemble `:762`). `forces` is the dominant contributor to
~1.6 GB trajectory files.
→ **Decision: reduce cadence, keep both (Phase 1).** Add a config field for a coarse
heavy-observable stride (default ~100× `observable_stride`) and record `forces`/`virial` on it.
Shrinks new trajectories while keeping the data available. JSON/NPZ schema unaffected.

### B-stride (Med) — Frame mis-alignment when `record_stride ≠ observable_stride`
In `_extract_frame_data`'s `trajectory.read()` path (`agglomeration_analysis.py:390-417`) it indexes
`topology_records_all[current_frame]`, assuming the i-th trajectory frame and i-th topology record
are the same simulation step. That only holds when `record_stride == observable_stride`. The config
`_validate` merely *warns* about unequal strides but allows them — when they differ, all structural
analysis silently pairs positions with the wrong topology record.
→ Enforce equality, or align by time (as the particles-observable path already does).

### B-searchsorted (Low–Med) — `np.searchsorted` on non-monotonic series
`get_binding_kinetics.find_half_time` (`:813`) and `compute_summary_metrics` (`ensemble :1071`) use
`searchsorted`, which assumes a sorted array. `fraction_bound(t)` / `bonds(t)` usually rise but are
noisy and can dip; the returned index can be silently wrong. True today (no unbinding) but fragile
if fission is added. → Use first-crossing `np.argmax(x >= threshold)`.

### B-except (Low) — `except (KeyError, Exception)` too broad
`agglomeration_analysis.py:319` — `Exception` already subsumes `KeyError`, and the broad catch hides
real failures behind the "particles observable not available" fallback. → Narrow it.

### B-times (Low) — Structural `times` docstrings say "ns" but return step numbers
`_extract_frame_data` and the `get_*` structural functions document `times … (ns)` (e.g. `:287,
635, 873`) while returning raw step counts. Exactly the footgun `CLAUDE.md` warns about. Audit that
every consumer re-converts, and fix the docstrings.

### B-xyz (Low) — `convert_h5_to_xyz` positional type mapping is fragile
Hard-codes `type_0→Qt, type_1→Ft, type_2→QtC, type_3→FtC` (`:1658`) and defaults unknown labels to
Qt (`:1707`). A reorder of `_add_species` would silently mislabel every exported frame.

### B-treewarn (Low) — Bond-count "n−1 fallback" warning is misleading
Every reaction adds exactly one edge and never closes a ring, so clusters are **always acyclic
trees** and `n_bonds = n_particles − 1` is *exact*, not an estimate. The three-method counter +
warning machinery in `get_bond_counts` is dead complexity. (Modelling note: this also means
`get_contact_analysis` coordination numbers measure the spanning tree, not true spatial contact
coordination — worth stating in the README.)

### B-prefix (Low) — Single-run filename prefix `qt_ft_` diverges from the convention
`_generate_output_filename` emits `qt_ft_{n}Qt_{n}Ft_…` (`:482`) while the documented convention and
the ensemble folder names use `{n}Qt_{n}Ft_…` (no prefix). The prefix does leak into stored
`ensemble_config.json`/`ensemble_state.json` as the base-config `output_file` string, but replicas
always load `trajectory.h5`, so unifying it will **not** break dataset loading. → Unify (see R3).

---

## 4. Redundancy / duplication

### R1 (Med) — `analyze_ensemble.main()` duplicates the EnsembleSimulation pipeline
~`analyze_ensemble.py:147-500` re-implements `collect_results` + `compute_statistics` +
`compute_structural_statistics` + `save_for_plotting`, including the per-replica frame loop and the
structural mean/std processing. Only real difference: the CLI adds parallel structural analysis.
**Schema-drift risk on the on-disk JSON/NPZ.** → Add `parallel`/`n_workers` to
`compute_structural_statistics`; CLI becomes `load → collect → compute_* → save_for_plotting()`.
Largest single cleanup.

> **Demonstrated (Phase-0 smoke harness):** the two paths already diverge. `save_for_plotting()`
> writes NPZ keys `std_nn_dist_{mean,std,all}` and `std_intra_nn_dist_{mean,std,all}`;
> `analyze_ensemble.py` omits all six (its spatial `keys` list drops the `std_*` series). So
> `ensemble_structural.npz` content depends on *which path produced it* — confirming the drift.
> Phase-3 gate: after the dedup, `tools/smoke_test.py` Stage 5 must report zero differences.

### R2 (Med) — Random placement duplicated
`place_particles` and `equilibrate_system` both draw uniform positions independently. → Shared helper.

### R3 (Med) — Param-string formatting duplicated (also fixes B-prefix)
`_generate_output_filename` (config) and `_generate_folder_name` (ensemble) re-derive the same
`eQQ/eFF/eQF/kon/dt/time` string with copy-pasted `fmt_eps`/`kon_str`/`dt_str`/`time_str`. They must
stay byte-identical to preserve the naming convention. → One `format_param_string()` in config.

### R4 (Med) — Topology observable re-read 3–4× per replica
`collect_results` calls `get_bond_counts`, `get_cluster_statistics`,
`get_binding_kinetics`(→`get_bond_counts` again), and later `get_size_fractions`, each re-reading the
topology observable. → Thread a single read through.

### R5 (Med) — Run pipeline triplicated
`_run_single_replica` (ensemble), `_run_replica_worker` (ensemble, multiprocessing), and
`run_replica.py` (CLI) each re-implement equilibrate → build → place → run. → One
`run_one(config, equilibration_steps, skip_equilibration)`.

### C3 (Low) — `convert_numpy` defined 3×
`ensemble :1494`, `analyze_ensemble :190`, and inline elsewhere. → One shared util.

### C5 (Low) — JSON/NPZ save logic duplicated
`EnsembleSimulation.save_for_plotting` and `analyze_ensemble.main` share the same `time_series_keys`
and write structure. → One I/O writer (subsumed by R1).

---

## 5. Suggested file structure & naming

Drop the redundant `agglomeration_` prefix; split config out of the engine; make CLIs thin
wrappers. **Keep on-disk formats/filenames unchanged.**

```
qtft/
├── config.py     # *Config dataclasses + shared format_param_string()   (R3 / B-prefix)
├── system.py     # create_system, _add_species / _add_potentials / _add_topologies
├── engine.py     # create_simulation, place_particles, run_simulation, equilibrate_system, run_one()  (R5)
├── analysis.py   # metrics (matplotlib-free)   [optionally split: kinetics / structure / export]
├── ensemble.py   # EnsembleSimulation
├── plotting.py   # [optionally split: single / ensemble / comparison]
└── io.py         # single source of the JSON/NPZ schema   (R1 / C5)
scripts/
├── run_replica.py        # → qtft.engine.run_one
└── analyze_ensemble.py   # → qtft.ensemble + qtft.io  (comparison helpers move to analysis)
```

⚠ Renaming breaks the notebooks' imports and the SLURM `scp`/run lines, and the comparison helpers
in `analyze_ensemble.py` are analysis (not script) code. Move in lockstep, or add re-export shims for
a gentle migration. **Do this last**, as a pure rename with no logic change.

---

## 6. Phased refactor plan

One concept per phase, one commit, verified before the next. Sequenced by blast radius: decisions →
isolated physics → cheap safe bugfixes → logic dedup (formats preserved) → file moves last.

| Phase | Scope | Format risk | Gate |
|------|-------|-------------|------|
| **0** | Safety net: this doc + smoke harness + capture baselines + collect decisions | none | **DONE** — `tools/smoke_test.py` green on current code; R1 baseline captured (drift demonstrated) |
| **1** | Cheap format-neutral fixes: B-forces (coarse-stride forces/virial), B-stride, B-searchsorted, B-except, B-times, B-xyz, B-treewarn | none (trajectory contents shrink) | **DONE** — harness green (exit 0); legacy configs load (back-compat verified); A5 skipped (perf) |
| **2** | Physics: P1 (equilibration_potential flag, default WCA); P2/P3/P4/P5 document-only | none | with default config, equilibration registers only repulsive potentials; harness green; README caveats added |
| **3** | Logic dedup, files in place: R3+B-prefix → R2 → R5 → R4 → R1/C3/C5 | **yes (R1)** | **byte-compare** CLI vs `save_for_plotting()` output against Phase-0 baseline |
| **4** | Restructure into `qtft/` package + shims; update notebooks & SLURM in lockstep | rename only | notebooks run; CLIs work; SLURM paths correct |

**Recommended first slice:** Phase 1 `B-forces` + Phase 2 `P1` — smallest, highest impact, isolated.

### Coverage checklist (every finding is assigned — nothing dropped)

| ID | Finding | Phase | Status |
|----|---------|-------|--------|
| P1 | Equilibration uses LJ not WCA → `equilibration_potential` flag (default WCA) | 2 | decided |
| P2 | LJ-min vs bond-length mismatch | deferred → doc | decided: defer |
| P3 | Cluster D = monomer D | doc-only | decided: doc |
| P4 | `kon` unit label | doc-only | decided: doc |
| P5 | Stokes–Einstein D ratio | doc-only | decided: doc |
| A5 | `skin=0.0` performance | 1 (opt) | skipped — perf-only, deferred |
| B-forces | forces/virial → coarse stride (keep, don't drop) | 1 | **done** |
| B-stride | frame mis-alignment when strides differ | 1 | **done** (time-matched) |
| B-searchsorted | searchsorted on non-monotonic series | 1 | **done** (first-crossing) |
| B-except | broad `except` | 1 | **done** |
| B-times | "ns" docstrings are step numbers | 1 | **done** |
| B-xyz | positional xyz type mapping | 1 | **done** (from `particle_types`) |
| B-treewarn | trees → `n−1` exact, drop warning | 1 | **done** |
| B-prefix | single-run filename prefix divergence | 3 (with R3) | open |
| R3 | param-string formatting dedup | 3 | open |
| R2 | placement dedup | 3 | open |
| R5 | run-pipeline triplication | 3 | open |
| R4 | topology re-read per replica | 3 | open |
| R1 | save/aggregate dedup in CLI | 3 | open |
| C3 | `convert_numpy` ×3 | 3 | open |
| C5 | JSON/NPZ save logic duplicated | 3 (with R1) | open |
| E | package restructure | 4 | open |

---

## 7. Verification protocol (no CI; verify by execution)

1. **Smoke run** — tiny config (`n_qt=n_ft≈20`, small box, short `n_steps`): completes and writes
   `trajectory.h5`; single-run + 2-replica ensemble + analysis + plotting all run.
2. **P1** — assert equilibration registers only repulsive (WCA-cutoff) potentials regardless of
   production `potential_type`.
3. **B-forces** — `trajectory.h5` shrinks; all analysis/plotting still runs.
4. **R1** — byte-compare `ensemble_statistics.json` / `ensemble_structural.npz` from the CLI vs
   `save_for_plotting()` on an existing `Different_Particle_Ratios/` ensemble.
5. Cross-check params/units against README §5 and `SimulationConfig` defaults after every phase.

---

## 8. Decisions (Phase 0 — RESOLVED)

1. **P1 → config flag.** Add an explicit equilibration-potential setting (default **WCA**) to
   `SimulationConfig` rather than hard-coding the switch. Equilibration uses that setting;
   production uses `lj.potential_type`. Keep JSON back-compat (default via `.get`).
2. **P2 → deferred.** Do **not** change the σ/bond convention this round. Document it as a known
   modelling caveat in README + CODE_REVIEW; revisit as a separate physics pass.
3. **B-forces → reduce cadence, do NOT drop.** Keep `forces` and `virial` registered, but record
   them on a **coarser stride** (~100× the normal `observable_stride`, i.e. "every 100th stride")
   via a new config field. Shrinks trajectories while keeping the data available. (Assumption:
   "every 100th stride" = 100 × `observable_stride`; confirm if you meant 100 absolute steps.)
4. **P3 / P4 / P5 → document-only** this round. Note as caveats; no code change.
