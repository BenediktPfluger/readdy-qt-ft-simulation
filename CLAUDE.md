# CLAUDE.md — Qt–Ft Particle Simulation with ReaDDy2

## Project

Coarse-grained ReaDDy2 Brownian-dynamics simulation of Qt encapsulin and ferritin (Ft)
agglomeration. Read `README.md` for the architecture, physical model, parameters, and
workflow before making changes.

## Architecture Rules

- **`SimulationConfig` is the single source of truth.** All parameters live in the config
  dataclasses in `agglomeration_simulation.py` and are JSON-serializable. Don't hard-code
  physical parameters elsewhere — add/extend the config and thread it through.
- **Keep the layer separation intact:**
  - `agglomeration_analysis.py` must stay **matplotlib-free** (pure data → numbers/arrays).
  - `agglomeration_plotting.py` owns all matplotlib/visualization.
  - `agglomeration_ensemble_simulation.py` orchestrates multi-replica runs; single-run
    building blocks stay in `agglomeration_simulation.py`.
- **Time axis = step numbers, not nanoseconds.** ReaDDy observables return step counts;
  convert with `step × timestep_ns × 1e-3 = µs` (`_steps_to_us` / `NS_TO_US`). Never plot or
  report raw step indices as time.
- **WCA = equilibration only; LJ = production.** Preserve the equilibrate-then-produce split.
- **Don't break on-disk formats** (`ensemble_statistics.json`, `ensemble_structural.npz`,
  config JSON, the auto-generated filename convention) — existing datasets in
  `Different_Particle_Ratios/` depend on them. If a format must change, say so explicitly.
- Notebooks (`Run_Simulation.ipynb`, `Plot_Ensemble_Results.ipynb`) are the user entry points;
  keep the public function signatures they call stable, or update the notebooks too.

## Workflow

- **Plan first**: present a short step-by-step plan and wait for approval before editing code.
- Make small changes — no big rewrites across many files at once.
- Keep the README in sync when you change parameters, public functions, or file formats.

## Definition of Done

There is **no automated test suite** and this is **not a git repo**, so "done" means verified
by running, not by a green CI:

1. **Review your own code** after you think you're done: re-read the diff and check it against
   what was asked.
2. **Verify by execution.** Run the smallest real thing that exercises the change — a short
   simulation (small `n_steps`/box), an analysis call on an existing `trajectory.h5`, or a
   plotting call on an existing ensemble in `Different_Particle_Ratios/`. Never claim something
   works without running it.
3. **Cross-check parameters** against `README.md` and the `SimulationConfig` defaults — names,
   units, and the epsilon-cascade behavior must stay consistent.
4. Explain in 2–3 sentences what you changed and why.

## Never

- Never commit secrets or API keys.
- Never silently change a saved-data format or the output-filename convention.
- Never report ReaDDy step numbers as physical time (always convert to µs).
- Never claim a change works without having actually run it.
