# Handover — ReaDDy Qt–Ft: timestep scaling, agglomeration↔deagglomeration cycling, and cluster-shape / DLA analysis

**For:** Benedikt Pfluger (model author) · **From:** Gil / Westmeyer Lab (via Claude Code) · **Date:** 2026-07-19
**Scope:** everything done on top of your thesis model since the 11-condition campaign — three workstreams,
all the new tooling, exactly how to run more on LRZ, and how to fold in the FIB-SEM data. Written to be
self-contained; you know the biology/model, so this focuses on the *computational* state and the *how-to*.

---

## 0. TL;DR — three things were done

1. **Maximized the integration timestep** (the original request). Verdict: **dt cannot be raised for this
   system — dt_max = 0.05 ns (1×)**. The agglomerated production state is stiff across every force channel
   at the ~0.05 ns scale; neither softening `k_bond` nor the LJ core (`ε`) unlocks a larger dt. The
   reference's 5 ms works only because its particles are ~100× coarser/slower; ours are fixed by the
   encapsulin/ferritin sizes. **Reach longer timescales via wall-clock (longer jobs, more cores, the
   `readdy-cell` fork), not a bigger dt.**
2. **Ran a long agglomeration↔deagglomeration production** — your Figure-6.4 protocol (WCA, kon = 10 ns⁻¹
   agg / koff = 10⁻⁴ ns⁻¹ deagg, dt = 0.05 ns) extended to **5 cycles (500 µs)**. Clean, reproducible
   cycling: every agg phase → one 600-particle percolated cluster (~599 bonds); every deagg phase → free
   monomers; reproduces your finding that cluster *size* collapses faster than the bond *count*.
3. **Quantified cluster morphology vs a sphere and vs DLA** (on Qt centroids, so it maps 1:1 onto FIB-SEM).
   Verdict: clusters are **robustly non-spherical for N ≥ 6**, and the LJ reference is **NOT consistent with
   pure DLA — it departs toward attraction-driven COMPACTION** (less elongated than a matched-N DLA null,
   compacting over time; attraction-off WCA recovers DLA-like morphology; shape is only weakly kon-dependent
   → not RLCA either). This refines the earlier "DLA-like" null study.

All three have a Google Doc (in the shared Drive folder) and a Markdown report + figures on LRZ under
`~/readdy-qt-ft-simulation/dt_ladder/`.

---

## 1. LRZ environment (unchanged from your setup — reuse it)

- **Login:** `ssh lrz` (host alias `cool.hpc.lrz.de`, user `ga49qal2`). Everything below lives on that
  account; if you run under your own LRZ account, clone the repo + copy the new `lrz/*.py` scripts (§3) and
  re-run `lrz/bootstrap_login.sh` once on a login node.
- **Repo:** `/dss/dsshome1/0C/ga49qal2/readdy-qt-ft-simulation/` (= your
  `github.com/BenediktPfluger/readdy-qt-ft-simulation` + the new scripts in §3, which are **not yet pushed
  to GitHub** — ask Gil to commit/push them, or copy from this account).
- **Env:** `~/miniforge3/envs/readdy` (readdy 2.0.14 linux-64). Activate:
  `source ~/miniforge3/bin/activate ~/miniforge3/envs/readdy`. Rebuild with `bash lrz/bootstrap_login.sh`.
- **SLURM (the important gotcha, unchanged):** submit to cluster **`serial`**, partition **`serial_std`**
  (1-day) or **`serial_long`** (7-day), QOS **`cm4_serial`** / **`cm4_serial_long`**, **≤ 32 cores/job**,
  per-user cap **192 CPU**. Do **NOT** use `cm4_std`/`cm4_tiny` — their QOS reject small jobs
  (`QOSMinCpuNotSatisfied`). Monitor with `squeue -M serial -u $USER`.
- **Do not run heavy Python on the LOGIN node** — it is throttled. Geometric-null generation and analysis
  either run locally (a laptop) or as a `serial_std` job; trajectory reading + shape extraction is light
  enough to run on the login node.

---

## 2. Workstream 1 — timestep maximization (dt-ladder)

**Result: dt_max = 0.05 ns (1×).** dt = 0.1 ns is *marginal* (1/3 seeds survive the agglomerated 25 µs
state, worse with length) and unsafe for long runs; dt ≥ 0.15 ns is unstable. Softening `k_bond` 10→0.3 and
softening ε_QQ/ε_FF 1.5→0.5 both fail to stabilize any larger dt. Large dt does not cleanly NaN — it
explodes positions and *hangs* ReaDDy's neighbour list (steps/s collapses 237→1 as dt 0.05→2). See
`dt_ladder/DT_SCALING_REPORT.md` and the Google Doc "dt-scaling + agg↔deagg cycling".

**How the ladder works** (`lrz/dt_ladder.py` + `dt_ladder_run.py`): each probe **equilibrates at a fixed
safe dt = 0.05** (WCA, no reactions), then runs **production at the test dt** from that identical state, so
the only variable is the production step size. Detectors: NaN/inf, energy runaway vs the dt=0.05 baseline,
**all-frame** core-overlap (silent noise-tunneling), and a hard `timeout` (exploded runs hang, so they must
be killed and marked unstable).

**Run a dt-ladder / any parameter sweep at dt = 0.05:**
```bash
source ~/miniforge3/bin/activate ~/miniforge3/envs/readdy && cd ~/readdy-qt-ft-simulation
# Phase-1 stability ladder (elevated kon so stiff bonds form fast; single-replica probes):
python lrz/dt_ladder.py --phase1 --submit
# Phase-2/3 invariance + recalibration at REAL kon, tagged folders (needed because k_bond isn't in the
# canonical folder name); e.g. sweep k_bond:
python lrz/dt_ladder.py --phase2 --kbond 1  --tag kb1  --dts 0.05,0.5,1  --sim-us 25 --reps 3 --submit
# analyze / compare:
python lrz/dt_ladder_analyze.py --phase-dir dt_ladder/phase1
python lrz/dt_ladder_compare.py --phase-dir dt_ladder/phase23 --master-kbond 10 --master-dt 0.05
```
`dt_ladder.py` flags: `--kon`, `--kbond`, `--eqq/--eff/--eqf` (LJ well depths), `--tag` (folder prefix),
`--dts`, `--sim-us`, `--reps`, `--timeout`, `--walltime`, `--outdir`, `--submit`, `--array`.

---

## 3. Workstream 2 — agglomeration↔deagglomeration cycled production

Your Figure-6.4 single cycle, extended to **5 cycles (10 phases, 500 µs, 10 M steps)**, completed on
`serial_long` (job 5320974, ~3.3 h). WCA is cheap so it ran ~1000 steps/s. Output:
`dt_ladder/production/aggdeagg_5cyc_500us/` (per-phase + combined trajectories, `progress.json`, and
`aggdeagg_analysis_fig.png`). Result: clean reversible cycling (bonds 0↔~599; largest cluster 1↔600), and
the size-collapses-faster-than-bond-count asymmetry reproduced.

**Run more agg↔deagg cycles (resumable, phase-level):**
```bash
# build a config with make_agg_deagg_phases (see setup_phased.py pattern, or edit n_cycles):
#   phases = make_agg_deagg_phases(agg_steps=1_000_000, deagg_steps=1_000_000, n_cycles=N,
#                                  agg_potential="WCA", deagg_potential="WCA")
#   TopologyConfig(kon=10.0, koff=1e-4, k_bond=10.0), timestep=0.05
python lrz/dt_ladder_phased.py --config <phased.json>            # fresh
python lrz/dt_ladder_phased.py --config <phased.json> --resume   # continue after preemption/node failure
python lrz/dt_ladder_phased_analyze.py --config <phased.json>    # bonds/cluster-size vs time, phase bands
```
`dt_ladder_phased.py` chains phases via ReaDDy checkpoints (positions **and** bonds carry across phases) and
is `--resume`-capable (verified by kill-and-resume). Tune the **kon/koff ratio** to model different sensor
reversibility (koff·T_deagg ≈ 5 gives ~99 % bond breakage over the phase).

---

## 4. Workstream 3 — cluster shape vs sphere & DLA (+ FIB-SEM framework)

**Result (verified by a 4-lens adversarial review):** clusters are **non-spherical for N ≥ 6** (matched-N
Cliff's δ vs a compact-sphere null +0.5→+1.0); the LJ reference is **more compact than a DLA null**
(δ −0.37→−0.53 at reliable N=4–22) and **compacts over time**; **attraction-off WCA recovers DLA-like
morphology** (δ≈0, df 1.86≈DLA 2.0) — so the mechanism is **attraction-driven compaction**, generic
attractive-colloid (not CaM-RS20-specific), and *not* reaction-limited (a 100× kon sweep barely moves the
shape). See `dt_ladder/SHAPE_DLA_REPORT.md`, `shape_dla_verified.png`, and the Google Doc.

**Pipeline (all on Qt CENTROIDS, so sim ↔ FIB-SEM are recipe-identical):**
- `lrz/sim_shape_plus.py --ensemble-dir <dir> --out <p> [--tfrac-min a --tfrac-max b]` — per-cluster κ²,
  aspect, Rg, coordination, loops, extent from a trajectory ensemble; time-windowing via `--tfrac-*`.
  Records the **percolation-exclusion fraction** (audit; it rises with agglomeration).
- `lrz/dla_null_plus.py --sim-npz <p>_data.npz --out <n> --repl 250` — matched-N nulls: **compact/sphere,
  random, DLA (Witten–Sander), RLCA (sticking 0.05)**; multiprocessing; records extent for the symmetric cut.
  **Run locally, not on the login node.**
- `lrz/analyze_shape_dla.py --sim-npz <p>_data.npz --null-npz <n>_data.npz --out <o> --extent-cap 250` —
  matched-N Mann-Whitney + Cliff's δ + Wilcoxon; **symmetric extent cut** applied to nulls; df context-only.

**Two methodological musts (learned the hard way):** (a) the `extent≥0.5L` percolation filter must be
applied to the nulls too (it excludes 17 % early → 33 % late sim clusters and can *manufacture* compaction);
(b) lead with **effect sizes**, not p-values (clusters are pooled over frames → pseudoreplication), and use
the **WCA-vs-LJ contrast** as the decisive, selection-immune evidence.

**FIB-SEM comparison (ready for your HEK / neuron Qt centroids):**
```bash
python lrz/vem_descriptors.py --qt hek_qt_centroids.npy --roi <xmin,xmax,...,zmax nm> --out em_hek
python lrz/analyze_shape_dla.py --sim-npz em_hek_data.npz --null-npz <nulls>_data.npz --out em_vs_nulls
```
`vem_descriptors.py` is the open-boundary twin (drops field-of-view-truncated clusters, no PBC). Report
**stimulated AND unstimulated — the DIFFERENCE is the signal.** This places the real clusters on the
sphere↔DLA↔compact axis and tests the sim's prediction that real clusters should be **compact / attraction-
dominated**, and identifies which simulated regime (density, kon, maturation time) matches. Compare
shape/topology, **never absolute counts/sizes/timescale** (the sim's diffusion is scaled).

---

## 5. New scripts added to `lrz/` (commit these to GitHub)

| script | purpose |
|---|---|
| `dt_ladder.py` | generate/submit dt (or kon/kbond/ε) sweeps; equilibrate-at-safe-dt; timeout-guarded |
| `dt_ladder_run.py` | per-replica worker (equilibrate at safe dt → produce at test dt; stability metrics) |
| `dt_ladder_analyze.py` | dt-ladder stability + invariance report + analytic diagnostics |
| `dt_ladder_compare.py` | cross-condition (k_bond/ε) comparison vs a master reference |
| `dt_ladder_production.py` | long single-condition production, segmented + checkpoint-resumable |
| `dt_ladder_phased.py` | long agg↔deagg cycled production, phase-checkpoint-chained + `--resume` |
| `dt_ladder_phased_analyze.py` | stitch bonds/cluster-size vs time across phases, shade phases |
| `sim_shape_plus.py` | per-cluster Qt-centroid shape + loops + extent (the shape extractor) |
| `dla_null_plus.py` | matched-N compact/random/DLA/RLCA nulls with loops + extent (parallel) |
| `analyze_shape_dla.py` | matched-N shape comparison sim vs nulls (symmetric extent cut) |

They reuse your existing `qtft/` package and `lrz/struct_descriptors.py` / `vem_descriptors.py` verbatim.

## 6. Key data locations (on `ga49qal2` LRZ)

- Campaign trajectories (reused): `~/readdy-qt-ft-simulation/200Qt_400Ft_LJ…100us/` (4 replicas, 2001 frames),
  `…WCA…50us/`, plus the ratio/density conditions.
- dt-ladder evidence: `~/readdy-qt-ft-simulation/dt_ladder/{phase1,phase23,phase2b}/dt_ladder_{report,compare}.{md,json}`.
- Production: `~/readdy-qt-ft-simulation/dt_ladder/production/aggdeagg_5cyc_500us/` (done).
- Shape/kon sweep: `~/readdy-qt-ft-simulation/dt_ladder/shape_kon/` (kon ∈ {2e-4,1e-3,5e-3,2e-2}, 50 µs).
- Reports + figures: `~/readdy-qt-ft-simulation/dt_ladder/*.md`, `*.png`.

## 7. Caveats and one thing to reconcile

- **kon discrepancy (please reconcile for the record):** thesis **Table 3.6 lists kon = 10⁻⁵ ns⁻¹** for the
  main LJ agglomeration study, but the **deposited campaign code uses 10⁻³** (`lrz/lrz_campaign.py`
  `build_base`). It doesn't affect the agg↔deagg run (which uses the unambiguous Figure-6.4 values) or the
  dt-scaling conclusions, but the main study's kon value should be pinned down.
- Shape claims are scoped to **N ≥ 6** (N=4–5 ≈ compact); large-N bins (≥36) are unreliable after the
  symmetric extent cut (DLA null depleted); the compaction mechanism is a **generic attraction**, not a
  chemistry-specific fingerprint; **fractal dimension is context-only** (the geometric DLA null is
  particle-cluster, whereas the system is cluster–cluster DLCA).

## 8. Suggested next simulations (highest value first)

1. **Fold in the FIB-SEM Qt centroids** (§4) — the single highest-value step: it constrains the model and
   tests the "real clusters are compact/attraction-dominated" prediction. Just needs the centroid `.npy/.csv`
   + imaged-volume ROI, per condition (stimulated + unstimulated).
2. **Match sim conditions to the EM densities** — once the EM Qt density/ratio is known, run the matching
   `n_qt/n_ft` at real kon and re-compare shape (the campaign already spans several ratios/densities).
3. **Discrete-patch valency ferritin** (your thesis §5 idea) — turns Finding B into a quantitative occluder
   titration; heavy, run supervised.
4. **`readdy-cell` fork** for ~3× wall-clock (no physics change) if longer trajectories are wanted — the only
   real lever for more simulated time, since dt is maxed.

## 9. Document index (all in the shared Drive folder "ReaDDy Qt-Ft analysis (reports + figures)")

- Google Docs: *dt-scaling + agg↔deagg cycling*; *Qt-cluster shape & DLA*; (older *timestep v1*).
- Figures: `dt_scaling_summary.png`, `aggdeagg_analysis_fig.png`, `shape_dla_verified.png`.
- Markdown sources on LRZ: `dt_ladder/{DT_SCALING_REPORT,SHAPE_DLA_REPORT,HANDOVER_Benedikt_2026-07-19}.md`.
- Prior null study: `~/claude_out/handoff-20260716-093014-79101-4010/readdy_null_2026-07-16/VERDICT.md`.

*Questions on the model/biology are yours; LRZ/SLURM mechanics and the new tooling are in this file + the
repo READMEs. Happy to walk through any of it or run the EM comparison the moment the centroids exist.*
