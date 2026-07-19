# Maximizing the Brownian-dynamics timestep — results

**Task:** find the largest integration timestep `dt` that is (a) numerically stable and
(b) preserves the equilibrium statistics of the ReaDDy 2 Qt–Ft agglomeration simulation,
then deploy a long production run. Reference condition: 200 Qt / 400 Ft, LJ, box 500³ nm,
T = 300 K, kon = 0.001 /ns, k_bond = 10, ε_QQ=ε_FF=1.5, ε_QF=3.0, current dt = 0.05 ns.
Cluster: LRZ CoolMUC‑4 `serial` (serial_std / serial_long, QOS cm4_serial[_long]).

---

## TL;DR

**dt cannot be raised for this system. dt_max = 0.05 ns (the current value) — 1× speedup.**

- **dt = 0.1 ns (2×) is *marginal*** — it survives short/dilute runs but only **1 of 3 seeds**
  in the agglomerated production state (25 µs), and degrades further with run length. It is
  **not safe for a long production run.**
- **dt ≥ 0.15 ns is unstable** in the agglomerated state.
- **Both force‑constant recalibration levers the handover names FAILED:**
  softening **k_bond** 10 → 0.3 stabilised *no* larger dt (and shifted the morphology);
  softening the **LJ core** ε_QQ/ε_FF 1.5 → 0.5 did not stabilise dt ≥ 0.2 either.
- The hoped **1–3 orders of magnitude is not achievable via dt** for these physical
  parameters. This is exactly the ceiling the handover flagged to *verify, not assume*
  ("our particles are smaller/faster than theirs, so our dt_max is smaller").

More simulated time here comes from **wall‑clock levers, not a bigger dt**: the
`readdy-cell` fork (~3×), more cores/threads, and long `serial_long` jobs with
checkpoint‑chaining. A **1 ms production run (10× the existing 100 µs campaign)** at
dt = 0.05 ns has been deployed on `serial_long` on that basis.

---

## 1. Method

A per‑dt "ladder" was run on the cluster (`lrz/dt_ladder*.py`). Each probe:

1. **Equilibrates at a fixed safe dt = 0.05 ns** (WCA, no reactions) to a relaxed state,
   then runs **production at the test dt** from that identical state — so the *only*
   variable is the production integrator's step size (not initial overlaps).
2. Detects instability by **NaN/inf**, **energy runaway vs the dt=0.05 baseline**,
   **all‑frame core overlap** (silent noise‑tunneling), and a **hard `timeout`** — because
   a blown‑up run does *not* cleanly NaN; it explodes positions and **hangs ReaDDy's
   neighbour list**, so it must be killed and recorded as unstable rather than left to
   squat on a node.
3. For stable dt, compares the four handover observables — relative shape anisotropy **κ²**,
   mean **Qt coordination**, **largest‑cluster fraction**, **g(r) bridging peak** — plus
   binding kinetics (n_bonds, fraction‑bound) against the dt=0.05 reference.

An adversarial 4‑lens design review (BD‑stability, ReaDDy‑API, statistics, reaction‑fidelity)
caught two must‑fixes that shaped the above: the silent‑tunneling detector and the
elevated‑kon Phase‑1 (so stiff bonds actually exist to stress the integrator).

## 2. Phase 1 — stability ladder (short, 5 µs, elevated kon = 0.1)

| dt (ns) | verdict | κ² | Qt coord | largest frac | n_bonds |
|---:|:--|--:|--:|--:|--:|
| 0.05 | **STABLE** | 0.408 | 3.29 | 0.211 | 561 |
| 0.1  | **STABLE** | 0.394 | 3.42 | 0.286 | 568 |
| ≥0.2 | UNSTABLE (explode → hang) | — | — | — | — |

Short‑run `dt_max = 0.1 ns`. The empirical cliff (0.1→0.2) is *tighter* than the single‑bond
divergence limit `dt_crit = 2/((D_QtC+D_FtC)/kT · k_bond) ≈ 0.5 ns`, because the agglomerate
is **multivalent** (each Qt bonds ~2.8 Ft → effective stiffness several‑fold higher).

## 3. Phase 2/3 — invariance + recalibration (real kon = 0.001, 25 µs)

At the true kon and 5× longer runtime, the system reaches the dense agglomerated production
state — and the cliff **drops**:

| k_bond | dt (ns) | speedup | verdict | note |
|---:|---:|---:|:--|:--|
| 10 | 0.05 | 1× | **STABLE** | master / reference |
| 10 | 0.1 | 2× | **UNSTABLE** | only 1/3 seeds complete (marginal) |
| 10 | 0.15, 0.2 | 3–4× | UNSTABLE | 0/3 |
| 1 | 0.05 | 1× | STABLE* | *largest‑frac drifts 0.41→0.23 (soft bond ≠ same physics) |
| 1 | 0.5, 1 | 10–20× | UNSTABLE | k_bond softening does **not** rescue |
| 0.3 | 0.05 | 1× | STABLE* | *largest‑frac drifts 0.41→0.28 |
| 0.3 | 1, 2 | 20–40× | UNSTABLE | k_bond softening does **not** rescue |

**Long‑timeout verification (90‑min cap) + LJ‑core softening:**

| condition | dt 0.05 | dt 0.1 | dt 0.15 | dt 0.2 | dt 0.5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| baseline core (ε=1.5) | 3/3 ✅ | **1/3 ⚠** | 0/3 ❌ | 0/3 ❌ | — |
| softened core (ε_QQ=ε_FF=0.5) | — | — | — | 0/3 ❌ | 0/3 ❌ |

Even with a 90‑minute wall‑clock allowance, dt=0.1 fails 2/3 seeds and dt≥0.15 fails
completely; softening the LJ cores 3× rescues nothing.

**Throughput collapse (the instability signature):** effective steps/s falls
**237 → 62 → 7 → 1.4 → 1.0** as dt rises **0.05 → 0.1 → 0.5 → 1 → 2** — a stable run's
per‑step cost is dt‑independent, so this collapse *is* the instability (positions spread →
degenerate neighbour lists), not mere slowness. See `dt_scaling_summary.png`.

## 4. Why the order‑of‑magnitude goal is unreachable here

In the **agglomerated production state** the system is stiff across *every* force channel at
the ~0.05 ns scale simultaneously: the **multivalent harmonic‑bond network** *and* the
**LJ repulsive cores** of densely‑packed particles. Consequently:

- **k_bond recalibration fails** — the bond is not the sole limiter, and softening it also
  changes the morphology (largest‑cluster fraction 0.41→0.23), so it is not a free knob.
- **ε (LJ‑core) softening fails** — softening the Qt‑Qt/Ft‑Ft cores did not stabilise larger dt.
- **The particles cannot be coarsened.** The reference study reached dt = 5 ms because its
  particles are ~100× larger/slower relative to their box, keeping per‑step displacement
  tiny. Ours (Ft 6 nm, Qt 21 nm; D 1.0 / 0.5 nm²/ns) are fixed by the encapsulin–ferritin
  model; the reference's dt is not transferable, precisely as the handover cautioned.

## 5. Speedup and achievable simulated time

**dt speedup = 1× (dt_max = 0.05 ns).** More simulated time is a wall‑clock question:

- Measured native throughput at dt=0.05: **~250 steps/s** (4 threads, LRZ cm4) ≈
  **45 µs simulated per hour**. Heavy agglomeration reduces this somewhat over a long run.
- **Max simulated time per job** (no dt change): **~1 ms per 1‑day job**,
  **~4–7 ms per 7‑day `serial_long` job** (agglomeration‑dependent), and **unbounded via
  checkpoint‑chaining** across jobs. For reference, the existing campaign reached 100 µs.
- **Levers that *do* help** (no physics change): `readdy-cell` fork (~3×), more cores/threads,
  and checkpoint‑chained multi‑job runs.

## 6. Phase 4 — deployed production run: reversible agglomeration ↔ deagglomeration (500 µs)

At the validated safe timestep, a long run was deployed that captures **both agglomeration and
deagglomeration**, following Benedikt Pfluger's thesis Figure‑6.4 protocol and extending its single
cycle to **five cycles (500 µs, 10 M steps, 10× the single cycle)**:

- **Protocol (Fig 6.4):** WCA potential throughout (cluster cohesion is from the harmonic bonds;
  the purely repulsive WCA lets freed particles disperse), alternating **agglomeration**
  (kon = 10 ns⁻¹, koff = 0, 50 µs) and **deagglomeration** (kon = 0, koff = 10⁻⁴ ns⁻¹, 50 µs)
  phases, dt = 0.05 ns, nQt = 200 / nFt = 400, k_bond = 10, rbind = 27.25.
- `lrz/dt_ladder_phased.py` runs the phases with ReaDDy checkpoints carrying positions **and** the
  topology bond network across phases; **`--resume`‑capable** (verified by kill‑and‑resume).
- **Completed on `serial_long` (job 5320974, ~3.3 h, 8 threads).** WCA is cheaper than LJ, so the
  run sustained ~1000 steps/s.
- **Result:** clean, reproducible cycling in all five cycles — every agglomeration phase drives all
  600 particles into one percolated cluster (~599 bonds), every deagglomeration phase breaks it back
  to free monomers (≤ 9 bonds, largest cluster ~2–3). It also reproduces the thesis's key structural
  asymmetry: cluster **size** collapses sharply while the **bond count** decays only exponentially,
  because a few bridging‑bond breaks split large clusters (Eq. 6.10–6.11).
- Output: `dt_ladder/production/aggdeagg_5cyc_500us/` (per‑phase + combined trajectories, cycling
  figure `aggdeagg_analysis_fig.png`, `progress.json`).

## 7. Recommendation

Do **not** raise dt for this system — 0.05 ns is at the stability ceiling and no
force‑constant recalibration moves it without changing the physics. To reach longer
simulated timescales, invest in **wall‑clock throughput** (the `readdy-cell` fork, more
cores, and checkpoint‑chained `serial_long` jobs), which the deployed production run
demonstrates.

## 8. Data note / caveat to reconcile

One note from Benedikt's thesis worth flagging: **Table 3.6 lists kon = 10⁻⁵ ns⁻¹** for the main
LJ agglomeration study, but the **deposited campaign code uses 10⁻³** — a discrepancy in that
study's rate. It does **not** affect the agg↔deagg run (which uses the unambiguous Figure‑6.4
values: WCA, kon = 10 ns⁻¹ agglomeration / koff = 10⁻⁴ ns⁻¹ deagglomeration), nor the dt‑scaling
conclusions (which concern integrator stability in the agglomerated state, reached regardless of
the exact kon). But the kon value of the main study should be reconciled for the record.

---

*Evidence: `dt_ladder/{phase1,phase23,phase2b}/dt_ladder_{report,compare}.{md,json}`,
`dt_ladder/phase1/dt_ladder_{energy_observables,gr}.png`, `dt_scaling_summary.png`.
Tooling: `lrz/dt_ladder.py` (generator), `dt_ladder_run.py` (worker), `dt_ladder_analyze.py`
+ `dt_ladder_compare.py` (analysis), `dt_ladder_production.py` (Phase‑4).*
