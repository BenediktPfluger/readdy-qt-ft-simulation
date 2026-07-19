# ReaDDy Qt–Ft agglomeration campaign — LRZ handover (for Benedikt)

**Author:** Gil / Westmeyer Lab (via Claude Code, on ambigram) · **Rev 2, 2026-07-17**
**Scope:** how to run and extend the CLEM-GECI ReaDDy structural campaign on LRZ
CoolMUC-4, plus the *vetted* state of the analysis so a collaborator stays aligned with
the manuscript's (deliberately conservative) claims.

> Rev 2 corrects Rev 1: fractal dimension is **not** the DLA discriminator (Rev 1 wrongly
> used it), the DLA/random-bond null model is **already done**, and the native-linux LRZ
> run is now **verified working** (Rev 1 listed it as untested). Details below.

---

## 0. TL;DR

Coarse-grained ReaDDy2 model of your **encapsulin–ferritin Ca²⁺ agglomeration sensor**
(master-thesis model; repo `github.com/BenediktPfluger/readdy-qt-ft-simulation`):
- **Qt** = encapsulin scaffold, one sphere r = 21 nm (protomer M-Qt-M-eUnaG-NES-RS20,
  presents **RS20**).
- **Ft** = ferritin-sized crosslinker, one sphere r = 6 nm (CaM-mScarlet-HpFtn, presents **CaM**).
- On Ca²⁺, CaM–RS20 binding crosslinks the cages → reversible agglomeration into
  fluorescent puncta. The crux is crosslinker **valency**.

The sim supports **two defensible morphology findings** (an adversarial integrator/skeptic/
reviewer pass fixed their exact scope — see §2). A matched-N **null-model study** (done
2026-07-16) bounds Finding A. The 11-condition campaign is **currently running natively on
LRZ** (serial cluster) — and that native-linux run is now confirmed to work, which was the
open cross-platform-validation item.

---

## 1. Rationale — what the model tests

Two morphology signatures, both aimed at a volume-EM (FIB-SEM/ExM) comparison:

1. **Is the agglomerate non-spherical / elongated?** — gyration-tensor relative shape
   anisotropy κ² (0 = sphere, 1 = rod) and aspect ratio, measured per Qt-cluster and
   compared against size-matched geometric nulls (not just "it aggregates").
2. **Is agglomeration gated by binding or by valency?** — does making ferritin
   *monovalent* (one CaM patch) abolish the percolating specific-bridging network even
   though CaM–RS20 binding still occurs?

Model: two topology species crosslink (`seed`/`grow`/`merge` topology reactions, kon =
0.001, binding radius 27.25 nm, harmonic bond k = 10) into a `QtFt_Cluster`; non-bonded
LJ (with attraction) or WCA (repulsive only); box 500³ nm, T = 300 K, dt = 50 ps, CPU
kernel, 4 threads. Descriptors: κ², aspect, Qt coordination, largest-/specific-network
fraction, g(r) bridging peak. Full spec: `READDY_BRIEF.md` (rechnma bundle / lab-notebooks).

---

## 2. What is established (vetted) — and what NOT to claim

### Finding A — clusters are elongated/branched, not spherical (bounded)
Robust across LJ *and* WCA: κ² ≈ 0.24–0.43, aspect ≈ 4–9 (well-sampled conditions only).
A dedicated **matched-N null study** (compact-sphere / spatially-random / off-lattice DLA
nulls, 300 replicas × each observed sim-N, identical `shape_metrics()`) pins the exact
scope:
- **vs a compact sphere:** far more elongated at *every* size (Cliff's δ +0.61→+1.00). This
  part is unambiguous.
- **vs an equal-size random point set:** significantly more elongated **for N ≥ 6**
  (δ grows to +0.99 with N; 28/28 N-bins sim > random). **Not** distinguishable at N = 4–5
  (29 % of clusters) — so lead with κ², restrict quoted aspect to N ≥ 6 (aspect at N = 4 is
  a degenerate-eigenvalue artifact).
- **vs a DLA null: DLA-like, not beyond it.** The sim is *less* branched than DLA at N ≤ 10,
  converges to DLA by N ≈ 15–19, only marginally exceeds it at the largest sparse sizes
  (Wilcoxon-greater p = 1.0). → The elongation is **"consistent with" generic diffusion-
  limited aggregation; it is NOT a distinguishable, specific signature of the CaM–RS20
  bridging chemistry.** Frame as *"consistent with"*, never *"validates"*.
- Verdict + figure: `readdy_null_2026-07-16/VERDICT.md` and Doc
  `docs.google.com/document/d/1Zx5e_8BPRYJyJ_E8PqU-rE2T5pkTYzP3u8-e7OaxFQQ`.

### Finding B — agglomeration is multivalency-gated, not binding-gated
Monovalent ferritin keeps ≈ 89 % of cages ferritin-bound yet collapses **specific-network
percolation 0.557 → 0.005**. Prediction: an internal-steric-occlusion / occluder titration
abolishes puncta while CaM–RS20 binding persists. (This is the highest-value experimentally-
testable output — and the target of the next sim, §5.)

### Do NOT claim (killed by the adversarial pass — do not resurrect)
- **Fractal dimension as the "branched" evidence.** Same-estimator df: **sim 2.30 > DLA 1.92
  < compact 2.89** — the sim is anisotropic yet *more* space-filling than an ideal DLA
  fractal. Elongation (κ²) and openness (df) are different axes; a naive df read would call
  the sim "compact" and contradict "branched". df is **context only**, never a discriminator.
- **"50–60 nm spacing is bridge-enforced"** (circular) and **"10× ferritin / percolation
  threshold"** (the sim never tests 10:1; percolation saturates by ~2:1).
- Absolute kinetics/timescale; spatial-graph clustering as *specific* bridging; κ²/aspect of
  the few-cluster conditions (600 Qt/50 Ft, 600 Qt/200 Ft); WCA absolutes (under-converged
  at 50 µs).

---

## 3. The current LRZ run (native-linux, VERIFIED)

- **Cluster / partition / QOS:** `serial` / `serial_std` / `cm4_serial`
  (**not** cm4_std/cm4_tiny — their QOS enforce a large-core floor: cm4_tiny MinTRES
  cpu = 17, cm4_std cpu = 112 → small jobs are rejected with `QOSMinCpuNotSatisfied`.
  `serial` = ≤ 32 cores/job, MaxSubmit = 200, 1-day walltime.) Each replica = 4 cores / 6 G.
- **✅ conda-forge `readdy` runs on the post-OS-upgrade cm4 compute nodes** — verified
  2026-07-17: a canary and all 32 replicas execute ReaDDy correctly on `cm4r*` compute
  nodes. This resolves the earlier "compiled-after-upgrade won't run / untested" concern
  (readdy is a prebuilt conda package, not locally compiled). This native run is the
  cross-platform check that an ambigram Rosetta rerun can *not* provide.
- **Paths (ga49qal2's LRZ home):** repo `/dss/dsshome1/0C/ga49qal2/readdy-qt-ft-simulation/`;
  env `~/miniforge3/envs/readdy` (readdy 2.0.14 linux-64) built by `lrz/bootstrap_login.sh`;
  each condition → `<repo>/<folder>/{configs,logs,replica_00N/trajectory.h5}`.
- **Jobs (2026-07-17):** arrays `5316953` (canary) + `5316959`–`5316969` = 11 conditions,
  32 replicas (same params/reps as the ambigram campaign). Postprocess job `5317035`
  (`--dependency=afterany:…`) auto-builds `results/manifest_lrz.json` when they drain.

---

## 4. How to run more campaigns (quick-start)

Two files in `lrz/` drive everything.

**One-time env build (on a login node — needs internet):**
```bash
cd ~/readdy-qt-ft-simulation && bash lrz/bootstrap_login.sh   # Miniforge + readdy env, idempotent
```

**Generate / submit / postprocess:**
```bash
source ~/miniforge3/bin/activate ~/miniforge3/envs/readdy
cd ~/readdy-qt-ft-simulation
python lrz/lrz_campaign.py                       # generate 11 SLURM scripts + campaign_index.json (no submit)
python lrz/lrz_campaign.py --submit              # generate + sbatch all
python lrz/lrz_campaign.py --only <label> --submit --array 0   # 1-replica canary
python lrz/lrz_campaign.py --postprocess         # descriptors -> results/manifest_lrz.json (after runs finish)
```
SLURM knobs and the `CONDITIONS` list live at the top of `lrz/lrz_campaign.py`
(`CLUSTER=serial`, `PARTITION=serial_std`, `QOS=cm4_serial`, `CPUS=4`, `MEM=6G`).

**Add a condition:** edit `CONDITIONS` — `C(label, nqt, nft, steps, reps, eqq, eff, eqf,
pot, mono)`. `steps × 0.05 ps` = length (1e6 = 50 µs). `pot="WCA"` = attraction off;
`mono=True` = monovalent Ft. Per-replica seeds derive deterministically from base
`rng_seed=1` via `np.random.SeedSequence` — **bump the base seed for independent
replicates**, and point at a fresh results dir (the driver skips finished conditions).

**Monitor / analyze / pull:**
```bash
squeue -M serial -u $USER ; tail -f <repo>/<folder>/logs/replica_0.out
python scripts/analyze_ensemble.py --ensemble-dir <repo>/<folder> --parallel --n-workers 4 --stride 10
rsync -rlpt lrz:'~/readdy-qt-ft-simulation/<folder>/ensemble_*' ./<folder>/
```

---

## 5. Suggested next campaigns (highest value first)

1. **Discrete-patch, explicit-valency ferritin** — each cage = core + N rigidly-bonded
   surface receptor particles; binding is patch–patch so valency & partial coverage become
   *emergent* (set N_active). Turns **Finding B into a quantitative occluder-titration
   prediction**. Heavy (200 Qt × up to ~60 patches) → run supervised, not unattended.
2. **Extend the matched-N null to the other 10 conditions** — the null study currently
   covers only the 200 Qt/400 Ft LJ reference; re-running it per condition would let Finding
   A be stated per-condition rather than for the reference alone.
3. **Full-production statistics** — 10 replicas (and ref/ratio at 100 µs) for real error
   bars on κ²/percolation. On `serial` replicas run concurrently, so wall-clock ≈ unchanged.
4. **vEM comparison** — run `lrz/vem_descriptors.py` on FIB-SEM/ExM Qt centroids in the same
   pipeline when the EM team exports them (`VEM_PROTOCOL.md` ready).

---

## 6. Path & document reference

- **This handover:** `<repo>/lrz/HANDOVER_Benedikt.md` (ambigram + LRZ mirrors).
- **LRZ driver / bootstrap:** `<repo>/lrz/lrz_campaign.py`, `lrz/bootstrap_login.sh`.
- **Analysis:** `<repo>/scripts/{run_replica,analyze_ensemble}.py`,
  `<repo>/lrz/{struct_descriptors,vem_descriptors,bond_network,campaign_figure}.py`.
- **Model source:** `<repo>/qtft/` (`config.py`, `system.py`, `engine.py`, `ensemble.py`).
- **Ambigram campaign (11/11 done, 1.0 GB):** `/Users/ambigram/readdy_campaign/`.
- **DLA null study:** `/Users/ambigram/claude_out/handoff-20260716-093014-79101-4010/readdy_null_2026-07-16/`.
- **Docs:** Findings `docs.google.com/document/d/1sBI8J43PF4sT3wyYBMLn-bIm7dBr1PBMnkYt9reZQng` ·
  DLA-null verdict `…/1Zx5e_8BPRYJyJ_E8PqU-rE2T5pkTYzP3u8-e7OaxFQQ` ·
  Manuscript id `13PYgb6-t_9Iidy9Ft5YstKzOlz1oWWxDlaGIi7WEYYI`.

*Biology/parameter questions → your call (it's your model). LRZ/SLURM mechanics → this file
+ the repo READMEs.*
