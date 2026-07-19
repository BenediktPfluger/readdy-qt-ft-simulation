# Qt-cluster morphology: deviation from spherical, and (in)consistency with DLA

**Question.** (1) To what degree do the simulated crosslinked clusters deviate statistically from a
spherical arrangement? (2) Is there any evidence the behaviour is *not* consistent with diffusion-
limited aggregation (DLA)? Analysis on Qt CENTROIDS only, so it maps 1:1 onto FIB-SEM Qt centroids.

## Executive Summary

- **Robustly non-spherical (for clusters of ≥ 6 cages).** Simulated Qt clusters lie far above a
  matched-N compact/spherical null at every size ≥ 6 (per-bin Cliff's δ **+0.5 → +1.0**). At N = 4–5
  the signal is weak/absent (indistinguishable from a compact blob in the aged clusters), so the
  elongation claim is scoped to **N ≥ 6**. Because the percolation-exclusion filter can only *lower*
  measured asphericity, this non-sphericity result is **conservative**.
- **NOT consistent with pure diffusion-limited aggregation — the departure is toward COMPACTION,
  not exotic branching.** With the attractive (LJ) potential on, the reference clusters are *less*
  elongated than a matched-N DLA null (Cliff's δ **−0.37 → −0.53** at the reliable N = 4–22 bins),
  and they **compact over time** (matched-N κ² falls from the early to the late window). A diffusion-
  limited, hit-and-stick aggregate cannot rearrange after sticking, so this is a genuine non-DLA
  signature.
- **The decisive, selection-proof control: attraction off (WCA) recovers DLA-like morphology.**
  The WCA full-physics run (identical to the reference but purely repulsive) is statistically
  indistinguishable from the DLA null (matched-N κ² δ ≈ **0**; fractal dimension 1.86 ≈ DLA 2.0),
  while the LJ reference is compact. Since WCA and LJ are filtered identically, this contrast is
  immune to the selection caveat below → **the compaction is driven by the short-range attraction.**
- **Mechanism is generic, not chemistry-specific.** LJ-vs-WCA isolates a generic isotropic short-
  range attraction (surface-tension-like coalescence), *not* the specific CaM–RS20 bridging. Shape
  alone cannot fingerprint the bridging chemistry.
- **Not classic reaction-limited (RLCA) either — it is attraction/coalescence-driven.** A kon sweep
  over a **100× range** (2×10⁻⁴ … 2×10⁻² ns⁻¹, matched 50 µs) barely moves the shape: matched-N κ²
  δ vs DLA stays flat at **−0.05 → −0.11** (κ² 0.25–0.32). Cluster morphology is thus **weakly
  dependent on the bonding rate**, which rules out a reaction-limited mechanism (where shape depends
  strongly on sticking probability). Combined with the WCA control, the compaction is set by the
  **attraction**, not the bonding kinetics — a coalescence regime distinct from both DLA and RLCA.

This **refines the earlier null study** (which, using only the 25 µs LJ reference and κ², concluded
"DLA-like / consistent with generic DLA"): with 100 µs, time-resolution, the attraction-off control,
and a symmetric-selection correction, the LJ reference clearly departs from DLA toward compaction.

## How it was done (and adversarially verified)

- **Sim side** (`lrz/sim_shape_plus.py`): Qt centroids per frame → d_cut contact graph (d_cut = 62 nm
  from g(r)) → per connected cluster the gyration-tensor shape (Rg, κ², aspect) via the **verbatim**
  `shape_metrics()` recipe. Reuses the canonical pipeline so sim and FIB-SEM are recipe-identical.
- **Matched-N nulls** (`lrz/dla_null_plus.py`): compact/sphere, random-Poisson, off-lattice DLA
  (Witten–Sander), and RLCA (sticking prob 0.05), generated at the sim's observed per-cluster N on the
  sim's length scale (bond = g(r) peak = 41 nm), scored with the identical `shape_metrics`.
- **Stats** (`lrz/analyze_shape_dla.py`): per-N-bin Mann–Whitney + **Cliff's δ** effect size and a
  matched-N Wilcoxon headline (δ > 0 ⇒ sim more elongated). κ² is scale-invariant, so the matched-N
  comparison is meaningful.
- **Verification (4-lens adversarial review) forced these corrections, now applied:**
  1. **Symmetric extent cut.** The percolation filter (`extent ≥ 0.5 L`, needed because a box-spanning
     cluster has undefined PBC shape) was originally applied to the sim only. It excludes 17 % of
     clusters early rising to **33 % late**, preferentially clipping elongated large clusters → it can
     *manufacture* compaction. Fixed by applying the same 250 nm extent cap to the nulls, and by
     leaning the conclusions on the two selection-immune signals: **matched-N per-bin** (not pooled
     medians) and the **WCA-vs-LJ contrast** (identical filtering). Post-correction the conclusions hold.
  2. **Fractal dimension is context only.** The geometric "DLA" null is particle-cluster (df ≈ 2.0),
     whereas the ReaDDy system is cluster–cluster (DLCA); the df gap is *not* used as evidence. The
     correct diffusion-limited reference is the **WCA full-physics run**.
  3. **Effect sizes are primary; p-values are order-of-magnitude only** (clusters are pooled across
     frames/replicas, so nominal n overstates the effective sample — the p ≪ 10⁻²⁰ headlines are not
     taken at face value).
  4. **Loops are not an independent DLA falsifier** (at d_cut > bond even the DLA null is ~98 % looped)
     — reported only as a matched-N density-normalised comparison, not as topological evidence.

## Numbers (matched-N Cliff's δ vs null, κ², symmetric extent cut)

| condition | κ² median | δ vs compact (N≥6) | δ vs DLA (N 4–22, reliable) | reading |
|---|---|---|---|---|
| compact null | 0.023 | — | — | sphere floor |
| **WCA** (attraction OFF) | 0.388 | +0.8 → +1.0 | **≈ 0** (−0.06…+0.12) | non-spherical; **DLA-like** |
| LJ ref early (10–35 µs) | 0.254 | +0.6 → +1.0 | −0.38 → −0.47 | non-spherical; ~DLA→compacting |
| **LJ ref late (65–100 µs)** | 0.119 | +0.5 → +1.0 | **−0.35 → −0.82** | non-spherical; **compact** |

(Large-N bins, N ≥ 36, are unreliable after the symmetric cut because the DLA null loses ~48 % of its
large aggregates to the extent cap; conclusions rest on N = 4–35.)

## Answers

**(1) Degree of deviation from spherical.** *"For aggregates of ≥ 6 cages the simulated clusters are
robustly non-spherical, lying far above a compact/spherical null across every replica and time window
(per-bin Cliff's δ +0.5 to +1.0). At N = 4–5 they are statistically indistinguishable from a compact
blob, so no elongation is claimed there. Because the percolation-exclusion filter can only lower
measured asphericity, this result is conservative."*

**(2) Evidence (in)consistent with DLA.** *"The morphology is **not** consistent with pure diffusion-
limited, hit-and-stick aggregation. With the attractive potential on, matched-N clusters are less
elongated and more compact than the diffusion-limited baseline, and they compact over time (κ² falls
early → late in the well-populated mid-N bins). The decisive control is the attraction-off (WCA)
full-physics run, which reproduces the diffusion-limited baseline morphology at matched N (df 1.86 vs
2.0; matched-N κ² δ ≈ 0). So the departure from DLA is **attraction-driven compaction**, a generic
attractive-colloid effect — not a fingerprint of the CaM–RS20 bridging chemistry, and not more
exotically branched than a diffusion-limited aggregate."*

## Do NOT claim
- Non-sphericity at N = 4–5 (indistinguishable from compact there).
- "More space-filling than DLA" from fractal dimension (df is context-only; wrong null family).
- p ≪ 10⁻²⁰ at face value (pooling pseudoreplication) — quote effect sizes.
- Loops as an independent topological DLA falsifier.
- A CaM–RS20-specific morphological signature (this is generic attractive-colloid compaction).

## FIB-SEM constraint framework (ready for the HEK / neuron centroids)

The pipeline is built so measured EM centroids drop straight in, recipe-identical:

1. **EM → shape:** `python lrz/vem_descriptors.py --qt <hek_or_neuron_qt_centroids>.npy --roi <imaged
   volume nm> --out em_hek` → per-cluster κ², aspect, Rg, coordination, g(r) (open-boundary twin;
   drops field-of-view-truncated clusters). Report **stimulated AND unstimulated — the difference is
   the signal.**
2. **EM vs the same nulls:** `python lrz/analyze_shape_dla.py --sim-npz em_hek_data.npz --null-npz
   <nulls>_data.npz --out em_vs_nulls` → where the real clusters sit on the sphere ↔ DLA ↔ compact
   axis (the analyzer already tolerates the EM npz's missing loop field).
3. **Constrain the model:** overlay EM κ²(N) on the sim conditions/time-windows to find which
   simulated regime (density, kon, maturation time) matches the real morphology — i.e. use the EM to
   pin the effective attraction/compaction state. Compare **shape/topology, never absolute
   counts/sizes/timescale** (the sim's diffusion is scaled).

## Provenance
- Tooling: `lrz/{sim_shape_plus,dla_null_plus,analyze_shape_dla}.py`; reuses `struct_descriptors.py`
  (`shape_metrics`) and `vem_descriptors.py` (EM twin).
- Sims: reused campaign `200Qt_400Ft_LJ…100us` (4 replicas, 2001 frames) and `…WCA…50us`; new
  kon-sweep (kon ∈ {2e-4,1e-3,5e-3,2e-2}, 50 µs, 4 replicas each) — shape weakly kon-dependent
  (matched-N κ² δ vs DLA −0.05…−0.11), i.e. not reaction-limited.
- Figure: `dt_ladder/shape_dla_verified.png`. Prior study: `readdy_null_2026-07-16/VERDICT.md`.
