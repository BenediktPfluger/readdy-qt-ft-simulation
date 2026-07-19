#!/usr/bin/env python
"""Consolidated manuscript figure from the overnight campaign."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.expanduser("~/readdy_campaign/results")
man = json.load(open(os.path.join(R, "manifest.json")))
spec = json.load(open(os.path.join(R, "specific_network.json")))


def gr(label):
    d = np.load(os.path.join(R, label + "_data.npz")); return d["gr_r"], d["gr"]


fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# A: specific percolation, mechanism 2x2 (multivalency required)
a = ax[0, 0]
labs = ["ref_200Qt_400Ft", "valency_monoFt_200Qt_400Ft", "mech_WCA_multi_200Qt_400Ft", "mech_WCA_mono_200Qt_400Ft"]
names = ["LJ · multivalent\n(reference)", "LJ · monovalent", "WCA · multivalent", "WCA · monovalent"]
vals = [spec[l]["largest_spec_frac"] for l in labs]
bars = a.bar(names, vals, color=["#2a7", "#e84", "#8ac", "#c44"])
a.set_ylabel("specific percolation\n(largest bond-cluster / total Qt)")
a.set_title("A  Multivalency drives the specific network", fontweight="bold")
for b, v in zip(bars, vals):
    a.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
a.set_ylim(0, 0.65)

# B: g(r) spacer shift (bridge vs contact)
b = ax[0, 1]
for label, nm, c in [("ref_200Qt_400Ft", "εQQ=1.5 (ref)", "#c44"),
                     ("spacer_eQQ0.5_200Qt_400Ft", "εQQ=0.5", "#e84"),
                     ("spacer_eQQ0.1_200Qt_400Ft", "εQQ=0.1", "#2a7"),
                     ("mech_WCA_mono_200Qt_400Ft", "WCA (bonds only)", "#468")]:
    r, g = gr(label); b.plot(r, g, label=nm, color=c, lw=1.8)
b.axvline(42, ls=":", c="gray"); b.axvline(54, ls="--", c="k")
b.text(42, b.get_ylim()[1]*0.9, " Qt-Qt\n contact", fontsize=8)
b.text(54, b.get_ylim()[1]*0.7, " ferritin\n bridge", fontsize=8)
b.set_xlim(20, 100); b.set_xlabel("Qt-Qt distance (nm)"); b.set_ylabel("g(r)")
b.set_title("B  50-60 nm spacing is bridge-enforced", fontweight="bold"); b.legend(fontsize=8)

# C: elongation kappa2 across clustering conditions
c = ax[1, 0]
order = ["ref_200Qt_400Ft", "ratio_200Qt_200Ft", "ratio_200Qt_1000Ft", "ratio_400Qt_200Ft",
         "valency_monoFt_200Qt_400Ft", "spacer_eQQ0.1_200Qt_400Ft", "mech_WCA_multi_200Qt_400Ft"]
short = ["200/400", "200/200", "200/1000", "400/200", "200/400\nmono", "εQQ0.1", "WCA"]
k2 = [man[l]["mean_shape_anisotropy_kappa2"] for l in order]
c.bar(short, k2, color="#68a")
c.axhline(0.0, c="k", lw=.5)
c.set_ylabel("shape anisotropy κ²  (0 sphere, 1 rod)")
c.set_title("C  Clusters are elongated in every condition", fontweight="bold")
c.set_ylim(0, 0.5)
for i, v in enumerate(k2):
    c.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

# D: stoichiometry -> ferritin engagement + percolation
d = ax[1, 1]
ratio = ["ratio_600Qt_50Ft", "ratio_600Qt_200Ft", "ratio_400Qt_200Ft", "ratio_200Qt_200Ft",
         "ref_200Qt_400Ft", "ratio_200Qt_1000Ft"]
rlab = ["600/50", "600/200", "400/200", "200/200", "200/400", "200/1000"]
rough = [spec[l]["specific_rough_Qt"] for l in ratio]
perc = [spec[l]["largest_spec_frac"] for l in ratio]
x = np.arange(len(ratio))
d.bar(x - 0.2, rough, 0.4, label="ferritin-engaged Qt", color="#7b5")
d.bar(x + 0.2, perc, 0.4, label="specific percolation", color="#37a")
d.set_xticks(x); d.set_xticklabels(rlab); d.set_xlabel("Qt / Ft")
d.set_ylabel("fraction"); d.set_title("D  Ferritin sufficiency gates the network", fontweight="bold")
d.legend(fontsize=8); d.set_ylim(0, 1.05)

fig.tight_layout()
out = os.path.join(R, "campaign_figure.png")
fig.savefig(out, dpi=140); print("FIGURE", out)
