# -*- coding: utf-8 -*-
# Combined figure; ONLY numbers and colors updated per the provided charts.

import matplotlib.pyplot as plt
import numpy as np

datasets = ["Test2016", "Test2017", "MSCOCO"]
x = np.arange(len(datasets))
w = 0.17

# ---- UPDATED DATA ----
# BLEU
bleu_cong = [38.63, 32.12, 28.73]
bleu_incg = [37.94, 31.00, 27.83]
# METEOR
meteor_cong = [64.59, 59.53, 54.40]
meteor_incg = [64.22, 58.72, 53.54]

# ---- UPDATED COLORS (match sample image) ----
# BLEU: purple family
C_BLEU_CONG = "#6a0dad"   # deep purple
C_BLEU_INCG = "#c39bd3"   # light lavender
# METEOR: green family
C_MET_CONG  = "#2ca02c"   # deep green
C_MET_INCG  = "#98df8a"   # light green

fig, ax1 = plt.subplots(figsize=(10, 5.2))

# BLEU (left axis)
b1 = ax1.bar(x - w*1.1, bleu_cong, width=w, label="BLEU (Congruent)",
             color=C_BLEU_CONG, edgecolor="black", linewidth=0.6)
b2 = ax1.bar(x - w*0.0, bleu_incg, width=w, label="BLEU (Incongruent)",
             color=C_BLEU_INCG, edgecolor="black", linewidth=0.6)
ax1.set_ylabel("BLEU"); ax1.set_ylim(24, 42)
ax1.set_xticks(x); ax1.set_xticklabels(datasets)
ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

# METEOR (right axis)
ax2 = ax1.twinx()
m1 = ax2.bar(x + w*1.1, meteor_cong, width=w, label="METEOR (Congruent)",
             color=C_MET_CONG, edgecolor="black", linewidth=0.6)
m2 = ax2.bar(x + w*2.2, meteor_incg, width=w, label="METEOR (Incongruent)",
             color=C_MET_INCG, edgecolor="black", linewidth=0.6)
ax2.set_ylabel("METEOR"); ax2.set_ylim(46, 67)

# value labels on bars
ax1.bar_label(b1, labels=[f"{v:.2f}" for v in bleu_cong], padding=2, fontsize=9)
ax1.bar_label(b2, labels=[f"{v:.2f}" for v in bleu_incg], padding=2, fontsize=9)
ax2.bar_label(m1, labels=[f"{v:.2f}" for v in meteor_cong], padding=2, fontsize=9)
ax2.bar_label(m2, labels=[f"{v:.2f}" for v in meteor_incg], padding=2, fontsize=9)

# legend below axes (unchanged)
handles = [b1, b2, m1, m2]
labels  = [h.get_label() for h in handles]
ax1.legend(handles, labels, ncol=2, loc="upper center",
           bbox_to_anchor=(0.5, -0.08), frameon=False)

fig.suptitle("Multi30K En–De: Congruent vs. Incongruent Decoding", y=0.99)
fig.tight_layout(); plt.subplots_adjust(bottom=0.18)
plt.savefig("multi30k_congruent_incongruent_combined_updated.png", dpi=300, bbox_inches="tight")
plt.show()
