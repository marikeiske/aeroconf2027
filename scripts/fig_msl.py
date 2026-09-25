import os
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis import COL, INK, INK2, GRID, FIG
d = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "msl", "msl_included.csv"))
X = ["Avionics", "Automotive", "Space/radar", "Embedded/edge", "Generic critical"]
Y = ["Mean performance", "Latency/overhead", "Schedulability/WCET", "Tail/jitter/pWCET", "Interference", "Qualitative only"]
fig, ax = plt.subplots(figsize=(3.6, 2.9))
for i, y in enumerate(Y):
    for j, x in enumerate(X):
        n = int(((d.domain == x) & (d.metric == y)).sum())
        if n:
            ax.scatter(j, i, s=60 + 70 * n, color="#86b6ef", edgecolor="#1c5cab", lw=0.8, zorder=3)
            ax.text(j, i, str(n), ha="center", va="center", fontsize=8, color=INK, zorder=4)
for i in (3, 4):
    ax.scatter(0, i, marker="*", s=150, color=COL["OpenCL"], edgecolor="white", lw=0.6, zorder=5)
ax.text(0.28, 3.5, "this work", fontsize=7.5, color=INK2, va="center")
ax.set_xticks(range(len(X))); ax.set_xticklabels(["Avionics", "Auto-\nmotive", "Space/\nradar", "Embed-\nded", "Generic\ncritical"], fontsize=7.5)
ax.set_yticks(range(len(Y))); ax.set_yticklabels(Y, fontsize=7.5)
ax.set_xlim(-0.6, len(X) - 0.4); ax.set_ylim(len(Y) - 0.5, -0.6)
ax.tick_params(length=0)
save_path = FIG + "/fig7_msl.png"; fig.savefig(save_path); print("saved", save_path)
