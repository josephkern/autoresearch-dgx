"""One-shot script to render v1 vs v2 comparison plot for DGX_SPARK_NOTES.md.

Run inside the container (has matplotlib):
  docker exec autoresearch-dgx python3 /workspace/plot_v1_v2.py

Output: dgx_v1_v2_comparison.png
"""

import csv
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["val_bpb"] = float(r["val_bpb"])
        r["memory_gb"] = float(r["memory_gb"])
    return rows


v1 = load("/workspace/results_v1.tsv")
v2 = load("/workspace/results.tsv")

# Running best (the actual descent line, including only keeps in chronological order)
def running_best(rows):
    out = []
    best = float("inf")
    for r in rows:
        if r["status"] == "keep":
            best = min(best, r["val_bpb"])
        out.append(best if best < float("inf") else r["val_bpb"])
    return out

v1_running = running_best(v1)
v2_running = running_best(v2)

# Per-experiment val_bpb (all rows including discards)
v1_vals = [r["val_bpb"] for r in v1]
v2_vals = [r["val_bpb"] for r in v2]
v1_keeps = [(i, r["val_bpb"]) for i, r in enumerate(v1) if r["status"] == "keep"]
v2_keeps = [(i, r["val_bpb"]) for i, r in enumerate(v2) if r["status"] == "keep"]
v1_disc = [(i, r["val_bpb"]) for i, r in enumerate(v1) if r["status"] == "discard"]
v2_disc = [(i, r["val_bpb"]) for i, r in enumerate(v2) if r["status"] == "discard"]

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

for ax, env, vals, keeps, disc, running, color in [
    (axes[0], "v1 (venv / PyTorch 2.9.1 / MAGMA fallback)",
        v1_vals, v1_keeps, v1_disc, v1_running, "#1f77b4"),
    (axes[1], "v2 (NVIDIA container / PyTorch 2.12 / cuBLAS)",
        v2_vals, v2_keeps, v2_disc, v2_running, "#d62728"),
]:
    # discards as light dots
    if disc:
        xs, ys = zip(*disc)
        ax.scatter(xs, ys, s=18, c="lightgray", alpha=0.7, label="discard", zorder=2)
    # keeps as dark dots
    if keeps:
        xs, ys = zip(*keeps)
        ax.scatter(xs, ys, s=42, c=color, label="keep", zorder=4)
    # running-best line
    ax.plot(range(len(running)), running, c=color, lw=2.0, alpha=0.85,
            label="best so far", zorder=3)
    ax.set_title(env, fontsize=11)
    ax.set_xlabel("experiment #")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(1.10, 1.50)

axes[0].set_ylabel("val_bpb (lower is better)")

# Annotate key transitions on v2 plot.
# x is the EXPERIMENT INDEX in results.tsv (the row of the keep), not the keep number.
# Text offsets are tuned by hand to avoid overlap.
ann_v2 = [
    # (exp_idx,       val_bpb,  text,                                  text_dx, text_dy)
    (2,             1.256720, "AR=128\n(arch win)",                    +6,  +0.05),
    (13,            1.204727, "TBS=2¹⁵\n(ga 4→2)",                     -3,  +0.08),
    (16,            1.152971, "TBS=2¹⁴\n(ga 2→1, 4K steps)",           +3,  +0.12),
    (43,            1.137750, "DBS=16, TBS=2¹⁵\n(big batch + ga=1)",   -23, +0.05),
    (55,            1.132811, "joint LR retune\n(final 1.133)",        -22, -0.025),
]
for x, y, txt, dx, dy in ann_v2:
    axes[1].annotate(txt, xy=(x, y), xytext=(x + dx, y + dy),
                     fontsize=8, ha="left",
                     arrowprops=dict(arrowstyle="->", color="gray", lw=0.7,
                                     connectionstyle="arc3,rad=0.1"))

# Summary text
v1_best = min(r["val_bpb"] for r in v1 if r["status"] == "keep")
v2_best = min(r["val_bpb"] for r in v2 if r["status"] == "keep")
fig.suptitle(
    f"DGX Spark autoresearch: v1 → v2 environment swap\n"
    f"v1 best: {v1_best:.4f} (12 keeps / 48 exp)  →  "
    f"v2 best: {v2_best:.4f} (16 keeps / 60 exp)  →  "
    f"Δ = {v1_best - v2_best:+.4f} val_bpb",
    fontsize=12, y=1.02,
)

plt.tight_layout()
out = "/workspace/dgx_v1_v2_comparison.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}")
