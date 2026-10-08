"""Consolida las curvas de aprendizaje de las 3 semillas de la vision 8x8
(Fig. vision_seeds) a partir de los JSON guardados por run_vision.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG = Path(__file__).resolve().parent.parent / "docs" / "latex" / "figuras" / "vision_seeds.png"

WINDOW = 100
SEEDS = (0, 1, 2)
COLORS = ["tab:blue", "tab:orange", "tab:green"]

fig, ax = plt.subplots(figsize=(9, 4.5))
for s, c in zip(SEEDS, COLORS):
    d = json.loads((OUT_DIR / f"run_vision_seed{s}.json").read_text())
    acc = np.array(d["correct"], dtype=float)
    if len(acc) >= WINDOW:
        smooth = np.convolve(acc, np.ones(WINDOW) / WINDOW, mode="valid")
        ax.plot(np.arange(len(smooth)), smooth, lw=1.6, color=c,
                label=f"semilla {s}")

ax.axhline(1 / 8, ls="--", color="gray", label="aleatorio (1/8)")
ax.set_xlabel("Episodio")
ax.set_ylabel(f"Accuracy (media m\u00f3vil {WINDOW})")
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 5000)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
ax.set_title("Robustez entre semillas \u2014 Visi\u00f3n 8\u00d78")
fig.tight_layout()
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=200, bbox_inches="tight")
print(f"Figura guardada: {FIG}")