"""Genera el diagrama conceptual del agente neuromorfico (Fig. agente_arquitectura).

Flujo: Observacion -> Codificador top-k -> Crossbar N x M -> LIF -> Softmax -> Accion,
con bucle de aprendizaje R-STDP modulado por recompensa sobre el crossbar.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parent.parent / "docs" / "latex" / "figuras" / "agente_arquitectura.png"

fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis("off")

FC = "#eaf3fb"
EC = "#2b6cb0"
RD = "#c53030"


def box(x, y, w, h, title, sub, fc=FC, ec=EC):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
            fontsize=11, fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
            fontsize=8.5, color="#333333", zorder=3)


def arrow(x1, y1, x2, y2, color=EC, ls="solid", lw=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=22, linewidth=lw, color=color,
                                 linestyle=ls, zorder=1))


# --- Cadena directa (fila superior) ---
box(0.15, 3.3, 1.9, 1.6, "Observaci\u00f3n", "estado del\nentorno")
box(2.6, 3.3, 1.9, 1.6, "Codificador", "unipolar \u00b7 top-$k$\nvoltajes $V_{rows}$")
box(5.05, 3.3, 1.9, 1.6, "Crossbar $N\\times M$", "conductancias\n$G_{ij}$")
box(7.5, 3.3, 1.9, 1.6, "Neuronas LIF", "integraci\u00f3n\n$I_{col}$")
box(9.95, 3.3, 1.9, 1.6, "Softmax", "temperatura\n$T\\searrow$")

arrow(2.05, 4.1, 2.6, 4.1)
arrow(4.5, 4.1, 5.05, 4.1)
arrow(6.95, 4.1, 7.5, 4.1)
arrow(9.4, 4.1, 9.95, 4.1)

# --- Bloque de aprendizaje R-STDP (inferior) ---
box(4.55, 0.3, 2.9, 1.5, "Actualizaci\u00f3n R-STDP", "$\\Delta G=-\\eta\\,E\\,(R-b)$\n(elegibilidad $E$)")

# --- Salida: Softmax -> Accion ---
arrow(10.9, 3.3, 10.9, 2.3)
ax.text(11.55, 2.8, "Acci\u00f3n", fontsize=10, fontweight="bold",
        ha="center", color=EC)

# --- Bucle de recompensa (entorno -> R-STDP) ---
arrow(11.5, 1.7, 7.45, 1.05, color=RD, ls="--")
ax.text(9.5, 1.05, "recompensa $R$ vs. baseline $b$", fontsize=8.5,
        color=RD, ha="center")
# --- Escritura de G sobre el crossbar (R-STDP -> Crossbar) ---
arrow(6.0, 1.8, 6.0, 3.3, color=RD, ls="--", lw=1.8)
ax.text(6.55, 2.4, "escribe $G$", fontsize=8.5, color=RD, ha="left")

fig.tight_layout()
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Diagrama guardado: {FIG}")