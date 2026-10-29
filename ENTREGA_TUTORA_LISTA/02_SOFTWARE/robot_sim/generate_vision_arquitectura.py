"""Genera el diagrama del flujo de inferencia y aprendizaje visual 8x8."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIGS = (
    ROOT / "docs" / "latex" / "figuras" / "vision_arquitectura.png",
    ROOT / "PLANTILLA_PDF_FINAL" / "figuras" / "vision_arquitectura.png",
)

BLUE = "#2563a6"
PALE_BLUE = "#eaf3fb"
RED = "#b42318"
PALE_RED = "#fff1f0"
GRAY = "#475467"

fig, ax = plt.subplots(figsize=(13, 7), dpi=220)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")


def box(x, y, width, height, title, detail, fill=PALE_BLUE, edge=BLUE):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.5, edgecolor=edge, facecolor=fill, zorder=2,
    ))
    ax.text(x + width / 2, y + height * 0.66, title,
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#172b4d", zorder=3)
    ax.text(x + width / 2, y + height * 0.30, detail,
            ha="center", va="center", fontsize=8.5, color=GRAY, zorder=3)


def arrow(start, end, color=BLUE, style="-", width=1.8,
          connection="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16,
        linewidth=width, color=color, linestyle=style,
        connectionstyle=connection, zorder=1,
    ))


# Flujo de inferencia.
top_y, top_h, box_w = 5.45, 1.35, 2.15
top_x = (0.25, 2.95, 5.65, 8.35, 11.05)
stages = (
    ("Imagen 8 × 8", "64 píxeles binarios"),
    ("Retina", "codificación de filas"),
    ("Crossbar 64 × 8", "conductancias $G_{ij}$"),
    ("Competencia suave", "softmax sobre $I_{col}$"),
    ("Clase predicha", "$\\hat{y}=\\operatorname{argmax}(p)$"),
)
for x, (title, detail) in zip(top_x, stages):
    box(x, top_y, box_w, top_h, title, detail)
for i in range(len(top_x) - 1):
    arrow((top_x[i] + box_w, top_y + top_h / 2),
          (top_x[i + 1], top_y + top_h / 2))

# Bucle de aprendizaje supervisado: clase predicha y etiqueta de referencia.
box(8.9, 3.05, 2.8, 1.25, "Evaluación supervisada",
    "$R=+1$ si $\\hat{y}=y$; $R=-1$ si no",
    fill="#f2f4f7", edge="#667085")
box(12.0, 3.05, 1.7, 1.25, "Etiqueta real", "$y$ del patrón",
    fill="#f2f4f7", edge="#667085")
arrow((12.12, top_y), (10.6, 4.30), color=GRAY, width=1.5)
arrow((12.0, 3.68), (11.7, 3.68), color=GRAY, width=1.5)

box(5.05, 0.65, 3.8, 1.45, "Actualización REINFORCE",
    "$\\Delta X=-\\eta E(R-b)$",
    fill=PALE_RED, edge=RED)
arrow((9.65, 3.05), (8.15, 2.10), color=RED, style="--", width=1.7)
arrow((6.95, 2.10), (6.95, top_y), color=RED, style="--", width=1.7)
ax.text(7.16, 3.9, "actualiza $X_{ij}$", rotation=90,
        ha="left", va="center", fontsize=8.5, color=RED)

fig.tight_layout(pad=1.2)
for target in FIGS:
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Diagrama guardado: {target}")
plt.close(fig)
