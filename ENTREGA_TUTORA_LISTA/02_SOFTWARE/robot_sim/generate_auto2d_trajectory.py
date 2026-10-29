"""Render a saved successful Auto 2D route for the thesis."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from car_env import GOAL_POS, GOAL_R, MAP_SIZE, OBST_R, CarEnv

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).resolve().parent / "outputs" / "run_car_seed0.json"
FIGS = (
    ROOT / "docs" / "latex" / "figuras" / "auto2d_trayectoria_seed0.png",
    ROOT / "PLANTILLA_PDF_FINAL" / "figuras" / "auto2d_trayectoria_seed0.png",
)


def main():
    record = json.loads(SOURCE.read_text(encoding="utf-8"))
    trail = np.asarray(record.get("win_trail", []), dtype=float)
    if trail.ndim != 2 or trail.shape[1] != 2 or len(trail) < 2:
        raise ValueError(f"No hay una trayectoria exitosa utilizable en {SOURCE}")
    if np.hypot(*(trail[-1] - np.asarray(GOAL_POS))) > GOAL_R:
        raise ValueError("La trayectoria archivada no termina dentro de la meta")

    env = CarEnv(seed=int(record["seed"]))
    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=220)
    ax.add_patch(Rectangle((0, 0), MAP_SIZE, MAP_SIZE,
                           facecolor="#f5f7fa", edgecolor="#344054",
                           linewidth=1.5, zorder=0))
    for x, y in env.obstacles:
        ax.add_patch(Circle((x, y), OBST_R, facecolor="#98a2b3",
                            edgecolor="#475467", linewidth=1.2, zorder=1))
    ax.add_patch(Circle(GOAL_POS, GOAL_R, facecolor="#abefc6",
                        edgecolor="#067647", linewidth=1.5,
                        alpha=0.9, zorder=2))
    ax.plot(trail[:, 0], trail[:, 1], color="#175cd3", linewidth=2.4,
            label="Trayectoria guardada", zorder=3)
    ax.scatter(*trail[0], color="#1570ef", edgecolor="white", linewidth=0.8,
               s=70, label="Inicio", zorder=4)
    ax.scatter(*trail[-1], color="#b42318", marker="*", edgecolor="white",
               linewidth=0.8, s=150, label="Llegada guardada", zorder=5)
    ax.set_xlim(-0.25, MAP_SIZE + 0.25)
    ax.set_ylim(-0.25, MAP_SIZE + 0.25)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, int(MAP_SIZE) + 1, 2))
    ax.set_yticks(range(0, int(MAP_SIZE) + 1, 2))
    ax.grid(color="white", linewidth=1.0, zorder=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Auto 2D — trayectoria exitosa guardada (semilla 0)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.tight_layout()

    for target in FIGS:
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Figura guardada: {target}")
    plt.close(fig)


if __name__ == "__main__":
    main()
