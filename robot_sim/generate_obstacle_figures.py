"""Figuras del gridworld 2D con obstaculos (3.3.7).

Genera en docs/latex/figuras/:
  obstacle_mapa.png         gridworld + obstaculos + meta (seed con mejor tasa)
  obstacle_trayectoria.png  antes (ep 0) vs despues (rollout greedy final)
  obstacle_learning.png     tasa de exito de las 3 semillas (media movil 50)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from obstacle_env import ObstacleGridworld

OUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "latex" / "figuras"
SEEDS = (0, 1, 2)
WINDOW = 50


def _log(seed):
    return json.loads((OUT_DIR / f"run_obstacle_seed{seed}.json").read_text())


def _paths(seed):
    return json.loads(
        (OUT_DIR / f"obstacle_seed{seed}_paths.json").read_text()
    )


def random_success_rate(env, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n):
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(int(rng.integers(4)))
        ok += int(env.agent == env.goal)
    return ok / n


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Mejor seed por tasa de exito en los ultimos 200 episodios
    best = max(SEEDS, key=lambda s: float(np.mean(_log(s)["success"][-200:])))
    print(f"semilla elegida para mapa/trayectoria: {best}")

    # ----- 1) Mapa -----
    env = ObstacleGridworld(seed=best)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    env.draw(ax, path=None,
             title="Gridworld 10$\times$10 — obst\u00e1culos y meta")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "obstacle_mapa.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ----- 2) Trayectoria antes / despues -----
    p = _paths(best)
    early = [tuple(q) for q in p["early_path"]]
    learned = [tuple(q) for q in p["learned_path"]]
    env2 = ObstacleGridworld(seed=best)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    env2.draw(axes[0], path=early,
              title="Antes — pol\u00edtica aleatoria (episodio 0)")
    env2.draw(axes[1], path=learned,
              title="Despu\u00e9s — roll-out determinista")
    fig.suptitle(f"Trayectorias — semilla {best}", fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "obstacle_trayectoria.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ----- 3) Curva de aprendizaje 3 semillas -----
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for s, c in zip(SEEDS, colors):
        succ = np.array(_log(s)["success"], dtype=float)
        if len(succ) >= WINDOW:
            sm = np.convolve(succ, np.ones(WINDOW) / WINDOW, mode="valid")
            ax.plot(np.arange(len(sm)), sm, lw=1.6, color=c,
                    label=f"semilla {s}")
    rand = random_success_rate(ObstacleGridworld(seed=0))
    ax.axhline(rand, ls="--", color="gray",
               label=f"aleatorio ({rand * 100:.0f}%)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel(f"Tasa de éxito (media m\u00f3vil {WINDOW})")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_title("Navegaci\u00f3n con obst\u00e1culos — 3 semillas")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "obstacle_learning.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[ok] figuras en {FIG_DIR}")


if __name__ == "__main__":
    main()
