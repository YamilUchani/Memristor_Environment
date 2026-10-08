"""
Genera las 4 figuras del capítulo a partir de los logs de las 4 corridas.
Uso:  python plot_results.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUT = Path(__file__).resolve().parent / "outputs"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------
# Configuración de las 4 corridas
# ------------------------------------------------------------------
RUNS = {
    "A  REINFORCE acumulado":     ("run_A_base.json",     "tab:blue"),
    "B  temp_min=0.1":            ("run_B_temp01.json",   "tab:orange"),
    "C  sin acumulación":         ("run_C_noaccum.json",  "tab:green"),
    "D  Hebbian sin homeostasis": ("run_D_hebbian.json",  "tab:red"),
}

# ------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------
def load(name):
    p = OUT / name
    if not p.exists():
        print(f"[warn] falta {p}, se omite")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def smooth(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")

def rolling_percentile(x, w=50, p=50):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.array([np.percentile(x[i:i+w], p) for i in range(len(x)-w+1)])


def find_X_file():
    """Busca el .npy de la matriz X final de la corrida A."""
    for name in ("run_A_base_X.npy", "cartpole_2000_X.npy", "cartpole_X.npy"):
        p = OUT / name
        if p.exists():
            return p
    return None


# ------------------------------------------------------------------
# Cargar datos
# ------------------------------------------------------------------
data = {label: load(fname) for label, (fname, _) in RUNS.items()}
data = {k: v for k, v in data.items() if v is not None}
if not data:
    raise RuntimeError(f"No hay ningún JSON en {OUT}")

print(f"[info] corridas cargadas: {list(data.keys())}")


# ==================================================================
# FIGURA 1 — Curva de aprendizaje
# ==================================================================
fig, ax = plt.subplots(figsize=(8, 5))
for label, log in data.items():
    color = RUNS[label][1]
    r = np.asarray(log["reward"], dtype=float)
    rs = smooth(r, w=50)
    r25 = rolling_percentile(r, w=50, p=25)
    r75 = rolling_percentile(r, w=50, p=75)
    x_axis = np.arange(len(rs))
    ax.plot(x_axis, rs, label=label, color=color, lw=1.8)
    ax.fill_between(x_axis, r25, r75, color=color, alpha=0.15)
ax.axhline(22, ls="--", color="gray", lw=1, label="política aleatoria (~22)")
ax.set_xlabel("Episodio")
ax.set_ylabel("Recompensa (media móvil 50)")
ax.set_title("Curva de aprendizaje en CartPole-v1")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig1_learning_curves.png", dpi=300)
plt.close(fig)
print("[ok] fig1_learning_curves.png")


# ==================================================================
# FIGURA 2 — Histograma de retornos (últimos 200 eps)
# ==================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for ax, (label, log) in zip(axes.ravel(), data.items()):
    color = RUNS[label][1]
    r = np.asarray(log["reward"], dtype=float)[-200:]
    ax.hist(r, bins=25, color=color, alpha=0.75, edgecolor="black", lw=0.4)
    ax.axvline(np.mean(r), color="black", ls="--", lw=1,
               label=f"media={np.mean(r):.1f}")
    ax.axvline(22, color="gray", ls=":", lw=1, label="aleatorio")
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
for ax in axes[-1]:
    ax.set_xlabel("Retorno del episodio")
for ax in axes[:, 0]:
    ax.set_ylabel("Frecuencia")
fig.suptitle("Distribución de retornos — últimos 200 episodios",
             fontsize=12, y=1.00)
fig.tight_layout()
fig.savefig(FIG / "fig2_return_histograms.png", dpi=300)
plt.close(fig)
print("[ok] fig2_return_histograms.png")


# ==================================================================
# FIGURA 3 — Heatmap de la matriz X final (corrida A)
# ==================================================================
xfile = find_X_file()
if xfile is not None:
    X = np.load(xfile)
    fig, ax = plt.subplots(figsize=(5, 7))
    im = ax.imshow(X, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Columna (acción)")
    ax.set_ylabel("Fila (sensor)")
    ax.set_title(f"Matriz X final — {xfile.name}\nmean={X.mean():.3f}  std={X.std():.3f}")
    ax.set_xticks(range(X.shape[1]))
    ax.set_yticks(range(X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            ax.text(j, i, f"{X[i,j]:.2f}", ha="center", va="center",
                    color="white" if X[i, j] < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="X ∈ [0, 1]")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_X_heatmap_runA.png", dpi=300)
    plt.close(fig)
    print("[ok] fig3_X_heatmap_runA.png")
else:
    print("[warn] no se encontró .npy de X; se omite fig3")


# ==================================================================
# FIGURA 4 — P(a=1 | ángulo del poste)
# ==================================================================
def plot_action_vs_angle(ax, log, label, color,
                         n_bins=12, angle_max=0.22,
                         n_last_eps=200):
    if "angles_per_ep" not in log or "actions_per_ep" not in log:
        ax.text(0.5, 0.5, f"{label}\n(sin datos de ángulo)",
                ha="center", va="center", transform=ax.transAxes)
        return

    # Aplanar últimos N episodios
    angles, actions = [], []
    for a_ep, ac_ep in zip(log["angles_per_ep"][-n_last_eps:],
                           log["actions_per_ep"][-n_last_eps:]):
        angles.extend(a_ep)
        actions.extend(ac_ep)
    angles = np.asarray(angles, dtype=float)
    actions = np.asarray(actions, dtype=int)

    # Bins uniformes en [-angle_max, angle_max]
    edges = np.linspace(-angle_max, angle_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    p1 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (angles >= edges[i]) & (angles < edges[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 5:
            p1[i] = float(np.mean(actions[mask] == 1))

    ax.axhline(0.5, color="gray", ls="--", lw=1, label="sin política (0.5)")
    ax.plot(centers, p1, "o-", color=color, lw=1.8, ms=5)
    ax.set_xlabel("Ángulo del poste (rad)")
    ax.set_ylabel("P(a=1 | ángulo)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{label}\n(últimos {n_last_eps} eps)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

# --- Panel 4: A vs D ---
labels_4b = list(data.keys())
if labels_4b:
    fig, axes = plt.subplots(1, len(labels_4b),
                             figsize=(4 * len(labels_4b), 4),
                             squeeze=False)
    for ax, label in zip(axes[0], labels_4b):
        plot_action_vs_angle(ax, data[label], label, RUNS[label][1])
    fig.tight_layout()
    fig.savefig(FIG / "fig4_action_vs_angle.png", dpi=300)
    plt.close(fig)
    print("[ok] fig4_action_vs_angle.png")

print(f"\n[listo] figuras en {FIG}")
