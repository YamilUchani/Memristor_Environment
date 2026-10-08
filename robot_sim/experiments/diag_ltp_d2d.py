"""
Diagnóstico específico de LTP/LTD y D2D.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurobot.crossbar_brain import CrossbarBrain

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True, parents=True)


# ==================================================================
# PARTE 1: barrido de voltaje para LTD
# ==================================================================
print("=" * 60)
print("PARTE 1: Barrido de voltaje para LTD y LTP")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
results_ltp_ltd = {}

# Subplot 1: LTD con distintos voltajes (partiendo de x=0.5)
ax = axes[0]
V_ltd_values = [-0.1, -0.3, -0.5, -0.7, -1.0]
for V_ltd in V_ltd_values:
    b = CrossbarBrain(N=1, M=1, R_sense=1e-3, seed=0)
    b.X = np.full((1, 1), 0.5)
    cell = b.cells[0][0]
    X_seq = [0.5]
    for _ in range(50):
        cell.update(voltage=V_ltd, dt=1e-3)
        x = float(getattr(cell, "x", X_seq[-1]))
        X_seq.append(x)
    ax.plot(X_seq, "o-", ms=3, label=f"V = {V_ltd:.1f} V")
    results_ltp_ltd[f"LTD_{V_ltd}"] = X_seq

ax.set_xlabel("Pulso")
ax.set_ylabel("Estado x")
ax.set_title("LTD con distintos voltajes (x0 = 0.5)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_ylim(-0.05, 1.05)

# Subplot 2: LTP con distintos voltajes (partiendo de x=0.5)
ax = axes[1]
V_ltp_values = [0.3, 0.5, 0.7, 1.0]
for V_ltp in V_ltp_values:
    b = CrossbarBrain(N=1, M=1, R_sense=1e-3, seed=0)
    b.X = np.full((1, 1), 0.5)
    cell = b.cells[0][0]
    X_seq = [0.5]
    for _ in range(50):
        cell.update(voltage=V_ltp, dt=1e-3)
        x = float(getattr(cell, "x", X_seq[-1]))
        X_seq.append(x)
    ax.plot(X_seq, "o-", ms=3, label=f"V = {V_ltp:.1f} V")
    results_ltp_ltd[f"LTP_{V_ltp}"] = X_seq

ax.set_xlabel("Pulso")
ax.set_ylabel("Estado x")
ax.set_title("LTP con distintos voltajes (x0 = 0.5)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(FIG / "diag_ltp_ltd_voltage_sweep.png", dpi=200)
plt.close()
print("[fig] diag_ltp_ltd_voltage_sweep.png")

# Tabla resumen
print("\nResumen LTP/LTD por voltaje:")
print(f"{'Operacion':>12} | {'x tras 1 pulso':>15} | {'x tras 5 pulsos':>15} | {'x tras 50 pulsos':>16}")
print("-" * 65)
for key, seq in results_ltp_ltd.items():
    if len(seq) >= 51:
        print(f"{key:>12} | {seq[1]:>15.4f} | {seq[5]:>15.4f} | {seq[50]:>16.4f}")


# ==================================================================
# PARTE 2: D2D - verificar varianza real
# ==================================================================
print("\n" + "=" * 60)
print("PARTE 2: D2D - verificar varianza real")
print("=" * 60)

N, M = 8, 8
n_crossbars = 20
V = np.full(N, 0.1)

I_cols = []
X_means = []
X_stds = []

for k in range(n_crossbars):
    b = CrossbarBrain(N=N, M=M, R_sense=1e-3, seed=k)
    b.X = np.random.default_rng(k).uniform(0.3, 0.7, (N, M))
    X_means.append(b.X.mean())
    X_stds.append(b.X.std())
    I_col, _ = b.read(V)
    I_cols.append(I_col)

X_means = np.array(X_means)
X_stds = np.array(X_stds)
I_cols = np.array(I_cols)

cv_I = (I_cols.std(axis=0) / (np.abs(I_cols.mean(axis=0)) + 1e-30)).mean() * 100
print(f"X mean entre crossbars:    media={X_means.mean():.4f}  std={X_means.std():.4f}  CV={X_means.std()/X_means.mean()*100:.2f}%")
print(f"X std  entre crossbars:    media={X_stds.mean():.4f}  std={X_stds.std():.4f}")
print(f"I_col media por columna:   mean={I_cols.mean():.4e} A  std_entre_crossbars={I_cols.std(axis=0).mean():.4e} A")
print(f"I_col CV entre crossbars:  {cv_I:.4f}%")

# Test con X extremos para ver si el solver responde
print("\nPrueba con X extremos (un solo crossbar):")
b = CrossbarBrain(N=N, M=M, R_sense=1e-3, seed=0)
for x_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
    b.X = np.full((N, M), x_val)
    I_col, _ = b.read(V)
    print(f"  X={x_val:.1f}  I_col_mean={I_col.mean()*1e6:.2f} uA  G_mean={b.G.mean()*1e3:.4f} mS")

# Guardar npz
np.savez(FIG / "diag_d2d.npz",
         X_means=X_means, X_stds=X_stds,
         I_cols=I_cols)
print("\n[ok] diagnóstico completado con éxito")
