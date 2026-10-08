import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurobot.crossbar_brain import CrossbarBrain

# Verificación 1: G(x) directo desde la property
print("=== Verificación 1: G(x) pura ===")
b = CrossbarBrain(N=4, M=4, R_sense=1e-3, seed=0)
for x_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
    b.X = np.full((4, 4), x_val)
    G_now = b.G.mean()   # property, recalcula
    R_eq = 1.0 / G_now
    print(f"  X={x_val:.1f}  G_mean={G_now*1e3:.4f} mS  R_eq={R_eq:.2f} Ohm")

# Verificación 2: I_col vs X (con R_sense bajo)
print("\n=== Verificación 2: I_col responde a X ===")
b = CrossbarBrain(N=8, M=8, R_sense=1e-3, seed=0)
V = np.full(8, 0.1)
for x_val in [0.1, 0.5, 0.9]:
    b.X = np.full((8, 8), x_val)
    I_col, sneak = b.read(V)
    print(f"  X={x_val:.1f}  G_mean={b.G.mean()*1e3:.3f} mS  "
          f"I_col_mean={I_col.mean()*1e6:.2f} uA  sneak={sneak:.4f}")

# Verificación 3: los valores de electrical config
print("\n=== Verificación 3: config del lab ===")
print(f"  r_on  = {b.elec.r_on} Ohm")
print(f"  r_off = {b.elec.r_off} Ohm")
print(f"  Rango esperado de G: {1/b.elec.r_off*1e6:.2f} uS a {1/b.elec.r_on*1e6:.2f} uS")
print(f"  Factor dinámico: {b.elec.r_off/b.elec.r_on:.1f}x")
