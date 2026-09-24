"""
demo_parte_c.py
===============
Ejemplo numérico paso a paso del Crossbar 4×4 con esquema V/2 y variabilidad D2D.
Demuestra la independencia celda a celda y la selectividad de LA CRUZ al programar M22.
"""

import sys
import os
import numpy as np

# Asegurar path de importación de neurolab
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neurolab.crossbar import Crossbar, CrossbarConfig


def print_matrix(G, title=""):
    print(f"\n{title}")
    print("        " + "".join(f"  COL{j+1}   " for j in range(G.shape[1])))
    for i in range(G.shape[0]):
        row = f"FILA {i+1}: "
        for j in range(G.shape[1]):
            row += f"{G[i,j]:8.2f} "
        print(row)


def main():
    print("=" * 75)
    print(" PARTE C -- DEMOSTRACION NUMERICA PASO A PASO DEL CROSSBAR 4x4")
    print("=" * 75)

    # 1. Crear Crossbar 4x4 con variabilidad D2D propia por celda
    cb = Crossbar(CrossbarConfig(n_rows=4, n_cols=4, seed=42, enable_d2d=True, d2d_sigma=0.08))

    # --- Estado inicial ---
    G_before = cb.G_matrix * 1e6
    print_matrix(G_before, "1) ESTADO INICIAL (Conductancias independientes por D2D en uS)")

    # --- Programar M22 con esquema V/2 ---
    print("\n" + "-" * 75)
    print("-> Programando M22 con V/2 (V_program = 1.2V, dt = 1ms)...")
    print("   Celda Objetivo M22 (Fila 2, Columna 2): Full Pulse (+1.2V)")
    print("   Celdas de la Cruz (Fila 2 y Columna 2): Half Pulse (+0.6V)")
    print("   Celdas Restantes: 0.0V")
    print("-" * 75)

    cb.program_V2(i_target=1, j_target=1, V_program=1.2, dt=1e-3)

    G_after = cb.G_matrix * 1e6
    print_matrix(G_after, "2) DESPUES DE PROGRAMAR M22 (uS)")

    # --- Mostrar deltas ---
    delta = G_after - G_before
    print_matrix(delta, "3) MATRIZ delta G (Cambio neto de conductancia en uS)")

    # --- Verificar selectividad ---
    print("\n" + "=" * 75)
    print("   VERIFICACION NUMERICA DE SELECTIVIDAD Y LA CRUZ")
    print("=" * 75)

    delta_target = delta[1, 1]
    delta_half = np.array([
        delta[1, 0], delta[1, 2], delta[1, 3],  # misma fila (Fila 2)
        delta[0, 1], delta[2, 1], delta[3, 1],  # misma columna (Columna 2)
    ])
    delta_rest = np.array([
        delta[0, 0], delta[0, 2], delta[0, 3],
        delta[2, 0], delta[2, 2], delta[2, 3],
        delta[3, 0], delta[3, 2], delta[3, 3],
    ])

    print(f"  (*) dG objetivo (M22 Full V):      {delta_target:+.3f} uS")
    print(f"  (~) dG half-selected (Cruz mean):  {delta_half.mean():+.3f} uS")
    print(f"  (.) dG no afectadas (Resto mean):  {delta_rest.mean():+.3f} uS")
    ratio = delta_target / max(1e-9, delta_half.mean())
    print(f"  [*] Ratio objetivo/half:           {ratio:.1f}x")
    print(f"  [OK] Selectividad:                 {'EXCELENTE' if ratio > 5 else 'BAJA'}")
    print("=" * 75)


if __name__ == '__main__':
    main()
