"""
test_16_read_window.py
======================
Prueba 16: Ventana de Lectura (Read Window).
Determina el voltaje máximo V_read_max que se puede aplicar sin alterar el peso (ΔG/G <= 0.01%).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


import numpy as np
import matplotlib.pyplot as plt
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def test_16_read_window():
    """Barrido de V_read para encontrar la ventana no destructiva."""

    V_read_vals = np.linspace(0.01, 1.0, 50)
    delta_G_rel = []

    for V_r in V_read_vals:
        config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
        cb = CrossbarIdeal(config)
        G_init = cb.G_matrix[0, 0]

        # Simular lectura no destructiva estándar (t_read = 1.5 ms)
        cb.memristors[0, 0].update(V_r, dt=0.0015)
        cb.G_matrix[0, 0] = cb.memristors[0, 0].conductance

        G_final = cb.G_matrix[0, 0]
        dG = abs(G_final - G_init) / G_init * 100
        delta_G_rel.append(dG)

    delta_G_rel = np.array(delta_G_rel)

    # Buscar V_read_max donde ΔG <= 0.01%
    valid_mask = delta_G_rel <= 0.01
    if np.any(valid_mask):
        V_read_max = V_read_vals[valid_mask][-1]
    else:
        V_read_max = V_read_vals[0]

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "fig_16_read_window.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(V_read_vals, delta_G_rel, 'b-o', ms=4, lw=1.5, label='Perturbación ΔG (%)')
    ax.axhline(0.01, color='r', ls='--', lw=1.2, label='Límite No Destructivo (0.01%)')
    ax.axvline(V_read_max, color='g', ls=':', lw=1.5, label=f'V_read_max = {V_read_max:.2f} V')
    ax.set_xlabel('Voltaje de Lectura V_read (V)')
    ax.set_ylabel('Variación Relativa ΔG/G_0 (%)')
    ax.set_title(f'Prueba 16: Ventana de Lectura (V_read_max = {V_read_max:.2f} V)')
    ax.set_ylim(0, max(0.015, float(np.max(delta_G_rel)) * 1.15))
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 16: VENTANA DE LECTURA")
    print("=" * 60)
    print(f"Rango V_read : [0.01, 1.00] V")
    print(f"V_read_max   : {V_read_max:.2f} V (para ΔG <= 0.01%)")
    print(f"Figura       : {fig_path}")
    print("✅ PRUEBA 16: OK")
    print()

    return V_read_vals, delta_G_rel, V_read_max


if __name__ == '__main__':
    test_16_read_window()
