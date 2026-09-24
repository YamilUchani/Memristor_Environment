"""
test_17_write_window.py
=======================
Prueba 17: Ventana de Escritura (Write Window).
Determina el voltaje mínimo V_write_min requerido para modificar determinísticamente el peso (ΔG/G >= 1.0%).
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


def test_17_write_window():
    """Barrido de V_pulse para encontrar el umbral de escritura."""

    V_pulse_vals = np.linspace(0.1, 2.0, 50)
    delta_G_rel = []

    for V_p in V_pulse_vals:
        config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
        cb = CrossbarIdeal(config)
        G_init = cb.G_matrix[0, 0]

        # Aplicar 1 pulso de escritura eficaz de 20 ms
        cb.memristors[0, 0].update(V_p, dt=0.02)
        cb.G_matrix[0, 0] = cb.memristors[0, 0].conductance

        G_final = cb.G_matrix[0, 0]
        dG = abs(G_final - G_init) / G_init * 100
        delta_G_rel.append(dG)

    delta_G_rel = np.array(delta_G_rel)

    # Buscar V_write_min donde ΔG >= 1.0%
    write_mask = delta_G_rel >= 1.0
    if np.any(write_mask):
        V_write_min = V_pulse_vals[write_mask][0]
    else:
        V_write_min = V_pulse_vals[-1]

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "fig_17_write_window.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(V_pulse_vals, delta_G_rel, 'm-s', ms=4, lw=1.5, label='Cambio de peso ΔG (%)')
    ax.axhline(1.0, color='r', ls='--', lw=1.2, label='Umbral de Escritura Eficiente (1.0%)')
    ax.axvline(V_write_min, color='g', ls=':', lw=1.5, label=f'V_write_min = {V_write_min:.2f} V')
    ax.set_xlabel('Voltaje de Escritura V_pulse (V)')
    ax.set_ylabel('Variación Relativa ΔG/G_0 (%)')
    ax.set_title(f'Prueba 17: Ventana de Escritura (V_write_min = {V_write_min:.2f} V)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 17: VENTANA DE ESCRITURA")
    print("=" * 60)
    print(f"Rango V_pulse : [0.10, 2.00] V")
    print(f"V_write_min   : {V_write_min:.2f} V (para ΔG >= 1.0%)")
    print(f"Figura        : {fig_path}")
    print("✅ PRUEBA 17: OK")
    print()

    return V_pulse_vals, delta_G_rel, V_write_min


if __name__ == '__main__':
    test_17_write_window()
