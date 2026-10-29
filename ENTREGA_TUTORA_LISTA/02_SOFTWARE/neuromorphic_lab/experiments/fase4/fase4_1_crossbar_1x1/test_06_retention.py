"""
test_06_retention.py
====================
Prueba 6: Persistencia de memoria.
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


def test_06_retention():
    """Verifica que el peso persiste sin voltaje."""

    config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
    cb = CrossbarIdeal(config)

    # Programar un peso específico
    G_programado = 200e-6
    cb.set_conductance(0, 0, G_programado)

    # Leer estado inicial
    G_inicial = cb.G_matrix[0, 0]

    # Simular 10 segundos SIN voltaje
    T_espera = 10.0
    dt = 1e-4
    t = np.arange(0, T_espera, dt)

    G_history = np.zeros_like(t)

    for k in range(len(t)):
        G_history[k] = cb.G_matrix[0, 0]
        cb.update_memristors(dt)  # V = 0 (no se aplica nada)

    G_final = cb.G_matrix[0, 0]
    drift = abs(G_final - G_inicial) / G_inicial * 100

    assert drift < 0.01, f"Drift debe ser < 0.01%, es {drift}"

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)

    # Figura
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, G_history * 1e6, 'b-', lw=2)
    ax.axhline(G_inicial * 1e6, color='r', ls='--', lw=1,
               label=f'G inicial = {G_inicial*1e6:.4f} μS')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('G_11 (μS)')
    ax.set_title(f'Prueba 6: Retención de peso (drift < {drift:.6f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'fig_06_retention.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 6: PERSISTENCIA (RETENCIÓN)")
    print("=" * 60)
    print(f"G programado  : {G_programado*1e6:.4f} μS")
    print(f"G inicial     : {G_inicial*1e6:.6f} μS")
    print(f"G final       : {G_final*1e6:.6f} μS")
    print(f"Duración      : {T_espera} s")
    print(f"Drift         : {drift:.8f} %")
    print(f"NO volátil    : {drift < 0.01}")
    print(f"Figura        : {fig_path}")
    print("✅ PRUEBA 6: OK")
    print()

    return t, G_history


if __name__ == '__main__':
    test_06_retention()
