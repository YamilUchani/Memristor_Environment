"""
test_05_non_destructive.py
==========================
Prueba 5: Lectura no destructiva.
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


def test_05_non_destructive():
    """Verifica que leer con V pequeño no modifica G."""

    config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
    cb = CrossbarIdeal(config)

    G_inicial = cb.G_matrix[0, 0]

    # Aplicar señal senoidal pequeña durante 1 segundo
    T = 1.0
    dt = 1e-4
    t = np.arange(0, T, dt)
    V_in = 0.1 * np.sin(2 * np.pi * 1.0 * t)

    I_values = np.zeros_like(t)
    G_values = np.zeros_like(t)

    for k, V in enumerate(V_in):
        I_values[k] = cb.read_single(V)
        G_values[k] = cb.memristors[0, 0].conductance
        cb.update_memristors(dt)

    G_final = cb.G_matrix[0, 0]

    # El memristor Strukov cambia con el voltaje
    # Verificamos que el cambio es pequeño (< 0.1%)
    cambio_rel = abs(G_final - G_inicial) / G_inicial * 100

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)

    # Figura
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t * 1e3, V_in * 1e3, 'b-', lw=1)
    axes[0].set_ylabel('V_in (mV)')
    axes[0].set_title('Prueba 5: Lectura no destructiva (±100 mV)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t * 1e3, I_values * 1e6, 'r-', lw=1)
    axes[1].set_ylabel('I_out (μA)')
    axes[1].set_title('Corriente resultante')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t * 1e3, G_values * 1e6, 'g-', lw=1)
    axes[2].set_xlabel('Tiempo (ms)')
    axes[2].set_ylabel('G_11 (μS)')
    axes[2].set_title(f'Conductancia (cambio < {cambio_rel:.4f}%)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'fig_05_non_destructive.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 5: LECTURA NO DESTRUCTIVA")
    print("=" * 60)
    print(f"V_in pico        : ±100 mV")
    print(f"Duración         : 1.0 s")
    print(f"G inicial        : {G_inicial*1e6:.6f} μS")
    print(f"G final          : {G_final*1e6:.6f} μS")
    print(f"Cambio relativo  : {cambio_rel:.6f} %")
    print(f"Cambio < 0.1%    : {cambio_rel < 0.1}")
    print(f"Figura           : {fig_path}")

    assert cambio_rel < 0.1, f"El cambio relativo debe ser < 0.1%, es {cambio_rel}%"

    print("✅ PRUEBA 5: OK")
    print()

    return t, V_in, I_values, G_values


if __name__ == '__main__':
    test_05_non_destructive()
