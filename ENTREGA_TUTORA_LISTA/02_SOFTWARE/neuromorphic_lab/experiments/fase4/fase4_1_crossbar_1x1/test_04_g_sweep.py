"""
test_04_g_sweep.py
==================
Prueba 4: Barrido de conductancia.
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


def test_04_conductance_sweep():
    """Barrido de conductancia de 62.5 a 500 μS."""

    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    G_values = np.linspace(62.5e-6, 500e-6, 41)
    V_in = 0.5
    I_values = []

    for G in G_values:
        cb.set_conductance(0, 0, G)
        I = cb.read_single(V_in)
        I_values.append(I)

    I_values = np.array(I_values)
    I_expected = G_values * V_in

    mae = np.mean(np.abs(I_values - I_expected))
    r2 = np.corrcoef(I_values, I_expected)[0, 1] ** 2

    assert r2 == 1.0
    assert mae < 1e-15

    # Directorio de salida
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)

    # Figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(G_values * 1e6, I_values * 1e6, 'b-o', ms=4)
    axes[0].plot(G_values * 1e6, I_expected * 1e6, 'r--', lw=1)
    axes[0].set_xlabel('G_11 (μS)')
    axes[0].set_ylabel('I_out (μA)')
    axes[0].set_title('Crossbar 1×1: Barrido de conductancia')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(G_values * 1e6, (I_values - I_expected) * 1e15, 'g-s', ms=4)
    axes[1].set_xlabel('G_11 (μS)')
    axes[1].set_ylabel('Error (fA)')
    axes[1].set_title('Error absoluto')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'fig_04_g_sweep.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 4: BARRIDO DE CONDUCTANCIA")
    print("=" * 60)
    print(f"G_11 rango   : [62.5, 500] μS")
    print(f"V_in         : 0.5 V")
    print(f"MAE          : {mae:.2e} A")
    print(f"R²           : {r2:.10f}")
    print(f"Figura       : {fig_path}")
    print("✅ PRUEBA 4: OK")
    print()

    return G_values, I_values, I_expected


if __name__ == '__main__':
    test_04_conductance_sweep()
