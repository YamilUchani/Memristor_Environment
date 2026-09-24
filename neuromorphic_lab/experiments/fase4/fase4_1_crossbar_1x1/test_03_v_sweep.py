"""
test_03_v_sweep.py
==================
Prueba 3: Barrido de voltaje.
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


def test_03_voltage_sweep():
    """Barrido de voltaje ±100 mV."""

    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)
    cb.set_conductance(0, 0, 200e-6)

    V_values = np.linspace(-0.1, 0.1, 41)
    I_values = []
    I_expected = []

    for V in V_values:
        I = cb.read_single(V)
        I_values.append(I)
        I_expected.append(200e-6 * V)

    I_values = np.array(I_values)
    I_expected = np.array(I_expected)

    mae = np.mean(np.abs(I_values - I_expected))
    rmse = np.sqrt(np.mean((I_values - I_expected) ** 2))
    r2 = np.corrcoef(I_values, I_expected)[0, 1] ** 2

    assert r2 > 0.999999, f"R² debe ser ~1.0, es {r2}"
    assert mae < 1e-15, f"MAE debe ser < 1e-15, es {mae}"

    # Directorio de salida
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)

    # Figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(V_values, I_values * 1e6, 'b-o', ms=4, label='I_sim')
    axes[0].plot(V_values, I_expected * 1e6, 'r--', lw=1, label='I_ideal')
    axes[0].set_xlabel('V_in (V)')
    axes[0].set_ylabel('I_out (μA)')
    axes[0].set_title('Crossbar 1×1: Barrido de voltaje')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(V_values, (I_values - I_expected) * 1e15, 'g-s', ms=4)
    axes[1].set_xlabel('V_in (V)')
    axes[1].set_ylabel('Error (fA)')
    axes[1].set_title('Error absoluto')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'fig_03_v_sweep.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 3: BARRIDO DE VOLTAJE")
    print("=" * 60)
    print(f"V_in         : [−0.1, +0.1] V")
    print(f"N puntos     : {len(V_values)}")
    print(f"G_11         : 200 μS")
    print(f"MAE          : {mae:.2e} A")
    print(f"RMSE         : {rmse:.2e} A")
    print(f"R²           : {r2:.10f}")
    print(f"Figura       : {fig_path}")
    print("✅ PRUEBA 3: OK")
    print()

    return V_values, I_values, I_expected


if __name__ == '__main__':
    test_03_voltage_sweep()
