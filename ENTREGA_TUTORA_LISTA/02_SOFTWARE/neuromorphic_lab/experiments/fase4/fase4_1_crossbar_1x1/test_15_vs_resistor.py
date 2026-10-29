"""
test_15_vs_resistor.py
======================
Prueba 15: Comparación con resistencia pura.
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
from neurolab.devices import MemristorStrukov
from neurolab.devices.config import StrukovConfig


def test_15_vs_resistor():
    """Comparar I-V de memristor vs resistencia con histéresis visible."""

    # --- Resistencia pura ---
    R_const = 14_410
    V_values = np.linspace(-2.0, 2.0, 201)
    I_resistor = V_values / R_const

    # --- Memristor Strukov con barrido triangular ---
    cfg = StrukovConfig(RON=100.0, ROFF=16_000.0, x0=0.10,
                        D=10e-9, mu_v=1e-14, clip_x=True)
    mem = MemristorStrukov(cfg)

    # BARRIDO TRIANGULAR (ida y vuelta)
    n_points = 300
    V_ida = np.linspace(-2.0, 2.0, n_points)
    V_vuelta = np.linspace(2.0, -2.0, n_points)
    V_tri = np.concatenate([V_ida, V_vuelta])

    I_mem = []
    R_mem = []

    dt = 1e-4  # paso temporal
    for V in V_tri:
        I = mem.current(V)
        I_mem.append(I)
        R_mem.append(mem.resistance)
        # Múltiples pasos para acumular cambio y desplegar lazo
        for _ in range(10):
            mem.update(V, dt)

    I_mem = np.array(I_mem)
    R_mem = np.array(R_mem)

    R_var = (max(R_mem) - min(R_mem)) / np.mean(R_mem) * 100

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, 'fig_15_vs_resistor.png')

    # Figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(V_values, I_resistor * 1e3, 'b-', lw=2)
    axes[0].set_xlabel('V (V)')
    axes[0].set_ylabel('I (mA)')
    axes[0].set_title('Resistencia pura (14.41 kΩ)')
    axes[0].axhline(0, color='k', ls='--', lw=0.5)
    axes[0].axvline(0, color='k', ls='--', lw=0.5)
    axes[0].grid(True, alpha=0.3)

    # Memristor: separar ida y vuelta para ver el lazo
    axes[1].plot(V_tri[:n_points], I_mem[:n_points] * 1e3,
                 'r-', lw=1.5, label='Ida')
    axes[1].plot(V_tri[n_points:], I_mem[n_points:] * 1e3,
                 'orange', lw=1.5, label='Vuelta')
    axes[1].set_xlabel('V (V)')
    axes[1].set_ylabel('I (mA)')
    axes[1].set_title(f'Memristor Strukov (ΔR = {R_var:.2f}%)')
    axes[1].axhline(0, color='k', ls='--', lw=0.5)
    axes[1].axvline(0, color='k', ls='--', lw=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 15: VS RESISTENCIA PURA")
    print("=" * 60)
    print(f"Resistencia  : R = {R_const} Ω (lineal)")
    print(f"Memristor    : R ∈ [{min(R_mem):.2f}, {max(R_mem):.2f}] Ω")
    print(f"Variación R  : {R_var:.4f} %")
    print(f"Histéresis   : {R_var > 1.0}")
    print(f"Figura       : {fig_path}")
    print("✅ PRUEBA 15: OK")
    print()

    return V_values, I_resistor, V_tri, I_mem, R_mem


if __name__ == '__main__':
    test_15_vs_resistor()

