"""
validacion_iv_ltp_ltd.py
========================
Curva I-V con lazo de histéresis pinzada y respuesta R-V del memristor
durante la caracterización de plasticidad sináptica.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reconfigurar codificación de consola para Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurolab.devices import MemristorStrukov
from neurolab.configs import StrukovConfig


def simulate_iv_sweep(V_max=1.0, n_cycles=2, n_points_per_cycle=400, dt=1e-4):
    """
    Simula un barrido senoidal V(t) = V_max * sin(2*pi*f*t) para obtener la curva I-V.
    """
    cfg = StrukovConfig(
        RON=100.0, ROFF=16_000.0, x0=0.10,
        D=10e-9, mu_v=1e-14, clip_x=True
    )
    mem = MemristorStrukov(cfg)

    t = np.linspace(0, n_cycles, n_cycles * n_points_per_cycle)
    V = V_max * np.sin(2 * np.pi * t)
    I = np.zeros_like(V)
    R = np.zeros_like(V)
    G = np.zeros_like(V)

    for k, vk in enumerate(V):
        I[k] = mem.current(vk)
        R[k] = mem.resistance
        G[k] = 1.0 / R[k]
        mem.update(vk, dt)

    return V, I, R, G


def main():
    base_dir = os.path.dirname(__file__)
    png_path = os.path.join(base_dir, "validacion_iv_ltp_ltd.png")
    csv_path = os.path.join(base_dir, "validacion_iv_ltp_ltd.csv")

    V, I, R, G = simulate_iv_sweep()

    # Exportar CSV
    df = pd.DataFrame({"V_Volt": V, "I_mA": I * 1e3, "R_kOhm": R / 1e3, "G_uS": G * 1e6})
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Curva I-V (Histéresis Pinzada)
    axes[0].plot(V, I * 1e3, 'b-', lw=1.5, label='Lazo I-V')
    axes[0].axhline(0, color='k', ls='--', lw=0.6)
    axes[0].axvline(0, color='k', ls='--', lw=0.6)
    axes[0].set_xlabel('Voltaje Aplicado V (V)', fontsize=11)
    axes[0].set_ylabel('Corriente I (mA)', fontsize=11)
    axes[0].set_title('Curva I-V del Memristor (Histéresis Pinzada)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')

    # Curva R-V
    axes[1].plot(V, R / 1e3, 'r-', lw=1.5, label='Resistencia R(V)')
    axes[1].axvline(0, color='k', ls='--', lw=0.6)
    axes[1].set_xlabel('Voltaje Aplicado V (V)', fontsize=11)
    axes[1].set_ylabel('Resistencia R (kΩ)', fontsize=11)
    axes[1].set_title('Evolución de Resistencia Dinámica R vs. V', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    print("=" * 65)
    print("  CARACTERIZACIÓN I-V E HISTÉRESIS PINZADA (LTP/LTD)")
    print("=" * 65)
    print(f"I_max = {I.max()*1e3:.3f} mA | I_min = {I.min()*1e3:.3f} mA")
    print(f"R_min = {R.min()/1e3:.2f} kΩ | R_max = {R.max()/1e3:.2f} kΩ")
    print(f"Gráfica guardada en: {png_path}")
    print(f"Datos exportados a:  {csv_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()
