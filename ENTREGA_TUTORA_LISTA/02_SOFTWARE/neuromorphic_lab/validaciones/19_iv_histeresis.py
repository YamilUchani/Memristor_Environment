"""
validacion_iv_ltp_ltd.py
========================
Curva I-V con lazo de histéresis pinzada en el origen.
Genera el lazo clásico en forma de "8" (Strukov).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Reconfigurar codificación de consola para Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurolab.devices import MemristorStrukov
from neurolab.configs import StrukovConfig

def simulate_iv_sweep():
    # Parámetros para un lazo "8" amplio y hermoso
    cfg = StrukovConfig(
        RON=10_000.0, ROFF=16_000.0, x0=0.50,
        D=10e-9, mu_v=2e-13, clip_x=True
    )
    mem = MemristorStrukov(cfg)

    # Onda senoidal, muy baja frecuencia (0.5 Hz) para máxima movilidad
    t = np.linspace(0, 2.0, 1000)
    actual_dt = t[1] - t[0]
    V = 1.0 * np.sin(2 * np.pi * 0.5 * t)
    
    I = np.zeros_like(V)

    for k, v in enumerate(V):
        I[k] = mem.current(v)
        mem.update(v, actual_dt)

    return V, I

def main():
    base_dir = os.path.dirname(__file__)
    png_path = os.path.join(base_dir, "validacion_iv_ltp_ltd.png")
    latex_png_path = os.path.abspath(os.path.join(base_dir, "..", "..", "docs", "latex", "figuras", "fig_19_iv_hysteresis.png"))

    V, I = simulate_iv_sweep()
    I_mA = I * 1e3

    fig, ax = plt.subplots(figsize=(7, 6))

    # Curva I-V (Histéresis Pinzada "8")
    ax.plot(V, I_mA, 'b-', lw=2.0, label='Lazo I-V Clásico')
    ax.axhline(0, color='k', ls='--', lw=0.6)
    ax.axvline(0, color='k', ls='--', lw=0.6)
    ax.set_xlabel('Voltaje Aplicado V (V)', fontsize=12)
    ax.set_ylabel('Corriente I (mA)', fontsize=12)
    ax.set_title('Curva I-V del Memristor (Histéresis Pinzada)', fontsize=14, fontweight='bold')
    
    # Limites simétricos
    lim_y = 1.1 * np.abs(I_mA).max()
    if lim_y == 0: lim_y = 1.0
    ax.set_ylim(-lim_y, lim_y)
    ax.set_xlim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(latex_png_path, dpi=300)
    plt.close()

    print("=" * 65)
    print("  CARACTERIZACIÓN I-V E HISTÉRESIS PINZADA ('8' CLÁSICO)")
    print("=" * 65)
    print(f"I_max = {I.max()*1e3:.3f} mA | I_min = {I.min()*1e3:.3f} mA")
    print("=" * 65)


if __name__ == '__main__':
    main()
