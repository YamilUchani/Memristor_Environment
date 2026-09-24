"""
validacion_vi_ltp_ltd.py
========================
Evidencia física completa de voltaje, corriente, resistencia y conductancia
durante pulsos LTP (+1V) y LTD (-1V).
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


def simulate_pulses(V_pulse, n_pulses=40, pulse_width=1e-3, dt=1e-4):
    """
    Simula n_pulses de voltaje V_pulse sobre un memristor Strukov.
    Registra V, I, R, G en cada instante.
    """
    cfg = StrukovConfig(
        RON=100.0, ROFF=16_000.0, x0=0.10,
        D=10e-9, mu_v=1e-14, clip_x=True
    )
    mem = MemristorStrukov(cfg)

    # Tiempo total
    T_total = n_pulses * pulse_width
    t = np.arange(0, T_total, dt)

    V = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    G = np.zeros_like(t)

    for k, tk in enumerate(t):
        phase = (tk % pulse_width) / pulse_width
        V[k] = V_pulse if phase < 0.5 else 0.0

        I[k] = mem.current(V[k])
        R[k] = mem.resistance
        G[k] = 1.0 / R[k]

        mem.update(V[k], dt)

    return t, V, I, R, G


def main():
    base_dir = os.path.dirname(__file__)
    png_path = os.path.join(base_dir, "validacion_vi_ltp_ltd.png")
    csv_path = os.path.join(base_dir, "validacion_vi_ltp_ltd.csv")

    # --- Simular LTP y LTD ---
    t_ltp, V_ltp, I_ltp, R_ltp, G_ltp = simulate_pulses(+1.0)
    t_ltd, V_ltd, I_ltd, R_ltd, G_ltd = simulate_pulses(-1.0)

    # Exportar CSV
    df = pd.DataFrame({
        "t_ms": t_ltp * 1e3,
        "V_ltp_V": V_ltp, "I_ltp_uA": I_ltp * 1e6, "R_ltp_kOhm": R_ltp / 1e3, "G_ltp_uS": G_ltp * 1e6,
        "V_ltd_V": V_ltd, "I_ltd_uA": I_ltd * 1e6, "R_ltd_kOhm": R_ltd / 1e3, "G_ltd_uS": G_ltd * 1e6,
    })
    df.to_csv(csv_path, index=False)

    # --- Figura 4 paneles ---
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    # Panel 1: Voltaje
    axes[0].plot(t_ltp * 1e3, V_ltp, 'b-', lw=1.2, label='LTP (+1 V)')
    axes[0].plot(t_ltd * 1e3, V_ltd, 'r-', lw=1.2, label='LTD (−1 V)')
    axes[0].set_ylabel('V (V)', fontsize=10)
    axes[0].set_title('Voltaje Aplicado V(t) — Pulsos Cuadrados (Duty Cycle 50%)', fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Corriente
    axes[1].plot(t_ltp * 1e3, I_ltp * 1e6, 'b-', lw=1.2, label='LTP')
    axes[1].plot(t_ltd * 1e3, I_ltd * 1e6, 'r-', lw=1.2, label='LTD')
    axes[1].set_ylabel('I (μA)', fontsize=10)
    axes[1].set_title('Corriente Resultante I(t) = V(t) / R(t)', fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Resistencia
    axes[2].plot(t_ltp * 1e3, R_ltp / 1e3, 'b-', lw=1.2, label='LTP (R decrece)')
    axes[2].plot(t_ltd * 1e3, R_ltd / 1e3, 'r-', lw=1.2, label='LTD (R crece)')
    axes[2].set_ylabel('R (kΩ)', fontsize=10)
    axes[2].set_title('Resistencia Dinámica R(t)', fontsize=11, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Conductancia
    axes[3].plot(t_ltp * 1e3, G_ltp * 1e6, 'b-', lw=1.2, label='LTP (G crece)')
    axes[3].plot(t_ltd * 1e3, G_ltd * 1e6, 'r-', lw=1.2, label='LTD (G decrece)')
    axes[3].set_ylabel('G (μS)', fontsize=10)
    axes[3].set_xlabel('Tiempo (ms)', fontsize=11)
    axes[3].set_title('Conductancia Sináptica G(t) = 1 / R(t)', fontsize=11, fontweight='bold')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    # --- Métricas ---
    print("=" * 65)
    print("  EVIDENCIA FÍSICA CONSOLIDADA: V - I - R - G (LTP vs LTD)")
    print("=" * 65)
    print(f"LTP (+1V): I_mean = {I_ltp[I_ltp > 0].mean()*1e6:.2f} μA | "
          f"R_ini = {R_ltp[0]/1e3:.2f} kΩ -> R_fin = {R_ltp[-1]/1e3:.2f} kΩ | "
          f"G_ini = {G_ltp[0]*1e6:.2f} μS -> G_fin = {G_ltp[-1]*1e6:.2f} μS")
    print(f"LTD (-1V): I_mean = {I_ltd[I_ltd < 0].mean()*1e6:.2f} μA | "
          f"R_ini = {R_ltd[0]/1e3:.2f} kΩ -> R_fin = {R_ltd[-1]/1e3:.2f} kΩ | "
          f"G_ini = {G_ltd[0]*1e6:.2f} μS -> G_fin = {G_ltd[-1]*1e6:.2f} μS")
    print(f"Gráfica guardada en: {png_path}")
    print(f"Datos exportados a:  {csv_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()
