"""
validacion_ltp_ltd_ciclo.py
===========================
Validación de ciclo completo continuo de plasticidad LTP (50 pulsos) + LTD (50 pulsos).
Demuestra la reversibilidad del peso sináptico en el memristor.
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
from neurolab.synapses import MemristiveSynapse, LTPRule, LTDRule


def main():
    base_dir = os.path.dirname(__file__)
    png_path = os.path.join(base_dir, "validacion_ltp_ltd_ciclo.png")
    csv_path = os.path.join(base_dir, "validacion_ltp_ltd_ciclo.csv")

    cfg = StrukovConfig(RON=100.0, ROFF=16_000.0, x0=0.10, D=10e-9, mu_v=1e-14)
    mem = MemristorStrukov(cfg)
    syn = MemristiveSynapse(mem)

    G_history = [syn.conductance]

    # 50 pulsos LTP
    ltp = LTPRule(n_pulses=50, V_pulse=+1.0)
    for _ in range(50):
        syn.update(pre_spike=True, post_spike=False, dt=1e-3, V_applied=+1.0)
        G_history.append(syn.conductance)

    # 50 pulsos LTD consecutivos
    ltd = LTDRule(n_pulses=50, V_pulse=-1.0)
    for _ in range(50):
        syn.update(pre_spike=False, post_spike=True, dt=1e-3, V_applied=-1.0)
        G_history.append(syn.conductance)

    G_history = np.array(G_history)
    pulses = np.arange(len(G_history))

    # Save CSV
    df = pd.DataFrame({"pulso": pulses, "conductancia_S": G_history})
    df.to_csv(csv_path, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(pulses[:51], G_history[:51] * 1e6, 'b-o', ms=3, label='Fase LTP (Potenciación +1V)')
    ax.plot(pulses[50:], G_history[50:] * 1e6, 'r-s', ms=3, label='Fase LTD (Depresión -1V)')
    ax.axvline(50, color='gray', ls='--', alpha=0.7, label='Transición LTP → LTD')
    ax.set_xlabel('Número de pulsos consecutivos', fontsize=11)
    ax.set_ylabel('Conductancia Sináptica (μS)', fontsize=11)
    ax.set_title('Ciclo Completo Reversible LTP (50 pulsos) → LTD (50 pulsos)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("  RESULTADOS DE VALIDACIÓN: CICLO COMPLETO LTP + LTD")
    print("=" * 60)
    print(f"G_inicial (pulso 0)  = {G_history[0]*1e6:.2f} μS")
    print(f"G_max     (pulso 50) = {G_history[50]*1e6:.2f} μS (LTP max)")
    print(f"G_final   (pulso 100)= {G_history[-1]*1e6:.2f} μS (LTD final)")
    print(f"Gráfica guardada en: {png_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
