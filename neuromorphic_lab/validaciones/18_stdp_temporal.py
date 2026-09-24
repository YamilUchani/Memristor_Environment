"""
validacion_stdp_temporal.py
===========================
Muestra la evolución temporal del peso sináptico W durante
una secuencia continua de pares de spikes con Δt variable (fase LTP y fase LTD).
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
from neurolab.synapses import MemristiveSynapse, STDPRule


def main():
    base_dir = os.path.dirname(__file__)
    png_path = os.path.join(base_dir, "validacion_stdp_temporal.png")
    csv_path = os.path.join(base_dir, "validacion_stdp_temporal.csv")

    rule = STDPRule(
        A_plus=0.05, A_minus=-0.025,
        tau_plus=17e-3, tau_minus=34e-3
    )

    # Secuencia de Δt: 20 pares LTP (Δt = +5 a +40 ms), luego 20 pares LTD (Δt = -5 a -40 ms)
    dt_ltp = np.linspace(0.005, 0.040, 20)
    dt_ltd = np.linspace(-0.005, -0.040, 20)
    dt_sequence = np.concatenate([dt_ltp, dt_ltd])

    cfg = StrukovConfig(RON=100.0, ROFF=16_000.0, x0=0.50)
    mem = MemristorStrukov(cfg)
    syn = MemristiveSynapse(mem)

    W_history = [syn.weight]
    G_history = [syn.conductance]

    for dt_val in dt_sequence:
        rule.apply(syn, dt_val)
        W_history.append(syn.weight)
        G_history.append(syn.conductance)

    W_history = np.array(W_history)
    G_history = np.array(G_history)
    pair_indices = np.arange(len(W_history))

    # Exportar CSV
    df = pd.DataFrame({
        "par_index": pair_indices,
        "delta_t_ms": np.concatenate([[0.0], dt_sequence * 1e3]),
        "peso_W": W_history,
        "conductancia_uS": G_history * 1e6
    })
    df.to_csv(csv_path, index=False)

    # --- Figura ---
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

    # Panel 1: Δt de cada par de spikes
    markerline, stemlines, baseline = axes[0].stem(
        range(1, len(dt_sequence) + 1), dt_sequence * 1e3,
        basefmt='k-', linefmt='C0-', markerfmt='C0o'
    )
    plt.setp(markerline, markersize=5)
    axes[0].axhline(0, color='k', ls='--', lw=0.8)
    axes[0].axvline(20.5, color='gray', ls='--', alpha=0.7, label='Transición LTP → LTD')
    axes[0].set_ylabel('Δt = t_post − t_pre (ms)', fontsize=10)
    axes[0].set_title('Secuencia de Pares de Spikes (Entrada Estimuladora)', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Panel 2: Evolución de W(t)
    axes[1].plot(pair_indices, W_history, 'g-o', ms=4, lw=1.5, label='Peso Sináptico W(t)')
    axes[1].axhline(W_history[0], color='k', ls='--', lw=0.8, label=f'W inicial ({W_history[0]:.2f})')
    axes[1].axvline(20, color='gray', ls='--', alpha=0.7)
    axes[1].set_xlabel('Número de Par de Spikes Consecutivos', fontsize=11)
    axes[1].set_ylabel('Peso Sináptico W ∈ [0, 1]', fontsize=10)
    axes[1].set_title('Evolución Temporal Emergente de W(t) (STDP Hebbiano)', fontsize=11, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    print("=" * 65)
    print("  EVOLUCIÓN TEMPORAL DEL PESO W(t) (STDP HEBBIANO)")
    print("=" * 65)
    print(f"W inicial (par 0)  = {W_history[0]:.4f}")
    print(f"W máximo  (par 20) = {W_history[20]:.4f} (tras fase LTP)")
    print(f"W final   (par 40) = {W_history[-1]:.4f} (tras fase LTD)")
    print(f"Gráfica guardada en: {png_path}")
    print(f"Datos exportados a:  {csv_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()
