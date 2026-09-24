"""
test_20_endurance.py
====================
Prueba 20: Endurance y Ciclado Continuo (1000 Ciclos LTP/LTD).
Evalúa la estabilidad de la ventana de memoria (G_max - G_min) a lo largo de 1000 ciclos de programación continua.
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
from neurolab.synapses import MemristiveSynapse, LTPRule, LTDRule


def test_20_endurance():
    """Prueba de Endurance a lo largo de 1000 ciclos continuos LTP -> LTD."""

    config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
    cb = CrossbarIdeal(config)
    syn = MemristiveSynapse(cb.memristors[0, 0])

    N_cycles = 1000
    G_max_history = []
    G_min_history = []

    ltp = LTPRule(n_pulses=10, V_pulse=+1.0)
    ltd = LTDRule(n_pulses=10, V_pulse=-1.0)

    for c in range(N_cycles):
        # Fase LTP: 10 pulsos
        ltp.apply(syn, dt=1e-3)
        G_max_history.append(syn.conductance)

        # Fase LTD: 10 pulsos
        ltd.apply(syn, dt=1e-3)
        G_min_history.append(syn.conductance)

    G_max_history = np.array(G_max_history) * 1e6
    G_min_history = np.array(G_min_history) * 1e6

    window_initial = G_max_history[0] - G_min_history[0]
    window_final = G_max_history[-1] - G_min_history[-1]
    retention_ratio = window_final / window_initial * 100

    assert retention_ratio > 95.0, f"Retención de ventana de memoria baja: {retention_ratio:.2f}%"

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "fig_20_endurance.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, N_cycles + 1), G_max_history, 'b-', lw=1.2, label='G_max (Post-LTP)')
    ax.plot(range(1, N_cycles + 1), G_min_history, 'r-', lw=1.2, label='G_min (Post-LTD)')
    ax.fill_between(range(1, N_cycles + 1), G_min_history, G_max_history, color='#bd93f9', alpha=0.2, label='Ventana de Memoria')
    ax.set_xlabel('Número de Ciclo (LTP → LTD)')
    ax.set_ylabel('Conductancia G_11 (μS)')
    ax.set_title(f'Prueba 20: Endurance (1000 Ciclos) — Retención de Ventana = {retention_ratio:.2f}%')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 20: ENDURANCE (CICLADO CONTINUO)")
    print("=" * 60)
    print(f"N ciclos     : {N_cycles}")
    print(f"Ventana ini  : {window_initial:.4f} μS")
    print(f"Ventana fin  : {window_final:.4f} μS")
    print(f"Retención MW : {retention_ratio:.2f} %")
    print(f"Figura       : {fig_path}")
    print("✅ PRUEBA 20: OK")
    print()

    return G_max_history, G_min_history, retention_ratio


if __name__ == '__main__':
    test_20_endurance()
