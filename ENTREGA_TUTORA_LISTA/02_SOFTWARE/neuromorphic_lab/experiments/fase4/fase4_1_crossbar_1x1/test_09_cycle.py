"""
test_09_cycle.py
================
Prueba 9: Ciclo reversible LTP → LTD → LTP.
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


def test_09_cycle():
    """Ciclo completo: 50 LTP, 50 LTD, 50 LTP."""

    config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
    cb = CrossbarIdeal(config)
    syn = MemristiveSynapse(cb.memristors[0, 0])

    G_inicial = syn.conductance
    G_history = [G_inicial]
    pulsos = [0]

    # Fase 1: LTP (50 pulsos)
    ltp = LTPRule(n_pulses=50, V_pulse=+1.0)
    for _ in range(50):
        syn.update(pre_spike=True, post_spike=False,
                   dt=1e-3, V_applied=+1.0)
        G_history.append(syn.conductance)
        pulsos.append(len(pulsos))

    G_max = G_history[-1]

    # Fase 2: LTD (50 pulsos)
    ltd = LTDRule(n_pulses=50, V_pulse=-1.0)
    for _ in range(50):
        syn.update(pre_spike=False, post_spike=True,
                   dt=1e-3, V_applied=-1.0)
        G_history.append(syn.conductance)
        pulsos.append(len(pulsos))

    G_min = G_history[-1]

    # Fase 3: LTP otra vez (50 pulsos)
    for _ in range(50):
        syn.update(pre_spike=True, post_spike=False,
                   dt=1e-3, V_applied=+1.0)
        G_history.append(syn.conductance)
        pulsos.append(len(pulsos))

    G_final = G_history[-1]

    G_history = np.array(G_history)

    # Verificar reversibilidad
    error_retorno = abs(G_final - G_max) / G_max * 100
    assert error_retorno < 1.0, f"Error retorno debe ser < 1%, es {error_retorno}"

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)

    # Figura
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pulsos[:51], G_history[:51] * 1e6, 'b-o', ms=3,
            label='LTP #1 (+1V)')
    ax.plot(pulsos[50:101], G_history[50:101] * 1e6, 'r-s', ms=3,
            label='LTD (−1V)')
    ax.plot(pulsos[100:], G_history[100:] * 1e6, 'g-^', ms=3,
            label='LTP #2 (+1V)')
    ax.axvline(50, color='gray', ls='--', alpha=0.5)
    ax.axvline(100, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Número de pulsos')
    ax.set_ylabel('G_11 (μS)')
    ax.set_title(f'Prueba 9: Ciclo reversible (error retorno = {error_retorno:.4f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'fig_09_cycle.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 9: CICLO LTP → LTD → LTP")
    print("=" * 60)
    print(f"G inicial     : {G_inicial*1e6:.6f} μS")
    print(f"G max (LTP#1) : {G_max*1e6:.6f} μS")
    print(f"G min (LTD)   : {G_min*1e6:.6f} μS")
    print(f"G final (LTP#2): {G_final*1e6:.6f} μS")
    print(f"Error retorno : {error_retorno:.6f} %")
    print(f"Figura        : {fig_path}")
    print("✅ PRUEBA 9: OK")
    print()

    return pulsos, G_history


if __name__ == '__main__':
    test_09_cycle()
