"""
test_08_ltd.py
==============
Prueba 8: Programación LTD.
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
from neurolab.synapses import MemristiveSynapse, LTDRule


def test_08_ltd():
    """Aplicar 40 pulsos −1 V y verificar que G baja."""

    config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
    cb = CrossbarIdeal(config)

    syn = MemristiveSynapse(cb.memristors[0, 0])
    G_inicial = syn.conductance

    # Aplicar LTD
    ltd = LTDRule(n_pulses=40, V_pulse=-1.0)
    G_history = ltd.apply(syn, dt=1e-3)

    G_final = G_history[-1]
    delta_G = (G_final - G_inicial) * 1e6

    assert G_final < G_inicial, "G debe bajar con LTD"

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)

    # Figura
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(G_history)), G_history * 1e6, 'r-s', ms=4, lw=1.5)
    ax.axhline(G_inicial * 1e6, color='gray', ls='--', lw=1,
               label=f'G inicial = {G_inicial*1e6:.4f} μS')
    ax.set_xlabel('Número de pulsos')
    ax.set_ylabel('G_11 (μS)')
    ax.set_title(f'Prueba 8: LTD (−1V, 40 pulsos) — ΔG = {delta_G:+.4f} μS')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'fig_08_ltd.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 8: PROGRAMACIÓN LTD")
    print("=" * 60)
    print(f"V_pulse       : −1.0 V")
    print(f"N pulsos      : 40")
    print(f"G inicial     : {G_inicial*1e6:.6f} μS")
    print(f"G final       : {G_final*1e6:.6f} μS")
    print(f"ΔG            : {delta_G:+.6f} μS")
    print(f"Figura        : {fig_path}")
    print("✅ PRUEBA 8: OK")
    print()

    return G_history


if __name__ == '__main__':
    test_08_ltd()
