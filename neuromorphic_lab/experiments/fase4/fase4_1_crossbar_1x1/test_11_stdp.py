"""
test_11_stdp.py
===============
Prueba 11: Integración con STDP.
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
from neurolab.synapses import MemristiveSynapse, STDPRule, AntiSTDPRule


def test_11_stdp():
    """Aplicar STDP al crossbar."""

    # --- Hebbiano ---
    config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
    cb = CrossbarIdeal(config)
    syn = MemristiveSynapse(cb.memristors[0, 0])
    stdp = STDPRule(A_plus=0.05, A_minus=-0.025,
                    tau_plus=17e-3, tau_minus=34e-3)

    dt_values = np.linspace(-0.08, 0.08, 41)
    dW_values = []

    for dt in dt_values:
        # Resetear sinapsis
        syn.reset()
        dW = stdp.apply(syn, dt)
        dW_values.append(dW)

    # --- Anti-Hebbiano ---
    cb_anti = CrossbarIdeal(config)
    syn_anti = MemristiveSynapse(cb_anti.memristors[0, 0])
    anti = AntiSTDPRule(A_plus=0.05, A_minus=-0.025,
                        tau_plus=17e-3, tau_minus=34e-3)

    dW_anti = []
    for dt in dt_values:
        syn_anti.reset()
        dW = anti.apply(syn_anti, dt)
        dW_anti.append(dW)

    dW_values = np.array(dW_values)
    dW_anti = np.array(dW_anti)

    # Verificar LTP para dt > 0 (Hebbiano)
    idx_pos = dt_values > 0
    assert np.all(dW_values[idx_pos] > 0), "LTP en dt > 0"

    # Verificar LTD para dt < 0 (Hebbiano)
    idx_neg = dt_values < 0
    assert np.all(dW_values[idx_neg] < 0), "LTD en dt < 0"

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)

    # Figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(dt_values * 1e3, dW_values, 'b-o', ms=4)
    axes[0].axhline(0, color='k', ls='--', lw=0.8)
    axes[0].axvline(0, color='k', ls='--', lw=0.8)
    axes[0].set_xlabel('Δt (ms)')
    axes[0].set_ylabel('ΔW')
    axes[0].set_title('Prueba 11: STDP Hebbiano')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dt_values * 1e3, dW_anti, 'r-s', ms=4)
    axes[1].axhline(0, color='k', ls='--', lw=0.8)
    axes[1].axvline(0, color='k', ls='--', lw=0.8)
    axes[1].set_xlabel('Δt (ms)')
    axes[1].set_ylabel('ΔW')
    axes[1].set_title('Prueba 11: STDP Anti-Hebbiano')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'fig_11_stdp.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 11: INTEGRACIÓN CON STDP")
    print("=" * 60)
    print(f"Hebbiano:")
    print(f"  ΔW max = {dW_values.max():+.6f} en Δt = {dt_values[np.argmax(dW_values)]*1e3:+.1f} ms")
    print(f"  ΔW min = {dW_values.min():+.6f} en Δt = {dt_values[np.argmin(dW_values)]*1e3:+.1f} ms")
    print(f"Anti-Hebbiano:")
    print(f"  ΔW max = {dW_anti.max():+.6f}")
    print(f"  ΔW min = {dW_anti.min():+.6f}")
    print(f"Figura    : {fig_path}")
    print("✅ PRUEBA 11: OK")
    print()

    return dt_values, dW_values, dW_anti


if __name__ == '__main__':
    test_11_stdp()
