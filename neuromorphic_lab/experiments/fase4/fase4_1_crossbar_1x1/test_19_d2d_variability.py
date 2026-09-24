"""
test_19_d2d_variability.py
==========================
Prueba 19: Variabilidad Device-to-Device (D2D Monte Carlo).
Simula 100 crossbars 1×1 independientes con variaciones gaussianas del 5% en RON, ROFF y x0.
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


def test_19_d2d_variability():
    """Simulación Monte Carlo D2D de 100 memristores."""

    N_devices = 100
    sigma_var = 0.05  # 5% variabilidad

    np.random.seed(42)
    G_init_list = []
    G_prog_list = []

    for _ in range(N_devices):
        # Muestreo gaussiano de parámetros físicos
        ron_i = np.random.normal(100.0, 100.0 * sigma_var)
        roff_i = np.random.normal(16_000.0, 16_000.0 * sigma_var)
        x0_i = np.random.normal(0.10, 0.10 * sigma_var)

        cfg = StrukovConfig(RON=max(10.0, ron_i), ROFF=max(1000.0, roff_i), x0=max(0.01, x0_i))
        mem = MemristorStrukov(cfg)
        
        G_init = mem.conductance
        G_init_list.append(G_init)

        # Aplicar 20 pulsos de programación LTP (+1V)
        for _ in range(20):
            mem.update(+1.0, 1e-3)
        
        G_prog = mem.conductance
        G_prog_list.append(G_prog)

    G_init_list = np.array(G_init_list) * 1e6
    G_prog_list = np.array(G_prog_list) * 1e6

    mean_init = np.mean(G_init_list)
    std_init = np.std(G_init_list)
    cv_init = std_init / mean_init * 100

    mean_prog = np.mean(G_prog_list)
    std_prog = np.std(G_prog_list)
    cv_prog = std_prog / mean_prog * 100

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "fig_19_d2d_variability.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(G_init_list, bins=15, color='#4a9eff', edgecolor='black', alpha=0.7)
    axes[0].axvline(mean_init, color='r', ls='--', lw=1.5, label=f'μ = {mean_init:.2f} μS')
    axes[0].set_xlabel('Conductancia Inicial G_init (μS)')
    axes[0].set_ylabel('Frecuencia (Dispositivos)')
    axes[0].set_title(f'Distribución G_init (CV = {cv_init:.2f}%)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].hist(G_prog_list, bins=15, color='#50fa7b', edgecolor='black', alpha=0.7)
    axes[1].axvline(mean_prog, color='r', ls='--', lw=1.5, label=f'μ = {mean_prog:.2f} μS')
    axes[1].set_xlabel('Conductancia Programada G_prog (μS)')
    axes[1].set_ylabel('Frecuencia (Dispositivos)')
    axes[1].set_title(f'Distribución Post-LTP (CV = {cv_prog:.2f}%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 19: VARIABILIDAD D2D (MONTE CARLO)")
    print("=" * 60)
    print(f"N dispositivos: {N_devices}")
    print(f"G_init medio  : {mean_init:.2f} ± {std_init:.2f} μS (CV = {cv_init:.2f}%)")
    print(f"G_prog medio  : {mean_prog:.2f} ± {std_prog:.2f} μS (CV = {cv_prog:.2f}%)")
    print(f"Figura        : {fig_path}")
    print("✅ PRUEBA 19: OK")
    print()

    return G_init_list, G_prog_list, cv_init, cv_prog


if __name__ == '__main__':
    test_19_d2d_variability()
