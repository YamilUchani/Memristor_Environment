"""
validacion_ltp_ltd.py
=====================
Validación de curvas LTP (Long-Term Potentiation) y LTD (Long-Term Depression).
Script reproducible que genera curvas de potenciación/depresión y exporta CSV y PNG.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Añadir el directorio raíz de neurolab al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reconfigurar codificación de consola para Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from neurolab.devices import MemristorStrukov
from neurolab.configs import StrukovConfig
from neurolab.synapses import MemristiveSynapse, LTPRule, LTDRule


def main():
    base_dir = os.path.dirname(__file__)
    png_path = os.path.join(base_dir, "validacion_ltp_ltd.png")
    csv_path = os.path.join(base_dir, "validacion_ltp_ltd.csv")

    # --- Configuración del memristor ---
    cfg = StrukovConfig(
        RON=100.0, ROFF=16_000.0, x0=0.10,
        D=10e-9, mu_v=1e-14, clip_x=True
    )
    mem = MemristorStrukov(cfg)
    syn = MemristiveSynapse(mem)

    # --- LTP: 40 pulsos positivos (+1 V) ---
    ltp = LTPRule(n_pulses=40, V_pulse=+1.0)
    G_ltp = ltp.apply(syn, dt=1e-3)

    # Resetear sinapsis a estado inicial
    syn.reset()

    # --- LTD: 40 pulsos negativos (-1 V) ---
    ltd = LTDRule(n_pulses=40, V_pulse=-1.0)
    G_ltd = ltd.apply(syn, dt=1e-3)

    # --- Exportar CSV ---
    df_ltp = pd.DataFrame({'pulso': np.arange(len(G_ltp)), 'conductancia_S': G_ltp, 'tipo': 'LTP'})
    df_ltd = pd.DataFrame({'pulso': np.arange(len(G_ltd)), 'conductancia_S': G_ltd, 'tipo': 'LTD'})
    df_total = pd.concat([df_ltp, df_ltd], ignore_index=True)
    df_total.to_csv(csv_path, index=False)

    # --- Figura ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(range(len(G_ltp)), G_ltp * 1e6, 'b-o', ms=4, lw=1.5, label='LTP (+1V)')
    axes[0].set_xlabel('Número de pulsos', fontsize=11)
    axes[0].set_ylabel('Conductancia (μS)', fontsize=11)
    axes[0].set_title('LTP — Potenciación sináptica gradual', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(len(G_ltd)), G_ltd * 1e6, 'r-s', ms=4, lw=1.5, label='LTD (-1V)')
    axes[1].set_xlabel('Número de pulsos', fontsize=11)
    axes[1].set_ylabel('Conductancia (μS)', fontsize=11)
    axes[1].set_title('LTD — Depresión sináptica gradual', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    # --- Imprimir Métricas ---
    dG_ltp = (G_ltp[-1] - G_ltp[0]) * 1e6
    dG_ltd = (G_ltd[-1] - G_ltd[0]) * 1e6

    print("=" * 60)
    print("  RESULTADOS DE VALIDACIÓN: LTP / LTD")
    print("=" * 60)
    print(f"LTP: G_inicial = {G_ltp[0]*1e6:.2f} μS | G_final = {G_ltp[-1]*1e6:.2f} μS | ΔG = +{dG_ltp:.2f} μS")
    print(f"LTD: G_inicial = {G_ltd[0]*1e6:.2f} μS | G_final = {G_ltd[-1]*1e6:.2f} μS | ΔG = {dG_ltd:.2f} μS")
    print(f"Gráfica guardada en: {png_path}")
    print(f"Datos exportados a:  {csv_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
