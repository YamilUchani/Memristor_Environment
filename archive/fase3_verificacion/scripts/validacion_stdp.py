"""
validacion_stdp.py
==================
Validación de STDP (Spike-Timing-Dependent Plasticity).
Genera la ventana STDP ΔW vs. Δt para reglas Hebbiana y anti-Hebbiana,
exportando gráficos de alta resolución y archivos de datos CSV.
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
from neurolab.synapses import MemristiveSynapse, STDPRule, AntiSTDPRule


def simulate_stdp(rule, dt_values, n_repeats=10):
    """
    Simula la ventana STDP para una regla dada.

    Parameters
    ----------
    rule : STDPRule
        Regla STDP a aplicar.
    dt_values : array
        Valores de Δt a barrer (s).
    n_repeats : int
        Número de repeticiones por cada Δt.

    Returns
    -------
    dW_mean, dW_std : arrays
        Cambio de peso promedio y desviación estándar.
    """
    dW_mean = np.zeros(len(dt_values))
    dW_std = np.zeros(len(dt_values))

    for i, dt in enumerate(dt_values):
        dWs = []
        for _ in range(n_repeats):
            cfg = StrukovConfig(
                RON=100.0, ROFF=16_000.0, x0=0.10,
                D=10e-9, mu_v=1e-14, clip_x=True
            )
            mem = MemristorStrukov(cfg)
            syn = MemristiveSynapse(mem)

            dW = rule.apply(syn, dt)
            dWs.append(dW)

        dW_mean[i] = np.mean(dWs)
        dW_std[i] = np.std(dWs)

    return dW_mean, dW_std


def main():
    base_dir = os.path.dirname(__file__)
    png_path = os.path.join(base_dir, "validacion_stdp.png")
    csv_path = os.path.join(base_dir, "validacion_stdp.csv")

    # --- Barrido de Δt (-80 ms a +80 ms) ---
    dt_values = np.linspace(-0.08, 0.08, 33)  # 33 puntos

    # --- Instanciación de Reglas ---
    hebbian = STDPRule(
        A_plus=0.05, A_minus=-0.025,
        tau_plus=17e-3, tau_minus=34e-3
    )
    anti_hebbian = AntiSTDPRule(
        A_plus=0.05, A_minus=-0.025,
        tau_plus=17e-3, tau_minus=34e-3
    )

    # --- Ejecutar Simulaciones ---
    dW_hebb, _ = simulate_stdp(hebbian, dt_values)
    dW_anti, _ = simulate_stdp(anti_hebbian, dt_values)

    # --- Exportar CSV ---
    df_stdp = pd.DataFrame({
        'delta_t_ms': dt_values * 1e3,
        'delta_w_hebbian': dW_hebb,
        'delta_w_anti_hebbian': dW_anti
    })
    df_stdp.to_csv(csv_path, index=False)

    # --- Figura de 2 Paneles ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: STDP Hebbiano
    axes[0].plot(dt_values * 1e3, dW_hebb, 'b-o', ms=5, lw=1.8, label='STDP Hebbiano')
    axes[0].axhline(0, color='k', ls='--', lw=0.8)
    axes[0].axvline(0, color='k', ls='--', lw=0.8)
    axes[0].set_xlabel('Δt = t_post − t_pre (ms)', fontsize=11)
    axes[0].set_ylabel('ΔW (Cambio de peso)', fontsize=11)
    axes[0].set_title('STDP Hebbiano (Ventana Asimétrica Biológica)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].text(25, 0.035, 'LTP (Δt > 0)', color='navy', fontsize=11, fontweight='bold')
    axes[0].text(-65, -0.018, 'LTD (Δt < 0)', color='darkred', fontsize=11, fontweight='bold')
    axes[0].legend()

    # Panel 2: STDP Anti-Hebbiano
    axes[1].plot(dt_values * 1e3, dW_anti, 'r-s', ms=5, lw=1.8, label='STDP Anti-Hebbiano')
    axes[1].axhline(0, color='k', ls='--', lw=0.8)
    axes[1].axvline(0, color='k', ls='--', lw=0.8)
    axes[1].set_xlabel('Δt = t_post − t_pre (ms)', fontsize=11)
    axes[1].set_ylabel('ΔW (Cambio de peso)', fontsize=11)
    axes[1].set_title('STDP Anti-Hebbiano (Regla Invertida)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    # --- Imprimir Métricas ---
    idx_max = np.argmax(dW_hebb)
    idx_min = np.argmin(dW_hebb)

    print("=" * 60)
    print("  RESULTADOS DE VALIDACIÓN: STDP (Spike-Timing Plasticity)")
    print("=" * 60)
    print(f"STDP Hebbiano Max: ΔW_max = +{dW_hebb[idx_max]:.4f} en Δt = {dt_values[idx_max]*1e3:.1f} ms")
    print(f"STDP Hebbiano Min: ΔW_min = {dW_hebb[idx_min]:.4f} en Δt = {dt_values[idx_min]*1e3:.1f} ms")
    print(f"Parámetros: τ+ = {hebbian.tau_plus*1e3:.1f} ms, τ- = {hebbian.tau_minus*1e3:.1f} ms")
    print(f"Gráfica guardada en: {png_path}")
    print(f"Datos exportados a:  {csv_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
