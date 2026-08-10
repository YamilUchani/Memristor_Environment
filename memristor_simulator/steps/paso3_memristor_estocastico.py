"""
paso3_memristor_estocastico.py
===============================
PASO 3: Caracterización estocástica (variabilidad ciclo a ciclo - C2C).

Objetivo:
    Demostrar el impacto de la variabilidad física Cycle-to-Cycle (C2C) 
    sobre un memristor aislado ante estímulos alternos repetidos.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Asegurar que el paquete memristor_simulator sea encontrable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from memristor_simulator.config.parameters import StrukovParameters
from memristor_simulator.models.stochastic_model import StochasticMemristor, C2CConfig

def main():
    print("Iniciando Paso 3: Simulación de variabilidad estocástica C2C...")

    # Parámetros nominales del dispositivo
    nom_params = StrukovParameters(
        R_on=100.0,
        R_off=16000.0,
        D=10e-9,
        mu_v=1e-14,
        w_init=0.1,
        enable_nonlinear_drift=True,
        enable_hard_switching=True
    )

    # Configuración estocástica C2C (15% CV para R_on, 10% CV para R_off)
    c2c_cfg = C2CConfig(
        enabled=True,
        r_on_cv=0.15,
        r_off_cv=0.10,
        state_noise_sigma=0.02,
        seed=42
    )

    device = StochasticMemristor(nom_params, c2c_cfg)

    # Simular 10 ciclos consecutivos a 0.5 Hz
    f = 0.5
    dt = 1e-3
    T_cycle = 1.0 / f  # 2.0 s por ciclo
    n_cycles = 10
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    print(f"Ejecutando {n_cycles} ciclos con variabilidad activa...")

    # Guardar trazas temporales para graficar
    for cycle in range(n_cycles):
        time = np.arange(0, T_cycle, dt)
        n_steps = len(time)
        voltage = np.sin(2 * np.pi * f * time)
        current = np.zeros(n_steps)
        state = np.zeros(n_steps)

        for i in range(n_steps):
            state[i] = device.state
            # detect_cycle=True evalúa e inyecta la variabilidad C2C al cruzar cero
            current[i] = device.step(voltage[i], dt, detect_cycle=True)

        # Graficar el ciclo actual en ambos paneles
        axes[0].plot(voltage, current * 1000, alpha=0.6, lw=1.5)
        axes[1].plot(time + (cycle * T_cycle), state, alpha=0.7, lw=1.5)

    axes[0].axhline(0, color='gray', ls='--', lw=0.5)
    axes[0].axvline(0, color='gray', ls='--', lw=0.5)
    axes[0].set_title("Lazos de Histéresis Ruidosos (I-V)", fontweight="bold")
    axes[0].set_xlabel("Voltaje (V)")
    axes[0].set_ylabel("Corriente (mA)")
    axes[0].grid(True, ls=":", alpha=0.5)

    axes[1].set_title("Evolución del Estado Interno (w/D)", fontweight="bold")
    axes[1].set_xlabel("Tiempo acumulado (s)")
    axes[1].set_ylabel("x = w/D")
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].grid(True, ls=":", alpha=0.5)

    plt.suptitle("PASO 3 — CARACTERIZACIÓN ESTOCÁSTICA C2C DEL DISPOSITIVO", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output_modular")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "paso3_memristor_estocastico.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Gráfica generada exitosamente en: {out_path}")
    # plt.show()

if __name__ == "__main__":
    main()
