"""
paso1_simulacion_basica.py
===========================
PASO 1: Simulación básica del memristor aislado (Modelo de Strukov determinista).

Objetivo:
    Demostrar el comportamiento fundamental del memristor de TiO2 obteniendo
    la curva característica I-V (lazo de histéresis estrangulado) en el origen.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Asegurar que el paquete memristor_simulator sea encontrable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from memristor_simulator.config.parameters import StrukovParameters
from memristor_simulator.models.strukov_model import StrukovMemristor

def main():
    print("Iniciando Paso 1: Simulación básica del memristor aislado...")

    # Parámetros físicos basados en HP Labs (2008)
    params = StrukovParameters(
        R_on=100.0,
        R_off=16000.0,  # Relación R_off/R_on reducida para visualización clara del lazo
        D=10e-9,
        mu_v=1e-14,
        w_init=0.1,
        enable_nonlinear_drift=True,
        enable_hard_switching=True
    )

    device = StrukovMemristor(params)

    # Configuración del estímulo senoidal (1 Hz, 1 V de amplitud)
    f = 1.0  # Hz
    T = 2.0  # s (2 ciclos completos)
    dt = 1e-4  # s
    time = np.arange(0, T, dt)
    n_steps = len(time)

    voltage = np.sin(2 * np.pi * f * time)
    current = np.zeros(n_steps)
    state = np.zeros(n_steps)
    resistance = np.zeros(n_steps)

    # Integración temporal
    print("Ejecutando integración temporal con estímulo senoidal...")
    for i in range(n_steps):
        state[i] = device.state
        resistance[i] = device.resistance
        current[i] = device.step(voltage[i], dt)

    print("Simulación finalizada. Graficando resultados...")

    # Gráficas
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # Panel 1: Curva I-V (Lazo de histéresis estrangulado)
    axes[0].plot(voltage, current * 1000, color='blue', lw=2)
    axes[0].axhline(0, color='gray', ls='--', lw=0.5)
    axes[0].axvline(0, color='gray', ls='--', lw=0.5)
    axes[0].set_title("Curva Corriente-Voltaje (Histéresis)", fontweight="bold")
    axes[0].set_xlabel("Voltaje (V)")
    axes[0].set_ylabel("Corriente (mA)")
    axes[0].grid(True, ls=":", alpha=0.5)

    # Panel 2: Evolución de la variable de estado w/D
    axes[1].plot(time, state, color='green', lw=2)
    axes[1].set_title("Evolución del Estado Interno (w/D)", fontweight="bold")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylabel("x = w/D")
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].grid(True, ls=":", alpha=0.5)

    plt.suptitle("PASO 1 — SIMULACIÓN BÁSICA DEL MEMRISTOR DE STRUKOV", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output_modular")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "paso1_simulacion_basica.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Gráfica generada exitosamente en: {out_path}")
    # plt.show()

if __name__ == "__main__":
    main()
