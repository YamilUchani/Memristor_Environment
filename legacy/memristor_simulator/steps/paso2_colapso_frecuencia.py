"""
paso2_colapso_frecuencia.py
============================
PASO 2: Colapso de la histéresis por frecuencia.

Objetivo:
    Demostrar la naturaleza inercial iónica del memristor simulando el colapso
    del lazo de histéresis al aumentar la frecuencia de la señal de entrada.
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
    print("Iniciando Paso 2: Simulación de colapso por frecuencia...")

    # Parámetros físicos (HP Labs)
    params = StrukovParameters(
        R_on=100.0,
        R_off=16000.0,
        D=10e-9,
        mu_v=1e-14,
        w_init=0.1,
        enable_nonlinear_drift=True,
        enable_hard_switching=True
    )

    frequencies = [0.5, 2.0, 10.0]  # Hz
    colors = ['#1f77b4', '#9467bd', '#d62728']

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("white")

    dt = 1e-4  # s
    T_cycle = 2.0  # s (evaluaremos una ventana de tiempo fija)

    for f, color in zip(frequencies, colors):
        device = StrukovMemristor(params)
        time = np.arange(0, T_cycle, dt)
        n_steps = len(time)

        voltage = np.sin(2 * np.pi * f * time)
        current = np.zeros(n_steps)

        for i in range(n_steps):
            current[i] = device.step(voltage[i], dt)

        # Graficar corriente en mA
        plt.plot(voltage, current * 1000, label=f"f = {f} Hz", color=color, lw=2.0)

    plt.axhline(0, color='gray', ls='--', lw=0.5)
    plt.axvline(0, color='gray', ls='--', lw=0.5)
    plt.title("Colapso de Histéresis por Frecuencia", fontsize=13, fontweight="bold")
    plt.xlabel("Voltaje (V)")
    plt.ylabel("Corriente (mA)")
    plt.legend(fontsize=11)
    plt.grid(True, ls=":", alpha=0.5)

    plt.suptitle("PASO 2 — EFECTO DE FRECUENCIA EN EL LAZO DE HISTÉRESIS", fontsize=14, fontweight="bold")
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output_modular")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "paso2_colapso_frecuencia.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Gráfica generada exitosamente en: {out_path}")
    # plt.show()

if __name__ == "__main__":
    main()
