"""
paso22_analisis_sneak_paths.py
==============================
Script para replicar el análisis de "Sneak Paths" (corrientes de fuga parásitas)
en arreglos Crossbar de distintos tamaños (4x4 hasta 64x64), en función de Vdd y Kon.

El modelo analítico para un esquema de lectura con líneas no seleccionadas flotantes es:
R_sneak = R_unsel / (N-1) + R_unsel / (N-1)^2 + R_unsel / (N-1)
I_sneak = Vdd / R_sneak
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.physical_crossbar import PhysicalCrossbarArray
from memristor_simulator.config.parameters import StrukovParameters

# ── Parámetros del barrido ───────────────────────────────────────────────────
Vdds = [1.0, 1.5, 2.0, 2.5, 3.0]
Kons = [1e-9, 3e-8, 5e-8, 8e-8, 1e-7]
Ns = [4, 8, 16, 32, 64]
ON_OFF_RATIO = 1e4  # Asumimos un ratio típico para reproducir las barras "All Zeros"

# ── Función de cálculo mediante red física ──────────────────────────────────
def calc_isneak(Vdd, K_val, N, pattern="All Ones"):
    """ Calcula la corriente Sneak Path instanciando físicamente el Crossbar """
    p_base = StrukovParameters(R_on=1.0/K_val, R_off=1e9, D=10e-9, mu_v=0.0)
    crossbar = PhysicalCrossbarArray(size=N, base_params=p_base, on_off_ratio=ON_OFF_RATIO)
    
    if pattern == "All Ones":
        crossbar.set_pattern_all_ones()
    else:
        crossbar.set_pattern_all_zeros()
        
    R_sneak = crossbar.get_sneak_resistance(target_row=0, target_col=0)
    return Vdd / R_sneak

# ── Crear la figura (Grid de 5x5) ────────────────────────────────────────────
fig, axes = plt.subplots(nrows=len(Vdds), ncols=len(Kons), figsize=(16, 10), sharey=True)
fig.patch.set_facecolor('white')

# Colores para los distintos tamaños de Crossbar
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Variables para crear la leyenda global al final
bars_for_legend = []
labels_for_legend = [f"{n}x{n}" for n in Ns]

for i, vdd in enumerate(Vdds):
    for j, kon in enumerate(Kons):
        ax = axes[i, j]
        
        # Datos a plotear
        iones_vals = []
        izeros_vals = []
        
        for n in Ns:
            # All Ones
            i_ones = calc_isneak(vdd, kon, n, pattern="All Ones")
            # All Zeros
            i_zeros = calc_isneak(vdd, kon, n, pattern="All Zeros")
            
            # Convertir a µA
            iones_vals.append(i_ones * 1e6)
            izeros_vals.append(i_zeros * 1e6)
            
        # Posiciones de las barras
        x = np.arange(2)  # [All Ones, All Zeros]
        width = 0.15
        
        for idx, n in enumerate(Ns):
            offset = (idx - 2) * width
            bar = ax.bar(x + offset, [iones_vals[idx], izeros_vals[idx]], 
                         width, color=colors[idx])
            if i == 0 and j == 0:
                bars_for_legend.append(bar)
                
        # Formato del subplot
        ax.set_yscale('log')
        ax.set_ylim([1e-5, 1e1])
        ax.set_xticks(x)
        ax.set_xticklabels(['All Ones', 'All Zeros'], fontsize=9)
        
        # Títulos de filas y columnas simulando la imagen original
        if i == 0:
            ax.set_title(f"Kon = {kon:.0e}".replace('e-0', 'e-'), fontsize=13, fontweight='bold', pad=15)
        if j == 0:
            ax.set_ylabel(f"Vdd = {vdd} V\n\nIsneak (µA)", fontsize=11, fontweight='bold', labelpad=10)
        else:
            ax.set_ylabel("Isneak (µA)", fontsize=9)
            
        if i == len(Vdds) - 1:
            ax.set_xlabel("Data Pattern", fontsize=10)

# Leyenda global en la parte inferior
fig.legend(bars_for_legend, labels_for_legend, loc='lower center', ncol=5, fontsize=12, bbox_to_anchor=(0.5, -0.05))

# Título global
fig.suptitle("Sneak-path current (Isneak) dependence on Crossbar Size, Read Voltage (Vdd) and Conductance (Kon)", 
             fontsize=16, fontweight='bold', y=1.05)

plt.tight_layout()

# ── Guardar la imagen ────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso22_analisis_sneak_paths.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[PNG] Guardado: {out_path}")
