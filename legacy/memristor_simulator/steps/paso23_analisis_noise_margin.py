"""
paso23_analisis_noise_margin.py
===============================
Script para replicar el análisis de "Noise Margin" (Margen de Ruido)
en arreglos Crossbar de distintos tamaños (4x4 hasta 64x64), en función de Vdd y Kon.

El Noise Margin decae debido a la caída de voltaje en las líneas metálicas o 
resistencias de los drivers/amplificadores (R_wire) provocada por la alta corriente 
de Sneak Path. 
Modelo utilizado: Divisor de tensión simple.
NM = R_sneak / (R_sneak + R_wire)
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
ON_OFF_RATIO = 1e4  

# Resistencia parásita equivalente (cables M3/M5/M6 o drivers de lectura)
# Calibrada para que coincida con la caída a NM=0.45 para el caso 64x64, Kon=1e-7
R_wire = 380e3 # 380 kOhm

# ── Función de cálculo mediante red física ──────────────────────────────────
def calc_rsneak(K_val, N, pattern="All Ones"):
    """ Calcula la resistencia equivalente instanciando físicamente el Crossbar """
    p_base = StrukovParameters(R_on=1.0/K_val, R_off=1e9, D=10e-9, mu_v=0.0)
    crossbar = PhysicalCrossbarArray(size=N, base_params=p_base, on_off_ratio=ON_OFF_RATIO)
    
    if pattern == "All Ones":
        crossbar.set_pattern_all_ones()
    else:
        crossbar.set_pattern_all_zeros()
        
    return crossbar.get_sneak_resistance(target_row=0, target_col=0)

# ── Crear la figura (Grid de 5x5) ────────────────────────────────────────────
fig, axes = plt.subplots(nrows=len(Vdds), ncols=len(Kons), figsize=(16, 10), sharey=True)
fig.patch.set_facecolor('white')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bars_for_legend = []
labels_for_legend = [f"{n}x{n}" for n in Ns]

for i, vdd in enumerate(Vdds):
    for j, kon in enumerate(Kons):
        ax = axes[i, j]
        
        nm_ones_vals = []
        nm_zeros_vals = []
        
        for n in Ns:
            # All Ones
            rsneak_ones = calc_rsneak(kon, n, pattern="All Ones")
            nm_ones = rsneak_ones / (rsneak_ones + R_wire)
            
            # All Zeros
            rsneak_zeros = calc_rsneak(kon, n, pattern="All Zeros")
            nm_zeros = rsneak_zeros / (rsneak_zeros + R_wire)
            
            nm_ones_vals.append(nm_ones)
            nm_zeros_vals.append(nm_zeros)
            
        x = np.arange(2)
        width = 0.15
        
        for idx, n in enumerate(Ns):
            offset = (idx - 2) * width
            bar = ax.bar(x + offset, [nm_ones_vals[idx], nm_zeros_vals[idx]], 
                         width, color=colors[idx])
            if i == 0 and j == 0:
                bars_for_legend.append(bar)
                
        # Formato del subplot
        ax.set_ylim([0, 1.1])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticks(x)
        ax.set_xticklabels(['All Ones', 'All Zeros'], fontsize=9)
        
        if i == 0:
            ax.set_title(f"Kon = {kon:.0e}".replace('e-0', 'e-'), fontsize=13, fontweight='bold', pad=15)
        if j == 0:
            ax.set_ylabel(f"Vdd = {vdd} V\n\nNoise Margin", fontsize=11, fontweight='bold', labelpad=10)
        else:
            ax.set_ylabel("Noise Margin", fontsize=9)
            
        if i == len(Vdds) - 1:
            ax.set_xlabel("Data Pattern", fontsize=10)

fig.legend(bars_for_legend, labels_for_legend, loc='lower center', ncol=5, fontsize=12, bbox_to_anchor=(0.5, -0.05))
fig.suptitle("Noise Margin dependence on Crossbar Size, Read Voltage (Vdd) and Conductance (Kon)", 
             fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()

# ── Guardar la imagen ────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso23_analisis_noise_margin.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[PNG] Guardado: {out_path}")
