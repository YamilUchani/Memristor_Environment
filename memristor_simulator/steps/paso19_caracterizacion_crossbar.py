"""
paso19_caracterizacion_crossbar.py
==================================
Script para replicar la caracterización de un crossbar 10x8 en su estado virgen 
(pre-forming), comparando con la Figura S3.

1. Simula la variabilidad dispositivo a dispositivo (Device-to-Device).
2. Extrae las curvas I-V.
3. Genera el mapa de calor (Heatmap) de las conductancias a 0.1V.
4. Genera el histograma de las conductancias.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.strukov_model import StrukovMemristor
from memristor_simulator.config.parameters import StrukovParameters

class NonLinearStrukovMemristor(StrukovMemristor):
    """
    Extensión del modelo físico base para incluir la conducción no lineal
    (tipo Schottky o Poole-Frenkel) característica del estado virgen pre-forming.
    """
    def step(self, voltage: float, dt: float) -> float:
        # Integramos físicamente el estado interno usando el modelo base
        linear_current = super().step(voltage, dt)
        
        # Añadimos la no linealidad estática del TiO2 no dopado
        if self.resistance > 1e6: # Solo aplicable fuertemente en estado HRS / Virgen
            non_linear_term = np.sign(voltage) * 0.5e-6 * (np.abs(voltage)**3)
            return linear_current + non_linear_term
        return linear_current

# ── 1. Configuración del Crossbar 10x8 ───────────────────────────────────────
ROWS = 10
COLS = 8

# Valores estadísticos basados en la Figura S3 (c)
# La conductancia media está alrededor de 0.45 µS con una desviación de ~0.08 µS
mean_G = 0.45e-6  # 0.45 uS
std_G  = 0.08e-6  # 0.08 uS

# Crear la matriz de memristores con variabilidad en R_off
memristors = []
G_matrix = np.zeros((ROWS, COLS))

np.random.seed(42) # Para reproducibilidad

for i in range(ROWS):
    row_mems = []
    for j in range(COLS):
        # Muestrear una conductancia inicial de la distribución normal
        g_val = np.random.normal(mean_G, std_G)
        g_val = np.clip(g_val, 0.25e-6, 0.65e-6) # Limitar a los valores de la figura
        G_matrix[i, j] = g_val
        
        # Calcular R_off correspondiente
        r_off_val = 1.0 / g_val
        
        p_mem = StrukovParameters(
            R_on=100.0, 
            R_off=r_off_val, 
            D=10e-9, 
            mu_v=1e-16, # Movilidad baja para no perturbarlo durante la lectura
            w_init=0.0
        )
        row_mems.append(NonLinearStrukovMemristor(p_mem))
    memristors.append(row_mems)

# ── 2. Extracción de datos ───────────────────────────────────────────────────

# (a) Curvas I-V de algunos dispositivos representativos
V_sweep = np.linspace(-0.8, 0.8, 100)
I_curves = []
# Seleccionamos 5 dispositivos al azar para la curva I-V
devices_for_iv = [(0,0), (2,3), (5,5), (7,1), (9,7)]

for r, c in devices_for_iv:
    mem = memristors[r][c]
    i_curve = []
    
    # Extraer I-V haciendo un barrido físico a través del integrador step()
    dt_sweep = 1e-4
    for v in V_sweep:
        i_val = mem.step(v, dt_sweep)
        i_curve.append(i_val)
    I_curves.append(i_curve)

# (b) y (c) Conductancias efectivas a 0.1V ya están en G_matrix
G_flat = G_matrix.flatten() * 1e6 # Convertir a µS

# ── 3. Generación de la Figura S3 ─────────────────────────────────────────────
fig = plt.figure(figsize=(8, 14))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(3, 1, hspace=0.4)

# Panel (a): Curvas I-V
ax_a = fig.add_subplot(gs[0])
colors = ['cyan', 'yellow', 'magenta', 'orange', 'dodgerblue']
for idx, i_curve in enumerate(I_curves):
    ax_a.plot(V_sweep, np.array(i_curve)*1e6, color=colors[idx], lw=2)
ax_a.set_xlabel("Voltage (V)", fontsize=12)
ax_a.set_ylabel("Current (µA)", fontsize=12)
ax_a.set_xlim([-1.0, 1.0])
ax_a.set_ylim([-3.0, 5.0])
ax_a.set_title("(a) Representative I-V curves", loc='left', fontsize=14, fontweight='bold')
ax_a.tick_params(axis='both', labelsize=11)
ax_a.grid(True, ls=':', alpha=0.5)

# Panel (b): Heatmap de Conductancia
ax_b = fig.add_subplot(gs[1])
# Colormap de Verde (baja G) a Rojo (alta G) simulando la imagen
cmap = plt.cm.RdYlGn_r 
im = ax_b.imshow(G_matrix, cmap=cmap, aspect='auto')

for i in range(ROWS):
    for j in range(COLS):
        # Formato notación científica ej: 4.5E-07
        text_val = f"{G_matrix[i, j]:.1E}".replace('E-0', 'E-0')
        ax_b.text(j, i, text_val, ha="center", va="center", color="black", fontsize=8)

ax_b.set_xticks([])
ax_b.set_yticks([])
ax_b.set_title("(b) Conductance map", loc='left', fontsize=14, fontweight='bold')

# Panel (c): Histograma
ax_c = fig.add_subplot(gs[2])
ax_c.hist(G_flat, bins=10, color='red', edgecolor='white', hatch='\\\\\\\\')
ax_c.set_xlabel("Conductance @ 0.1V (µS)", fontsize=12)
ax_c.set_ylabel("Count", fontsize=12)
ax_c.set_xlim([0.25, 0.65])
ax_c.set_title("(c) Histogram", loc='left', fontsize=14, fontweight='bold')
ax_c.tick_params(axis='both', labelsize=11)
ax_c.grid(True, ls=':', alpha=0.5, axis='y')

fig.suptitle("Pre-forming characterization of a 10x8 portion of the crossbar", fontsize=14, y=0.94)

out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso19_caracterizacion_crossbar.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[PNG] Guardado: {out_path}")
