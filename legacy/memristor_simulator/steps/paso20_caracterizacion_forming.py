"""
paso20_caracterizacion_forming.py
=================================
Script para replicar la caracterización del proceso de FORMING (electroformado) 
en un crossbar 10x8, comparando con la Figura S4.

Simula:
1. La distribución estadística de los voltajes de electroformado (V_form).
2. Curvas I-V típicas de forming (con salto abrupto de corriente).
3. Mapa de calor de los voltajes de forming.
4. Histograma de voltajes de forming.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.strukov_model import StrukovMemristor
from memristor_simulator.config.parameters import StrukovParameters

class FormingMemristor(StrukovMemristor):
    """
    Extensión del modelo físico base para modelar el evento de electroformado
    de un solo uso.
    """
    def __init__(self, params: StrukovParameters, v_form: float):
        super().__init__(params)
        self.v_form = v_form
        self.is_formed = False
        self.r_virgin = 2.5e6  # ~0.4 uS resistencia pre-forming
        self.compliance = 200e-6 # Límite de corriente para evitar daño

    def step(self, voltage: float, dt: float) -> float:
        if not self.is_formed:
            if abs(voltage) >= self.v_form:
                self.is_formed = True
                # Tras el rompimiento, pasa a comportarse como el modelo Strukov normal
                current = super().step(voltage, dt)
                return np.clip(current, -self.compliance, self.compliance)
            else:
                return voltage / self.r_virgin
        else:
            current = super().step(voltage, dt)
            return np.clip(current, -self.compliance, self.compliance)

# ── 1. Generación de datos estadísticos de Forming (10x8) ────────────────────
ROWS = 10
COLS = 8

np.random.seed(100)

# El histograma S4(c) muestra voltajes de forming entre 1.7V y 2.1V, con pico en ~1.9V
mean_Vf = 1.9
std_Vf = 0.08
V_forming_matrix = np.random.normal(mean_Vf, std_Vf, (ROWS, COLS))
V_forming_matrix = np.clip(V_forming_matrix, 1.7, 2.15)

# Para simular las curvas I-V de forming (Figura S4a)
# El forming es un evento de un solo uso donde la resistencia cae abruptamente
R_virgin = 2.5e6  # ~0.4 uS
R_formed = 10e3   # ~100 uS
I_compliance = 200e-6 # 200 uA de corriente de cumplimiento (compliance)

# Seleccionamos algunas curvas simuladas con distintos V_forming
V_forms_to_plot = [1.80, 1.85, 1.90, 1.95, 2.00, 2.05, 2.10]
I_curves = []

for vf in V_forms_to_plot:
    # Instanciamos un dispositivo físico con su voltaje de forming respectivo
    p_mem = StrukovParameters(R_on=10e3, R_off=100e3, D=10e-9, mu_v=0.0) # mu_v=0.0 porque el forming es casi instantáneo
    mem = FormingMemristor(p_mem, vf)
    
    v_sweep = np.linspace(0, 2.2, 200)
    dt_sweep = 1e-4
    i_curve = []
    
    for v in v_sweep:
        i_curve.append(mem.step(v, dt_sweep))
    
    # Sweep back (retorno) ya en estado formado
    v_sweep_back = np.linspace(2.2, 0, 200)
    i_curve_back = []
    for v in v_sweep_back:
        i_curve_back.append(mem.step(v, dt_sweep))
        
    I_curves.append((v_sweep, i_curve, v_sweep_back, i_curve_back))

# ── 2. Generación de la Figura S4 ────────────────────────────────────────────
fig = plt.figure(figsize=(10, 12))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(3, 1, hspace=0.4)

# Panel (a): Curvas I-V de Forming
ax_a = fig.add_subplot(gs[0])
colors = plt.cm.jet(np.linspace(0, 1, len(V_forms_to_plot)))
for idx, (v_up, i_up, v_dn, i_dn) in enumerate(I_curves):
    # Trazar ida
    ax_a.plot(v_up, np.array(i_up)*1e6, color=colors[idx], lw=1.5)
    # Trazar vuelta
    ax_a.plot(v_dn, np.array(i_dn)*1e6, color=colors[idx], lw=1.5)

ax_a.set_xlabel("Voltage (V)", fontsize=12)
ax_a.set_ylabel("Current (µA)", fontsize=12)
ax_a.set_xlim([0.0, 2.3])
ax_a.set_title("(a) Typical forming switching curves", loc='left', fontsize=14, fontweight='bold')
ax_a.grid(True, ls=':', alpha=0.5)

# Panel (b): Heatmap de Forming Voltage
ax_b = fig.add_subplot(gs[1])
im = ax_b.imshow(V_forming_matrix, cmap='gray_r', aspect='auto', vmin=1.7, vmax=2.15)
plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04, label="Forming Voltage (V)")

ax_b.set_xticks(range(COLS))
ax_b.set_xticklabels(range(1, COLS+1))
ax_b.set_yticks(range(ROWS))
ax_b.set_yticklabels(range(1, ROWS+1))
ax_b.set_xlabel("column", fontsize=12)
ax_b.set_ylabel("row", fontsize=12)
ax_b.set_title("(b) Map of forming voltages", loc='left', fontsize=14, fontweight='bold')

# Panel (c): Histograma
ax_c = fig.add_subplot(gs[2])
ax_c.hist(V_forming_matrix.flatten(), bins=10, color='red', edgecolor='white', hatch='\\\\\\\\', alpha=0.7)
ax_c.set_xlabel("Forming voltage (V)", fontsize=12)
ax_c.set_ylabel("Count", fontsize=12)
ax_c.set_xlim([1.65, 2.2])
ax_c.set_title("(c) Histogram of forming voltages", loc='left', fontsize=14, fontweight='bold')
ax_c.grid(True, ls=':', alpha=0.5, axis='y')

fig.suptitle("Characterization of forming in a 10x8 portion of the crossbar", fontsize=15, y=0.94)

out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso20_caracterizacion_forming.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[PNG] Guardado: {out_path}")
