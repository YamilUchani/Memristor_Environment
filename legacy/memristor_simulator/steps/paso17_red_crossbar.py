"""
paso17_red_crossbar.py
======================
Red Neuromórfica Colectiva: Matriz Crossbar 3x3
6 Neuronas (3 entrada + 3 salida) conectadas por 9 sinapsis de memristor.

Demuestra:
- Procesamiento paralelo colectivo.
- Integración de múltiples fuentes sinápticas.
- Actividad diferenciada según los pesos de la matriz.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.lif_neuron import default_lif_params, LIFNeuron
from memristor_simulator.models.strukov_model import StrukovMemristor
from memristor_simulator.config.parameters import StrukovParameters

# ── Configuración ─────────────────────────────────────────────────────────────
dt = 0.05e-3
T = 0.5  # 500 ms
n = int(T / dt)
t = np.arange(n) * dt
t_ms = t * 1e3

# ── Matriz de pesos sinápticos (3x3) ─────────────────────────────────────────
# Cada valor es w_init del memristor (0.01 = HRS/débil, 1.0 = LRS/fuerte)
# Filas = Neuronas de Entrada (In0, In1, In2)
# Columnas = Neuronas de Salida (Out0, Out1, Out2)
#
# Diseño intencional:
#   Out0: Responde fuertemente a In0 y In1 (detector de actividad conjunta)
#   Out1: Responde solo a In2 (canal exclusivo)
#   Out2: Responde débilmente a todo (inhibida)
W_matrix = np.array([
    [1.00, 0.01, 0.05],  # In0 -> Out0 fuerte, Out1 débil, Out2 débil
    [0.90, 0.01, 0.05],  # In1 -> Out0 fuerte, Out1 débil, Out2 débil
    [0.01, 1.00, 0.05],  # In2 -> Out0 débil,  Out1 fuerte, Out2 débil
])

# ── Crear Neuronas de Entrada ─────────────────────────────────────────────────
p_lif_in = default_lif_params()
p_lif_in.R_s = 80e3  # Menor Rs para que disparen mas facil

# Estimulos diferenciados para cada neurona de entrada
# In0: Estimulo fuerte y constante -> dispara rapido (~alta frecuencia)
# In1: Estimulo moderado -> dispara mas lento
# In2: Estimulo debil con rafaga tardia -> solo dispara al final
V_stim = np.zeros((3, n))
V_stim[0, :] = 2.5                          # Constante fuerte
V_stim[1, :] = 2.0                          # Constante moderado
V_stim[2, :int(0.25/dt)] = 0.3              # Silencio inicial
V_stim[2, int(0.25/dt):] = 2.5              # Rafaga tardia fuerte

# ── Crear Neuronas de Salida ──────────────────────────────────────────────────
p_lif_out = default_lif_params()
p_lif_out.R_s = 80e3  # Menor Rs para que sean más sensibles a corriente colectiva

# ── Crear Memristores (Matriz 3x3) ───────────────────────────────────────────
memristors = []
for i in range(3):
    row = []
    for j in range(3):
        p_mem = StrukovParameters(
            R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=0.0,  # mu_v=0 -> peso congelado
            w_init=W_matrix[i, j]
        )
        row.append(StrukovMemristor(p_mem))
    memristors.append(row)

# ── Instanciar neuronas ───────────────────────────────────────────────────────
neurons_in = [LIFNeuron(p_lif_in) for _ in range(3)]
neurons_out = [LIFNeuron(p_lif_out) for _ in range(3)]

# ── Almacenamiento ────────────────────────────────────────────────────────────
Vc_in = np.zeros((3, n))
w_in = np.zeros((3, n))
Vc_out = np.zeros((3, n))
w_out = np.zeros((3, n))
I_total_out = np.zeros((3, n))

# Temporizadores de spike para cada neurona de entrada
spike_timers = [0.0, 0.0, 0.0]
spike_duration = 2e-3
spike_amplitude = 5.0
V_out_in = np.zeros((3, n))  # Voltaje de salida de cada neurona de entrada

# ── Simulación ────────────────────────────────────────────────────────────────
for k in range(n):
    # --- Paso 1: Actualizar neuronas de entrada ---
    for i in range(3):
        V_eff = V_stim[i, k]
        neurons_in[i].step(V_eff, dt)
        Vc_in[i, k] = neurons_in[i].V
        w_in[i, k] = neurons_in[i].w
        
        # Detectar spike y emitir pulso físico
        if neurons_in[i].w >= 0.5 and spike_timers[i] <= 0:
            spike_timers[i] = spike_duration
        
        if spike_timers[i] > 0:
            V_out_in[i, k] = spike_amplitude
            spike_timers[i] -= dt
        else:
            V_out_in[i, k] = 0.0
    
    # --- Paso 2: Calcular corrientes sinápticas a través de la matriz ---
    for j in range(3):  # Para cada neurona de salida
        I_total = 0.0
        for i in range(3):  # Sumar contribuciones de todas las neuronas de entrada
            R_m = memristors[i][j].resistance
            I_syn = (V_out_in[i, k] - neurons_out[j].V) / (R_m + p_lif_out.R_s)
            I_total += I_syn
        
        I_total_out[j, k] = I_total
        
        # --- Paso 3: Actualizar neurona de salida ---
        V_eff_out = neurons_out[j].V + I_total * p_lif_out.R_s
        neurons_out[j].step(V_eff_out, dt)
        Vc_out[j, k] = neurons_out[j].V
        w_out[j, k] = neurons_out[j].w

# ── Detectar spikes ──────────────────────────────────────────────────────────
def detect_spikes(w_array):
    edges = np.diff((w_array >= 0.5).astype(int)) > 0
    return t_ms[1:][edges]

spikes_in = [detect_spikes(w_in[i]) for i in range(3)]
spikes_out = [detect_spikes(w_out[j]) for j in range(3)]

# Calcular resistencias para el heatmap
R_matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        R_matrix[i, j] = memristors[i][j].resistance

# ── Gráfica ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35, top=0.88, bottom=0.08, left=0.07, right=0.97)

colors_in = ['#1565C0', '#2E7D32', '#E65100']
colors_out = ['#D32F2F', '#7B1FA2', '#00838F']
labels_in = ['In₀ (Alta Freq)', 'In₁ (Media Freq)', 'In₂ (Ráfaga Tardía)']
labels_out = ['Out₀ (Detector AND)', 'Out₁ (Canal In₂)', 'Out₂ (Inhibida)']

# Panel 1: Raster Plot de Entrada
ax1 = fig.add_subplot(gs[0, 0])
for i in range(3):
    if len(spikes_in[i]) > 0:
        ax1.scatter(spikes_in[i], np.full_like(spikes_in[i], i), 
                   marker='|', s=200, lw=2, color=colors_in[i], label=labels_in[i])
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(['In₀', 'In₁', 'In₂'])
ax1.set_xlabel("Tiempo (ms)")
ax1.set_title("① Actividad de Entrada\n(Raster Plot)", fontweight='bold')
ax1.set_xlim([0, T*1e3])
ax1.grid(True, ls=':', alpha=0.5)
ax1.legend(loc='upper right', fontsize=7)

# Panel 2: Heatmap de la Matriz Sináptica
ax2 = fig.add_subplot(gs[0, 1])
# Usar conductancia normalizada (1/R) para que azul = fuerte
G_norm = (1.0 / R_matrix) / (1.0 / 100.0)  # Normalizar respecto a R_on
im = ax2.imshow(G_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(['Out₀', 'Out₁', 'Out₂'])
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(['In₀', 'In₁', 'In₂'])
ax2.set_title("② Matriz Sináptica (Crossbar 3×3)\nVerde = Fuerte | Rojo = Débil", fontweight='bold')

# Añadir valores de R en cada celda
for i in range(3):
    for j in range(3):
        r_val = R_matrix[i, j]
        if r_val < 10000:
            label = f"{r_val/1000:.1f}k"
        else:
            label = f"{r_val/1000:.0f}k"
        text_color = 'white' if G_norm[i, j] > 0.5 else 'black'
        ax2.text(j, i, label, ha='center', va='center', fontweight='bold', color=text_color, fontsize=10)

cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Conductancia Normalizada', fontsize=9)

# Panel 3: Voltaje de Entrada
ax3 = fig.add_subplot(gs[0, 2])
for i in range(3):
    ax3.plot(t_ms, Vc_in[i], color=colors_in[i], alpha=0.7, lw=1, label=labels_in[i])
ax3.set_ylabel("$V_c$ (V)")
ax3.set_xlabel("Tiempo (ms)")
ax3.set_title("③ Voltaje Interno Neuronas de Entrada", fontweight='bold')
ax3.grid(True, ls=':', alpha=0.5)
ax3.legend(fontsize=7)
ax3.set_xlim([0, T*1e3])

# Panel 4: Corriente Sináptica Colectiva
ax4 = fig.add_subplot(gs[1, 0])
for j in range(3):
    ax4.plot(t_ms, I_total_out[j]*1e6, color=colors_out[j], lw=1.5, label=labels_out[j])
ax4.set_ylabel(r"$I_{total}$ ($\mu$A)")
ax4.set_xlabel("Tiempo (ms)")
ax4.set_title("④ Corriente Colectiva por Neurona de Salida", fontweight='bold')
ax4.grid(True, ls=':', alpha=0.5)
ax4.legend(fontsize=7)
ax4.set_xlim([0, T*1e3])

# Panel 5: Voltaje de Salida
ax5 = fig.add_subplot(gs[1, 1])
for j in range(3):
    ax5.plot(t_ms, Vc_out[j], color=colors_out[j], lw=1.5, label=labels_out[j])
ax5.axhline(0.95, color='gray', ls='--', lw=1, label='$V_{th}$')
ax5.set_ylabel("$V_c$ (V)")
ax5.set_xlabel("Tiempo (ms)")
ax5.set_title("⑤ Integración Neuronal de Salida", fontweight='bold')
ax5.grid(True, ls=':', alpha=0.5)
ax5.legend(fontsize=7)
ax5.set_xlim([0, T*1e3])

# Panel 6: Raster Plot de Salida
ax6 = fig.add_subplot(gs[1, 2])
for j in range(3):
    n_sp = len(spikes_out[j])
    if n_sp > 0:
        ax6.scatter(spikes_out[j], np.full_like(spikes_out[j], j),
                   marker='|', s=300, lw=3, color=colors_out[j], label=f"{labels_out[j]}: {n_sp} spikes")
    else:
        ax6.scatter([], [], marker='|', s=300, lw=3, color=colors_out[j], label=f"{labels_out[j]}: 0 spikes")
ax6.set_yticks([0, 1, 2])
ax6.set_yticklabels(['Out₀', 'Out₁', 'Out₂'])
ax6.set_xlabel("Tiempo (ms)")
ax6.set_title("⑥ Actividad Colectiva de Salida\n(Raster Plot)", fontweight='bold')
ax6.set_xlim([0, T*1e3])
ax6.grid(True, ls=':', alpha=0.5)
ax6.legend(loc='upper right', fontsize=7)

fig.suptitle(
    "PASO 17 — RED NEUROMÓRFICA COLECTIVA (Crossbar 3×3)\n"
    "3 Neuronas de Entrada + 9 Sinapsis de Memristor + 3 Neuronas de Salida",
    fontsize=15, fontweight='bold', y=0.96
)

out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso17_red_crossbar.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[PNG] Guardado: {out_path}")
print(f"\nResumen de Actividad:")
for j in range(3):
    print(f"  {labels_out[j]}: {len(spikes_out[j])} spikes")
