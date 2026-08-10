"""
paso18_crossbar_escalable.py
============================
Arquitectura Neuromórfica Escalable: Crossbar 8x4
8 Neuronas de Entrada + 32 Sinapsis de Memristor + 4 Neuronas de Salida = 44 elementos.

Demuestra:
- Escalabilidad de la arquitectura Crossbar.
- Operacion Multiplicacion-Acumulacion (MAC) en hardware: I_j = Sum(V_i * G_ij)
- Clasificacion de patrones espaciales con pesos preconfigurados.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.lif_neuron import default_lif_params, LIFNeuron
from memristor_simulator.models.strukov_model import StrukovMemristor
from memristor_simulator.config.parameters import StrukovParameters

# ── Configuracion ─────────────────────────────────────────────────────────────
dt   = 0.05e-3
T    = 0.4          # 400 ms
n    = int(T / dt)
t    = np.arange(n) * dt
t_ms = t * 1e3

N_IN  = 8   # Filas del crossbar (neuronas de entrada)
N_OUT = 4   # Columnas del crossbar (neuronas de salida)

# ── Definir 2 patrones de entrada (como pixeles de una imagen 8x1) ────────────
# PatronA: Actividad en la mitad superior (In0-In3 fuertes, In4-In7 silenciosas)
# PatronB: Actividad en la mitad inferior (In0-In3 silenciosas, In4-In7 fuertes)

stim_A = np.array([2.5, 2.3, 2.4, 2.2, 0.3, 0.3, 0.3, 0.3])  # Mitad superior activa
stim_B = np.array([0.3, 0.3, 0.3, 0.3, 2.5, 2.3, 2.4, 2.2])  # Mitad inferior activa

# ── Matriz de pesos sinapticos (8x4) ─────────────────────────────────────────
# Diseñada para que:
#   Out0: Detector de Patron A (pesos fuertes en filas 0-3)
#   Out1: Detector de Patron B (pesos fuertes en filas 4-7)
#   Out2: Detector de actividad global (pesos moderados en todas las filas)
#   Out3: Inhibida (todos los pesos debiles)

W = np.array([
    # Out0    Out1    Out2    Out3
    [1.00,   0.01,   0.50,   0.02],  # In0
    [0.95,   0.01,   0.50,   0.02],  # In1
    [0.90,   0.01,   0.50,   0.02],  # In2
    [0.85,   0.01,   0.50,   0.02],  # In3
    [0.01,   1.00,   0.50,   0.02],  # In4
    [0.01,   0.95,   0.50,   0.02],  # In5
    [0.01,   0.90,   0.50,   0.02],  # In6
    [0.01,   0.85,   0.50,   0.02],  # In7
])

# ── Funcion de simulacion para un patron dado ────────────────────────────────
def simulate_crossbar(stim_vector, pattern_name):
    """
    Simula el crossbar completo para un vector de estimulo dado.
    Retorna arrays de voltaje, corrientes y spikes.
    """
    # Neuronas de entrada
    p_in = default_lif_params()
    p_in.R_s = 80e3
    neurons_in = [LIFNeuron(p_in) for _ in range(N_IN)]
    
    # Neuronas de salida
    p_out = default_lif_params()
    p_out.R_s = 50e3  # Rs bajo para que sean sensibles a la suma colectiva
    neurons_out = [LIFNeuron(p_out) for _ in range(N_OUT)]
    
    # Memristores (8x4)
    memristors = []
    for i in range(N_IN):
        row = []
        for j in range(N_OUT):
            p_mem = StrukovParameters(
                R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=0.0,
                w_init=W[i, j]
            )
            row.append(StrukovMemristor(p_mem))
        memristors.append(row)
    
    # Almacenamiento
    Vc_in   = np.zeros((N_IN, n))
    w_in    = np.zeros((N_IN, n))
    V_out_in = np.zeros((N_IN, n))
    Vc_out  = np.zeros((N_OUT, n))
    w_out   = np.zeros((N_OUT, n))
    I_out   = np.zeros((N_OUT, n))
    
    spike_timers = np.zeros(N_IN)
    spike_dur = 2e-3
    spike_amp = 5.0
    
    for k in range(n):
        # --- Neuronas de entrada ---
        for i in range(N_IN):
            neurons_in[i].step(stim_vector[i], dt)
            Vc_in[i, k] = neurons_in[i].V
            w_in[i, k] = neurons_in[i].w
            
            if neurons_in[i].w >= 0.5 and spike_timers[i] <= 0:
                spike_timers[i] = spike_dur
            
            if spike_timers[i] > 0:
                V_out_in[i, k] = spike_amp
                spike_timers[i] -= dt
            else:
                V_out_in[i, k] = 0.0
        
        # --- Corrientes sinapticas (MAC: Multiply-Accumulate) ---
        for j in range(N_OUT):
            I_total = 0.0
            for i in range(N_IN):
                R_m = memristors[i][j].resistance
                I_syn = (V_out_in[i, k] - neurons_out[j].V) / (R_m + p_out.R_s)
                I_total += I_syn
            
            I_out[j, k] = I_total
            V_eff = neurons_out[j].V + I_total * p_out.R_s
            neurons_out[j].step(V_eff, dt)
            Vc_out[j, k] = neurons_out[j].V
            w_out[j, k] = neurons_out[j].w
    
    # Detectar spikes
    def get_spikes(w_arr):
        edges = np.diff((w_arr >= 0.5).astype(int)) > 0
        return t_ms[1:][edges]
    
    spikes_in  = [get_spikes(w_in[i]) for i in range(N_IN)]
    spikes_out = [get_spikes(w_out[j]) for j in range(N_OUT)]
    
    # Resistencias
    R_mat = np.array([[memristors[i][j].resistance for j in range(N_OUT)] for i in range(N_IN)])
    
    return spikes_in, spikes_out, Vc_out, I_out, R_mat

# ── Ejecutar ambos patrones ───────────────────────────────────────────────────
print("Simulando Patron A (mitad superior activa)...")
sp_in_A, sp_out_A, Vc_out_A, I_out_A, R_mat = simulate_crossbar(stim_A, "A")

print("Simulando Patron B (mitad inferior activa)...")
sp_in_B, sp_out_B, Vc_out_B, I_out_B, _     = simulate_crossbar(stim_B, "B")

# ── Grafica ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(3, 3, hspace=0.5, wspace=0.35, top=0.9, bottom=0.06, left=0.06, right=0.97)

colors_out = ['#D32F2F', '#1565C0', '#2E7D32', '#757575']
labels_out = ['Out0 (Detector A)', 'Out1 (Detector B)', 'Out2 (Global)', 'Out3 (Inhibida)']
colors_in  = plt.cm.viridis(np.linspace(0.1, 0.9, N_IN))

# --- Panel 1: Heatmap de la Matriz Sinaptica (8x4) ---
ax_mat = fig.add_subplot(gs[0, 0])
G_norm = (1.0 / R_mat) / (1.0 / 100.0)
im = ax_mat.imshow(G_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax_mat.set_xticks(range(N_OUT))
ax_mat.set_xticklabels(['Out0\nDetect A', 'Out1\nDetect B', 'Out2\nGlobal', 'Out3\nInhib.'], fontsize=8)
ax_mat.set_yticks(range(N_IN))
ax_mat.set_yticklabels([f'In{i}' for i in range(N_IN)])
ax_mat.set_title("Matriz Sinaptica Crossbar 8x4\n(32 Memristores)", fontweight='bold', fontsize=11)
for i in range(N_IN):
    for j in range(N_OUT):
        r = R_mat[i, j]
        lbl = f"{r/1000:.0f}k" if r > 5000 else f"{r/1000:.1f}k"
        tc = 'white' if G_norm[i, j] > 0.4 else 'black'
        ax_mat.text(j, i, lbl, ha='center', va='center', fontweight='bold', color=tc, fontsize=8)
cbar = plt.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
cbar.set_label('Conductancia Norm.', fontsize=8)

# --- Panel 2: Raster de Entrada - Patron A ---
ax_rA = fig.add_subplot(gs[0, 1])
for i in range(N_IN):
    if len(sp_in_A[i]) > 0:
        ax_rA.scatter(sp_in_A[i], np.full_like(sp_in_A[i], i),
                     marker='|', s=150, lw=2, color=colors_in[i])
ax_rA.set_yticks(range(N_IN))
ax_rA.set_yticklabels([f'In{i}' for i in range(N_IN)])
ax_rA.set_title("Entrada: Patron A\n(Mitad Superior Activa)", fontweight='bold', fontsize=11)
ax_rA.set_xlim([0, T*1e3]); ax_rA.grid(True, ls=':', alpha=0.5)
ax_rA.axhspan(-0.5, 3.5, color='#E3F2FD', alpha=0.3)
ax_rA.axhspan(3.5, 7.5, color='#FFF9C4', alpha=0.3)
ax_rA.set_xlabel("Tiempo (ms)")

# --- Panel 3: Raster de Entrada - Patron B ---
ax_rB = fig.add_subplot(gs[0, 2])
for i in range(N_IN):
    if len(sp_in_B[i]) > 0:
        ax_rB.scatter(sp_in_B[i], np.full_like(sp_in_B[i], i),
                     marker='|', s=150, lw=2, color=colors_in[i])
ax_rB.set_yticks(range(N_IN))
ax_rB.set_yticklabels([f'In{i}' for i in range(N_IN)])
ax_rB.set_title("Entrada: Patron B\n(Mitad Inferior Activa)", fontweight='bold', fontsize=11)
ax_rB.set_xlim([0, T*1e3]); ax_rB.grid(True, ls=':', alpha=0.5)
ax_rB.axhspan(-0.5, 3.5, color='#FFF9C4', alpha=0.3)
ax_rB.axhspan(3.5, 7.5, color='#E3F2FD', alpha=0.3)
ax_rB.set_xlabel("Tiempo (ms)")

# --- Panel 4: Voltaje de Salida - Patron A ---
ax_vA = fig.add_subplot(gs[1, 0:2])
for j in range(N_OUT):
    n_sp = len(sp_out_A[j])
    ax_vA.plot(t_ms, Vc_out_A[j], color=colors_out[j], lw=1.5,
              label=f"{labels_out[j]}: {n_sp} spikes")
ax_vA.axhline(0.95, color='gray', ls='--', lw=1)
ax_vA.set_ylabel("$V_c$ (V)")
ax_vA.set_xlabel("Tiempo (ms)")
ax_vA.set_title("Respuesta Neuronal a Patron A", fontweight='bold', fontsize=11)
ax_vA.grid(True, ls=':', alpha=0.5)
ax_vA.legend(loc='upper right', fontsize=8)
ax_vA.set_xlim([0, T*1e3])

# --- Panel 5: Raster de Salida - Patron A ---
ax_sA = fig.add_subplot(gs[1, 2])
for j in range(N_OUT):
    n_sp = len(sp_out_A[j])
    if n_sp > 0:
        ax_sA.scatter(sp_out_A[j], np.full_like(sp_out_A[j], j),
                     marker='|', s=300, lw=3, color=colors_out[j],
                     label=f"{labels_out[j]}: {n_sp}")
    else:
        ax_sA.scatter([], [], marker='|', color=colors_out[j],
                     label=f"{labels_out[j]}: 0")
ax_sA.set_yticks(range(N_OUT))
ax_sA.set_yticklabels(['Out0', 'Out1', 'Out2', 'Out3'])
ax_sA.set_title("Salida Patron A\n(Raster)", fontweight='bold', fontsize=11)
ax_sA.set_xlim([0, T*1e3]); ax_sA.grid(True, ls=':', alpha=0.5)
ax_sA.legend(loc='upper right', fontsize=7)
ax_sA.set_xlabel("Tiempo (ms)")

# --- Panel 6: Voltaje de Salida - Patron B ---
ax_vB = fig.add_subplot(gs[2, 0:2])
for j in range(N_OUT):
    n_sp = len(sp_out_B[j])
    ax_vB.plot(t_ms, Vc_out_B[j], color=colors_out[j], lw=1.5,
              label=f"{labels_out[j]}: {n_sp} spikes")
ax_vB.axhline(0.95, color='gray', ls='--', lw=1)
ax_vB.set_ylabel("$V_c$ (V)")
ax_vB.set_xlabel("Tiempo (ms)")
ax_vB.set_title("Respuesta Neuronal a Patron B", fontweight='bold', fontsize=11)
ax_vB.grid(True, ls=':', alpha=0.5)
ax_vB.legend(loc='upper right', fontsize=8)
ax_vB.set_xlim([0, T*1e3])

# --- Panel 7: Raster de Salida - Patron B ---
ax_sB = fig.add_subplot(gs[2, 2])
for j in range(N_OUT):
    n_sp = len(sp_out_B[j])
    if n_sp > 0:
        ax_sB.scatter(sp_out_B[j], np.full_like(sp_out_B[j], j),
                     marker='|', s=300, lw=3, color=colors_out[j],
                     label=f"{labels_out[j]}: {n_sp}")
    else:
        ax_sB.scatter([], [], marker='|', color=colors_out[j],
                     label=f"{labels_out[j]}: 0")
ax_sB.set_yticks(range(N_OUT))
ax_sB.set_yticklabels(['Out0', 'Out1', 'Out2', 'Out3'])
ax_sB.set_title("Salida Patron B\n(Raster)", fontweight='bold', fontsize=11)
ax_sB.set_xlim([0, T*1e3]); ax_sB.grid(True, ls=':', alpha=0.5)
ax_sB.legend(loc='upper right', fontsize=7)
ax_sB.set_xlabel("Tiempo (ms)")

fig.suptitle(
    "PASO 18 -- CROSSBAR ESCALABLE 8x4 (12 Neuronas + 32 Sinapsis)\n"
    "Clasificacion de Patrones Espaciales mediante Operacion MAC en Hardware",
    fontsize=14, fontweight='bold', y=0.97
)

out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso18_crossbar_escalable.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[PNG] Guardado: {out_path}")

# Resumen
print("\n=== RESUMEN DE CLASIFICACION ===")
print(f"Patron A -> Out0 (Detector A): {len(sp_out_A[0])} spikes | Out1 (Detector B): {len(sp_out_A[1])} spikes")
print(f"Patron B -> Out0 (Detector A): {len(sp_out_B[0])} spikes | Out1 (Detector B): {len(sp_out_B[1])} spikes")
winner_A = "Out0 (CORRECTO)" if len(sp_out_A[0]) > len(sp_out_A[1]) else "Out1 (ERROR)"
winner_B = "Out1 (CORRECTO)" if len(sp_out_B[1]) > len(sp_out_B[0]) else "Out0 (ERROR)"
print(f"Ganador Patron A: {winner_A}")
print(f"Ganador Patron B: {winner_B}")
