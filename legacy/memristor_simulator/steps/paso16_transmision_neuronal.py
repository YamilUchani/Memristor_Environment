"""
paso16_transmision_neuronal.py
==============================
Demuestra la transmisión de señales entre dos neuronas conectadas
mediante un memristor. (OBJ-3: Red básica de 2 neuronas + sinapsis).

Arquitectura:
[Estímulo Constante] -> Neurona Pre (N1) -> [Memristor] -> Neurona Post (N2)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.lif_neuron    import default_lif_params, LIFNeuron
from memristor_simulator.models.strukov_model import StrukovMemristor
from memristor_simulator.config.parameters import StrukovParameters

dt = 0.05e-3
T = 0.5 # 500 ms
n = int(T / dt)
t = np.arange(n) * dt
t_ms = t * 1e3

def simulate_network(w_init_mem, title):
    """
    Simula una red simple de 2 neuronas.
    w_init_mem define el estado fijo del memristor (0.01 = HRS, 1.0 = LRS).
    """
    p_lif_pre = default_lif_params()
    p_lif_post = default_lif_params()
    p_lif_post.R_s = 100e3 # Menor resistencia de entrada para la N2
    
    # Memristor con plasticidad congelada (mu_v = 0) para evaluar solo la transmision
    p_mem = StrukovParameters(
        R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=0.0, w_init=w_init_mem
    )
    
    n_pre = LIFNeuron(p_lif_pre)
    n_post = LIFNeuron(p_lif_post)
    mem = StrukovMemristor(p_mem)
    
    # Almacenamiento de variables
    V_c_pre = np.zeros(n)
    V_out_pre = np.zeros(n)
    I_syn_arr = np.zeros(n)
    V_c_post = np.zeros(n)
    w_post_arr = np.zeros(n)
    
    # Generador de V_out_pre basado en los disparos de n_pre
    spike_timer_pre = 0.0
    spike_duration = 2e-3 # 2 ms ancho del spike
    spike_voltage = 5.0   # 5 Voltios de amplitud del spike
    
    for k in range(n):
        # --- 1. NEURONA PRE-SINÁPTICA (N1) ---
        # Recibe un estimulo constante de 1.8V para obligarla a disparar a ~20Hz
        V_stim = 1.8 
        I_pre = (V_stim - n_pre.V) / p_lif_pre.R_s
        V_eff_pre = n_pre.V + I_pre * p_lif_pre.R_s
        n_pre.step(V_eff_pre, dt)
        V_c_pre[k] = n_pre.V
        
        # Deteccion de disparo y emision del Spike Fisico
        if n_pre.w >= 0.5 and spike_timer_pre <= 0:
            spike_timer_pre = spike_duration
            
        if spike_timer_pre > 0:
            current_vout = spike_voltage
            spike_timer_pre -= dt
        else:
            current_vout = 0.0
            
        V_out_pre[k] = current_vout
        
        # --- 2. SINAPSIS (MEMRISTOR) ---
        R_m = mem.resistance
        # I_syn fluye desde V_out_pre hacia el capacitor de la N_post
        I_syn = (current_vout - n_post.V) / (R_m + p_lif_post.R_s)
        I_syn_arr[k] = I_syn
        
        # --- 3. NEURONA POST-SINÁPTICA (N2) ---
        V_eff_post = n_post.V + I_syn * p_lif_post.R_s
        n_post.step(V_eff_post, dt)
        
        V_c_post[k] = n_post.V
        w_post_arr[k] = n_post.w

    # Detectar spikes de N2 para contarlos
    sp_edges = np.diff((w_post_arr >= 0.5).astype(int)) > 0
    spikes_count = np.sum(sp_edges)

    return V_c_pre, V_out_pre, I_syn_arr, V_c_post, spikes_count

# Simular ambos casos
# Caso A: HRS (Conexion Debil)
vc_pre1, vout_pre1, isyn1, vc_post1, spikes1 = simulate_network(0.01, "HRS")

# Caso B: LRS (Conexion Fuerte)
vc_pre2, vout_pre2, isyn2, vc_post2, spikes2 = simulate_network(1.0, "LRS")


# Graficar
fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.2, top=0.9)

def plot_col(col_idx, vc_pre, vout_pre, isyn, vc_post, spikes, title_text, color_theme):
    # Neurona PRE
    ax1 = fig.add_subplot(gs[0, col_idx])
    ax1.set_title(title_text, fontsize=14, fontweight='bold')
    ax1.plot(t_ms, vc_pre, color='gray', label='$V_{c1}$ (Interno)', alpha=0.5)
    ax1.plot(t_ms, vout_pre, color='blue', label='$V_{out1}$ (Spikes)')
    ax1.set_ylabel("Voltaje Pre (V)")
    ax1.grid(True, ls=':', alpha=0.6)
    ax1.legend(loc='upper right')
    ax1.set_ylim([-0.2, 5.5])
    
    # Corriente Sinaptica
    ax2 = fig.add_subplot(gs[1, col_idx])
    ax2.plot(t_ms, isyn * 1e6, color='purple', lw=2)
    ax2.set_ylabel("Corriente $I_{syn}$ ($\mu$A)")
    ax2.grid(True, ls=':', alpha=0.6)
    # Uniformizar ejes Y para comparación justa
    ax2.set_ylim([-2, 55]) 
    
    # Neurona POST
    ax3 = fig.add_subplot(gs[2, col_idx])
    ax3.plot(t_ms, vc_post, color=color_theme, lw=2, label='$V_{c2}$')
    ax3.axhline(0.95, color='gray', ls='--', label='Umbral ($V_{th}$)')
    
    # Añadir lineas verticales para los spikes esteticos
    # Buscamos los picos
    sp_idx = np.where(np.diff((vc_post < 0.2).astype(int)) < 0)[0] # cuando cae a 0
    # Dibujamos spikes artificiales para visualizacion
    ax3.set_ylim([-0.1, 1.4])
    for s in sp_idx:
        if vc_post[s-1] > 0.8: # Era un disparo
            ax3.plot([t_ms[s], t_ms[s]], [vc_post[s-1], 1.25], color='black', lw=2)
    
    ax3.set_ylabel("Voltaje Post (V)")
    ax3.set_xlabel("Tiempo (ms)")
    ax3.grid(True, ls=':', alpha=0.6)
    ax3.text(250, 1.15, f"Total Spikes Post: {spikes}", color=color_theme, fontweight='bold', ha='center', fontsize=12)
    ax3.legend(loc='upper left')

plot_col(0, vc_pre1, vout_pre1, isyn1, vc_post1, spikes1, "Caso A: Memristor en Alta Resistencia (HRS = Conexión Rota)", "red")
plot_col(1, vc_pre2, vout_pre2, isyn2, vc_post2, spikes2, "Caso B: Memristor en Baja Resistencia (LRS = Conexión Fuerte)", "green")

fig.suptitle("PASO 16 — TRANSMISIÓN NEURONAL A TRAVÉS DE SINAPSIS ARTIFICIAL", fontsize=16, fontweight='bold', y=0.97)

out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso16_transmision.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[PNG] Guardado: {out_path}")
