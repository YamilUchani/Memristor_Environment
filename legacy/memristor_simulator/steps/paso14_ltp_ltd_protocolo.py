"""
paso14_ltp_ltd_protocolo.py
===========================
Demostración de Potenciación a Largo Plazo (LTP) y Depresión a Largo Plazo (LTD)
utilizando pulsos pre y post-sinápticos con un modelo de memristor con umbral.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.strukov_model import StrukovMemristor, dxdt_strukov, window_biolek
from memristor_simulator.config.parameters import StrukovParameters

dt = 0.1e-3

# Definir la forma del Spike (Pulso asimétrico o bifásico)
# Un pulso bifásico común para STDP:
# Parte positiva: +1.2V de 0 a 2ms
# Parte negativa: -1.2V de 2 a 4ms
def get_spike(t_array, start_time):
    v = np.zeros_like(t_array)
    # Spike pre-sinaptico: +1.2V, luego -1.2V
    idx_pos = (t_array >= start_time) & (t_array < start_time + 2e-3)
    idx_neg = (t_array >= start_time + 2e-3) & (t_array < start_time + 4e-3)
    v[idx_pos] = 1.2
    v[idx_neg] = -1.2
    return v

def run_protocol(delta_t, title):
    # Simular 30 ms
    t = np.arange(0, 30e-3, dt)
    
    # 5 pares de pulsos separados por 5 ms
    V_pre = np.zeros_like(t)
    V_post = np.zeros_like(t)
    
    # Tren de 3 pares de pulsos para observar la acumulacion
    for i in range(3):
        t_pre = 2e-3 + i * 8e-3
        t_post = t_pre + delta_t
        V_pre += get_spike(t, t_pre)
        # Para V_post invertimos la polaridad para lograr la regla de Hebb
        # o usamos la misma polaridad. Usar misma polaridad requiere V_M = V_pre - V_post.
        # Si V_pre es +1.2 y V_post es -1.2 superpuestos, V_M = +2.4V
        V_post += get_spike(t, t_post)
        
    V_M = V_pre - V_post
    
    p_mem = StrukovParameters(
        R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=5e-12,
        v_th_mem=1.5, # UMBRAL DE VOLTAJE (solo aprende si |Vm| > 1.5V)
        w_init=0.5, enable_nonlinear_drift=True, enable_hard_switching=True
    )
    mem = StrukovMemristor(p_mem)
    
    R_history = np.zeros_like(t)
    x_history = np.zeros_like(t)
    
    for k in range(len(t)):
        v = V_M[k]
        i_in = v / mem.resistance
        # Solo calculamos deriva si supera el umbral
        dxdt = dxdt_strukov(i_in, p_mem.R_on, p_mem.D, p_mem.mu_v, v_mem=v, v_th_mem=p_mem.v_th_mem)
        if p_mem.enable_nonlinear_drift: dxdt *= window_biolek(mem.x, p=5)
        
        mem.x = float(np.clip(mem.x + dxdt * dt, 0.0, 1.0))
        mem._recalculate_resistance()
        
        R_history[k] = mem.resistance
        x_history[k] = mem.x
        
    return t, V_pre, V_post, V_M, R_history

# Ejecutar LTP (Pre precede a Post, dt = +2ms)
# Vpre(t) tiene cola negativa de 2 a 4ms. Vpost(t) tiene pico positivo de 2 a 4ms.
# VM = Vpre - Vpost = (-1.2V) - (+1.2V) = -2.4V. (Deriva negativa -> LTD)
# Para que Pre antes que Post de LTP, necesitamos que VM sea positivo!
# Por tanto, si delta_t > 0, VM = Vpre(t) - Vpost(t) debe ser > +1.5V.
# Esto ocurre si la cola de Vpre es POSITIVA o la cabeza de Vpost es NEGATIVA.
# Vamos a usar pulsos inversos:
def get_spike(t_array, start_time):
    v = np.zeros_like(t_array)
    # Cabeza negativa, cola positiva (forma clásica para STDP)
    v[(t_array >= start_time) & (t_array < start_time + 2e-3)] = -1.0
    v[(t_array >= start_time + 2e-3) & (t_array < start_time + 8e-3)] = 0.6
    return v

def run_protocol_stdp(delta_t):
    t = np.arange(0, 40e-3, dt)
    V_pre = np.zeros_like(t)
    V_post = np.zeros_like(t)
    
    for i in range(3):
        t_pre = 5e-3 + i * 12e-3
        t_post = t_pre + delta_t
        V_pre += get_spike(t, t_pre)
        V_post += get_spike(t, t_post)
        
    V_M = V_pre - V_post
    
    p_mem = StrukovParameters(
        R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=5e-12,
        v_th_mem=1.2, # UMBRAL DE VOLTAJE (solo aprende si |Vm| > 1.2V)
        w_init=0.5, enable_nonlinear_drift=True, enable_hard_switching=True
    )
    mem = StrukovMemristor(p_mem)
    R_history = np.zeros_like(t)
    
    for k in range(len(t)):
        v = V_M[k]
        i_in = v / mem.resistance
        dxdt = dxdt_strukov(i_in, p_mem.R_on, p_mem.D, p_mem.mu_v, v_mem=v, v_th_mem=p_mem.v_th_mem)
        if p_mem.enable_nonlinear_drift: dxdt *= window_biolek(mem.x, p=5)
        mem.x = float(np.clip(mem.x + dxdt * dt, 0.0, 1.0))
        mem._recalculate_resistance()
        R_history[k] = mem.resistance
        
    return t, V_pre, V_post, V_M, R_history

# Protocolo LTP: Post dispara justo DESPUES de Pre (+3 ms)
t_ltp, pre_ltp, post_ltp, vm_ltp, R_ltp = run_protocol_stdp(3.0e-3)

# Protocolo LTD: Post dispara justo ANTES de Pre (-3 ms)
t_ltd, pre_ltd, post_ltd, vm_ltd, R_ltd = run_protocol_stdp(-3.0e-3)

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(4, 2, hspace=0.6, wspace=0.3, top=0.9)

def plot_col(col_idx, t, pre, post, vm, R, title, color_r, r_label):
    ax1 = fig.add_subplot(gs[0, col_idx])
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.plot(t*1000, pre, color='blue', label='$V_{pre}$')
    ax1.plot(t*1000, post, color='red', label='$V_{post}$', ls='--')
    ax1.set_ylabel("Voltaje (V)")
    ax1.grid(True, ls=':', alpha=0.6)
    ax1.legend(loc='upper right')
    
    ax2 = fig.add_subplot(gs[1, col_idx])
    ax2.plot(t*1000, vm, color='purple', label='$V_M = V_{pre} - V_{post}$')
    ax2.axhline(1.2, color='gray', ls='--', label='$+V_{th}$')
    ax2.axhline(-1.2, color='gray', ls='--', label='$-V_{th}$')
    # Sombrear areas que superan umbral
    ax2.fill_between(t*1000, vm, 1.2, where=(vm > 1.2), color='green', alpha=0.3, label='Potenciación')
    ax2.fill_between(t*1000, vm, -1.2, where=(vm < -1.2), color='red', alpha=0.3, label='Depresión')
    ax2.set_ylabel("$V_M$ (V)")
    ax2.grid(True, ls=':', alpha=0.6)
    if col_idx == 1: ax2.legend(loc='upper right')
    
    ax3 = fig.add_subplot(gs[2:, col_idx])
    ax3.plot(t*1000, R/1000, color=color_r, lw=3)
    ax3.set_ylabel("Resistencia $R_M$ (k$\Omega$)")
    ax3.set_xlabel("Tiempo (ms)")
    ax3.grid(True, ls=':', alpha=0.6)
    ax3.set_title(r_label, fontweight='bold', color=color_r)

plot_col(0, t_ltp, pre_ltp, post_ltp, vm_ltp, R_ltp, "LTP: Pre dispara antes que Post ($\Delta t > 0$)", 'green', "Resistencia Baja = Conexión más fuerte (LTP)")
plot_col(1, t_ltd, pre_ltd, post_ltd, vm_ltd, R_ltd, "LTD: Post dispara antes que Pre ($\Delta t < 0$)", 'red', "Resistencia Sube = Conexión más débil (LTD)")

fig.suptitle("PASO 14 — APRENDIZAJE SINÁPTICO (Plasticidad Hebbiana)", fontsize=16, fontweight='bold', y=0.97)

out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso14_ltp_ltd.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[PNG] Guardado: {out_path}")
