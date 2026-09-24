"""
paso15_curva_stdp.py
====================
Extracción de la Curva STDP (Spike-Timing-Dependent Plasticity)
Barre el valor de Delta t y grafica el cambio relativo de conductancia (Peso Sináptico).
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.strukov_model import StrukovMemristor, dxdt_strukov, window_biolek
from memristor_simulator.config.parameters import StrukovParameters

dt = 0.1e-3

def get_spike(t_array, start_time):
    v = np.zeros_like(t_array)
    v[(t_array >= start_time) & (t_array < start_time + 2e-3)] = -1.0
    v[(t_array >= start_time + 2e-3) & (t_array < start_time + 8e-3)] = 0.6
    return v

def calculate_dw(delta_t):
    """
    Calcula el cambio de conductancia Delta w para un delta_t dado.
    Aplica 5 pares de pulsos para acumular un cambio medible.
    """
    t = np.arange(0, 80e-3, dt)
    V_pre = np.zeros_like(t)
    V_post = np.zeros_like(t)
    
    # Tren de 5 pares de pulsos
    for i in range(5):
        t_pre = 10e-3 + i * 14e-3
        t_post = t_pre + delta_t
        V_pre += get_spike(t, t_pre)
        V_post += get_spike(t, t_post)
        
    V_M = V_pre - V_post
    
    p_mem = StrukovParameters(
        R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=6e-12,
        v_th_mem=1.2, # UMBRAL FÍSICO
        w_init=0.5, enable_nonlinear_drift=True, enable_hard_switching=True
    )
    mem = StrukovMemristor(p_mem)
    x_initial = mem.x
    
    for v in V_M:
        i_in = v / mem.resistance
        dxdt = dxdt_strukov(i_in, p_mem.R_on, p_mem.D, p_mem.mu_v, v_mem=v, v_th_mem=p_mem.v_th_mem)
        if p_mem.enable_nonlinear_drift: dxdt *= window_biolek(mem.x, p=5)
        mem.x = float(np.clip(mem.x + dxdt * dt, 0.0, 1.0))
        mem._recalculate_resistance()
        
    x_final = mem.x
    # Retornar el cambio absoluto del estado interno (Peso Sináptico w)
    return (x_final - x_initial) * 100.0

# ── Barrido de Delta t ────────────────────────────────────────────────────────
delta_t_vals = np.linspace(-15e-3, 15e-3, 61)
dw_vals = []

print("Calculando curva STDP... Esto tomará unos segundos.")
for dt_val in delta_t_vals:
    dw = calculate_dw(dt_val)
    dw_vals.append(dw)

dt_ms = delta_t_vals * 1000.0

# ── Generar Gráfica ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_facecolor('white')

# Simulated STDP curve (from memristor dynamics)
ax.plot(dt_ms, dw_vals, marker='o', linestyle='-', color='#2c3e50', linewidth=2, markersize=6, label='Simulación')

# Theoretical exponential STDP model (biologically inspired)
A_plus = 0.5   # % potentiation max (LTP)
A_minus = 0.3  # % depression max (LTD)
tau_plus = 10.0  # ms decay constant for LTP
tau_minus = 10.0  # ms decay constant for LTD
dw_theor = np.where(dt_ms > 0,
                    A_plus * np.exp(-dt_ms / tau_plus),
                    -A_minus * np.exp(dt_ms / tau_minus))
ax.plot(dt_ms, dw_theor, '--', color='#e67e22', linewidth=2, label='Modelo Exponencial')

# Reference lines
ax.axhline(0, color='gray', linestyle='--')
ax.axvline(0, color='gray', linestyle='--')

# Shading for LTP/LTD regions based on simulated data
ax.fill_between(dt_ms, dw_vals, 0, where=(np.array(dw_vals) > 0), color='#2ecc71', alpha=0.3, label='LTP (Sim.)')
ax.fill_between(dt_ms, dw_vals, 0, where=(np.array(dw_vals) < 0), color='#e74c3c', alpha=0.3, label='LTD (Sim.)')

ax.set_title("Curva de Aprendizaje STDP (Spike‑Timing‑Dependent Plasticity)", fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel("$\\Delta t = t_{post} - t_{pre}$ (ms)", fontsize=13)
ax.set_ylabel("Cambio de Conductancia Sináptica $\\Delta w$ (%)", fontsize=13)
ax.grid(True, ls=':', alpha=0.7)
ax.legend(fontsize=12)

# Anotaciones
ax.annotate('Pre precede a Post\n(Fortalece Sinapsis)', xy=(5, np.max(dw_vals)*0.8), xytext=(8, np.max(dw_vals)*0.9),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), ha='center', color='green', fontweight='bold')
ax.annotate('Post precede a Pre\n(Debilita Sinapsis)', xy=(-5, np.min(dw_vals)*0.8), xytext=(-8, np.min(dw_vals)*0.9),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), ha='center', color='red', fontweight='bold')

out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso15_curva_stdp.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[PNG] Guardado: {out_path}")
