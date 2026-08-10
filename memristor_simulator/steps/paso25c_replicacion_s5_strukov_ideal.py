"""
paso25c_replicacion_s5_strukov_ideal.py
========================================
Script para replicar los paneles (a, b, c, d) de la Figura S5, pero
utilizando estrictamente el modelo ideal de Strukov (sin variabilidad C2C ni ruido).
Se aplica únicamente variabilidad D2D (Device-to-Device) para generar los
diferentes dispositivos del arreglo.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.strukov_model import StrukovMemristor
from memristor_simulator.config.parameters import StrukovParameters

# ── 1. Generar Curvas (a) y (b) usando Modelo Ideal (Strukov) ──────────

def run_ideal_sweeps(n_devices=3):
    """
    Simula pocos dispositivos sin variabilidad D2D para mostrar la 
    naturaleza pura, determinista y limpia del modelo ideal.
    """
    pulse_width = 500e-6
    dt = 10e-6
    n_steps_pulse = int(pulse_width / dt)
    
    v_amplitudes_set = np.arange(0.8, 1.85, 0.02)
    v_amplitudes_reset = np.arange(-0.8, -2.05, -0.02)
    
    cycles_data_set = []
    cycles_data_reset = []
    
    for cyc in range(n_devices):
        # Variabilidad D2D casi nula (Líneas limpias de libro de texto)
        r_on_s = 15e3 + (cyc * 500)  # Ligera separación visual
        r_off_s = 45e3 + (cyc * 1000)
        mu_set_s = 1.8e-14
        mu_reset_s = 1.8e-14
        
        # --- Simular SET ---
        p_set = StrukovParameters(R_on=r_on_s, R_off=r_off_s, D=10e-9, mu_v=mu_set_s, enable_nonlinear_drift=True)
        dev_set = StrukovMemristor(p_set)
        dev_set.params.w_init = 0.05 # HRS
        dev_set.x = dev_set.params.w_init
        dev_set._recalculate_resistance()
        
        g_set = []
        for v in v_amplitudes_set:
            # INTEGRACIÓN PURA STRUKOV (Sin el salto "abrupto" artificial)
            for _ in range(n_steps_pulse):
                dev_set.step(v, dt)
            
            dev_set._recalculate_resistance()
            g_read = 1.0 / dev_set.resistance
            g_set.append(g_read)
            
        cycles_data_set.append((v_amplitudes_set, g_set))
        
        # --- Simular RESET ---
        p_reset = StrukovParameters(R_on=r_on_s, R_off=r_off_s, D=10e-9, mu_v=mu_reset_s, enable_nonlinear_drift=True)
        dev_reset = StrukovMemristor(p_reset)
        dev_reset.params.w_init = 0.95 # LRS
        dev_reset.x = dev_reset.params.w_init
        dev_reset._recalculate_resistance()
        
        g_reset = []
        for v in v_amplitudes_reset:
            for _ in range(n_steps_pulse):
                dev_reset.step(v, dt)
                
            dev_reset._recalculate_resistance()
            g_read = 1.0 / dev_reset.resistance
            g_reset.append(g_read)
            
        cycles_data_reset.append((v_amplitudes_reset, g_reset))
        
    return cycles_data_set, cycles_data_reset

cycles_data_set, cycles_data_reset = run_ideal_sweeps(n_devices=30)

# ── 2. Generar Mapas (c) y (d): 10x8 Array Thresholds ─────────────────────────

def get_thresholds_10x8():
    np.random.seed(200)
    ROWS, COLS = 10, 8
    
    vset_map = np.zeros((ROWS, COLS))
    vreset_map = np.zeros((ROWS, COLS))
    
    from paso25_replicacion_prezioso_s5 import find_threshold
    
    for r in range(ROWS):
        for c in range(COLS):
            r_off_sample = max(30e3, np.random.normal(50e3, 8e3))
            mu_set_sample = max(0.5e-14, np.random.normal(1.8e-14, 0.3e-14))
            mu_reset_sample = max(0.1e-14, np.random.normal(0.4e-14, 0.08e-14))
            
            p_set = StrukovParameters(R_on=2e3, R_off=r_off_sample, D=10e-9, mu_v=mu_set_sample, enable_nonlinear_drift=True)
            mem_set = StrukovMemristor(p_set)
            vset = find_threshold(mem_set, is_set=True)
            
            p_reset = StrukovParameters(R_on=2e3, R_off=r_off_sample, D=10e-9, mu_v=mu_reset_sample, enable_nonlinear_drift=True)
            mem_reset = StrukovMemristor(p_reset)
            vreset = find_threshold(mem_reset, is_set=False)
            
            vset_map[r, c] = vset if not np.isnan(vset) else np.nan
            vreset_map[r, c] = vreset if not np.isnan(vreset) else np.nan
            
    return vset_map, vreset_map

vset_map, vreset_map = get_thresholds_10x8()

# ── 3. Graficar ───────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.2)

# Panel (a): RESET
ax_a = fig.add_subplot(gs[0, 0])
for v_arr, g_arr in cycles_data_reset:
    # Usamos línea sólida pura para resaltar que es el modelo Ideal Determinista
    ax_a.plot(v_arr, g_arr, color='mediumblue', lw=1.5, alpha=0.6)
    
ax_a.set_title("(a) RESET Transition (Ideal Strukov)", fontsize=14, fontweight='bold', loc='left')
ax_a.set_xlabel("Voltage (v)", fontsize=12)
ax_a.set_ylabel("Conductance (S)", fontsize=12)
ax_a.set_xlim([-2.0, -0.8])
ax_a.grid(True, linestyle='--', alpha=0.6)

# Panel (b): SET
ax_b = fig.add_subplot(gs[0, 1])
for v_arr, g_arr in cycles_data_set:
    # Línea sólida pura
    ax_b.plot(v_arr, g_arr, color='crimson', lw=1.5, alpha=0.6)

ax_b.set_title("(b) SET Transition (Ideal Strukov)", fontsize=14, fontweight='bold', loc='left')
ax_b.set_xlabel("Voltage (v)", fontsize=12)
ax_b.set_ylabel("Conductance (S)", fontsize=12)
ax_b.set_xlim([0.8, 1.8])
ax_b.grid(True, linestyle='--', alpha=0.6)

# Panel (c): VSET Map
ax_c = fig.add_subplot(gs[1, 0])
cmap_set = plt.cm.RdYlGn
cmap_set.set_bad(color='lightblue')
im_c = ax_c.imshow(vset_map, cmap=cmap_set, aspect='auto')
for r in range(10):
    for c in range(8):
        val = vset_map[r, c]
        if np.isnan(val):
            ax_c.text(c, r, 'x', ha='center', va='center', color='black', fontweight='bold')
        else:
            ax_c.text(c, r, f"{val:.3g}", ha='center', va='center', color='black', fontsize=9)
ax_c.set_xticks([])
ax_c.set_yticks([])
ax_c.set_title("(c) V_SET Map (D2D Variability)", fontsize=14, fontweight='bold', loc='left')

# Panel (d): VRESET Map
ax_d = fig.add_subplot(gs[1, 1])
cmap_reset = plt.cm.RdYlGn_r
cmap_reset.set_bad(color='lightblue')
im_d = ax_d.imshow(vreset_map, cmap=cmap_reset, aspect='auto')
for r in range(10):
    for c in range(8):
        val = vreset_map[r, c]
        if np.isnan(val):
            ax_d.text(c, r, 'x', ha='center', va='center', color='black', fontweight='bold')
        else:
            ax_d.text(c, r, f"{val:.3g}", ha='center', va='center', color='black', fontsize=9)
ax_d.set_xticks([])
ax_d.set_yticks([])
ax_d.set_title("(d) V_RESET Map (D2D Variability)", fontsize=14, fontweight='bold', loc='left')

plt.suptitle("Figure S5 Replication: Strukov Ideal Model (Deterministic)", fontsize=16, y=0.96)

out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "paso25c_replicacion_s5_strukov_ideal.png")
plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[PNG] Guardado en: {png_path}")
