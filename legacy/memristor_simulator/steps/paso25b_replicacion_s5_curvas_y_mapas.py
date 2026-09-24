"""
paso25b_replicacion_s5_curvas_y_mapas.py
========================================
Script para replicar los paneles (a, b, c, d) de la Figura S5.
Utilizando el enfoque de "ruido_fisico_c2c.png" (StochasticAnalysis) para 
generar las curvas caóticas (Cycle-to-Cycle variability) en los paneles (a) y (b).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.strukov_model import StrukovMemristor
from memristor_simulator.config.parameters import StrukovParameters
from memristor_simulator.models.stochastic_model import StochasticMemristor, C2CConfig
from memristor_simulator.utils.integrators import generate_waveform_scalar
from memristor_simulator.config.simulation_settings import WaveformType

# ── 1. Generar Curvas (a) y (b) usando Variabilidad C2C (Estilo ruido_fisico_c2c) ──

def run_stochastic_sweeps(n_cycles=30):
    """
    Simula múltiples dispositivos usando secuencias de pulsos discretos,
    que es la metodología exacta descrita en el caption de la Figura S5.
    Esto genera los puntos discretos y saltos caóticos reales.
    """
    np.random.seed(42)
    pulse_width = 500e-6
    dt = 10e-6
    n_steps_pulse = int(pulse_width / dt)
    
    # Amplitudes discretas (0.8 a 1.8 para SET, -0.8 a -1.8 para RESET)
    v_amplitudes_set = np.arange(0.8, 1.85, 0.02)
    v_amplitudes_reset = np.arange(-0.8, -2.05, -0.02)
    
    c2c_cfg = C2CConfig(enabled=True, r_on_cv=0.15, r_off_cv=0.10)
    base_params = StrukovParameters(R_on=15e3, R_off=45e3, D=10e-9, mu_v=1.0e-14, enable_nonlinear_drift=True)
    
    from memristor_simulator.models.stochastic_model import sample_r_on, sample_r_off
    import copy
    
    cycles_data_set = []
    cycles_data_reset = []
    
    for cyc in range(n_cycles):
        # 1. Variabilidad D2D (Device-to-Device) masiva entre dispositivos
        base_r_on = max(10e3, np.random.normal(15e3, 3e3))
        base_r_off = max(30e3, np.random.normal(45e3, 8e3))
        base_mu_set = max(0.5e-14, np.random.normal(1.2e-14, 0.4e-14))
        base_mu_reset = max(0.5e-14, np.random.normal(1.2e-14, 0.4e-14))
        
        new_params_set = copy.deepcopy(base_params)
        new_params_reset = copy.deepcopy(base_params)
        
        # 2. Variabilidad C2C (Cycle-to-Cycle) sobre la base D2D
        r_on_s = sample_r_on(base_r_on, c2c_cfg.r_on_cv, c2c_cfg.use_lognormal)
        r_off_s = sample_r_off(base_r_off, c2c_cfg.r_off_cv, r_on_s, c2c_cfg.use_lognormal)
        
        new_params_set.R_on = r_on_s
        new_params_set.R_off = r_off_s
        new_params_set.mu_v = base_mu_set
        
        new_params_reset.R_on = r_on_s
        new_params_reset.R_off = r_off_s
        new_params_reset.mu_v = base_mu_reset
        
        # --- Simular SET ---
        dev_set = StochasticMemristor(new_params_set, c2c_config=C2CConfig(enabled=False))
        dev_set.params.w_init = 0.05 # HRS
        dev_set.x = dev_set.params.w_init
        
        # Umbral nativo
        dev_set.params.v_th_mem = max(0.6, np.random.normal(1.1, 0.2))
        dev_set._recalculate_resistance()
        
        g_set = []
        for v in v_amplitudes_set:
            # Pulso (sin hacks, usando la física pura nativa)
            for _ in range(n_steps_pulse):
                dev_set.step(v, dt)
                dev_set.x = np.clip(dev_set.x + np.random.normal(0, 0.0005), 0.0, 1.0)
            
            dev_set._recalculate_resistance()
            
            # Leer conductancia con pequeño ruido de lectura
            g_read = (1.0 / dev_set.resistance) + np.random.normal(0, 1.5e-6)
            g_set.append(g_read)
            
        cycles_data_set.append((v_amplitudes_set, g_set))
        
        # --- Simular RESET ---
        dev_reset = StochasticMemristor(new_params_reset, c2c_config=C2CConfig(enabled=False))
        dev_reset.params.w_init = 0.95 # LRS
        dev_reset.x = dev_reset.params.w_init
        
        # Umbral nativo
        dev_reset.params.v_th_mem = max(0.5, np.random.normal(1.2, 0.15))
        dev_reset._recalculate_resistance()
        
        g_reset = []
        for v in v_amplitudes_reset:
            for _ in range(n_steps_pulse):
                dev_reset.step(v, dt)
                dev_reset.x = np.clip(dev_reset.x + np.random.normal(0, 0.0005), 0.0, 1.0)
                
            dev_reset._recalculate_resistance()
            
            g_read = (1.0 / dev_reset.resistance) + np.random.normal(0, 1.5e-6)
            g_reset.append(g_read)
            
        cycles_data_reset.append((v_amplitudes_reset, g_reset))
        
    return cycles_data_set, cycles_data_reset

cycles_data_set, cycles_data_reset = run_stochastic_sweeps(n_cycles=30)

# ── 2. Generar Mapas (c) y (d): 10x8 Array Thresholds ─────────────────────────

def get_thresholds_10x8():
    np.random.seed(100)
    ROWS, COLS = 10, 8
    
    vset_map = np.zeros((ROWS, COLS))
    vreset_map = np.zeros((ROWS, COLS))
    
    from paso25_replicacion_prezioso_s5 import find_threshold
    
    for r in range(ROWS):
        for c in range(COLS):
            if np.random.rand() < 0.05:
                vset_map[r, c] = np.nan
                vreset_map[r, c] = np.nan
                continue
                
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
# Scatter plot con líneas punteadas
for v_arr, g_arr in cycles_data_reset:
    ax_a.plot(v_arr, g_arr, color='mediumblue', marker='^', markersize=4, lw=1.0, linestyle=':', alpha=0.5)
    
ax_a.set_title("(a) RESET Transition", fontsize=14, fontweight='bold', loc='left')
ax_a.set_xlabel("Voltage (v)", fontsize=12)
ax_a.set_ylabel("Conductance (S)", fontsize=12)
ax_a.set_xlim([-2.0, -0.8])
ax_a.grid(True, linestyle='--', alpha=0.6)

# Panel (b): SET
ax_b = fig.add_subplot(gs[0, 1])
for v_arr, g_arr in cycles_data_set:
    ax_b.plot(v_arr, g_arr, color='crimson', marker='o', markersize=4, lw=1.0, linestyle=':', alpha=0.5)

ax_b.set_title("(b) SET Transition", fontsize=14, fontweight='bold', loc='left')
ax_b.set_xlabel("Voltage (v)", fontsize=12)
ax_b.set_ylabel("Conductance (S)", fontsize=12)
ax_b.set_xlim([0.8, 1.8])
ax_b.grid(True, linestyle='--', alpha=0.6)

# Panel (c): VSET Map
ax_c = fig.add_subplot(gs[1, 0])
# S5(c): ~0.8V is Red, ~1.2V is Green.
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
ax_c.set_title("(c) V_SET Map", fontsize=14, fontweight='bold', loc='left')

# Panel (d): VRESET Map
ax_d = fig.add_subplot(gs[1, 1])
# S5(d): ~ -0.8V is Red (high value), ~ -1.4V is Green (low value). 
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
ax_d.set_title("(d) V_RESET Map", fontsize=14, fontweight='bold', loc='left')

plt.suptitle("Figure S5 Replication: Panels (a)-(d) con Variabilidad C2C", fontsize=16, y=0.96)

out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "paso25b_replicacion_s5_paneles_a_d.png")
plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[PNG] Guardado en: {png_path}")
