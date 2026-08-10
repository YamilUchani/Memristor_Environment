"""
paso26_replicacion_prezioso_s6.py
==================================
Script para replicar la Figura S6: Evolución de la conductancia bajo trenes
de 500 pulsos de amplitud fija para las polaridades SET y RESET.
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.stochastic_model import StochasticMemristor, C2CConfig
from memristor_simulator.config.parameters import StrukovParameters

def simulate_pulse_train(v_amplitude, n_pulses=500, pulse_width=500e-6, dt=10e-6, 
                         init_cond_us=50.0, v_th=1.0, state_noise_sigma=0.005):
    """
    Simula un tren de pulsos de amplitud fija y extrae la conductancia tras cada pulso.
    """
    np.random.seed(int(abs(v_amplitude)*1000)) # Semilla única para cada curva
    
    # Parámetros calibrados
    base_params = StrukovParameters(
        R_on=10e3,
        R_off=200e3,
        D=10e-9,
        mu_v=2.5e-14, 
        enable_nonlinear_drift=True
    )
    
    # Configuramos el umbral físico
    base_params.v_th_mem = 0.0 # Desactivamos el hard-threshold nativo porque usamos el codo continuo
    
    # Configuración de ruido ciclo a ciclo
    cfg = C2CConfig(enabled=True, r_on_cv=0.1, r_off_cv=0.1, state_noise_sigma=state_noise_sigma)
    
    mem = StochasticMemristor(base_params, cfg)
    
    # Ajustar w_init para que coincida con la conductancia inicial deseada (en uS)
    target_r = 1.0 / (init_cond_us * 1e-6)
    w_calc = (base_params.R_off - target_r) / (base_params.R_off - base_params.R_on)
    mem.params.w_init = np.clip(w_calc, 0.01, 0.99)
    mem.x = mem.params.w_init
    mem._recalculate_resistance()
    
    n_steps_pulse = int(pulse_width / dt)
    conductances = []
    
    # Física experimental: Movilidad depende exponencialmente del voltaje
    actual_mu = base_params.mu_v * (abs(v_amplitude) / v_th)**15
    
    for p in range(n_pulses):
        # Inyectar variabilidad interna C2C
        mem._apply_c2c_variability()
        mem.params.mu_v = actual_mu # Aplicar movilidad no-lineal al ciclo
        
        # Aplicar el pulso de voltaje
        for _ in range(n_steps_pulse):
            mem.step(v_amplitude, dt, detect_cycle=False)
            
        # Añadir ruido extra a la variable de estado para simular la dispersión visual del paper
        if state_noise_sigma > 0:
            mem.x = np.clip(mem.x + np.random.normal(0, 0.002), 0.0, 1.0)
            mem._recalculate_resistance()
        
        # Lectura con ruido realista que mantenga las bandas separadas pero con dispersión visible
        g_read = (1.0 / mem.resistance) + np.random.normal(0, 5e-6)
        conductances.append(max(g_read, 1e-7) * 1e6) # Guardar en micro-Siemens (uS), mínimo cercano a 0
        
    return np.array(conductances)

# ── 1. Simulación ─────────────────────────────────────────────────────────────
print("Simulando Panel (a) RESET...")
# Valores iniciales calibrados estrictamente con el texto detallado del usuario
reset_11 = simulate_pulse_train(-1.1, init_cond_us=70.0, v_th=1.6)
reset_12 = simulate_pulse_train(-1.2, init_cond_us=30.0, v_th=1.52)
reset_13 = simulate_pulse_train(-1.3, init_cond_us=18.0, v_th=1.45)

print("Simulando Panel (b) SET...")
set_11 = simulate_pulse_train(1.1, init_cond_us=38.0, v_th=1.66, state_noise_sigma=0.0)
set_12 = simulate_pulse_train(1.2, init_cond_us=55.0, v_th=1.93, state_noise_sigma=0.0)
set_13 = simulate_pulse_train(1.3, init_cond_us=62.0, v_th=1.95, state_noise_sigma=0.0)

# ── 2. Ploteo ────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(2, 1, figsize=(6, 9))
fig.patch.set_facecolor('white')

pulses = np.arange(1, 501)

# Función para la línea punteada (filtro Savitzky-Golay o Media Móvil)
def trendline(data):
    # Se usa un fit polinomial de bajo grado o filtro suave para el trend
    return savgol_filter(data, window_length=151, polyorder=2)

# --- Panel (a) RESET ---
axs[0].scatter(pulses, reset_11, color='none', edgecolor='lime', marker='s', s=15, label='- 1.1V')
axs[0].scatter(pulses, reset_12, color='none', edgecolor='red', marker='o', s=15, label='- 1.2V')
axs[0].scatter(pulses, reset_13, color='none', edgecolor='blue', marker='^', s=15, label='- 1.3V')

axs[0].plot(pulses, trendline(reset_11), color='black', linestyle='--', linewidth=2.5)
axs[0].plot(pulses, trendline(reset_12), color='black', linestyle='--', linewidth=2.5)
axs[0].plot(pulses, trendline(reset_13), color='black', linestyle='--', linewidth=2.5)

axs[0].set_title("(a)", fontsize=16, fontweight='bold', loc='left')
axs[0].set_xlabel("Pulse #", fontsize=14)
axs[0].set_ylabel("Conductance (S)", fontsize=14) # El Y tick format llevará el 'μ'
axs[0].set_xlim([-50, 550])
axs[0].set_ylim([0, 80])
axs[0].set_yticks(np.arange(0, 100, 20))
axs[0].set_yticklabels([f"{y}μ" if y > 0 else "0" for y in np.arange(0, 100, 20)])
axs[0].legend(loc='upper right', frameon=True, edgecolor='black', fontsize=11, handletextpad=0.1)
axs[0].tick_params(axis='both', direction='in', labelsize=12, top=True, right=True)

# --- Panel (b) SET ---
axs[1].scatter(pulses, set_11, color='none', edgecolor='blue', marker='o', s=15, label='1.1V')
axs[1].scatter(pulses, set_12, color='none', edgecolor='red', marker='o', s=15, label='1.2V')
axs[1].scatter(pulses, set_13, color='none', edgecolor='lime', marker='o', s=15, label='1.3V')

axs[1].plot(pulses, trendline(set_11), color='black', linestyle='--', linewidth=2.5)
axs[1].plot(pulses, trendline(set_12), color='black', linestyle='--', linewidth=2.5)
axs[1].plot(pulses, trendline(set_13), color='black', linestyle='--', linewidth=2.5)

axs[1].set_title("(b)", fontsize=16, fontweight='bold', loc='left')
axs[1].set_xlabel("Pulse #", fontsize=14)
axs[1].set_ylabel("Conductance (S)", fontsize=14)
axs[1].set_xlim([-50, 550])
axs[1].set_ylim([10, 110])
axs[1].set_yticks(np.arange(20, 120, 20))
axs[1].set_yticklabels([f"{y}μ" for y in np.arange(20, 120, 20)])

# Legend custom position como en el paper
axs[1].legend(loc='upper left', bbox_to_anchor=(0.1, 0.95), frameon=True, edgecolor='black', fontsize=11, handletextpad=0.1)
axs[1].tick_params(axis='both', direction='in', labelsize=12, top=True, right=True)

# Estilo global de bordes
for ax in axs:
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "paso26_replicacion_s6.png")
plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[PNG] Imagen guardada en: {png_path}")
