"""
paso21_caracterizacion_pulsos.py
================================
Script para replicar la Figura S6: Evolución de la conductancia bajo trenes 
de pulsos (500 µs) de amplitud constante.

Compara la modulación de peso (SET y RESET) para distintos voltajes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.strukov_model import StrukovMemristor
from memristor_simulator.config.parameters import StrukovParameters

dt = 0.05e-3
pulse_width = 500e-6
n_steps_per_pulse = int(pulse_width / dt)
num_pulses = 500

# Voltajes a evaluar según la Figura S6
voltages_set = [1.1, 1.2, 1.3]
voltages_reset = [-1.1, -1.2, -1.3]

def run_pulse_train(voltage, initial_w, is_set):
    # Parametrización del memristor
    p_mem = StrukovParameters(
        R_on=10e3,      # ~100 uS max
        R_off=100e3,    # ~10 uS min
        D=10e-9, 
        mu_v=1.5e-14 if is_set else 2.0e-14, # Ajuste fenomenológico de movilidad
        w_init=initial_w,
        enable_nonlinear_drift=True
    )
    mem = StrukovMemristor(p_mem)
    
    conductances = []
    
    for _ in range(num_pulses):
        # Medir conductancia (estado actual)
        conductances.append(1.0 / mem.resistance)
        
        # Aplicar pulso
        # La fuerte dependencia no lineal del voltaje se modela físicamente
        # ajustando la movilidad iónica efectiva durante el pulso
        base_mu_v = p_mem.mu_v
        mem.params.mu_v = base_mu_v * (abs(voltage) / 1.0)**5
        
        for _ in range(n_steps_per_pulse):
            mem.step(voltage, dt)
            
        mem.params.mu_v = base_mu_v

    return np.array(conductances) * 1e6 # Convertir a µS

# ── 1. Simular Trenes de Pulsos ───────────────────────────────────────────────

results_set = {}
for v in voltages_set:
    # Para SET, iniciamos en estado de baja conductancia (w=0.1)
    results_set[v] = run_pulse_train(v, initial_w=0.1, is_set=True)

results_reset = {}
for v in voltages_reset:
    # Para RESET, iniciamos en estado de alta conductancia (w=0.9)
    results_reset[v] = run_pulse_train(v, initial_w=0.9, is_set=False)

# ── 2. Generar Figura S6 ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(1, 2, wspace=0.3)

# Panel (a): RESET
ax_a = fig.add_subplot(gs[0])
colors_reset = { -1.1: 'green', -1.2: 'red', -1.3: 'blue' }
markers_reset = { -1.1: 's', -1.2: 'o', -1.3: '^' }

for v in voltages_reset:
    ax_a.plot(range(num_pulses), results_reset[v], 
              marker=markers_reset[v], color=colors_reset[v], 
              linestyle='none', markersize=3, alpha=0.6, label=f"{v}V")
    
    # Línea de tendencia (media móvil)
    trend = np.convolve(results_reset[v], np.ones(20)/20, mode='valid')
    ax_a.plot(range(9, num_pulses-10), trend, color='black', linestyle='--', lw=1.5)

ax_a.set_title("(a) RESET Polarities", fontweight='bold', fontsize=12)
ax_a.set_xlabel("Pulse #", fontsize=11)
ax_a.set_ylabel("Conductance (µS)", fontsize=11)
ax_a.set_ylim([0, 80])
ax_a.legend(loc='upper right', framealpha=1.0, edgecolor='black')
ax_a.grid(True, ls=':', alpha=0.5)

# Panel (b): SET
ax_b = fig.add_subplot(gs[1])
colors_set = { 1.1: 'blue', 1.2: 'red', 1.3: 'green' }
markers_set = { 1.1: 'o', 1.2: 's', 1.3: '^' }

for v in voltages_set:
    ax_b.plot(range(num_pulses), results_set[v], 
              marker=markers_set[v], color=colors_set[v], 
              linestyle='none', markersize=3, alpha=0.6, label=f"{v}V")
    
    # Línea de tendencia
    trend = np.convolve(results_set[v], np.ones(20)/20, mode='valid')
    ax_b.plot(range(9, num_pulses-10), trend, color='black', linestyle='--', lw=1.5)

ax_b.set_title("(b) SET Polarities", fontweight='bold', fontsize=12)
ax_b.set_xlabel("Pulse #", fontsize=11)
ax_b.set_ylabel("Conductance (µS)", fontsize=11)
ax_b.set_ylim([20, 110])
ax_b.legend(loc='upper left', framealpha=1.0, edgecolor='black')
ax_b.grid(True, ls=':', alpha=0.5)

fig.suptitle("Evolution of memristor's effective conductance under 500-µs pulse trains", fontsize=14, y=0.98)

out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso21_caracterizacion_pulsos.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[PNG] Guardado: {out_path}")
