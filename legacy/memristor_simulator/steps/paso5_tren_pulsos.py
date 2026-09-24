"""
plot_vin_vc_vout.py
====================
Script para graficar Vin vs Time, Vc vs Time y Vout vs Time.

Simula el circuito RC clasico de la neurona LIF:

    Vin(t) --> [R_in] --> Vc (nodo del capacitor C)
                                |
                               [R_leak]  (fuga de membrana)
                                |
                               GND

    Vc(t)  = potencial de membrana (voltage across C)
    Vout   = spike: pulso breve cuando Vc >= V_th, 0 en otro caso

Se usan los parametros estandar del modelo LIF del proyecto.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from memristor_simulator.models.lif_neuron import default_lif_params, LIFNeuron

# â”€â”€ Parametros â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
p     = default_lif_params()
dt    = 0.05e-3      # 0.05 ms
T     = 0.50         # 500 ms para observar múltiples ciclos de disparo
n     = int(T / dt)
f_in   = 100.0        # Hz
A_in   = 5.0         # V
t = np.arange(n) * dt

# Generar picos erráticos neuromórficos (trapezoidales/cuadráticos) a 100 Hz
# Replicando el comportamiento errático real (con subida/bajada y meseta)
t_points = [0.0]
v_points = [0.0]

t_current = 0.0  # en ms
while t_current < T * 1e3 + 20.0:
    # Parámetros del pulso aleatorio (sintonizados para promediar 100 Hz y mantener la energía de disparo)
    gap = np.random.uniform(3.0, 5.0)       # tiempo apagado (ms)
    trise = np.random.uniform(0.6, 1.2)     # tiempo de subida (ms)
    tflat = np.random.uniform(3.2, 4.5)     # tiempo plano en alto (ms)
    tfall = np.random.uniform(0.6, 1.2)     # tiempo de bajada (ms)
    
    t_start = t_current + gap
    t_peak_start = t_start + trise
    t_peak_end = t_peak_start + tflat
    t_end = t_peak_end + tfall
    
    vpeak = np.random.uniform(4.8, 5.15)
    vvalley1 = np.random.uniform(-0.02, 0.03)
    vvalley2 = np.random.uniform(-0.02, 0.03)
    
    t_points.extend([t_start, t_peak_start, t_peak_end, t_end])
    v_points.extend([vvalley1, vpeak, vpeak, vvalley2])
    
    t_current = t_end

# Convertir tiempos a segundos e interpolar
t_points = np.array(t_points) * 1e-3
v_points = np.array(v_points)
Vin_ideal = np.interp(t, t_points, v_points)

# Agregar ruido de medición de alta frecuencia y recortar
Vin = Vin_ideal + np.random.normal(0, 0.03, len(t))
Vin = np.clip(Vin, -0.05, 6.0)

# ── Simulacion del circuito TSM-LIF Físico ───────────────────────────────────
neuron = LIFNeuron(p)
Vc   = np.zeros(n)    # potencial en el capacitor
Vout = np.zeros(n)    # voltaje de salida (sobre R0)
w    = np.zeros(n)    # estado del TSM

for k in range(n):
    fired, V_curr = neuron.step(Vin[k], dt)
    Vc[k] = V_curr
    Vout[k] = neuron.Vout
    w[k] = neuron.w

# ── Post-proceso: Deformaciones físicas realistas ────────────────────────────
# (modela imperfecciones de medición, ruido térmico y acoplamiento parasitario
#  observadas en los datos experimentales de examen.py)

# --- Vc: ruido térmico + ripple de acoplamiento RC con Vin ---
# Ruido térmico de baja amplitud filtrado (no afecta la escala del disparo)
noise_raw = np.random.normal(0, 1.0, n)
# Filtro de primer orden: tau ~ 2 ms → suaviza el ruido para que sea correlacionado
alpha_filt = dt / (dt + 2e-3)
noise_vc = np.zeros(n)
for k in range(1, n):
    noise_vc[k] = noise_vc[k-1] + alpha_filt * (noise_raw[k] - noise_vc[k-1])
noise_vc *= 0.007   # amplitud ~7 mV pico — visible pero sutil

# Ripple de acoplamiento con Vin: el capacitor "ve" pequeñas oscilaciones del bus
ripple_vc = Vin_ideal * 0.003  # 0.3% de Vin se cuela en Vc

Vc = Vc + noise_vc + ripple_vc

# --- Vout: cola exponencial asimétrica en spikes + variación de piso basal ---
# Detectar los picos de cada spike (w cruza 0.5 hacia arriba)
spike_mask_post = w >= 0.5
spike_edges_post = np.diff(spike_mask_post.astype(int)) > 0
spike_idx = np.where(spike_edges_post)[0] + 1   # índices de inicio de spike

# Añadir cola exponencial de apagado (tau ~ 3 ms) a cada spike
tau_tail = 3e-3   # 3 ms
Vout_deformed = Vout.copy()
for idx in spike_idx:
    # Busca el máximo local dentro de los próximos 20 ms
    end_search = min(idx + int(20e-3 / dt), n)
    local_segment = Vout[idx:end_search]
    if len(local_segment) == 0:
        continue
    peak_rel = np.argmax(local_segment)
    peak_abs = idx + peak_rel
    peak_val = Vout[peak_abs]

    # Aplica la cola exponencial desde el pico hacia adelante
    tail_len = min(int(25e-3 / dt), n - peak_abs)
    decay = peak_val * np.exp(-np.arange(tail_len) * dt / tau_tail) * 0.30
    Vout_deformed[peak_abs:peak_abs + tail_len] += decay

# Variación suave del piso basal (ruido muy bajo-frecuencia, ~1 Hz)
t_norm = t / T  # 0 → 1
baseline_drift = 0.004 * np.sin(2 * np.pi * 1.3 * t)   # drift lento
baseline_hf    = np.random.normal(0, 0.003, n)          # granularidad de medición
Vout_deformed = Vout_deformed + baseline_drift + baseline_hf
Vout_deformed = np.clip(Vout_deformed, -0.04, 0.80)     # límites físicos

Vout = Vout_deformed

# Convertir a ms para el eje X
t_ms  = t * 1e3

# ── Figura ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(3, 1, figure=fig,
                       hspace=0.52,
                       left=0.09, right=0.96,
                       top=0.92, bottom=0.07)

# Colores
C_VIN  = "#E65100"   # naranja oscuro
C_VC   = "#1565C0"   # azul oscuro
C_VOUT = "#2E7D32"   # verde oscuro
C_TH   = "#C62828"   # rojo â€” umbral
C_EL   = "gray"      # de hold

# â”€â”€ Panel 1: Vin vs Time â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# -- Panel 1: Vin vs Time -----------------------------------------------------
ax1 = fig.add_subplot(gs[0])
ax1.plot(t_ms, Vin, color=C_VIN, lw=2.0, label="$V_{in}(t)$")
ax1.fill_between(t_ms, 0, Vin, alpha=0.12, color=C_VIN)
ax1.set_ylabel("$V_{in}$ (V)", fontsize=11, color=C_VIN)
ax1.tick_params(axis="y", labelcolor=C_VIN)
ax1.set_title(
    f"$V_{{in}}(t)$ — Entrada de onda trapezoidal errática (cuadrática con pendiente) "
    f"($f$ = {f_in:.0f} Hz,  $A \\approx$ {A_in:.1f} V)",
    fontsize=11, fontweight="bold"
)
ax1.grid(True, alpha=0.22, ls=":")
ax1.legend(loc="upper right", fontsize=10)
ax1.set_xlim([0, T * 1e3])
ax1.set_ylim([-0.2, A_in * 1.15])
ax1.text(0.01, 0.92,
         f"$R_{{s}}$ = {p.R_s/1e3:.0f} kOhm\n$f_{{in}}$ = {f_in:.0f} Hz",
         transform=ax1.transAxes, fontsize=9,
         bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# â”€â”€ Panel 2: Vc vs Time â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax2 = fig.add_subplot(gs[1])
ax2.plot(t_ms, Vc, color=C_VC, lw=2.0, label="$V_c(t)$ â€” Voltaje del Capacitor")
ax2.axhline(p.V_th,   color=C_TH,  lw=1.5, ls="--", alpha=0.85,
            label=f"$V_{{th}}$ = {p.V_th:.2f} V")
ax2.axhline(p.V_hold, color=C_EL,  lw=1.2, ls="-.", alpha=0.60,
            label=f"$V_{{hold}}$ = {p.V_hold:.2f} V")

# Marcar spikes en Vc
spike_mask = w >= 0.5
# Detectar flancos de subida
spike_edges = np.diff(spike_mask.astype(int)) > 0
spike_times_ms = t_ms[1:][spike_edges]
for t_sp in spike_times_ms:
    ax2.axvline(t_sp, color=C_VOUT, lw=0.8, alpha=0.35)

ax2.set_ylabel("$V_c$ (V)", fontsize=11, color=C_VC)
ax2.tick_params(axis="y", labelcolor=C_VC)
ax2.set_title(
    "$V_c(t)$ â€” Potencial del capacitor de membrana ($C_m$)",
    fontsize=11, fontweight="bold"
)
ax2.grid(True, alpha=0.22, ls=":")
ax2.legend(loc="upper right", fontsize=9, ncol=2)
ax2.set_xlim([0, T * 1e3])
ax2.set_ylim([-0.05, 1.2])

n_spikes_real = len(spike_times_ms)
f_real = n_spikes_real / T if T > 0 else 0
T_spike_ms = (1.0 / f_real * 1e3) if f_real > 0 else 0
ax2.text(0.01, 0.92,
         f"$C$ = {p.C*1e9:.0f} nF\n"
         f"Spikes = {n_spikes_real}\n"
         f"$f_{{out}}$ = {f_real:.1f} Hz\n"
         f"$T_{{spike}}$ = {T_spike_ms:.2f} ms",
         transform=ax2.transAxes, fontsize=9,
         bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# â”€â”€ Panel 3: Vout vs Time â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax3 = fig.add_subplot(gs[2])
ax3.plot(t_ms, Vout, color=C_VOUT, lw=2.0, label="$V_{out}(t)$ (Spike físico)")
ax3.fill_between(t_ms, 0, Vout, alpha=0.15, color=C_VOUT)
ax3.set_ylabel("$V_{out}$ (V)", fontsize=11, color=C_VOUT)
ax3.tick_params(axis="y", labelcolor=C_VOUT)
ax3.set_xlabel("Tiempo (ms)", fontsize=11)
ax3.set_title(
    "$V_{out}(t)$ â€” Voltaje de salida en $R_0$ (corriente de descarga)",
    fontsize=11, fontweight="bold"
)
ax3.grid(True, alpha=0.22, ls=":")
ax3.legend(loc="upper right", fontsize=10)
ax3.set_xlim([0, T * 1e3])
ax3.set_ylim([-0.05, 0.75])
ax3.text(0.01, 0.82,
         f"Objetivo Fspike: ~6 Hz\n"
         f"Medido Fspike  : {f_real:.2f} Hz\n"
         f"Vspike pico    : {Vout.max():.3f} V",
         transform=ax3.transAxes, fontsize=9,
         bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# â”€â”€ Titulo global â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig.suptitle(
    f"Circuito Físico TSM-LIF: $V_{{in}}$ Trapezoidal Errática (100 Hz) | "
    f"$f_{{out}}$ = {f_real:.1f} Hz | "
    f"$V_{{spike}}$ máx = {Vout.max():.2f} V",
    fontsize=13, fontweight="bold", y=0.97
)

# â”€â”€ Guardar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
out_dir  = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso5_tren_pulsos.png")
print(f"  Frecuencia medida : {f_real:.2f} Hz")
if len(spike_times_ms) > 1:
    periods = np.diff(spike_times_ms)
    print(f"  Períodos entre spikes: {periods} ms")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"  [PNG] Guardado: {out_path}")

plt.show()

