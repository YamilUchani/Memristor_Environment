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

# ── Parametros ────────────────────────────────────────────────────────────────
p     = default_lif_params()
dt    = 0.05e-3      # 0.05 ms
T     = 0.50         # 500 ms para observar múltiples ciclos de disparo
n     = int(T / dt)

# Entrada Vin: onda cuadrada de 100 Hz, Vlow = 0 V, Vhigh = 5 V (Toff = 5 ms, Ton = 5 ms)
f_in   = 100.0        # Hz
A_in   = 5.0         # V
t = np.arange(n) * dt
Vin = np.where((t * f_in) % 1.0 < 0.5, A_in, 0.0)

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
C_TH   = "#C62828"   # rojo — umbral
C_EL   = "gray"      # de hold

# ── Panel 1: Vin vs Time ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(t_ms, Vin, color=C_VIN, lw=2.0, label="$V_{in}(t)$")
ax1.fill_between(t_ms, 0, Vin, alpha=0.12, color=C_VIN)
ax1.set_ylabel("$V_{in}$ (V)", fontsize=11, color=C_VIN)
ax1.tick_params(axis="y", labelcolor=C_VIN)
ax1.set_title(
    f"$V_{{in}}(t)$ — Entrada de onda cuadrada  "
    f"($f$ = {f_in:.0f} Hz,  $A$ = {A_in:.1f} V)",
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

# ── Panel 2: Vc vs Time ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.plot(t_ms, Vc, color=C_VC, lw=2.0, label="$V_c(t)$ — Voltaje del Capacitor")
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
    "$V_c(t)$ — Potencial del capacitor de membrana ($C_m$)",
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

# ── Panel 3: Vout vs Time ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.plot(t_ms, Vout, color=C_VOUT, lw=2.0, label="$V_{out}(t)$ (Spike físico)")
ax3.fill_between(t_ms, 0, Vout, alpha=0.15, color=C_VOUT)
ax3.set_ylabel("$V_{out}$ (V)", fontsize=11, color=C_VOUT)
ax3.tick_params(axis="y", labelcolor=C_VOUT)
ax3.set_xlabel("Tiempo (ms)", fontsize=11)
ax3.set_title(
    "$V_{out}(t)$ — Voltaje de salida en $R_0$ (corriente de descarga)",
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

# ── Titulo global ─────────────────────────────────────────────────────────────
fig.suptitle(
    f"Circuito Físico TSM-LIF: $V_{{in}}$ = {A_in:.0f} V (100 Hz) | "
    f"$f_{{out}}$ = {f_real:.1f} Hz | "
    f"$V_{{spike}}$ máx = {Vout.max():.2f} V",
    fontsize=13, fontweight="bold", y=0.97
)

# ── Guardar ───────────────────────────────────────────────────────────────────
out_dir  = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "Vin_Vc_Vout_vs_Time.png")
print(f"  Frecuencia medida : {f_real:.2f} Hz")
if len(spike_times_ms) > 1:
    periods = np.diff(spike_times_ms)
    print(f"  Períodos entre spikes: {periods} ms")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"  [PNG] Guardado: {out_path}")

plt.show()

