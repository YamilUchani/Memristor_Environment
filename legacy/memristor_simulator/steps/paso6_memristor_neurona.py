"""
paso6_memristor_neurona.py
==========================
PASO 6: Conexión del Memristor Strukov a la Neurona TSM-LIF.

Topología del circuito:
    Vin(t) ──[M(x)]──[Rs]──Vc──[TSM]──[R0]──GND
              Strukov         LIF física

La señal de entrada atraviesa el memristor Strukov (no-volátil).
La corriente resultante I_M = (Vin - Vc) / (R_M(x) + Rs) carga el capacitor.
El estado x del memristor evoluciona con esa corriente ⟹
el historial de actividad modifica el ritmo de disparo de la neurona.

Ecuaciones acopladas:
    I_M    = (Vin - Vc) / (R_M(x) + Rs)
    dVc/dt = (I_M - Vc/(R_tsm + R0)) / C
    dx/dt  = β · I_M · f(x)          [Strukov]
    dw/dt  = dinámica TSM             [lif_neuron]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from memristor_simulator.models.lif_neuron   import default_lif_params, LIFNeuron
from memristor_simulator.models.strukov_model import (
    StrukovMemristor, calculate_resistance, dxdt_strukov, window_biolek
)
from memristor_simulator.config.parameters import StrukovParameters

# ── Parámetros ────────────────────────────────────────────────────────────────
p_lif = default_lif_params()

# Memristor Strukov con movilidad aumentada para operar en escala de 500 ms
# (el mu_v original está pensado para señales de ~Hz con ciclos de segundos;
#  escalamos para que x cambie de forma visible en 500 ms con corrientes de ~µA)
p_mem = StrukovParameters(
    R_on   = 100.0,        # 100 Ohm — estado ON
    R_off  = 16_000.0,     # 16 kOhm — estado OFF
    D      = 10e-9,        # 10 nm
    # mu_v calibrado para que x evolucione de 0.1 -> ~0.57 en 500 ms:
    # I_media ~ 21 uA, dx/dt = mu_v * Ron/D^2 * I * duty = mu_v * 1e18 * 21e-6 * 0.45
    # Para delta_x ~ 0.47 en 0.5 s -> mu_v = 1e-13 m^2/V*s
    mu_v   = 1e-13,
    w_init = 0.1,          # empieza casi en OFF (x=0.1 -> R~14.5 kOhm)
    enable_nonlinear_drift = True,
    enable_hard_switching  = True,
)

dt    = 0.05e-3    # 0.05 ms
T     = 0.50       # 500 ms
n     = int(T / dt)
f_in  = 100.0      # Hz
A_in  = 5.0        # V
t     = np.arange(n) * dt

# ── Señal de entrada: trapezoidal errática 100 Hz (igual que Paso 5) ─────────
np.random.seed(42)
t_points = [0.0]; v_points = [0.0]; t_current = 0.0
while t_current < T * 1e3 + 20.0:
    gap   = np.random.uniform(3.0, 5.0)
    trise = np.random.uniform(0.6, 1.2)
    tflat = np.random.uniform(3.2, 4.5)
    tfall = np.random.uniform(0.6, 1.2)
    t_start      = t_current + gap
    t_peak_start = t_start + trise
    t_peak_end   = t_peak_start + tflat
    t_end        = t_peak_end + tfall
    vpeak = np.random.uniform(4.8, 5.15)
    t_points.extend([t_start, t_peak_start, t_peak_end, t_end])
    v_points.extend([np.random.uniform(-0.02,0.03), vpeak, vpeak,
                     np.random.uniform(-0.02,0.03)])
    t_current = t_end

t_points  = np.array(t_points) * 1e-3
v_points  = np.array(v_points)
Vin_ideal = np.interp(t, t_points, v_points)
Vin       = np.clip(Vin_ideal + np.random.normal(0, 0.03, n), -0.05, 6.0)

# ── Simulación acoplada Strukov + TSM-LIF ────────────────────────────────────
neuron  = LIFNeuron(p_lif)
mem     = StrukovMemristor(p_mem)

Vc      = np.zeros(n)   # potencial de membrana
Vout    = np.zeros(n)   # voltaje de salida
w_tsm   = np.zeros(n)   # estado del TSM
x_mem   = np.zeros(n)   # estado del memristor Strukov
R_mem   = np.zeros(n)   # resistencia instantánea del memristor
I_mem   = np.zeros(n)   # corriente a través del memristor

for k in range(n):
    vin_k = Vin[k]

    # 1. Resistencia actual del memristor Strukov
    r_m = mem.resistance                  # R_M(x) en Ω

    # 2. Corriente que pasa por el memristor y llega al nodo Vc
    #    El memristor está en SERIE con Rs antes del capacitor
    i_in = (vin_k - neuron.V) / (r_m + p_lif.R_s)

    # 3. Avanzar el estado del memristor con esa corriente
    #    (usamos la función interna directamente para control total)
    dxdt = dxdt_strukov(i_in, p_mem.R_on, p_mem.D, p_mem.mu_v)
    if p_mem.enable_nonlinear_drift:
        dxdt *= window_biolek(mem.x, p=5)
    mem.x = float(np.clip(mem.x + dxdt * dt, 0.0, 1.0))
    mem._recalculate_resistance()

    # 4. Inyectar esa corriente a la neurona directamente
    #    (sobreescribimos el voltaje de entrada efectivo al integrador LIF)
    #    V_eff = voltaje equivalente que produce la misma corriente i_in a través de Rs
    V_eff = neuron.V + i_in * p_lif.R_s   # = Vin * Rs/(R_M+Rs) + Vc * R_M/(R_M+Rs)

    fired, Vc_k = neuron.step(V_eff, dt)

    # Guardar estado
    Vc[k]    = Vc_k
    Vout[k]  = neuron.Vout
    w_tsm[k] = neuron.w
    x_mem[k] = mem.x
    R_mem[k] = mem.resistance
    I_mem[k] = i_in

# ── Post-proceso: deformaciones físicas realistas (igual que Paso 5) ─────────
noise_raw  = np.random.normal(0, 1.0, n)
alpha_f    = dt / (dt + 2e-3)
noise_vc   = np.zeros(n)
for k in range(1, n):
    noise_vc[k] = noise_vc[k-1] + alpha_f * (noise_raw[k] - noise_vc[k-1])
noise_vc *= 0.007
Vc = Vc + noise_vc + Vin_ideal * 0.003

spike_mask_p  = w_tsm >= 0.5
spike_edges_p = np.diff(spike_mask_p.astype(int)) > 0
spike_idx     = np.where(spike_edges_p)[0] + 1
tau_tail      = 3e-3
Vout_d        = Vout.copy()
for idx in spike_idx:
    end_s  = min(idx + int(20e-3/dt), n)
    seg    = Vout[idx:end_s]
    if len(seg) == 0: continue
    pk_abs = idx + np.argmax(seg)
    pk_val = Vout[pk_abs]
    tl     = min(int(25e-3/dt), n - pk_abs)
    Vout_d[pk_abs:pk_abs+tl] += pk_val * np.exp(-np.arange(tl)*dt/tau_tail)*0.30
Vout_d += 0.004*np.sin(2*np.pi*1.3*t) + np.random.normal(0, 0.003, n)
Vout    = np.clip(Vout_d, -0.04, 0.80)

# ── Métricas ──────────────────────────────────────────────────────────────────
t_ms = t * 1e3
spike_mask    = w_tsm >= 0.5
spike_edges   = np.diff(spike_mask.astype(int)) > 0
spike_times_ms = t_ms[1:][spike_edges]
n_spikes = len(spike_times_ms)
f_real   = n_spikes / T if T > 0 else 0.0

# ── Figura ────────────────────────────────────────────────────────────────────
C_VIN  = "#E65100"
C_VC   = "#1565C0"
C_VOUT = "#2E7D32"
C_MEM  = "#6A1B9A"   # morado — memristor
C_TH   = "#C62828"
C_EL   = "gray"

fig = plt.figure(figsize=(14, 13))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(4, 1, figure=fig,
                       hspace=0.55,
                       left=0.09, right=0.96,
                       top=0.93, bottom=0.06)

# ── Panel 1: Vin ──────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(t_ms, Vin, color=C_VIN, lw=1.4, label=r"$V_{in}(t)$")
ax1.fill_between(t_ms, 0, Vin, alpha=0.10, color=C_VIN)
ax1.set_ylabel(r"$V_{in}$ (V)", fontsize=11, color=C_VIN)
ax1.tick_params(axis="y", labelcolor=C_VIN)
ax1.set_title(r"$V_{in}(t)$ — Entrada trapezoidal errática (100 Hz)", fontsize=11, fontweight="bold")
ax1.grid(True, alpha=0.22, ls=":")
ax1.legend(loc="upper right", fontsize=10)
ax1.set_xlim([0, T*1e3]); ax1.set_ylim([-0.2, A_in*1.15])

# ── Panel 2: Estado y Resistencia del Memristor ───────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2_r = ax2.twinx()

ax2.plot(t_ms, x_mem, color=C_MEM, lw=1.8, label=r"$x(t)$ — Estado Strukov")
ax2.set_ylabel(r"Estado $x = w/D$", fontsize=11, color=C_MEM)
ax2.tick_params(axis="y", labelcolor=C_MEM)
ax2.set_ylim([0, 1])

ax2_r.plot(t_ms, R_mem/1e3, color="#AB47BC", lw=1.2, ls="--", alpha=0.70,
           label=r"$R_M(x)$ (kΩ)")
ax2_r.set_ylabel(r"$R_M$ (kΩ)", fontsize=10, color="#AB47BC")
ax2_r.tick_params(axis="y", labelcolor="#AB47BC")

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_r.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, loc="upper right", fontsize=9)
ax2.set_title(r"Memristor Strukov: $x(t)$ y $R_M(t)$ — Estado interno del dispositivo",
              fontsize=11, fontweight="bold")
ax2.grid(True, alpha=0.22, ls=":")
ax2.set_xlim([0, T*1e3])
ax2.text(0.01, 0.88,
         f"$R_{{on}}$ = {p_mem.R_on:.0f} Ω\n"
         f"$R_{{off}}$ = {p_mem.R_off/1e3:.0f} kΩ\n"
         f"$x_0$ = {p_mem.w_init:.2f}",
         transform=ax2.transAxes, fontsize=9,
         bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# ── Panel 3: Vc ──────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.plot(t_ms, Vc, color=C_VC, lw=1.8, label=r"$V_c(t)$ — Potencial de membrana")
ax3.axhline(p_lif.V_th,   color=C_TH, lw=1.5, ls="--", alpha=0.85,
            label=f"$V_{{th}}$ = {p_lif.V_th:.2f} V")
ax3.axhline(p_lif.V_hold, color=C_EL, lw=1.2, ls="-.", alpha=0.60,
            label=f"$V_{{hold}}$ = {p_lif.V_hold:.2f} V")
for t_sp in spike_times_ms:
    ax3.axvline(t_sp, color=C_VOUT, lw=0.8, alpha=0.35)
ax3.set_ylabel(r"$V_c$ (V)", fontsize=11, color=C_VC)
ax3.tick_params(axis="y", labelcolor=C_VC)
ax3.set_title(r"$V_c(t)$ — Carga del capacitor modulada por $R_M(x)$",
              fontsize=11, fontweight="bold")
ax3.grid(True, alpha=0.22, ls=":")
ax3.legend(loc="upper right", fontsize=9, ncol=2)
ax3.set_xlim([0, T*1e3]); ax3.set_ylim([-0.05, 1.2])
ax3.text(0.01, 0.88,
         f"Spikes = {n_spikes}\n$f_{{out}}$ = {f_real:.1f} Hz",
         transform=ax3.transAxes, fontsize=9,
         bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# ── Panel 4: Vout ─────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3])
ax4.plot(t_ms, Vout, color=C_VOUT, lw=1.8, label=r"$V_{out}(t)$ (Spike físico)")
ax4.fill_between(t_ms, 0, Vout, alpha=0.15, color=C_VOUT)
ax4.set_ylabel(r"$V_{out}$ (V)", fontsize=11, color=C_VOUT)
ax4.tick_params(axis="y", labelcolor=C_VOUT)
ax4.set_xlabel("Tiempo (ms)", fontsize=11)
ax4.set_title(r"$V_{out}(t)$ — Spikes de la neurona (corriente de descarga del TSM)",
              fontsize=11, fontweight="bold")
ax4.grid(True, alpha=0.22, ls=":")
ax4.legend(loc="upper right", fontsize=10)
ax4.set_xlim([0, T*1e3]); ax4.set_ylim([-0.05, 0.75])
ax4.text(0.01, 0.82,
         f"$f_{{out}}$ = {f_real:.2f} Hz\n"
         f"$V_{{spike}}$ = {Vout.max():.3f} V",
         transform=ax4.transAxes, fontsize=9,
         bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# ── Título global ─────────────────────────────────────────────────────────────
fig.suptitle(
    r"PASO 6 — Memristor Strukov + Neurona TSM-LIF"
    f" | $f_{{out}}$ = {f_real:.1f} Hz | $V_{{spike}}$ = {Vout.max():.2f} V",
    fontsize=13, fontweight="bold", y=0.97
)

# ── Guardar ───────────────────────────────────────────────────────────────────
out_dir  = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso6_memristor_neurona.png")

print(f"  Frecuencia medida : {f_real:.2f} Hz")
print(f"  Spikes            : {n_spikes}")
print(f"  R_M inicial       : {R_mem[0]/1e3:.2f} kOhm  (x={x_mem[0]:.3f})")
print(f"  R_M final         : {R_mem[-1]/1e3:.2f} kOhm  (x={x_mem[-1]:.3f})")
if n_spikes > 1:
    print(f"  Períodos entre spikes: {np.diff(spike_times_ms)} ms")

fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"  [PNG] Guardado: {out_path}")
plt.show()
