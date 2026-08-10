"""
paso9_estado_lrs.py
===================
PASO 9: Experimento con el memristor en Baja Resistencia (LRS).

Se mantiene el dispositivo cerca de LRS (x ~ 1, R_M ~ R_on).
Objetivo: Demostrar que una resistencia baja facilita la activacion neuronal
          (se incrementa la corriente transmitida y aumenta el numero de spikes).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from memristor_simulator.models.lif_neuron    import default_lif_params, LIFNeuron
from memristor_simulator.models.strukov_model import (
    StrukovMemristor, dxdt_strukov, window_biolek
)
from memristor_simulator.config.parameters import StrukovParameters

# ── Parametros ────────────────────────────────────────────────────────────────
p_lif = default_lif_params()
p_lif.R_s = 100e3  # Reducido para que el memristor domine la dinamica

# Configuramos el memristor para que se mantenga en LRS
p_mem = StrukovParameters(
    R_on   = 100.0,
    R_off  = 500_000.0, # Aumentado pedagogicamente
    D      = 10e-9,
    # mu_v extremadamente bajo para que x casi no cambie y se quede en LRS
    mu_v   = 1e-16,
    w_init = 1.0,   # Inicia en x=1.0 -> R_M = 100 Ohm
    enable_nonlinear_drift = True,
    enable_hard_switching  = True,
)

dt   = 0.05e-3
T    = 0.50
n    = int(T / dt)
f_in = 100.0
A_in = 5.0
t    = np.arange(n) * dt

# ── Entrada: trapezoidal erratica 100 Hz ─────────────────────────────────────
np.random.seed(42)
t_pts = [0.0]; v_pts = [0.0]; t_cur = 0.0
while t_cur < T * 1e3 + 20.0:
    gap   = np.random.uniform(3.0, 5.0)
    trise = np.random.uniform(0.6, 1.2)
    tflat = np.random.uniform(3.2, 4.5)
    tfall = np.random.uniform(0.6, 1.2)
    ts = t_cur + gap
    tp = ts + trise
    te = tp + tflat
    tf = te + tfall
    vp = np.random.uniform(4.8, 5.15)
    t_pts.extend([ts, tp, te, tf])
    v_pts.extend([np.random.uniform(-0.02, 0.03), vp, vp,
                  np.random.uniform(-0.02, 0.03)])
    t_cur = tf

t_pts     = np.array(t_pts) * 1e-3
v_pts     = np.array(v_pts)
Vin_ideal = np.interp(t, t_pts, v_pts)
Vin       = np.clip(Vin_ideal + np.random.normal(0, 0.03, n), -0.05, 6.0)

# ── Simulacion acoplada ───────────────────────────────────────────────────────
neuron = LIFNeuron(p_lif)
mem    = StrukovMemristor(p_mem)

Vc    = np.zeros(n)
Vout  = np.zeros(n)
w_tsm = np.zeros(n)
x_mem = np.zeros(n)
R_mem = np.zeros(n)
I_mem = np.zeros(n)

for k in range(n):
    r_m  = mem.resistance
    i_in = (Vin[k] - neuron.V) / (r_m + p_lif.R_s)

    dxdt = dxdt_strukov(i_in, p_mem.R_on, p_mem.D, p_mem.mu_v)
    if p_mem.enable_nonlinear_drift:
        dxdt *= window_biolek(mem.x, p=5)
    mem.x = float(np.clip(mem.x + dxdt * dt, 0.0, 1.0))
    mem._recalculate_resistance()

    V_eff = neuron.V + i_in * p_lif.R_s
    fired, Vc_k = neuron.step(V_eff, dt)

    Vc[k]    = Vc_k
    Vout[k]  = neuron.Vout
    w_tsm[k] = neuron.w
    x_mem[k] = mem.x
    R_mem[k] = mem.resistance
    I_mem[k] = i_in

# ── Post-proceso: deformaciones fisicas realistas ─────────────────────────────
noise_raw = np.random.normal(0, 1.0, n)
a = dt / (dt + 2e-3)
nvc = np.zeros(n)
for k in range(1, n):
    nvc[k] = nvc[k-1] + a * (noise_raw[k] - nvc[k-1])
Vc = Vc + nvc * 0.007 + Vin_ideal * 0.003

sp_mask  = w_tsm >= 0.5
sp_edges = np.diff(sp_mask.astype(int)) > 0
sp_idx   = np.where(sp_edges)[0] + 1
Vout_d   = Vout.copy()
for idx in sp_idx:
    seg = Vout[idx:min(idx + int(20e-3/dt), n)]
    if len(seg) == 0: continue
    pk  = idx + np.argmax(seg)
    tl  = min(int(25e-3/dt), n - pk)
    Vout_d[pk:pk+tl] += Vout[pk] * np.exp(-np.arange(tl)*dt/3e-3) * 0.30
Vout_d += 0.004*np.sin(2*np.pi*1.3*t) + np.random.normal(0, 0.003, n)
Vout     = np.clip(Vout_d, -0.04, 0.80)

I_mem = I_mem + np.random.normal(0, 0.05e-6, n)

# ── Metricas ──────────────────────────────────────────────────────────────────
t_ms = t * 1e3
spike_mask     = w_tsm >= 0.5
spike_edges    = np.diff(spike_mask.astype(int)) > 0
spike_times_ms = t_ms[1:][spike_edges]
n_spikes = len(spike_times_ms)
f_real   = n_spikes / T if T > 0 else 0.0

# ── Figura ────────────────────────────────────────────────────────────────────
C_VIN  = "#E65100"
C_X    = "#6A1B9A"
C_IMEM = "#00838F"
C_VC   = "#1565C0"
C_VOUT = "#2E7D32"

fig = plt.figure(figsize=(14, 13))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.55, left=0.09, right=0.94, top=0.93, bottom=0.06)

def style_ax(ax, color, ylabel, title, ylim=None):
    ax.set_ylabel(ylabel, fontsize=10, color=color)
    ax.tick_params(axis="y", labelcolor=color, labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.grid(True, alpha=0.20, ls=":")
    ax.set_xlim([0, T * 1e3])
    if ylim: ax.set_ylim(ylim)

# Panel 1: Vin y R_M
ax1 = fig.add_subplot(gs[0])
ax1.plot(t_ms, Vin, color=C_VIN, lw=1.3, label=r"$V_{in}(t)$")
ax1.fill_between(t_ms, 0, Vin, alpha=0.10, color=C_VIN)
style_ax(ax1, C_VIN, r"$V_{in}$ (V)", r"① $V_{in}(t)$ — Entrada", ylim=[-0.3, A_in*1.18])
ax1_r = ax1.twinx()
ax1_r.plot(t_ms, R_mem/1e3, color="#AB47BC", lw=2.0, ls="--", label=r"$R_M(t)$")
ax1_r.set_ylabel(r"$R_M$ (kOhm)", color="#AB47BC", fontsize=10)
ax1_r.tick_params(axis="y", labelcolor="#AB47BC")
ax1_r.set_ylim([0, 520])
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc="upper right", fontsize=9)
ax1.text(0.01, 0.86, f"Estado: Mantenido en LRS\n$R_M \\approx$ {R_mem[0]/1e3:.2f} kOhm",
         transform=ax1.transAxes, fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# Panel 2: I_mem
ax2 = fig.add_subplot(gs[1])
I_uA = I_mem * 1e6
ax2.plot(t_ms, I_uA, color=C_IMEM, lw=1.1, label=r"$I_{mem}(t)$", alpha=0.85)
ax2.fill_between(t_ms, 0, I_uA, alpha=0.12, color=C_IMEM)
style_ax(ax2, C_IMEM, r"$I_{mem}$ (µA)", r"② $I_{mem}(t)$ — Corriente incrementada por la baja resistencia")
ax2.legend(loc="upper right", fontsize=9)
ax2.text(0.01, 0.82, f"$I_{{max}}$ = {I_uA.max():.2f} µA\n$I_{{media}}$ = {I_uA[I_uA>0].mean():.2f} µA",
         transform=ax2.transAxes, fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# Panel 3: Vc
ax3 = fig.add_subplot(gs[2])
ax3.plot(t_ms, Vc, color=C_VC, lw=1.8, label=r"$V_c(t)$")
ax3.axhline(p_lif.V_th, color="#C62828", lw=1.4, ls="--", alpha=0.85, label=f"$V_{{th}}$ = {p_lif.V_th:.2f} V")
for t_sp in spike_times_ms: ax3.axvline(t_sp, color=C_VOUT, lw=0.9, alpha=0.35, zorder=0)
style_ax(ax3, C_VC, r"$V_c$ (V)", r"③ $V_c(t)$ — Carga rápida debido a la alta corriente", ylim=[-0.05, 1.25])
ax3.legend(loc="upper right", fontsize=8, ncol=2)

# Panel 4: Vout
ax4 = fig.add_subplot(gs[3])
ax4.plot(t_ms, Vout, color=C_VOUT, lw=1.6, label=r"$V_{out}(t)$")
ax4.fill_between(t_ms, 0, Vout, alpha=0.15, color=C_VOUT)
for t_sp in spike_times_ms: ax4.axvline(t_sp, color=C_VOUT, lw=0.9, ls="--", alpha=0.30, zorder=0)
style_ax(ax4, C_VOUT, r"$V_{out}$ (V)", r"④ $V_{out}(t)$ — Mayor cantidad de spikes", ylim=[-0.05, 0.80])
ax4.set_xlabel("Tiempo (ms)", fontsize=11)
ax4.legend(loc="upper right", fontsize=9)
ax4.text(0.01, 0.78, f"Spikes: {n_spikes}\n$f_{{out}}$ = {f_real:.2f} Hz",
         transform=ax4.transAxes, fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.85))

fig.suptitle(
    "PASO 9 — Experimento Memristor en LRS (Baja Resistencia)\n"
    f"Spikes facilitados al maximo: {n_spikes} (vs 0 en HRS, vs 3 en Aprendizaje)",
    fontsize=13, fontweight="bold", y=0.975
)

out_dir  = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso9_estado_lrs.png")

print(f"  Frecuencia medida : {f_real:.2f} Hz")
print(f"  Spikes            : {n_spikes}")
print(f"  R_M inicial / final: {R_mem[0]/1e3:.3f} / {R_mem[-1]/1e3:.3f} kOhm")

fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"  [PNG] Guardado: {out_path}")
plt.show()
