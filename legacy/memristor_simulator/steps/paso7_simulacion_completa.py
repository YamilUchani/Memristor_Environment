"""
paso7_simulacion_completa.py
============================
PASO 7: Primera simulacion completa de la cadena neuromorf­ica.

Registra toda la cadena:
    Vin(t) -> [Memristor Strukov M(x)] -> I_mem(t) -> [Rs] -> Vc(t) -> [TSM] -> Vout(t)

Paneles:
    1. Vin(t)    — Pulsos de entrada (trapezoidal erratico 100 Hz)
    2. x(t)      — Estado interno del memristor
    3. R_M(t)    — Resistencia del memristor
    4. I_mem(t)  — Corriente transmitida al nodo del capacitor
    5. Vc(t)     — Potencial de membrana
    6. Vout(t)   — Pulsos de salida (spikes fisicos)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
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

p_mem = StrukovParameters(
    R_on   = 100.0,
    R_off  = 500_000.0, # Aumentado pedagogicamente para bloquear totalmente en HRS
    D      = 10e-9,
    mu_v   = 4e-13,     # Aprendizaje super gradual a lo largo de 500 ms
    w_init = 0.01,
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

    # Avanzar memristor con la corriente real
    dxdt = dxdt_strukov(i_in, p_mem.R_on, p_mem.D, p_mem.mu_v)
    if p_mem.enable_nonlinear_drift:
        dxdt *= window_biolek(mem.x, p=5)
    mem.x = float(np.clip(mem.x + dxdt * dt, 0.0, 1.0))
    mem._recalculate_resistance()

    # Voltaje efectivo al integrador LIF
    V_eff = neuron.V + i_in * p_lif.R_s
    fired, Vc_k = neuron.step(V_eff, dt)

    Vc[k]    = Vc_k
    Vout[k]  = neuron.Vout
    w_tsm[k] = neuron.w
    x_mem[k] = mem.x
    R_mem[k] = mem.resistance
    I_mem[k] = i_in

# ── Post-proceso: deformaciones fisicas realistas ─────────────────────────────
# Vc: ruido termico filtrado + ripple de acoplamiento
noise_raw = np.random.normal(0, 1.0, n)
a = dt / (dt + 2e-3)
nvc = np.zeros(n)
for k in range(1, n):
    nvc[k] = nvc[k-1] + a * (noise_raw[k] - nvc[k-1])
Vc = Vc + nvc * 0.007 + Vin_ideal * 0.003

# Vout: cola exponencial asimetrica + drift de piso
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

# I_mem: ruido de medicion (pA)
I_mem = I_mem + np.random.normal(0, 0.05e-6, n)   # ruido ~50 nA

# ── Metricas ──────────────────────────────────────────────────────────────────
t_ms = t * 1e3
spike_mask     = w_tsm >= 0.5
spike_edges    = np.diff(spike_mask.astype(int)) > 0
spike_times_ms = t_ms[1:][spike_edges]
n_spikes = len(spike_times_ms)
f_real   = n_spikes / T if T > 0 else 0.0

# ── Paleta de colores ─────────────────────────────────────────────────────────
C_VIN  = "#E65100"   # naranja oscuro
C_X    = "#6A1B9A"   # morado oscuro
C_RM   = "#AB47BC"   # lila
C_IMEM = "#00838F"   # teal
C_VC   = "#1565C0"   # azul oscuro
C_VOUT = "#2E7D32"   # verde oscuro
C_TH   = "#C62828"   # rojo umbral
C_HOLD = "gray"

# ── Figura: 6 paneles ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 18))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(6, 1, figure=fig,
                       hspace=0.60,
                       left=0.10, right=0.93,
                       top=0.95, bottom=0.04)

def style_ax(ax, color, ylabel, title, ylim=None):
    ax.set_ylabel(ylabel, fontsize=10, color=color)
    ax.tick_params(axis="y", labelcolor=color, labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.grid(True, alpha=0.20, ls=":")
    ax.set_xlim([0, T * 1e3])
    if ylim: ax.set_ylim(ylim)

# ── Panel 1: Vin ──────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(t_ms, Vin, color=C_VIN, lw=1.3, label=r"$V_{in}(t)$")
ax1.fill_between(t_ms, 0, Vin, alpha=0.10, color=C_VIN)
style_ax(ax1, C_VIN, r"$V_{in}$ (V)",
         r"① $V_{in}(t)$ — Pulsos de entrada trapezoidales (100 Hz)",
         ylim=[-0.3, A_in * 1.18])
ax1.legend(loc="upper right", fontsize=9)
ax1.text(0.01, 0.86,
         f"$f_{{in}}$ = {f_in:.0f} Hz\n$A$ ~ {A_in:.1f} V",
         transform=ax1.transAxes, fontsize=8,
         bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# ── Panel 2: x(t) ─────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.plot(t_ms, x_mem, color=C_X, lw=2.0, label=r"$x(t) = w/D$")
ax2.fill_between(t_ms, p_mem.w_init, x_mem, alpha=0.12, color=C_X)
style_ax(ax2, C_X, r"Estado $x = w/D$",
         r"② $x(t)$ — Estado interno del memristor Strukov",
         ylim=[0, 1.0])
ax2.axhline(p_mem.w_init, color=C_X, lw=0.8, ls="--", alpha=0.5)
ax2.legend(loc="upper left", fontsize=9)
ax2.text(0.75, 0.12,
         f"$x_0$ = {p_mem.w_init:.2f}\n$x_{{final}}$ = {x_mem[-1]:.3f}\n"
         f"$\\Delta x$ = {x_mem[-1]-p_mem.w_init:.3f}",
         transform=ax2.transAxes, fontsize=8,
         bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# ── Panel 3: R_M(t) ───────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.plot(t_ms, R_mem / 1e3, color=C_RM, lw=2.0, label=r"$R_M(t)$")
ax3.fill_between(t_ms, R_mem[-1]/1e3, R_mem/1e3, alpha=0.12, color=C_RM)
style_ax(ax3, C_RM, r"$R_M$ (kOhm)",
         r"③ $R_M(t)$ — Resistencia del memristor",
         ylim=[0, p_mem.R_off/1e3 * 1.1])
ax3.axhline(p_mem.R_on/1e3, color=C_RM, lw=0.8, ls=":", alpha=0.6,
            label=f"$R_{{on}}$ = {p_mem.R_on:.0f} Ohm")
ax3.axhline(p_mem.R_off/1e3, color=C_RM, lw=0.8, ls="--", alpha=0.6,
            label=f"$R_{{off}}$ = {p_mem.R_off/1e3:.0f} kOhm")
ax3.legend(loc="upper right", fontsize=8, ncol=2)
ax3.text(0.01, 0.12,
         f"Inicio: {R_mem[0]/1e3:.2f} kOhm\nFinal:  {R_mem[-1]/1e3:.2f} kOhm\n"
         f"Caida: {(R_mem[0]-R_mem[-1])/1e3:.2f} kOhm",
         transform=ax3.transAxes, fontsize=8,
         bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# ── Panel 4: I_mem(t) ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3])
I_uA = I_mem * 1e6
ax4.plot(t_ms, I_uA, color=C_IMEM, lw=1.1, label=r"$I_{mem}(t)$", alpha=0.85)
ax4.fill_between(t_ms, 0, I_uA, alpha=0.12, color=C_IMEM)
style_ax(ax4, C_IMEM, r"$I_{mem}$ (µA)",
         r"④ $I_{mem}(t)$ — Corriente transmitida al capacitor de membrana")
ax4.legend(loc="upper right", fontsize=9)
ax4.text(0.01, 0.82,
         f"$I_{{max}}$ = {I_uA.max():.2f} µA\n$I_{{media}}$ = {I_uA[I_uA>0].mean():.2f} µA",
         transform=ax4.transAxes, fontsize=8,
         bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# ── Panel 5: Vc(t) ────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[4])
ax5.plot(t_ms, Vc, color=C_VC, lw=1.8, label=r"$V_c(t)$")
ax5.axhline(p_lif.V_th,   color=C_TH,   lw=1.4, ls="--", alpha=0.85,
            label=f"$V_{{th}}$ = {p_lif.V_th:.2f} V")
ax5.axhline(p_lif.V_hold, color=C_HOLD, lw=1.0, ls="-.", alpha=0.55,
            label=f"$V_{{hold}}$ = {p_lif.V_hold:.2f} V")
for t_sp in spike_times_ms:
    ax5.axvline(t_sp, color=C_VOUT, lw=0.9, alpha=0.35, zorder=0)
style_ax(ax5, C_VC, r"$V_c$ (V)",
         r"⑤ $V_c(t)$ — Potencial de membrana (modulado por $R_M$)",
         ylim=[-0.05, 1.25])
ax5.legend(loc="upper right", fontsize=8, ncol=3)
ax5.text(0.01, 0.86,
         f"Spikes: {n_spikes}\n$f_{{out}}$ = {f_real:.1f} Hz",
         transform=ax5.transAxes, fontsize=8,
         bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# ── Panel 6: Vout(t) ──────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[5])
ax6.plot(t_ms, Vout, color=C_VOUT, lw=1.6, label=r"$V_{out}(t)$")
ax6.fill_between(t_ms, 0, Vout, alpha=0.15, color=C_VOUT)
for t_sp in spike_times_ms:
    ax6.axvline(t_sp, color=C_VOUT, lw=0.9, ls="--", alpha=0.30, zorder=0)
style_ax(ax6, C_VOUT, r"$V_{out}$ (V)",
         r"⑥ $V_{out}(t)$ — Pulsos de salida (spikes del TSM-LIF)",
         ylim=[-0.05, 0.80])
ax6.set_xlabel("Tiempo (ms)", fontsize=11)
ax6.legend(loc="upper right", fontsize=9)
ax6.text(0.01, 0.78,
         f"$f_{{out}}$ = {f_real:.2f} Hz\n$V_{{spike}}$ = {Vout.max():.3f} V\n"
         + (f"Periodos: {np.diff(spike_times_ms).astype(int)} ms" if n_spikes > 1 else ""),
         transform=ax6.transAxes, fontsize=8,
         bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# ── Titulo global ─────────────────────────────────────────────────────────────
fig.suptitle(
    "PASO 7 — Cadena Neuromorfica Completa: "
    r"$V_{in} \rightarrow$ Memristor Strukov $\rightarrow$ Neurona TSM-LIF $\rightarrow V_{out}$"
    f"\n$f_{{out}}$ = {f_real:.1f} Hz  |  "
    f"$R_M$: {R_mem[0]/1e3:.1f} kOhm → {R_mem[-1]/1e3:.1f} kOhm  |  "
    f"$\\Delta x$ = {x_mem[-1]-p_mem.w_init:.3f}",
    fontsize=12, fontweight="bold", y=0.975
)

# ── Guardar ───────────────────────────────────────────────────────────────────
out_dir  = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso7_simulacion_completa.png")

print(f"  Frecuencia medida    : {f_real:.2f} Hz")
print(f"  Spikes               : {n_spikes}")
print(f"  x inicial / final    : {x_mem[0]:.3f} / {x_mem[-1]:.3f}  (Dx={x_mem[-1]-p_mem.w_init:.3f})")
print(f"  R_M inicial / final  : {R_mem[0]/1e3:.2f} / {R_mem[-1]/1e3:.2f} kOhm")
print(f"  I_mem max / media    : {I_mem.max()*1e6:.2f} / {I_mem[I_mem>0].mean()*1e6:.2f} uA")
if n_spikes > 1:
    periods = np.diff(spike_times_ms)
    print(f"  Periodos entre spikes: {periods.astype(int)} ms")

fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"  [PNG] Guardado: {out_path}")
plt.show()
