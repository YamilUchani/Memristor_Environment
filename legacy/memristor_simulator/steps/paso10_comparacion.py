"""
paso10_comparacion.py
=====================
PASO 10: Comparacion de experimentos (HRS vs LRS vs Aprendizaje).

Objetivo:
    Cerrar la primera validacion neuromorfica demostrando que:
    1. El estado del memristor (R_M) modifica la corriente transmitida (I_mem).
    2. La corriente modifica la velocidad de carga (V_c) y la actividad neuronal.
    3. El memristor actua como una sinapsis artificial de peso variable.
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

# ── Configuracion comun ───────────────────────────────────────────────────────
dt   = 0.05e-3
T    = 0.50
n    = int(T / dt)
t    = np.arange(n) * dt
t_ms = t * 1e3

# Generar la misma entrada Vin para todos
np.random.seed(42)
t_pts = [0.0]; v_pts = [0.0]; t_cur = 0.0
while t_cur < T * 1e3 + 20.0:
    gap   = np.random.uniform(3.0, 5.0)
    trise = np.random.uniform(0.6, 1.2)
    tflat = np.random.uniform(3.2, 4.5)
    tfall = np.random.uniform(0.6, 1.2)
    ts = t_cur + gap; tp = ts + trise; te = tp + tflat; tf = te + tfall
    vp = np.random.uniform(4.8, 5.15)
    t_pts.extend([ts, tp, te, tf])
    v_pts.extend([np.random.uniform(-0.02, 0.03), vp, vp, np.random.uniform(-0.02, 0.03)])
    t_cur = tf

Vin_ideal = np.interp(t, np.array(t_pts)*1e-3, np.array(v_pts))
Vin       = np.clip(Vin_ideal + np.random.normal(0, 0.03, n), -0.05, 6.0)

# Ruido comun para V_c
noise_raw = np.random.normal(0, 1.0, n)
a = dt / (dt + 2e-3)
nvc = np.zeros(n)
for k in range(1, n): nvc[k] = nvc[k-1] + a * (noise_raw[k] - nvc[k-1])
vc_noise = nvc * 0.007 + Vin_ideal * 0.003

# ── Funcion de Simulacion ─────────────────────────────────────────────────────
def simular_caso(w_i, mu):
    p_lif = default_lif_params()
    p_lif.R_s = 100e3  # Resistencia de neurona ajustada

    p_mem = StrukovParameters(
        R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=mu,
        w_init=w_i, enable_nonlinear_drift=True, enable_hard_switching=True
    )
    
    neuron = LIFNeuron(p_lif)
    mem    = StrukovMemristor(p_mem)
    
    Vc = np.zeros(n); Vout = np.zeros(n); w_tsm = np.zeros(n)
    R_mem = np.zeros(n); I_mem = np.zeros(n)

    for k in range(n):
        r_m = mem.resistance
        i_in = (Vin[k] - neuron.V) / (r_m + p_lif.R_s)
        
        dxdt = dxdt_strukov(i_in, p_mem.R_on, p_mem.D, p_mem.mu_v)
        if p_mem.enable_nonlinear_drift: dxdt *= window_biolek(mem.x, p=5)
        mem.x = float(np.clip(mem.x + dxdt * dt, 0.0, 1.0))
        mem._recalculate_resistance()
        
        V_eff = neuron.V + i_in * p_lif.R_s
        neuron.step(V_eff, dt)
        
        Vc[k] = neuron.V
        Vout[k] = neuron.Vout
        w_tsm[k] = neuron.w
        R_mem[k] = mem.resistance
        I_mem[k] = i_in

    # Post-proceso
    Vc = Vc + vc_noise
    sp_edges = np.diff((w_tsm >= 0.5).astype(int)) > 0
    sp_idx   = np.where(sp_edges)[0] + 1
    
    Vout_d = Vout.copy()
    for idx in sp_idx:
        seg = Vout[idx:min(idx + int(20e-3/dt), n)]
        if len(seg) == 0: continue
        pk = idx + np.argmax(seg)
        tl = min(int(25e-3/dt), n - pk)
        Vout_d[pk:pk+tl] += Vout[pk] * np.exp(-np.arange(tl)*dt/3e-3) * 0.30
    Vout_d += 0.004*np.sin(2*np.pi*1.3*t) + np.random.normal(0, 0.003, n)
    Vout = np.clip(Vout_d, -0.04, 0.80)
    
    I_mem = I_mem + np.random.normal(0, 0.05e-6, n)
    
    spikes = t_ms[1:][sp_edges]
    return R_mem, I_mem, Vc, Vout, spikes

# ── Ejecutar los 3 casos ──────────────────────────────────────────────────────
print("Simulando Estado HRS...")
R_hrs, I_hrs, Vc_hrs, Vo_hrs, sp_hrs = simular_caso(0.01, 1e-16)

print("Simulando Estado Aprendizaje (Dinámico)...")
R_apr, I_apr, Vc_apr, Vo_apr, sp_apr = simular_caso(0.1, 4e-13)

print("Simulando Estado LRS...")
R_lrs, I_lrs, Vc_lrs, Vo_lrs, sp_lrs = simular_caso(1.0, 1e-16)

# ── Figura Comparativa ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 14))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45, left=0.08, right=0.95, top=0.92, bottom=0.06)

C_HRS = "#E53935" # Rojo
C_APR = "#8E24AA" # Morado
C_LRS = "#1E88E5" # Azul

# Panel 1: Resistencia del Memristor
ax1 = fig.add_subplot(gs[0])
ax1.plot(t_ms, R_hrs/1e3, color=C_HRS, lw=2.5, label="HRS Constante (~500 kOhm)")
ax1.plot(t_ms, R_apr/1e3, color=C_APR, lw=2.5, label="Aprendizaje (500 kOhm → 100 Ohm)")
ax1.plot(t_ms, R_lrs/1e3, color=C_LRS, lw=2.5, label="LRS Constante (100 Ohm)")
ax1.set_ylabel(r"$R_M$ (kOhm)", fontsize=11)
ax1.set_title(r"① Función de Sinapsis Variable — Resistencia del Memristor $R_M(t)$", fontweight="bold")
ax1.grid(True, ls=":", alpha=0.4)
ax1.set_xlim([0, T*1e3])
ax1.legend(loc="right", fontsize=10)

# Panel 2: Corriente Transmitida
ax2 = fig.add_subplot(gs[1])
ax2.plot(t_ms, I_hrs*1e6, color=C_HRS, lw=1.2, alpha=0.7, label=r"I (HRS)")
ax2.plot(t_ms, I_apr*1e6, color=C_APR, lw=1.5, alpha=0.8, label=r"I (Aprendizaje)")
ax2.plot(t_ms, I_lrs*1e6, color=C_LRS, lw=1.2, alpha=0.6, label=r"I (LRS)")
ax2.set_ylabel(r"$I_{mem}$ (µA)", fontsize=11)
ax2.set_title(r"② Control de Corriente — El memristor regula la magnitud de la corriente inyectada", fontweight="bold")
ax2.grid(True, ls=":", alpha=0.4)
ax2.set_xlim([0, T*1e3])

# Panel 3: Carga del Capacitor Vc
ax3 = fig.add_subplot(gs[2])

import pandas as pd
csv_vc_path = os.path.join(os.path.dirname(__file__), "../../data/Vc vs Time_ Figure 3.csv")
if os.path.exists(csv_vc_path):
    df_vc = pd.read_csv(csv_vc_path, header=None, names=['t', 'v']).sort_values('t')
    # Filtrar un poco los valores anómalos o negativos grandes si los hay al inicio
    df_vc = df_vc[df_vc['t'] >= 0]
    ax3.plot(df_vc['t'].values * 1e3, df_vc['v'].values, color="black", ls="--", lw=2.0, alpha=0.6, label="Experimental (Fig 3)")

ax3.plot(t_ms, Vc_hrs, color=C_HRS, lw=1.5, label=f"HRS (Carga Bloqueada)")
ax3.plot(t_ms, Vc_apr, color=C_APR, lw=2.0, label=f"Aprendizaje (Carga Acelerada)")
ax3.plot(t_ms, Vc_lrs, color=C_LRS, lw=1.5, label=f"LRS (Carga Facilitada)")
ax3.axhline(0.95, color="gray", ls="--", lw=1.5, label="$V_{th}$ = 0.95 V")
ax3.set_ylabel(r"$V_c$ (V)", fontsize=11)
ax3.set_title(r"③ Integración Neuronal — La diferencia de corriente modifica la velocidad de carga", fontweight="bold")
ax3.grid(True, ls=":", alpha=0.4)
ax3.set_xlim([0, T*1e3])
ax3.legend(loc="upper left", fontsize=10, ncol=4)

# Panel 4: Raster Plot de Spikes
ax4 = fig.add_subplot(gs[3])
ax4.eventplot(sp_hrs, lineoffsets=0.8, linelengths=0.3, color=C_HRS, lw=3)
ax4.eventplot(sp_apr, lineoffsets=0.5, linelengths=0.3, color=C_APR, lw=3)
ax4.eventplot(sp_lrs, lineoffsets=0.2, linelengths=0.3, color=C_LRS, lw=3)

ax4.text(-10, 0.8, f"HRS: {len(sp_hrs)} spikes", color=C_HRS, va="center", ha="right", fontweight="bold")
ax4.text(-10, 0.5, f"Aprendizaje: {len(sp_apr)} spikes", color=C_APR, va="center", ha="right", fontweight="bold")
ax4.text(-10, 0.2, f"LRS: {len(sp_lrs)} spikes", color=C_LRS, va="center", ha="right", fontweight="bold")

ax4.set_yticks([])
ax4.set_ylim([0.0, 1.0])
ax4.set_xlabel("Tiempo (ms)", fontsize=12)
ax4.set_title(r"④ Disparo Neuronal (Spikes) — Traducción de la resistencia en tasa de disparo (Firing Rate)", fontweight="bold")
ax4.grid(True, ls=":", alpha=0.4, axis='x')
ax4.set_xlim([0, T*1e3])

fig.suptitle(
    "PASO 10 — VALIDACIÓN NEUROMÓRFICA COMPLETA\nDemostración del Memristor como Sinapsis Artificial",
    fontsize=16, fontweight="bold"
)

out_dir  = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso10_comparacion.png")

fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\n[PNG] Guardado: {out_path}")
plt.show()
