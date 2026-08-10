"""
paso13_plasticidad_y_memoria.py
===============================
Demostracion definitiva del comportamiento Neuromorfico:
Plasticidad, Memoria No Volatil e Integracion dependiente del historial.

Protocolo:
1. Entrenamiento (0.0 - 0.4s): Tren de pulsos. El memristor aprende.
2. Apagon/Silencio (0.4 - 0.8s): 0 Voltios. Prueba de Memoria No Volatil.
3. Re-evaluacion (0.8 - 1.2s): El MISMO tren de pulsos. La neurona dispara mas 
   porque la sinapsis "recordo" el entrenamiento previo.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from memristor_simulator.models.lif_neuron    import default_lif_params, LIFNeuron
from memristor_simulator.models.strukov_model import StrukovMemristor, dxdt_strukov, window_biolek
from memristor_simulator.config.parameters import StrukovParameters

# ── Configuracion ─────────────────────────────────────────────────────────────
dt = 0.05e-3
T = 1.3  # 1.3 segundos
n = int(T / dt)
t = np.arange(n) * dt
t_ms = t * 1e3

# ── Crear Señal Vin (Tren 1 -> Silencio -> Tren 2 idéntico) ───────────────────
Vin = np.zeros(n)

def add_pulse_train(start_t, end_t, freq):
    t_curr = start_t
    period = 1.0 / freq
    while t_curr < end_t:
        idx_start = int(t_curr / dt)
        idx_end = int((t_curr + 0.005) / dt) # 5ms width
        if idx_end < n:
            Vin[idx_start:idx_end] = 5.0
        t_curr += period

# Tren 1: 0.05s a 0.4s (50 Hz)
add_pulse_train(0.05, 0.4, 50)
# Tren 2: 0.8s a 1.15s (50 Hz, identico al Tren 1)
add_pulse_train(0.8, 1.15, 50)

# ── Simulacion ────────────────────────────────────────────────────────────────
p_lif = default_lif_params()
p_lif.R_s = 100e3  # Resistencia de neurona ajustada

p_mem = StrukovParameters(
    R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=1e-12,
    w_init=0.01, enable_nonlinear_drift=True, enable_hard_switching=True
)

neuron = LIFNeuron(p_lif)
mem = StrukovMemristor(p_mem)

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
    
    # Simular apagado de energia total durante el silencio
    if 0.45 <= t[k] <= 0.75:
        neuron.V = 0.0
    else:
        neuron.step(V_eff, dt)
    
    Vc[k] = neuron.V
    Vout[k] = neuron.Vout
    w_tsm[k] = neuron.w
    R_mem[k] = mem.resistance
    I_mem[k] = i_in

# Detectar spikes
sp_edges = np.diff((w_tsm >= 0.5).astype(int)) > 0
spikes = t_ms[1:][sp_edges]
spikes_t1 = spikes[spikes < 450]
spikes_t2 = spikes[spikes > 750]

# Estilizar Vout para grafico
Vout_d = Vout.copy()
sp_idx = np.where(sp_edges)[0] + 1
for idx in sp_idx:
    seg = Vout[idx:min(idx + int(20e-3/dt), n)]
    if len(seg) == 0: continue
    pk = idx + np.argmax(seg)
    tl = min(int(25e-3/dt), n - pk)
    Vout_d[pk:pk+tl] += Vout[pk] * np.exp(-np.arange(tl)*dt/3e-3) * 0.30
Vout_d += np.random.normal(0, 0.003, n)
Vout = np.clip(Vout_d, -0.04, 0.80)

# ── Figura ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4, left=0.08, right=0.95, top=0.9, bottom=0.08)

# Panel 1: Entrada y Zonas
ax1 = fig.add_subplot(gs[0])
ax1.plot(t_ms, Vin, color="#F57C00", lw=2, label="$V_{in}(t)$")
ax1.axvspan(0, 450, color="#E3F2FD", alpha=0.5, label="Fase 1: Entrenamiento")
ax1.axvspan(450, 750, color="#FFF9C4", alpha=0.5, label="Fase 2: Apagón (Retención de Memoria)")
ax1.axvspan(750, 1200, color="#E8F5E9", alpha=0.5, label="Fase 3: Re-evaluación")
ax1.set_ylabel("Voltaje (V)", fontsize=11)
ax1.set_title("① Estímulo: Dos trenes de pulsos idénticos separados por un período de inactividad", fontweight="bold")
ax1.grid(True, ls=":", alpha=0.5)
ax1.set_xlim([0, T*1e3])
ax1.legend(loc="upper right")

# Panel 2: Memristor (Plasticidad y No-Volatilidad)
ax2 = fig.add_subplot(gs[1])
ax2.plot(t_ms, R_mem/1e3, color="#8E24AA", lw=3, label=r"$R_M(t)$")
ax2.set_ylabel(r"$R_M$ (kOhm)", fontsize=11)
ax2.set_title("② Sinapsis Artificial: Demostración de Plasticidad y Memoria No Volátil", fontweight="bold")
ax2.annotate('Aprende (Cae la R)', xy=(200, 350), xytext=(200, 450), 
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), ha='center')
ax2.annotate('Retiene Memoria sin Energía\n(Estado No Volátil)', xy=(600, 250), xytext=(600, 100), 
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), ha='center')
ax2.grid(True, ls=":", alpha=0.5)
ax2.set_xlim([0, T*1e3])
ax2.set_ylim([0, 520])

# Panel 3: Neurona (Resultados)
ax3 = fig.add_subplot(gs[2])
ax3.plot(t_ms, Vc, color="#1E88E5", lw=1.5, label="$V_c(t)$")
ax3.axhline(0.95, color="gray", ls="--", lw=1.5, label="$V_{th}$")

# Añadir spikes al Vc para que se vean bonitos
for sp in spikes:
    idx = int(sp / 1000 / dt)
    if idx < n:
        ax3.plot([sp, sp], [Vc[idx], 1.2], color="#43A047", lw=2)

ax3.set_ylabel("$V_c$ (V)", fontsize=11)
ax3.set_xlabel("Tiempo (ms)", fontsize=12)
ax3.set_title("③ Respuesta Neuromórfica: El mismo estímulo produce diferentes respuestas gracias a la experiencia previa", fontweight="bold")
ax3.text(225, 1.05, f"Spikes: {len(spikes_t1)}", color="#43A047", fontweight="bold", ha="center")
ax3.text(975, 1.05, f"Spikes: {len(spikes_t2)}", color="#43A047", fontweight="bold", ha="center")
ax3.grid(True, ls=":", alpha=0.5)
ax3.set_xlim([0, T*1e3])
ax3.set_ylim([-0.1, 1.3])
ax3.legend(loc="upper left")

fig.suptitle(
    "PASO 13 — PRUEBA DEFINITIVA DE COMPORTAMIENTO NEUROMÓRFICO\n"
    "Integración de Plasticidad (LTP) y Memoria No Volátil",
    fontsize=14, fontweight="bold", y=0.96
)

out_dir  = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "paso13_demostracion_memoria.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\n[PNG] Guardado: {out_path}")
