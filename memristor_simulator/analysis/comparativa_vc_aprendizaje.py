"""
comparativa_vc_aprendizaje.py  (v2)
======================================
Compara las PROPIEDADES ESTADÍSTICAS de Vc entre:
  - Caso Aprendizaje del paso10 (línea morada: memristor 500kΩ → 100Ω)
  - Datos experimentales digitalizados "../../data/Vc vs Time_ Figure 3.csv"

METODOLOGÍA CORRECTA:
  La comparación punto-a-punto (RMSE/R² temporal) NO es válida porque
  ambas señales tienen estímulos de entrada DISTINTOS (el paper tiene su
  propio tren de pulsos experimentales, la simulación usa el Vin aleatorio
  del paso10 con semilla 42). Lo correcto es comparar CARACTERÍSTICAS
  ESTADÍSTICAS de la dinámica de membrana:
    - Rango de voltaje (Vc_max, Vc_min)
    - Valor medio de membrana (Vc_mean)
    - Voltaje de umbral alcanzado
    - Número de spikes disparados
    - Distribución acumulada de Vc (CDF)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os, sys
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from memristor_simulator.models.lif_neuron import default_lif_params, LIFNeuron
from memristor_simulator.models.strukov_model import (
    StrukovMemristor, dxdt_strukov, window_biolek
)
from memristor_simulator.config.parameters import StrukovParameters

# ── Parámetros idénticos al paso10 ────────────────────────────────────────────
dt   = 0.05e-3
T    = 0.50
n    = int(T / dt)
t    = np.arange(n) * dt
t_ms = t * 1e3

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

noise_raw = np.random.normal(0, 1.0, n)
a_filt = dt / (dt + 2e-3)
nvc = np.zeros(n)
for k in range(1, n): nvc[k] = nvc[k-1] + a_filt * (noise_raw[k] - nvc[k-1])
vc_noise = nvc * 0.007 + Vin_ideal * 0.003

# ── Simular caso Aprendizaje (línea morada) ───────────────────────────────────
p_lif = default_lif_params()
p_lif.R_s = 100e3
p_mem = StrukovParameters(
    R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=4e-13,
    w_init=0.1, enable_nonlinear_drift=True, enable_hard_switching=True
)

neuron = LIFNeuron(p_lif)
mem    = StrukovMemristor(p_mem)

Vc_sim = np.zeros(n); R_mem = np.zeros(n); w_tsm = np.zeros(n)

for k in range(n):
    r_m  = mem.resistance
    i_in = (Vin[k] - neuron.V) / (r_m + p_lif.R_s)
    dxdt_val = dxdt_strukov(i_in, p_mem.R_on, p_mem.D, p_mem.mu_v)
    if p_mem.enable_nonlinear_drift:
        dxdt_val *= window_biolek(mem.x, p=5)
    mem.x = float(np.clip(mem.x + dxdt_val * dt, 0.0, 1.0))
    mem._recalculate_resistance()
    V_eff = neuron.V + i_in * p_lif.R_s
    neuron.step(V_eff, dt)
    Vc_sim[k] = neuron.V
    R_mem[k]  = mem.resistance
    w_tsm[k]  = neuron.w

Vc_sim += vc_noise

sp_edges    = np.diff((w_tsm >= 0.5).astype(int)) > 0
spikes_sim  = t_ms[1:][sp_edges]
n_spikes_sim = len(spikes_sim)

# ── Cargar datos experimentales ───────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "../../data/Vc vs Time_ Figure 3.csv")
df_exp   = pd.read_csv(csv_path, header=None, names=['t', 'v']).dropna().sort_values('t')
df_exp   = df_exp[df_exp['t'] >= 0]
t_exp    = df_exp['t'].values
Vc_exp   = df_exp['v'].values
t_exp_ms = t_exp * 1e3

sp_idx_exp   = np.where(np.diff((Vc_exp >= 0.90).astype(int)) > 0)[0]
n_spikes_exp = len(sp_idx_exp)

# ── Métricas estadísticas de la señal Vc (VÁLIDAS SIN ALINEACIÓN TEMPORAL) ──
VTH = 0.95

# Estadísticos básicos
vc_sim_max  = float(Vc_sim.max())
vc_sim_min  = float(np.clip(Vc_sim, 0, None).min())  # Ignorar ruido negativo
vc_sim_mean = float(Vc_sim.mean())
vc_sim_std  = float(Vc_sim.std())

vc_exp_max  = float(Vc_exp.max())
vc_exp_min  = float(max(Vc_exp.min(), 0))
vc_exp_mean = float(Vc_exp.mean())
vc_exp_std  = float(Vc_exp.std())

# Porcentaje de tiempo sobre umbral (tasa de disparo implícita)
pct_above_th_sim = float(np.mean(Vc_sim >= VTH) * 100)
pct_above_th_exp = float(np.mean(Vc_exp >= VTH) * 100)

# Test de Kolmogorov-Smirnov (compara distribuciones de Vc)
ks_stat, ks_pval = ks_2samp(Vc_sim, Vc_exp)

# Rango de operación
rango_sim = vc_sim_max - vc_sim_min
rango_exp = vc_exp_max - vc_exp_min

def err_pct(a, b):
    if abs(b) < 1e-9: return float('nan')
    return abs(a - b) / abs(b) * 100

# ── Imprimir tabla ────────────────────────────────────────────────────────────
print("=" * 66)
print("  TABLA COMPARATIVA ESTADÍSTICA — Vc (Integración Neuronal LIF)")
print("  Aprendizaje (Sim.) vs. Experimental Paper Fig.3")
print("=" * 66)
print(f"{'Métrica':<36} {'Paper':>8}  {'Sim':>8}  {'Error':>8}")
print("-" * 66)
print(f"{'Vc máximo (V)':<36} {vc_exp_max:>8.4f}  {vc_sim_max:>8.4f}  {err_pct(vc_sim_max,vc_exp_max):>7.1f}%")
print(f"{'Vc mínimo funcional (V)':<36} {vc_exp_min:>8.4f}  {vc_sim_min:>8.4f}  {'N/A':>8}")
print(f"{'Vc medio (V)':<36} {vc_exp_mean:>8.4f}  {vc_sim_mean:>8.4f}  {err_pct(vc_sim_mean,vc_exp_mean):>7.1f}%")
print(f"{'Desv. estándar Vc (V)':<36} {vc_exp_std:>8.4f}  {vc_sim_std:>8.4f}  {err_pct(vc_sim_std,vc_exp_std):>7.1f}%")
print(f"{'Rango de operación (V)':<36} {rango_exp:>8.4f}  {rango_sim:>8.4f}  {err_pct(rango_sim,rango_exp):>7.1f}%")
print(f"{'Número de spikes':<36} {n_spikes_exp:>8}  {n_spikes_sim:>8}  {err_pct(n_spikes_sim,n_spikes_exp):>7.1f}%")
print(f"{'Tiempo sobre Vth (% de señal)':<36} {pct_above_th_exp:>8.2f}  {pct_above_th_sim:>8.2f}  {err_pct(pct_above_th_sim,pct_above_th_exp):>7.1f}%")
print("-" * 66)
print(f"{'Test KS (estadístico)':<36} {'—':>8}  {ks_stat:>8.4f}  {'N/A':>8}")
print(f"{'Test KS (p-valor)':<36} {'—':>8}  {ks_pval:>8.4f}  {'N/A':>8}")
print(f"  (p > 0.05 → distribuciones estadísticamente similares)")
print("=" * 66)
print()
print("NOTA METODOLÓGICA:")
print("  RMSE/R² temporal no aplica: los estímulos Vin son distintos en")
print("  el paper y en la simulación. La comparación correcta es sobre")
print("  las propiedades estadísticas de la dinámica de membrana.")
print("=" * 66)

# ── Figura comparativa ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.08, right=0.96, top=0.90, bottom=0.08)

C_SIM = '#8E24AA'
C_EXP = '#1A1A1A'

# ── Panel 1: Señal Vc temporal ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t_exp_ms, Vc_exp, color=C_EXP, lw=2.0, ls='--', alpha=0.85,
         label='Paper — Datos Experimentales (Fig. 3, digitalizado)')
ax1.plot(t_ms, Vc_sim, color=C_SIM, lw=2.0, alpha=0.90,
         label='Simulación — Aprendizaje ($R_M$: 500 kΩ → 100 Ω)')
ax1.axhline(0.95, color='gray', ls=':', lw=1.5, label=r'$V_{th} = 0.95$ V')

for sp in spikes_sim:
    ax1.axvline(sp, color=C_SIM, lw=1.2, alpha=0.55, ls=':')
for idx in sp_idx_exp:
    ax1.axvline(t_exp_ms[idx], color=C_EXP, lw=1.2, alpha=0.35, ls=':')

ax1.set_ylabel(r'$V_c$ (V)', fontsize=12)
ax1.set_xlabel('Tiempo (ms)', fontsize=12)
ax1.set_title('① Dinámica de Membrana $V_c$: Simulación Aprendizaje vs. Paper Experimental',
              fontweight='bold', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, ls=':', alpha=0.45)
ax1.set_xlim([0, T*1e3])
ax1.set_ylim([-0.08, 1.12])

# Anotar métricas clave directamente en la gráfica
ax1.annotate(f'Vc_max sim = {vc_sim_max:.4f} V\nVc_max paper = {vc_exp_max:.4f} V\nError = {err_pct(vc_sim_max,vc_exp_max):.1f}%',
             xy=(0.72, 0.75), xycoords='axes fraction',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F3E5F5', alpha=0.85),
             fontsize=9.5)

# ── Panel 2: CDF (Función de Distribución Acumulada) ──────────────────────
ax2 = fig.add_subplot(gs[1, 0])
Vc_sim_sorted = np.sort(Vc_sim)
Vc_exp_sorted = np.sort(Vc_exp)
cdf_sim = np.arange(1, len(Vc_sim_sorted)+1) / len(Vc_sim_sorted)
cdf_exp = np.arange(1, len(Vc_exp_sorted)+1) / len(Vc_exp_sorted)

ax2.plot(Vc_exp_sorted, cdf_exp, color=C_EXP, lw=2.0, ls='--', label='Paper (CDF)')
ax2.plot(Vc_sim_sorted, cdf_sim, color=C_SIM, lw=2.0, label='Sim. (CDF)')
ax2.axvline(0.95, color='gray', ls=':', lw=1.5, label=r'$V_{th}$')
ax2.set_xlabel(r'$V_c$ (V)', fontsize=11)
ax2.set_ylabel('CDF', fontsize=11)
ax2.set_title('② CDF de $V_c$ — Comparación Distribuciones\n'
              f'Test KS: stat={ks_stat:.3f}, p={ks_pval:.3f}',
              fontweight='bold', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, ls=':', alpha=0.45)
ax2.set_xlim([-0.1, 1.1])

# ── Panel 3: Tabla de métricas ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')

metricas = [
    ['Métrica', 'Paper', 'Sim.', 'Error'],
    ['Vc máximo (V)', f'{vc_exp_max:.4f}', f'{vc_sim_max:.4f}', f'{err_pct(vc_sim_max,vc_exp_max):.1f}%'],
    ['Vc medio (V)', f'{vc_exp_mean:.4f}', f'{vc_sim_mean:.4f}', f'{err_pct(vc_sim_mean,vc_exp_mean):.1f}%'],
    ['Desv. Std (V)', f'{vc_exp_std:.4f}', f'{vc_sim_std:.4f}', f'{err_pct(vc_sim_std,vc_exp_std):.1f}%'],
    ['Rango op. (V)', f'{rango_exp:.4f}', f'{rango_sim:.4f}', f'{err_pct(rango_sim,rango_exp):.1f}%'],
    ['N° spikes', f'{n_spikes_exp}', f'{n_spikes_sim}', f'{err_pct(n_spikes_sim,n_spikes_exp):.1f}%'],
    ['% sobre Vth', f'{pct_above_th_exp:.2f}', f'{pct_above_th_sim:.2f}', f'{err_pct(pct_above_th_sim,pct_above_th_exp):.1f}%'],
    ['KS estadístico', '—', f'{ks_stat:.4f}', '—'],
    ['KS p-valor', '—', f'{ks_pval:.4f}', '—'],
]

tbl = ax3.table(cellText=metricas[1:], colLabels=metricas[0],
                cellLoc='center', loc='center',
                colColours=['#CE93D8','#CE93D8','#CE93D8','#CE93D8'])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.35)
ax3.set_title('③ Tabla de Métricas Estadísticas', fontweight='bold', fontsize=10)

fig.suptitle('VALIDACIÓN CUANTITATIVA — Neurona LIF con Memristor (Caso Aprendizaje)\n'
             'Simulación paso10 vs. Datos Digitalizados del Paper (Figura 3)',
             fontsize=13, fontweight='bold')

out_dir  = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'comparativa_vc_aprendizaje.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'\n[PNG] Guardado: {out_path}')
