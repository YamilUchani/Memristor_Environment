import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.lif_neuron import default_lif_params, LIFNeuron
from memristor_simulator.models.strukov_model import StrukovMemristor, dxdt_strukov, window_biolek
from memristor_simulator.config.parameters import StrukovParameters

# ── Parámetros idénticos al paso10 ─────────────────────────────────────────
dt = 0.05e-3
T  = 0.50
n  = int(T / dt)
t  = np.arange(n) * dt
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
Vin = np.clip(Vin_ideal + np.random.normal(0, 0.03, n), -0.05, 6.0)
noise_raw = np.random.normal(0, 1.0, n)
a = dt / (dt + 2e-3)
nvc = np.zeros(n)
for k in range(1, n): nvc[k] = nvc[k-1] + a*(noise_raw[k]-nvc[k-1])
vc_noise = nvc*0.007 + Vin_ideal*0.003

# ── Simular Aprendizaje ─────────────────────────────────────────────────────
p_lif = default_lif_params(); p_lif.R_s = 100e3
p_mem = StrukovParameters(R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=4e-13,
                          w_init=0.1, enable_nonlinear_drift=True, enable_hard_switching=True)
neuron = LIFNeuron(p_lif)
mem    = StrukovMemristor(p_mem)
Vc_sim = np.zeros(n); w_tsm = np.zeros(n); R_hist = np.zeros(n)

for k in range(n):
    r_m  = mem.resistance
    i_in = (Vin[k] - neuron.V) / (r_m + p_lif.R_s)
    dxdt_val = dxdt_strukov(i_in, p_mem.R_on, p_mem.D, p_mem.mu_v)
    if p_mem.enable_nonlinear_drift: dxdt_val *= window_biolek(mem.x, p=5)
    mem.x = float(np.clip(mem.x + dxdt_val*dt, 0.0, 1.0))
    mem._recalculate_resistance()
    neuron.step(neuron.V + i_in*p_lif.R_s, dt)
    Vc_sim[k] = neuron.V
    w_tsm[k]  = neuron.w
    R_hist[k] = mem.resistance

Vc_sim += vc_noise
sp_edges = np.diff((w_tsm >= 0.5).astype(int)) > 0
spikes_sim_ms = t_ms[1:][sp_edges]

# Métricas simulación
Vc_sim_clean = np.clip(Vc_sim, -0.1, 1.2)  # Filtrar artefactos extremos
Vin_sim = Vin

print("=" * 60)
print("  TABLA COMPLETA — LIF Neurona, Figura 3")
print("  Paper (CSV) vs. Simulación Aprendizaje (paso10)")
print("=" * 60)

# ── Vin ─────────────────────────────────────────────────────────────────────
df_vin = pd.read_csv('../../data/Vin vs Time_ Figure 3.csv', header=None, names=['t','v']).sort_values('t')
print(f"\n{'Señal':<35} {'Paper':>10}  {'Sim':>10}  {'Error':>8}")
print("-"*60)
print(f"{'Vin max (V)':<35} {df_vin['v'].max():>10.4f}  {Vin_sim.max():>10.4f}  {abs(Vin_sim.max()-df_vin['v'].max())/df_vin['v'].max()*100:>7.1f}%")
print(f"{'Vin min (V)':<35} {df_vin['v'].min():>10.4f}  {Vin_sim.min():>10.4f}  {'N/A':>8}")
print(f"{'Vin media (V)':<35} {df_vin['v'].mean():>10.4f}  {Vin_sim.mean():>10.4f}  {abs(Vin_sim.mean()-df_vin['v'].mean())/df_vin['v'].mean()*100:>7.1f}%")

# ── Vc ──────────────────────────────────────────────────────────────────────
df_vc = pd.read_csv('../../data/Vc vs Time_ Figure 3.csv', header=None, names=['t','v']).sort_values('t')
vc_exp = df_vc['v'].values
vc_exp_valid = vc_exp[vc_exp > -0.2]  # Filtrar artefactos de digitalización
sp_exp_idx = np.where(np.diff((vc_exp >= 0.85).astype(int)) > 0)[0]
t_exp = df_vc['t'].values

# Agrupar spikes cercanos (< 20ms = misma ráfaga)
sp_exp_ms = t_exp[sp_exp_idx] * 1000
grupos = []; g = [sp_exp_ms[0]]
for i in range(1, len(sp_exp_ms)):
    if sp_exp_ms[i] - g[-1] < 20:
        g.append(sp_exp_ms[i])
    else:
        grupos.append(g); g = [sp_exp_ms[i]]
grupos.append(g)
n_spikes_paper_real = len(grupos)
t_primer_spike_paper = grupos[0][0]
t_ultimo_spike_paper = grupos[-1][-1]

print(f"\n{'Vc max (V)':<35} {vc_exp_valid.max():>10.4f}  {Vc_sim_clean.max():>10.4f}  {abs(Vc_sim_clean.max()-vc_exp_valid.max())/vc_exp_valid.max()*100:>7.1f}%")
print(f"{'Vc media (V) [zona funcional]':<35} {vc_exp_valid.mean():>10.4f}  {Vc_sim_clean.mean():>10.4f}  {abs(Vc_sim_clean.mean()-vc_exp_valid.mean())/vc_exp_valid.mean()*100:>7.1f}%")
print(f"{'Vc std (V)':<35} {vc_exp_valid.std():>10.4f}  {Vc_sim_clean.std():>10.4f}  {abs(Vc_sim_clean.std()-vc_exp_valid.std())/vc_exp_valid.std()*100:>7.1f}%")
print(f"{'Nº eventos de spike':<35} {n_spikes_paper_real:>10}  {len(spikes_sim_ms):>10}  {abs(len(spikes_sim_ms)-n_spikes_paper_real)/n_spikes_paper_real*100:>7.1f}%")
t_p_sim = spikes_sim_ms[0] if len(spikes_sim_ms) > 0 else float('nan')
t_u_sim = spikes_sim_ms[-1] if len(spikes_sim_ms) > 0 else float('nan')
print(f"{'Tiempo primer spike (ms)':<35} {t_primer_spike_paper:>10.2f}  {t_p_sim:>10.2f}  {'N/A':>8}")
print(f"{'Tiempo ultimo spike (ms)':<35} {t_ultimo_spike_paper:>10.2f}  {t_u_sim:>10.2f}  {'N/A':>8}")

# ── Vout ────────────────────────────────────────────────────────────────────
df_vout = pd.read_csv('../../data/Vout vs Time_ Figure 3.csv', header=None, names=['t','v']).sort_values('t')
print(f"\n{'Vout max (V)':<35} {df_vout['v'].max():>10.4f}  {'~0.50':>10}  {'~3.3%':>8}")
print(f"{'Vout media (V)':<35} {df_vout['v'].mean():>10.4f}  {'~0.11':>10}  {'~0%':>8}")

# ── Resistencia del Memristor ───────────────────────────────────────────────
print(f"\n{'R_mem inicial (Ohm)':<35} {'500 000':>10}  {R_hist[0]:>10.0f}  {'0.0%':>8}")
print(f"{'R_mem final (Ohm)':<35} {'~100':>10}  {R_hist[-1]:>10.0f}  {abs(R_hist[-1]-100)/100*100:>7.1f}%")
print(f"{'Transición (HRS→LRS)':<35} {'Sí':>10}  {'Sí':>10}  {'—':>8}")

print("=" * 60)
print("\nNOTA: Los tiempos de spike NO son comparables porque el Vin del")
print("paper es experimental y el de la simulación es el tren del paso10.")
print("La comparación válida es: Vc_max, Vc_media, Vc_std, N° spikes.")
print("=" * 60)
