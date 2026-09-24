"""
generate_all_metrics.py
========================
Script para extraer datos y calcular métricas para:
1. STDP (Spike-Timing-Dependent Plasticity)
2. LTP/LTD (Potenciación y Depresión)
3. Retención (Memoria no volátil)
4. Crossbar (Simulación de clasificación y métricas de rendimiento)

Guarda las curvas en CSV y genera archivos de métricas formateados.
"""

import numpy as np
import pandas as pd
import os, sys
from scipy.optimize import curve_fit

# Agregar directorio actual al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from memristor_simulator.models.strukov_model import StrukovMemristor, dxdt_strukov, window_biolek
from memristor_simulator.config.parameters import StrukovParameters
from memristor_simulator.models.lif_neuron import default_lif_params, LIFNeuron

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. STDP
# ─────────────────────────────────────────────────────────────────────────────
print("--- Simulando STDP ---")
dt_stdp = 0.1e-3

def get_spike(t_array, start_time):
    v = np.zeros_like(t_array)
    v[(t_array >= start_time) & (t_array < start_time + 2e-3)] = -1.0
    v[(t_array >= start_time + 2e-3) & (t_array < start_time + 8e-3)] = 0.6
    return v

def calculate_dw(delta_t):
    t = np.arange(0, 80e-3, dt_stdp)
    V_pre = np.zeros_like(t)
    V_post = np.zeros_like(t)
    for i in range(5):
        t_pre = 10e-3 + i * 14e-3
        t_post = t_pre + delta_t
        V_pre += get_spike(t, t_pre)
        V_post += get_spike(t, t_post)
    V_M = V_pre - V_post
    p_mem = StrukovParameters(
        R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=6e-12,
        v_th_mem=1.2, w_init=0.5, enable_nonlinear_drift=True, enable_hard_switching=True
    )
    mem = StrukovMemristor(p_mem)
    x_init = mem.x
    for v in V_M:
        i_in = v / mem.resistance
        dxdt = dxdt_strukov(i_in, p_mem.R_on, p_mem.D, p_mem.mu_v, v_mem=v, v_th_mem=p_mem.v_th_mem)
        if p_mem.enable_nonlinear_drift: dxdt *= window_biolek(mem.x, p=5)
        mem.x = float(np.clip(mem.x + dxdt * dt_stdp, 0.0, 1.0))
        mem._recalculate_resistance()
    return (mem.x - x_init) * 100.0

delta_t_vals = np.linspace(-15e-3, 15e-3, 101)
dw_vals = np.array([calculate_dw(dt_val) for dt_val in delta_t_vals])

# Guardar datos de la curva STDP
stdp_df = pd.DataFrame({
    'Delta_t_s': delta_t_vals,
    'Delta_t_ms': delta_t_vals * 1000.0,
    'Delta_W_percent': dw_vals
})
stdp_df.to_csv(os.path.join(out_dir, "stdp_data.csv"), index=False)

# Calcular métricas de STDP
ltp_max = np.max(dw_vals)
ltd_max = np.min(dw_vals)

# Cruce por cero (interpolado)
zero_idx = np.where(np.diff(np.sign(dw_vals)))[0]
if len(zero_idx) > 0:
    best_idx = zero_idx[np.argmin(np.abs(delta_t_vals[zero_idx]))]
    x1, x2 = delta_t_vals[best_idx], delta_t_vals[best_idx+1]
    y1, y2 = dw_vals[best_idx], dw_vals[best_idx+1]
    cruce_cero = x1 - y1 * (x2 - x1) / (y2 - y1)
else:
    cruce_cero = 0.0

def trapezoid(y, x):
    return np.sum((y[:-1] + y[1:]) * np.diff(x) / 2.0)

# Área LTP y LTD (integración por regla del trapecio)
area_ltp = trapezoid(np.maximum(dw_vals, 0.0), delta_t_vals * 1000.0)
area_ltd = trapezoid(np.minimum(dw_vals, 0.0), delta_t_vals * 1000.0)

# Ajuste exponencial de tau
ltp_mask = (delta_t_vals > 0) & (dw_vals > 0.05)
tau_ltp = np.nan
if np.sum(ltp_mask) > 2:
    try:
        popt, _ = curve_fit(lambda x, A, tau: A * np.exp(-x / tau), delta_t_vals[ltp_mask] * 1000.0, dw_vals[ltp_mask], p0=[ltp_max, 3.0], bounds=(0, [100, 100]))
        tau_ltp = popt[1]
    except Exception:
        pass

ltd_mask = (delta_t_vals < 0) & (dw_vals < -0.05)
tau_ltd = np.nan
if np.sum(ltd_mask) > 2:
    try:
        popt, _ = curve_fit(lambda x, B, tau: B * np.exp(x / tau), delta_t_vals[ltd_mask] * 1000.0, dw_vals[ltd_mask], p0=[ltd_max, 3.0], bounds=([-100, 0], [0, 100]))
        tau_ltd = popt[1]
    except Exception:
        pass

ltp_ltd_ratio = abs(ltp_max / ltd_max) if ltd_max != 0 else np.nan

stdp_metrics = {
    'Metrica': ['LTP maximo', 'LTD maximo', 'Cruce por cero (ms)', 'Area LTP', 'Area LTD', 'tau LTP (ms)', 'tau LTD (ms)', 'Ratio LTP/LTD'],
    'Valor': [ltp_max, ltd_max, cruce_cero * 1000.0, area_ltp, area_ltd, tau_ltp, tau_ltd, ltp_ltd_ratio]
}
pd.DataFrame(stdp_metrics).to_csv(os.path.join(out_dir, "stdp_metrics.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LTP / LTD
# ─────────────────────────────────────────────────────────────────────────────
print("--- Simulando LTP/LTD ---")
dt_lt = 0.1e-3
t_lt = np.arange(0, 100e-3, dt_lt)

p_mem_lt = StrukovParameters(
    R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=5e-12,
    v_th_mem=0.0, w_init=0.5, enable_nonlinear_drift=True, enable_hard_switching=True
)
mem_ltp = StrukovMemristor(p_mem_lt)
mem_ltd = StrukovMemristor(p_mem_lt)

R_ltp = []
R_ltd = []
for _ in t_lt:
    mem_ltp.step(1.5, dt_lt)
    mem_ltd.step(-1.5, dt_lt)
    R_ltp.append(mem_ltp.resistance)
    R_ltd.append(mem_ltd.resistance)

R_ltp = np.array(R_ltp)
R_ltd = np.array(R_ltd)

ltp_ltd_df = pd.DataFrame({
    'Tiempo_s': t_lt,
    'Tiempo_ms': t_lt * 1000.0,
    'R_LTP_Ohm': R_ltp,
    'G_LTP_uS': (1.0 / R_ltp) * 1e6,
    'R_LTD_Ohm': R_ltd,
    'G_LTD_uS': (1.0 / R_ltd) * 1e6
})
ltp_ltd_df.to_csv(os.path.join(out_dir, "ltp_ltd_data.csv"), index=False)

# Métricas LTP
r_init_ltp = R_ltp[0]
r_final_ltp = R_ltp[-1]
dr_ltp = r_final_ltp - r_init_ltp
pct_ltp = (dr_ltp / r_init_ltp) * 100.0

target_change_ltp = r_init_ltp + 0.95 * dr_ltp
sat_idx_ltp = np.where(R_ltp <= target_change_ltp if dr_ltp < 0 else R_ltp >= target_change_ltp)[0]
sat_time_ltp = t_lt[sat_idx_ltp[0]] * 1000.0 if len(sat_idx_ltp) > 0 else t_lt[-1] * 1000.0

# Métricas LTD
r_init_ltd = R_ltd[0]
r_final_ltd = R_ltd[-1]
dr_ltd = r_final_ltd - r_init_ltd
pct_ltd = (dr_ltd / r_init_ltd) * 100.0

target_change_ltd = r_init_ltd + 0.95 * dr_ltd
sat_idx_ltd = np.where(R_ltd >= target_change_ltd if dr_ltd > 0 else R_ltd <= target_change_ltd)[0]
sat_time_ltd = t_lt[sat_idx_ltd[0]] * 1000.0 if len(sat_idx_ltd) > 0 else t_lt[-1] * 1000.0

ltp_ltd_metrics = {
    'Metrica': ['R inicial LTP (Ohm)', 'R final LTP (Ohm)', 'dR LTP (Ohm)', '% cambio LTP', 'Tiempo saturacion LTP (ms)',
                'R inicial LTD (Ohm)', 'R final LTD (Ohm)', 'dR LTD (Ohm)', '% cambio LTD', 'Tiempo saturacion LTD (ms)'],
    'Valor': [r_init_ltp, r_final_ltp, dr_ltp, pct_ltp, sat_time_ltp,
              r_init_ltd, r_final_ltd, dr_ltd, pct_ltd, sat_time_ltd]
}
pd.DataFrame(ltp_ltd_metrics).to_csv(os.path.join(out_dir, "ltp_ltd_metrics.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MEMORIA NO VOLÁTIL (Retención)
# ─────────────────────────────────────────────────────────────────────────────
print("--- Simulando Retencion ---")
dt_ret = 0.1e-3
t_ret = np.arange(0, 1.0, dt_ret)

p_mem_ret = StrukovParameters(
    R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=5e-12,
    v_th_mem=1.0, w_init=0.01, enable_nonlinear_drift=True, enable_hard_switching=True
)
mem_ret = StrukovMemristor(p_mem_ret)

R_ret_history = []
for t_val in t_ret:
    if t_val < 0.3:
        v = 1.5
    else:
        v = 0.0
    mem_ret.step(v, dt_ret)
    R_ret_history.append(mem_ret.resistance)

R_ret_history = np.array(R_ret_history)

ret_df = pd.DataFrame({
    'Tiempo_s': t_ret,
    'Tiempo_ms': t_ret * 1000.0,
    'Resistencia_Ohm': R_ret_history
})
ret_df.to_csv(os.path.join(out_dir, "retention_data.csv"), index=False)

r_init_ret = R_ret_history[0]
r_trained_ret = R_ret_history[int(0.3 / dt_ret) - 1]
r_retained_ret = R_ret_history[-1]

ret_percent = (1.0 - abs(r_retained_ret - r_trained_ret) / r_trained_ret) * 100.0
ret_time_ms = 700.0

ret_metrics = {
    'Metrica': ['R inicial (Ohm)', 'R entrenada (Ohm)', 'R retenida (Ohm)', 'Retencion (%)', 'Tiempo de retencion (ms)'],
    'Valor': [r_init_ret, r_trained_ret, r_retained_ret, ret_percent, ret_time_ms]
}
pd.DataFrame(ret_metrics).to_csv(os.path.join(out_dir, "retention_metrics.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CROSSBAR CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────
print("--- Simulando Clasificacion Crossbar ---")
np.random.seed(42)
test_patterns = [
    ([1, 1, 0], 0),
    ([0, 0, 1], 1),
    ([1, 0, 0], 0),
    ([0, 1, 0], 0),
    ([0, 0, 0], 2),
    ([1, 1, 0], 0),
    ([0, 0, 1], 1),
    ([1, 0, 1], 1),
    ([0, 0, 0], 2),
    ([0, 1, 1], 1),
]

W_matrix = np.array([
    [1.00, 0.01, 0.05],
    [0.90, 0.01, 0.05],
    [0.01, 1.00, 0.05],
])

results = []
for idx, (pattern, expected) in enumerate(test_patterns):
    T_test = 0.1
    n_test = int(T_test / dt_lt)
    
    p_lif_in = default_lif_params()
    p_lif_in.R_s = 80e3
    p_lif_out = default_lif_params()
    p_lif_out.R_s = 80e3
    
    neurons_in = [LIFNeuron(p_lif_in) for _ in range(3)]
    neurons_out = [LIFNeuron(p_lif_out) for _ in range(3)]
    
    mem_matrix = []
    for i in range(3):
        row = []
        for j in range(3):
            p_mem = StrukovParameters(R_on=100.0, R_off=500_000.0, D=10e-9, mu_v=0.0, w_init=W_matrix[i, j])
            row.append(StrukovMemristor(p_mem))
        mem_matrix.append(row)
        
    spike_timers = [0.0, 0.0, 0.0]
    out_spikes = [0, 0, 0]
    
    for k in range(n_test):
        V_in_val = [3.0 if pattern[i] == 1 else 0.2 for i in range(3)]
        V_out_in = [0.0, 0.0, 0.0]
        for i in range(3):
            neurons_in[i].step(V_in_val[i], dt_lt)
            if neurons_in[i].w >= 0.5 and spike_timers[i] <= 0:
                spike_timers[i] = 2e-3
            if spike_timers[i] > 0:
                V_out_in[i] = 5.0
                spike_timers[i] -= dt_lt
                
        for j in range(3):
            I_total = 0.0
            for i in range(3):
                R_m = mem_matrix[i][j].resistance
                I_syn = (V_out_in[i] - neurons_out[j].V) / (R_m + p_lif_out.R_s)
                I_total += I_syn
            
            V_eff_out = neurons_out[j].V + I_total * p_lif_out.R_s
            neurons_out[j].step(V_eff_out, dt_lt)
            if neurons_out[j].w >= 0.5:
                out_spikes[j] += 1
                
    obtained = np.argmax(out_spikes)
    results.append({
        'Muestra': idx,
        'Patron': str(pattern),
        'Salida_esperada': f'Out{expected}',
        'Salida_obtenida': f'Out{obtained}',
        'Spikes': str(out_spikes)
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(out_dir, "crossbar_data.csv"), index=False)

y_true = np.array([p[1] for p in test_patterns])
y_pred = np.array([int(r['Salida_obtenida'][-1]) for r in results])

accuracy = np.mean(y_true == y_pred)

precisions = []
recalls = []
f1s = []

for c in [0, 1, 2]:
    tp = np.sum((y_true == c) & (y_pred == c))
    fp = np.sum((y_true != c) & (y_pred == c))
    fn = np.sum((y_true == c) & (y_pred != c))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

macro_precision = np.mean(precisions)
macro_recall = np.mean(recalls)
macro_f1 = np.mean(f1s)

crossbar_metrics = {
    'Metrica': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)', 'Tasa de acierto'],
    'Valor': [accuracy, macro_precision, macro_recall, macro_f1, accuracy * 100.0]
}
pd.DataFrame(crossbar_metrics).to_csv(os.path.join(out_dir, "crossbar_metrics.csv"), index=False)

print("\n--- ¡Simulaciones completadas con éxito y métricas generadas! ---")
