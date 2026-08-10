"""
calibrate_lif.py
================
Script de calibración automática del modelo TSM-LIF físico.

Optimiza los parámetros libres (Rs, C, Ron, Roff, R0, Vth, Vhold, alpha, beta)
para minimizar la diferencia cuadrática (RMSE) frente a los datos experimentales
digitalizados del capacitor (Vc) y de la salida (Vout).

Uso:
    python calibrate_lif.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
import os, sys

# Asegurar importación del modelo y datos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from memristor_simulator.models.lif_neuron import LIFNeuron, LIFParameters
from memristor_simulator.validation.lif_experimental_data import VC_T, VC_V, VOUT_T, VOUT_V, VIN_T, VIN_V

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Función de simulación con parámetros dados
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(p: LIFParameters, T: float = 0.50, dt: float = 10e-6):
    n = int(T / dt)
    t = np.arange(n) * dt
    
    # Onda cuadrada a 100 Hz, amplitud 5V, 50% duty cycle
    f_in = 100.0
    Vin = np.where((t * f_in) % 1.0 < 0.5, 5.0, 0.0)
    
    neuron = LIFNeuron(p)
    Vc_sim = np.zeros(n)
    Vout_sim = np.zeros(n)
    w_sim = np.zeros(n)
    
    for k in range(n):
        neuron.step(Vin[k], dt)
        Vc_sim[k] = neuron.V
        Vout_sim[k] = neuron.Vout
        w_sim[k] = neuron.w
        
    return t, Vc_sim, Vout_sim, w_sim


# ─────────────────────────────────────────────────────────────────────────────
# Función de Costo (Error Cuadrático Medio)
# ─────────────────────────────────────────────────────────────────────────────

def loss_function(x):
    # Desempaquetar parámetros del vector de optimización
    # Usamos escala logarítmica para resistencias y capacitancias para mejorar estabilidad del optimizador
    Rs = 10**x[0]
    C = 10**x[1]
    R_on = 10**x[2]
    R_off = 10**x[3]
    R_0 = 10**x[4]
    
    V_th = x[5]
    V_hold = x[6]
    alpha = x[7]
    beta = x[8]
    
    p = LIFParameters(
        C=C, R_s=Rs, R_on=R_on, R_off=R_off, R_0=R_0,
        V_th=V_th, V_hold=V_hold, alpha=alpha, beta=beta
    )
    
    # Simular
    dt = 50e-6 # 50 us para optimización rápida (10,000 pasos en lugar de 50,000)
    t_sim, Vc_sim, Vout_sim, _ = run_simulation(p, T=0.50, dt=dt)
    
    # Control de inestabilidad numérica
    if np.any(np.isnan(Vc_sim)) or np.any(np.isinf(Vc_sim)) or np.any(np.isnan(Vout_sim)):
        return 1e6
        
    # Interpolación a los puntos de tiempo de los datos experimentales
    Vc_interp = np.interp(VC_T, t_sim, Vc_sim)
    Vout_interp = np.interp(VOUT_T, t_sim, Vout_sim)
    
    # Calcular RMSE
    rmse_vc = np.sqrt(np.mean((Vc_interp - VC_V)**2))
    rmse_vout = np.sqrt(np.mean((Vout_interp - VOUT_V)**2))
    
    # Peso balanceado: Vc va de 0 a 1V, Vout va de 0 a 0.55V.
    # Penalizamos si V_hold >= V_th
    penalty = 0.0
    if V_hold >= V_th:
        penalty = 10.0 * (V_hold - V_th + 0.1)
        
    return rmse_vc + 1.2 * rmse_vout + penalty


# ─────────────────────────────────────────────────────────────────────────────
# Ejecutar optimización
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  CALIBRACION AUTOMATICA DE LA NEURONA TSM-LIF")
    print("=" * 65)
    
    # Parámetros iniciales propuestos por el usuario (en escala adecuada para el optimizador)
    # x0: [log10(Rs), log10(C), log10(Ron), log10(Roff), log10(R0), Vth, Vhold, alpha, beta]
    x0 = [
        np.log10(100e3),  # Rs = 100k
        np.log10(500e-9),  # C = 500n
        np.log10(1e3),     # Ron = 1k
        np.log10(1e6),     # Roff = 1M
        np.log10(530.0),   # R0 = 530
        0.95,              # Vth
        0.15,              # Vhold
        25000.0,           # alpha
        220.0              # beta
    ]
    
    # Límites físicos (bounds)
    bounds = [
        (4.0, 6.0),       # Rs: 10k - 1M
        (-8.0, -5.0),     # C: 10n - 10u
        (2.0, 4.0),       # Ron: 100 - 10k
        (5.0, 7.0),       # Roff: 100k - 10M
        (2.0, 4.0),       # R0: 100 - 10k
        (0.60, 1.20),     # Vth: 0.6V - 1.2V
        (0.05, 0.40),     # Vhold: 50mV - 400mV
        (1000.0, 100000.0),# alpha
        (1.0, 2000.0)     # beta: permitimos desactivación lenta del TSM
    ]
    
    print("\n  Ejecutando ajuste por Nelder-Mead en SciPy...")
    
    iter_count = 0
    def callback(xk):
        nonlocal iter_count
        iter_count += 1
        if iter_count % 50 == 0:
            # Imprimir pérdida aproximada sin penalización extra para claridad
            print(f"    Iteración {iter_count:3d} | Pérdida actual: {loss_function(xk):.4f}")
            
    res = minimize(
        loss_function, x0, method='Nelder-Mead', bounds=bounds,
        callback=callback,
        options={'maxiter': 1500, 'disp': True}
    )
    
    # Extraer parámetros optimizados
    x_opt = res.x
    p_opt = LIFParameters(
        R_s = 10**x_opt[0],
        C = 10**x_opt[1],
        R_on = 10**x_opt[2],
        R_off = 10**x_opt[3],
        R_0 = 10**x_opt[4],
        V_th = x_opt[5],
        V_hold = x_opt[6],
        alpha = x_opt[7],
        beta = x_opt[8]
    )
    
    print("\n" + "=" * 65)
    print("  PARAMETROS OPTIMIZADOS ENCONTRADOS")
    print("=" * 65)
    print(f"  Rs     = {p_opt.R_s/1e3:.2f} kOhm  (inicial: 100.00 kOhm)")
    print(f"  C      = {p_opt.C*1e9:.2f} nF   (inicial: 500.00 nF)")
    print(f"  Ron    = {p_opt.R_on/1e3:.3f} kOhm   (inicial: 1.000 kOhm)")
    print(f"  Roff   = {p_opt.R_off/1e6:.2f} MOhm   (inicial: 1.00 MOhm)")
    print(f"  R0     = {p_opt.R_0:.1f} Ohm    (inicial: 530.0 Ohm)")
    print(f"  Vth    = {p_opt.V_th:.3f} V      (inicial: 0.950 V)")
    print(f"  Vhold  = {p_opt.V_hold:.3f} V      (inicial: 0.150 V)")
    print(f"  alpha  = {p_opt.alpha:.1f}       (inicial: 25000.0)")
    print(f"  beta   = {p_opt.beta:.1f}        (inicial: 220.0)")
    print("-" * 65)
    
    # Simular con parámetros iniciales y optimizados para graficar comparación
    p_init = LIFParameters(
        R_s=100e3, C=500e-9, R_on=1e3, R_off=1e6, R_0=530.0,
        V_th=0.95, V_hold=0.15, alpha=25000.0, beta=220.0
    )
    
    dt_fine = 5e-6 # 5 us para alta precisión en la gráfica
    t_init, Vc_init, Vout_init, _ = run_simulation(p_init, dt=dt_fine)
    t_opt, Vc_opt, Vout_opt, _ = run_simulation(p_opt, dt=dt_fine)
    
    # ── Gráficas de Comparación ───────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor("white")
    
    gs = gridspec.GridSpec(2, 1, hspace=0.4)
    
    # Panel 1: Voltaje del capacitor Vc
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(VC_T * 1e3, VC_V, color="red", marker="x", label="Datos Digitalizados (Papel)", zorder=5)
    ax1.plot(t_init * 1e3, Vc_init, color="gray", linestyle="--", alpha=0.7, label="Inicial (No Ajustado)")
    ax1.plot(t_opt * 1e3, Vc_opt, color="#1565C0", linewidth=2.0, label="Optimizado (Ajuste Físico)")
    ax1.set_ylabel("Voltaje del Capacitor $V_c$ (V)", fontsize=11)
    ax1.set_title("Calibración del Voltaje en el Capacitor ($V_c$)", fontsize=12, fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_xlim([0, 500])
    ax1.set_ylim([-0.05, 1.2])
    
    # Panel 2: Voltaje de salida Vout
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(VOUT_T * 1e3, VOUT_V, color="red", marker="x", label="Datos Digitalizados (Papel)", zorder=5)
    ax2.plot(t_init * 1e3, Vout_init, color="gray", linestyle="--", alpha=0.7, label="Inicial (No Ajustado)")
    ax2.plot(t_opt * 1e3, Vout_opt, color="#2E7D32", linewidth=2.0, label="Optimizado (Ajuste Físico)")
    ax2.set_xlabel("Tiempo (ms)", fontsize=11)
    ax2.set_ylabel("Voltaje de Salida $V_{out}$ (V)", fontsize=11)
    ax2.set_title("Calibración de los Spikes de Salida ($V_{out}$)", fontsize=12, fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.3)
    ax2.legend(loc="upper right")
    ax2.set_xlim([0, 500])
    ax2.set_ylim([-0.05, 0.75])
    
    # Calcular RMSE final
    Vc_interp = np.interp(VC_T, t_opt, Vc_opt)
    Vout_interp = np.interp(VOUT_T, t_opt, Vout_opt)
    rmse_vc = np.sqrt(np.mean((Vc_interp - VC_V)**2))
    rmse_vout = np.sqrt(np.mean((Vout_interp - VOUT_V)**2))
    
    fig.suptitle(
        f"Ajuste Cuantitativo del Gemelo Digital TSM-LIF\n"
        f"RMSE $V_c$ = {rmse_vc*1e3:.1f} mV | RMSE $V_{{out}}$ = {rmse_vout*1e3:.1f} mV",
        fontsize=14, fontweight="bold", y=0.98
    )
    
    save_path = os.path.join(OUTPUT_DIR, "tsm_calibration_results.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n  [PNG] Comparación guardada en:\n        {save_path}")
    print("=" * 65)
    
    # Sugerir copiar estos parámetros a default_lif_params()
    print("\n  Copia estos valores en memristor_simulator/models/lif_neuron.py:")
    print("  --------------------------------------------------------------")
    print(f"  C: float = {p_opt.C:.7e}")
    print(f"  R_s: float = {p_opt.R_s:.3e}")
    print(f"  R_off: float = {p_opt.R_off:.3e}")
    print(f"  R_on: float = {p_opt.R_on:.3e}")
    print(f"  R_0: float = {p_opt.R_0:.3f}")
    print(f"  V_th: float = {p_opt.V_th:.4f}")
    print(f"  V_hold: float = {p_opt.V_hold:.4f}")
    print(f"  alpha: float = {p_opt.alpha:.2f}")
    print(f"  beta: float = {p_opt.beta:.2f}")
    print("  --------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
