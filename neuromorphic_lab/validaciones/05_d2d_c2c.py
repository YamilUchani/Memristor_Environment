"""
validaciones/validacion_d2d_c2c.py
===================================
Script de Validación Estadísticas D2D (Device-to-Device) y C2C (Cycle-to-Cycle).

Comprobaciones:
  1. D2D (Monte Carlo N=100): Prueba de Kolmogorov-Smirnov (KS) p-valor > 0.05
     para verificar distribución Gaussiana estática de R_ON y R_OFF.
  2. C2C (Ornstein-Uhlenbeck): Varianza teórica Var(eta) = sigma^2 / (2*theta)
     vs varianza muestral simulada (Error rel < 5%).
  3. Gráficas: Envolvente I-V D2D, histogramas y función de autocorrelación C2C.

Salidas (generadas en esta misma carpeta):
  - validacion_d2d_c2c.png
  - validacion_d2d_c2c.csv
"""

import sys
from pathlib import Path

# Garantizar que el directorio raíz de neuromorphic_lab esté en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from neurolab.core.memristor import Memristor
from neurolab.core.config import DeviceIdentity, ElectricalConfig, StrukovConfig
from neurolab.devices.models.strukov import StrukovMathModel
from neurolab.devices.realism.d2d import D2DVariabilityModifier
from neurolab.devices.realism.c2c import C2CVariabilityModifier


def run_d2d_validation():
    print("=" * 80)
    print(" 1. VALIDACION ESTADISTICA D2D (Device-to-Device Monte Carlo)")
    print("=" * 80)

    n_samples = 100
    sigma_d2d = 0.05
    base_r_on = 100.0
    base_r_off = 16000.0

    r_on_samples = []
    r_off_samples = []
    factors = []

    base_elec = ElectricalConfig(r_on=base_r_on, r_off=base_r_off, initial_state=0.1)

    for seed in range(n_samples):
        mod = D2DVariabilityModifier(variability_std=sigma_d2d, seed=seed)
        elec_mod = mod.apply_to_electrical(base_elec)
        r_on_samples.append(elec_mod.r_on)
        r_off_samples.append(elec_mod.r_off)
        factors.append(mod._d2d_factor)

    r_on_samples = np.array(r_on_samples)
    r_off_samples = np.array(r_off_samples)
    factors = np.array(factors)

    norm_r_on = (r_on_samples - base_r_on) / (base_r_on * sigma_d2d)
    ks_stat, p_value = stats.kstest(norm_r_on, 'norm')

    print(f" Muestras D2D:            N = {n_samples}")
    print(f" R_ON Media / Std:        {np.mean(r_on_samples):.2f} Ohm +/- {np.std(r_on_samples):.2f} Ohm")
    print(f" R_OFF Media / Std:       {np.mean(r_off_samples):.2f} Ohm +/- {np.std(r_off_samples):.2f} Ohm")
    print(f" Primitiva KS Stat:       {ks_stat:.4f}")
    print(f" Primitiva KS p-valor:    {p_value:.4f}  ({'PASADO (p > 0.05)' if p_value > 0.05 else 'REVISAR'})")

    return r_on_samples, r_off_samples, factors, p_value


def run_c2c_validation():
    print("\n" + "=" * 80)
    print(" 2. VALIDACION ESTADISTICA C2C (Proceso de Ornstein-Uhlenbeck)")
    print("=" * 80)

    sigma = 0.05
    theta = 1.0
    dt_sim = 0.001
    duration = 500.0
    steps = int(duration / dt_sim)

    c2c_mod = C2CVariabilityModifier(sigma=sigma, theta=theta, seed=42, dt_simulation=dt_sim)

    eta_values = []
    dxdt_dummy = 1.0

    for step in range(steps):
        _ = c2c_mod.modify_dxdt(dxdt_dummy, 0.5, 1.0, 0.001, None, None)
        eta_values.append(c2c_mod.eta)

    eta_values = np.array(eta_values)
    warmup_steps = int(10.0 / dt_sim)
    eta_ss = eta_values[warmup_steps:]

    var_sim = float(np.var(eta_ss))
    var_th = (sigma ** 2) / (2.0 * theta)
    err_pct = abs(var_sim - var_th) / var_th * 100.0

    print(f" Pasos de integracion:    {steps:,} (100 s @ dt={dt_sim*1e3:.1f} ms)")
    print(f" Varianza Teorica OU:     {var_th:.6f}")
    print(f" Varianza Simulada OU:    {var_sim:.6f}")
    print(f" Error de Varianza (%):   {err_pct:.2f}%  ({'PASADO (< 5%)' if err_pct < 5.0 else 'REVISAR'})")

    autocorr = np.correlate(eta_ss - np.mean(eta_ss), eta_ss - np.mean(eta_ss), mode='full')
    autocorr = autocorr[len(eta_ss)-1:]
    autocorr /= autocorr[0]
    t_lags = np.arange(len(autocorr)) * dt_sim

    return eta_ss, var_sim, var_th, err_pct, t_lags[:5000], autocorr[:5000], theta


def main():
    output_dir = Path(__file__).resolve().parent

    r_on_samples, r_off_samples, factors, p_value = run_d2d_validation()
    eta_ss, var_sim, var_th, err_pct, t_lags, autocorr, theta = run_c2c_validation()
    sigma_d2d = 0.05

    with open(output_dir / "datos" / "validacion_d2d_c2c.csv", "w", encoding="utf-8") as f:
        f.write("Metrica,Valor,Esperado,Diagnostico\n")
        f.write(f"KS_p_value,{p_value:.4f},>0.05,{'PASADO' if p_value > 0.05 else 'REVISAR'}\n")
        f.write(f"C2C_Var_Sim,{var_sim:.6f},{var_th:.6f},Error={err_pct:.2f}%\n")
        f.write(f"C2C_Var_Error_Pct,{err_pct:.2f}%,<5.0%,{'PASADO' if err_pct < 5.0 else 'REVISAR'}\n")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor='#ffffff')

    for ax in axes.flat:
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#000000', labelcolor='#000000')
        ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
            spine.set_linewidth(1.1)

    ax1 = axes[0, 0]
    ax1.hist(r_on_samples, bins=15, color='#2563eb', edgecolor='#ffffff', alpha=0.85)
    ax1.axvline(100.0, color='#dc2626', linestyle='--', linewidth=1.8, label='Nominal (100 Ω)')
    ax1.set_title("D2D — Distribución de R_ON (N=100 Monte Carlo)", color='#0f172a', fontsize=10, fontweight='bold')
    ax1.set_xlabel("Resistencia R_ON (Ω)", color='#000000', fontweight='bold')
    ax1.set_ylabel("Frecuencia", color='#000000', fontweight='bold')
    ax1.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a')

    ax2 = axes[0, 1]
    t = np.linspace(0, 0.02, 1000)
    v_in = 1.0 * np.sin(2 * np.pi * 100 * t)
    base_elec_envelope = ElectricalConfig(r_on=100.0, r_off=16000.0, initial_state=0.1)
    
    # Colormap para diferenciar sutilmente las 20 celdas dentro del envolvente
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for idx in range(20):
        mod_i = D2DVariabilityModifier(variability_std=sigma_d2d, seed=idx)
        elec_i = mod_i.apply_to_electrical(base_elec_envelope)
        dev = Memristor(
            math_model=StrukovMathModel(),
            electrical=elec_i,
            identity=DeviceIdentity(device_name=f"D2D Cell {idx}", device_family="oxide_memristor", model_name="strukov"),
            model_config=StrukovConfig(D=1e-8, mu_v=1e-14),
            modifiers=[mod_i],
            clip_x=True,
        )
        dt = t[1] - t[0]
        i_list = [dev.step(voltage=v, dt=dt) for v in v_in]
        ax2.plot(v_in, np.array(i_list) * 1e3, color=colors[idx], alpha=0.55, linewidth=1.0)
        
    ax2.set_title("D2D — Envolvente Histerética I-V (20 Celdas)", color='#0f172a', fontsize=10, fontweight='bold')
    ax2.set_xlabel("Voltaje (V)", color='#000000', fontweight='bold')
    ax2.set_ylabel("Corriente (mA)", color='#000000', fontweight='bold')

    ax3 = axes[1, 0]
    ax3.plot(np.arange(len(eta_ss[:2000])) * 0.001, eta_ss[:2000], color='#059669', linewidth=1.2)
    ax3.axhline(0.0, color='#64748b', linestyle='--', alpha=0.8)
    ax3.set_title("C2C — Trayectoria Proceso OU η(t)", color='#0f172a', fontsize=10, fontweight='bold')
    ax3.set_xlabel("Tiempo (s)", color='#000000', fontweight='bold')
    ax3.set_ylabel("Fluctuación C2C η", color='#000000', fontweight='bold')

    ax4 = axes[1, 1]
    ax4.plot(t_lags[:2000], autocorr[:2000], color='#2563eb', linewidth=1.8, label='[Sim] Neuromorphic Lab')
    ax4.plot(t_lags[:2000], np.exp(-theta * t_lags[:2000]), ':', color='#d97706', linewidth=2.0, label='[Teórico] exp(-θ·t)')
    ax4.set_title("C2C — Función de Autocorrelación", color='#0f172a', fontsize=10, fontweight='bold')
    ax4.set_xlabel("Desfasaje τ (s)", color='#000000', fontweight='bold')
    ax4.set_ylabel("Autocorrelación R(τ)", color='#000000', fontweight='bold')
    ax4.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a')

    plt.tight_layout()
    plt.savefig(output_dir / "figuras" / "validacion_d2d_c2c.png", dpi=300)
    print("\n[OK] Archivos generados en validaciones/:")
    print("   - validacion_d2d_c2c.png")
    print("   - validacion_d2d_c2c.csv")


if __name__ == "__main__":
    main()
