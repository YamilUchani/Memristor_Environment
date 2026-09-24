"""
validaciones/validacion_strukov.py
===================================
Script de Validación Físico-Matemática Completa (Tema Blanco Académico):
Strukov et al. (2008, Nature 453) vs Simulador Neuromorphic Lab (6 Paneles).

Compara directamente en cada panel:
  - Datos de Referencia del Paper / CSV (Strukov et al. 2008 Fig 2b)
  - Simulación Numérica en Tiempo Real de Neuromorphic Lab (StrukovMathModel)

Salidas (generadas en esta misma carpeta):
  - validacion_strukov.png
  - validacion_strukov.csv
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

from neurolab.core.config import DeviceIdentity, ElectricalConfig, StrukovConfig
from neurolab.core.memristor import Memristor
from neurolab.devices.models.strukov import StrukovMathModel
from neurolab.io.validation_loader import ValidationDataLoader
from neurolab.core.validation_metrics import compute_all_metrics


def build_strukov_device() -> Memristor:
    return Memristor(
        math_model=StrukovMathModel(),
        electrical=ElectricalConfig(
            r_on=100.0,
            r_off=16_000.0,
            initial_state=0.10,
        ),
        identity=DeviceIdentity(
            device_name="Strukov TiO2 Ideal",
            device_family="oxide_memristor",
            model_name="strukov",
        ),
        model_config=StrukovConfig(
            D=10e-9,      # 10 nm
            mu_v=1e-14,   # 10^-14 m^2/(V s)
        ),
        modifiers=[],  # Modo 1 — Ideal (Sin Ventana, igual a strukov_ideal.json)
        clip_x=True,
    )


def main():
    print("=" * 80)
    print(" VALIDACION STRUKOV et al. (2008) vs SIMULADOR NEUROMORPHIC LAB (6 PANELES)")
    print("=" * 80)

    output_dir = Path(__file__).resolve().parent

    # 1. Cargar datos de referencia CSV de Strukov 2008 (Fig 2b)
    loader = ValidationDataLoader()
    val_data = loader.load_all(dataset="default")
    if val_data is None:
        print("[ADVERTENCIA] No se encontraron datos CSV en 'data for validation'.")

    # 2. Ejecutar Simulación Numérica en Neuromorphic Lab
    dev = build_strukov_device()
    v0 = 1.0
    f0 = 0.5
    t_total = 6.0
    dt = 1e-4
    steps = int(t_total / dt)
    t = np.linspace(0.0, t_total, steps)
    v_signal = v0 * np.sin(2.0 * np.pi * f0 * t)

    i_out = np.zeros(steps)
    x_state = np.zeros(steps)
    r_hist = np.zeros(steps)
    g_hist = np.zeros(steps)
    p_hist = np.zeros(steps)

    for k in range(steps):
        v = v_signal[k]
        i_inst = dev.step(voltage=v, dt=dt)
        i_out[k] = i_inst
        x_state[k] = dev.x
        r_hist[k] = dev.resistance
        g_hist[k] = dev.conductance
        p_hist[k] = abs(v * i_inst)

    # 3. Calcular Métricas Cuantitativas (R^2, MAE, RMSE)
    metrics = None
    if val_data is not None:
        metrics = compute_all_metrics(
            t_sim=t, i_sim_mA=i_out * 1e3, x_sim=x_state,
            r_sim_kohm=r_hist / 1e3, g_sim_us=g_hist * 1e6,
            val_data=val_data, v_sim=v_signal
        )
        print(f"\n {'Magnitud':<25} {'MAE':<12} {'RMSE':<12} {'R²':<10}")
        print("-" * 80)
        for m in metrics:
            print(f" {m.name:<25} {m.mae:<12.4g} {m.rmse:<12.4g} {m.r2:<10.4f}")
        print("=" * 80)

    # 4. Graficar Comparativa en Fondo Blanco Académico
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), facecolor='#ffffff')
    fig.suptitle("Validación Comparativa: Paper Strukov et al. (Nature 2008) vs Simulador Neuromorphic Lab", color='#0f172a', fontsize=14, fontweight='bold')

    for ax in axes.flat:
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#000000', labelcolor='#000000')
        ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
            spine.set_linewidth(1.1)

    step_ds = max(1, steps // 3000)
    t_ms = t[::step_ds] * 1e3

    # Extraer arrays del CSV de referencia si están disponibles
    t_v_csv = val_data["t_v"] * 1e3 if val_data is not None else None
    v_csv = val_data["v_val"] if val_data is not None else None
    t_i_csv = val_data["t_i"] * 1e3 if val_data is not None else None
    i_csv_mA = val_data["i_val_mA"] if val_data is not None else None
    t_w_csv = val_data["t_w"] * 1e3 if val_data is not None else None
    w_csv = val_data["wd_val"] if val_data is not None else None
    v_interp_csv = val_data["v_interp"] if val_data is not None else None
    r_csv_kohm = val_data["r_val_kohm"] if val_data is not None else None
    g_csv_us = val_data["g_val_us"] if val_data is not None else None

    # Panel 1: Histéresis I-V (Paper vs Simulador)
    ax1 = axes[0, 0]
    if val_data is not None:
        ax1.scatter(v_interp_csv[::10], i_csv_mA[::10], color='#d97706', s=14, alpha=0.85, label='[CSV] Paper Strukov (2008)', zorder=2)
    ax1.plot(v_signal[::step_ds], i_out[::step_ds] * 1e3, color='#dc2626', linewidth=2.0, label='[Sim] Neuromorphic Lab', zorder=3)
    ax1.set_title("① Histéresis I-V (Paper vs Simulador)", color='#0f172a', fontsize=11, fontweight='bold')
    ax1.set_xlabel("Voltaje V_IN (V)", color='#000000', fontweight='bold')
    ax1.set_ylabel("Corriente I (mA)", color='#000000', fontweight='bold')
    ax1.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a', loc='upper left')

    # Panel 2: Señales Temporales V(t) e I(t)
    ax2 = axes[0, 1]
    if val_data is not None:
        ax2.plot(t_i_csv, i_csv_mA, '--', color='#d97706', linewidth=1.8, label='[CSV] I(t) Paper 2008')
    ax2.plot(t_ms, i_out[::step_ds] * 1e3, color='#dc2626', linewidth=2.0, label='[Sim] I(t) Neuromorphic Lab')
    ax2.set_title("② Corriente I(t) en el Tiempo", color='#0f172a', fontsize=11, fontweight='bold')
    ax2.set_xlabel("Tiempo (ms)", color='#000000', fontweight='bold')
    ax2.set_ylabel("Corriente I (mA)", color='#000000', fontweight='bold')
    ax2.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a', loc='upper right')

    # Panel 3: Estado Interno x(t) = w/D
    ax3 = axes[0, 2]
    if val_data is not None:
        ax3.plot(t_w_csv, w_csv, '--', color='#d97706', linewidth=1.8, label='[CSV] x(t) Paper 2008')
    ax3.plot(t_ms, x_state[::step_ds], color='#059669', linewidth=2.0, label=r'[Sim] x(t) Neuromorphic Lab')
    ax3.set_title(r"③ Estado Interno x(t) ∈ [0,1]", color='#0f172a', fontsize=11, fontweight='bold')
    ax3.set_xlabel("Tiempo (ms)", color='#000000', fontweight='bold')
    ax3.set_ylabel("Ancho de Capa Doped x", color='#000000', fontweight='bold')
    ax3.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a', loc='upper right')

    # Panel 4: Resistencia R(t)
    ax4 = axes[1, 0]
    if val_data is not None:
        valid_r = ~np.isnan(r_csv_kohm)
        ax4.scatter(t_i_csv[valid_r][::15], r_csv_kohm[valid_r][::15], color='#d97706', s=12, alpha=0.7, label='[CSV] R(t) Paper 2008')
    ax4.plot(t_ms, r_hist[::step_ds] / 1e3, color='#2563eb', linewidth=2.0, label='[Sim] R(t) Neuromorphic Lab')
    ax4.set_title("④ Resistencia Memristiva R(t)", color='#0f172a', fontsize=11, fontweight='bold')
    ax4.set_xlabel("Tiempo (ms)", color='#000000', fontweight='bold')
    ax4.set_ylabel("Resistencia (kΩ)", color='#000000', fontweight='bold')
    ax4.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a', loc='upper right')

    # Panel 5: Conductancia G(t)
    ax5 = axes[1, 1]
    if val_data is not None:
        valid_g = ~np.isnan(g_csv_us)
        ax5.scatter(t_i_csv[valid_g][::15], g_csv_us[valid_g][::15], color='#d97706', s=12, alpha=0.7, label='[CSV] G(t) Paper 2008')
    ax5.plot(t_ms, g_hist[::step_ds] * 1e6, color='#7c3aed', linewidth=2.0, label='[Sim] G(t) Neuromorphic Lab')
    ax5.set_title("⑤ Conductancia Memristiva G(t)", color='#0f172a', fontsize=11, fontweight='bold')
    ax5.set_xlabel("Tiempo (ms)", color='#000000', fontweight='bold')
    ax5.set_ylabel("Conductancia (µS)", color='#000000', fontweight='bold')
    ax5.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a', loc='upper right')

    # Panel 6: Potencia Instantánea P(t)
    ax6 = axes[1, 2]
    ax6.plot(t_ms, p_hist[::step_ds] * 1e3, color='#db2777', linewidth=2.0, label='[Sim] P(t) Neuromorphic Lab')
    ax6.set_title("⑥ Potencia Disipada P(t)", color='#0f172a', fontsize=11, fontweight='bold')
    ax6.set_xlabel("Tiempo (ms)", color='#000000', fontweight='bold')
    ax6.set_ylabel("Potencia (mW)", color='#000000', fontweight='bold')
    ax6.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a', loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / "figuras" / "validacion_strukov.png", dpi=300)

    # 5. Guardar CSV con las métricas y series temporales
    csv_path = output_dir / "datos" / "validacion_strukov.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t_s,V_in_V,I_sim_A,x_sim,R_sim_ohm,G_sim_S,P_sim_W\n")
        for k in range(0, steps, step_ds):
            f.write(f"{t[k]:.6f},{v_signal[k]:.6f},{i_out[k]:.6e},{x_state[k]:.6f},{r_hist[k]:.2f},{g_hist[k]:.6e},{p_hist[k]:.6e}\n")

    print("\n[OK] Archivos comparativos generados exitosamente en validaciones/:")
    print("   - validacion_strukov.png  (Fondo Blanco Académico)")
    print("   - validacion_strukov.csv")


if __name__ == "__main__":
    main()
